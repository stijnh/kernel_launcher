#pragma once

#include <nvrtc.h>

#include <fstream>
#include <future>
#include <sstream>
#include <vector>

#include "module.hpp"
#include "utils.hpp"

namespace kernel_launcher {

struct Source {
    Source(std::string filename) :
        _filename(std::move(filename)),
        _has_content(false) {}

    Source(const char* filename) : Source(std::string(filename)) {}

    Source(std::string filename, std::string content) :
        _filename(std::move(filename)),
        _content(std::move(content)),
        _has_content(true) {}

    std::string file_name() const {
        return _filename;
    }

    std::string read() const {
        if (_has_content) {
            return _content;
        }

        std::ifstream t(_filename);
        return {
            (std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>()};
    }

  private:
    std::string _filename;
    std::string _content;
    bool _has_content;
};

struct Compiler {
    virtual std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const = 0;
};

struct NvrtcException: std::runtime_error {
    NvrtcException(nvrtcResult err) :
        std::runtime_error(nvrtcGetErrorString(err)),
        _err(err) {
        //
    }

    NvrtcException(nvrtcResult err, std::string msg) :
        std::runtime_error(
            nvrtcGetErrorString(err) + std::string(": ") + std::move(msg)),
        _err(err) {
        //
    }

    nvrtcResult error() const {
        return _err;
    }

  private:
    nvrtcResult _err;
};

void nvrtc_assert(nvrtcResult err) {
    if (err != NVRTC_SUCCESS) {
        throw NvrtcException(err);
    }
}

struct NvrtcCompiler: Compiler {
    NvrtcCompiler() {
        //
    }

    void add_option(std::string opt) {
        _global_options.push_back(opt);
    }

    static std::string generate_expression(
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types) {
        std::stringstream oss;
        oss << "(void(*)(";

        bool is_first = true;
        for (const Type& ty : parameter_types) {
            if (!is_first) {
                oss << ",";
            } else {
                is_first = false;
            }

            oss << ty.name();
        }

        oss << "))";
        oss << kernel_name;

        if (template_args.size() > 0) {
            oss << "<";

            is_first = true;
            for (const TemplateArg& arg : template_args) {
                if (!is_first) {
                    oss << ",";
                } else {
                    is_first = false;
                }

                oss << arg.get();
            }

            oss << ">";
        }

        return oss.str();
    }

    static std::string arch_flag(CUdevice* device_opt) {
        CUdevice device;
        if (device_opt) {
            device = *device_opt;
        } else {
            cu_assert(cuCtxGetDevice(&device));
        }

        int major, minor;
        cu_assert(cuDeviceGetAttribute(
            &minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device));
        cu_assert(cuDeviceGetAttribute(
            &major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device));

        std::stringstream oss;
        oss << "--gpu-architecture=compute_" << major << minor;
        return oss.str();
    }

    std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const override {
        std::string symbol =
            generate_expression(kernel_name, template_args, parameter_types);

        std::vector<const char*> all_options;
        bool mentions_std = false;

        for (auto& opt : _global_options) {
            all_options.push_back(opt.c_str());
            mentions_std |= opt.find("-std") == 0;
        }

        for (auto& opt : options) {
            all_options.push_back(opt.c_str());
            mentions_std |= opt.find("-std") == 0;
        }

        if (!mentions_std) {
            all_options.push_back("-std=c++11");
        }

        std::string arch = arch_flag(device_opt);
        all_options.push_back(arch.c_str());

        std::string source_name = kernel_source.file_name();
        std::string source_content = kernel_source.read();

        nvrtcProgram program = nullptr;
        nvrtc_assert(nvrtcCreateProgram(
            &program,
            source_content.c_str(),
            source_name.c_str(),
            0,  // numHeaders
            nullptr,  // headers
            nullptr  // includeNames
            ));

        try {
            nvrtc_assert(nvrtcAddNameExpression(program, symbol.c_str()));

            nvrtcResult result = nvrtcCompileProgram(
                program,
                (int)all_options.size(),
                all_options.data());

            if (result != NVRTC_SUCCESS) {
                size_t size = 0;
                nvrtc_assert(nvrtcGetProgramLogSize(program, &size));

                std::vector<char> log(size + 1, '\0');
                nvrtc_assert(nvrtcGetProgramLog(program, log.data()));

                throw NvrtcException(result, std::string(log.data()));
            }

            const char* lowered_name = nullptr;
            nvrtc_assert(
                nvrtcGetLoweredName(program, symbol.c_str(), &lowered_name));

            size_t size = 0;
            nvrtc_assert(nvrtcGetPTXSize(program, &size));

            std::vector<char> ptx(size + 1, '\0');
            nvrtc_assert(nvrtcGetPTX(program, ptx.data()));

            CudaModule module = CudaModule(ptx.data(), lowered_name);

            nvrtc_assert(nvrtcDestroyProgram(&program));

            auto promise = std::promise<CudaModule>();
            promise.set_value(std::move(module));
            return promise.get_future();
        } catch (std::exception& e) {
            nvrtcDestroyProgram(&program);
            throw;
        }
    }

  private:
    std::vector<std::string> _global_options;
};

struct AsyncCompiler: Compiler {
    template<typename C>
    AsyncCompiler(C compiler) : _inner(std::make_shared<C>(compiler)) {
        //
    }

    std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const override {
        CUdevice device;
        if (device_opt) {
            device = *device_opt;
        } else {
            cu_assert(cuCtxGetDevice(&device));
        }

        auto out = std::async(std::launch::async, [=]() {
            CUdevice d = device;
            CUcontext context;
            cu_assert(cuDevicePrimaryCtxRetain(&context, device));
            cu_assert(cuCtxSetCurrent(context));

            return _inner
                ->compile(
                    kernel_source,
                    kernel_name,
                    template_args,
                    parameter_types,
                    options,
                    &d)
                .get();
        });
        return out;
    }

    std::shared_ptr<Compiler> _inner;
};

}  // namespace kernel_launcher