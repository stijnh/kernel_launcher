#include "kernel_launcher/compile.hpp"

#include "../include/kernel_launcher/compile.hpp"

namespace kernel_launcher {

static inline void nvrtc_assert(nvrtcResult err) {
    if (err != NVRTC_SUCCESS) {
        throw NvrtcException(err);
    }
}

static inline std::string generate_expression(
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

    if (!template_args.empty()) {
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

static inline std::string arch_flag(CUdevice* device_opt) {
    CUdevice device;
    if (device_opt) {
        device = *device_opt;
    } else {
        KERNEL_LAUNCHER_ASSERT(cuCtxGetDevice(&device));
    }

    int major, minor;
    KERNEL_LAUNCHER_ASSERT(cuDeviceGetAttribute(
        &minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device));
    KERNEL_LAUNCHER_ASSERT(cuDeviceGetAttribute(
        &major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device));

    std::stringstream oss;
    oss << "--gpu-architecture=compute_" << major << minor;
    return oss.str();
}

std::future<CudaModule> NvrtcCompiler::compile(
    const Source& kernel_source,
    const std::string& kernel_name,
    const std::vector<TemplateArg>& template_args,
    const std::vector<Type>& parameter_types,
    const std::vector<std::string>& options,
    CUdevice* device_opt) const {
    std::string symbol =
        generate_expression(kernel_name, template_args, parameter_types);

    std::vector<const char*> all_options;
    bool mentions_std = false;

    for (auto& opt : global_options_) {
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

std::future<CudaModule> AsyncCompiler::compile(
    const Source& kernel_source,
    const std::string& kernel_name,
    const std::vector<TemplateArg>& template_args,
    const std::vector<Type>& parameter_types,
    const std::vector<std::string>& options,
    CUdevice* device_opt) const {
    CUcontext context;
    KERNEL_LAUNCHER_ASSERT(cuCtxGetCurrent(&context));

    auto out = std::async(std::launch::async, [=]() {
        KERNEL_LAUNCHER_ASSERT(cuCtxSetCurrent(context));

        return inner_
            ->compile(
                kernel_source,
                kernel_name,
                template_args,
                parameter_types,
                options,
                nullptr)
            .get();
    });
    return out;
}

}  // namespace kernel_launcher