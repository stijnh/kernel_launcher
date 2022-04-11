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
        filename_(std::move(filename)),
        has_content_(false) {
        //
    }

    Source(const char* filename) : Source(std::string(filename)) {
        //
    }

    Source(std::string filename, std::string content) :
        filename_(std::move(filename)),
        content_(std::move(content)),
        has_content_(true) {
        //
    }

    std::string file_name() const {
        return filename_;
    }

    std::string read() const {
        if (has_content_) {
            return content_;
        }

        std::ifstream t(filename_);
        return {
            (std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>()};
    }

  private:
    std::string filename_;
    std::string content_;
    bool has_content_;
};

struct NvrtcException: std::runtime_error {
    NvrtcException(nvrtcResult err) :
        std::runtime_error(nvrtcGetErrorString(err)),
        err_(err) {
        //
    }

    NvrtcException(nvrtcResult err, std::string msg) :
        std::runtime_error(
            nvrtcGetErrorString(err) + std::string(": ") + std::move(msg)),
        err_(err) {
        //
    }

    nvrtcResult error() const {
        return err_;
    }

  private:
    nvrtcResult err_;
};

struct CompilerBase {
    virtual std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const = 0;
    virtual ~CompilerBase() = default;
};

struct Compiler: CompilerBase {
    Compiler() = default;
    Compiler(const Compiler&) = default;
    Compiler(Compiler&&) = default;

    template<typename C>
    Compiler(std::shared_ptr<C> inner) : inner_(std::move(inner)) {}

    template<typename C>
    Compiler(C compiler) :
        inner_(std::make_shared<typename std::decay<C>::type>(
            std::move(compiler))) {}

    void reset() {
        inner_.reset();
    }

    std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const {
        KERNEL_LAUNCHER_ASSERT((bool)inner_);
        return inner_->compile(
            kernel_source,
            kernel_name,
            template_args,
            parameter_types,
            options,
            device_opt);
    }

  private:
    std::shared_ptr<CompilerBase> inner_;
};

struct NvrtcCompiler: CompilerBase {
    NvrtcCompiler() {
        //
    }

    void add_option(std::string opt) {
        global_options_.push_back(opt);
    }

    std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const override;

  private:
    std::vector<std::string> global_options_;
};

struct AsyncCompiler: CompilerBase {
    AsyncCompiler(Compiler compiler) : inner_(std::move(compiler)) {
        //
    }

    std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const override;

  private:
    Compiler inner_;
};

}  // namespace kernel_launcher