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
        _has_content(false) {
        //
    }

    Source(const char* filename) : Source(std::string(filename)) {
        //
    }

    Source(std::string filename, std::string content) :
        _filename(std::move(filename)),
        _content(std::move(content)),
        _has_content(true) {
        //
    }

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

struct Compiler {
    virtual std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const = 0;
};

struct NvrtcCompiler: Compiler {
    NvrtcCompiler() {
        //
    }

    void add_option(std::string opt) {
        _global_options.push_back(opt);
    }

    std::future<CudaModule> compile(
        const Source& kernel_source,
        const std::string& kernel_name,
        const std::vector<TemplateArg>& template_args,
        const std::vector<Type>& parameter_types,
        const std::vector<std::string>& options,
        CUdevice* device_opt) const override;

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
        CUdevice* device_opt) const override;

  private:
    std::shared_ptr<Compiler> _inner;
};

}  // namespace kernel_launcher