#pragma once

#include <array>

#include "kernel_launcher/compile.hpp"
#include "kernel_launcher/config.hpp"
#include "kernel_launcher/expr.hpp"
#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

const NvrtcCompiler DEFAULT_COMPILER = {};

struct RawKernel {
    RawKernel() {
        //
    }

    RawKernel(
        CudaModule module,
        dim3 block_size,
        dim3 grid_divisor,
        uint32_t shared_mem) :
        _module(std::move(module)),
        _block_size(block_size),
        _grid_divisor(grid_divisor),
        _shared_mem(shared_mem) {}

    void launch(cudaStream_t stream, dim3 problem_size, void** args) const {
        dim3 grid_size = {
            div_ceil(problem_size.x, _grid_divisor.x),
            div_ceil(problem_size.y, _grid_divisor.y),
            div_ceil(problem_size.z, _grid_divisor.z)};

        return _module
            .launch(grid_size, _block_size, _shared_mem, stream, args);
    }

  private:
    CudaModule _module;
    dim3 _block_size;
    dim3 _grid_divisor;
    uint32_t _shared_mem;
};

struct KernelBuilder: ConfigSpace {
    KernelBuilder(Source kernel_source, std::string kernel_name) :
        _kernel_source(std::move(kernel_source)),
        _kernel_name(std::move(kernel_name)) {
        //
    }

    KernelBuilder& block_size(
        Expr<uint32_t> x,
        Expr<uint32_t> y = {1},
        Expr<uint32_t> z = {1}) {
        grid_divisors(x, y, z);
        _block_size[0] = std::move(x);
        _block_size[1] = std::move(y);
        _block_size[2] = std::move(z);
        return *this;
    }

    std::array<Expr<uint32_t>, 3> tune_block_size(std::vector<uint32_t> xs, std::vector<uint32_t> ys = {1}, std::vector<uint32_t> zs = {1}) {
        Expr<uint32_t> x = tune("block_size_x", std::move(xs));
        Expr<uint32_t> y = tune("block_size_y", std::move(ys));
        Expr<uint32_t> z = tune("block_size_z", std::move(zs));
        block_size(x, y, z);
        return {x, y, z};
    }

    KernelBuilder& grid_divisors(
        Expr<uint32_t> x,
        Expr<uint32_t> y = {1},
        Expr<uint32_t> z = {1}) {
        _grid_divisors[0] = std::move(x);
        _grid_divisors[1] = std::move(y);
        _grid_divisors[2] = std::move(z);
        return *this;
    }

    KernelBuilder& shared_memory(Expr<uint32_t> s) {
        _shared_mem = std::move(s);
        return *this;
    }

    template<typename T>
    KernelBuilder& template_type() {
        return template_arg(Type::of<T>());
    }

    KernelBuilder& template_arg(Expr<TemplateArg> arg) {
        _template_args.emplace_back(std::move(arg));
        return *this;
    }

    template<typename T, typename... Ts>
    KernelBuilder& template_args(T&& first, Ts&&... rest) {
        template_arg(std::forward<T>(first));
        return template_args(std::forward<Ts>(rest)...);
    }

    KernelBuilder& template_args() {
        return *this;
    }

    Expr<std::string> tune_compiler_flag(std::string name, std::vector<std::string> values) {
        Expr<std::string> e = tune(std::move(name), std::move(values));
        compiler_flag(e);
        return e;
    }

    KernelBuilder& compiler_flag(Expr<std::string> opt) {
        _compile_flags.emplace_back(std::move(opt));
        return *this;
    }

    template<typename T, typename... Ts>
    KernelBuilder& compiler_flags(T&& first, Ts&&... rest) {
        compiler_flag(std::forward<T>(first));
        return compiler_flags(std::forward<Ts>(rest)...);
    }

    KernelBuilder& compiler_flags() {
        return *this;
    }

    template<typename T>
    KernelBuilder& define(ParamExpr<T> value) {
        return define(value.name(), value);
    }

    KernelBuilder& define(std::string name, Expr<std::string> value) {
        if (_defines.find(name) != _defines.end()) {
            throw std::runtime_error("variable already defined: " + name);
        }

        _defines.insert({std::move(name), std::move(value)});
        return *this;
    }

    Expr<std::string> tune_define(std::string name, std::vector<std::string> values) {
        Expr<std::string> e = tune(name, values);
        define(name, e);
        return e;
    }

    KernelBuilder& assertion(Expr<bool> fun) {
        restrict(fun);
        _assertions.emplace_back(std::move(fun));
        return *this;
    }

    RawKernel compile(
        const Config& config,
        const std::vector<Type>& parameter_types,
        const Compiler& compiler = DEFAULT_COMPILER) const {
        Eval eval = {config.get()};

        for (const auto& p : _assertions) {
            if (!eval(p)) {
                throw std::runtime_error("assertion failed: " + p.name());
            }
        }

        std::vector<TemplateArg> template_args;
        for (const auto& p : _template_args) {
            template_args.push_back(eval(p));
        }

        std::vector<std::string> options;
        for (const auto& p : _compile_flags) {
            options.push_back(eval(p));
        }

        for (const auto& p : _defines) {
            options.push_back("--define-macro");
            options.push_back(p.first + "=" + eval(p.second));
        }

        dim3 block_size = {
            eval(_block_size[0]),
            eval(_block_size[1]),
            eval(_block_size[2])};

        dim3 grid_divisor = {
            eval(_grid_divisors[0]),
            eval(_grid_divisors[1]),
            eval(_grid_divisors[2])};

        uint32_t shared_mem = eval(_shared_mem);

        CudaModule module = compiler.compile(
            _kernel_source,
            _kernel_name,
            template_args,
            parameter_types,
            options,
            nullptr,
            block_size).get();

        return RawKernel(
            std::move(module),
            block_size,
            grid_divisor,
            shared_mem);
    }

    Source _kernel_source;
    std::string _kernel_name;
    std::array<Expr<uint32_t>, 3> _block_size = {1, 1, 1};
    std::array<Expr<uint32_t>, 3> _grid_divisors = {1, 1, 1};
    Expr<uint32_t> _shared_mem = {0};
    std::vector<Expr<TemplateArg>> _template_args {};
    std::vector<Expr<std::string>> _compile_flags {};
    std::vector<Expr<bool>> _assertions {};
    std::unordered_map<std::string, Expr<std::string>> _defines {};
};

template<typename... Args>
struct KernelInstantiation {
    KernelInstantiation(
        cudaStream_t stream,
        dim3 problem_size,
        const RawKernel& kernel) :
        _stream(stream),
        _problem_size(problem_size),
        _kernel(kernel) {
        //
    }

    void launch(Args... args) const {
        std::array<void*, sizeof...(Args)> raw_args = {&args...};
        _kernel.launch(_stream, _problem_size, raw_args.data());
    }

    void operator()(Args... args) const {
        return launch(args...);
    }

  private:
    cudaStream_t _stream;
    dim3 _problem_size;
    const RawKernel& _kernel;
};

template<typename... Args>
struct Kernel {
    Kernel() {
        //
    }

    Kernel(
        const KernelBuilder& builder,
        const Config& config,
        const Compiler& compiler = DEFAULT_COMPILER) {
        load(builder, config, compiler);
    }

    static Kernel<Args...> compile(
        const KernelBuilder& builder,
        const Config& config,
        const Compiler& compiler = DEFAULT_COMPILER) {
        Kernel<Args...> kernel;
        kernel.load(builder, config, compiler);
        return kernel;
    }

    void load(
        const KernelBuilder& builder,
        const Config& config,
        const Compiler& compiler = DEFAULT_COMPILER) {
        _kernel = builder.compile(config, {type_of<Args>()...}, compiler);
    }

    KernelInstantiation<Args...>
    instantiate(cudaStream_t stream, dim3 problem_size) const {
        return KernelInstantiation<Args...>(stream, problem_size, _kernel);
    }

    KernelInstantiation<Args...>
    operator()(cudaStream_t stream, dim3 problem_size) const {
        return instantiate(stream, problem_size);
    }

    KernelInstantiation<Args...> operator()(dim3 problem_size) const {
        return instantiate(nullptr, problem_size);
    }

    KernelInstantiation<Args...> operator()(
        cudaStream_t stream,
        uint32_t problem_x,
        uint32_t problem_y = 1,
        uint32_t problem_z = 1) const {
        return instantiate(stream, dim3(problem_x, problem_y, problem_z));
    }

    KernelInstantiation<Args...> operator()(
        uint32_t problem_x,
        uint32_t problem_y = 1,
        uint32_t problem_z = 1) const {
        return instantiate(nullptr, dim3(problem_x, problem_y, problem_z));
    }

  private:
    RawKernel _kernel;
};

}  // namespace kernel_launcher