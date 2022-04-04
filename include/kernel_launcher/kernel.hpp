#pragma once

#include <array>

#include "kernel_launcher/compile.hpp"
#include "kernel_launcher/config.hpp"
#include "kernel_launcher/expr.hpp"
#include "kernel_launcher/utils.hpp"
#include "nlohmann/json.hpp"

namespace kernel_launcher {

const NvrtcCompiler DEFAULT_COMPILER = {};

struct RawKernel {
    RawKernel() {}

    RawKernel(
        std::future<CudaModule> future,
        dim3 block_size,
        dim3 grid_divisor,
        uint32_t shared_mem) :
        _future(std::move(future)),
        _block_size(block_size),
        _grid_divisor(grid_divisor),
        _shared_mem(shared_mem) {}

    bool ready() const;
    void wait_ready() const;
    void launch(cudaStream_t stream, dim3 problem_size, void** args);

  private:
    bool _ready = false;
    std::future<CudaModule> _future;
    CudaModule _module;
    dim3 _block_size = 0;
    dim3 _grid_divisor = 0;
    uint32_t _shared_mem = 0;
};

struct KernelBuilder: ConfigSpace {
    KernelBuilder(Source kernel_source, std::string kernel_name) :
        _kernel_source(std::move(kernel_source)),
        _kernel_name(std::move(kernel_name)) {
        //
    }

    const std::string& kernel_name() const {
        return _kernel_name;
    }

    const Source& kernel_source() const {
        return _kernel_source;
    }

    template<typename T>
    KernelBuilder& template_type() {
        return template_arg(Type::of<T>());
    }

    template<typename T, typename... Ts>
    KernelBuilder& template_args(T&& first, Ts&&... rest) {
        template_arg(std::forward<T>(first));
        return template_args(std::forward<Ts>(rest)...);
    }

    KernelBuilder& template_args() {
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

    KernelBuilder& block_size(
        Expr<uint32_t> x,
        Expr<uint32_t> y = {1u},
        Expr<uint32_t> z = {1u});

    KernelBuilder& grid_divisors(
        Expr<uint32_t> x,
        Expr<uint32_t> y = {1u},
        Expr<uint32_t> z = {1u});

    KernelBuilder& shared_memory(Expr<uint32_t> s);
    KernelBuilder& template_arg(Expr<TemplateArg> arg);
    KernelBuilder& compiler_flag(Expr<std::string> opt);
    KernelBuilder& define(std::string name, Expr<std::string> value);
    KernelBuilder& assertion(Expr<bool> fun);

    std::array<Expr<uint32_t>, 3> tune_block_size(
        std::vector<uint32_t> xs,
        std::vector<uint32_t> ys = {1u},
        std::vector<uint32_t> zs = {1u});

    Expr<std::string>
    tune_compiler_flag(std::string name, std::vector<std::string> values);

    Expr<std::string>
    tune_define(std::string name, std::vector<std::string> values);

    RawKernel compile(
        const Config& config,
        const std::vector<Type>& parameter_types,
        const Compiler& compiler = DEFAULT_COMPILER) const;

    nlohmann::json to_json() const;

  private:
    Source _kernel_source;
    std::string _kernel_name;
    std::array<Expr<uint32_t>, 3> _block_size = {1u, 1u, 1u};
    std::array<Expr<uint32_t>, 3> _grid_divisors = {1u, 1u, 1u};
    Expr<uint32_t> _shared_mem = {0u};
    std::vector<Expr<TemplateArg>> _template_args {};
    std::vector<Expr<std::string>> _compile_flags {};
    std::vector<Expr<bool>> _assertions {};
    std::unordered_map<std::string, Expr<std::string>> _defines {};
};

template<typename T>
struct KernelArg {
    using type = T;
};

template<typename T>
struct KernelArg<T*> {
    using type = MemoryView<T>;
};

template<typename T>
struct KernelArg<const T*> {
    using type = MemoryView<T>;
};

template<typename T>
using kernel_arg_t = typename KernelArg<T>::type;

template<typename K, typename... Args>
struct KernelInstantiation {
    KernelInstantiation(cudaStream_t stream, dim3 problem_size, K& kernel) :
        _stream(stream),
        _problem_size(problem_size),
        _kernel(kernel) {
        //
    }

    void launch(kernel_arg_t<Args>... args) {
        std::array<void*, sizeof...(Args)> raw_args = {&args...};
        _kernel.launch(_stream, _problem_size, raw_args.data());
    }

    void operator()(kernel_arg_t<Args>... args) {
        return launch(args...);
    }

  private:
    cudaStream_t _stream;
    dim3 _problem_size;
    K& _kernel;
};

template<typename... Args>
struct Kernel {
    using instance_type = KernelInstantiation<RawKernel, Args...>;

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

    instance_type instantiate(cudaStream_t stream, dim3 problem_size) {
        return instance_type(stream, problem_size, _kernel);
    }

    instance_type operator()(cudaStream_t stream, dim3 problem_size) {
        return instantiate(stream, problem_size);
    }

    instance_type operator()(dim3 problem_size) {
        return instantiate(nullptr, problem_size);
    }

    instance_type operator()(
        cudaStream_t stream,
        uint32_t problem_x,
        uint32_t problem_y,
        uint32_t problem_z = 1) {
        return instantiate(stream, dim3(problem_x, problem_y, problem_z));
    }

    instance_type
    operator()(uint32_t problem_x, uint32_t problem_y, uint32_t problem_z = 1) {
        return instantiate(nullptr, dim3(problem_x, problem_y, problem_z));
    }

  private:
    RawKernel _kernel;
};

}  // namespace kernel_launcher