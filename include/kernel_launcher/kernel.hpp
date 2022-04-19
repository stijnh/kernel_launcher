#pragma once

#include <array>

#include "compile.hpp"
#include "config.hpp"
#include "expr.hpp"
#include "utils.hpp"

#if KERNEL_LAUNCHER_JSON
    #include "nlohmann/json.hpp"
#endif

namespace kernel_launcher {

const NvrtcCompiler DEFAULT_COMPILER = {};

struct RawKernel {
    RawKernel() {}

    RawKernel(
        std::future<CudaModule> future,
        dim3 block_size,
        dim3 grid_divisor,
        uint32_t shared_mem) :
        future_(std::move(future)),
        block_size_(block_size),
        grid_divisor_(grid_divisor),
        shared_mem_(shared_mem) {}

    bool ready() const;
    void wait_ready() const;
    void launch(cudaStream_t stream, dim3 problem_size, void** args);

  private:
    bool ready_ = false;
    std::future<CudaModule> future_;
    CudaModule module_;
    dim3 block_size_ = 0;
    dim3 grid_divisor_ = 0;
    uint32_t shared_mem_ = 0;
};

struct KernelBuilder: ConfigSpace {
    KernelBuilder(Source kernel_source, std::string kernel_name) :
        kernel_source_(std::move(kernel_source)),
        kernel_name_(std::move(kernel_name)) {
        //
    }

    const std::string& kernel_name() const {
        return kernel_name_;
    }

    const Source& kernel_source() const {
        return kernel_source_;
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
        const CompilerBase& compiler = DEFAULT_COMPILER) const;

#if KERNEL_LAUNCHER_JSON
    nlohmann::json to_json() const;
#endif

  private:
    Source kernel_source_;
    std::string kernel_name_;
    std::array<Expr<uint32_t>, 3> block_size_ = {1u, 1u, 1u};
    std::array<Expr<uint32_t>, 3> grid_divisors_ = {1u, 1u, 1u};
    Expr<uint32_t> shared_mem_ = {0u};
    std::vector<Expr<TemplateArg>> template_args_ {};
    std::vector<Expr<std::string>> compile_flags_ {};
    std::vector<Expr<bool>> assertions_ {};
    std::unordered_map<std::string, Expr<std::string>> defines_ {};
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
        stream_(stream),
        problem_size_(problem_size),
        kernel_(kernel) {
        //
    }

    void launch(kernel_arg_t<Args>... args) {
        std::array<void*, sizeof...(Args)> raw_args = {&args...};
        kernel_.launch(stream_, problem_size_, raw_args.data());
    }

    void operator()(kernel_arg_t<Args>... args) {
        return launch(args...);
    }

  private:
    cudaStream_t stream_;
    dim3 problem_size_;
    K& kernel_;
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
        const CompilerBase& compiler = DEFAULT_COMPILER) {
        initialize(builder, config, compiler);
    }

    static Kernel<Args...> load(
        const KernelBuilder& builder,
        const Config& config,
        const CompilerBase& compiler = DEFAULT_COMPILER) {
        Kernel<Args...> kernel;
        kernel.load(builder, config, compiler);
        return kernel;
    }

    void initialize(
        const KernelBuilder& builder,
        const Config& config,
        const CompilerBase& compiler = DEFAULT_COMPILER) {
        kernel_ = builder.compile(config, {type_of<Args>()...}, compiler);
    }

    instance_type instantiate(cudaStream_t stream, dim3 problem_size) {
        return instance_type(stream, problem_size, kernel_);
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
        return instantiate(dim3(problem_x, problem_y, problem_z));
    }

  private:
    RawKernel kernel_;
};

}  // namespace kernel_launcher

#if KERNEL_LAUNCHER_HEADERONLY
    #include KERNEL_LAUNCHER_IMPL("kernel.cpp")
#endif