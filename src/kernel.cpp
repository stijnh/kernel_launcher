#include "kernel_launcher/kernel.hpp"

#include "../include/kernel_launcher/kernel.hpp"

namespace kernel_launcher {

bool RawKernel::ready() const {
    return ready_
        || (future_.valid()
            && future_.wait_for(std::chrono::seconds(0))
                == std::future_status::ready);
}

void RawKernel::wait_ready() const {
    return future_.wait();
}

void RawKernel::launch(cudaStream_t stream, dim3 problem_size, void** args) {
    if (!ready_) {
        ready_ = true;
        module_ = future_.get();
    }

    dim3 grid_size = {
        div_ceil(problem_size.x, grid_divisor_.x),
        div_ceil(problem_size.y, grid_divisor_.y),
        div_ceil(problem_size.z, grid_divisor_.z)};

    return module_.launch(grid_size, block_size_, shared_mem_, stream, args);
}

KernelBuilder& KernelBuilder::block_size(
    Expr<uint32_t> x,
    Expr<uint32_t> y,
    Expr<uint32_t> z) {
    grid_divisors(x, y, z);
    block_size_[0] = std::move(x);
    block_size_[1] = std::move(y);
    block_size_[2] = std::move(z);
    return *this;
}

KernelBuilder& KernelBuilder::grid_divisors(
    Expr<uint32_t> x,
    Expr<uint32_t> y,
    Expr<uint32_t> z) {
    grid_divisors_[0] = std::move(x);
    grid_divisors_[1] = std::move(y);
    grid_divisors_[2] = std::move(z);
    return *this;
}

KernelBuilder& KernelBuilder::shared_memory(Expr<uint32_t> s) {
    shared_mem_ = std::move(s);
    return *this;
}

KernelBuilder& KernelBuilder::template_arg(Expr<TemplateArg> arg) {
    template_args_.emplace_back(std::move(arg));
    return *this;
}

KernelBuilder& KernelBuilder::compiler_flag(Expr<std::string> opt) {
    compile_flags_.emplace_back(std::move(opt));
    return *this;
}

KernelBuilder&
KernelBuilder::define(std::string name, Expr<std::string> value) {
    if (defines_.find(name) != defines_.end()) {
        throw std::runtime_error("variable already defined: " + name);
    }

    defines_.insert({std::move(name), std::move(value)});
    return *this;
}

KernelBuilder& KernelBuilder::assertion(Expr<bool> fun) {
    restrict(fun);
    assertions_.emplace_back(std::move(fun));
    return *this;
}

std::array<Expr<uint32_t>, 3> KernelBuilder::tune_block_size(
    std::vector<uint32_t> xs,
    std::vector<uint32_t> ys,
    std::vector<uint32_t> zs) {
    Expr<uint32_t> x = tune("block_size_x", std::move(xs));
    Expr<uint32_t> y = tune("block_size_y", std::move(ys));
    Expr<uint32_t> z = tune("block_size_z", std::move(zs));
    block_size(x, y, z);
    return {x, y, z};
}

Expr<std::string> KernelBuilder::tune_compiler_flag(
    std::string name,
    std::vector<std::string> values) {
    Expr<std::string> e = tune(std::move(name), std::move(values));
    compiler_flag(e);
    return e;
}

Expr<std::string>
KernelBuilder::tune_define(std::string name, std::vector<std::string> values) {
    Expr<std::string> e = tune(name, values);
    define(name, e);
    return e;
}

RawKernel KernelBuilder::compile(
    const Config& config,
    const std::vector<Type>& parameter_types,
    const Compiler& compiler) const {
    Eval eval = {config.get()};

    for (const auto& p : assertions_) {
        if (!eval(p)) {
            throw std::runtime_error("assertion failed: " + p.to_string());
        }
    }

    std::vector<TemplateArg> template_args;
    for (const auto& p : template_args_) {
        template_args.push_back(eval(p));
    }

    std::vector<std::string> options;
    for (const auto& p : compile_flags_) {
        options.push_back(eval(p));
    }

    for (const auto& p : defines_) {
        options.push_back("--define-macro");
        options.push_back(p.first + "=" + eval(p.second));
    }

    options.push_back("-DKERNEL_LAUNCHER=1");
    options.push_back("-Dkernel_tuner=1");

    dim3 block_size = {
        eval(block_size_[0]),
        eval(block_size_[1]),
        eval(block_size_[2])};

    dim3 grid_divisor = {
        eval(grid_divisors_[0]),
        eval(grid_divisors_[1]),
        eval(grid_divisors_[2])};

    uint32_t shared_mem = eval(shared_mem_);

    std::future<CudaModule> module = compiler.compile(
        kernel_source_,
        kernel_name_,
        template_args,
        parameter_types,
        options,
        nullptr);

    return RawKernel(std::move(module), block_size, grid_divisor, shared_mem);
}

nlohmann::json KernelBuilder::to_json() const {
    using nlohmann::json;
    json result = ConfigSpace::to_json();

    result["kernel_name"] = kernel_name_;
    result["block_size"] = {
        block_size_[0].to_json(),
        block_size_[1].to_json(),
        block_size_[2].to_json()};
    result["grid_divisors"] = {
        grid_divisors_[0].to_json(),
        grid_divisors_[1].to_json(),
        grid_divisors_[2].to_json()};
    result["shared_mem"] = shared_mem_.to_json();

    std::vector<json> template_args;
    for (const auto& p : template_args_) {
        template_args.push_back(p.to_json());
    }
    result["template_arg"] = template_args;

    std::vector<json> compile_flags;
    for (const auto& p : compile_flags_) {
        compile_flags.push_back(p.to_json());
    }
    result["compile_flags"] = compile_flags;

    std::unordered_map<std::string, json> defines;
    for (const auto& d : defines_) {
        defines[d.first] = d.second.to_json();
    }
    result["defines"] = defines;

    return result;
}

}  // namespace kernel_launcher