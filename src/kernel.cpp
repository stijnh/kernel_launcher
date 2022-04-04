#include "kernel_launcher/kernel.hpp"

#include "../include/kernel_launcher/kernel.hpp"

namespace kernel_launcher {

bool RawKernel::ready() const {
    return _ready
        || (_future.valid()
            && _future.wait_for(std::chrono::seconds(0))
                == std::future_status::ready);
}

void RawKernel::wait_ready() const {
    return _future.wait();
}

void RawKernel::launch(cudaStream_t stream, dim3 problem_size, void** args) {
    if (!_ready) {
        _ready = true;
        _module = _future.get();
    }

    dim3 grid_size = {
        div_ceil(problem_size.x, _grid_divisor.x),
        div_ceil(problem_size.y, _grid_divisor.y),
        div_ceil(problem_size.z, _grid_divisor.z)};

    return _module.launch(grid_size, _block_size, _shared_mem, stream, args);
}

KernelBuilder& KernelBuilder::block_size(
    Expr<uint32_t> x,
    Expr<uint32_t> y,
    Expr<uint32_t> z) {
    grid_divisors(x, y, z);
    _block_size[0] = std::move(x);
    _block_size[1] = std::move(y);
    _block_size[2] = std::move(z);
    return *this;
}

KernelBuilder& KernelBuilder::grid_divisors(
    Expr<uint32_t> x,
    Expr<uint32_t> y,
    Expr<uint32_t> z) {
    _grid_divisors[0] = std::move(x);
    _grid_divisors[1] = std::move(y);
    _grid_divisors[2] = std::move(z);
    return *this;
}

KernelBuilder& KernelBuilder::shared_memory(Expr<uint32_t> s) {
    _shared_mem = std::move(s);
    return *this;
}

KernelBuilder& KernelBuilder::template_arg(Expr<TemplateArg> arg) {
    _template_args.emplace_back(std::move(arg));
    return *this;
}

KernelBuilder& KernelBuilder::compiler_flag(Expr<std::string> opt) {
    _compile_flags.emplace_back(std::move(opt));
    return *this;
}

KernelBuilder&
KernelBuilder::define(std::string name, Expr<std::string> value) {
    if (_defines.find(name) != _defines.end()) {
        throw std::runtime_error("variable already defined: " + name);
    }

    _defines.insert({std::move(name), std::move(value)});
    return *this;
}

KernelBuilder& KernelBuilder::assertion(Expr<bool> fun) {
    restrict(fun);
    _assertions.emplace_back(std::move(fun));
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

    for (const auto& p : _assertions) {
        if (!eval(p)) {
            throw std::runtime_error("assertion failed: " + p.to_string());
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

    std::future<CudaModule> module = compiler.compile(
        _kernel_source,
        _kernel_name,
        template_args,
        parameter_types,
        options,
        nullptr);

    return RawKernel(std::move(module), block_size, grid_divisor, shared_mem);
}

nlohmann::json KernelBuilder::to_json() const {
    using nlohmann::json;
    json result = ConfigSpace::to_json();

    result["kernel_name"] = _kernel_name;
    result["block_size"] = {
        _block_size[0].to_json(),
        _block_size[1].to_json(),
        _block_size[2].to_json()};
    result["grid_divisors"] = {
        _grid_divisors[0].to_json(),
        _grid_divisors[1].to_json(),
        _grid_divisors[2].to_json()};
    result["shared_mem"] = _shared_mem.to_json();

    std::vector<json> template_args;
    for (const auto& p : _template_args) {
        template_args.push_back(p.to_json());
    }
    result["template_arg"] = template_args;

    std::vector<json> compile_flags;
    for (const auto& p : _compile_flags) {
        compile_flags.push_back(p.to_json());
    }
    result["compile_flags"] = compile_flags;

    std::unordered_map<std::string, json> defines;
    for (const auto& d : _defines) {
        defines[d.first] = d.second.to_json();
    }
    result["defines"] = defines;

    return result;
}

}  // namespace kernel_launcher