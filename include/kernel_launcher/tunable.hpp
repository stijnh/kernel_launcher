#pragma once

#include <ios>

#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {

struct TuningStrategy {
    virtual bool init(const KernelBuilder& builder, Config& config) = 0;
    virtual bool submit(double performance, Config& config) = 0;
};

struct RandomStrategy: TuningStrategy {
    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    ConfigIterator _iter;
};

struct CachingStrategy: TuningStrategy {
    template<typename T>
    CachingStrategy(std::string filename, T inner = {}) :
        _inner(std::make_unique<std::decay_t<T>>(std::forward<T>(inner))),
        _filename(std::move(filename)) {
        //
    }

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    bool read_cache(const KernelBuilder& builder, Config& best_config);
    void write_cache(const Config& config, double performance);
    bool submit_internal(Config& config);

  private:
    std::unique_ptr<TuningStrategy> _inner;
    std::string _filename;
    nlohmann::json _json;
    bool _first_run;
    Config _first_config;
    std::string _current;
    std::unordered_map<std::string, double> _cache;
};

struct RawTuneKernel {
    RawTuneKernel() : _state(state_uninitialized) {
        //
    }

    RawTuneKernel(
        KernelBuilder builder,
        std::vector<Type> parameter_types,
        std::unique_ptr<Compiler> compiler = {},
        std::unique_ptr<TuningStrategy> strategy = {}) :
        _state(state_init),
        _builder(std::make_unique<KernelBuilder>(std::move(builder))),
        _strategy(std::move(strategy)),
        _compiler(std::move(compiler)),
        _parameter_types(std::move(parameter_types)) {
        if (!_strategy) {
            _strategy = std::make_unique<RandomStrategy>();
        }

        if (!_compiler) {
            _compiler = std::make_unique<NvrtcCompiler>();
        }

        next_configuration();
    }

    void launch(cudaStream_t stream, dim3 problem_size, void** args);

  private:
    void next_configuration();

    enum {
        state_uninitialized,
        state_init,
        state_tuning,
        state_compiling,
        state_finished,
    } _state;

    std::unique_ptr<KernelBuilder> _builder;
    std::unique_ptr<TuningStrategy> _strategy;
    std::unique_ptr<Compiler> _compiler;
    std::vector<Type> _parameter_types;

    CudaEvent _before_event;
    CudaEvent _after_event;

    double _best_performance = 1e9;
    RawKernel _best_kernel;

    Config _current_config;
    double _current_time = 0;
    uint64_t _current_workload = 0;
    RawKernel _current_kernel;
    bool _first_run;
};

template<typename... Args>
struct TuneKernel {
    using instance_type = KernelInstantiation<RawTuneKernel, Args...>;

    TuneKernel() {}

    TuneKernel(
        KernelBuilder builder,
        std::unique_ptr<TuningStrategy> strategy = {},
        std::unique_ptr<Compiler> compiler = {}) :
        _kernel(
            std::move(builder),
            {type_of<Args>()...},
            std::move(compiler),
            std::move(strategy)) {}

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

    RawTuneKernel _kernel;
};
}  // namespace kernel_launcher
