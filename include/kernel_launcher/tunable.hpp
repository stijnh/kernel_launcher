#pragma once

#include <ios>

#include "kernel_launcher/strategy.hpp"
#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {


struct RawTuneKernel {
    RawTuneKernel() : state_(state_uninitialized) {
        //
    }

    RawTuneKernel(
        KernelBuilder builder,
        std::vector<Type> parameter_types,
        std::unique_ptr<Compiler> compiler = {},
        std::unique_ptr<TuningStrategy> strategy = {}) :
        state_(state_init),
        builder_(std::make_unique<KernelBuilder>(std::move(builder))),
        strategy_(std::move(strategy)),
        compiler_(std::move(compiler)),
        parameter_types_(std::move(parameter_types)) {
        if (!strategy_) {
            strategy_ = std::make_unique<RandomStrategy>();
        }

        if (!compiler_) {
            compiler_ = std::make_unique<NvrtcCompiler>();
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
    } state_;

    std::unique_ptr<KernelBuilder> builder_;
    std::unique_ptr<TuningStrategy> strategy_;
    std::unique_ptr<Compiler> compiler_;
    std::vector<Type> parameter_types_;

    CudaEvent before_event_;
    CudaEvent after_event_;

    double best_performance_ = -1e99;
    RawKernel best_kernel_;

    Config current_config_;
    double current_time_ = 0;
    uint64_t current_workload_ = 0;
    RawKernel current_kernel_;
    bool first_run_;
};

template<typename... Args>
struct TuneKernel {
    using instance_type = KernelInstantiation<RawTuneKernel, Args...>;

    TuneKernel() {}

    TuneKernel(
        KernelBuilder builder,
        std::unique_ptr<TuningStrategy> strategy = {},
        std::unique_ptr<Compiler> compiler = {}) :
        kernel_(
            std::move(builder),
            {type_of<Args>()...},
            std::move(compiler),
            std::move(strategy)) {}

    static TuneKernel load(
        KernelBuilder builder,
        std::unique_ptr<TuningStrategy> strategy = {},
        std::unique_ptr<Compiler> compiler = {}) {
        return TuneKernel(
            std::move(builder),
            std::move(strategy),
            std::move(compiler));
    }

    void initialize(
        KernelBuilder builder,
        std::unique_ptr<TuningStrategy> strategy = {},
        std::unique_ptr<Compiler> compiler = {}) {
        *this = TuneKernel(
            std::move(builder),
            std::move(strategy),
            std::move(compiler));
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
        return instantiate(nullptr, dim3(problem_x, problem_y, problem_z));
    }

    RawTuneKernel kernel_;
};
}  // namespace kernel_launcher
