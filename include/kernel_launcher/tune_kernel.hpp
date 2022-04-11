#pragma once

#include <ios>

#include "kernel_launcher/kernel.hpp"
#include "kernel_launcher/strategy.hpp"

namespace kernel_launcher {

struct Aggregator {
    Aggregator(size_t max_evals = 20, double max_seconds = 1.0) :
        max_evals_(max_evals),
        max_seconds_(max_seconds) {
        //
    }

    void reset();
    void add(dim3 problem_size, double time);
    bool collect(double& performance);

  private:
    std::vector<std::pair<dim3, double>> records_;
    size_t max_evals_;
    double max_seconds_;
};

struct RawTuneKernel {
    RawTuneKernel() = default;

    RawTuneKernel(
        KernelBuilder builder,
        std::vector<Type> parameter_types,
        Strategy strategy = {},
        std::unique_ptr<Compiler> compiler = std::make_unique<NvrtcCompiler>(),
        Aggregator aggregator = {}) :
        builder_(std::make_unique<KernelBuilder>(std::move(builder))),
        strategy_(std::move(strategy)),
        compiler_(std::move(compiler)),
        parameter_types_(std::move(parameter_types)),
        aggregator_(std::move(aggregator)) {
        if (!strategy_) {
            strategy_ = RandomStrategy();
        }

        if (!strategy_.init(*builder_, current_config_)) {
            throw std::runtime_error("search strategy failed to initialize");
        }

        next_configuration();
    }

    void launch(cudaStream_t stream, dim3 problem_size, void** args);

  private:
    void next_configuration();

    enum {
        state_uninitialized,
        state_tuning,
        state_compiling,
        state_measuring,
        state_finished,
    } state_ = state_uninitialized;

    std::unique_ptr<KernelBuilder> builder_;
    Strategy strategy_;
    std::unique_ptr<Compiler> compiler_;
    std::vector<Type> parameter_types_;

    CudaEvent before_event_;
    CudaEvent after_event_;

    double best_performance_ = -1e99;
    RawKernel best_kernel_;

    Config current_config_;
    RawKernel current_kernel_;
    dim3 current_problem_;
    Aggregator aggregator_;
    bool first_run_;
};

template<typename... Args>
struct TuneKernel {
    using instance_type = KernelInstantiation<RawTuneKernel, Args...>;

    TuneKernel() {}

    TuneKernel(
        KernelBuilder builder,
        Strategy strategy = {},
        std::unique_ptr<Compiler> compiler = {}) :
        kernel_(
            std::move(builder),
            {type_of<Args>()...},
            std::move(strategy),
            std::move(compiler)) {}

    static TuneKernel load(
        KernelBuilder builder,
        Strategy strategy = {},
        std::unique_ptr<Compiler> compiler = {}) {
        return TuneKernel(
            std::move(builder),
            std::move(strategy),
            std::move(compiler));
    }

    void initialize(
        KernelBuilder builder,
        Strategy strategy = {},
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
