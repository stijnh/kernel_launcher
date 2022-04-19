#pragma once

#include <ios>

#include "kernel.hpp"
#include "strategy.hpp"

namespace kernel_launcher {

struct KernelResults {
    KernelResults(
        size_t min_evals = 0,
        size_t max_evals = 20,
        double max_seconds = 1.0,
        size_t num_outliers = 1) :
        min_evals_(min_evals),
        max_evals_(max_evals),
        max_seconds_(max_seconds),
        num_outliers_(num_outliers) {
        //
    }

    void reset();
    void add(dim3 problem_size, double time);
    bool collect(double& performance);

  private:
    std::vector<std::pair<dim3, double>> records_;
    size_t min_evals_;
    size_t max_evals_;
    double max_seconds_;
    size_t num_outliers_;
};

struct RawTuneKernel {
    RawTuneKernel() = default;

    RawTuneKernel(
        KernelBuilder builder,
        std::vector<Type> parameter_types,
        Strategy strategy = {},
        Compiler compiler = DEFAULT_COMPILER,
        KernelResults aggregator = {}) :
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
    Compiler compiler_;
    std::vector<Type> parameter_types_;

    CudaEvent before_event_;
    CudaEvent after_event_;

    double best_performance_ = -1e99;
    RawKernel best_kernel_;

    Config current_config_;
    RawKernel current_kernel_;
    dim3 current_problem_;
    KernelResults aggregator_;
    bool first_run_;
};

template<typename... Args>
struct TuneKernel {
    using instance_type = KernelInstantiation<RawTuneKernel, Args...>;

    TuneKernel() {}

    TuneKernel(
        KernelBuilder builder,
        Strategy strategy = {},
        Compiler compiler = DEFAULT_COMPILER) :
        kernel_(
            std::move(builder),
            {type_of<Args>()...},
            std::move(strategy),
            std::move(compiler)) {}

    static TuneKernel load(
        KernelBuilder builder,
        Strategy strategy = {},
        Compiler compiler = DEFAULT_COMPILER) {
        return TuneKernel(
            std::move(builder),
            std::move(strategy),
            std::move(compiler));
    }

    void initialize(
        KernelBuilder builder,
        Strategy strategy = {},
        Compiler compiler = DEFAULT_COMPILER) {
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

namespace experimental {
    static inline Config tune(
        std::string filename,
        Strategy strategy,
        const KernelBuilder& builder,
        const std::function<double(const Config&)>& callback) {
        TuningCache cache;
        Config current;
        Config best_config;
        double best_performance = -1e99;

        if (cache.open(std::move(filename), builder, current)) {
            return current;
        }

        if (!strategy.init(builder, current)) {
            throw std::runtime_error("");
        }

        while (true) {
            double performance;

            if (!cache.find(current, performance)) {
                performance = callback(current);
                cache.append(current, performance);
            }

            if (performance > best_performance) {
                best_performance = performance;
                best_config = Config(current);
            }

            if (!strategy.submit(performance, current)) {
                break;
            }
        }

        return best_config;
    }

    template<typename... Args>
    static inline Config tune_kernel(
        std::string filename,
        Strategy strategy,
        const KernelBuilder& builder,
        KernelResults results,
        Kernel<Args...>& kernel,
        dim3 problem_size,
        Args... args) {
        Config best_config = tune(
            std::move(filename),
            builder,
            std::move(strategy),
            [&](const Config& config) -> double {
                results.reset();
                CudaEvent before_event;
                CudaEvent after_event;
                cudaStream_t stream = nullptr;

                auto kernel = Kernel<Args...>::compile(builder, config);

                while (true) {
                    before_event.record(stream);
                    kernel.instantiate(problem_size, stream).launch(args...);
                    after_event.record(stream);
                    after_event.synchronize();

                    double time =
                        after_event.seconds_elapsed_since(before_event);
                    results.add(problem_size, time);

                    double performance;
                    if (results.collect(performance)) {
                        return performance;
                    }
                }
            });

        kernel.initialize(builder, best_config);
        return best_config;
    }
}  // namespace experimental

}  // namespace kernel_launcher

#if KERNEL_LAUNCHER_HEADERONLY
    #include KERNEL_LAUNCHER_IMPL("tune_kernel.cpp")
#endif