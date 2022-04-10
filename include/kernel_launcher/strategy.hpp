#pragma once

#include <random>
#include <unordered_set>

#include "kernel_launcher/cache.hpp"
#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {

struct TuningStrategy {
    virtual bool init(const KernelBuilder& builder, Config& config) = 0;
    virtual bool submit(double performance, Config& config) = 0;
    virtual ~TuningStrategy() = default;
};

struct AnyTuningStrategy: TuningStrategy {
    AnyTuningStrategy() = default;
    AnyTuningStrategy(AnyTuningStrategy&&) = default;
    AnyTuningStrategy& operator=(AnyTuningStrategy&&) = default;

    template<typename T>
    AnyTuningStrategy(std::unique_ptr<T>&& inner) : inner_(std::move(inner)) {}

    template<typename T>
    AnyTuningStrategy(T inner) :
        inner_(
            std::make_unique<typename std::decay<T>::type>(std::move(inner))) {}

    operator bool() const {
        return (bool)inner_;
    }

    void reset() {
        inner_.reset();
    }

    bool init(const KernelBuilder& builder, Config& config) override {
        if (!inner_) return false;
        return inner_->init(builder, config);
    }

    bool submit(double performance, Config& config) override {
        if (!inner_) return false;
        return inner_->submit(performance, config);
    }

  private:
    std::unique_ptr<TuningStrategy> inner_;
};

struct RandomStrategy: TuningStrategy {
    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    ConfigIterator iter_ {};
};

struct HillClimbingStrategy: TuningStrategy {
    HillClimbingStrategy(AnyTuningStrategy inner = RandomStrategy {}) :
        inner_(std::move(inner)) {}

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    void update_best(double performance, const Config& config);

    std::default_random_engine rng_ {1};
    std::vector<bool> attempted_neighbors_ = {};
    std::vector<std::pair<TunableParam, TunableValue>> neighbors_ = {};
    size_t attempted_neighbors_count_ = 0;
    ConfigSpace space_ {};
    AnyTuningStrategy inner_ {};
    double best_performance_ = -1e99;
    Config best_config_ {};
};

struct LimitStrategy: TuningStrategy {
    LimitStrategy(
        uint64_t max_eval,
        AnyTuningStrategy inner = RandomStrategy {}) :
        max_eval_(max_eval),
        inner_(std::move(inner)) {}

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    uint64_t curr_eval_ = 0;
    uint64_t max_eval_ = 0;
    AnyTuningStrategy inner_ {};
};

struct CachingStrategy: TuningStrategy {
    CachingStrategy(
        std::string filename,
        AnyTuningStrategy inner = RandomStrategy {}) :
        filename_(std::move(filename)),
        inner_(std::move(inner)) {
        //
    }

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    std::string filename_ {};
    TuningCache cache_ {};
    AnyTuningStrategy inner_ {};
    bool first_run_ = true;
    Config first_config_ {};
};

namespace experimental {
    static inline Config tune_callback(
        std::string filename,
        AnyTuningStrategy strategy,
        const KernelBuilder& builder,
        std::function<double(const Config&)> callback) {
        TuningCache cache;
        Config current;
        Config best_config;
        double best_performance = -1e99;

        if (cache.initialize(std::move(filename), builder, current)) {
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
                best_config = current;
            }

            if (!strategy.submit(performance, current)) {
                break;
            }
        }

        return current;
    }

    template<typename... Args>
    static inline Config tune_kernel(
        std::string filename,
        AnyTuningStrategy strategy,
        const KernelBuilder& builder,
        Kernel<Args...>& kernel,
        dim3 problem_size,
        Args... args) {
        Config config = tune_callback(
            std::move(filename),
            builder,
            std::move(strategy),
            [&](const Config& config) -> double {
                CudaEvent before_event;
                CudaEvent after_event;
                cudaStream_t stream = nullptr;

                auto kernel = Kernel<Args...>::compile(builder, config);

                double total_time = 0.0;
                long long int total_workload = 0;

                for (size_t runs = 0; runs < 100 && total_time < 1.0; runs++) {
                    before_event.record(stream);
                    kernel(problem_size, stream)(args...);
                    after_event.record(stream);
                    after_event.synchronize();

                    // Skip measuring the first run
                    if (runs == 0) {
                        continue;
                    }

                    total_time +=
                        after_event.seconds_elapsed_since(before_event);
                    total_workload +=
                        problem_size.x * problem_size.y * problem_size.z;
                }

                return double(total_workload) / total_time;
            });

        kernel.initialize(builder, config);
        return config;
    }
}  // namespace experimental

}  // namespace kernel_launcher