#pragma once

#include <random>
#include <unordered_set>

#include "cache.hpp"
#include "kernel.hpp"

namespace kernel_launcher {
struct BaseStrategy {
    virtual bool init(const KernelBuilder& builder, Config& config) = 0;
    virtual bool submit(double performance, Config& config) = 0;
    virtual ~BaseStrategy() = default;
};

struct Strategy: BaseStrategy {
    Strategy() = default;
    Strategy(Strategy&&) = default;
    Strategy& operator=(Strategy&&) = default;

    template<typename T>
    Strategy(std::unique_ptr<T>&& inner) : inner_(std::move(inner)) {}

    template<typename T>
    Strategy(T inner) :
        inner_(
            std::make_unique<typename std::decay<T>::type>(std::move(inner))) {}

    operator bool() const {
        return (bool)inner_;
    }

    void reset() {
        inner_.reset();
    }

    bool init(const KernelBuilder& builder, Config& config) override {
        if (!inner_)
            return false;
        return inner_->init(builder, config);
    }

    bool submit(double performance, Config& config) override {
        if (!inner_)
            return false;
        return inner_->submit(performance, config);
    }

  private:
    std::unique_ptr<BaseStrategy> inner_;
};

struct RandomStrategy: BaseStrategy {
    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    ConfigIterator iter_ {};
};

struct HillClimbingStrategy: BaseStrategy {
    HillClimbingStrategy(Strategy inner = RandomStrategy {}) :
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
    Strategy inner_ {};
    double best_performance_ = -1e99;
    Config best_config_ {};
};

struct LimitStrategy: BaseStrategy {
    LimitStrategy(uint64_t max_eval, Strategy inner = RandomStrategy {}) :
        max_eval_(max_eval),
        inner_(std::move(inner)) {}

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    uint64_t curr_eval_ = 0;
    uint64_t max_eval_ = 0;
    Strategy inner_ {};
};

struct CachingStrategy: BaseStrategy {
    CachingStrategy(std::string filename, Strategy inner = RandomStrategy {}) :
        filename_(std::move(filename)),
        inner_(std::move(inner)) {
        //
    }

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    std::string filename_ {};
    TuningCache cache_ {};
    Strategy inner_ {};
    bool first_run_ = true;
    Config first_config_ {};
};

namespace strategy {
    inline RandomStrategy random() {
        return RandomStrategy();
    }

    inline LimitStrategy limit(size_t max_eval, Strategy strategy = random()) {
        return LimitStrategy(max_eval, std::move(strategy));
    }

    inline HillClimbingStrategy hill_climbing(Strategy strategy = random()) {
        return HillClimbingStrategy(std::move(strategy));
    }

    inline CachingStrategy
    cache(std::string filename, Strategy strategy = random()) {
        return CachingStrategy(std::move(filename), std::move(strategy));
    }
}  // namespace strategy

}  // namespace kernel_launcher

#if KERNEL_LAUNCHER_HEADERONLY
    #include KERNEL_LAUNCHER_IMPL("strategy.cpp")
#endif