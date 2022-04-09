#pragma once

#include <random>
#include <unordered_set>

#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {

struct TuningCache {
    TuningCache(std::string filename) : filename_(std::move(filename)) {
        //
    }

    bool initialize(const KernelBuilder& builder, Config& best_config);
    void append(const Config& config, double performance);
    bool find(const Config& config, double& performance) const;

  private:
    std::string filename_;
    bool initialized_ = false;
    std::unordered_map<std::string, double> cache_ {};
    std::vector<TunableParam> parameters_ {};
};

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
        KERNEL_LAUNCHER_ASSERT(*this);
        return inner_->init(builder, config);
    }

    bool submit(double performance, Config& config) override {
        KERNEL_LAUNCHER_ASSERT(*this);
        return inner_->submit(performance, config);
    }

  private:
    std::unique_ptr<TuningStrategy> inner_;
};

struct RandomStrategy: TuningStrategy {
    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    ConfigIterator iter_;
};

struct HillClimbingStrategy: TuningStrategy {
    HillClimbingStrategy(AnyTuningStrategy inner) : inner_(std::move(inner)) {}

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    void update_best(double performance, const Config& config);

    std::default_random_engine rng_;
    std::vector<bool> attempted_neighbors_;
    std::vector<std::pair<TunableParam, TunableValue>> neighbors_;
    size_t attempted_neighbors_count_;
    ConfigSpace space_;
    AnyTuningStrategy inner_;
    double best_performance_;
    Config best_config_;
};

struct LimitStrategy: TuningStrategy {
    LimitStrategy(uint64_t max_eval, AnyTuningStrategy inner) :
        max_eval_(max_eval),
        inner_(std::move(inner)) {}

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    uint64_t curr_eval_;
    uint64_t max_eval_;
    AnyTuningStrategy inner_;
};

struct CachingStrategy: TuningStrategy {
    CachingStrategy(std::string filename, AnyTuningStrategy inner) :
        cache_(std::move(filename)),
        inner_(std::move(inner)) {
        //
    }

    bool init(const KernelBuilder& builder, Config& config) override;
    bool submit(double performance, Config& config) override;

  private:
    TuningCache cache_;
    AnyTuningStrategy inner_;
    bool first_run_;
    Config first_config_;
};

}  // namespace kernel_launcher