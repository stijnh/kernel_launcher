#include "kernel_launcher/strategy.hpp"

#include <unistd.h>

namespace kernel_launcher {

KERNEL_LAUNCHER_API
bool RandomStrategy::init(const KernelBuilder& builder, Config& config) {
    iter_ = builder.iterate();
    return iter_.next(config);
}

KERNEL_LAUNCHER_API
bool RandomStrategy::submit(double, Config& config) {
    return iter_.next(config);
}

KERNEL_LAUNCHER_API
bool LimitStrategy::init(const KernelBuilder& builder, Config& config) {
    curr_eval_ = 0;
    return inner_.init(builder, config);
}

KERNEL_LAUNCHER_API
bool LimitStrategy::submit(double performance, Config& config) {
    bool x = inner_.submit(performance, config) && curr_eval_++ < max_eval_;
    return x;
}

KERNEL_LAUNCHER_API
void HillClimbingStrategy::update_best(
    double performance,
    const Config& config) {
    std::fill(attempted_neighbors_.begin(), attempted_neighbors_.end(), false);
    attempted_neighbors_count_ = 0;
    best_performance_ = performance;
    best_config_ = Config(config);
}

KERNEL_LAUNCHER_API
bool HillClimbingStrategy::init(const KernelBuilder& builder, Config& config) {
    rng_ = std::default_random_engine {std::random_device {}()};
    space_ = ConfigSpace(builder);
    neighbors_.clear();
    attempted_neighbors_.clear();

    for (const auto& param : space_.parameters()) {
        for (const auto& value : param.values()) {
            neighbors_.push_back(std::make_pair(param, value));
            attempted_neighbors_.push_back(false);
        }
    }

    if (!inner_.init(builder, config)) {
        return false;
    }

    update_best(0.0, config);
    return space_.size() > 0;
}

KERNEL_LAUNCHER_API
bool HillClimbingStrategy::submit(double performance, Config& config) {
    if (performance > best_performance_) {
        update_best(performance, config);
    } else {
        config = Config(best_config_);  // Reset config
    }

    while (attempted_neighbors_count_ < neighbors_.size()) {
        std::uniform_int_distribution<size_t> dist {0, neighbors_.size() - 1};
        size_t index = dist(rng_);

        if (attempted_neighbors_[index]) {
            continue;
        }

        attempted_neighbors_count_++;
        attempted_neighbors_[index] = true;

        const auto& pair = neighbors_[index];
        const TunableValue& new_val = pair.second;
        const TunableValue& old_val = config[pair.first];

        if (old_val == new_val) {
            continue;
        }

        config.insert(pair.first, new_val);

        if (!space_.is_valid(config)) {
            config.insert(pair.first, old_val);  // put back the old value.
            continue;
        }

        return true;
    }

    if (!inner_.submit(performance, config)) {
        return false;
    }

    update_best(0, config);
    return true;
}

static bool
internal_submit(const TuningCache& cache, BaseStrategy& inner, Config& config) {
    double perf;

    while (true) {
        if (!cache.find(config, perf)) {
            return true;
        } else {
        }

        if (!inner.submit(perf, config)) {
            return false;
        } else {
        }
    }
}

KERNEL_LAUNCHER_API
bool CachingStrategy::init(const KernelBuilder& builder, Config& config) {
    if (!inner_.init(builder, config)) {
        return false;
    }

    Config best_config;
    if (cache_.open(filename_, builder, best_config)) {
        first_run_ = true;
        first_config_ = std::move(config);
        config = std::move(best_config);
        return true;
    }

    first_run_ = false;
    return internal_submit(cache_, inner_, config);
}

KERNEL_LAUNCHER_API
bool CachingStrategy::submit(double performance, Config& config) {
    if (first_run_) {
        first_run_ = false;
        config = std::move(first_config_);
    } else {
        cache_.append(config, performance);

        if (!inner_.submit(performance, config)) {
            return false;
        }
    }

    return internal_submit(cache_, inner_, config);
}

}  // namespace kernel_launcher
