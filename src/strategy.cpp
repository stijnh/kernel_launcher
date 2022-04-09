#include "kernel_launcher/strategy.hpp"

#include <unistd.h>

namespace kernel_launcher {
bool RandomStrategy::init(const KernelBuilder& builder, Config& config) {
    iter_ = builder.iterate();
    return iter_.next(config);
}

bool RandomStrategy::submit(double, Config& config) {
    return iter_.next(config);
}

bool LimitStrategy::init(const KernelBuilder& builder, Config& config) {
    curr_eval_ = 0;
    return inner_->init(builder, config);
}

bool LimitStrategy::submit(double performance, Config& config) {
    curr_eval_++;
    bool response = inner_->submit(performance, config);
    return response && (curr_eval_ <= max_eval_);
}

void HillClimbingStrategy::update_best(
    double performance,
    const Config& config) {
    std::fill(attempted_neighbors_.begin(), attempted_neighbors_.end(), false);
    attempted_neighbors_count_ = 0;
    best_performance_ = performance;
    best_config_ = config;
}

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

    config = space_.default_config();
    update_best(0.0, config);
    return space_.size() > 0;
}

bool HillClimbingStrategy::submit(double performance, Config& config) {
    if (performance > best_performance_) {
        update_best(performance, config);
    } else {
        config = best_config_;  // Reset config
    }

    while (attempted_neighbors_count_ < neighbors_.size()) {
        std::uniform_int_distribution<size_t> dist {0, neighbors_.size()};
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

    config = space_.random_config();
    update_best(0, config);
    return true;
}
}  // namespace kernel_launcher
