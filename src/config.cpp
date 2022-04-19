#include "kernel_launcher/config.hpp"

#include <random>
#include <unordered_set>

namespace kernel_launcher {

KERNEL_LAUNCHER_API
const TunableValue& Config::at(const TunableParam& param) const {
    auto it = inner_.find(param);

    if (it == inner_.end()) {
        throw std::runtime_error(
            std::string("unknown parameter: ") + param.name());
    }

    return it->second;
}

KERNEL_LAUNCHER_API
const std::unordered_map<TunableParam, TunableValue>& Config::get() const {
    return inner_;
}

KERNEL_LAUNCHER_API
void Config::insert(TunableParam p, TunableValue v) {
    inner_[std::move(p)] = std::move(v);
}

KERNEL_LAUNCHER_API
nlohmann::json Config::to_json() const {
    std::unordered_map<std::string, nlohmann::json> results;

    for (const auto& p : inner_) {
        results[p.first.name()] = p.second.to_json();
    }

    return results;
}

KERNEL_LAUNCHER_API
Config Config::from_json(const nlohmann::json& json, const ConfigSpace& space) {
    return space.load_config(json);
}

KERNEL_LAUNCHER_API
TunableParam ConfigSpace::create_param(
    std::string name,
    Type type,
    std::vector<TunableValue> values,
    TunableValue default_value) {
    for (const auto& p : params_) {
        if (p.name() == name) {
            throw std::runtime_error("duplicate parameter '" + name + "'");
        }
    }

    TunableParam p = TunableParam(name, type, values, default_value);
    params_.push_back(p);
    return p;
}

KERNEL_LAUNCHER_API
void ConfigSpace::restrict(Expr<bool> expr) {
    restrictions_.push_back(std::move(expr));
}

KERNEL_LAUNCHER_API
uint64_t ConfigSpace::size() const {
    uint64_t n = 1;
    for (const auto& p : params_) {
        uint64_t k = (uint64_t)p.size();
        if (k == 0)
            return 0;

        // Check for overflows
        if ((n * k) / k != n) {
            throw std::runtime_error("integer overflow");
        }

        n *= k;
    }

    return n;
}

KERNEL_LAUNCHER_API
bool ConfigSpace::get(uint64_t index, Config& config) const {
    for (const auto& p : params_) {
        uint64_t n = (uint64_t)p.size();
        uint64_t i = index % n;
        index /= n;

        config.insert(p, p[i]);
    }

    Eval eval = {config.get()};

    for (const auto& r : restrictions_) {
        if (!eval(r)) {
            return false;
        }
    }

    return true;
}

KERNEL_LAUNCHER_API
bool ConfigSpace::is_valid(const Config& config) const {
    if (config.size() != params_.size()) {
        return false;
    }

    for (const auto& p : params_) {
        TunableValue value = config[p];
        bool is_allowed = false;

        for (const auto& allowed_value : p.values()) {
            is_allowed |= value == allowed_value;
        }

        if (!is_allowed) {
            return false;
        }
    }

    Eval eval = {config.get()};

    for (const auto& r : restrictions_) {
        if (!eval(r)) {
            return false;
        }
    }

    return true;
}

KERNEL_LAUNCHER_API
Config ConfigSpace::random_config() const {
    Config config;
    if (iterate().next(config)) {
        return config;
    } else {
        throw std::runtime_error("no valid configurations found");
    }
}

KERNEL_LAUNCHER_API
Config ConfigSpace::default_config() const {
    Config config;

    for (const auto& param : params_) {
        config.insert(param, param.default_value());
    }

    Eval eval = {config.get()};
    for (const auto& r : restrictions_) {
        if (!eval(r)) {
            throw std::runtime_error(
                "config does not pass restriction: " + r.to_string());
        }
    }

    return config;
}

KERNEL_LAUNCHER_API
Config ConfigSpace::load_config(const nlohmann::json& obj) const {
    Config config;

    for (const auto& param : params_) {
        TunableValue value = TunableValue::from_json(obj[param.name()]);
        bool is_valid = param.default_value() == value;

        for (const auto& allowed_value : param.values()) {
            is_valid |= value == allowed_value;
        }

        if (!is_valid) {
            throw std::runtime_error("key not found: " + param.name());
        }

        config.insert(param, std::move(value));
    }

    Eval eval = {config.get()};
    for (const auto& r : restrictions_) {
        if (!eval(r)) {
            throw std::runtime_error(
                "config does not pass restriction: " + r.to_string());
        }
    }

    return config;
}

KERNEL_LAUNCHER_API
const TunableParam& ConfigSpace::at(const char* s) const {
    for (const auto& param : params_) {
        if (param.name() == s) {
            return param;
        }
    }

    throw std::runtime_error(std::string("parameter not found: ") + s);
}

KERNEL_LAUNCHER_API
ConfigIterator ConfigSpace::iterate() const {
    return *this;
}

KERNEL_LAUNCHER_API
nlohmann::json ConfigSpace::to_json() const {
    using nlohmann::json;
    json results = json::object();

    std::unordered_map<std::string, json> params;
    for (const auto& param : params_) {
        std::vector<json> values;
        for (const auto& value : param.values()) {
            values.push_back(value.to_json());
        }

        params[param.name()] = values;
    }
    results["parameters"] = params;

    std::vector<json> restrictions;
    for (const auto& d : restrictions_) {
        restrictions.push_back(d.to_json());
    }
    results["restrictions"] = restrictions;

    return results;
}

KERNEL_LAUNCHER_API
void ConfigIterator::reset() {
    size_ = space_.size();
    visited_.clear();
    remaining_ = size_;
    rng_ = std::default_random_engine {std::random_device {}()};
}

KERNEL_LAUNCHER_API
bool ConfigIterator::next(Config& config) {
    while (remaining_ > 0) {
        std::uniform_int_distribution<uint64_t> dist {0, size_ - 1};
        uint64_t index = dist(rng_);

        uint64_t word_offset = index / 64;
        uint64_t bit_offset = index % 64;

        uint64_t mask = 1 << bit_offset;
        uint64_t& word = visited_[word_offset];

        // Bit already set, continue
        if ((word & mask) != 0) {
            continue;
        }

        // Set bit to 1
        word |= mask;
        remaining_--;

        if (space_.get(index, config)) {
            return true;
        }
    }

    return false;
}

}  // namespace kernel_launcher