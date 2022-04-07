#include "kernel_launcher/config.hpp"

#include <random>
#include <unordered_set>

#include "../include/kernel_launcher/config.hpp"

namespace kernel_launcher {

const TunableValue& Config::at(const TunableParam& param) const {
    auto it = inner_.find(param);

    if (it == inner_.end()) {
        throw std::runtime_error(
            std::string("unknown parameter: ") + param.name());
    }

    return it->second;
}

const std::unordered_map<TunableParam, TunableValue>& Config::get() const {
    return inner_;
}

void Config::insert(TunableParam p, TunableValue v) {
    inner_[std::move(p)] = std::move(v);
}

nlohmann::json Config::to_json() const {
    std::unordered_map<std::string, nlohmann::json> results;

    for (const auto& p : inner_) {
        results[p.first.name()] = p.second.to_json();
    }

    return results;
}

Config Config::from_json(const nlohmann::json& json, const ConfigSpace& space) {
    return space.load_config(json);
}

TunableParam ConfigSpace::create_param(
    std::string name,
    Type type,
    std::vector<TunableValue> values,
    TunableValue default_value) {
    TunableParam p = TunableParam(name, type, values, default_value);
    params_.push_back(p);
    return p;
}

void ConfigSpace::restrict(Expr<bool> expr) {
    restrictions_.push_back(std::move(expr));
}

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

bool ConfigSpace::get(uint64_t index, Config& config) const {
    for (const auto& p : params_) {
        uint64_t n = (uint64_t)p.size();
        uint64_t i = index % n;
        index /= n;

        config.insert(p, p[i]);
    }

    return is_valid(config);
}

bool ConfigSpace::is_valid(const Config& config) const {
    Eval eval = {config.get()};

    for (const auto& r : restrictions_) {
        if (!eval(r)) {
            return false;
        }
    }

    return true;
}

Config ConfigSpace::sample_config() const {
    Config config;
    if (iterate().next(config)) {
        return config;
    } else {
        throw std::runtime_error("no valid configurations found");
    }
}

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

const TunableParam& ConfigSpace::at(const char* s) const {
    for (const auto& param : params_) {
        if (param.name() == s) {
            return param;
        }
    }

    throw std::runtime_error(std::string("parameter not found: ") + s);
}

ConfigIterator ConfigSpace::iterate() const {
    return *this;
}

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

void ConfigIterator::reset() {
    index_ = 0;
    size_ = space_.size();

    log4_ = 1;
    while (size_ >= (1 << (2 * log4_))) {
        log4_ += 1;
    }

    std::random_device device;
    for (uint32_t& seed : murmur_rounds_) {
        seed = device();
    }
}

static uint32_t murmur_hash2(uint32_t key, uint32_t seed) {
    const uint32_t m = 0x5bd1e995;
    uint32_t h = seed;
    uint32_t k = key;
    k *= m;
    k ^= k >> 24;
    k *= m;
    h *= m;
    h ^= k;
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
}

template<size_t Rounds>
static uint64_t encrypt_index(
    uint64_t index,
    uint64_t half_bits,
    std::array<uint32_t, Rounds> seeds) {
    uint64_t mask = (1 << half_bits) - 1;

    // break our index into the left and right half
    uint32_t left = (uint32_t)((index >> half_bits) & mask);
    uint32_t right = (uint32_t)((index & mask));

    // do 4 feistel rounds
    for (uint32_t seed : seeds) {
        uint32_t new_left = right;
        uint32_t new_right = (left ^ murmur_hash2(right, seed)) & mask;
        left = new_left;
        right = new_right;
    }

    // put the left and right back together to form the encrypted index
    return uint64_t(left << half_bits) | uint64_t(right);
}

bool ConfigIterator::next(Config& config) {
    while (index_ < (1 << (2 * log4_))) {
        uint64_t result = encrypt_index(index_, log4_, murmur_rounds_);
        index_++;

        if (result < size_ && space_.get(result, config)) {
            return true;
        }
    }

    return false;
}

}  // namespace kernel_launcher