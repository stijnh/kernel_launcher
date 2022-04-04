#include "kernel_launcher/config.hpp"
#include "../include/kernel_launcher/config.hpp"


#include <random>
#include <unordered_set>

namespace kernel_launcher {

const TunableValue& Config::at(const TunableParam& param) const {
    auto it = _inner.find(param);

    if (it == _inner.end()) {
        throw std::runtime_error(
            std::string("unknown parameter: ") + param.name());
    }

    return it->second;
}

const std::unordered_map<TunableParam, TunableValue>& Config::get() const {
    return _inner;
}

void Config::insert(TunableParam p, TunableValue v) {
    _inner[std::move(p)] = std::move(v);
}

nlohmann::json Config::to_json() const {
    std::unordered_map<std::string, nlohmann::json> results;

    for (const auto& p : _inner) {
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
    std::vector<TunableValue> values) {
    TunableParam p = TunableParam(name, type);
    _params.insert({p, std::move(values)});
    return p;
}

void ConfigSpace::restrict(Expr<bool> expr) {
    _restrictions.push_back(std::move(expr));
}

size_t ConfigSpace::size() const {
    size_t n = 1;
    for (const auto& p : _params) {
        size_t k = p.second.size();
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

bool ConfigSpace::get(size_t index, Config& config) const {
    for (const auto& p : _params) {
        size_t n = p.second.size();
        size_t i = index % n;
        index /= n;

        config.insert(p.first, p.second[i]);
    }

    return is_valid(config);
}

bool ConfigSpace::is_valid(const Config& config) const {
    Eval eval = {config.get()};

    for (const auto& r : _restrictions) {
        if (!eval(r)) {
            return false;
        }
    }

    return true;
}

Config ConfigSpace::sample() const {
    size_t n = size();
    std::random_device rd;
    std::uniform_int_distribution<size_t> rng(0, n);
    std::unordered_set<size_t> attempted;
    Config config;

    while (attempted.size() < n) {
        size_t i = rng(rd);

        if (attempted.insert(i).second) {
            if (get(i, config)) {
                return config;
            }
        }
    }

    throw std::runtime_error("no valid configurations found");
}

Config ConfigSpace::load_config(const nlohmann::json& obj) const {
    Config config;

    for (const auto& p : _params) {
        const TunableParam& param = p.first;
        const std::vector<TunableValue>& valid_values = p.second;

        TunableValue value = TunableValue::from_json(obj[param.name()]);
        bool is_valid = false;

        for (const auto& valid_value : valid_values) {
            is_valid |= valid_value == value;
        }

        if (!is_valid) {
            throw std::runtime_error("key not found: " + param.name());
        }

        config.insert(param, std::move(value));
    }

    Eval eval = {config.get()};
    for (const auto& r : _restrictions) {
        if (!eval(r)) {
            throw std::runtime_error(
                "config does not pass restriction: " + r.to_string());
        }
    }

    return config;
}

const TunableParam& ConfigSpace::at(std::string& s) const {
    for (const auto& p : _params) {
        if (p.first.name() == s) {
            return p.first;
        }
    }

    throw std::runtime_error("parameter not found: " + s);
}

ConfigIterator ConfigSpace::iterate() const {
    return *this;
}

nlohmann::json ConfigSpace::to_json() const {
    using nlohmann::json;
    json results = json::object();

    std::unordered_map<std::string, json> params;
    for (const auto& p : _params) {
        std::vector<json> values;
        for (const auto& v : p.second) {
            values.push_back(v.to_json());
        }

        params[p.first.name()] = values;
    }
    results["parameters"] = params;

    std::vector<json> restrictions;
    for (const auto& d : _restrictions) {
        restrictions.push_back(d.to_json());
    }
    results["restrictions"] = restrictions;

    return results;
}

void ConfigIterator::reset() {
    _attempts = range(_space.size());

    std::random_device device;
    std::default_random_engine rng {device()};
    std::shuffle(_attempts.begin(), _attempts.end(), rng);
}

bool ConfigIterator::next(Config& config) {
    while (!_attempts.empty()) {
        size_t index = _attempts.back();
        _attempts.pop_back();

        if (_space.get(index, config)) {
            return true;
        }
    }

    return false;
}

}  // namespace kernel_launcher