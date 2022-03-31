#pragma once

#include <random>
#include <unordered_set>

#include "kernel_launcher/expr.hpp"
#include "kernel_launcher/utils.hpp"
#include "kernel_launcher/value.hpp"

namespace kernel_launcher {

struct ConfigSpace;
struct ConfigIterator;

struct Config {
    Config() {
        //
    }

    const TunableValue& operator[](const TunableParam& param) const {
        auto it = _inner.find(param);

        if (it == _inner.end()) {
            throw std::runtime_error(
                std::string("unknown parameter: ") + param.name());
        }

        return it->second;
    }

    const std::unordered_map<TunableParam, TunableValue>& get() const {
        return _inner;
    }

    void insert(TunableParam p, TunableValue v) {
        _inner.insert({std::move(p), std::move(v)});
    }

    nlohmann::json to_json() const {
        std::unordered_map<std::string, nlohmann::json> results;

        for (const auto& p : _inner) {
            results[p.first.name()] = p.second.to_json();
        }

        return results;
    }

    static Config
    from_json(const nlohmann::json& json, const ConfigSpace& space);

  private:
    std::unordered_map<TunableParam, TunableValue> _inner;
};

struct ConfigSpace {
    TunableParam create_param(
        std::string name,
        Type type,
        std::vector<TunableValue> values) {
        TunableParam p = TunableParam(name, type);
        _params.insert({p, std::move(values)});
        return p;
    }

    template<typename T>
    ParamExpr<T> tune(std::string name, const std::vector<T>& values) {
        std::vector<TunableValue> tvalues;
        for (const auto& p : values) {
            tvalues.push_back(p);
        }

        return ParamExpr<T>(
            create_param(std::move(name), type_of<T>(), std::move(tvalues)));
    }

    template<typename T>
    ParamExpr<T>
    tune(std::string name, const std::initializer_list<T>& values) {
        std::vector<TunableValue> tvalues;
        for (const auto& p : values) {
            tvalues.push_back(p);
        }

        return ParamExpr<T>(
            create_param(std::move(name), type_of<T>(), std::move(tvalues)));
    }

    void restrict(Expr<bool> expr) {
        _restrictions.push_back(std::move(expr));
    }

    size_t size() const {
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

    Config get(size_t index) const {
        Config config;

        for (const auto& p : _params) {
            size_t n = p.second.size();
            size_t i = index % n;
            index /= n;

            config.insert(p.first, p.second[i]);
        }

        return config;
    }

    bool is_valid(const Config& config) const {
        Eval eval = {config.get()};

        for (const auto& r : _restrictions) {
            if (!eval(r)) {
                return false;
            }
        }

        return true;
    }

    Config sample() const {
        size_t n = size();
        std::random_device rd;
        std::uniform_int_distribution<size_t> rng(0, n);
        std::unordered_set<size_t> attempted;

        while (attempted.size() < n) {
            size_t i = rng(rd);

            if (attempted.insert(i).second) {
                Config config = get(i);

                if (is_valid(config)) {
                    return config;
                }
            }
        }

        throw std::runtime_error("no valid configurations found");
    }

    Config load_config(const nlohmann::json& obj) const {
        Config config;

        for (const auto& p : _params) {
            const TunableParam& param = p.first;
            const std::vector<TunableValue>& valid_values = p.second;

            TunableValue value = TunableValue::from_json(obj[param.name()]);
            bool is_valid = false;

            for (const auto& valid_value : valid_values) {
                is_valid |= valid_value == value;
            }

            if (is_valid) {
                throw std::runtime_error("key not found: " + param.name());
            }

            config.insert(param, std::move(value));
        }

        Eval eval = {config.get()};
        for (const auto &r: _restrictions) {
            if (!eval(r)) {
                throw std::runtime_error("config does not pass restriction: " + r.name());
            }
        }

        return config;
    }

    ConfigIterator iterate() const;

  private:
    std::unordered_map<TunableParam, std::vector<TunableValue>> _params;
    std::vector<Expr<bool>> _restrictions;
};

struct ConfigIterator {
    ConfigIterator(ConfigSpace space): _space(std::move(space)) {
        _attempts = range(_space.size());

        std::default_random_engine rng {0};
        std::shuffle(_attempts.begin(), _attempts.end(), rng);
    }

    bool next(Config &config) {
        while (!_attempts.empty()) {
            size_t i = _attempts.back();
            _attempts.pop_back();

            config = _space.get(i);

            if (_space.is_valid(config)) {
                return true;
            }
        }

        return false;
    }

  private:
    ConfigSpace _space;
    std::vector<size_t> _attempts;
};

ConfigIterator ConfigSpace::iterate() const {
    return *this;
}

Config Config::from_json(const nlohmann::json& json, const ConfigSpace& space) {
    return space.load_config(json);
}

}  // namespace kernel_launcher