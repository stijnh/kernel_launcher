#pragma once

#include "kernel_launcher/expr.hpp"
#include "kernel_launcher/utils.hpp"
#include "kernel_launcher/value.hpp"

namespace kernel_launcher {

struct ConfigSpace;
struct ConfigIterator;

struct Config {
    Config() = default;
    const TunableValue& operator[](const TunableParam& param) const {
        return at(param);
    }

    const TunableValue& at(const TunableParam& param) const;
    void insert(TunableParam p, TunableValue v);
    const std::unordered_map<TunableParam, TunableValue>& get() const;

    nlohmann::json to_json() const;
    static Config
    from_json(const nlohmann::json& json, const ConfigSpace& space);

  private:
    std::unordered_map<TunableParam, TunableValue> _inner;
};

struct ConfigSpace {
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

    const std::unordered_map<TunableParam, std::vector<TunableValue>>&
    parameters() const {
        return _params;
    }

    const TunableParam& operator[](std::string& s) const {
        return at(s);
    }

    TunableParam
    create_param(std::string name, Type type, std::vector<TunableValue> values);
    const TunableParam& at(std::string& s) const;
    void restrict(Expr<bool> expr);
    size_t size() const;
    bool get(size_t index, Config& config) const;
    bool is_valid(const Config& config) const;
    Config sample() const;
    ConfigIterator iterate() const;

    Config load_config(const nlohmann::json& obj) const;
    nlohmann::json to_json() const;

  private:
    std::unordered_map<TunableParam, std::vector<TunableValue>> _params;
    std::vector<Expr<bool>> _restrictions;
};

struct ConfigIterator {
    ConfigIterator() = default;
    ConfigIterator(ConfigSpace space) : _space(std::move(space)) {
        reset();
    }

    void reset();
    bool next(Config& config);

  private:
    ConfigSpace _space;
    std::vector<size_t> _attempts;
};

}  // namespace kernel_launcher