#pragma once

#include "kernel_launcher/expr.hpp"
#include "kernel_launcher/utils.hpp"
#include "kernel_launcher/value.hpp"

namespace kernel_launcher {

struct ConfigSpace;
struct ConfigIterator;

struct Config {
    Config() = default;
    explicit Config(const Config&) = default;
    Config(Config&&) = default;
    Config& operator=(Config&&) = default;
    Config& operator=(const Config&) = delete;

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
    std::unordered_map<TunableParam, TunableValue> inner_;
};

struct ConfigSpace {
    ConfigSpace() = default;
    explicit ConfigSpace(const ConfigSpace&) = default;

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
        return params_;
    }

    const TunableParam& operator[](std::string& s) const {
        return at(s);
    }

    TunableParam
    create_param(std::string name, Type type, std::vector<TunableValue> values);
    const TunableParam& at(std::string& s) const;
    void restrict(Expr<bool> expr);
    uint64_t size() const;
    bool get(uint64_t index, Config& config) const;
    bool is_valid(const Config& config) const;
    Config sample() const;
    ConfigIterator iterate() const;

    Config load_config(const nlohmann::json& obj) const;
    nlohmann::json to_json() const;

  private:
    std::unordered_map<TunableParam, std::vector<TunableValue>> params_;
    std::vector<Expr<bool>> restrictions_;
};

struct ConfigIterator {
    explicit ConfigIterator(const ConfigIterator&) = default;
    ConfigIterator() = default;
    ConfigIterator(ConfigSpace space) : space_(std::move(space)) {
        reset();
    }

    void reset();
    bool next(Config& config);

  private:
    ConfigSpace space_;
    uint64_t index_;
    uint64_t size_;
    uint64_t log4_;
    std::array<uint32_t, 4> murmur_rounds_;
};

}  // namespace kernel_launcher