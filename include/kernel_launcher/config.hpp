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
    Config() = default;
    Config(Config&&) = default;
    explicit Config(const Config&) = default;
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

    bool operator==(const Config& that) const {
        return inner_ == that.inner_;
    }

    bool operator!=(const Config& that) const {
        return !operator==(that);
    }

  private:
    std::unordered_map<TunableParam, TunableValue> inner_;
};

struct ConfigSpace {
    ConfigSpace() = default;
    ConfigSpace(ConfigSpace&&) = default;
    explicit ConfigSpace(const ConfigSpace&) = default;
    ConfigSpace& operator=(ConfigSpace&&) = default;
    ConfigSpace& operator=(const ConfigSpace&) = delete;

    template<typename T, typename It>
    ParamExpr<T> tune(std::string name, It begin, It end, T default_value) {
        std::vector<TunableValue> tvalues;
        for (It current = begin; current != end; ++current) {
            T value = *current;
            tvalues.push_back(std::move(value));
        }

        return ParamExpr<T>(create_param(
            std::move(name),
            type_of<T>(),
            std::move(tvalues),
            default_value));
    }

    template<typename Collection, typename T>
    ParamExpr<T>
    tune(std::string name, const Collection& values, T default_type) {
        return tune(
            std::move(name),
            std::begin(values),
            std::end(values),
            std::move(default_type));
    }

    template<typename Collection, typename T = typename Collection::value_type>
    ParamExpr<T> tune(std::string name, const Collection& values) {
        KERNEL_LAUNCHER_ASSERT(values.begin() != values.end());
        return tune(std::move(name), values, *values.begin());
    }

    // Special case for initializer_list since it is not recognized by the function above
    template<typename T>
    ParamExpr<T> tune(
        std::string name,
        const std::initializer_list<T>& values,
        T default_type) {
        return tune(
            std::move(name),
            std::begin(values),
            std::end(values),
            std::move(default_type));
    }

    template<typename T>
    ParamExpr<T>
    tune(std::string name, const std::initializer_list<T>& values) {
        KERNEL_LAUNCHER_ASSERT(values.begin() != values.end());
        return tune(std::move(name), values, *values.begin());
    }

    const std::vector<TunableParam>& parameters() const {
        return params_;
    }

    const TunableParam& operator[](const char* s) const {
        return at(s);
    }

    TunableParam create_param(
        std::string name,
        Type type,
        std::vector<TunableValue> values,
        TunableValue default_value);
    const TunableParam& at(const char* s) const;
    void restrict(Expr<bool> expr);
    uint64_t size() const;
    bool get(uint64_t index, Config& config) const;
    bool is_valid(const Config& config) const;
    Config random_config() const;
    Config default_config() const;
    ConfigIterator iterate() const;

    Config load_config(const nlohmann::json& obj) const;
    nlohmann::json to_json() const;

  private:
    std::vector<TunableParam> params_;
    std::vector<Expr<bool>> restrictions_;
};

struct ConfigIterator {
    ConfigIterator() = default;
    ConfigIterator(ConfigIterator&&) = default;
    explicit ConfigIterator(const ConfigIterator&) = default;
    ConfigIterator& operator=(ConfigIterator&&) = default;
    ConfigIterator& operator=(const ConfigIterator&) = delete;

    ConfigIterator(ConfigSpace space) : space_(std::move(space)) {
        reset();
    }

    void reset();
    bool next(Config& config);

  private:
    ConfigSpace space_;
    uint64_t size_;
    std::default_random_engine rng_;
    std::unordered_set<uint64_t> visited_;
};

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::Config> {
    size_t operator()(const kernel_launcher::Config& config) const noexcept {
        size_t hash = 0;

        for (const auto& p : config.get()) {
            hash ^= p.first.hash() ^ p.second.hash();
        }

        return hash;
    }
};
}  // namespace std