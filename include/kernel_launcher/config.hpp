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

    template<typename T, typename It>
    ParamExpr<T> tune(std::string name, It begin, It end, T default_value) {
        std::vector<TunableValue> tvalues;
        for (auto current = begin; current != end; ++current) {
            tvalues.push_back(*current);
        }

        return ParamExpr<T>(create_param(
            std::move(name),
            type_of<T>(),
            std::move(tvalues),
            default_value));
    }

    template<typename T>
    ParamExpr<T>
    tune(std::string name, const std::vector<T>& values, T default_value) {
        return tune(
            std::move(name),
            values.begin(),
            values.end(),
            std::move(default_value));
    }

    template<typename T>
    ParamExpr<T> tune(std::string name, const std::vector<T>& values) {
        return tune(std::move(name), values, values.at(0));
    }

    template<typename T>
    ParamExpr<T> tune(
        std::string name,
        const std::initializer_list<T>& values,
        T default_value) {
        return tune(
            std::move(name),
            values.begin(),
            values.end(),
            std::move(default_value));
    }

    template<typename T>
    ParamExpr<T>
    tune(std::string name, const std::initializer_list<T>& values) {
        KERNEL_LAUNCHER_ASSERT(values.size() > 0);
        return tune(std::move(name), values, *values.begin());
    }

    const std::vector<TunableParam>& parameters() const {
        return params_;
    }

    const TunableParam& operator[](std::string& s) const {
        return at(s);
    }

    TunableParam create_param(
        std::string name,
        Type type,
        std::vector<TunableValue> values,
        TunableValue default_value);
    const TunableParam& at(std::string& s) const;
    void restrict(Expr<bool> expr);
    uint64_t size() const;
    bool get(uint64_t index, Config& config) const;
    bool is_valid(const Config& config) const;
    Config sample_config() const;
    Config default_config() const;
    ConfigIterator iterate() const;

    Config load_config(const nlohmann::json& obj) const;
    nlohmann::json to_json() const;

  private:
    std::vector<TunableParam> params_;
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

namespace std {
template<>
struct hash<kernel_launcher::Config> {
    size_t operator()(const kernel_launcher::Config& config) const noexcept {
        size_t hash = 0;

        for (const auto& p : config.get()) {
            size_t left = std::hash<std::string> {}(p.first.name());
            size_t right =
                std::hash<kernel_launcher::TunableValue> {}(p.second);

            // Combine using XOR to ensure that the order of elements is not important while hashing.
            hash ^=
                right + 0x9e3779b9 + (left << 6) + (left >> 2);  // From BOOST
        }

        return hash;
    }
};
}  // namespace std