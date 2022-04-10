#pragma once

#include <iostream>
#include <new>
#include <stdexcept>

#include "kernel_launcher/utils.hpp"
#include "nlohmann/json.hpp"

namespace kernel_launcher {

const std::string& intern_string(const std::string& input);

struct TunableValue;

struct CastException: std::runtime_error {
    CastException(const TunableValue& value, Type type);
};

struct TunableValue {
    template<typename T>
    struct TypeIndicator {};

    TunableValue(TunableValue&& val) noexcept {
        *this = std::move(val);
    }

    TunableValue(const TunableValue& val) {
        *this = val;
    }

    TunableValue() {}

    TunableValue(const std::string& value) :
        type_(type_string),
        string_val_(&intern_string(value)) {}

    TunableValue(const char* value) : TunableValue(std::string(value)) {}

    TunableValue(Type t) : TunableValue(t.name()) {}

    TunableValue(TemplateArg t) : TunableValue(t.get()) {}

    TunableValue(double i) : type_(type_double), double_val_(i) {}

    TunableValue(float i) : type_(type_double), double_val_(i) {}

    template<typename T>
    TunableValue(std::initializer_list<T> list) :
        type_(type_list),
        list_val_({}) {
        for (const auto& p : list) {
            list_val_.push_back(TunableValue(p));
        }
    }

    template<typename T>
    TunableValue(std::vector<T> list) : type_(type_list), list_val_({}) {
        for (auto&& p : list) {
            list_val_.push_back(TunableValue(std::move(p)));
        }
    }

    template<typename T, size_t N>
    TunableValue(std::array<T, N> list) : type_(type_list), list_val_({}) {
        for (auto&& p : list) {
            list_val_.push_back(TunableValue(std::move(p)));
        }
    }

    template<typename A, typename B>
    TunableValue(std::pair<A, B> list) : type_(type_list), list_val_({}) {
        list_val_.push_back(std::move(list.first));
        list_val_.push_back(std::move(list.second));
    }

    ~TunableValue() {
        clear();
    }

    void clear() {
        auto oldtype = type_;
        type_ = type_empty;

        switch (oldtype) {
            case type_list:
                using std::vector;
                list_val_.~vector();
                break;
            case type_string:
            case type_bool:
            case type_int:
            case type_double:
            case type_empty:
                break;
        }
    }

    TunableValue& operator=(const TunableValue& val) {
        auto old_type = type_;
        auto new_type = val.type_;

        if (old_type == type_list && new_type == type_list && 0) {
            list_val_ = val.list_val_;
        } else {
            clear();  // Sets type_ to empty

            switch (new_type) {
                case type_double:
                    double_val_ = val.double_val_;
                    break;
                case type_int:
                    int_val_ = val.int_val_;
                    break;
                case type_bool:
                    bool_val_ = val.bool_val_;
                    break;
                case type_string:
                    string_val_ = val.string_val_;
                    break;
                case type_list:
                    new (&list_val_) std::vector<TunableValue>(val.list_val_);
                    break;
                case type_empty:
                    break;
            }

            type_ =
                new_type;  // Set type_ now since the copy constructors might throw
        }

        return *this;
    }

    TunableValue& operator=(TunableValue&& that) noexcept {
        clear();
        auto new_type = that.type_;
        that.type_ = type_empty;

        switch (new_type) {
            case type_double:
                double_val_ = that.double_val_;
                break;
            case type_int:
                int_val_ = that.int_val_;
                break;
            case type_bool:
                bool_val_ = that.bool_val_;
                break;
            case type_string:
                string_val_ = that.string_val_;
                break;
            case type_list:
                new (&list_val_)
                    std::vector<TunableValue>(std::move(that.list_val_));
                break;
            case type_empty:
                break;
        }

        type_ =
            new_type;  // type_ now since the move constructors might throw (is it true?)
        return *this;
    }

    bool operator==(const TunableValue& that) const {
        if (this->type_ != that.type_) {
            return false;
        }

        switch (type_) {
            case type_empty:
                return true;
            case type_int:
                return this->int_val_ == that.int_val_;
            case type_double:
                return this->double_val_ == that.double_val_;
            case type_string:
                return *this->string_val_ == *that.string_val_;
            case type_bool:
                return this->bool_val_ == that.bool_val_;
            case type_list:
                return this->list_val_ == that.list_val_;
            default:
                return false;
        }
    }

    bool operator!=(const TunableValue& that) const {
        return !(*this == that);
    }

    bool operator<(const TunableValue& that) const {
        if (this->type_ != that.type_) {
            return this->type_ < that.type_;
        }

        switch (type_) {
            case type_empty:
                return false;
            case type_int:
                return this->int_val_ < that.int_val_;
            case type_double:
                return this->double_val_ < that.double_val_;
            case type_string:
                return *this->string_val_ < *that.string_val_;
            case type_bool:
                return this->bool_val_ < that.bool_val_;
            case type_list:
                return this->list_val_ < that.list_val_;
            default:
                return false;
        }
    }

    bool operator>(const TunableValue& that) const {
        return that < *this;
    }

    bool operator<=(const TunableValue& that) const {
        return *this < that || *this == that;
    }

    bool operator>=(const TunableValue& that) const {
        return that <= *this;
    }

    size_t hash() const {
        switch (type_) {
            case type_int:
                return std::hash<intmax_t> {}(int_val_);
            case type_double:
                return std::hash<double> {}(double_val_);
            case type_string:
                return std::hash<std::string> {}(*string_val_);
            case type_bool:
                return std::hash<bool> {}(bool_val_);
            case type_list: {
                size_t h = 0;
                for (const auto& p : list_val_) {
                    h ^= p.hash();  // TODO: find better hash combiner
                }

                return h;
            }
            default:
                return 0;
        }
    }

    bool is_empty() const {
        return type_ == type_empty;
    }

    bool is_string() const {
        return !is_empty();
    }

    std::string to_string() const {
        switch (type_) {
            case type_int:
                return std::to_string(int_val_);
            case type_double:
                return std::to_string(double_val_);
            case type_string:
                return *string_val_;
            case type_bool:
                return bool_val_ ? "true" : "false";
            case type_list: {
                std::stringstream out;
                out << "[";
                bool is_first = true;

                for (const auto& p : list_val_) {
                    if (is_first) {
                        is_first = false;
                    } else {
                        out << ", ";
                    }

                    out << p.to_string();
                }

                out << "]";
                return out.str();
            }
            default:
                return "";
        }
    }

    bool is_double() const {
        return type_ == type_double;
    }

    bool is_float() const {
        return is_double();
    }

    double to_double() const {
        switch (type_) {
            case type_double:
                return double_val_;
            default:
                throw CastException(*this, type_of<double>());
        }
    }

    float to_float() const {
        return (float)to_double();
    }

    explicit operator double() const {
        return to_double();
    }

    explicit operator float() const {
        return to_float();
    }

    bool is_list() const {
        return type_ == type_list;
    }

    template<typename T>
    bool is_vector() const {
        if (!is_list()) {
            return false;
        }

        for (const auto& p : list_val_) {
            if (!p.is<T>()) {
                return false;
            }
        }

        return true;
    }

    template<typename T>
    std::vector<T> to_vector() const {
        if (!is_vector<T>()) {
            throw CastException(*this, type_of<std::vector<T>>());
        }

        std::vector<T> result;
        for (const auto& p : list_val_) {
            result.push_back(p.to(TypeIndicator<T> {}));
        }
        return result;
    }

    template<typename T>
    bool is(TypeIndicator<std::vector<T>>) const {
        return is_vector<T>();
    }

    template<typename T>
    std::vector<T> to(TypeIndicator<std::vector<T>>) const {
        return to_vector<T>();
    }

    template<typename A, typename B>
    bool is_pair() const {
        return is_list() && list_val_.size() == 2 && list_val_[0].is<A>()
            && list_val_[1].is<B>();
    }

    template<typename A, typename B>
    std::pair<A, B> to_pair() const {
        if (!is_pair<A, B>()) {
            throw CastException(*this, type_of<std::pair<A, B>>());
        }

        return {
            list_val_[0].to(TypeIndicator<A>()),
            list_val_[1].to(TypeIndicator<B>())};
    }

    template<typename A, typename B>
    bool is(TypeIndicator<std::pair<A, B>>) const {
        return is_pair<A, B>();
    }

    template<typename A, typename B>
    std::pair<A, B> to(TypeIndicator<std::pair<A, B>>) const {
        return to_pair<A, B>();
    }

    template<typename T, size_t N>
    bool is_array() const {
        if (!is_list() && list_val_.size() == N) {
            return false;
        }

        for (const auto& p : list_val_) {
            if (!p.is<T>()) {
                return false;
            }
        }

        return true;
    }

    template<typename T, size_t N>
    std::array<T, N> to_array() const {
        if (!is_array<T, N>()) {
            throw CastException(*this, type_of<std::array<T, N>>());
        }

        std::array<T, N> result;
        for (size_t i = 0; i < N; i++) {
            result[i] = list_val_[i].to(TypeIndicator<T>());
        }
        return result;
    }

    template<typename T, size_t N>
    bool is(TypeIndicator<std::array<T, N>>) const {
        return is_array<T, N>();
    }

    template<typename T, size_t N>
    std::array<T, N> to(TypeIndicator<std::array<T, N>>) const {
        return to_array<T, N>();
    }

    template<typename T>
    bool is() const {
        return is(TypeIndicator<T> {});
    }

    template<typename T>
    T to() const {
        return to(TypeIndicator<T> {});
    }

    nlohmann::json to_json() const {
        using nlohmann::json;

        switch (type_) {
            case type_int:
                return int_val_;
            case type_double:
                return double_val_;
            case type_string:
                return *string_val_;
            case type_bool:
                return bool_val_;
            case type_list: {
                std::vector<json> result;
                for (const auto& p : list_val_) {
                    result.push_back(p.to_json());
                }

                return result;
            }
            case type_empty:
                break;  // fallthrough
        }

        return nullptr;
    }

    static TunableValue from_json(const nlohmann::json& obj) {
        using nlohmann::json;

        switch (obj.type()) {
            case json::value_t::null:
            case json::value_t::discarded:
                return {};
            case json::value_t::string:
                return (std::string)obj;
            case json::value_t::boolean:
                return (bool)obj;
            case json::value_t::number_integer:
                return (json::number_integer_t)obj;
            case json::value_t::number_unsigned:
                return (json::number_unsigned_t)obj;
            case json::value_t::number_float:
                return (json::number_float_t)obj;
            case json::value_t::array: {
                std::vector<TunableValue> result;
                for (const auto& item : obj) {
                    result.push_back(TunableValue::from_json(item));
                }

                return result;
            }
            default:
                break;  // fallthrough
        }

        throw std::runtime_error("unknown json object");
    }

  private:
    bool is(TypeIndicator<TunableValue>) const {
        return true;
    }

    bool is(TypeIndicator<std::string>) const {
        return this->is_string();
    }

    bool is(TypeIndicator<double>) const {
        return this->is_double();
    }

    bool is(TypeIndicator<float>) const {
        return this->is_double();
    }

    TunableValue to(TypeIndicator<TunableValue>) const {
        return *this;
    }

    std::string to(TypeIndicator<std::string>) const {
        return this->to_string();
    }

    double to(TypeIndicator<double>) const {
        return this->to_double();
    }

    float to(TypeIndicator<float>) const {
        return this->to_float();
    }

#define FOR_INTEGER(type, human_name)                               \
  public:                                                           \
    TunableValue(type i) : type_(type_int), int_val_(i) {}          \
    bool is_##human_name() const {                                  \
        if (type_ == type_bool) {                                   \
            return true;                                            \
        } else if (type_ == type_int && in_range<type>(int_val_)) { \
            return true;                                            \
        }                                                           \
        return false;                                               \
    }                                                               \
    type to_##human_name() const {                                  \
        if (type_ == type_bool) {                                   \
            return (type)bool_val_;                                 \
        } else if (type_ == type_int && in_range<type>(int_val_)) { \
            return (type)int_val_;                                  \
        }                                                           \
        throw CastException(*this, type_of<type>());                \
    }                                                               \
    explicit operator type() const {                                \
        return to_##human_name();                                   \
    }                                                               \
                                                                    \
  private:                                                          \
    bool is(TypeIndicator<type>) const {                            \
        return this->is_##human_name();                             \
    }                                                               \
    type to(TypeIndicator<type>) const {                            \
        return this->to_##human_name();                             \
    }

    FOR_INTEGER(char, char)
    FOR_INTEGER(short, short)
    FOR_INTEGER(int, int)
    FOR_INTEGER(long, long)
    FOR_INTEGER(long long, longlong)
    FOR_INTEGER(unsigned char, uchar)
    FOR_INTEGER(unsigned short, ushort)
    FOR_INTEGER(unsigned int, uint)
    FOR_INTEGER(unsigned long, ulong)
    FOR_INTEGER(unsigned long long, ulonglong)
    FOR_INTEGER(bool, bool)
#undef FOR_INTEGER

  private:
    enum {
        type_empty,
        type_int,
        type_double,
        type_string,
        type_bool,
        type_list,
    } type_ = type_empty;

    union {
        intmax_t int_val_;
        double double_val_;
        bool bool_val_;
        std::vector<TunableValue> list_val_;
        const std::string* string_val_;
    };
};

inline CastException::CastException(const TunableValue& value, Type type) :
    std::runtime_error(
        value.to_string() + " cannot be cast to " + type.name()) {}

struct TunableParam {
  private:
    struct Impl {
        friend TunableParam;

        Impl(
            std::string name,
            Type type,
            std::vector<TunableValue> values,
            TunableValue default_value) :
            name_(std::move(name)),
            type_(std::move(type)),
            values_(std::move(values)),
            default_value_(std::move(default_value)) {}

      private:
        std::string name_;
        Type type_;
        std::vector<TunableValue> values_;
        TunableValue default_value_;
    };

  public:
    TunableParam(
        std::string name,
        Type type,
        std::vector<TunableValue> values,
        TunableValue default_value) {
        inner_ = std::make_shared<Impl>(
            std::move(name),
            std::move(type),
            std::move(values),
            std::move(default_value));
    }

    const std::string& name() const {
        return inner_->name_;
    }

    size_t hash() const {
        return (size_t)inner_.get();
    }

    Type type() const {
        return inner_->type_;
    }

    const TunableValue& default_value() const {
        return inner_->default_value_;
    }

    const std::vector<TunableValue>& values() const {
        return inner_->values_;
    }

    const TunableValue& at(size_t i) const {
        return values().at(i);
    }

    const TunableValue& operator[](size_t i) const {
        return at(i);
    }

    size_t size() const {
        return values().size();
    }

    bool operator==(const TunableParam& that) const {
        return inner_.get() == that.inner_.get();
    }

    bool operator!=(const TunableParam& that) const {
        return !(*this == that);
    }

  private:
    std::shared_ptr<Impl> inner_;
};

}  // namespace kernel_launcher

namespace std {

template<>
struct hash<kernel_launcher::TunableParam> {
    std::size_t operator()(const kernel_launcher::TunableParam& k) const {
        return k.hash();
    }
};

template<>
struct hash<kernel_launcher::TunableValue> {
    size_t operator()(const kernel_launcher::TunableValue& val) const noexcept {
        return val.hash();
    }
};
}  // namespace std