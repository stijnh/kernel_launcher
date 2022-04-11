#pragma once

#include <iostream>
#include <new>
#include <stdexcept>

#include "kernel_launcher/utils.hpp"
#include "nlohmann/json.hpp"

namespace kernel_launcher {

const std::string& intern_string(const char* input);

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

    TunableValue(const char* value) :
        type_(type_string),
        string_val_(&intern_string(value)) {}

    TunableValue(const std::string& value) : TunableValue(value.c_str()) {}

    TunableValue(Type t) : TunableValue(t.name()) {}

    TunableValue(TemplateArg t) : TunableValue(t.get()) {}

    TunableValue(double i) : type_(type_double), double_val_(i) {}

    TunableValue(float i) : type_(type_double), double_val_(i) {}

    TunableValue(bool b) : type_(type_bool), bool_val_(b) {}

    ~TunableValue() {
        clear();
    }

    void clear() {
        type_ = type_empty;
    }

    TunableValue& operator=(const TunableValue& val) {
        type_ = val.type_;
        switch (type_) {
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
            case type_empty:
                break;
        }

        return *this;
    }

    TunableValue& operator=(TunableValue&& that) noexcept {
        type_ = that.type_;
        that.type_ = type_empty;

        switch (type_) {
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
            case type_empty:
                break;
        }

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
                // ptr comparison is sufficient since strings are interned
                return this->string_val_ == that.string_val_;
            case type_bool:
                return this->bool_val_ == that.bool_val_;
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

    bool is_bool() const {
        if (type_ == type_bool) {
            return true;
        } else if (type_ == type_int && in_range<bool>(int_val_)) {
            return true;
        }

        return false;
    }

    bool to_bool() const {
        if (type_ == type_bool) {
            return bool_val_;
        } else if (type_ == type_int && in_range<bool>(int_val_)) {
            return (bool)int_val_;
        }

        throw CastException(*this, type_of<bool>());
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

    bool is(TypeIndicator<bool>) const {
        return this->is_bool();
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

    bool to(TypeIndicator<bool>) const {
        return this->to_bool();
    }

  public:
    explicit operator bool() const {
        return to_bool();
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
#undef FOR_INTEGER

  private:
    enum {
        type_empty,
        type_int,
        type_double,
        type_string,
        type_bool,
    } type_ = type_empty;

    union {
        intmax_t int_val_;
        double double_val_;
        bool bool_val_;
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