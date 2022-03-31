#pragma once

#include <stdexcept>

#include "kernel_launcher/utils.hpp"
#include "nlohmann/json.hpp"

namespace kernel_launcher {

struct TunableValue;

struct CastException: std::runtime_error {
    CastException(const TunableValue& value, Type type);
};

struct TunableValue {
    template<typename T>
    struct TypeIndicator {};

    TunableValue(TunableValue&& val) {
        *this = std::move(val);
    }

    TunableValue(const TunableValue& val) {
        *this = val;
    }

    TunableValue() {}

    TunableValue(std::string value) :
        _type(type_string),
        _string_val(std::move(value)) {}

    TunableValue(const char* value) : TunableValue(std::string(value)) {}

    TunableValue(Type t) : TunableValue(t.name()) {}

    TunableValue(double i) : _type(type_double), _double_val(i) {}

    TunableValue(float i) : _type(type_double), _double_val(i) {}

    ~TunableValue() {
        clear();
    }

    void clear() {
        auto old_type = _type;
        _type = type_empty;

        switch (old_type) {
            case type_string:
                using std::string;
                _string_val.~string();
                break;
            case type_bool:
            case type_int:
            case type_double:
            case type_empty:
                break;
        }
    }

    TunableValue& operator=(TunableValue&& that) {
        *this = that;  // TODO: make actual move operator=
        return *this;
    }

    TunableValue& operator=(const TunableValue& val) {
        clear();

        switch (val._type) {
            case type_double:
                _type = type_double;
                _double_val = val._double_val;
                break;
            case type_int:
                _type = type_int;
                _int_val = val._int_val;
                break;
            case type_string:
                _type = type_string;
                _string_val = val._string_val;
                break;
            case type_empty:
                _type = type_empty;
                break;
            case type_bool:
                _type = type_bool;
                _bool_val = val._bool_val;
                break;
        }

        return *this;
    }

    bool is_empty() const {
        return _type == type_empty;
    }

    bool is_string() const {
        return !is_empty();
    }

    std::string to_string() const {
        switch (_type) {
            case type_int:
                return std::to_string(_int_val);
            case type_double:
                return std::to_string(_double_val);
            case type_string:
                return _string_val;
            case type_bool:
                return _bool_val ? "true" : "false";
            default:
                return "";
        }
    }

    bool is_double() const {
        return _type == type_double;
    }

    bool is_float() const {
        return is_double();
    }

    double to_double() const {
        switch (_type) {
            case type_double:
                return _double_val;
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

    bool operator==(const TunableValue& that) const {
        if (this->_type != that._type) {
            return false;
        }

        switch (_type) {
            case type_empty:
                return true;
            case type_int:
                return this->_int_val == that._int_val;
            case type_double:
                return this->_double_val == that._double_val;
            case type_string:
                return this->_string_val == that._string_val;
            case type_bool:
                return this->_bool_val == that._bool_val;
            default:
                return false;
        }
    }

    bool operator!=(const TunableValue& that) const {
        return !(*this == that);
    }

    bool operator<(const TunableValue& that) const {
        if (this->_type != that._type) {
            return this->_type < that._type;
        }

        switch (_type) {
            case type_empty:
                return false;
            case type_int:
                return this->_int_val < that._int_val;
            case type_double:
                return this->_double_val < that._double_val;
            case type_string:
                return this->_string_val < that._string_val;
            case type_bool:
                return this->_bool_val < that._bool_val;
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

        switch (_type) {
            case type_int:
                return _int_val;
            case type_double:
                return _double_val;
            case type_string:
                return _string_val;
            case type_bool:
                return _bool_val;
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
    TunableValue(type i) : _type(type_int), _int_val(i) {}          \
    bool is_##human_name() const {                                  \
        if (_type == type_bool) {                                   \
            return true;                                            \
        } else if (_type == type_int && in_range<type>(_int_val)) { \
            return true;                                            \
        }                                                           \
        return false;                                               \
    }                                                               \
    type to_##human_name() const {                                  \
        if (_type == type_bool) {                                   \
            return (type)_bool_val;                                 \
        } else if (_type == type_int && in_range<type>(_int_val)) { \
            return (type)_int_val;                                  \
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
    } _type = type_empty;

    union {
        intmax_t _int_val;
        double _double_val;
        bool _bool_val;
        std::string _string_val;
    };
};

CastException::CastException(const TunableValue& value, Type type) :
    std::runtime_error(
        value.to_string() + " cannot be cast to " + type.name()) {}

}  // namespace kernel_launcher