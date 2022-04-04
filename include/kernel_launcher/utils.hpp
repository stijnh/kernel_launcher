#pragma once

#include <cxxabi.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace kernel_launcher {

struct Type {
    Type(const std::type_info& t) : _inner(t) {
        //
    }

    template<typename T>
    static inline Type of() {
        return Type(typeid(T));
    }

    const std::type_info& get() const {
        return _inner;
    }

    const std::string& name() const;

    bool operator==(const Type& that) {
        return this->_inner == that._inner;
    }

    bool operator!=(const Type& that) {
        return !(*this == that);
    }

  private:
    const std::type_info& _inner;
};

template<typename T>
static inline Type type_of() {
    return Type::of<T>();
}

template<typename T>
static inline Type type_of(const T&) {
    return Type::of<T>();
}

static inline std::ostream& operator<<(std::ostream& os, const Type& t) {
    return os << t.name();
}

struct TemplateArg {
#define CONSTRUCTOR(type)                                        \
    TemplateArg(type i) {                                        \
        _inner = std::string("(" #type ")") + std::to_string(i); \
    }
    CONSTRUCTOR(signed char);
    CONSTRUCTOR(short);
    CONSTRUCTOR(int);
    CONSTRUCTOR(long);
    CONSTRUCTOR(long long);
    CONSTRUCTOR(unsigned char);
    CONSTRUCTOR(unsigned short);
    CONSTRUCTOR(unsigned int);
    CONSTRUCTOR(unsigned long);
    CONSTRUCTOR(unsigned long long);
    CONSTRUCTOR(float);
    CONSTRUCTOR(double);
#undef CONSTRUCTOR

    TemplateArg(bool b) {
        _inner = b ? "(bool)true" : "(bool)false";
    }

    TemplateArg(Type type) {
        _inner = type.name();
    }

    template<typename T>
    static TemplateArg from_type() {
        return TemplateArg(Type::of<T>());
    }

    static TemplateArg from_string(std::string s) {
        TemplateArg t(0);
        t._inner = std::move(s);
        return t;
    }

    const std::string& get() const {
        return _inner;
    }

  private:
    std::string _inner;
};

template<typename T>
TemplateArg template_arg(const T& value) {
    return TemplateArg(value);
}

template<typename T>
TemplateArg template_type() {
    return TemplateArg(typeid(T));
}

struct TunableParam {
  private:
    struct Impl {
        friend TunableParam;

        Impl(std::string name, Type type) :
            _name(std::move(name)),
            _type(std::move(type)) {
            static std::atomic<uint64_t> COUNTER = {1};
            _key = COUNTER++;
        }

      private:
        uint64_t _key;
        std::string _name;
        Type _type;
    };

  public:
    TunableParam(std::string name, Type type) {
        _inner = std::make_shared<Impl>(std::move(name), std::move(type));
    }

    const std::string& name() const {
        return _inner->_name;
    }

    uint64_t key() const {
        return _inner->_key;
    }

    Type type() const {
        return _inner->_type;
    }

    bool operator==(const TunableParam& that) const {
        return _inner.get() == that._inner.get();
    }

    bool operator!=(const TunableParam& that) const {
        return !(*this == that);
    }

  private:
    std::shared_ptr<Impl> _inner;
};

template<
    typename L,
    typename R,
    typename std::enable_if<
        std::is_signed<L>::value == std::is_signed<R>::value,
        std::nullptr_t>::type = nullptr>
bool cmp_less(L left, R right) {
    return left < right;
}

template<
    typename L,
    typename R,
    typename std::enable_if<
        std::is_signed<L>::value && !std::is_signed<R>::value,
        std::nullptr_t>::type = nullptr>
bool cmp_less(L left, R right) {
    using UL = std::make_unsigned_t<L>;
    return left < 0 || UL(left) < right;
}

template<
    typename L,
    typename R,
    typename std::enable_if<
        !std::is_signed<L>::value && std::is_signed<R>::value,
        std::nullptr_t>::type = nullptr>
bool cmp_less(L left, R right) {
    using UR = std::make_unsigned_t<R>;
    return right >= 0 && left < UR(right);
}

template<typename T, typename L, typename R>
bool in_range(T val, L min, R max) {
    return !cmp_less(val, min) && !cmp_less(max, val);
}

template<typename R, typename T>
bool in_range(T val) {
    return in_range(
        val,
        std::numeric_limits<R>::min(),
        std::numeric_limits<R>::max());
}

template<typename T>
std::vector<T> range(T start, T end, T step) {
    std::vector<T> results;
    while (start < end) {
        results.push_back(start);
        start += step;
    }

    return results;
}

template<typename T>
std::vector<T> range(T start, T end) {
    return range(start, end, T {1});
}

template<typename T>
std::vector<T> range(T end) {
    return range(T {0}, end);
}

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::TunableParam> {
    std::size_t operator()(const kernel_launcher::TunableParam& k) const {
        return k.key();
    }
};
}  // namespace std