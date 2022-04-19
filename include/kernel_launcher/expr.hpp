#pragma once

#include <sstream>
#include <vector>

#include "kernel_launcher/utils.hpp"
#include "kernel_launcher/value.hpp"

namespace kernel_launcher {

template<typename R>
struct BaseExpr;

using TunableMap = std::unordered_map<TunableParam, TunableValue>;
static const TunableMap EMPTY_CONFIG = {};

struct Eval {
    Eval(const TunableMap& mapping = EMPTY_CONFIG) : inner_(mapping) {
        //
    }

    template<typename T>
    T lookup(const TunableParam& param) const {
        return inner_.at(param).to<T>();
    }

    template<typename R>
    R eval(const BaseExpr<R>& expr) const {
        return expr.eval(*this);
    }

    template<typename R>
    R operator()(const BaseExpr<R>& expr) const {
        return eval(expr);
    }

  private:
    const TunableMap& inner_;
};

template<typename R>
struct BaseExpr {
    using return_type = R;

    virtual ~BaseExpr() = default;
    virtual std::string to_string() const = 0;
    virtual return_type eval(const Eval& eval) const = 0;

#if KERNEL_LAUNCHER_JSON
    virtual nlohmann::json to_json() const {
        throw std::runtime_error(
            "expression cannot be converted to json: " + to_string());
    }
#endif
};

template<typename T>
struct ParamExpr: BaseExpr<T> {
    ParamExpr(TunableParam p) : _param(std::move(p)) {
        //
    }

    std::string to_string() const override {
        return "$" + _param.name();
    }

    T eval(const Eval& eval) const override {
        return eval.lookup<T>(_param);
    }

#if KERNEL_LAUNCHER_JSON
    nlohmann::json to_json() const override {
        return {
            {"operator", "parameter"},
            {"name", _param.name()},
        };
    }
#endif

    const TunableParam& parameter() const {
        return _param;
    }

  private:
    TunableParam _param;
};

template<typename T>
struct ScalarExpr: BaseExpr<T> {
    ScalarExpr(T value) : _value(std::move(value)) {
        //
    }

    std::string to_string() const override {
        std::stringstream oss;
        oss << _value;
        return oss.str();
    }

    T eval(const Eval&) const override {
        return _value;
    }

#if KERNEL_LAUNCHER_JSON
    // TODO: Add to_json for arbitrary T
    //    nlohmann::json to_json() const override {
    //        return TunableValue(_value).to_json();
    //    }
#endif

  private:
    T _value;
};

template<typename T>
ScalarExpr<T> scalar(T value = {}) {
    return ScalarExpr<T>(std::move(value));
}

template<typename F, typename R = typename std::result_of<F(const Eval&)>::type>
struct FunExpr: BaseExpr<R> {
    FunExpr(std::string name, F fun) : _name(name), _fun(fun) {
        //
    }

    FunExpr(F fun) : _fun(std::move(fun)) {
        std::stringstream oss;
        oss << "<anonymous: " << Type::of<F>().name() << ">";
        _name = oss.str();
    }

    R eval(const Eval& eval) const override {
        return _fun(eval);
    }

    std::string to_string() const override {
        return _name;
    }

  private:
    std::string _name;
    F _fun;
};

template<class...>
using void_t = void;

namespace detail {
    template<typename T>
    std::true_type is_expr_helper(const BaseExpr<T>*);
    std::false_type is_expr_helper(...);
}  // namespace detail

template<typename T>
constexpr bool is_expr =
    decltype(detail::is_expr_helper(std::declval<T*>()))::value;

namespace detail {
    template<typename T, typename = void>
    struct ToExprHelper {
        using type = ScalarExpr<T>;

        static ScalarExpr<T> convert(T value) {
            return ScalarExpr<T>(std::move(value));
        }
    };

    template<typename F>
    struct ToExprHelper<
        F,
        void_t<typename std::result_of<F(const Eval&)>::type>> {
        using type = FunExpr<F>;

        static FunExpr<F> convert(F fun) {
            return FunExpr<F>(std::move(fun));
        }
    };

    template<typename T>
    struct ToExprHelper<T, typename std::enable_if<is_expr<T>>::type> {
        using type = T;

        static T convert(T value) {
            return value;
        }
    };
}  // namespace detail

template<typename T>
using expr_type = typename detail::ToExprHelper<T>::type;

template<typename T>
expr_type<T> expr(T value) {
    return detail::ToExprHelper<T>::convert(std::move(value));
}
#define UNARY_OP_IMPL(class_name, fun, op)                                     \
    template<                                                                  \
        typename T,                                                            \
        typename O = decltype(fun(std::declval<typename T::return_type>()))>   \
    struct class_name: BaseExpr<O> {                                           \
        class_name(T input) : _input(std::move(input)) {}                      \
                                                                               \
        O eval(const Eval& eval) const override {                              \
            return fun(eval(_input));                                          \
        }                                                                      \
                                                                               \
        std::string to_string() const override {                               \
            return "(" #op + _input.to_string() + ")";                         \
        }                                                                      \
                                                                               \
        nlohmann::json to_json() const override {                              \
            return {                                                           \
                {"operator", #op},                                             \
                {"operand", _input.to_json()},                                 \
            };                                                                 \
        }                                                                      \
                                                                               \
      private:                                                                 \
        T _input;                                                              \
    };                                                                         \
                                                                               \
    template<typename L, typename = typename std::enable_if<is_expr<L>>::type> \
    auto operator op(L lhs) {                                                  \
        return class_name<expr_type<L>>(expr(std::move(lhs)));                 \
    }

UNARY_OP_IMPL(NegExpr, std::negate<void> {}, -)
UNARY_OP_IMPL(NotExpr, std::logical_not<void> {}, !)
UNARY_OP_IMPL(InvExpr, std::bit_not<void> {}, ~)
#undef UNARY_OP_IMPL

#define BINARY_OP_IMPL(class_name, fun, op)                                 \
    template<                                                               \
        typename L,                                                         \
        typename R,                                                         \
        typename O = decltype(                                              \
            fun(std::declval<typename L::return_type>(),                    \
                std::declval<typename R::return_type>()))>                  \
    struct class_name: BaseExpr<O> {                                        \
        class_name(L lhs, R rhs) :                                          \
            _lhs(std::move(lhs)),                                           \
            _rhs(std::move(rhs)) {}                                         \
                                                                            \
        O eval(const Eval& eval) const override {                           \
            return fun(eval(_lhs), eval(_rhs));                             \
        }                                                                   \
                                                                            \
        std::string to_string() const override {                            \
            return "(" + _lhs.to_string() + #op + _rhs.to_string() + ")";   \
        }                                                                   \
                                                                            \
        nlohmann::json to_json() const override {                           \
            return {                                                        \
                {"operator", #op},                                          \
                {"left", _lhs.to_json()},                                   \
                {"right", _rhs.to_json()},                                  \
            };                                                              \
        }                                                                   \
                                                                            \
      private:                                                              \
        L _lhs;                                                             \
        R _rhs;                                                             \
    };                                                                      \
                                                                            \
    template<                                                               \
        typename L,                                                         \
        typename R,                                                         \
        typename = typename std::enable_if<is_expr<L> || is_expr<R>>::type> \
    auto operator op(L lhs, R rhs) {                                        \
        return class_name<expr_type<L>, expr_type<R>>(                      \
            expr(std::move(lhs)),                                           \
            expr(std::move(rhs)));                                          \
    }

BINARY_OP_IMPL(AddExpr, std::plus<void> {}, +)
BINARY_OP_IMPL(MulExpr, std::multiplies<void> {}, *)
BINARY_OP_IMPL(SubExpr, std::minus<void> {}, -)
BINARY_OP_IMPL(DivExpr, std::divides<void> {}, /)
BINARY_OP_IMPL(ModExpr, std::modulus<void> {}, %)

BINARY_OP_IMPL(OrExpr, std::bit_or<void> {}, |)
BINARY_OP_IMPL(AndExpr, std::bit_and<void> {}, &)
BINARY_OP_IMPL(XorExpr, std::bit_xor<void> {}, ^)

BINARY_OP_IMPL(EqExpr, std::equal_to<void> {}, ==)
BINARY_OP_IMPL(NeqExpr, std::not_equal_to<void> {}, !=)
BINARY_OP_IMPL(LtExpr, std::less<void> {}, <)
BINARY_OP_IMPL(GtExpr, std::greater<void> {}, >)
BINARY_OP_IMPL(LteExpr, std::less_equal<void> {}, <=)
BINARY_OP_IMPL(GteExpr, std::greater_equal<void> {}, >=)

// This are rarely used and they cause a lot of confusion when expressions
// are used in ostreams/istreams.
// BINARY_OP_IMPL(ShlOp, <<)
// BINARY_OP_IMPL(ShrOp, >>)
#undef BINARY_OP_IMPL

namespace detail {
    template<typename A, typename B, typename = void>
    struct CastHelper {
        static B call(A val) {
            return val;
        }
    };

    template<>
    struct CastHelper<bool, std::string> {
        static std::string call(bool val) {
            return val ? "true" : "false";
        }
    };

    template<typename T>
    struct CastHelper<
        T,
        std::string,
        typename std::enable_if<
            std::is_integral<T>::value
            || std::is_floating_point<T>::value>::type> {
        static std::string call(T val) {
            return std::to_string(val);
        }
    };

    template<typename I>
    struct CastHelper<I, bool> {
        static bool call(I val) {
            return (bool)val;
        }
    };

    template<typename I, typename O>
    struct CastHelper<
        I,
        O,
        typename std::enable_if<
            std::is_integral<I>::value && std::is_integral<O>::value
            && !std::is_same<O, bool>::value>::type> {
        static O call(I val) {
            if (!in_range<O>(val)) {
                throw std::runtime_error(
                    "invalid cast of " + std::to_string(val) + " from "
                    + Type::of<I>().name() + " to " + Type::of<O>().name());
            }

            return (O)val;
        }
    };
}  // namespace detail

template<typename E, typename O, typename I = typename E::return_type>
struct ConvertExpr: BaseExpr<O> {
    ConvertExpr(E value) : inner_(std::move(value)) {
        //
    }

    std::string to_string() const override {
        return inner_.to_string();
    }

    O eval(const Eval& eval) const override {
        return detail::CastHelper<I, O>::call(eval(inner_));
    }

#if KERNEL_LAUNCHER_JSON
    nlohmann::json to_json() const override {
        return {
            {"operator", "convert"},
            {"type", type_of<O>().name()},
            {"operand", inner_.to_json()},
        };
    }
#endif

  private:
    E inner_;
};

template<typename E, typename I>
struct ConvertExpr<E, I, I>: BaseExpr<I> {
    ConvertExpr(E value) : inner_(std::move(value)) {
        //
    }

    std::string to_string() const override {
        return inner_.to_string();
    }

    I eval(const Eval& eval) const override {
        return eval(inner_);
    }

#if KERNEL_LAUNCHER_JSON
    nlohmann::json to_json() const override {
        return inner_.to_json();
    }
#endif

  private:
    E inner_;
};

template<typename O, typename E>
auto cast(E val) {
    return ConvertExpr<expr_type<E>, O>(expr(std::move(val)));
}

template<
    typename C,
    typename L,
    typename R,
    typename O = typename std::
        common_type<typename L::return_type, typename R::return_type>::type>
struct CondExpr: BaseExpr<O> {
    CondExpr(C cond, L left, R right) :
        cond_(std::move(cond)),
        left_(std::move(left)),
        right_(std::move(right)) {}

    std::string to_string() const override {
        return "(" + cond_.to_string() + " ? " + left_.to_string() + " : "
            + right_.to_string() + ")";
    }

    O eval(const Eval& eval) const override {
        return eval(cond_) ? eval(left_) : eval(right_);
    }

#if KERNEL_LAUNCHER_JSON
    nlohmann::json to_json() const override {
        return {
            {"operator", "conditional"},
            {"condition", cond_.to_json()},
            {"left", left_.to_json()},
            {"right", right_.to_json()},
        };
    }
#endif

  private:
    C cond_;
    L left_;
    R right_;
};

template<typename C, typename L, typename R>
auto ifelse(C cond, L left, R right) {
    return CondExpr<expr_type<C>, expr_type<L>, expr_type<R>> {
        expr(std::move(cond)),
        expr(std::move(left)),
        expr(std::move(right))};
}

template<typename A, typename B>
auto div_ceil(A left, B right) {
    return (left / right) + (left % right != 0);
}

template<typename T>
struct Expr: BaseExpr<T> {
    template<typename S, typename = expr_type<S>>
    Expr(S value) {
        auto e = cast<T>(expr(std::move(value)));
        inner_ = std::make_shared<decltype(e)>(e);
    }

    T eval(const Eval& eval) const override {
        return inner_->eval(eval);
    }

    std::string to_string() const override {
        return inner_->to_string();
    }

#if KERNEL_LAUNCHER_JSON
    nlohmann::json to_json() const override {
        return inner_->to_json();
    }
#endif

  private:
    std::shared_ptr<BaseExpr<T>> inner_ {};
};

}  // namespace kernel_launcher

#if KERNEL_LAUNCHER_HEADERONLY
    #include KERNEL_LAUNCHER_IMPL("expr.cpp")
#endif
