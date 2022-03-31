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
    Eval(const TunableMap& mapping = EMPTY_CONFIG) : _inner(mapping) {
        //
    }

    template<typename T>
    T lookup(const TunableParam& param) const {
        return _inner.at(param).to<T>();
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
    const TunableMap& _inner;
};

template<typename R>
struct BaseExpr {
    using return_type = R;

    virtual std::string name() const = 0;
    virtual return_type eval(const Eval& eval) const = 0;
    virtual nlohmann::json to_json() const {
        throw std::runtime_error(
            "expression cannot be converted to json: " + name());
    }
};

template<typename T>
struct ParamExpr: BaseExpr<T> {
    ParamExpr(TunableParam p) : _param(std::move(p)) {
        //
    }

    std::string name() const override {
        return "$" + _param.name();
    }

    T eval(const Eval& eval) const override {
        return eval.lookup<T>(_param);
    }

    nlohmann::json to_json() const override {
        return _param.name();
    }

  private:
    TunableParam _param;
};

template<typename T>
struct ScalarExpr: BaseExpr<T> {
    ScalarExpr(T value) : _value(std::move(value)) {
        //
    }

    std::string name() const override {
        std::stringstream oss;
        oss << _value;
        return oss.str();
    }

    T eval(const Eval&) const override {
        return _value;
    }

    nlohmann::json to_json() const override {
        return TunableValue(_value).to_json();
    }

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

    std::string name() const override {
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

template<typename Op, typename... Args>
using op_return_type = decltype(
    std::declval<Op>().eval(std::declval<typename Args::return_type>()...));

template<typename Op, typename... Args>
struct OpExpr: BaseExpr<op_return_type<Op, Args...>> {
    using return_type = op_return_type<Op, Args...>;

    OpExpr(Op op, Args... args) :
        _op(std::move(op)),
        _args(std::move(args)...) {
        //
    }

    OpExpr(Args... args) : _op({}), _args(std::move(args)...) {
        //
    }

    return_type eval(const Eval& eval) const override {
        return _eval_helper(eval, std::index_sequence_for<Args...>());
    }

    std::string name() const override {
        return _name_helper(std::index_sequence_for<Args...>());
    }

    //    nlohmann::json to_json() const override {
    //        return _json_helper(std::index_sequence_for<Args...>());
    //    }

  private:
    template<size_t... I>
    return_type
    _eval_helper(const Eval& eval, std::index_sequence<I...>) const {
        return _op.eval(eval(std::get<I>(_args))...);
    }

    template<size_t... I>
    std::string _name_helper(std::index_sequence<I...>) const {
        return _op.name(std::get<I>(_args).name()...);
    }

    template<size_t... I>
    nlohmann::json to_json(std::index_sequence<I...>) const {
        //return _op.to_json(std::get<I>(_args).to_json()...);
        return nullptr;
    }

  private:
    Op _op;
    std::tuple<Args...> _args;
};

template<typename Op, typename... Args>
OpExpr<Op, Args...> make_op(Op op, Args... args) {
    return OpExpr<Op, Args...>(std::move(op), std::move(args)...);
}

#define UNARY_OP_IMPL(class_name, op)                                          \
    struct class_name##Op {                                                    \
        template<typename T>                                                   \
        auto eval(T val) const {                                               \
            return op val;                                                     \
        }                                                                      \
                                                                               \
        std::string name(const std::string& name) const {                      \
            std::stringstream oss;                                             \
            oss << "(" << #op << name << ")";                                  \
            return oss.str();                                                  \
        }                                                                      \
    };                                                                         \
    template<typename T, typename = typename std::enable_if<is_expr<T>>::type> \
    auto operator op(T value) {                                                \
        return make_op(class_name##Op {}, expr(std::move(value)));             \
    }

UNARY_OP_IMPL(Neg, -)
UNARY_OP_IMPL(Not, !)
UNARY_OP_IMPL(Inv, ~)
#undef UNARY_OP_IMPL

#define BINARY_OP_IMPL(class_name, op)                                       \
    struct class_name {                                                      \
        template<typename L, typename R>                                     \
        auto eval(L left, R right) const {                                   \
            return left op right;                                            \
        }                                                                    \
                                                                             \
        std::string name(const std::string& l, const std::string& r) const { \
            std::stringstream oss;                                           \
            oss << "(" << l << #op << r << ")";                              \
            return oss.str();                                                \
        }                                                                    \
    };                                                                       \
    template<                                                                \
        typename L,                                                          \
        typename R,                                                          \
        typename = typename std::enable_if<is_expr<L> || is_expr<R>>::type>  \
    auto operator op(L left, R right) {                                      \
        return make_op(                                                      \
            class_name {},                                                   \
            expr(std::move(left)),                                           \
            expr(std::move(right)));                                         \
    }

BINARY_OP_IMPL(MulOp, *)
BINARY_OP_IMPL(AddOp, +)
BINARY_OP_IMPL(SubdOp, -)
BINARY_OP_IMPL(DivOp, /)
BINARY_OP_IMPL(RemOp, %)
BINARY_OP_IMPL(OrOp, |)
BINARY_OP_IMPL(AndOp, &)
BINARY_OP_IMPL(XorOp, ^)
BINARY_OP_IMPL(EqOp, ==)
BINARY_OP_IMPL(NeqOp, !=)
BINARY_OP_IMPL(LeOp, <)
BINARY_OP_IMPL(GtOp, >)
BINARY_OP_IMPL(LteOp, <=)
BINARY_OP_IMPL(GteOp, >=)

// This are rarely used and they cause a lot of confusion when expressions
// are used in ostreams/istreams.
// BINARY_OP_IMPL(ShlOp, <<)
// BINARY_OP_IMPL(ShrOp, >>)
#undef BINARY_OP_IMPL

template<typename L>
struct AccessorOp {
    AccessorOp(L list) : _inner(std::move(list)) {
        //
    }

    template<typename I>
    auto eval(I index) const {
        return _inner.at(index);
    }

    std::string name(const std::string& index) const {
        std::stringstream oss;
        oss << "{";
        bool is_first = true;

        for (const auto& it : _inner) {
            if (!is_first) {
                oss << ", ";
            } else {
                is_first = false;
            }

            oss << it;
        }

        oss << "}[" << index << "]";
        return oss.str();
    }

  private:
    L _inner;
};

template<typename T, typename I>
auto get(const std::initializer_list<T>& list, I index) {
    std::vector<T> v = list;
    return get(std::move(v), index);
}

template<typename L, typename I>
auto get(L list, I index) {
    return make_op(AccessorOp<L> {std::move(list)}, expr(std::move(index)));
}

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

template<typename T>
struct CastOp {
    template<typename I>
    T eval(I value) const {
        return detail::CastHelper<I, T>::call(std::move(value));
    }

    std::string name(const std::string& name) const {
        return name;
    }
};

template<typename T, typename E>
auto cast(E val) {
    return make_op(CastOp<T> {}, expr(std::move(val)));
}

struct TernaryOp {
    template<typename L, typename R>
    auto eval(bool cond, L left, R right) const {
        return cond ? std::move(left) : std::move(right);
    }

    std::string name(
        const std::string& cond,
        const std::string& left,
        const std::string& right) const {
        std::stringstream oss;
        oss << "(" << cond << " ? " << left << " : " << right << ")";
        return oss.str();
    }
};

template<typename C, typename L, typename R>
auto ifelse(C cond, L left, R right) {
    return make_op(
        TernaryOp {},
        expr(std::move(cond)),
        expr(std::move(left)),
        expr(std::move(right)));
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
        _inner = std::make_shared<decltype(e)>(e);
    }

    T eval(const Eval& eval) const override {
        return _inner->eval(eval);
    }

    std::string name() const override {
        return _inner->name();
    }

  private:
    std::shared_ptr<BaseExpr<T>> _inner {};
};

}  // namespace kernel_launcher
