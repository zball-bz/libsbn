// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/borrow.hpp"
#include "sbn3/division.hpp"
#include "sbn3/detail/expression.hpp"
#include <array>
#include <bit>
namespace sbn::expr {
using op=sbn::detail::expression_op;
class emitter {
    std::span<sbn::detail::expression_instruction>storage_;std::size_t used_=0;
public:
    explicit emitter(std::span<sbn::detail::expression_instruction>s)noexcept:storage_(s){}
    unsigned append(sbn::detail::expression_instruction i)noexcept{
        sbn::detail::require(used_<storage_.size(),SBN3_FATAL_SIZE,"expression instruction bound");storage_[used_]=i;return unsigned(used_++);
    }
    std::size_t size()const noexcept{return used_;}
};
template<class T>concept expression=requires(const std::remove_cvref_t<T>&t,emitter&w){{std::remove_cvref_t<T>::max_nodes}->std::convertible_to<std::size_t>;{t.emit(w)}->std::same_as<unsigned>;};
class borrowed_value {
    borrowed_integer value_;
public:
    static constexpr std::size_t max_nodes=1;
    explicit borrowed_value(const integer&x)noexcept:value_(x){}
    unsigned emit(emitter&w)const noexcept{return w.append({op::value,0,0,0,value_.view().c_view(),0});}
};
class owned_value {
    integer value_;
public:
    static constexpr std::size_t max_nodes=1;
    explicit owned_value(integer x)noexcept:value_(std::move(x)){}
    unsigned emit(emitter&w)const noexcept{return w.append({op::value,0,0,0,value_.view().c_view(),0});}
};
class viewed_value {
    integer_view value_;
public:
    static constexpr std::size_t max_nodes=1;
    explicit viewed_value(integer_view x)noexcept:value_(x){}
    unsigned emit(emitter&w)const noexcept{return w.append({op::value,0,0,0,value_.c_view(),0});}
};
template<unsigned Index>class parameter {
    std::size_t bits_;
public:
    static constexpr std::size_t max_nodes=1;
    explicit parameter(std::size_t bits)noexcept:bits_(bits){}
    unsigned emit(emitter&w)const noexcept{return w.append({op::parameter,0,0,Index,{},bits_});}
};
template<unsigned Index,sbn::detail::builtin_integer I>auto input_bits(I bits)noexcept{return parameter<Index>(sbn::detail::checked_count(bits));}
template<unsigned Index,sbn::detail::builtin_integer I>auto input(I limb_capacity)noexcept{
    const auto n=sbn::detail::checked_count(limb_capacity);return parameter<Index>(n>SIZE_MAX/64?SIZE_MAX:n*64);
}
template<expression E>auto capture(E&&e)noexcept{return std::remove_cvref_t<E>(std::forward<E>(e));}
template<expression E>auto capture(const E&e)noexcept{return E(e);}
inline auto capture(const integer&x)noexcept{return borrowed_value(x);}
inline auto capture(integer&&x)noexcept{return owned_value(std::move(x));}
inline auto capture(const integer&&x)noexcept{return owned_value(integer(x));}
inline auto capture(integer_view x)noexcept{return viewed_value(x);}
template<sbn::detail::builtin_integer I>auto capture(I x)noexcept{return owned_value(integer{x});}
template<class V>requires(!expression<V>&&!std::same_as<V,integer>&&requires(const V&v){{v.view()}->std::same_as<integer_view>;})
auto capture(const V&v)noexcept{return viewed_value(v.view());}
// Raw buffer views cannot safely be captured from a temporary descriptor.
template<class V>requires(!std::is_lvalue_reference_v<V>&&!expression<V>&&!std::same_as<std::remove_cvref_t<V>,integer>&&requires(const V&v){{v.view()}->std::same_as<integer_view>;})
auto capture(V&&)=delete;
template<op Op,expression L,expression R>class binary {
    L left_;R right_;std::size_t argument_;
public:
    static constexpr std::size_t max_nodes=L::max_nodes+R::max_nodes+1;
    binary(L l,R r,std::size_t a=0)noexcept:left_(std::move(l)),right_(std::move(r)),argument_(a){}
    unsigned emit(emitter&w)const noexcept{const auto a=left_.emit(w),b=right_.emit(w);return w.append({Op,a,b,argument_,{},0});}
};
template<op Op,expression E>class unary {
    E value_;std::size_t argument_;
public:
    static constexpr std::size_t max_nodes=E::max_nodes+1;
    unary(E e,std::size_t a=0)noexcept:value_(std::move(e)),argument_(a){}
    unsigned emit(emitter&w)const noexcept{return w.append({Op,value_.emit(w),0,argument_,{},0});}
};
template<op Op,class A,class B>auto make_binary(A&&a,B&&b,std::size_t argument=0)noexcept{
    auto l=capture(std::forward<A>(a));auto r=capture(std::forward<B>(b));
    return binary<Op,decltype(l),decltype(r)>(std::move(l),std::move(r),argument);
}
template<op Op,class A>auto make_unary(A&&a,std::size_t argument=0)noexcept{
    auto v=capture(std::forward<A>(a));return unary<Op,decltype(v)>(std::move(v),argument);
}
template<class A>auto value(A&&a)noexcept{return capture(std::forward<A>(a));}
template<class A,class B>auto add(A&&a,B&&b)noexcept{return make_binary<op::add>(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>auto sub(A&&a,B&&b)noexcept{return make_binary<op::subtract>(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>auto mul(A&&a,B&&b)noexcept{return make_binary<op::multiply>(std::forward<A>(a),std::forward<B>(b));}
template<class A>auto square(A&&a)noexcept{return make_unary<op::square>(std::forward<A>(a));}
template<class A>auto abs(A&&a)noexcept{return make_unary<op::absolute>(std::forward<A>(a));}
template<class A,class B>auto divide(A&&a,B&&b,division_rounding m=division_rounding::trunc)noexcept{return make_binary<op::quotient>(std::forward<A>(a),std::forward<B>(b),std::size_t(m));}
template<class A,class B>auto remainder(A&&a,B&&b,division_rounding m=division_rounding::trunc)noexcept{return make_binary<op::remainder>(std::forward<A>(a),std::forward<B>(b),std::size_t(m));}
template<class A,class B>auto exact_divide(A&&a,B&&b)noexcept{return make_binary<op::exact_quotient>(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>auto gcd(A&&a,B&&b)noexcept{return make_binary<op::gcd>(std::forward<A>(a),std::forward<B>(b));}
template<class A,sbn::detail::builtin_integer I>auto trunc_shift(A&&a,I bits)noexcept{return make_unary<op::trunc_shift>(std::forward<A>(a),sbn::detail::checked_count(bits));}
template<expression E>class power {
    E value_;std::size_t exponent_;
public:
    static constexpr std::size_t max_nodes=E::max_nodes+2*sizeof(std::size_t)*8;
    power(E e,std::size_t n)noexcept:value_(std::move(e)),exponent_(n){}
    unsigned emit(emitter&w)const noexcept{
        const auto base=value_.emit(w);auto acc=base;
        if(!exponent_){static constexpr uint64_t one=1;return w.append({op::value,0,0,0,{&one,1,0},0});}
        for(unsigned bit=unsigned(std::bit_width(exponent_)-1);bit-->0;){acc=w.append({op::square,acc,0,0,{},0});if((exponent_>>bit)&1)acc=w.append({op::multiply,acc,base,0,{},0});}
        return acc;
    }
};
template<class A,sbn::detail::builtin_integer I>auto pow(A&&a,I n)noexcept{auto e=capture(std::forward<A>(a));return power<decltype(e)>(std::move(e),sbn::detail::checked_count(n));}
template<class A,class B>requires(expression<std::remove_cvref_t<A>>||expression<std::remove_cvref_t<B>>)
auto operator+(A&&a,B&&b)noexcept{return add(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>requires(expression<std::remove_cvref_t<A>>||expression<std::remove_cvref_t<B>>)
auto operator-(A&&a,B&&b)noexcept{return sub(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>requires(expression<std::remove_cvref_t<A>>||expression<std::remove_cvref_t<B>>)
auto operator*(A&&a,B&&b)noexcept{return mul(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>requires(expression<std::remove_cvref_t<A>>||expression<std::remove_cvref_t<B>>)
auto operator/(A&&a,B&&b)noexcept{return divide(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>requires(expression<std::remove_cvref_t<A>>||expression<std::remove_cvref_t<B>>)
auto operator%(A&&a,B&&b)noexcept{return remainder(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>requires(expression<std::remove_cvref_t<A>>||expression<std::remove_cvref_t<B>>)
auto operator&(A&&a,B&&b)noexcept{return make_binary<op::bit_and>(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>requires(expression<std::remove_cvref_t<A>>||expression<std::remove_cvref_t<B>>)
auto operator|(A&&a,B&&b)noexcept{return make_binary<op::bit_or>(std::forward<A>(a),std::forward<B>(b));}
template<class A,class B>requires(expression<std::remove_cvref_t<A>>||expression<std::remove_cvref_t<B>>)
auto operator^(A&&a,B&&b)noexcept{return make_binary<op::bit_xor>(std::forward<A>(a),std::forward<B>(b));}
template<expression E>auto operator-(E&&e)noexcept{return make_unary<op::negate>(std::forward<E>(e));}
template<expression E>auto operator~(E&&e)noexcept{return make_unary<op::complement>(std::forward<E>(e));}
template<expression E,sbn::detail::builtin_integer I>auto operator<<(E&&e,I n)noexcept{return make_unary<op::left_shift>(std::forward<E>(e),sbn::detail::checked_count(n));}
template<expression E,sbn::detail::builtin_integer I>auto operator>>(E&&e,I n)noexcept{return make_unary<op::right_shift>(std::forward<E>(e),sbn::detail::checked_count(n));}
template<expression E>struct instruction_list {
    static_assert(E::max_nodes<=sbn::detail::max_expression_instructions,"expression too large; split it at an explicit value boundary");
    std::array<sbn::detail::expression_instruction,E::max_nodes>storage;std::size_t count;
    explicit instruction_list(const E&e)noexcept{emitter w(storage);const auto root=e.emit(w);count=w.size();sbn::detail::require(root+1==count,SBN3_FATAL_MATH,"expression root order");}
    std::span<const sbn::detail::expression_instruction>view()const noexcept{return {storage.data(),count};}
};
}
