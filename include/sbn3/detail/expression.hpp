// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/integer.hpp"
namespace sbn {
struct expression_requirements : requirements {
    std::size_t control_bytes=0,value_bytes=0,kernel_bytes=0,instructions=0,parameters=0;
};
namespace detail {
enum class expression_op : unsigned {
    value,parameter,add,subtract,multiply,square,negate,absolute,bit_and,bit_or,bit_xor,complement,
    left_shift,right_shift,trunc_shift,quotient,remainder,exact_quotient,gcd
};
struct expression_instruction {
    expression_op op;
    unsigned left,right;std::size_t argument;
    sbn3_int_view input;
    std::size_t bound_bits;
};
static_assert(std::is_trivially_default_constructible_v<expression_instruction>);
inline constexpr std::size_t max_expression_instructions=1024;
expression_requirements expression_query(std::span<const expression_instruction>,unsigned workers)noexcept;
// Compile and execute use only caller-provisioned bytes. The compiled form
// contains offsets, shapes and resolved kernel plans; input addresses are
// rebound on execute, which makes owning expression moves safe.
expression_requirements expression_compile(std::span<const expression_instruction>,execution,std::span<const integer_view> inputs={})noexcept;
void expression_execute(integer_ref,std::span<const expression_instruction>,execution,std::span<const integer_view> inputs={})noexcept;
inline integer_view argument_view(integer_view v)noexcept{return v;}
template<class V>requires requires(const V&v){{v.view()}->std::same_as<integer_view>;}
inline integer_view argument_view(const V&v)noexcept{return v.view();}
template<class V>concept expression_argument=requires(const V&v){{argument_view(v)}->std::same_as<integer_view>;};
}
}
