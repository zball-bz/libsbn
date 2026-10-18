// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/integer.hpp"
namespace sbn {
// Ordinary nonnegative integer exponentiation. 0^0 is 1. This is not modular
// powering. The fixed binary ladder is sized from the input bit-length;
// arithmetic uses only the queried caller workspace. Exact alias is allowed.
requirements power_requirements(integer_ref,integer_view,std::size_t exponent,unsigned workers=1)noexcept;
void power_into(integer_ref,integer_view,std::size_t exponent,execution={})noexcept;
integer pow(const integer&,std::size_t exponent)noexcept;
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline integer pow(const integer&a,I exponent)noexcept{return pow(a,detail::checked_count(exponent));}
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline requirements power_requirements(integer_ref out,integer_view a,I e,unsigned workers=1)noexcept{return power_requirements(out,a,detail::checked_count(e),workers);}
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline void power_into(integer_ref out,integer_view a,I e,execution ex={})noexcept{power_into(out,a,detail::checked_count(e),ex);}
}
