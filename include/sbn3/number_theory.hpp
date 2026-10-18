// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/division.hpp"
namespace sbn {
struct bezout_result {integer gcd,x,y;};
struct bezout_requirements : requirements {std::size_t gcd_limbs=0,x_limbs=0,y_limbs=0;};
requirements gcd_requirements(integer_ref,integer_view,integer_view,unsigned workers=1)noexcept;
void gcd_into(integer_ref,integer_view,integer_view,execution={})noexcept;
integer gcd(const integer&,const integer&)noexcept;
integer lcm(const integer&,const integer&)noexcept;
inline bool coprime(const integer&a,const integer&b)noexcept{return gcd(a,b)==1;}
bezout_requirements extended_gcd_requirements(integer_ref g,std::optional<integer_ref>x,std::optional<integer_ref>y,
                                             integer_view,integer_view,unsigned workers=1)noexcept;
void extended_gcd_into(integer_ref g,std::optional<integer_ref>x,std::optional<integer_ref>y,
                       integer_view,integer_view,execution={})noexcept;
bezout_result extended_gcd(const integer&,const integer&)noexcept;
// Modulus may have either sign, but must be nonzero. Noninvertibility is a
// normal empty result. The inverse, when present, is in [0,abs(modulus)).
std::optional<integer> inverse_mod(const integer&,const integer&modulus)noexcept;
requirements jacobi_requirements(integer_view,integer_view,unsigned workers=1)noexcept;
int jacobi(integer_view numerator,integer_view positive_odd_denominator,execution)noexcept;
int jacobi(const integer&,const integer&)noexcept;
}
