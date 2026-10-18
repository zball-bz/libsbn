// Original libsbn code; LGPL-3.0-or-later.
// Quotient-direction automaton: Moller, arXiv:1907.07795.
// Low bits refer to the complete logical operands, never to an HGCD prefix.
#pragma once
#include <array>
#include <stdint.h>
namespace sbn::v3::gcd {
struct NoSymbol { void operator()(unsigned,uint64_t) const noexcept {} };
struct JacobiState {
    unsigned bits;
    static constexpr unsigned transition(unsigned s,unsigned d,unsigned q) {
        unsigned e=s&1,a=(s>>1)&3,b=(s>>3)&3,old=s>>5;
        if(old!=d&&(a&b&1))e^=((a-1)*(b-1)/4)&1;
        if(d){if(b==2)e^=(q*(a-1)/2+q*(q-1)/2)&1;a=(a-q*b)&3;}
        else {if(a==2)e^=(q*(b-1)/2+q*(q-1)/2)&1;b=(b-q*a)&3;}
        return e|(a<<1)|(b<<3)|(d<<5);
    }
    void operator()(unsigned d,uint64_t q) noexcept;
    static JacobiState initial(uint64_t a,uint64_t b,unsigned sign=0) {return {sign|((unsigned(a)&3)<<1)|((unsigned(b)&3)<<3)|32};}
    bool numerator_is_a() const {return bits&32?bool(bits&8):!(bits&2);}
    int sign() const {return 1-2*int(bits&1);}
};
inline constexpr auto jacobi_transitions=[](){std::array<uint8_t,512> t{};
    for(unsigned s=0;s<64;++s)for(unsigned d=0;d<2;++d)for(unsigned q=0;q<4;++q)
        t[(s<<3)|(d<<2)|q]=JacobiState::transition(s,d,q);
    return t;}();
inline void JacobiState::operator()(unsigned d,uint64_t q) noexcept {bits=jacobi_transitions[(bits<<3)|(d<<2)|(q&3)];}
}
