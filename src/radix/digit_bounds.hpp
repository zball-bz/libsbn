#pragma once
#include "radix/log2_bounds.hpp"
#include <array>
namespace sbn::v3::radix {
struct InverseLogBound {uint64_t down,up;};
consteval auto make_inverse_logs() {
    std::array<InverseLogBound,65> out{};
    for(unsigned b=3;b<=64;++b){
        const __uint128_t numerator=__uint128_t(1)<<120;
        const auto lo=log2_floor_q56[b];
        out[b]={uint64_t(numerator/(lo+1)),uint64_t((numerator+lo-1)/lo)};
    }
    return out;
}
inline constexpr auto inverse_logs=make_inverse_logs();
// log2(b) lies in [L,L+1]/2^56, so these reciprocals enclose 1/log2(b).
// Used for non-power-of-two bases; shifts handle the powers of two.
inline uint64_t integer_digit_capacity(uint64_t bits,unsigned base) noexcept {
    return uint64_t((__uint128_t(bits)*inverse_logs[base].up)>>64)+1;
}
inline uint64_t fraction_digit_capacity(uint64_t bits,unsigned base) noexcept {
    return uint64_t((__uint128_t(bits)*inverse_logs[base].down)>>64);
}
}
