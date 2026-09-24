#pragma once
#include "algorithms/newton_contract.hpp"
#include "product/window.hpp"
#include "value/parallel_limbs.hpp"

namespace sbn::v3 {
// The arithmetic recipe is independent of product geometry and representation.
// Providers lower its windows/cancellation bounds, and control the lifetime of
// the returned windows. m/n are mathematical precision, never padded lengths.
struct Refinement {
    size_t m, n;
    uint64_t *coarse = nullptr; // m+1 words for Quotient; unused by Inverse
    uint64_t *rho = nullptr;   // n-m+3 words, or unused if provider retains it
    sbn3_team *team = nullptr;
};
enum class RefinementKind { Inverse, Quotient };

template<RefinementKind kind>
constexpr product::WindowGroupShape refinement_products(size_t m,size_t n) noexcept {
    constexpr bool quotient=kind==RefinementKind::Quotient;
    const size_t rn=newton_contract::residual_words(m,n);
    product::WindowGroupShape s{};s.count=quotient?3:2;
    if constexpr(quotient)
        s.products[0]={{m+1,0},{m+1,1},{m,m+1,0},{2*m,4,false},{},false};
    s.products[quotient?1:0]={{m+1,quotient?2u:0u},{n,3},
        {newton_contract::residual_shift(m),rn,64},
        {n,quotient?newton_contract::division_residual_limit:newton_contract::inverse_residual_limit},
        quotient?product::AddendShape{n+1,m,false}:product::AddendShape{1,n+m,true},true};
    s.products[s.count-1]={{m+1,0},{rn,4},{newton_contract::correction_offset(m),n-m+1,0},
        {n+2,quotient?newton_contract::division_correction_limit:newton_contract::inverse_correction_limit},{},false};
    return s;
}

template<RefinementKind kind, class Products>
[[gnu::always_inline]] inline void bounded_refinement(const Refinement &p, Products &products, const uint64_t *a,
                        const uint64_t *d, const uint64_t *u, uint64_t *out) noexcept {
    constexpr bool quotient = kind == RefinementKind::Quotient;
    using namespace product;
    const size_t m = p.m, n = p.n;
    require(m >= 3 && m < n && n - m <= m - 1 && d && u && out &&
                (d[n-1] >> 63) && u[m] == 1,
            SBN3_FATAL_ARGUMENT, "bounded refinement input");
    const uint64_t *coarse = u;
    if constexpr (quotient) {
        require(a && a[n] <= 1 && p.coarse, SBN3_FATAL_ARGUMENT, "quotient refinement input");
        auto first = products.multiply(0, {u, m+1}, {a+n-m, m+1}, {m, m+1, 0}, {2*m, 4, false});
        require(!first.negative && first.words == m+1 && !first.error_bits,
                SBN3_FATAL_MATH, "coarse quotient window");
        coarse = products.retain(first, p.coarse).data;
    }
    constexpr uint64_t residual_limit = quotient ? newton_contract::division_residual_limit
                                                : newton_contract::inverse_residual_limit;
    constexpr uint64_t correction_limit = quotient ? newton_contract::division_correction_limit
                                                  : newton_contract::inverse_correction_limit;
    const uint64_t one = 1;
    const ShiftedSpan subtrahend = quotient ? ShiftedSpan{{a, n+1}, m}
                                           : ShiftedSpan{{&one, 1}, n+m};
    const size_t rn = newton_contract::residual_words(m, n);
    auto residual = products.cancel(quotient ? 1 : 0, {coarse, m+1}, {d, n}, subtrahend,
                                    {newton_contract::residual_shift(m), rn, 64},
                                    {n, residual_limit});
    // Up to 64 low error bits in rho contribute less than 2/B to the final
    // correction because it retains two guard words. The ordinary recurrence
    // contributes <440/B, leaving the same strict <3-ulp output contract.
    require(residual.words == rn && residual.error_bits <= 64,
            SBN3_FATAL_MATH, "refinement residual window");
    residual = products.retain(residual, p.rho);
    const size_t count = n-m+1;
    auto correction = products.multiply(quotient ? 2 : 1, {u, m+1}, {residual.data, rn},
                                        {newton_contract::correction_offset(m), count, 0},
                                        {n+2, correction_limit});
    require(!correction.negative && !correction.error_bits && correction.words == count,
            SBN3_FATAL_MATH, "refinement correction window");
    // Both numerator and divisor have had their last read. For an in-place
    // inverse, move the old approximation before clearing its former low part.
    if (out == coarse) {
        const size_t shift=n-m,overlap=m+1-shift;
        if(overlap<=3){
            uint64_t tail[3];memcpy(tail,coarse+shift,overlap*8);
            // The remaining source and destination are disjoint. Preserve
            // parallel bandwidth for large rungs instead of serial memmove.
            parallel_limbs::copy(p.team,out+shift,coarse,shift);
            memcpy(out+2*shift,tail,overlap*8);
        }else memmove(out+shift,coarse,(m+1)*8);
    }else parallel_limbs::copy(p.team, out+n-m, coarse, m+1);
    parallel_limbs::fill(p.team, out, n-m);
    const uint64_t spill = residual.negative
        ? parallel_limbs::add_to(p.team, out, n+1, correction.data, count)
        : parallel_limbs::sub_from(p.team, out, n+1, correction.data, count);
    require(!spill, SBN3_FATAL_MATH, "bounded refinement correction overflow");
    if constexpr (!quotient) {
        if (!out[n]) memset(out, 0, n*8);
        else if (out[n] > 1) memset(out, 0xff, n*8);
        out[n] = 1;
    }
}
} // namespace sbn::v3
