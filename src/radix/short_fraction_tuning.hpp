#pragma once
#include "radix/geometry.hpp"
#include "radix/word_base.hpp"
namespace sbn::v3::radix {
// Zen5 / Clang 21.1.8, complete fresh direct/tree calls, bases 3/10/63.
// Even g16 points at 96..512 limbs fit the costs; the remaining points are
// held out. Tree cost follows rounded fragments and the distinct nonleaf
// classes queried/bound once, rather than a smooth input-limb surrogate.
// The margin is the P90 tree/direct relative prediction error on training
// points; it is not an acceptance tolerance. Evidence and fit:
// experiments/results/radix_rework_2026-09-20/r11-current-crossover/policy.json.
// No value-dependent measurements or retained plans enter this decision.
inline unsigned short_fraction_classes(uint64_t fragments) noexcept {
    static_assert(group_fragments==8 && fragment_digits==64);
    if(fragments<=group_fragments)return 0;
    // All perfect classes 16,32,... occur. Each nonperfect remainder above
    // one leaf group contributes one further class on the ragged right edge.
    unsigned count=63-unsigned(__builtin_clzll(fragments))-3;
    while(fragments>group_fragments && (fragments&(fragments-1))){
        ++count;
        fragments-=uint64_t(1)<<(63-unsigned(__builtin_clzll(fragments)));
    }
    return count;
}
inline bool short_fraction_preferred(const BaseInfo &base,size_t limbs,uint64_t digits,bool enclosed) noexcept {
    const auto wb=word_bases[base.base];
    const double rounds=double((digits+(enclosed?word_digits:0)+wb.digits-1)/wb.digits);
    const double rectangular=double(limbs)*rounds;
    // Multiplication by an even radix progressively makes low limbs zero.
    // Fraction bits/word rounding change this estimate by at most one limb
    // per round; the execution skips only limbs it actually observes zero.
    const double zeros=double(base.twos*wb.digits)*rounds*(rounds-1)/128.;
    const double steps=rectangular>zeros?rectangular-zeros:0.;
    // Copy cost follows input precision; emission cost follows the number
    // of radix words. Keep them separate for short/overprecise requests.
    const double direct=590.8548001058812+.3*double(limbs)+2.914049631961575*rounds+.2611952941172749*steps;
    const uint64_t fragments=(digits+fragment_digits-1)/fragment_digits;
    const double tree_digits=double(fragments*fragment_digits);
    const double tree_limbs=tree_digits*double(base.log2_base)/64.;
    const double odd_share=double(base.log2_odd/base.log2_base);
    const double tree=6.565754821255059*tree_limbs+.19933465332567643*tree_digits+
                      37.10954684739962*tree_limbs*odd_share+568.9565748379164*short_fraction_classes(fragments);
    return direct<=1.127203338239927*tree;
}
}
