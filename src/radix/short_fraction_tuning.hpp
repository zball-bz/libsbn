#pragma once
#include "radix/geometry.hpp"
#include "radix/word_base.hpp"
namespace sbn::v3::radix {
// Zen5 / Clang 21.1.8: full-call direct/tree comparisons, bases 3/10/63.
// Fit on 1/8-octave sizes; additional 1/16 points were used to inspect the
// crossover. A 5% margin covers local fitting uncertainty. The crossover
// was recalibrated after the tree query pruning; earlier table costs must
// not keep the direct path alive beyond its useful range. Evidence:
// experiments/results/radix_rework_2026-09-20/r4-final-crossover/policy.json.
// No value-dependent measurements or retained plans enter this decision.
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
    const double direct=400.+.3*double(limbs)+8.667478*rounds+.24971751*steps;
    const double tree_limbs=double(digits)*double(base.log2_base)/64.;
    const double odd_share=double(base.log2_odd/base.log2_base);
    const double tree=4875.661270+5.576283*tree_limbs+.358832225*double(digits)+67.842975*tree_limbs*odd_share;
    return direct<=.95*tree;
}
}
