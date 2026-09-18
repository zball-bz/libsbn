#pragma once
#include "sbn3/product.h"
#include "product/small_tuning.hpp"
#include "product/short_tuning.hpp"
#include "product/tuning_native.hpp"
#include "tuning/native_policy.hpp"

namespace sbn::v3::cost_model {
enum class Model { SmallNtt, DeepNtt, Fft, Scalar, U52 };
enum class Evidence { Calibrated, Transferred, Extrapolated };
struct Estimate {
    double nanoseconds;
    Model model;
    Evidence evidence;
};

// Work score used to rank deep NTT plans. Unit: active trunk-primes weighted
// by measured geometry factors. Convert to nanoseconds only when mixing models.
inline double deep_score(const sbn3_mul_info &i, bool cyclic = false) {
    const double tower = mul_m2_cost(i.M2);
    const double fill = cyclic ? 1. : double(i.nat + i.nyt - 1) / double(i.transform_trunks);
    const double np_factor = i.np <= 8 ? mul_np_cost[i.np] : 1.;
    return double(i.np) * 8. * double(i.C) * double(i.lbv) * np_factor * tower * mul_column_cost(i.C) *
           mul_codec_cost(i.trunk_bits) * mul_large_rows_cost(i.np, i.M2) * (1. + .12 * (1. - fill));
}

inline bool small_domain(const sbn3_product_spec &s, const sbn3_mul_options &o) {
    const auto hi = std::max(s.a_limbs, s.b_limbs);
    return s.a_limbs && s.b_limbs && hi >= (size_t(1) << 15) && hi <= native_policy::small_model_max_words &&
           !o.column_log2 && !o.row_log2 && o.prime_count <= 8 &&
           (o.workers == 1 || o.workers == 2 || o.workers == 4 || o.workers == 8 || o.workers == 16 ||
            o.workers == 32);
}

// The saturated small-NTT calibration covers this machine's 16 physical
// cores. SMT must still enumerate its small transforms, instead of falling
// into a deep-tower recipe. Transfer the W16 ranking without inventing a
// measured 2x throughput gain; candidate legality/resources keep actual W32.
inline Estimate small_ntt(const sbn3_mul_info &i, bool fresh = true) {
    if (i.workers != 32)
        return {small_mul_cost(i, fresh), Model::SmallNtt, Evidence::Calibrated};
    auto ranking = i;
    ranking.workers = 16;
    return {small_mul_cost(ranking, fresh), Model::SmallNtt, Evidence::Transferred};
}

// A complete cyclic transform has N trunks, irrespective of output prefix.
// Preserve the calibrated evaluation order to keep plan tie decisions stable.
inline Estimate cyclic_product(const sbn3_mul_info &i, bool deep) {
    if (i.algorithm == SBN3_MUL_BAILEY && deep) {
        const double np_factor = i.np <= 8 ? mul_np_cost[i.np] : 1.;
        const double ns = native_policy::deep_ns_per_work * 16 / std::min(i.workers, 16u) * i.np *
                          double(i.transform_trunks) * np_factor * mul_m2_cost(i.M2) * mul_column_cost(i.C) *
                          mul_codec_cost(i.trunk_bits) * mul_large_rows_cost(i.np, i.M2);
        return {ns, Model::DeepNtt, i.np <= 8 ? Evidence::Transferred : Evidence::Extrapolated};
    }
    if (i.np)
        return {small_ntt(i, false).nanoseconds, Model::SmallNtt, Evidence::Transferred};
    return {pq16_product_cost(i, i.nat * i.trunk_bits / 64, i.nyt * i.trunk_bits / 64), Model::Fft,
            Evidence::Transferred};
}

// Seed conversion from an ordinary 2-forward/1-inverse product model. These
// are work-count assumptions, not a claim that measured F and I costs match.
// Explicit recipe counts make it possible to replace this transfer model later.
inline constexpr unsigned ordinary_transforms = 3;
inline constexpr unsigned cached_product_transforms = 2;
inline double prepare_share(double product_ns) {
    return product_ns / ordinary_transforms;
}
inline double inverse_share(double product_ns) {
    return product_ns / ordinary_transforms;
}
inline double cached_share(double product_ns, unsigned applications = 1) {
    return (applications * cached_product_transforms) * product_ns / ordinary_transforms;
}
inline Estimate linear_product(const sbn3_mul_info &i, size_t an, size_t bn) {
    if (i.algorithm == SBN3_MUL_SCALAR) return {scalar_product_cost(an,bn,i.workers),Model::Scalar,Evidence::Calibrated};
    if (i.algorithm == SBN3_MUL_U52) return {u52_product_cost(an,bn,i.workers),Model::U52,Evidence::Calibrated};
    if (!i.np) return {pq16_product_cost(i,an,bn),Model::Fft,Evidence::Calibrated};
    if (std::max(an,bn) <= native_policy::small_model_max_words)
        return small_ntt(i, true);
    return {native_policy::deep_ns_per_work * 16 / std::min(i.workers,16u) * deep_score(i),
            Model::DeepNtt,i.np<=8?Evidence::Transferred:Evidence::Extrapolated};
}
struct RecipeCost { double nanoseconds; size_t peak_bytes; };
// Whole-recipe admission includes value lifetimes, not just kernel scratch.
// Same near-tie/memory policy as Newton; no consumer or length special cases.
inline unsigned choose_recipe(const RecipeCost *c, unsigned count, size_t budget = 0) {
    unsigned best=count;
    for (unsigned j=0;j<count;++j)
        if ((!budget || c[j].peak_bytes<=budget) && c[j].nanoseconds>=0 &&
            (best==count || c[j].nanoseconds<c[best].nanoseconds ||
             (c[j].nanoseconds==c[best].nanoseconds && c[j].peak_bytes<c[best].peak_bytes))) best=j;
    if (best==count) return best;
    unsigned selected=best;
    for (unsigned j=0;j<count;++j)
        if ((!budget || c[j].peak_bytes<=budget) &&
            c[j].nanoseconds<=native_policy::memory_trade_time_ratio*c[best].nanoseconds &&
            c[j].peak_bytes<=native_policy::memory_trade_resident_ratio*c[best].peak_bytes &&
            (selected==best || c[j].nanoseconds<c[selected].nanoseconds)) selected=j;
    return selected;
}
} // namespace sbn::v3::cost_model
