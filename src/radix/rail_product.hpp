#pragma once
// A tree split as a wrap-around product with a cached rail spectrum (the "FFT middle product" and
// "transform-only interface" of the y-cruncher radix conversion notes), on the double-precision FFT
// kernels: C = common * fresh mod (2^(64 ring) - 1), the transform of `common` (a rail power) computed
// once at bind and shared read-only by every worker.
//
// The format tree needs the limbs [window, window + right) of the full product Z. With
//     ring > common + fresh - window          (the wrapped part ends at least one limb below the window)
// the high part of Z is added into the limbs [0, h), h = common + fresh - ring, and can reach the window
// only as a carry that runs through every gap limb [h, window). That needs all gap limbs of Z to be
// ones, and leaves the gap of the residue all zero; the residue's gap is also zero when Z's gap is zero
// and nothing carried. Any nonzero gap limb therefore proves the window exact. Otherwise limb h of Z is
// computed exactly from the low h + 1 limbs of both operands (structured inputs only: zeros, all ones),
// and the carry is taken back out. The window returned is always the exact one.
#include "backend/pq16/kernels.hpp"
#include "runtime/scratch.hpp"
namespace sbn::v3::radix {
inline constexpr size_t rail_product_min_limbs = 400;   // below: the exact word kernels win
inline constexpr size_t rail_product_max_ring = 32768;  // the FFT kernels' cyclic cap (16-bit digits)
struct RailProductPlan {
    bool enabled = false;
    pq16::Shape shape{};
    size_t ring = 0, fresh_limbs = 0, common_limbs = 0;
    size_t table_bytes = 0, cache_bytes = 0, scratch_bytes = 0;
    size_t prepared_bytes() const noexcept { return enabled ? table_bytes + cache_bytes + 512 : 0; }
    size_t output_limbs() const noexcept { return ring ? ring : common_limbs + fresh_limbs; }
    size_t episode_bytes() const noexcept { return ((output_limbs() * 8 + 63) & ~size_t(63)) + 128 + scratch_bytes; }
};
// False when no supported ring of at least `minimum_ring` limbs exists for these operand lengths.
bool rail_product_plan(size_t fresh_limbs, size_t common_limbs, size_t minimum_ring, RailProductPlan &) noexcept;
// The parse tree needs the whole product: a linear product with the same cached spectrum
// (ring == 0, the output has common + fresh limbs). False outside the FFT kernels' single-worker band.
bool rail_linear_plan(size_t fresh_limbs, size_t common_limbs, RailProductPlan &) noexcept;
struct RailProduct {
    RailProductPlan plan{};
    const pq16::Tables *tables = nullptr;
    const double *cache = nullptr;
    const uint64_t *common = nullptr;
};
// Tables and the spectrum of `common` (plan.common_limbs limbs, zero padded, outliving the object) in `storage`.
void rail_product_prepare(RailProduct &, const RailProductPlan &, Frame &storage, const uint64_t *common) noexcept;
// out: plan.output_limbs() limbs (the canonical residue, or the full product). work: at least
// plan.scratch_bytes. One worker.
void rail_product_execute(const RailProduct &, Frame &work, sbn3_team_scope *self, const uint64_t *fresh,
                          uint64_t *out) noexcept;
} // namespace sbn::v3::radix
