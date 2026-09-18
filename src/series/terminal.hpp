#pragma once
#include "sbn3/newton.h"
#include "sbn3/series.h"
// Terminal recipes consume a root's exact or limited numerator/denominator
// state and produce the caller's binary fixed-point view. They only compose
// existing services (scaled ratio normalization, Newton divide); no new
// arithmetic. A recipe is queried once and executed in a caller-prepared
// arena range; there is no allocation on the execute path.
namespace sbn::v3::series {
enum class RatioOutput { Certified, Approximate };
struct RatioTerminalPlan {
    // guard_words is the request; working = max(4, fractional + guard_words)
    // is the actual quotient precision and dropped_words = working - fractional
    // the low words actually discarded (>= guard_words).
    size_t fractional = 0, working = 0, guard_words = 0, dropped_words = 0, output_limbs = 0;
    size_t storage_bytes = 0, storage_alignment = 128, values_bytes = 0, division_offset = 0;
    unsigned workers = 0;
    // Consuming form: the caller's writable numerator and denominator slots are
    // normalized in place and the quotient overwrites the numerator slot, so
    // the terminal range holds the division alone (values_bytes == 0).
    bool consume_inputs = false;
    uint64_t error_units = 0; // absolute error bound at scale 2^-(64*working), before guard separation
    sbn3_newton_plan divide{};
    sbn3_newton_info info{};
};
// floor((T/D) * 2^(64*fractional)) with 0 <= T/D < 2^63, D > 0, plus one
// integer limb: output has fractional+1 limbs. The caller certifies that its
// input ratio is within input_error_units (at scale 2^-(64*working)) of the
// value it wants, for example an infinite-series tail; guard separation then
// proves the floor is exact. Support boundary: a value within error_units of
// a multiple of 2^-(64*fractional) cannot be certified and is a deterministic
// SBN3_FATAL_MATH ("constant output guard separation") on the compute face;
// the one exception is a retained result of zero, which is certified as 0
// because the value is known nonnegative. A negative numerator is rejected.
sbn3_query_result ratio_terminal_query(size_t fractional, size_t guard_words, unsigned workers,
                                       uint64_t input_error_units, RatioTerminalPlan &,
                                       bool consume_inputs = false) noexcept;
// Writable-slot capacities of the consuming form, in limbs.
inline size_t ratio_terminal_numerator_limbs(const RatioTerminalPlan &p) noexcept { return p.working + 3; }
inline size_t ratio_terminal_denominator_limbs(const RatioTerminalPlan &p) noexcept { return p.working + 2; }
// Arena range [offset, offset+storage_bytes) is unleased and aligned; the
// output span must be disjoint from it and 64-byte aligned. Inputs are
// caller-owned mantissas with binary exponents (series values).
void ratio_terminal_execute(const RatioTerminalPlan &, const sbn3_series_value &numerator,
                            const sbn3_series_value &denominator, sbn3_arena *, size_t offset, sbn3_team *,
                            sbn3_limbs out, RatioOutput = RatioOutput::Certified) noexcept;
// Consuming form (plan queried with consume_inputs). Both mantissas start at
// the base of caller-owned, 64-byte aligned, mutually disjoint slots with the
// capacities above, outside the terminal range; both are destroyed. The
// result (fractional+1 limbs) is left at the base of the numerator slot and
// stays valid for as long as the caller keeps that slot. Value lifetime: the
// reduced pair is the division's input, the quotient replaces the numerator,
// and no second copy of any full-precision value exists at any time.
sbn3_const_limbs ratio_terminal_execute_consuming(const RatioTerminalPlan &, sbn3_series_value &numerator,
                                                  sbn3_series_value &denominator, sbn3_arena *, size_t offset,
                                                  sbn3_team *, RatioOutput = RatioOutput::Certified) noexcept;
} // namespace sbn::v3::series
