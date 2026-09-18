#include "series/terminal.hpp"
#include "series/scaled_ratio.hpp"
#include "runtime/arena.hpp"
#include "runtime/team.hpp"
#include "common/checked.hpp"
#include <algorithm>
#include <cstring>
namespace sbn::v3::series {
namespace {
size_t up(size_t n, size_t a) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "terminal layout alignment");
    return r;
}
} // namespace
sbn3_query_result ratio_terminal_query(size_t fractional, size_t guard_words, unsigned workers,
                                       uint64_t input_error_units, RatioTerminalPlan &p,
                                       bool consume_inputs) noexcept {
    p = {};
    p.consume_inputs = consume_inputs;
    if (!fractional || !guard_words || guard_words > 64 || !workers || workers > 32 ||
        fractional > (size_t(1) << 28) || input_error_units > (uint64_t(1) << 40))
        return SBN3_UNSUPPORTED;
    p.fractional = fractional;
    p.guard_words = guard_words;
    p.working = std::max(size_t(4), fractional + guard_words);
    p.dropped_words = p.working - fractional;
    p.output_limbs = fractional + 1;
    p.workers = workers;
    const sbn3_newton_options no{workers, 0, 0, 0};
    const auto rc = sbn3_newton_query(SBN3_NEWTON_DIVIDE, p.working + 2, &no, &p.divide, &p.info);
    if (rc != SBN3_SUPPORTED)
        return rc;
    // ratio_inputs: <2 units from both truncations; Newton: <3 units; the
    // dropped low word of the quotient adds <1; caller-certified input error.
    p.error_units = 6 + input_error_units;
    p.storage_alignment = std::max<size_t>(128, p.info.storage_alignment);
    // Values: A (working+3 limbs), D (working+2 limbs), Q (working+3 limbs).
    const size_t n = p.working;
    p.values_bytes = consume_inputs ? 0 : up(8 * (n + 3), 128) + up(8 * (n + 2), 128) + up(8 * (n + 3), 128);
    p.division_offset = up(p.values_bytes, p.storage_alignment);
    p.storage_bytes = p.division_offset + p.info.storage_bytes;
    return SBN3_SUPPORTED;
}
void ratio_terminal_execute(const RatioTerminalPlan &p, const sbn3_series_value &numerator,
                            const sbn3_series_value &denominator, sbn3_arena *arena, size_t offset,
                            sbn3_team *team, sbn3_limbs out, RatioOutput mode) noexcept {
    require(arena && team && p.working && !p.consume_inputs && out.data && !(uintptr_t(out.data) & 63) &&
                out.capacity >= p.output_limbs && team->width == p.workers && !team->busy,
            SBN3_FATAL_ARGUMENT, "ratio terminal arguments");
    require(arena->contains(offset, offset + p.storage_bytes) && arena->unleased(offset, p.storage_bytes) &&
                !(uintptr_t(arena->base + offset) & (p.storage_alignment - 1)) &&
                !overlaps(out.data, out.capacity * 8, arena->base + offset, p.storage_bytes),
            SBN3_FATAL_WORKSPACE, "ratio terminal region");
    const auto &dm = denominator.mantissa;
    require(dm.size && !dm.negative && !numerator.mantissa.negative, SBN3_FATAL_MATH, "ratio terminal signs");
    const size_t n = p.working;
    auto values = arena->acquire(offset, p.values_bytes);
    auto *A = reinterpret_cast<uint64_t *>(arena->base + offset);
    auto *D = reinterpret_cast<uint64_t *>(arena->base + offset + up(8 * (n + 3), 128));
    auto *Q = reinterpret_cast<uint64_t *>(arena->base + offset + up(8 * (n + 3), 128) + up(8 * (n + 2), 128));
    ratio_inputs({{numerator.mantissa.data, numerator.mantissa.size, 0}, numerator.exponent2},
                 {{dm.data, dm.size, 0}, denominator.exponent2}, n, A, D, team);
    sbn3_newton_binding *division = nullptr;
    sbn3_newton_bind(&p.divide, arena, offset + p.division_offset, team, &division);
    const sbn3_newton_inputs in{{A, n + 3}, {D, n + 2}, 0};
    sbn3_newton_execute(division, &in, {Q, n + 3});
    sbn3_newton_unbind(division);
    const auto quotient = ratio_result(Q, n); // n+1 limbs: integer limb on top
    bool retained_zero = true;
    for (size_t j = p.dropped_words; j < n + 1; ++j)
        retained_zero &= quotient.data[j] == 0;
    // Nonnegativity permits touching zero, but a zero approximation must
    // still stay below the first positive output cell including its error.
    if (mode == RatioOutput::Certified)
        guard_separated(quotient.data, p.dropped_words, p.error_units, retained_zero);
    std::memcpy(out.data, quotient.data + p.dropped_words, p.output_limbs * 8);
    arena->release(values);
}
sbn3_const_limbs ratio_terminal_execute_consuming(const RatioTerminalPlan &p, sbn3_series_value &numerator,
                                                  sbn3_series_value &denominator, sbn3_arena *arena,
                                                  size_t offset, sbn3_team *team, RatioOutput mode) noexcept {
    auto &am = numerator.mantissa;
    auto &dm = denominator.mantissa;
    const size_t n = p.working;
    require(arena && team && n && p.consume_inputs && team->width == p.workers && !team->busy && am.data && dm.data &&
                !(uintptr_t(am.data) & 63) && !(uintptr_t(dm.data) & 63) &&
                am.capacity >= ratio_terminal_numerator_limbs(p) && dm.capacity >= ratio_terminal_denominator_limbs(p) &&
                am.size <= am.capacity && dm.size <= dm.capacity,
            SBN3_FATAL_ARGUMENT, "consuming ratio terminal arguments");
    require(arena->contains(offset, offset + p.storage_bytes) && arena->unleased(offset, p.storage_bytes) &&
                !(uintptr_t(arena->base + offset) & (p.storage_alignment - 1)) &&
                !overlaps(am.data, am.capacity * 8, arena->base + offset, p.storage_bytes) &&
                !overlaps(dm.data, dm.capacity * 8, arena->base + offset, p.storage_bytes) &&
                !overlaps(am.data, am.capacity * 8, dm.data, dm.capacity * 8),
            SBN3_FATAL_WORKSPACE, "consuming ratio terminal region");
    require(dm.size && !dm.negative && !am.negative, SBN3_FATAL_MATH, "ratio terminal signs");
    // In-place normalization (the exact alias is part of scaled_slice's contract),
    // then DIVIDE with output == numerator, which its contract permits.
    auto *A = am.data, *D = dm.data;
    ratio_inputs({{A, am.size, 0}, numerator.exponent2}, {{D, dm.size, 0}, denominator.exponent2}, n, A, D, team);
    sbn3_newton_binding *division = nullptr;
    sbn3_newton_bind(&p.divide, arena, offset + p.division_offset, team, &division);
    const sbn3_newton_inputs in{{A, n + 3}, {D, n + 2}, 0};
    sbn3_newton_execute(division, &in, {A, n + 3});
    sbn3_newton_unbind(division);
    const auto quotient = ratio_result(A, n); // n+1 limbs: integer limb on top
    bool retained_zero = true;
    for (size_t j = p.dropped_words; j < n + 1; ++j)
        retained_zero &= quotient.data[j] == 0;
    if (mode == RatioOutput::Certified)
        guard_separated(quotient.data, p.dropped_words, p.error_units, retained_zero);
    std::memmove(A, quotient.data + p.dropped_words, p.output_limbs * 8);
    am.size = dm.size = 0; // both inputs are consumed
    return {A, p.output_limbs};
}
} // namespace sbn::v3::series
