// SPDX-License-Identifier: LGPL-3.0-or-later
#include "series/precision_policy.hpp"
#include "common/identity.hpp"
#include <algorithm>
namespace sbn::v3::series {
size_t precision_words(const PsrSpec &s, uint64_t at) noexcept {
    const uint64_t lost = s.precision.decay_bits(s.precision.context, at) / 64;
    return lost >= s.fractional_limbs - 2 ? 2 : s.fractional_limbs - size_t(lost);
}
static size_t limited_words(const void *p, uint64_t a) {
    return precision_words(*static_cast<const PsrSpec *>(p), a) + precision_guard_words;
}
// An exact block runs the same canonical whole-word values (known low zero
// words live in the exponent) under one constant limit: the words its first
// index keeps. The exact-suffix rule proved that the span bound of the whole
// block fits below it, and the span bound of every subrange is no larger (D
// divides, the term count and the |P/Q| window only shrink), so no node ever
// drops a nonzero bit; the root needs no more capacity than a limited one.
static size_t exact_words(const void *p, uint64_t) {
    return *static_cast<const size_t *>(p);
}
FiniteFormula precision_formula(const PsrSpec &s, const size_t *exact) noexcept {
    auto f = s.formula;
    f.limit_context = exact ? static_cast<const void *>(exact) : &s;
    f.limit_words = exact ? exact_words : limited_words;
    f.reuse_values = true;
    if(f.normalized_bounds)f.bounds=f.normalized_bounds;
    if(f.normalized_serial_envelope)f.serial_envelope=f.normalized_serial_envelope;
    f.parameter_id = exact ? identity::word(identity::word(f.parameter_id, 0x5053524558414354ULL), *exact)
                           : identity::word(identity::word(f.parameter_id, 0x5053525041495231ULL), s.fractional_limbs);
    return f;
}
struct Cut {
    uint64_t end = 0;
    bool exact = false;
};
// Exact-suffix rule (docs/series-precision-planner-math-2026-09-17.md, (22) and
// section 7): the words a node starting at a must keep are
// precision(a)+guard. When the exact values of the whole remaining range
// [a,end) fit them with a word to spare, truncation cannot save anything
// there: for x in [a,end) the demand falls by the attenuation of [a,x), which
// never exceeds the growth log2 D[a,x) removed from the suffix (|U| >= 1), so
// no node inside would be truncated either. The suffix is then planned as
// one exact tree. Hyperdescent series stopped by their tail bound satisfy
// this from the first index (log2 D[begin,N) = p + O(log N)); expanding
// Common/BinaryBBP chains do not, and keep limited blocks. This is a
// planning result of the same precision rule, not a formula-family switch.
static bool exact_suffix(const PsrSpec &s, uint64_t a) {
    if (!s.exact_suffix)
        return false;
    const uint64_t end = s.range.end;
    if (s.precision.maximum_block_terms && end - a > s.precision.maximum_block_terms)
        return false;
    sbn3_series_shape exact{};
    if (s.formula.exact_span) {
        if (s.formula.exact_span(s.formula.context, {a, end}, 3, &exact) != SBN3_SUPPORTED)
            return false;
    } else {
        const auto bounds = s.formula.normalized_bounds ? s.formula.normalized_bounds : s.formula.bounds;
        if (bounds(s.formula.context, {a, end}, end - a, 3, &exact) != SBN3_SUPPORTED)
            return false;
    }
    return std::max(exact.limbs[0], exact.limbs[1]) + 1 <= precision_words(s, a) + precision_guard_words;
}
static Cut cut(const PsrSpec &s, uint64_t a) {
    const auto &p = s.precision;
    const uint64_t end = s.range.end;
    if (exact_suffix(s, a))
        return {end, true};
    uint64_t low = a + 1, high = end;
    while (low < high) {
        const uint64_t m = low + (high - low) / 2;
        if (s.formula.work(s.formula.context, {a, m}) >=
            std::max(double(precision_words(s, m)) * 64, p.minimum_block_bits))
            high = m;
        else
            low = m + 1;
    }
    low = std::max(low, a + std::min(p.minimum_block_terms, end - a));
    if (p.maximum_block_terms)
        low = std::min(low, a + std::min(p.maximum_block_terms, end - a));
    return {low, false};
}
bool precision_spec_supported(const PsrSpec &s) noexcept {
    return s.formula.bounds && s.formula.work && s.formula.leaf && s.precision.decay_bits &&
           s.range.begin < s.range.end && s.range.end - s.range.begin <= (uint64_t(1) << 48) &&
           s.fractional_limbs >= 2 && s.fractional_limbs <= (size_t(1) << 28) &&
           s.precision.minimum_block_terms &&
           (!s.precision.maximum_block_terms ||
            s.precision.maximum_block_terms >= s.precision.minimum_block_terms) &&
           s.formula.recipe >= SBN3_SERIES_HYPERDESCENT && s.formula.recipe <= SBN3_SERIES_BINARY_BBP &&
           s.precision.decay_bits(s.precision.context, s.range.begin) == 0;
}
sbn3_query_result precision_block_query(const PsrSpec &s, uint64_t begin, PrecisionBlock &out) noexcept {
    if (!precision_spec_supported(s) || begin < s.range.begin || begin >= s.range.end)
        return SBN3_UNSUPPORTED;
    const Cut selected = cut(s, begin);
    if (selected.end <= begin || selected.end > s.range.end ||
        s.precision.decay_bits(s.precision.context, selected.end) < s.precision.decay_bits(s.precision.context, begin))
        return SBN3_UNSUPPORTED;
    const size_t p = precision_words(s, begin);
    const bool more = selected.end < s.range.end;
    out = {{begin, selected.end}, p, more ? precision_words(s, selected.end) : 0,
           selected.exact ? p + precision_guard_words : 0,
           more && s.formula.recipe == SBN3_SERIES_COMMON_P2B3 ? 7u : 3u};
    return SBN3_SUPPORTED;
}
} // namespace sbn::v3::series
