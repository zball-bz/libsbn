#pragma once
#include <stddef.h>
#include <stdint.h>

namespace sbn::v3::newton_contract {
// Derivations and units: docs/arithmetic-contracts.md. These are arithmetic
// guarantees, not machine-tuning parameters.
inline constexpr size_t guard_words = 2;
inline constexpr uint64_t inverse_input_error = 8;
inline constexpr uint64_t rsqrt_input_error = 4;
inline constexpr uint64_t inverse_residual_limit = 16;
inline constexpr uint64_t inverse_correction_limit = 32;
inline constexpr uint64_t division_residual_limit = 32;
inline constexpr uint64_t division_correction_limit = 64;
inline constexpr uint64_t rsqrt_residual_limit = uint64_t(1) << 36;

constexpr size_t next_precision(size_t n) noexcept {
    return n / 2 + 1;
}
constexpr size_t residual_shift(size_t m) noexcept {
    return m - guard_words;
}
constexpr size_t residual_words(size_t m, size_t n) noexcept {
    return n + 1 - residual_shift(m);
}
constexpr size_t correction_offset(size_t m) noexcept {
    return m + guard_words;
}
constexpr size_t inverse_ring_min(size_t n) noexcept {
    return n + guard_words + 2;
}
constexpr size_t rsqrt_ring_min(size_t m) noexcept {
    return 2 * m + guard_words;
}
constexpr size_t division_ring_min(size_t m) noexcept {
    return 2 * m + guard_words + 2;
}
// Mathematical predecessor support, independently of its placement.
// Reciprocal ladders keep it in caller output; algorithms still using
// separate predecessor buffers can use this bound. Seed-only programs
// need no separate value arrays.
constexpr size_t minimum_approximation_words(size_t target, size_t seed_limit) noexcept {
    return target <= seed_limit ? 0 : next_precision(target) + 1;
}
} // namespace sbn::v3::newton_contract
