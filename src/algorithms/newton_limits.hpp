#pragma once
#include <bit>
#include <stddef.h>

namespace sbn::v3::newton_limits {
// Implemented seed and transform capacities. Keep storage sizes stable during
// the structural cleanup; assertions tie them to the actual program bounds.
inline constexpr size_t precision_words = (size_t(1) << 31) - 16;
inline constexpr size_t inverse_seed_words = 15;
inline constexpr size_t rsqrt_seed_words = 3;
inline constexpr unsigned rational_iterations = 40;
inline constexpr unsigned first_ntt_prime_count = 4, last_ntt_prime_count = 10;
inline constexpr unsigned ladder_bound = std::bit_width(precision_words - 2);
// n -> floor(n/2)+1 is exactly (n-2) -> floor((n-2)/2).
inline constexpr unsigned required_products = 2 * (rational_iterations - 1) + ladder_bound + 3;
inline constexpr unsigned required_spectra = rational_iterations - 1 + ladder_bound + 1;
inline constexpr unsigned required_leases = 2 * required_products + required_spectra + 8;
// 3 NP4 digit choices + 6*4 NP5..10 choices, two NTT layouts;
// four FFT radices * two widths, four wide digit widths * four radices, scalar.
inline constexpr unsigned candidate_bound =
    (3 + (last_ntt_prime_count - first_ntt_prime_count) * 4) * 2 + 4 * 2 + 4 * 4 + 1;
inline constexpr unsigned stages = 40, products = 128, spectra = 96, leases = 384, candidates = 80;
static_assert(stages >= ladder_bound && stages >= rational_iterations);
static_assert(products >= required_products && spectra >= required_spectra);
static_assert(leases >= required_leases && candidates >= candidate_bound);
} // namespace sbn::v3::newton_limits
