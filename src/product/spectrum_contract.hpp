#pragma once
#include <stdint.h>

namespace sbn::v3::spectrum_contract {
// Lifecycle is distinct from the mathematical transform frontier. Keep the
// existing uint32 atomic representation; no additional hot-path dispatch.
inline constexpr uint32_t reserved = 0, computing = 1, ready = 2;
inline constexpr unsigned blocked48_format = 1, flat_lazy64_format = 2, fft_aosoa_format = 3;
namespace fft_codec {
inline constexpr unsigned recipe_mask = 3;
inline constexpr unsigned balanced = 8;
inline constexpr unsigned centered = 16;
inline constexpr unsigned allowed = recipe_mask | balanced | centered;
} // namespace fft_codec
} // namespace sbn::v3::spectrum_contract
