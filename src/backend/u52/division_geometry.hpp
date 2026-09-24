#pragma once
#include <stdint.h>
namespace sbn::v3::u52 {
// Shared by the arithmetic kernel and its operation-count cost model.
inline constexpr uint64_t division_block_digits = 8;
inline constexpr uint64_t division_block_bits = 52 * division_block_digits;
inline constexpr uint64_t division_leaf_blocks = 12;
inline constexpr uint64_t division_min_blocks = 3; // the reciprocal uses 18 u52 digits
}
