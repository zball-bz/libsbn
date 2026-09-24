#pragma once
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3::radix {
inline constexpr unsigned fixed_power_levels=9;
struct FixedPower { const uint64_t *data=nullptr; size_t capacity=0; };
// Published mathematical constants odd^(64*2^level), odd=3,5,...,63.
// Entries are 64-byte aligned and include at least eight zero padding words.
FixedPower fixed_power(unsigned odd,unsigned level) noexcept;
size_t fixed_power_bytes() noexcept;
}
