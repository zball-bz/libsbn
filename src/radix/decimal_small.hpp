#pragma once
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3::radix {
// Bound for the shared fixed powers, word reciprocals and SIMD byte constants
// in the direct decimal module, including alignment (not runtime storage).
inline constexpr size_t decimal_constant_bound_bytes=8192;
// Native full-fresh crossover bounds. The algorithms have quadratic work;
// use the shared blocked/scaled/tree paths beyond these small domains.
inline constexpr size_t decimal_write_limbs=11;
inline constexpr size_t decimal_fraction_write_limbs=26,decimal_fraction_read_limbs=10;
// A retained binding has a different fixed cost from the one-shot entry.
inline constexpr size_t decimal_fraction_read_binding_limbs=9;
size_t decimal_write_padded(uint8_t *,size_t width,const uint64_t *,size_t limbs) noexcept;
void decimal_fraction_write(uint8_t *,size_t digits,const uint64_t *,size_t limbs) noexcept;
bool decimal_fraction_read(uint64_t *,size_t limbs,const uint8_t *,size_t digits) noexcept;
}
