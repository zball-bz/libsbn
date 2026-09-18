#pragma once
#include <stdint.h>
#include <stddef.h>
namespace sbn::v3 {
// v2's reciprocal seed, normalized D, 1<=n<=15. Writes n+1 limbs:
// B^n <= U < 2B^n and |U-B^(2n)/D| < 3. No scratch/heap allocation.
void inverse_seed(uint64_t *U, const uint64_t *D, size_t n) noexcept;
} // namespace sbn::v3
