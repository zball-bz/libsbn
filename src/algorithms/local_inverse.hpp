#pragma once
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3 {
class Frame;
// Normalized n-limb D, 1<=n<=8192. Exact floor((B^(2n)-1)/D),
// n+1 limbs with the high limb 1. Disjoint inputs/output/scratch. Local
// u52/FFT products follow the ordinary Newton residual/correction proof.
// Length-based local recipes need no operation-level product search or
// spectrum. The byte query includes all temporary FFT tables and work;
// arithmetic performs no ordinary heap allocation.
size_t local_inverse_bytes(size_t n) noexcept;
void local_inverse(uint64_t *out,const uint64_t *d,size_t n,Frame &) noexcept;
// Same framing, but only |U-B^(2n)/D|<3 is required. Skips the full
// multiply-back that makes the all-ones numerator quotient exact.
// This is the contract consumed by block Barrett and Newton scaling.
size_t local_inverse_approximate_bytes(size_t n) noexcept;
void local_inverse_approximate(uint64_t *out,const uint64_t *d,size_t n,Frame &) noexcept;
// Query-time estimate of the actual local recurrence and selected kernels.
double local_inverse_approximate_cost(size_t n) noexcept;
}
