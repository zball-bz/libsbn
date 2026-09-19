#pragma once
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3 {
class Frame;
// Normalized n-limb D, 1<=n<=8192. Exact floor((B^(2n)-1)/D),
// n+1 limbs with the high limb 1. Disjoint inputs/output/scratch. Local
// u52 products follow the ordinary Newton residual/correction proof;
// no product planner, root table, spectrum or heap allocation is involved.
size_t local_inverse_bytes(size_t n) noexcept;
void local_inverse(uint64_t *out,const uint64_t *d,size_t n,Frame &) noexcept;
}
