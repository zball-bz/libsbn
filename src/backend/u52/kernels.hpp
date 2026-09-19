#pragma once
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3 {
class Frame;
namespace u52 {
enum class Algorithm:unsigned {automatic,basecase,karatsuba,toom32,toom33,toom42,stripmine,zero};
struct Prepared {
    void *a=nullptr,*b=nullptr,*product=nullptr;
    size_t a_vectors=0,b_vectors=0;
    uint64_t *strip_product=nullptr;
    size_t strip_limbs=0;
};
// Bound the converted working set for rectangular products. The long input
// stays in u64 and is read one strip at a time; outputs must be disjoint.
inline constexpr size_t strip_limbs=4096, strip_short_limit=1024;
inline bool streams(size_t an,size_t bn) noexcept {
    return (an>strip_limbs && bn && bn<=strip_short_limit) ||
           (bn>strip_limbs && an && an<=strip_short_limit);
}
size_t middle_scratch_bytes(size_t an,size_t bn) noexcept;
void middle(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,Frame &) noexcept;
Prepared prepare_buffers(Frame &,size_t an,size_t bn,bool allow_streaming=true) noexcept;
void multiply_prepared(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,
                       const Prepared &,Frame &) noexcept;
// Bounds include conversion buffers, vector padding and recursive temporary
// storage; proved linear recursive envelope, independent of input values.
size_t scratch_bytes(size_t an,size_t bn,Algorithm root=Algorithm::automatic) noexcept;
// Experiment/query helper: counts are actual u52 digits, not u64 limbs.
bool root_supported(Algorithm,size_t a_digits,size_t b_digits) noexcept;
Algorithm multiply(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,
                   Frame &,Algorithm root=Algorithm::automatic) noexcept;
}
}
