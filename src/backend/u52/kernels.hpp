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
// The same exact polynomial band with four additional low diagonals.
// Let na=ceil(64*an/52), nb=ceil(64*bn/52). Output begins at bit
// 52*(na-5) of the full product and has ceil(52*(nb-na+7)/64) words.
// Omitted lower coefficients contribute a nonnegative carry <2*na*2^52.
// an<=832 (na<=1024), bn>=an, bn<=8192, na>=5;
// output/input/scratch disjoint. This retains the donor's audited n<=1024.
size_t middle_guard_scratch_bytes(size_t an,size_t bn) noexcept;
void middle_guard(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,Frame &) noexcept;
// Same guarded band as a logical (bn+2)-word B ending in two zero words,
// while reading exactly bn source words. Scratch is queried at bn+2.
void middle_guard_zero2(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,Frame &) noexcept;
// Guarded high-prefix estimate: floor(A*B/2^(64*an)), or one less.
// Only the highest bn+2 words of A are read. an>=bn+2, 1<=bn<=510;
// writes bn words; all spans disjoint. Used for a short quotient tail.
size_t high_prefix_scratch_bytes(size_t bn) noexcept;
void high_prefix(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,Frame &) noexcept;
Prepared prepare_buffers(Frame &,size_t an,size_t bn,bool allow_streaming=true) noexcept;
void multiply_prepared(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,
                       const Prepared &,Frame &) noexcept;
// Bounds include conversion buffers, vector padding and recursive temporary
// storage; proved linear recursive envelope, independent of input values.
size_t scratch_bytes(size_t an,size_t bn,Algorithm root=Algorithm::automatic) noexcept;
// Exact D&C division over 416-bit blocks. 3<=dn<=2^20, nn<=2^40;
// d[dn-1]!=0. Q has max(nn-dn+1,0) words, R has dn words. All spans
// are disjoint; every output word is written, including leading zeros.
size_t divide_scratch_bytes(size_t nn,size_t dn) noexcept;
// Persistent denominator representation. It contains the same two-vector
// reciprocal used at every div2b/DC leaf; execute never converts D again.
struct alignas(64) DivisionDivisor {
    const uint64_t *digits = nullptr;
    size_t limbs = 0, bits = 0, blocks = 0, shift = 0;
    alignas(64) uint64_t inverse[24];
};
size_t divide_divisor_bytes(size_t dn) noexcept;
size_t divide_work_bytes(size_t nn,size_t dn) noexcept;
DivisionDivisor divide_prepare(const uint64_t *d,size_t dn,Frame &) noexcept;
void divide_prepared(uint64_t *q,uint64_t *r,const uint64_t *n,size_t nn,
                      const DivisionDivisor &,Frame &) noexcept;
void divide(uint64_t *q,uint64_t *r,const uint64_t *n,size_t nn,const uint64_t *d,size_t dn,Frame &) noexcept;
// floor(B^n*A/D), normalized n-word D and (n+1)-word A with A[n]<=1.
// Writes n+1 words; Q may equal A. Uses divide_scratch_bytes(2*n+1,n).
// Shares the exact division core, but decodes the low zero words implicitly
// and does not convert or emit an unused remainder.
void divide_dyadic_quotient(uint64_t *q,const uint64_t *a,const uint64_t *d,size_t n,Frame &) noexcept;
// Experiment/query helper: counts are actual u52 digits, not u64 limbs.
bool root_supported(Algorithm,size_t a_digits,size_t b_digits) noexcept;
Algorithm multiply(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,
                   Frame &,Algorithm root=Algorithm::automatic) noexcept;
}
}
