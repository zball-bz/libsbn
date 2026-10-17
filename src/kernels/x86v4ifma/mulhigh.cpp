#include "backend/u52/island.hpp"
#include "backend/u52/mulhigh.hpp"
#include "core/x86_64/gen/mulhigh.hpp"
#include "core/x86_64/word.hpp"
namespace sbn::v3::u52 {
namespace {
// Out-of-line scalar kernels from 5 (register columns to 10, rows beyond) up to these sizes
// (write + call + readback crossovers against the u52 kernels, both producer models); the inline
// L0 kernels below 5, u52 beyond.
constexpr size_t fixed_min=5,mul_rows_max=14,sqr_rows_max=20;
using mul_fn=void (*)(uint64_t *,const uint64_t *,const uint64_t *);
using sqr_fn=void (*)(uint64_t *,const uint64_t *);
constexpr mul_fn mul_rows[mul_rows_max-fixed_min+1]={sbn3i_mulhigh_5,sbn3i_mulhigh_6,sbn3i_mulhigh_7,sbn3i_mulhigh_8,
                                                     sbn3i_mulhigh_9,sbn3i_mulhigh_10,sbn3i_mulhigh_11,
                                                     sbn3i_mulhigh_12,sbn3i_mulhigh_13,sbn3i_mulhigh_14};
constexpr sqr_fn sqr_rows[sqr_rows_max-fixed_min+1]={sbn3i_sqrhigh_5,sbn3i_sqrhigh_6,sbn3i_sqrhigh_7,sbn3i_sqrhigh_8,
                                                     sbn3i_sqrhigh_9,sbn3i_sqrhigh_10,sbn3i_sqrhigh_11,sbn3i_sqrhigh_12,
                                                     sbn3i_sqrhigh_13,sbn3i_sqrhigh_14,sbn3i_sqrhigh_15,sbn3i_sqrhigh_16,
                                                     sbn3i_sqrhigh_17,sbn3i_sqrhigh_18,sbn3i_sqrhigh_19,sbn3i_sqrhigh_20};
static size_t digits(size_t n){return (64*n+51)/52;}
static size_t vectors(size_t d){return (d+7)/8;}
// Planned digit counts including leading zero digits: the kernels read all d
// digits of each operand, conversion writes whole vectors up to its own count.
static sb_pvec convert(sb_pvec A,const uint64_t *a,size_t n,size_t va) noexcept {
    for(size_t v=vectors(u52_from_u64(A,a,n));v<va;++v)A[v]=sb_zero();
    return A;
}
// digit columns >= K of the d x d product (one conversion for a square), then
// bits 64n.. of their canonical value; scratch: R, then A (and B), 64-byte aligned
static void high(uint64_t *out,const uint64_t *a,const uint64_t *b,size_t n,void *scratch) noexcept {
    const size_t d=digits(n),va=vectors(d),K=u52_high_column(n),kv=K/8;
    auto *R=static_cast<sb_pvec>(scratch);
    const auto A=convert(R+vectors(2*d),a,n,va);
    if(b==a)fixed::sqr_blocks_high(reinterpret_cast<uint64_t *>(R),reinterpret_cast<const uint64_t *>(A),
                                   int64_t(d),int64_t(kv),int(K%8));
    else{
        const auto B=convert(A+va,b,n,va);
        // 17..24 digits (n = 14..19, kv = 2 for each): the window rows
        if(d>=17 && d<=24)fixed::mul_rows_window_high<2>(reinterpret_cast<uint64_t *>(R),reinterpret_cast<const uint64_t *>(A),
                                                       reinterpret_cast<const uint64_t *>(B),int64_t(d),int(K%8));
        else if(d>=56)u52_blocks_high<true>(R,A,B,d,kv,int(K%8));
        else u52_blocks_high<false>(R,A,B,d,kv,int(K%8));
    }
    u52_emit_high(out,R,kv,2*d,n);
}
static size_t high_bytes(size_t n,size_t operands) noexcept {
    const size_t d=digits(n);
    return 64*(operands*vectors(d)+vectors(2*d));
}
static void high(uint64_t *out,const uint64_t *a,const uint64_t *b,size_t n,Frame &f) noexcept {
    FrameMark mark(f);
    high(out,a,b,n,f.alloc<uint8_t>(high_bytes(n,b==a?1:2)));
}
} // namespace
size_t mulhigh_u52_bytes(size_t n,bool square) noexcept { return high_bytes(n,square?1:2); }
void mulhigh_u52(uint64_t *out,const uint64_t *a,const uint64_t *b,size_t n,void *scratch) noexcept {
    high(out,a,b,n,scratch);
}
// the same for assembly callers (src/fp/x86_64/mul.S)
extern "C" void sbn3i_mulhigh_u52(uint64_t *out,const uint64_t *a,const uint64_t *b,size_t n,void *scratch) noexcept {
    high(out,a,b,n,scratch);
}
size_t mulhigh_scratch_bytes(size_t n) noexcept {
    require(n&&n<=mulhigh_limit,SBN3_FATAL_SIZE,"mulhigh size");
    return n<=mul_rows_max?0:high_bytes(n,2)+64;
}
size_t sqrhigh_scratch_bytes(size_t n) noexcept {
    require(n&&n<=mulhigh_limit,SBN3_FATAL_SIZE,"sqrhigh size");
    return n<=sqr_rows_max?0:high_bytes(n,1)+64;
}
void sqrhigh(uint64_t *out,const uint64_t *a,size_t n,Frame &f) noexcept {
    require(n&&n<=mulhigh_limit,SBN3_FATAL_ARGUMENT,"sqrhigh size");
    switch(n){
    case 1:return l0::sqrhigh<1>(out,a);
    case 2:return l0::sqrhigh<2>(out,a);
    case 3:return l0::sqrhigh<3>(out,a);
    case 4:return l0::sqrhigh<4>(out,a);
    default:break;
    }
    if(n<=sqr_rows_max)return sqr_rows[n-fixed_min](out,a);
    high(out,a,a,n,f);
}
void mulhigh(uint64_t *out,const uint64_t *a,const uint64_t *b,size_t n,Frame &f) noexcept {
    require(n&&n<=mulhigh_limit,SBN3_FATAL_ARGUMENT,"mulhigh size");
    switch(n){
    case 1:return l0::mulhigh<1>(out,a,b);
    case 2:return l0::mulhigh<2>(out,a,b);
    case 3:return l0::mulhigh<3>(out,a,b);
    case 4:return l0::mulhigh<4>(out,a,b);
    default:break;
    }
    // a == b takes the square kernel of the same family (equal words)
    if(n<=mul_rows_max)return b==a?sqr_rows[n-fixed_min](out,a):mul_rows[n-fixed_min](out,a,b);
    high(out,a,b,n,f);
}
} // namespace sbn::v3::u52
