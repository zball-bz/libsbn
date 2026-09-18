#include "verify/bbp.hpp"
#include <immintrin.h>
namespace sbn::v3::bbp {
namespace {
using V=__m512i;
using D=__m512d;
inline V vi(uint64_t v) { return _mm512_set1_epi64(v); }
inline D vd(double v) { return _mm512_set1_pd(v); }
inline D square_mod(D x,D d,D inv) {
    // d<2^48, x<d. FMA recovers the exact product residual. Quotient
    // estimation is within one integer; both correction paths are exact.
    D h=_mm512_mul_pd(x,x);
    D l=_mm512_fmsub_pd(x,x,h);
    D q=_mm512_floor_pd(_mm512_mul_pd(h,inv));
    D r=_mm512_add_pd(_mm512_fnmadd_pd(q,d,h),l);
    r=_mm512_mask_add_pd(r,_mm512_cmp_pd_mask(r,vd(0),_CMP_LT_OQ),r,d);
    return _mm512_mask_sub_pd(r,_mm512_cmp_pd_mask(r,d,_CMP_GE_OQ),r,d);
}
template<unsigned N> void sum(Fixed &out,uint64_t offset,Term t,uint64_t begin,uint64_t end) {
    V sums[N];
    for(auto &s:sums) s=vi(0);
    uint64_t k=begin;
    // Four independent batches expose instruction-level parallelism in the
    // modular exponentiation dependency chain (32 terms, eight SIMD lanes).
    for(;end-k>=32;k+=32) {
        const uint64_t max_e=offset+t.shift-10*k;
        const unsigned length=max_e?64-__builtin_clzll(max_e):0;
        const unsigned remaining=length>5?length-5:0;
        V denominator[4], exponent[4];
        D d[4], inv[4], r[4];
        for(unsigned b=0;b<4;++b) {
            const V index=_mm512_add_epi64(vi(k+8*b),_mm512_setr_epi64(0,1,2,3,4,5,6,7));
            denominator[b]=_mm512_add_epi64(_mm512_mullo_epi64(index,vi(t.a)),vi(t.b));
            exponent[b]=_mm512_sub_epi64(vi(offset+t.shift),_mm512_mullo_epi64(index,vi(10)));
            d[b]=_mm512_cvtepu64_pd(denominator[b]);
            inv[b]=_mm512_div_pd(vd(1),d[b]);
            // First five exponent bits produce at most 2^31: construct this
            // exactly instead of performing five modular squarings.
            const V prefix=_mm512_srlv_epi64(exponent[b],vi(remaining));
            const D start=_mm512_cvtepu64_pd(_mm512_sllv_epi64(vi(1),prefix));
            const D q=_mm512_floor_pd(_mm512_mul_pd(start,inv[b]));
            r[b]=_mm512_fnmadd_pd(q,d[b],start);
            r[b]=_mm512_mask_add_pd(r[b],_mm512_cmp_pd_mask(r[b],vd(0),_CMP_LT_OQ),r[b],d[b]);
            r[b]=_mm512_mask_sub_pd(r[b],_mm512_cmp_pd_mask(r[b],d[b],_CMP_GE_OQ),r[b],d[b]);
        }
        for(unsigned bit=remaining;bit--;) {
            for(unsigned b=0;b<4;++b) {
                r[b]=square_mod(r[b],d[b],inv[b]);
                const __mmask8 double_it=_mm512_test_epi64_mask(exponent[b],vi(uint64_t(1)<<bit));
                r[b]=_mm512_mask_add_pd(r[b],double_it,r[b],r[b]);
                r[b]=_mm512_mask_sub_pd(r[b],_mm512_cmp_pd_mask(r[b],d[b],_CMP_GE_OQ),r[b],d[b]);
            }
        }
        for(unsigned b=0;b<4;++b) {
            V rem=_mm512_cvttpd_epu64(r[b]);
            const D scale=_mm512_mul_pd(inv[b],vd(double(uint64_t(1)<<radix_bits)));
            for(unsigned j=N;j--;) {
                V q=_mm512_cvttpd_epi64(_mm512_mul_pd(_mm512_cvtepu64_pd(rem),scale));
                V next=_mm512_sub_epi64(_mm512_slli_epi64(rem,radix_bits),
                                       _mm512_mullo_epi64(q,denominator[b]));
                // True difference is in (-d,2d), so its signed low 64 bits
                // recover it even when rem*2^48 itself exceeds 64 bits.
                const auto below=_mm512_cmp_epi64_mask(next,vi(0),_MM_CMPINT_LT);
                q=_mm512_mask_sub_epi64(q,below,q,vi(1));
                next=_mm512_mask_add_epi64(next,below,next,denominator[b]);
                const auto above=_mm512_cmp_epu64_mask(next,denominator[b],_MM_CMPINT_GE);
                q=_mm512_mask_add_epi64(q,above,q,vi(1));
                rem=_mm512_mask_sub_epi64(next,above,next,denominator[b]);
                const __mmask8 negative=(t.negative != bool(k&1))?0x55:0xaa;
                q=_mm512_mask_sub_epi64(q,negative,vi(0),q);
                sums[j]=_mm512_add_epi64(sums[j],q);
            }
        }
    }
    int64_t carry=0;
    Fixed total{};
    for(unsigned j=0;j<N;++j) {
        carry+=_mm512_reduce_add_epi64(sums[j]);
        total.d[j]=uint64_t(carry)&mask; carry>>=radix_bits;
    }
    out.add(total,N);
    for(;k<end;++k) out.add(scalar_term(offset,t,k,N),N,t.negative != bool(k&1));
}
}
void native_sum(Fixed &out,uint64_t offset,Term t,uint64_t begin,uint64_t end,unsigned digits) {
    if(digits==4) sum<4>(out,offset,t,begin,end);
    else sum<8>(out,offset,t,begin,end);
}
}
