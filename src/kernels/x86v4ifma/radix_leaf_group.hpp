#pragma once
// Native full leaf group: keep the u52 fractions in registers, emit each
// round with shared constants, then transpose digit words into fragments.
#include "radix/leaf.hpp"
#include <immintrin.h>
namespace sbn::v3::radix::leaf_detail {
[[gnu::always_inline]] inline void transpose(__m512i (&x)[8]) {
    const __m512i a0=_mm512_unpacklo_epi64(x[0],x[1]),a1=_mm512_unpackhi_epi64(x[0],x[1]);
    const __m512i a2=_mm512_unpacklo_epi64(x[2],x[3]),a3=_mm512_unpackhi_epi64(x[2],x[3]);
    const __m512i a4=_mm512_unpacklo_epi64(x[4],x[5]),a5=_mm512_unpackhi_epi64(x[4],x[5]);
    const __m512i a6=_mm512_unpacklo_epi64(x[6],x[7]),a7=_mm512_unpackhi_epi64(x[6],x[7]);
    const __m512i b0=_mm512_shuffle_i64x2(a0,a2,0x44),b1=_mm512_shuffle_i64x2(a1,a3,0x44);
    const __m512i b2=_mm512_shuffle_i64x2(a0,a2,0xee),b3=_mm512_shuffle_i64x2(a1,a3,0xee);
    const __m512i b4=_mm512_shuffle_i64x2(a4,a6,0x44),b5=_mm512_shuffle_i64x2(a5,a7,0x44);
    const __m512i b6=_mm512_shuffle_i64x2(a4,a6,0xee),b7=_mm512_shuffle_i64x2(a5,a7,0xee);
    x[0]=_mm512_shuffle_i64x2(b0,b4,0x88);x[1]=_mm512_shuffle_i64x2(b1,b5,0x88);
    x[2]=_mm512_shuffle_i64x2(b0,b4,0xdd);x[3]=_mm512_shuffle_i64x2(b1,b5,0xdd);
    x[4]=_mm512_shuffle_i64x2(b2,b6,0x88);x[5]=_mm512_shuffle_i64x2(b3,b7,0x88);
    x[6]=_mm512_shuffle_i64x2(b2,b6,0xdd);x[7]=_mm512_shuffle_i64x2(b3,b7,0xdd);
}
struct Emitter {
    __m512i m4,m2,b4,b2,m1,b1,table;
    __m128i s4,s2,s1;
    explicit Emitter(const DigitPlan &p):m4(_mm512_set1_epi64(p.m4)),m2(_mm512_set1_epi64(p.m2)),
      b4(_mm512_set1_epi64(p.b4)),b2(_mm512_set1_epi64(p.b2)),m1(_mm512_set1_epi16(p.m1)),
      b1(_mm512_set1_epi16(p.base)),table(_mm512_loadu_si512(p.encode)),
      s4(_mm_cvtsi32_si128(p.s4)),s2(_mm_cvtsi32_si128(p.s2)),s1(_mm_cvtsi32_si128(p.s1)){}
    [[gnu::always_inline]] inline __m512i operator()(__m512i x)const {
        const __m512i z=_mm512_setzero_si512();
        const auto a=_mm512_srl_epi64(_mm512_madd52hi_epu64(z,x,m4),s4);
        const auto b=_mm512_sub_epi64(x,_mm512_mullo_epi64(a,b4));
        const auto qa=_mm512_srl_epi64(_mm512_madd52hi_epu64(z,a,m2),s2);
        const auto ra=_mm512_sub_epi64(a,_mm512_mullo_epi64(qa,b2));
        const auto qb=_mm512_srl_epi64(_mm512_madd52hi_epu64(z,b,m2),s2);
        const auto rb=_mm512_sub_epi64(b,_mm512_mullo_epi64(qb,b2));
        const auto y=_mm512_or_si512(_mm512_or_si512(qa,_mm512_slli_epi64(ra,16)),
                                    _mm512_or_si512(_mm512_slli_epi64(qb,32),_mm512_slli_epi64(rb,48)));
        const auto q=_mm512_srl_epi16(_mm512_mulhi_epu16(y,m1),s1);
        const auto r=_mm512_sub_epi16(y,_mm512_mullo_epi16(q,b1));
        return _mm512_permutexvar_epi8(_mm512_or_si512(q,_mm512_slli_epi16(r,8)),table);
    }
};
template<unsigned Digits>
void emit_full_group(uint8_t *out,uint64_t *first,uint64_t *overlap,const uint64_t *const input[8],unsigned limbs,const DigitPlan &p){
    const __m512i zero=_mm512_setzero_si512(),mask=_mm512_set1_epi64((uint64_t(1)<<52)-1),b=_mm512_set1_epi64(p.b8);
    constexpr __mmask8 active=0xff;
    __m512i x[Digits],digits[8];const Emitter emit(p);
    #pragma clang loop unroll(full)
    for(unsigned j=0;j<Digits;++j){
        const int bit=int(64*limbs)-int(52*(Digits-j));
        const unsigned lo=bit<0?0:unsigned(bit),skip=lo-bit,q=lo/64,r=lo%64;
        alignas(64) uint64_t lanes[8];
        for(unsigned u=0;u<8;++u){
            uint64_t v=0;if(input[u]){v=q<limbs?input[u][q]>>r:0;if(r && q+1<limbs)v|=input[u][q+1]<<(64-r);}
            lanes[u]=(v<<skip)&((uint64_t(1)<<52)-1);
        }
        x[j]=_mm512_load_si512(lanes);
    }
    #pragma clang loop unroll(full)
    for(unsigned r=0;r<9;++r){
        __m512i carry=zero;
        #pragma clang loop unroll(full)
        for(unsigned j=0;j<Digits;++j){
            const auto low=_mm512_madd52lo_epu64(zero,x[j],b),high=_mm512_madd52hi_epu64(zero,x[j],b);
            const auto t=_mm512_add_epi64(low,carry);
            carry=_mm512_add_epi64(high,_mm512_srli_epi64(t,52));x[j]=_mm512_and_si512(t,mask);
        }
        if(!r)_mm512_mask_storeu_epi64(first,active,carry);
        if(r<8)digits[r]=emit(carry);else _mm512_mask_storeu_epi64(overlap,active,carry);
    }
    transpose(digits);
    for(unsigned u=0;u<8;++u)_mm512_storeu_si512(out+64*u,digits[u]);
}
}
