// Original libsbn code; LGPL-3.0-or-later. Private native SIMD island.
#pragma once
#include "value/bitwise.hpp"
#include <immintrin.h>
#include <cstring>
namespace sbn::v3::bits {
template<unsigned W>struct Vec;
template<>struct Vec<8>{
    using V=__m512i;static constexpr unsigned all=255;
    static V zero(){return _mm512_setzero_si512();}static V one(){return _mm512_set1_epi64(1);}
    static V load(const U*p){return _mm512_loadu_si512(p);}static V load(const U*p,unsigned m){return _mm512_maskz_loadu_epi64(__mmask8(m),p);}
    static void store(U*p,V a){_mm512_storeu_si512(p,a);}
    static void store(U*p,V a,unsigned m){_mm512_mask_storeu_epi64(p,__mmask8(m),a);}
    static unsigned nonzero(V a){return _mm512_cmpneq_epi64_mask(a,zero());}
    static V dec(V a,unsigned m){return _mm512_mask_sub_epi64(a,__mmask8(m),a,one());}
    static V inc(V a,unsigned m){return _mm512_mask_add_epi64(a,__mmask8(m),a,one());}
    static unsigned ones(V a){return _mm512_cmpeq_epi64_mask(a,_mm512_set1_epi64(-1));}
    static V And(V a,V b){return _mm512_and_si512(a,b);}static V Or(V a,V b){return _mm512_or_si512(a,b);}
    static V Xor(V a,V b){return _mm512_xor_si512(a,b);}static V AndNot(V a,V b){return _mm512_andnot_si512(a,b);}
};
template<>struct Vec<4>{
    using V=__m256i;static constexpr unsigned all=15;
    static V zero(){return _mm256_setzero_si256();}static V one(){return _mm256_set1_epi64x(1);}
    static V load(const U*p){return _mm256_loadu_si256(reinterpret_cast<const V*>(p));}static V load(const U*p,unsigned m){return _mm256_maskz_loadu_epi64(__mmask8(m),p);}
    static void store(U*p,V a){_mm256_storeu_si256(reinterpret_cast<V*>(p),a);}
    static void store(U*p,V a,unsigned m){_mm256_mask_storeu_epi64(p,__mmask8(m),a);}
    static unsigned nonzero(V a){return _mm256_cmpneq_epi64_mask(a,zero());}
    static V dec(V a,unsigned m){return _mm256_mask_sub_epi64(a,__mmask8(m),a,one());}
    static V inc(V a,unsigned m){return _mm256_mask_add_epi64(a,__mmask8(m),a,one());}
    static unsigned ones(V a){return _mm256_cmpeq_epi64_mask(a,_mm256_set1_epi64x(-1));}
    static V And(V a,V b){return _mm256_and_si256(a,b);}static V Or(V a,V b){return _mm256_or_si256(a,b);}
    static V Xor(V a,V b){return _mm256_xor_si256(a,b);}static V AndNot(V a,V b){return _mm256_andnot_si256(a,b);}
};
template<int Op,bool A,bool B>inline U boolean(U a,U b){
    if constexpr(Op==SBN3_BIT_XOR)return a^b;
    if constexpr(Op==SBN3_BIT_AND){if constexpr(A&&B)return a|b;else if constexpr(A)return ~a&b;else if constexpr(B)return a&~b;else return a&b;}
    if constexpr(A&&B)return a&b;else if constexpr(A)return a&~b;else if constexpr(B)return ~a&b;else return a|b;
}
template<unsigned W,int Op,bool A,bool B>inline typename Vec<W>::V boolean(typename Vec<W>::V a,typename Vec<W>::V b){
    using O=Vec<W>;
    if constexpr(Op==SBN3_BIT_XOR)return O::Xor(a,b);
    if constexpr(Op==SBN3_BIT_AND){if constexpr(A&&B)return O::Or(a,b);else if constexpr(A)return O::AndNot(a,b);else if constexpr(B)return O::AndNot(b,a);else return O::And(a,b);}
    if constexpr(A&&B)return O::And(a,b);else if constexpr(A)return O::AndNot(b,a);else if constexpr(B)return O::AndNot(a,b);else return O::Or(a,b);
}
template<unsigned W,bool Negative>inline typename Vec<W>::V prefix_load(const U*a,size_t n,size_t i,size_t pivot){
    using O=Vec<W>;auto v=i>=n?O::zero():n-i>=W?O::load(a+i):O::load(a+i,(1u<<unsigned(n-i))-1);
    if constexpr(Negative)if(i<=pivot)v=O::dec(v,pivot-i>=W-1?O::all:(1u<<unsigned(pivot-i+1))-1);
    return v;
}
template<unsigned W,int Op,bool A,bool B,bool Encoded=false>
[[gnu::always_inline]]inline size_t single_vector(U*out,const U*a,size_t an,const U*b,size_t bn,size_t n)noexcept{
    if(!n)return 0;
    using O=Vec<W>;constexpr bool negative=Op==SBN3_BIT_AND?A&&B:Op==SBN3_BIT_OR?A||B:A!=B;
    const unsigned live=_bzhi_u32(O::all,unsigned(n));
    auto x=O::load(a,_bzhi_u32(O::all,unsigned(std::min(an,n)))),y=O::load(b,_bzhi_u32(O::all,unsigned(std::min(bn,n))));
    bool easy=true;
    if constexpr(A||B){
        const U lo_a=an?a[0]:0,lo_b=bn?b[0]:0;
        easy=(!A||lo_a!=0)&&(!B||lo_b!=0);
        if constexpr(negative)easy=easy&&boolean<Op,A,B>(lo_a-U(A),lo_b-U(B))!=UINT64_MAX;
    }
    unsigned am=1,bm=1;
    if constexpr(A)if(!easy){const unsigned m=O::nonzero(x)&live;am=(m^(m-1))&live;}
    if constexpr(B)if(!easy){const unsigned m=O::nonzero(y)&live;bm=(m^(m-1))&live;}
    if constexpr(A)x=O::dec(x,am);if constexpr(B)y=O::dec(y,bm);
    auto v=boolean<W,Op,A,B>(x,y);bool carry=false;
    if constexpr(negative){unsigned inc=1;if(!easy){const unsigned m=O::ones(v)&live;inc=(m^(m+1))&live;carry=m==live;}v=O::inc(v,inc);}
    const unsigned mask=O::nonzero(v)&live;O::store(out,v,live);
    size_t size=mask?32-unsigned(__builtin_clz(mask)):0;if(carry){out[n]=1;size=n+1;}
    if constexpr(Encoded&&negative)if(size)size|=size_t(1)<<63;
    return size;
}
template<unsigned W,unsigned Unroll,int Op,bool A,bool B>
size_t binary_impl(U*out,const U*a,size_t an,const U*b,size_t bn) noexcept {
    constexpr bool negative=Op==SBN3_BIT_AND?A&&B:Op==SBN3_BIT_OR?A||B:A!=B;
    const size_t n=shape(sbn3_bit_operation(Op),an,A,bn,B).words;if(!n)return 0;
    if constexpr(W>1)if(n<=W)return single_vector<W,Op,A,B>(out,a,an,b,bn,n);
    const size_t ap=A?first_nonzero(a,std::min(an,n)):0,bp=B?first_nonzero(b,std::min(bn,n)):0;
    const size_t prefix=std::max(A?std::min(n,ap+1):0,B?std::min(n,bp+1):0);
    size_t i=0;bool carry=negative;
    if constexpr(W>1){using O=Vec<W>;
        for(;n-i>=W&&(i<prefix||carry);i+=W){
            auto v=boolean<W,Op,A,B>(prefix_load<W,A>(a,an,i,ap),prefix_load<W,B>(b,bn,i,bp));
            if constexpr(negative)if(carry){const unsigned m=O::ones(v);v=O::inc(v,(m^(m+1))&O::all);carry=m==O::all;}
            O::store(out+i,v);
        }
    }
    // Only a partial last vector, or scalar diagnostic, can remain here.
    while(i<n&&(i<prefix||carry)){
        U x=i<an?a[i]:0,y=i<bn?b[i]:0;
        if constexpr(A)x-=U(i<=ap);if constexpr(B)y-=U(i<=bp);
        U v=boolean<Op,A,B>(x,y);if constexpr(negative){const U old=v;v+=carry;carry=carry&&v<old;}
        out[i++]=v;
    }
    const size_t common=std::min({n,an,bn});
    if constexpr(W>1){using O=Vec<W>;
        for(;i+W*Unroll<=common;i+=W*Unroll){
#pragma clang loop unroll(full)
            for(unsigned k=0;k<Unroll;++k)O::store(out+i+k*W,boolean<W,Op,A,B>(O::load(a+i+k*W),O::load(b+i+k*W)));
        }
        for(;i+W<=common;i+=W)O::store(out+i,boolean<W,Op,A,B>(O::load(a+i),O::load(b+i)));
    }
    #pragma clang loop vectorize(disable) interleave(disable) unroll_count(4)
    for(;i<common;++i)out[i]=boolean<Op,A,B>(a[i],b[i]);
    // Every admitted nonempty tail is the unchanged longer magnitude. Its
    // decrement pivot and the result's increment have already been consumed.
    if(i<n){const U*longer=an>bn?a:b;if(out+i!=longer+i)memmove(out+i,longer+i,(n-i)*8);}
    if(carry){out[n]=1;return n+1;}
    return normalized_size(out,n);
}
template<unsigned W,unsigned Unroll,int Op>
[[gnu::always_inline]] inline size_t dispatch_signs(U*r,const U*a,size_t an,bool sa,const U*b,size_t bn,bool sb) noexcept {
    switch(unsigned(sa)*2+sb){
    case 0:return binary_impl<W,Unroll,Op,false,false>(r,a,an,b,bn);
    case 1:return binary_impl<W,Unroll,Op,false,true>(r,a,an,b,bn);
    case 2:return binary_impl<W,Unroll,Op,true,false>(r,a,an,b,bn);
    default:return binary_impl<W,Unroll,Op,true,true>(r,a,an,b,bn);
    }
}
template<unsigned W,unsigned Unroll>
size_t binary_width(U*r,const U*a,size_t an,bool sa,const U*b,size_t bn,bool sb,sbn3_bit_operation op) noexcept {
    switch(op){case SBN3_BIT_AND:return dispatch_signs<W,Unroll,SBN3_BIT_AND>(r,a,an,sa,b,bn,sb);
    case SBN3_BIT_OR:return dispatch_signs<W,Unroll,SBN3_BIT_OR>(r,a,an,sa,b,bn,sb);
    default:return dispatch_signs<W,Unroll,SBN3_BIT_XOR>(r,a,an,sa,b,bn,sb);}
}
}
