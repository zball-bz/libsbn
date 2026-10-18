#pragma once
// Original libsbn GCD kernels; LGPL-3.0-or-later.
// The leading-word reducer is GMP-derived; see gcd_head.hpp and its notice.
#include "sbn3/gcd.h"
#include "gcd_head.hpp"
#include "gcd_ifma52.hpp"
#include "value/divrem_words.hpp"
#include "common/checked.hpp"
#include "common/small_checks.h"
#include "gcd_odd_reduce.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <cassert>
namespace sbn::v3::gcd_small {
using U=uint64_t;using I=__int128_t;using V=__m512i;
// Two measured representation/kernel crossovers, no per-size plan search.
inline constexpr size_t vector_from=48,persistent_from=160;
[[gnu::always_inline]] inline uint64_t word(U a,U b) noexcept {
    if(!a||!b)return a|b;
    const unsigned s=__builtin_ctzll(a|b);a>>=__builtin_ctzll(a);b>>=__builtin_ctzll(b);
    while(a!=b){const U d=a-b;const unsigned z=__builtin_ctzll(d);const bool less=a<b;
        const U lo=less?a:b;a=(less?U(0)-d:d)>>z;b=lo;}
    return a<<s;
}
[[maybe_unused]] static unsigned zeros128(__uint128_t a){return U(a)?unsigned(__builtin_ctzll(U(a))):64+unsigned(__builtin_ctzll(U(a>>64)));}
[[maybe_unused]] static __uint128_t word2(__uint128_t a,__uint128_t b){
    if(!a||!b)return a|b;
    const unsigned s=zeros128(a|b);a>>=zeros128(a);b>>=zeros128(b);
    // One useful full-precision head batch, then the binary terminal. With
    // short or nearly equal heads the head reducer cannot repay its setup.
    const U ah=U(a>>64),bh=U(b>>64);
    if((ah>>32)&&(bh>>32)){
        const U delta=ah>bh?ah-bh:bh-ah;
        if(delta>=2){
            Matrix m;
            if(head_matrix<33,true>(ah,U(a),bh,U(b),&m)){
                // Accepted rows are nonnegative and fit 128 bits. Unsigned
                // low products recover them exactly despite intermediate wrap.
                const auto x=a*m.u[1][1]-b*m.u[0][1],y=b*m.u[0][0]-a*m.u[1][0];
                if(!x||!y)return (x|y)<<s;
                a=x>>zeros128(x);b=y>>zeros128(y);
            }
        }
    }
    while(a!=b){if(!(a>>64)&&!(b>>64))return __uint128_t(word(U(a),U(b)))<<s;
        const __uint128_t d=a-b;const unsigned z=zeros128(d);const bool less=a<b;
        const auto lo=less?a:b;a=(less?__uint128_t(0)-d:d)>>z;b=lo;}
    return a<<s;
}
[[maybe_unused]] static size_t norm(const U*a,size_t n){while(n&&!a[n-1])--n;return n;}
[[maybe_unused]] static size_t shared_zeros(const U*a,const U*b,size_t n){
    if(const U v=a[0]|b[0])return unsigned(__builtin_ctzll(v));
    size_t j=0;for(;j+8<=n;j+=8){const V v=_mm512_or_si512(_mm512_loadu_si512(a+j),_mm512_loadu_si512(b+j));const unsigned mask=_mm512_test_epi64_mask(v,v);
        if(mask){j+=unsigned(__builtin_ctz(mask));return 64*j+unsigned(__builtin_ctzll(a[j]|b[j]));}}
    while(!(a[j]|b[j]))++j;return 64*j+unsigned(__builtin_ctzll(a[j]|b[j]));
}
[[maybe_unused]] static size_t copy_shift(U*out,const U*a,size_t n,size_t bits){const size_t skip=bits/64;const unsigned s=bits%64;a+=skip;n-=skip;
    if(!s){if(out!=a)memmove(out,a,n*8);return n;}
    for(size_t j=0;j<n;++j)out[j]=(a[j]>>s)|(j+1<n?a[j+1]<<(64-s):0);
    return n-unsigned(!out[n-1]);
}
[[maybe_unused]] static size_t emit(U*out,const U*a,size_t n,size_t shift){if(!n)return 0;const size_t lo=shift/64;const unsigned s=shift%64;
    memset(out,0,lo*8);U c=0;for(size_t j=0;j<n;++j){const U v=a[j];out[lo+j]=s?(v<<s)|c:v;c=s?v>>(64-s):0;}
    if(c)out[lo+n++]=c;return lo+n;}
[[maybe_unused]] static size_t scalar_matrix(U*r,U*s,const U*a,const U*b,size_t n,const Matrix&m){
    I ca=0,cb=0;
    for(size_t j=0;j<n;++j){const U x=a[j],y=b[j];
        const I v=I(x)*m.u[1][1]-I(y)*m.u[0][1]+ca;
        const I w=I(y)*m.u[0][0]-I(x)*m.u[1][0]+cb;
        r[j]=U(v);s[j]=U(w);ca=v>>64;cb=w>>64;}
    require(!ca&&!cb,SBN3_FATAL_MATH,"GCD matrix signed carry");return n-unsigned((r[n-1]|s[n-1])==0);
}
struct Product {V lo,hi;};
[[maybe_unused]] static inline Product multiply(V x,U c){
    const V mask=_mm512_set1_epi64((U(1)<<52)-1),zero=_mm512_setzero_si512();
    const V xl=_mm512_and_si512(x,mask),xh=_mm512_srli_epi64(x,52);
    const V cl=_mm512_set1_epi64(c&((U(1)<<52)-1));
    V l=_mm512_madd52lo_epu64(zero,xl,cl),h=_mm512_madd52hi_epu64(zero,xl,cl);
    V mid=_mm512_madd52lo_epu64(h,xh,cl),high=_mm512_madd52hi_epu64(zero,xh,cl);
    {const V ch=_mm512_set1_epi64(c>>52);
        mid=_mm512_madd52lo_epu64(mid,xl,ch);high=_mm512_madd52hi_epu64(high,xl,ch);
        high=_mm512_madd52lo_epu64(high,xh,ch);}
    return {_mm512_or_si512(l,_mm512_slli_epi64(mid,52)),_mm512_add_epi64(_mm512_srli_epi64(mid,12),_mm512_slli_epi64(high,40))};
}
// Same u64 value representation; scalar coefficient is below 2^63.
// Four 32x32 products avoid the seven radix-2^52 partial products.
[[maybe_unused]] static inline Product multiply32(V x,U c){
    const V mask=_mm512_set1_epi64(0xffffffffu),lo=_mm512_set1_epi64(uint32_t(c)),hi=_mm512_set1_epi64(c>>32),xh=_mm512_srli_epi64(x,32);
    const V p00=_mm512_mul_epu32(x,lo),p10=_mm512_mul_epu32(xh,lo),p01=_mm512_mul_epu32(x,hi),p11=_mm512_mul_epu32(xh,hi);
    const V mid=_mm512_add_epi64(_mm512_add_epi64(_mm512_srli_epi64(p00,32),_mm512_and_si512(p10,mask)),p01);
    return {_mm512_or_si512(_mm512_and_si512(p00,mask),_mm512_slli_epi64(mid,32)),
            _mm512_add_epi64(_mm512_add_epi64(p11,_mm512_srli_epi64(p10,32)),_mm512_srli_epi64(mid,32))};
}
template<bool Word32>[[maybe_unused]] static inline Product multiply_selected(V x,U c){
    if constexpr(Word32)return multiply32(x,c);
    else return multiply(x,c);
}
[[maybe_unused]] static inline Product difference(Product a,Product b){V l=_mm512_sub_epi64(a.lo,b.lo);
    const auto borrow=_mm512_cmplt_epu64_mask(a.lo,b.lo);
    return {l,_mm512_mask_sub_epi64(_mm512_sub_epi64(a.hi,b.hi),borrow,_mm512_sub_epi64(a.hi,b.hi),_mm512_set1_epi64(1))};}
// Carry from each 128-bit digit is signed. Away from low-word endpoints,
// the residual +/-1 carry is independent of its predecessor. Endpoint blocks
// take the exact scalar stitch; no probabilistic correctness assumption.
[[maybe_unused]] static inline V carry_row(Product d,V &carry){
    const V zero=_mm512_setzero_si512(),one=_mm512_set1_epi64(1);
    const V prev=_mm512_alignr_epi64(d.hi,carry,7);
    V v=_mm512_add_epi64(d.lo,prev);
    const auto plus=_mm512_cmplt_epu64_mask(v,d.lo),minus=_mm512_cmplt_epi64_mask(prev,zero);
    const V e=_mm512_sub_epi64(_mm512_maskz_mov_epi64(plus,one),_mm512_maskz_mov_epi64(minus,one));
    const auto edge=_mm512_cmpeq_epi64_mask(v,zero)|_mm512_cmpeq_epi64_mask(v,_mm512_set1_epi64(-1));
    if(!edge){v=_mm512_add_epi64(v,_mm512_alignr_epi64(e,zero,7));
        carry=_mm512_add_epi64(d.hi,e);return v;}
    
    alignas(64) U lo[8],hh[8];_mm512_store_si512(lo,d.lo);_mm512_store_si512(hh,d.hi);
    I c=int64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(carry,3),1));
    for(unsigned j=0;j<8;++j){const I t=I(lo[j])+c;lo[j]=U(t);c=I(int64_t(hh[j]))+(t>>64);}carry=_mm512_set1_epi64(U(c));
    return _mm512_load_si512(lo);
}
template<bool Word32=false>[[maybe_unused]] static size_t ifma_matrix(U*r,U*s,const U*a,const U*b,size_t n,const Matrix&m){
    V va=_mm512_setzero_si512(),vb=va;size_t j=0;
    for(;j+8<n;j+=8){const V x=_mm512_loadu_si512(a+j),y=_mm512_loadu_si512(b+j);
        const auto v=difference(multiply_selected<Word32>(x,m.u[1][1]),multiply_selected<Word32>(y,m.u[0][1]));
        const auto w=difference(multiply_selected<Word32>(y,m.u[0][0]),multiply_selected<Word32>(x,m.u[1][0]));
        const V ra=carry_row(v,va),rb=carry_row(w,vb);_mm512_storeu_si512(r+j,ra);_mm512_storeu_si512(s+j,rb);}
    I ca=int64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(va,3),1)),cb=int64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(vb,3),1));
    for(;j<n;++j){const U x=a[j],y=b[j];const I v=I(x)*m.u[1][1]-I(y)*m.u[0][1]+ca;
        const I w=I(y)*m.u[0][0]-I(x)*m.u[1][0]+cb;r[j]=U(v);s[j]=U(w);ca=v>>64;cb=w>>64;}
    require(!ca&&!cb,SBN3_FATAL_MATH,"GCD matrix signed carry");return n-unsigned((r[n-1]|s[n-1])==0);
}
[[maybe_unused]] static U remainder_word(const U*a,size_t n,U d){
    using namespace sbn::v3::divrem_words;
    if(d==1)return 0;
    const unsigned s=__builtin_clzll(d);const U D=d<<s,inv=invert_limb(D);U r=s?a[n-1]>>(64-s):0;
    for(size_t j=n;j-->0;){const U lo=s?(a[j]<<s)|(j?a[j-1]>>(64-s):0):a[j];U q;udiv_qrnnd_preinv(q,r,r,lo,D,inv);(void)q;}
    return r>>s;
}
[[maybe_unused]] static U gcd_word_operand(const U*a,size_t n,U d){
    const unsigned z=__builtin_ctzll(d),common=a[0]?std::min(z,unsigned(__builtin_ctzll(a[0]))):z;
    d>>=z;return word(d,odd_gcd_residue(a,n,d))<<common;
}
[[maybe_unused]] static __uint128_t remainder_two(const U*a,size_t n,const U*d){
    using namespace sbn::v3::divrem_words;
    const unsigned s=__builtin_clzll(d[1]);const U dh=s?(d[1]<<s)|(d[0]>>(64-s)):d[1],dl=d[0]<<s;
    const U inverse=invert_pi1(dh,dl);U rh=0,rl=s?a[n-1]>>(64-s):0;
    for(size_t j=n;j-->0;){const U low=s?(a[j]<<s)|(j?a[j-1]>>(64-s):0):a[j];U q;udiv_qr_3by2(q,rh,rl,rh,rl,low,dh,dl,inverse);(void)q;}
    return ((__uint128_t(rh)<<64)|rl)>>s;
}
template<bool Capture=false>[[maybe_unused]] static size_t remainder(U*a,size_t an,U*b,size_t bn,U*qlo=nullptr){
    if constexpr(Capture)*qlo=0;
    using namespace sbn::v3::divrem_words;
    if(an<bn)return an;
    if(bn==1){a[0]=remainder_word(a,an,b[0]);return bool(a[0]);}
    if(bn==2){const auto r=remainder_two(a,an,b);a[0]=U(r);a[1]=U(r>>64);return norm(a,2);}
    const unsigned s=__builtin_clzll(b[bn-1]);
    if(s){const U c=shift_left(a,a,an,s);a[an]=c;an+=bool(c);shift_left(b,b,bn,s);}
    a[an]=0;const U inverse=invert_pi1(b[bn-1],b[bn-2]);
    // Normalized schoolbook remainder. The leading 3/2 estimate is an upper
    // bound on each quotient digit; addback restores the exact residue.
    // Quotient digits never leave registers or borrow the caller's output.
    for(size_t pos=an-bn+1;pos-->0;){
        U q=UINT64_MAX;
        if(a[pos+bn]!=b[bn-1]||a[pos+bn-1]!=b[bn-2]){
            U r1,r0;udiv_qr_3by2(q,r1,r0,a[pos+bn],a[pos+bn-1],a[pos+bn-2],b[bn-1],b[bn-2],inverse);
        }
        const U cy=sbn3i_submul_1(a+pos,b,long(bn),q);unsigned neg=a[pos+bn]<cy;a[pos+bn]-=cy;
        unsigned fixes=0;
        while(neg){
            require(++fixes<=2,SBN3_FATAL_MATH,"GCD remainder correction");
            const U c=sbn3i_add_n(a+pos,a+pos,b,long(bn)),old=a[pos+bn];a[pos+bn]+=c;neg-=unsigned(a[pos+bn]<old);
        }
        if constexpr(Capture)if(!pos)*qlo=q-fixes;
        require(!a[pos+bn],SBN3_FATAL_MATH,"GCD remainder high word");
    }
    if(s){shift_right(a,a,bn,s);shift_right(b,b,bn,s);}return norm(a,bn);
}
[[maybe_unused]] static V trailing(V x){return _mm512_popcnt_epi64(_mm512_andnot_si512(x,_mm512_sub_epi64(x,_mm512_set1_epi64(1))));}
[[maybe_unused]] static void batch_chunk(U*out,const U*aa,const U*bb,__mmask8 lanes) noexcept {
    V a=_mm512_maskz_loadu_epi64(lanes,aa),b=_mm512_maskz_loadu_epi64(lanes,bb);const V zero=_mm512_setzero_si512();
    const auto nonzero=_mm512_cmpneq_epi64_mask(a,zero)&_mm512_cmpneq_epi64_mask(b,zero);
    const V trivial=_mm512_or_si512(a,b),sh=trailing(trivial);
    a=_mm512_maskz_mov_epi64(nonzero,_mm512_srlv_epi64(a,trailing(a)));b=_mm512_srlv_epi64(b,trailing(b));
    auto active=_mm512_cmpneq_epi64_mask(a,zero);
    while(active){const V lo=_mm512_min_epu64(a,b),d=_mm512_sub_epi64(_mm512_max_epu64(a,b),lo);
        a=_mm512_mask_mov_epi64(a,active,_mm512_srlv_epi64(d,trailing(d)));b=_mm512_mask_mov_epi64(b,active,lo);active=_mm512_cmpneq_epi64_mask(a,zero);}
    _mm512_mask_storeu_epi64(out,lanes,_mm512_mask_mov_epi64(trivial,nonzero,_mm512_sllv_epi64(b,sh)));
}
[[maybe_unused]] static constexpr U mask52=(U(1)<<52)-1;
[[maybe_unused]] static U read52(const U*a,size_t n,size_t bit){const size_t j=bit/52;const unsigned s=bit%52;
    U v=j<n?a[j]>>s:0;if(j+1<n)v|=a[j+1]<<(52-s);if(s>40&&j+2<n)v|=a[j+2]<<(104-s);return v;}
[[maybe_unused]] static size_t expand52(U*a,size_t words){const size_t n=(64*words+51)/52;
    for(size_t i=n;i-->0;){const size_t bit=52*i,j=bit/64;const unsigned s=bit%64;
        a[i]=((a[j]>>s)|(s&&j+1<words?a[j+1]<<(64-s):0))&mask52;}return norm(a,n);}
[[maybe_unused]] static size_t compact52(U*a,size_t n){const size_t words=(52*n+63)/64;
    for(size_t i=0;i<words;++i)a[i]=read52(a,n,64*i);return norm(a,words);}
[[maybe_unused]] static size_t emit52(U*out,const U*a,size_t n,size_t shift){if(!n)return 0;
    const size_t bits=52*(n-1)+64-unsigned(__builtin_clzll(a[n-1]));
    const size_t words=(bits+63)/64,offset=shift/64;const unsigned s=shift%64;memset(out,0,offset*8);U carry=0;
    for(size_t j=0;j<words;++j){const U v=read52(a,n,64*j);out[offset+j]=s?(v<<s)|carry:v;carry=s?v>>(64-s):0;}
    size_t used=offset+words;if(carry)out[used++]=carry;return norm(out,used);
}
[[maybe_unused]] static size_t matrix52(U*a,U*b,size_t n,const Matrix&m){
    require(std::max({m.u[0][0],m.u[0][1],m.u[1][0],m.u[1][1]})<(U(1)<<63),SBN3_FATAL_MATH,"GCD matrix coefficient bound");
    // A final unusually large quotient may produce a wider matrix even when
    // the usual head stop leaves ~76 bits. Its exact transformation is valid;
    // it simply cannot be passed to a 52-bit coefficient multiply.
    if(std::max({m.u[0][0],m.u[0][1],m.u[1][0],m.u[1][1]})>mask52){
        I ca=0,cb=0;for(size_t j=0;j<n;++j){const U x=a[j],y=b[j];const I u=I(x)*m.u[1][1]-I(y)*m.u[0][1]+ca,v=I(y)*m.u[0][0]-I(x)*m.u[1][0]+cb;
            a[j]=U(u)&mask52;b[j]=U(v)&mask52;ca=u>>52;cb=v>>52;}
        require(!ca&&!cb,SBN3_FATAL_MATH,"GCD matrix signed carry");while(n&&(a[n-1]|b[n-1])==0)--n;return n;
    }
    const V zero=_mm512_setzero_si512(),c00=_mm512_set1_epi64(m.u[0][0]),c11=_mm512_set1_epi64(m.u[1][1]);
    const V neg01=_mm512_set1_epi64((-m.u[0][1])&mask52),neg10=_mm512_set1_epi64((-m.u[1][0])&mask52);
    const __mmask8 has01=m.u[0][1]?255:0,has10=m.u[1][0]?255:0;
    V va=zero,vb=zero;size_t j=0;
    for(;j+8<n;j+=8){const V a0=_mm512_loadu_si512(a+j),b0=_mm512_loadu_si512(b+j);
        const auto v=ifma52::difference(a0,b0,c11,neg01,has01),w=ifma52::difference(b0,a0,c00,neg10,has10);
        const V x=ifma52::carry(v.lo,v.hi,va),y=ifma52::carry(w.lo,w.hi,vb);_mm512_storeu_si512(a+j,x);_mm512_storeu_si512(b+j,y);}
    I ca=int64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(va,3),1)),cb=int64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(vb,3),1));
    for(;j<n;++j){const U x=a[j],y=b[j];const I u=I(x)*m.u[1][1]-I(y)*m.u[0][1]+ca,v=I(y)*m.u[0][0]-I(x)*m.u[1][0]+cb;
        a[j]=U(u)&mask52;b[j]=U(v)&mask52;ca=u>>52;cb=v>>52;}
    require(!ca&&!cb,SBN3_FATAL_MATH,"GCD matrix signed carry");while(n&&(a[n-1]|b[n-1])==0)--n;return n;
}
}
