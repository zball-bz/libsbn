// Original libsbn code; LGPL-3.0-or-later.
#include "gcd_small_core.hpp"
namespace sbn::v3::gcd_small {
static size_t finish_short(U*out,const U*a,size_t an,const U*b,size_t bn,size_t common){
    if(!bn)return emit(out,a,an,common);
    if(an<=2){
        const auto av=__uint128_t(a[0])|(an>1?__uint128_t(a[1])<<64:0);
        const auto bv=__uint128_t(b[0])|(bn>1?__uint128_t(b[1])<<64:0);
        const auto g=word2(av,bv);U v[2]{U(g),U(g>>64)};return emit(out,v,1+bool(v[1]),common);
    }
    if(bn==1){U g=gcd_word_operand(a,an,b[0]);return emit(out,&g,1,common);}
    const auto d=__uint128_t(b[0])|(__uint128_t(b[1])<<64),g=word2(d,remainder_two(a,an,b));
    U v[2]{U(g),U(g>>64)};return emit(out,v,1+bool(v[1]),common);
}
static size_t gcd52(U*out,const U*aa,size_t an,const U*bb,size_t bn,void*work){
    const size_t cap=(((64*(std::max(an,bn)+1)+51)/52)+7)&~size_t(7);U*a=static_cast<U*>(work),*b=a+cap;
    const size_t common=shared_zeros(aa,bb,std::min(an,bn));
    an=copy_shift(a,aa,an,common);bn=copy_shift(b,bb,bn,common);size_t n=std::max(an,bn);
    memset(a+an,0,(n-an+1)*8);memset(b+bn,0,(n-bn+1)*8);
    // Delay representation conversion until a productive wide head exists.
    // Near-equal/imbalanced inputs may collapse to a short divisor first.
    for(;;){
        if(an<bn){std::swap(a,b);std::swap(an,bn);}
        if(!bn)return emit(out,a,an,common);
        if(bn<=2)return finish_short(out,a,an,b,bn,common);
        Matrix m;const U mask=a[n-1]|b[n-1];const unsigned sh=__builtin_clzll(mask);
        const U ah=sh?(a[n-1]<<sh)|(a[n-2]>>(64-sh)):a[n-1],bh=sh?(b[n-1]<<sh)|(b[n-2]>>(64-sh)):b[n-1];
        const U al=sh?(a[n-2]<<sh)|(a[n-3]>>(64-sh)):a[n-2],bl=sh?(b[n-2]<<sh)|(b[n-3]>>(64-sh)):b[n-2];
        const bool wide=n>=persistent_from;
        const bool got=wide?head_matrix<44,true>(ah,al,bh,bl,&m):head_matrix<33,true>(ah,al,bh,bl,&m);
        if(got){
            if(wide){
                an=expand52(a,an);bn=expand52(b,bn);n=std::max(an,bn);
                memset(a+an,0,(n-an)*8);memset(b+bn,0,(n-bn)*8);
                n=matrix52(a,b,n,m);break;
            }
            require(std::max({m.u[0][0],m.u[0][1],m.u[1][0],m.u[1][1]})<(U(1)<<63),SBN3_FATAL_MATH,"GCD matrix coefficient bound");
            n=n>=vector_from?ifma_matrix(a,b,a,b,n,m):scalar_matrix(a,b,a,b,n,m);
            an=norm(a,n);bn=norm(b,n);n=std::max(an,bn);continue;
        }
        const int cmp=sbn::v3::divrem_words::compare(a,b,n);if(!cmp)return emit(out,a,n,common);
        if(cmp<0)std::swap(a,b);
        sbn3i_sub_n(a,a,b,long(n));an=norm(a,n);bn=norm(b,n);if(!an)return emit(out,b,bn,common);
        if(an<bn){std::swap(a,b);std::swap(an,bn);}
        if(bn<=2)return finish_short(out,a,an,b,bn,common);
        an=remainder(a,an,b,bn);if(!an)return emit(out,b,bn,common);
        n=std::max(an,bn);memset(a+an,0,(n-an+1)*8);memset(b+bn,0,(n-bn+1)*8);
    }
    for(;;){assert(n);const size_t bits=52*(n-1)+64-unsigned(__builtin_clzll(a[n-1]|b[n-1]));
        if(bits<=128){const auto av=__uint128_t(read52(a,n,0))|(__uint128_t(read52(a,n,64))<<64),bv=__uint128_t(read52(b,n,0))|(__uint128_t(read52(b,n,64))<<64);
            const auto g=word2(av,bv);U v[2]{U(g),U(g>>64)};return emit(out,v,1+bool(v[1]),common);}
        Matrix m;const U ah=read52(a,n,bits-64),al=read52(a,n,bits-128),bh=read52(b,n,bits-64),bl=read52(b,n,bits-128);
        const bool got=head_matrix<44,true>(ah,al,bh,bl,&m);
        if(got){n=matrix52(a,b,n,m);continue;}
        
        const int cmp=sbn::v3::divrem_words::compare(a,b,n);if(!cmp)return emit52(out,a,n,common);if(cmp<0)std::swap(a,b);
        int64_t borrow=0;for(size_t j=0;j<n;++j){const int64_t d=int64_t(a[j])-int64_t(b[j])-borrow;a[j]=U(d)&mask52;borrow=d<0;}require(!borrow,SBN3_FATAL_MATH,"GCD subtraction borrow");
        an=norm(a,n);bn=norm(b,n);if(!an)return emit52(out,b,bn,common);
        if(an<bn){std::swap(a,b);std::swap(an,bn);}
        an=compact52(a,an);bn=compact52(b,bn);
        if(bn<=2)return finish_short(out,a,an,b,bn,common);
        an=remainder(a,an,b,bn);
        if(!an)return emit(out,b,bn,common);
        an=expand52(a,an);bn=expand52(b,bn);n=std::max(an,bn);memset(a+an,0,(n-an)*8);memset(b+bn,0,(n-bn)*8);
    }
}
static size_t compute(U*out,const U*aa,size_t an,const U*bb,size_t bn,void*work) noexcept {
    an=norm(aa,an);bn=norm(bb,bn);if(!an){if(bn)memcpy(out,bb,bn*8);return bn;}if(!bn){memcpy(out,aa,an*8);return an;}
    if(an==1&&bn==1){out[0]=word(aa[0],bb[0]);return 1;}
    if(std::min(an,bn)==1){const U d=an==1?aa[0]:bb[0];out[0]=gcd_word_operand(an==1?bb:aa,an==1?bn:an,d);return 1;}
    if(an==bn&&aa[an-1]==bb[bn-1]&&!memcmp(aa,bb,an*8)){memcpy(out,aa,an*8);return an;}
    if(std::max(an,bn)<=2){const auto a=__uint128_t(aa[0])|(an>1?__uint128_t(aa[1])<<64:0),b=__uint128_t(bb[0])|(bn>1?__uint128_t(bb[1])<<64:0);
        const auto g=word2(a,b);out[0]=U(g);if(g>>64){out[1]=U(g>>64);return 2;}return 1;}
    if(std::min(an,bn)==2){const U*d=an==2?aa:bb;const auto modulus=(__uint128_t(d[1])<<64)|d[0];
        const auto g=word2(modulus,remainder_two(an==2?bb:aa,an==2?bn:an,d));out[0]=U(g);if(g>>64){out[1]=U(g>>64);return 2;}return 1;}
    if(std::max(an,bn)>=persistent_from)return gcd52(out,aa,an,bb,bn,work);
    const size_t cap=(std::max(an,bn)+9)&~size_t(7);auto*a=static_cast<U*>(work),*b=a+cap;
    const size_t common=shared_zeros(aa,bb,std::min(an,bn));
    an=copy_shift(a,aa,an,common);bn=copy_shift(b,bb,bn,common);size_t n=std::max(an,bn);memset(a+an,0,(n-an+1)*8);memset(b+bn,0,(n-bn+1)*8);
    while(n>2){Matrix m;const U mask=a[n-1]|b[n-1];assert(mask);const unsigned sh=__builtin_clzll(mask);
        const U ah=sh?(a[n-1]<<sh)|(a[n-2]>>(64-sh)):a[n-1],bh=sh?(b[n-1]<<sh)|(b[n-2]>>(64-sh)):b[n-1];
        const U al=sh?(a[n-2]<<sh)|(a[n-3]>>(64-sh)):a[n-2],bl=sh?(b[n-2]<<sh)|(b[n-3]>>(64-sh)):b[n-2];
        const bool got=head_matrix<33,true>(ah,al,bh,bl,&m);
        if(got){
            require(std::max({m.u[0][0],m.u[0][1],m.u[1][0],m.u[1][1]})<(U(1)<<63),SBN3_FATAL_MATH,"GCD matrix coefficient bound");
            n=n>=vector_from?ifma_matrix(a,b,a,b,n,m):scalar_matrix(a,b,a,b,n,m);continue;}
        
        int cmp=sbn::v3::divrem_words::compare(a,b,n);if(!cmp)return emit(out,a,n,common);
        if(cmp<0)std::swap(a,b);
        sbn3i_sub_n(a,a,b,long(n));an=norm(a,n);bn=norm(b,n);if(!an)return emit(out,b,bn,common);
        if(an<bn){std::swap(a,b);std::swap(an,bn);}
        if(bn<=2)return finish_short(out,a,an,b,bn,common);
        an=remainder(a,an,b,bn);if(!an)return emit(out,b,bn,common);
        n=std::max(an,bn);memset(a+an,0,(n-an+1)*8);memset(b+bn,0,(n-bn+1)*8);
    }
    const auto av=__uint128_t(a[0])|(n>1?__uint128_t(a[1])<<64:0),bv=__uint128_t(b[0])|(n>1?__uint128_t(b[1])<<64:0);
    const auto g=word2(av,bv);U v[2]{U(g),U(g>>64)};return emit(out,v,1+bool(v[1]),common);
}
}
namespace g=sbn::v3::gcd_small;
extern "C" uint64_t sbn3_gcd_u64(uint64_t a,uint64_t b){return g::word(a,b);}
extern "C" size_t sbn3_gcd_small_scratch_bytes(size_t an,size_t bn){
    const size_t n=std::max(an,bn);if(n>SBN3_GCD_SMALL_LIMBS)return SIZE_MAX;
    if(n<=2||std::min(an,bn)<=2)return 0;
    if(n>=g::persistent_from)return 16*((((64*(n+1)+51)/52)+7)&~size_t(7));
    return 16*((n+9)&~size_t(7));
}
extern "C" size_t sbn3_gcd_small(uint64_t*out,const uint64_t*a,size_t an,const uint64_t*b,size_t bn,void*scratch){
    if constexpr(SBN3_CHECK_SMALL){
        using sbn::v3::require;const size_t bytes=sbn3_gcd_small_scratch_bytes(an,bn);
        require(bytes!=SIZE_MAX,SBN3_FATAL_ARGUMENT,"small GCD domain");
        require((a||!an)&&(b||!bn)&&(scratch||!bytes),SBN3_FATAL_ARGUMENT,"small GCD span");
        require(!((reinterpret_cast<uintptr_t>(out)|reinterpret_cast<uintptr_t>(a)|reinterpret_cast<uintptr_t>(b)|reinterpret_cast<uintptr_t>(scratch))&7),SBN3_FATAL_ARGUMENT,"small GCD alignment");
        const size_t al=g::norm(a,an),bl=g::norm(b,bn),n=al&&bl?std::min(al,bl):std::max(al,bl);
        require(out||!n,SBN3_FATAL_ARGUMENT,"small GCD output");
        using sbn::v3::overlaps;
        require(!overlaps(out,n*8,a,an*8)&&!overlaps(out,n*8,b,bn*8)&&!overlaps(scratch,bytes,a,an*8)&&!overlaps(scratch,bytes,b,bn*8)&&!overlaps(scratch,bytes,out,n*8),SBN3_FATAL_ARGUMENT,"small GCD alias");
    }
    return g::compute(out,a,an,b,bn,scratch);
}
extern "C" void sbn3_gcd_u64_batch(uint64_t*out,const uint64_t*a,const uint64_t*b,size_t n){
    if constexpr(SBN3_CHECK_SMALL){using sbn::v3::require;
        require(n<=SIZE_MAX/8&&(!n||(out&&a&&b)),SBN3_FATAL_ARGUMENT,"word GCD batch span");
        require((out==a||!sbn::v3::overlaps(out,n*8,a,n*8))&&(out==b||!sbn::v3::overlaps(out,n*8,b,n*8)),SBN3_FATAL_ARGUMENT,"word GCD batch alias");}
    size_t j=0;for(;j+8<=n;j+=8)g::batch_chunk(out+j,a+j,b+j,255);
    const size_t left=n-j;
    if(left>=4)g::batch_chunk(out+j,a+j,b+j,(__mmask8)((1u<<left)-1));
    else for(;j<n;++j)out[j]=g::word(a[j],b[j]);
}

namespace sbn::v3::gcd {
// Positive reconstruction rows use the same 64-bit IFMA split as residual
// rows. Their high carries are UNSIGNED; the signed residual stitch cannot
// be reused when the sum's high word has its top bit set.
static inline gcd_small::Product unsigned_sum(gcd_small::Product a,gcd_small::Product b){
    const auto lo=_mm512_add_epi64(a.lo,b.lo),hi=_mm512_add_epi64(a.hi,b.hi);
    return {lo,_mm512_mask_add_epi64(hi,_mm512_cmplt_epu64_mask(lo,a.lo),hi,_mm512_set1_epi64(1))};
}
static inline __m512i unsigned_carry(gcd_small::Product a,__m512i &carry){
    const auto zero=_mm512_setzero_si512(),one=_mm512_set1_epi64(1);
    const auto previous=_mm512_alignr_epi64(a.hi,carry,7);
    auto out=_mm512_add_epi64(a.lo,previous);
    const auto correction=_mm512_maskz_mov_epi64(_mm512_cmplt_epu64_mask(out,a.lo),one);
    if(!_mm512_cmpeq_epi64_mask(out,_mm512_set1_epi64(-1))){
        carry=_mm512_add_epi64(a.hi,correction);
        return _mm512_add_epi64(out,_mm512_alignr_epi64(correction,zero,7));
    }
    alignas(64) uint64_t lo[8],hi[8];_mm512_store_si512(lo,a.lo);_mm512_store_si512(hi,a.hi);
    uint64_t c=uint64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(carry,3),1));
    for(unsigned i=0;i<8;++i){const __uint128_t t=__uint128_t(lo[i])+c;lo[i]=uint64_t(t);c=hi[i]+uint64_t(t>>64);}
    carry=_mm512_set1_epi64(c);return _mm512_load_si512(lo);
}
template<bool Narrow,bool Word32=false>static inline gcd_small::Product positive_product(__m512i x,uint64_t c){
    if constexpr(Word32)return gcd_small::multiply32(x,c);
    if constexpr(!Narrow)return gcd_small::multiply(x,c);
    const auto zero=_mm512_setzero_si512(),cv=_mm512_set1_epi64(c),xh=_mm512_srli_epi64(x,52);
    const auto lo=_mm512_madd52lo_epu64(zero,x,cv),hi=_mm512_madd52hi_epu64(zero,x,cv);
    const auto middle=_mm512_madd52lo_epu64(hi,xh,cv),high=_mm512_madd52hi_epu64(zero,xh,cv);
    return {_mm512_or_si512(lo,_mm512_slli_epi64(middle,52)),_mm512_add_epi64(_mm512_srli_epi64(middle,12),_mm512_slli_epi64(high,40))};
}
template<bool Narrow,bool Word32=false>static void compose_head_impl(uint64_t *a,uint64_t *b,size_t n,const gcd_small::Matrix &m){
    auto ca=_mm512_setzero_si512(),cb=ca;size_t j=0;
    for(;j+8<=n;j+=8){const auto x=_mm512_loadu_si512(a+j),y=_mm512_loadu_si512(b+j);
        const auto pa=unsigned_sum(positive_product<Narrow,Word32>(x,m.u[0][0]),positive_product<Narrow,Word32>(y,m.u[1][0]));
        const auto pb=unsigned_sum(positive_product<Narrow,Word32>(x,m.u[0][1]),positive_product<Narrow,Word32>(y,m.u[1][1]));
        _mm512_storeu_si512(a+j,unsigned_carry(pa,ca));_mm512_storeu_si512(b+j,unsigned_carry(pb,cb));
    }
    uint64_t va=uint64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(ca,3),1)),vb=uint64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(cb,3),1));
    for(;j<n;++j){const uint64_t x=a[j],y=b[j];
        const __uint128_t aa=__uint128_t(x)*m.u[0][0]+__uint128_t(y)*m.u[1][0]+va;
        const __uint128_t bb=__uint128_t(x)*m.u[0][1]+__uint128_t(y)*m.u[1][1]+vb;
        a[j]=uint64_t(aa);b[j]=uint64_t(bb);va=aa>>64;vb=bb>>64;
    }
    require(!va&&!vb,SBN3_FATAL_MATH,"HGCD positive matrix carry");
}
void compose_head(uint64_t *a,uint64_t *b,size_t n,const gcd_small::Matrix &m){
    const uint64_t largest=std::max({m.u[0][0],m.u[0][1],m.u[1][0],m.u[1][1]});
    require(largest<(uint64_t(1)<<63),SBN3_FATAL_MATH,"HGCD positive matrix coefficient bound");
    if(largest<(uint64_t(1)<<52))compose_head_impl<true>(a,b,n,m);
    else compose_head_impl<false>(a,b,n,m);
}
void compose_head_word32(uint64_t*a,uint64_t*b,size_t n,const gcd_small::Matrix&m){
    require(!((m.u[0][0]|m.u[0][1]|m.u[1][0]|m.u[1][1])>>63),SBN3_FATAL_MATH,"HGCD positive matrix coefficient bound");
    compose_head_impl<false,true>(a,b,n,m);
}
void apply_head_word32(uint64_t*a,uint64_t*b,size_t n,const gcd_small::Matrix&m){
    require(!((m.u[0][0]|m.u[0][1]|m.u[1][0]|m.u[1][1])>>63),SBN3_FATAL_MATH,"HGCD head coefficient bound");
    // Two vector groups amortize coefficient setup; the final group retains
    // the exact scalar endpoint stitch. No u64/u52 representation change.
    if(n>=16)gcd_small::ifma_matrix<true>(a,b,a,b,n,m);
    else gcd_small::scalar_matrix(a,b,a,b,n,m);
}
void apply_head(uint64_t *a,uint64_t *b,size_t n,const gcd_small::Matrix &m) {
    require(std::max({m.u[0][0],m.u[0][1],m.u[1][0],m.u[1][1]})<(uint64_t(1)<<63),
            SBN3_FATAL_MATH,"HGCD head coefficient bound");
    if(n>=gcd_small::vector_from)gcd_small::ifma_matrix(a,b,a,b,n,m);
    else gcd_small::scalar_matrix(a,b,a,b,n,m);
}
}
namespace sbn::v3::gcd {
size_t finish(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,void *scratch) {
    return gcd_small::compute(out,a,an,b,bn,scratch);
}
}
