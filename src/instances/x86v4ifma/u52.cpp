#include "backend/u52/island.hpp"
#include "value/limbs.hpp"
#include <algorithm>
#define SCRATCH(s) ::sbn::v3::AssumedFrameMark SBN3_U52_CAT(mark_,__LINE__)(*(s))
#define SALLOC(s,T,n) (s)->alloc_assumed<T>(n)
#include "backend/u52/mulmid.hpp"
#include "backend/u52/mulmid_kara.hpp"
#undef INLINE
#undef canonize
#include "backend/u52/division_estimate.hpp"
#include "backend/u52/division_core.hpp"
#include "backend/u52/dc_division.hpp"
#undef INLINE
#undef canonize
#undef SCRATCH
#undef SALLOC
namespace sbn::v3::u52 {
size_t divide_scratch_bytes(size_t nn,size_t dn) noexcept {
    require(dn>=17 && dn<=(size_t(1)<<20) && nn<=(size_t(1)<<40),SBN3_FATAL_SIZE,"u52 division size");
    const size_t db=(64*dn+415)/416,nb=(64*nn+830)/416;
    // Normalized D/N/Q and the shared cross-product, each with conversion
    // padding. D&C frames allocate no other arrays. Every cross-product has
    // at most 8*db u52 digits in total: S(D)<=5D+64 (u52-workspace.md).
    return 64*(2*nb+2*db+16)+64*((40*db+64+7)/8)+256;
}
void divide(uint64_t *qp,uint64_t *rp,const uint64_t *np,size_t nn64,const uint64_t *dp,size_t dn64,Frame &space) noexcept {
    require(dn64>=17 && dn64<=(size_t(1)<<20) && nn64<=(size_t(1)<<40) && dp[dn64-1],SBN3_FATAL_ARGUMENT,"u52 division arguments");
    const size_t qw=nn64>=dn64?nn64-dn64+1:0;
    const uint64_t dbits=u64_bit_length(dp,dn64),nbits=u64_bit_length(np,nn64);
    if(nbits<dbits){
        if(qw)memset(qp,0,qw*8);
        const size_t copy=std::min(nn64,dn64);
        if(copy)memcpy(rp,np,copy*8);
        memset(rp+copy,0,(dn64-copy)*8);return;
    }
    FrameMark mark(space);
    const size_t dn=(dbits+415)/416,shift=416*dn-dbits,nn=(nbits+shift+415)/416,qn=nn-dn;
    auto zero=[&](size_t count){auto *p=space.alloc<sb_vec>(count);memset(p,0,count*64);return p;};
    // Fused conversion may store one full spill vector after its shifted
    // last group, so two padding vectors are required, not just one.
    auto *d=zero(dn+2),*n=zero(nn+2),*q=zero(qn+3),*product=zero(dn+1);
    u52_from_u64_lsh(d,dp,dn64,shift);u52_from_u64_lsh(n,np,nn64,shift);
    alignas(64) sb_limb inverse[24];div2b_recip(inverse,dp,dn64);
    int high;
    if(!qn){high=block_cmp((sb_limb*)n,(const sb_limb*)d,dn)>=0;if(high)block_sub_n((sb_limb*)n,(const sb_limb*)d,dn);}
    else if(dn<dc_leaf_blocks)high=div2b_core((sb_limb*)q,(sb_limb*)n,nn,(const sb_limb*)d,dn,inverse,3);
    else high=blk_dcpi1_div_qr((sb_limb*)q,(sb_limb*)n,nn,(const sb_limb*)d,dn,inverse,(sb_limb*)product,space);
    // The old wrapper added an all-zero dividend block to make this digit
    // implicit. Keep it explicitly and avoid the extra full-divisor step.
    require(high==0 || high==1,SBN3_FATAL_MATH,"u52 division high quotient");
    ((sb_limb*)q)[8*qn]=uint64_t(high);
    size_t top=8*qn+1;while(top&&!((sb_limb*)q)[top-1])--top;
    require(!top || 52*(top-1)+64-unsigned(__builtin_clzll(((sb_limb*)q)[top-1]))<=64*qw,SBN3_FATAL_MATH,"u52 division quotient capacity");
    if(qw)u64_from_u52_canon(qp,q,qw);
    memset((sb_limb*)n+8*dn,0,64);
    u52_rshift((sb_limb*)n,(const sb_limb*)n,dn,shift);
    u64_from_u52_canon(rp,n,dn64);
}
bool root_supported(Algorithm root,size_t x,size_t y) noexcept {
    if(x<y){auto t=x;x=y;y=t;}
    if(root==Algorithm::automatic)return true;
    if(!y)return root==Algorithm::zero;
    switch(root){
        case Algorithm::basecase:return y<=112;
        case Algorithm::karatsuba:return y>x/2;
        case Algorithm::toom32:return u52_toom32_shape_ok(x,y);
        case Algorithm::toom33:return u52_toom33_shape_ok(x,y);
        case Algorithm::toom42:return u52_toom42_shape_ok(x,y);
        case Algorithm::stripmine:return x>=3*y && u52_toom42_shape_ok(2*y,y);
        default:return false;
    }
}
size_t scratch_bytes(size_t an,size_t bn,Algorithm root) noexcept {
    const bool streaming=root==Algorithm::automatic && streams(an,bn);
    require(an<=(size_t(1)<<31) && bn<=(size_t(1)<<31) &&
            (streaming || (an<=(1u<<20) && bn<=(1u<<20))),SBN3_FATAL_SIZE,"u52 input bound");
    if(!an || !bn)return 0;
    size_t extra=0;
    if(streaming){
        bn=an<bn?an:bn;an=strip_limbs;
        extra=(bn*8+63)&~size_t(63);
    }
    const size_t a=(an*64+51)/52,b=(bn*64+51)/52,d=a+b;
    const size_t va=(a+7)/8+1,vb=(b+7)/8+1;
    // Conversion owns A, B and their product: 2*(va+vb) vectors.
    // Recursive envelope S(D)<=5D+64 digits; see u52-workspace.md.
    const bool leaf=root==Algorithm::basecase || (root==Algorithm::automatic && (a<b?a:b)<MUL_U52_T22_THRESHOLD);
    const size_t recursive=leaf?0:((5*d+64+7)/8)*64;
    return extra+128*(va+vb)+recursive;
}

static Algorithm choose(size_t an,size_t bn) noexcept {
    if(an<bn){auto t=an;an=bn;bn=t;}
    if(!bn)return Algorithm::zero;
    if(bn<MUL_U52_T22_THRESHOLD)return Algorithm::basecase;
    if(bn<MUL_U52_T33_THRESHOLD){
        if(an>=3*bn)return Algorithm::stripmine;
        if(4*an>=5*bn && bn>=MUL_U52_T32_OK_THRESHOLD){
            if(4*an<7*bn){if(u52_toom32_shape_ok(an,bn))return Algorithm::toom32;}
            else if(bn>=MUL_U52_T42_OK_THRESHOLD && u52_toom42_shape_ok(an,bn))return Algorithm::toom42;
        }
        return Algorithm::karatsuba;
    }
    if(2*an>=5*bn)return Algorithm::stripmine;
    if(6*an<7*bn && u52_toom33_shape_ok(an,bn))return Algorithm::toom33;
    if(4*an<7*bn && u52_toom32_shape_ok(an,bn))return Algorithm::toom32;
    if(u52_toom42_shape_ok(an,bn))return Algorithm::toom42;
    return Algorithm::karatsuba;
}
template<bool track_root>
static inline Algorithm multiply_buffers(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,
                                        sb_vec *ax,sb_vec *bx,sb_vec *px,size_t va,size_t vb,Frame &space,Algorithm root) noexcept {
    if(!an || !bn){if(an+bn)memset(out,0,(an+bn)*8);return Algorithm::zero;}
    const size_t na=u52_from_u64(ax,a,an),nb=u52_from_u64(bx,b,bn);
    ax[va-1]=bx[vb-1]=sb_zero();
    if(!na || !nb){memset(out,0,(an+bn)*8);return Algorithm::zero;}
    size_t x=na,y=nb;if(x<y){auto *p=ax;ax=bx;bx=p;auto n=x;x=y;y=n;}
    int cls=0;
    if(root==Algorithm::automatic){if constexpr(track_root){root=choose(x,y);if(root==Algorithm::karatsuba && y<=x/2)root=Algorithm::basecase;}cls=mul_u52_dispatch(px,ax,bx,x,y,&space);}
    else switch(root){
        case Algorithm::basecase:require(y<=112,SBN3_FATAL_ARGUMENT,"forced u52 basecase lane bound");mul_u52_basecase(px,ax,bx,x,y);break;
        case Algorithm::karatsuba:require(y>x/2,SBN3_FATAL_ARGUMENT,"forced Karatsuba shape");cls=mul_u52_karatsuba(px,ax,bx,x,y,&space);break;
        case Algorithm::toom32:require(u52_toom32_shape_ok(x,y),SBN3_FATAL_ARGUMENT,"forced Toom32 shape");cls=mul_u52_toom32(px,ax,bx,x,y,&space);break;
        case Algorithm::toom33:require(u52_toom33_shape_ok(x,y),SBN3_FATAL_ARGUMENT,"forced Toom33 shape");cls=mul_u52_toom33(px,ax,bx,x,y,&space);break;
        case Algorithm::toom42:require(u52_toom42_shape_ok(x,y),SBN3_FATAL_ARGUMENT,"forced Toom42 shape");cls=mul_u52_toom42(px,ax,bx,x,y,&space);break;
        case Algorithm::stripmine:require(x>=3*y,SBN3_FATAL_ARGUMENT,"forced strip shape");cls=mul_u52_stripmine(px,ax,bx,x,y,&space);break;
        default:fatal(SBN3_FATAL_ARGUMENT,"u52 root algorithm");
    }
    const size_t need=8*((8*(an+bn)+51)/52);auto *digits=(uint64_t *)px;for(size_t j=na+nb;j<need;++j)digits[j]=0;
    if(cls)u64_from_u52_canonneg(out,px,an+bn);else u64_from_u52_canon(out,px,an+bn);
    return root;
}
Prepared prepare_buffers(Frame &space,size_t an,size_t bn,bool allow_streaming) noexcept {
    if(!an||!bn)return {};
    const bool streaming=allow_streaming && streams(an,bn);
    if(streaming){bn=an<bn?an:bn;an=strip_limbs;}
    const size_t va=u52_vec_count((an*16+12)/13)+1,vb=u52_vec_count((bn*16+12)/13)+1;
    auto *a=space.alloc<sb_vec>(va),*b=space.alloc<sb_vec>(vb),*p=space.alloc<sb_vec>(va+vb);
    auto *strip=streaming?space.alloc<uint64_t>(bn):nullptr;
    return {a,b,p,va,vb,strip,streaming?strip_limbs:0};
}
void multiply_prepared(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const Prepared &p,Frame &space) noexcept {
    if(p.strip_limbs){
        if(an<bn){auto *t=a;a=b;b=t;auto n=an;an=bn;bn=n;}
        for(size_t at=0;at<an;at+=p.strip_limbs){
            const size_t n=an-at<p.strip_limbs?an-at:p.strip_limbs;
            if(at)memcpy(p.strip_product,out+at,bn*8);
            (void)multiply_buffers<false>(out+at,a+at,n,b,bn,static_cast<sb_vec *>(p.a),
                                          static_cast<sb_vec *>(p.b),static_cast<sb_vec *>(p.product),
                                          p.a_vectors,p.b_vectors,space,Algorithm::automatic);
            // The preceding strips have written exactly [0,at+bn). Only
            // those bn overlap limbs are added; no whole-output clear or
            // reread of the long operand, and the carry stays in this strip.
            if(at)require(!limbs::add_to(out+at,n+bn,p.strip_product,bn),SBN3_FATAL_MATH,"u52 strip carry");
        }
        return;
    }
    (void)multiply_buffers<false>(out,a,an,b,bn,static_cast<sb_vec *>(p.a),static_cast<sb_vec *>(p.b),static_cast<sb_vec *>(p.product),p.a_vectors,p.b_vectors,space,Algorithm::automatic);
}
Algorithm multiply(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,Frame &space,Algorithm root) noexcept {
    FrameMark mark(space);const auto p=prepare_buffers(space,an,bn,root==Algorithm::automatic);
    if(p.strip_limbs){multiply_prepared(out,a,an,b,bn,p,space);return Algorithm::stripmine;}
    return multiply_buffers<true>(out,a,an,b,bn,static_cast<sb_vec *>(p.a),static_cast<sb_vec *>(p.b),static_cast<sb_vec *>(p.product),p.a_vectors,p.b_vectors,space,root);
}
}

namespace sbn::v3::u52 {
size_t middle_scratch_bytes(size_t an,size_t bn) noexcept {
    const size_t a=(an*64+51)/52,b=(bn*64+51)/52;
    // Conversion/result/padding <= 2*(a+b+64) words. Recursive maximum:
    // at most 7*m+80 words per halving level, m=a/2. The public TMP
    // wrapper limits a<=370; the guarded consumer permits a<=1024.
    return 16*(a+b+64)+8*(8*a+1024)+512;
}
template<unsigned Guards>
static void middle_impl(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,Frame &f) noexcept {
    FrameMark mark(f);const size_t na=(an*64+51)/52,nb=(bn*64+51)/52,rn=nb-na+1;
    if constexpr(Guards)require(na>Guards && na<=1024 && bn>=an && bn<=8192,SBN3_FATAL_ARGUMENT,"guarded middle shape");
    auto *A=f.alloc<uint64_t>(na+24),*B=f.alloc<uint64_t>(nb+56),*R=f.alloc<uint64_t>(rn+Guards+8);
    memset(A,0,(na+24)*8);memset(B,0,(nb+56)*8);memset(R,0,(rn+Guards+8)*8);
    // Fixed planned counts, including leading zero digits; no value-dependent
    // certificate or allocation. Front padding makes donor masked bases valid.
    A+=8;B+=8;u52_from_u64((sb_pvec)A,a,an);u52_from_u64((sb_pvec)B,b,bn);
    mulmid_dc(R+Guards,A,B,int64_t(na),int64_t(nb),&f);
    if constexpr(Guards){
        constexpr uint64_t mask=(uint64_t(1)<<52)-1;
        for(unsigned g=0;g<Guards;++g){
            const size_t diagonal=na-1-Guards+g;
            __m512i low=_mm512_setzero_si512(),high=low;
            const __m512i order=_mm512_setr_epi64(0,1,2,3,4,5,6,7);
            for(size_t i=0;i<=diagonal;i+=8){
                const unsigned count=unsigned(std::min<size_t>(8,diagonal-i+1));
                const __mmask8 lanes=__mmask8((1u<<count)-1);
                const __m512i x=_mm512_maskz_loadu_epi64(lanes,A+i);
                const __m512i raw=_mm512_maskz_loadu_epi64(lanes,B+diagonal-i+1-count);
                const __m512i indices=_mm512_sub_epi64(_mm512_set1_epi64(count-1),order);
                const __m512i y=_mm512_permutexvar_epi64(indices,raw);
                low=_mm512_madd52lo_epu64(low,x,y);high=_mm512_madd52hi_epu64(high,x,y);
            }
            const __uint128_t sum=(__uint128_t(uint64_t(_mm512_reduce_add_epi64(high)))<<52)+
                                  uint64_t(_mm512_reduce_add_epi64(low));
            R[g]+=uint64_t(sum)&mask;R[g+1]+=uint64_t(sum>>52);
        }
    }
    __int128 carry=0;uint64_t acc=0;unsigned bits=0;size_t w=0;
    for(size_t j=0;j<rn+Guards+2;++j){
        if(j<rn+Guards+1)carry+=(__int128)(int64_t)R[j];
        const uint64_t d=uint64_t(carry)&((uint64_t(1)<<52)-1);carry=(carry-d)>>52;
        acc|=d<<bits;
        if(bits+52>=64){out[w++]=acc;acc=bits?d>>(64-bits):0;bits=bits+52-64;}
        else bits+=52;
    }
    if(bits)out[w++]=acc;
}
void middle(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,Frame &f) noexcept {
    middle_impl<0>(out,a,an,b,bn,f);
}
size_t middle_guard_scratch_bytes(size_t an,size_t bn) noexcept {
    return middle_scratch_bytes(an,bn)+128;
}
void middle_guard(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,Frame &f) noexcept {
    middle_impl<4>(out,a,an,b,bn,f);
}
}
