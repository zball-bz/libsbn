#pragma once
#ifdef CR_FLAT_PROFILE
#include <time.h>
#define FLAT_STAMP(n) do{if(c.phase){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);stamp[n]=uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}}while(0)
#else
#define FLAT_STAMP(n) do{}while(0)
#endif
/* Flat in-cache recipe, over the same x^8 coefficient vectors as the
 * Bailey kernels. Only completed integer-coefficient vectors are packed. */
namespace sbn::v3::SBN3_P48_NS {
inline bool flat_packa_active(const Plan &g){
    if constexpr(CR_FLAT_PACKA==0)return false;
    if constexpr(CR_FLAT_PACKA==1)return true;
    return (NP==8 && g.M2>=32768) || (NP==10 && g.M2>=16384);
}
struct FlatConstants {Dec a{},b{};};
struct FlatCtx {
    const Plan *pl=nullptr;const Primes *primes=nullptr;const FlatConstants *constants=nullptr;
    const uint64_t *a[2]{},*b[2]{};size_t an[2]{},bn[2]{};
    const V *cached[2][NP]{};
    bool packed_cache[2]{},packed_output=false;
    uint8_t *output[NP]{};
    const uint64_t *k[2]{},*rec[2]{};
    PassCounts *counts=nullptr;
#ifdef CR_FLAT_PROFILE
    uint64_t (*phase)[8]=nullptr;
#endif
};
inline size_t flat_input_vectors(size_t limbs,int T){return ((limbs*64+T-1)/T+7)/8;}
inline void flat_decode(V *dst,const uint64_t *src,size_t limbs,const Dec &d,const Plan &g,unsigned q,bool reversed=false){
    const size_t reads=g.T>192?32:g.T>128?24:16,nv=flat_input_vectors(limbs,g.T);
    const V reverse=_mm512_setr_epi64(7,6,5,4,3,2,1,0);
    for(size_t v=0;v<nv;++v){const size_t sv=reversed?nv-1-v:v,offset=(sv*g.T)/8;alignas(64) uint64_t padded[40];const uint64_t *in;
        if(offset+reads<=limbs)in=src+offset;
        else {memset(padded,0,sizeof padded);if(offset<limbs)memcpy(padded,src+offset,(limbs-offset)*8);in=padded;}
        const V value=dec1q(d,in,q,int((sv*g.T%8)/4));dst[v]=reversed?_mm512_permutexvar_epi64(reverse,value):value;
    }
    const size_t e=g.M2/8;
    // Mode 2 preserves the streaming write warmup for every block that the
    // top forward pass will overwrite, but never touches dead output blocks.
    const size_t used=CR_FLAT_TRIM_ZERO==2 && g.lbv>nv?g.lbv:nv;
    const size_t zero_end=CR_FLAT_TRIM_ZERO && !reversed && !g.full && g.M2>CR_BLK_MIN?((used+e-1)/e)*e:g.M2;
    for(size_t v=nv;v<zero_end;++v)dst[v]=_mm512_setzero_si512();
}
inline void flat_forward(V *x,const Plan &g,size_t live,const Prime &p,const PrimeV &pv){
    forward_prefix(x,g.M2,0,g.lbv,live,p,pv);
}
inline void flat_inverse(V *x,const Plan &g,const Prime &p,const PrimeV &pv){
    if constexpr(!CR_FLAT_TRIM_ZERO)for(size_t v=g.lbv;v<g.M2;++v)x[v]=_mm512_setzero_si512();
    if(g.full)full<true,true,1>((uint64_t *)x,g.M2,0,p,pv,nullptr);
    else if constexpr(CR_FLAT_INVERSE==1)itft<0,1,1,true>(x,g.M2,0,g.lbv,p,pv);
    else itft_blk<true>(x,g.M2,0,g.lbv,p,pv);
}
inline void flat_pad_frequency(V *x,const Plan &g){for(size_t v=g.lbv;v<g.lbw;++v)x[v]=_mm512_setzero_si512();}
inline void flat_prepare_fn(void *arg,uint64_t lo,uint64_t hi,int w,Frame *space){
    auto &c=*static_cast<FlatCtx *>(arg);const auto &g=*c.pl;auto *x=space->alloc<V>(g.M2+16);
    for(unsigned q=lo;q<hi;++q){flat_decode(x,c.a[0],c.an[0],c.constants->a,g,q);flat_forward(x,g,flat_input_vectors(c.an[0],g.T),c.primes->P[q],c.primes->V[q]);flat_pad_frequency(x,g);
        if(c.packed_output){const auto pack=pk52_mk();for(size_t v=0;v<g.lbw;++v)stslot<48>(c.output[q]+48*v,x[v],pack,c.primes->V[q]);}
        else memcpy(c.output[q],x,g.lbw*64);if(c.counts)++c.counts[w].row_forward;}
}
// Private two-product cache. Public flat spectra remain lazy64.
inline void flat_program_prepare48(void *arg,uint64_t lo,uint64_t hi,int w,Frame *space){
    auto &c=*static_cast<FlatCtx *>(arg);const auto &g=*c.pl;auto *x=space->alloc<V>(g.M2+16);const auto pack=pk52_mk();
    for(unsigned q=lo;q<hi;++q){
        flat_decode(x,c.a[0],c.an[0],c.constants->a,g,q);
        flat_forward(x,g,flat_input_vectors(c.an[0],g.T),c.primes->P[q],c.primes->V[q]);flat_pad_frequency(x,g);
        for(size_t v=0;v<g.lbw;++v)stslot<48>(c.output[q]+v*48,x[v],pack,c.primes->V[q]);
        if(c.counts)++c.counts[w].row_forward;
    }
}
inline void flat_program_apply48(void *arg,uint64_t lo,uint64_t hi,int w,Frame *space){
    auto &c=*static_cast<FlatCtx *>(arg);const auto &g=*c.pl;auto *y=space->alloc<V>(g.M2+16);const auto pack=pk52_mk();
    for(unsigned q=lo;q<hi;++q){
        const auto &p=c.primes->P[q];const auto &pv=c.primes->V[q];
        // cached[] is an erased-address carrier here, selected only by the
        // private program_cache argument; it is never dereferenced as V[].
        const auto *saved=reinterpret_cast<const uint8_t *>(c.cached[0][q]);
        flat_decode(y,c.b[0],c.bn[0],c.constants->b,g,q);
        flat_forward(y,g,flat_input_vectors(c.bn[0],g.T),p,pv);flat_pad_frequency(y,g);
        for(size_t v=0;v<g.lbw;v+=8){
            V a[8];for(unsigned j=0;j<8;++j)a[j]=ldslot<48>(saved+(v+j)*48,pack);
            conv8x8<false,true>(y+v,a,tower8<false>(p,v),pv);
        }
        flat_inverse(y,g,p,pv);flat_pad_frequency(y,g);
        for(size_t v=0;v<g.lbw;++v)st52(c.output[q]+v*SLOT,y[v],pack,pv);
        if(c.counts){++c.counts[w].row_forward;++c.counts[w].row_inverse;c.counts[w].leaf_products+=g.lbw/8;}
    }
}
// Kind agrees with the public recipe enum: mul, square, middle, two-term MAC.
template<bool Scaled>
void flat_product_packed_a(FlatCtx &c,uint64_t lo,uint64_t hi,int worker,Frame *space){
    const auto &g=*c.pl;V *y=space->alloc<V>(g.M2+16);const Pk52 pack=pk52_mk();
    for(unsigned q=lo;q<hi;++q){
        const auto &p=c.primes->P[q];const auto &pv=c.primes->V[q];auto *saved=c.output[q];
#ifdef CR_FLAT_PROFILE
        uint64_t stamp[9]{};
#endif
        FLAT_STAMP(0);flat_decode(y,c.a[0],c.an[0],c.constants->a,g,q);FLAT_STAMP(1);flat_forward(y,g,flat_input_vectors(c.an[0],g.T),p,pv);flat_pad_frequency(y,g);FLAT_STAMP(2);
        for(size_t v=0;v<g.lbw;++v)st52(saved+v*SLOT,y[v],pack,pv);
        FLAT_STAMP(3);flat_decode(y,c.b[0],c.bn[0],c.constants->b,g,q);FLAT_STAMP(4);flat_forward(y,g,flat_input_vectors(c.bn[0],g.T),p,pv);flat_pad_frequency(y,g);FLAT_STAMP(5);
        for(size_t v=0;v<g.lbw;v+=8){V a[8];for(unsigned j=0;j<8;++j)a[j]=ld52(saved+(v+j)*SLOT,pack);
            conv8x8<Scaled,true>(y+v,a,tower8<false>(p,v),pv,Scaled?vset(c.k[0][q]):V{},Scaled?vset(c.rec[0][q]):V{});}
        FLAT_STAMP(6);flat_inverse(y,g,p,pv);for(size_t v=g.lbv;v<g.lbw;++v)y[v]=_mm512_setzero_si512();FLAT_STAMP(7);
        for(size_t v=0;v<g.lbw;++v)st52(saved+v*SLOT,y[v],pack,pv);
        FLAT_STAMP(8);
#ifdef CR_FLAT_PROFILE
        if(c.phase)for(unsigned k=0;k<8;++k)c.phase[q][k]=stamp[k+1]-stamp[k];
#endif
        if(c.counts){c.counts[worker].row_forward+=2;++c.counts[worker].row_inverse;c.counts[worker].leaf_products+=g.lbw/8;}
    }
}
template<unsigned Kind,bool Scaled>
void flat_product_fn(void *arg,uint64_t lo,uint64_t hi,int w,Frame *space){
    auto &c=*static_cast<FlatCtx *>(arg);const auto &g=*c.pl;const bool cached0=c.cached[0][0]!=nullptr;
    if constexpr(CR_FLAT_PACKA && Kind==0)if(!cached0 && flat_packa_active(g)){flat_product_packed_a<Scaled>(c,lo,hi,w,space);return;}
    V *a=cached0?nullptr:space->alloc<V>(g.M2+16);
    V *y=(Kind==1 && !cached0)?a:space->alloc<V>(g.M2+16);
    V *a1=nullptr,*y1=nullptr;
    if constexpr(Kind==3){a1=c.cached[1][0]?nullptr:space->alloc<V>(g.M2+16);y1=space->alloc<V>(g.M2+16);}
    const Pk52 pack=pk52_mk();
    for(unsigned q=lo;q<hi;++q){const auto &p=c.primes->P[q];const auto &pv=c.primes->V[q];
#ifdef CR_FLAT_PROFILE
        uint64_t stamp[9]{};
#endif
        FLAT_STAMP(0);
        if(a){flat_decode(a,c.a[0],c.an[0],c.constants->a,g,q);FLAT_STAMP(1);flat_forward(a,g,flat_input_vectors(c.an[0],g.T),p,pv);flat_pad_frequency(a,g);if(c.counts)++c.counts[w].row_forward;}
        FLAT_STAMP(2);FLAT_STAMP(3);
        const V *ap=cached0?c.cached[0][q]:a;
        if constexpr(Kind==1){if(cached0){for(size_t v=g.lbw;v<g.M2;++v)y[v]=_mm512_setzero_si512();}}
        else {
            flat_decode(y,c.b[0],c.bn[0],c.constants->b,g,q,Kind==2);
            FLAT_STAMP(4);
            if constexpr(Kind==2){if(g.mrow_unnorm)full<false,true,1>((uint64_t *)y,g.M2,0,p,pv,nullptr);else itft<1,1,1>(y,g.M2,0,g.lbv,p,pv,nullptr);if(c.counts)++c.counts[w].row_mid;}
            else {flat_forward(y,g,flat_input_vectors(c.bn[0],g.T),p,pv);if(c.counts)++c.counts[w].row_forward;}
            flat_pad_frequency(y,g);
        }
        const V *ap1=nullptr;
        if constexpr(Kind==3){
            if(a1){flat_decode(a1,c.a[1],c.an[1],c.constants->a,g,q);flat_forward(a1,g,flat_input_vectors(c.an[1],g.T),p,pv);flat_pad_frequency(a1,g);if(c.counts)++c.counts[w].row_forward;}
            ap1=c.cached[1][q]?c.cached[1][q]:a1;
            flat_decode(y1,c.b[1],c.bn[1],c.constants->b,g,q);flat_forward(y1,g,flat_input_vectors(c.bn[1],g.T),p,pv);flat_pad_frequency(y1,g);if(c.counts)++c.counts[w].row_forward;
        }
        const V kc=Scaled?vset(c.k[0][q]):V{},kr=Scaled?vset(c.rec[0][q]):V{};
        FLAT_STAMP(5);
        for(size_t v=0;v<g.lbw;v+=8){
            V decoded[8],decoded1[8];const V *block=c.packed_cache[0]?decoded:ap+v,
                *block1=c.packed_cache[1]?decoded1:ap1?ap1+v:nullptr;
            if(c.packed_cache[0]){const auto *bytes=reinterpret_cast<const uint8_t *>(ap);
                for(unsigned j=0;j<8;++j)decoded[j]=ldslot<48>(bytes+48*(v+j),pack);block=decoded;}
            if constexpr(Kind==3)if(c.packed_cache[1]){const auto *bytes=reinterpret_cast<const uint8_t *>(ap1);
                for(unsigned j=0;j<8;++j)decoded1[j]=ldslot<48>(bytes+48*(v+j),pack);block1=decoded1;}
            if constexpr(Kind==1)if(cached0)for(unsigned j=0;j<8;++j)y[v+j]=v+j<g.lbv?block[j]:_mm512_setzero_si512();
            const V roots=tower8<false>(p,v);
            if constexpr(Kind==2)conv8x8T<Scaled>(y+v,block,roots,tower8<true>(p,v),pv,kc,kr);
            else conv8x8<Scaled>(y+v,block,roots,pv,kc,kr);
            if constexpr(Kind==3){conv8x8<Scaled>(y1+v,block1,roots,pv,Scaled?vset(c.k[1][q]):V{},Scaled?vset(c.rec[1][q]):V{});
                for(unsigned j=0;j<8;++j)y[v+j]=_mm512_add_epi64(y[v+j],y1[v+j]);}
        }
        if(c.counts)c.counts[w].leaf_products+=(g.lbw/8)*(Kind==3?2:1);
        FLAT_STAMP(6);
        if constexpr(Kind==2){for(size_t v=g.lbv;v<g.M2;++v)y[v]=_mm512_setzero_si512();tft<1,1>(y,g.M2,0,g.lbv,p,pv,nullptr);if(c.counts)++c.counts[w].row_transpose_forward;}
        else {flat_inverse(y,g,p,pv);if(c.counts)++c.counts[w].row_inverse;}
        for(size_t v=g.lbv;v<g.lbw;++v)y[v]=_mm512_setzero_si512();
        FLAT_STAMP(7);
        for(size_t v=0;v<g.lbw;++v)st52(c.output[q]+v*SLOT,y[v],pack,pv);
        FLAT_STAMP(8);
#ifdef CR_FLAT_PROFILE
        if constexpr(Kind==0)if(c.phase&&!cached0)for(unsigned k=0;k<8;++k)c.phase[q][k]=stamp[k+1]-stamp[k];
#endif
    }
}
} // namespace
#undef FLAT_STAMP
