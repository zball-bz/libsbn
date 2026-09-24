#pragma once
#include <new>
// Right-angle recipe for 16-bit odd-radix W1 plans. Experiment, gates and
// measurements: bench/probes/rac_2026-09-08, results/rac_2026-09-08,
// docs/rac-results-2026-09-08.md.
// ============================================================================
// Right-angle convolution (RAC) recipe (2026-09-08).
// Product in C[x]/(x^N - i): z[p] = x[p] + i*x[p+N] (digit p and digit N+p),
// evaluated at the roots of x^N = i, i.e. the e^{+} DFT of z[p]*w^{p/4}.
// CT split p = s*n + j, k' = b + h + M*k2 with h = (rM-1)/4, r = M mod 4:
//   Z[k'] = sum_j w_n^{j k2} T_b(j) sum_s w_M^{s b} i^{r s} z[s n + j],
//   T_b(j) = exp(2 pi i j (4b + rM) / (4N)).
// The pre-weight i^{rs} is an exact lane rotation, the twist lives in T_b.
// Inverse: conj(T_b), e^{-} M-DFT, then i^{-rs}. Scale 1/N stays pointwise.
// Emit: 32 consecutive positions of a stripe (four tiles) are staged, then
// their re digits form one 8-limb front step and their im digits another;
// 2M fronts, the stock 8-lane carry chain, full-width stores.
// ============================================================================
namespace sbn::v3::pq16 {
struct RacTables {
    const pq16_plan *core;Shape shape;const double *tw;unsigned r;
    sb_vec dec_idx[4]; // vector u <- 8 consecutive digits 8u..8u+7 of one raw zmm (32 digits)
};
// M branches x n positions of T_b(j), [re x8 | im x8] per tile; b = 0 carries the twist too.
static inline size_t rac_table_bytes(Shape s){return sizeof(RacTables)+256+(root_bank::rac_twiddle(s.radix,s.branch)?0:16*size_t(s.nfull));}
static inline RacTables *rac_prepare(Frame &f,const pq16_plan &core,Shape s){
    auto *p=new(f.allocate(sizeof(RacTables),128)) RacTables{};p->core=&core;p->shape=s;
    const unsigned M=s.radix,n=s.branch,N=s.nfull;p->r=M%4;
    p->tw=root_bank::rac_twiddle(M,n);
    if(!p->tw){
        auto *tw=static_cast<double *>(f.allocate(16*size_t(N),128));p->tw=tw;
        for(unsigned b=0;b<M;++b)for(unsigned j=0;j<n;++j){
            long double c,sn;ct_root_ratio(c,sn,uint64_t(4*b+p->r*M)*j,4*uint64_t(N));
            const size_t at=2*size_t(b)*n+2*(j&~7u)+(j&7u);tw[at]=(double)c;tw[at+8]=(double)sn;
        }
    }
    for(unsigned u=0;u<4;++u){char idx[64];for(int i=0;i<64;++i)idx[i]=(char)0x40;
        for(unsigned l=0;l<8;++l){const unsigned d=8*u+l;idx[4*l]=(char)(2*d);idx[4*l+1]=(char)(2*d+1);}
        memcpy(&p->dec_idx[u],idx,64);}
    return p;
}
static inline qcv rac_rot_i(qcv x,unsigned phase){ // x * i^phase, exact
    switch(phase&3){case 0:return x;case 1:return q_j(x);case 2:return {sb_sub(sb_dzero(),x.re),sb_sub(sb_dzero(),x.im)};default:return q_mj(x);}
}
// ---- emit primitives ---------------------------------------------------------
// 16 consecutive digits (two 8-lane double vectors, biased rounding) -> 8 u64 pairs (digit 2m + digit(2m+1)<<16)
static inline sb_vec rac_pairs16(sb_dvec a,sb_dvec b){
    constexpr uint64_t bits=0x4338000000000000ull;static_assert((bits<<16)==0);
    const sb_vec bias=sb_set1_64(bits);const sb_dvec db=sb_as_dvec(bias);
    const sb_vec IE=sb_setr_64(0,2,4,6,8,10,12,14),IO=sb_setr_64(1,3,5,7,9,11,13,15);
    const sb_vec da=sb_as_ivec(sb_add(a,db)),dbv=sb_as_ivec(sb_add(b,db));
    return sb_sub(sb_add(sb__fn(permutex2var_epi64)(da,IE,dbv),sb_slli(sb__fn(permutex2var_epi64)(da,IO,dbv),16)),bias);
}
// (re pairs, im pairs) -> lanes 0-3: 4 re limbs, lanes 4-7: 4 im limbs; hi = quad carries
// two independent 4-lane carry chains in one vector (lanes 0-3 / 4-7); cin bit0 = low chain, bit1 = high chain
// one emit step: two consecutive output tiles (positions p, p+8) of a stripe -> re front (digits p..p+15) + im front (N+p..)
// ---- pointwise fused into the inverse leaves ----------------------------------
static inline void rac_ileaf64_run(double *d,const double *o,uint32_t span,sb_dvec sc,const pq16_plan *pl){
    qcv lw[4];q_l64const(lw,pl);
    for(uint32_t g=0;g<span/64u;++g){double *p=d+128u*(size_t)g;const double *y=o+128u*(size_t)g;qcv v[8];
        for(int i=0;i<8;++i){v[i]=q_mul(q_ld(p+16*i),q_ld(y+16*i));v[i].re=sb_mul(v[i].re,sc);v[i].im=sb_mul(v[i].im,sc);}
        q_ibody64(v,lw);for(int i=0;i<8;++i)q_st(p+16*i,v[i]);}
}
static inline void rac_ileaf32x2_one(double *p,const double *y,qcv W1,qcv W2,sb_dvec sc){
    qcv v[8];for(int i=0;i<8;++i){v[i]=q_mul(q_ld(p+16*i),q_ld(y+16*i));v[i].re=sb_mul(v[i].re,sc);v[i].im=sb_mul(v[i].im,sc);}
    q_idft8(v);q_tr8cv(v);q_ibf4(v,W1,W2);q_ibf4(v+4,W1,W2);for(int i=0;i<8;++i)q_st(p+16*i,v[i]);
}
static inline void rac_ileaf32x2_run(double *d,const double *o,uint32_t span,sb_dvec sc,const pq16_plan *pl){
    qcv W1=q_ld(pl->l32w),W2=q_ld(pl->l32w+16);
    for(uint32_t g=0;g<span/64u;++g)rac_ileaf32x2_one(d+128u*(size_t)g,o+128u*(size_t)g,W1,W2,sc);
}
static inline void rac_ileaf128_run(double *d,const double *o,uint32_t span,sb_dvec sc,const pq16_plan *pl){
    qcv W1_32=q_ld(pl->l32w),W2_32=q_ld(pl->l32w+16);
    for(uint32_t g=0;g<span/128u;++g){double *p=d+256u*(size_t)g;const double *y=o+256u*(size_t)g;
        rac_ileaf32x2_one(p,y,W1_32,W2_32,sc);rac_ileaf32x2_one(p+128,y+128,W1_32,W2_32,sc);
        for(int t=0;t<4;++t){qcv v[4]={q_ld(p+16*t),q_ld(p+16*(t+4)),q_ld(p+16*(t+8)),q_ld(p+16*(t+12))};
            qcv W1=q_ld(pl->l128w+32*t),W2=q_ld(pl->l128w+32*t+16);q_ibf4(v,W1,W2);
            q_st(p+16*t,v[0]);q_st(p+16*(t+4),v[1]);q_st(p+16*(t+8),v[2]);q_st(p+16*(t+12),v[3]);}}
}
static inline void rac_pointwise_branch(double *A,const double *B,unsigned n,double sc_d,unsigned leaf,const pq16_plan *pl){
    const sb_dvec sc=sb_set1_d(sc_d);
    if(leaf==32u){rac_ileaf32x2_run(A,B,n,sc,pl);return;}
    // 64/128-point leaves spill when the product rides on their loads: multiply per block, then the stock leaf
    const unsigned block=leaf==128u?128u:64u;
    for(unsigned k=0;k<n;k+=block){
        for(unsigned p=0;p<block;p+=8){double *x=A+2*size_t(k+p);const double *y=B+2*size_t(k+p);qcv z=q_mul(q_ld(x),q_ld(y));z.re=sb_mul(z.re,sc);z.im=sb_mul(z.im,sc);q_st(x,z);}
        pq16_ileaf_run(A+2*size_t(k),block,leaf,pl);
    }
}
template<unsigned M>static inline void rac_pointwise(double *A,const double *B,const RacTables &t){
    const unsigned n=t.shape.branch,N=t.shape.nfull,leaf=pq16_odd_leaf(n);
    for(unsigned b=0;b<M;++b)rac_pointwise_branch(A+2*size_t(b)*n,B+2*size_t(b)*n,n,1./N,leaf,t.core);
}
// ---- odd M (CT) ----------------------------------------------------------------
template<unsigned M>static inline void rac_forward(double *data,const uint64_t *a,size_t count,const RacTables &t){
    const unsigned n=t.shape.branch,N=t.shape.nfull;const auto *pl=t.core;
    for(unsigned j=0;j<n;j+=32){
        sb_vec rre[M],rim[M];bool has_im=false;
        for(unsigned s=0;s<M;++s){const size_t pre=(size_t(s)*n+j)/4,pim=(size_t(N)+size_t(s)*n+j)/4;
            rre[s]=pre<count?q_raw8(a+pre,int64_t(count-pre)):sb_zero();
            if(pim<count){rim[s]=q_raw8(a+pim,int64_t(count-pim));has_im=true;}else rim[s]=sb_zero();}
        for(unsigned u=0;u<4;++u){qcv x[M],y[M];
            for(unsigned s=0;s<M;++s){x[s].re=q_dec1(rre[s],t.dec_idx[u]);x[s].im=has_im?q_dec1(rim[s],t.dec_idx[u]):sb_dzero();x[s]=rac_rot_i(x[s],t.r*s);}
            q_pfa_bfly(x,y,M,1);
            for(unsigned b=0;b<M;++b)q_st(data+2*size_t(b)*n+2*j+16*u,q_mul(y[b],q_ld(t.tw+2*size_t(b)*n+2*j+16*u)));
        }
    }
    for(unsigned b=0;b<M;++b)pq16_odd_fwd(data+2*size_t(b)*n,n,pl);
}
// 4 vectors of 8 consecutive biased-rounded digits -> 8 limbs (lo/hi), same bound as q_pack2<true>
static inline q_lohi rac_pack32(const sb_dvec x[4]){
    const sb_vec p01=rac_pairs16(x[0],x[1]),p23=rac_pairs16(x[2],x[3]);
    const sb_vec IE=sb_setr_64(0,2,4,6,8,10,12,14),IO=sb_setr_64(1,3,5,7,9,11,13,15);
    const sb_vec elo=sb__fn(permutex2var_epi64)(p01,IE,p23),olo=sb__fn(permutex2var_epi64)(p01,IO,p23);
    q_lohi q;q.lo=sb_add(elo,sb_slli(olo,32));q.hi=sb_add(sb_srli(olo,32),sb_maskz(sb_ltu(q.lo,elo),sb_set1_64(1)));return q;
}
static inline void rac_front_step(uint64_t *r,size_t size,size_t off,const sb_dvec x[4],q_chain &ch){
#ifdef RAC_COEFFICIENT_OBSERVER
    for(unsigned u=0;u<4;++u){alignas(64) double tmp[8];store_dvec(tmp,x[u]);for(unsigned l=0;l<8;++l)RAC_COEFFICIENT_OBSERVER(4*off+8*u+l,tmp[l]);}
#endif
    const auto out=q_chain_step(&ch,rac_pack32(x));if(off<size)q_st_tail(r+off,out,int64_t(size-off));
}
template<unsigned M>static inline bool rac_emit(uint64_t *r,size_t size,double *data,const RacTables &t){
    const unsigned n=t.shape.branch,N=t.shape.nfull;q_chain ch[2*M]{};
    for(unsigned j=0;j<n;j+=32){
        alignas(64) double stage[M][4][16];
        for(unsigned u=0;u<4;++u){qcv x[M],y[M];
            for(unsigned b=0;b<M;++b)x[b]=q_mulc(q_ld(data+2*size_t(b)*n+2*j+16*u),q_ld(t.tw+2*size_t(b)*n+2*j+16*u));
            q_pfa_bfly(x,y,M,0);
            for(unsigned s=0;s<M;++s)q_st(stage[s][u],rac_rot_i(y[s],4-((t.r*s)&3)));
        }
        for(unsigned s=0;s<M;++s){sb_dvec re[4],im[4];
            for(unsigned u=0;u<4;++u){re[u]=load_dvec(stage[s][u]);im[u]=load_dvec(stage[s][u]+8);}
            rac_front_step(r,size,(size_t(s)*n+j)/4,re,ch[s]);
            rac_front_step(r,size,(size_t(N)+size_t(s)*n+j)/4,im,ch[M+s]);
        }
    }
    return q_chains_close(r,size,ch,2*M,n/4);
}
template<unsigned M>static inline bool rac_multiply(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const RacTables &t,Frame &f,const double *cached=nullptr,bool cached_square=false){
    static_assert(M==3||M==5||M==7,"right-angle recipe: odd radix only (pow2 keeps the fused PQ pipeline)");
    const bool square=cached?cached_square:a==b && an==bn;
    const unsigned n=t.shape.branch,N=t.shape.nfull;FrameMark mark(f);double *A=f.alloc<double>((cached||square?2:4)*size_t(N)+48),*B=cached?const_cast<double *>(cached):square?A:A+2*size_t(N)+32;
    if(cached){if(square)memcpy(A,cached,2*size_t(N)*8);else rac_forward<M>(A,b,bn,t);}
    else{if(!square)rac_forward<M>(B,b,bn,t);rac_forward<M>(A,a,an,t);}rac_pointwise<M>(A,B,t);
    for(unsigned s=0;s<M;++s)pq16_odd_inv(A+2*size_t(s)*n,n,t.core);
    return rac_emit<M>(r,an+bn,A,t);
}
// Canonical integer source modulo 2^(32*N)+1. The extra-word endpoint is
// exactly -1, so its evaluations are constant -1 (not a discarded digit).
template<unsigned M>static inline void rac_plus_forward(double *out,const uint64_t *a,size_t count,const RacTables &t){
    const size_t ring=t.shape.nfull/2;
    if(count==ring+1){
        require(a[ring]<=1,SBN3_FATAL_ARGUMENT,"plus ring canonical high word");
        if(a[ring]){uint64_t any=0;for(size_t j=0;j<ring;++j)any|=a[j];require(!any,SBN3_FATAL_ARGUMENT,"plus ring endpoint");
            for(size_t j=0;j<t.shape.nfull;j+=8)q_st(out+2*j,{sb_set1_d(-1.),sb_dzero()});return;}
        --count;
    }
    rac_forward<M>(out,a,count,t);
}
template<unsigned M>static inline void rac_plus_emit(uint64_t *out,int64_t *digits,double *data,const RacTables &t){
    const size_t n=t.shape.branch,N=t.shape.nfull,ring=N/2;
    for(size_t j=0;j<n;j+=8){qcv x[M],y[M];
        for(unsigned b=0;b<M;++b)x[b]=q_mulc(q_ld(data+2*size_t(b)*n+2*j),q_ld(t.tw+2*size_t(b)*n+2*j));
        q_pfa_bfly(x,y,M,0);
        for(unsigned s=0;s<M;++s){const auto z=rac_rot_i(y[s],4-((t.r*s)&3));const size_t k=s*n+j;
#ifdef RAC_COEFFICIENT_OBSERVER
            alignas(64) double re[8],im[8];store_dvec(re,z.re);store_dvec(im,z.im);
            for(unsigned l=0;l<8;++l){RAC_COEFFICIENT_OBSERVER(k+l,re[l]);RAC_COEFFICIENT_OBSERVER(N+k+l,im[l]);}
#endif
            _mm512_storeu_si512(digits+k,_mm512_cvt_roundpd_epi64(z.re,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
            _mm512_storeu_si512(digits+N+k,_mm512_cvt_roundpd_epi64(z.im,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
        }
    }
    // Normalize the signed negacyclic coefficients. Final carry C has weight
    // 2^(64*ring) == -1, so the remaining reduction is L-C.
    __int128 carry=0;uint64_t word=0;
    for(size_t j=0;j<2*N;++j){carry+=digits[j];const uint64_t d=uint64_t(carry)&65535;carry=(carry-d)>>16;
        word|=d<<(16*(j%4));if(j%4==3){out[j/4]=word;word=0;}}
    require(carry>-(int64_t(1)<<48)&&carry<(int64_t(1)<<48),SBN3_FATAL_MATH,"plus ring carry bound");out[ring]=0;
    if(carry>=0){uint64_t by=uint64_t(carry);for(size_t j=0;j<ring&&by;++j){const uint64_t old=out[j];out[j]-=by;by=old<by;}
        if(by){uint64_t c=1;for(size_t j=0;j<ring&&c;++j)c=++out[j]==0;out[ring]=c;}}
    else{uint64_t c=uint64_t(-carry);for(size_t j=0;j<ring&&c;++j){const __uint128_t v=__uint128_t(out[j])+c;out[j]=uint64_t(v);c=uint64_t(v>>64);}out[ring]=c;}
    if(out[ring]){uint64_t any=0;for(size_t j=0;j<ring;++j)any|=out[j];
        if(any){out[ring]=0;for(size_t j=0;j<ring;++j)if(out[j]--)break;}}
}
template<unsigned M>static inline void rac_plus_multiply(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,bool square,const double *cached,const RacTables &t,Frame &f){
    FrameMark mark(f);const size_t N=t.shape.nfull,n=t.shape.branch;
    auto *A=f.alloc<double>(2*N),*B=A;const double *other=cached;
    if(!cached&&!square)B=f.alloc<double>(2*N);
    if(cached){if(square)memcpy(A,cached,16*N);else rac_plus_forward<M>(A,b,bn,t);}
    else{if(!square)rac_plus_forward<M>(B,b,bn,t);rac_plus_forward<M>(A,a,an,t);other=B;}
    rac_pointwise<M>(A,other,t);for(unsigned s=0;s<M;++s)pq16_odd_inv(A+2*size_t(s)*n,unsigned(n),t.core);
    auto *digits=f.alloc<int64_t>(2*N);rac_plus_emit<M>(out,digits,A,t);
}

}
