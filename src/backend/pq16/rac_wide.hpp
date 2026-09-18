#pragma once
// Right-angle recipe for the 17..20-bit codec (odd radix). Included after
// variable.hpp (uses vx_emit64 / vx_fronts) and rac.hpp (RacTables, rac_pointwise).
// Envelope: results/rac_2026-09-08/gate-wide.jsonl, docs/rac-results-2026-09-08.md.
// ============================================================================
// Right-angle recipe for the 17..20-bit codec (odd radix). Digit p of the
// operand is position p (re), digit N+p is position p (im). The transform,
// cross table (RacTables) and pointwise are the 16-bit RAC ones; only the
// decode (8 consecutive digits per vector) and the emit (adjacent-lane pairs
// feed the existing 2^(2B)-slot packer) differ. S selects the locally
// balanced digits (variable.hpp): the carry t_{k-1} comes from the previous
// lane, lane 0 from the digit before the vector.
// ============================================================================
namespace sbn::v3::pq16 {
template<unsigned B,unsigned U,unsigned Half>static inline __m256i vx_rgather(sb_vec raw){
    alignas(64) static constexpr auto index=[] {struct X {unsigned char v[64]{};} x{};
        for(unsigned k=0;k<8;++k)for(unsigned q=0;q<4;++q)x.v[4*k+q]=((16*U*B)%64+(8*Half+k)*B)/8+q;return x;}();
    return _mm512_castsi512_si256(_mm512_permutexvar_epi8(_mm512_load_si512(&index),raw));
}
// 8 consecutive digits 16U+8*Half .. +7 of the 64-digit block whose raw window starts at limb (16*U*B)/64
template<unsigned B,unsigned U,unsigned Half>static inline sb_dvec vx_rdec(sb_vec raw){
    alignas(32) static constexpr auto shift=[] {struct X {unsigned v[8]{};} x{};
        for(unsigned k=0;k<8;++k)x.v[k]=((8*Half+k)*B)%8;return x;}();
    auto v=vx_rgather<B,U,Half>(raw);
    v=_mm256_and_si256(_mm256_srlv_epi32(v,_mm256_load_si256((const __m256i *)&shift)),_mm256_set1_epi32((1u<<B)-1));
    return _mm512_cvtepu32_pd(v);
}
template<unsigned B,unsigned U,unsigned Half>static inline sb_dvec vx_rdec_signed(sb_vec raw,uint32_t tprev){
    alignas(32) static constexpr auto shift=[] {struct X {unsigned v[8]{};} x{};
        for(unsigned k=0;k<8;++k)x.v[k]=32-B-((8*Half+k)*B)%8;return x;}();
    const auto e=_mm256_srai_epi32(_mm256_sllv_epi32(vx_rgather<B,U,Half>(raw),_mm256_load_si256((const __m256i *)&shift)),32-B);
    const auto t=_mm256_srli_epi32(e,31),prev=_mm256_insert_epi32(_mm256_setzero_si256(),int(tprev),7);
    return _mm512_cvtepi32_pd(_mm256_add_epi32(e,_mm256_alignr_epi32(t,prev,7)));
}
static inline uint32_t vx_tprev_bit(const uint64_t *a,size_t count,int64_t bit){ // operand bit `bit` (0 outside)
    if(bit<0)return 0;const size_t limb=size_t(bit)/64;return limb<count?uint32_t((a[limb]>>(bit%64))&1):0u;
}
// four consecutive re/im vectors (positions p0..p0+31, p0 % 32 == 0) of the digit stream starting at digit `base` (base % 64 == 0)
template<unsigned B,unsigned U0,bool S>static inline void vx_rdec4(sb_dvec out[4],const uint64_t *a,size_t count,size_t first){
    const size_t pos0=first+(16*U0*B)/64,pos1=first+(16*(U0+1)*B)/64;
    const auto raw0=pos0<count?q_raw8(a+pos0,int64_t(count-pos0)):sb_zero(),raw1=pos1<count?q_raw8(a+pos1,int64_t(count-pos1)):sb_zero();
    if constexpr(!S){out[0]=vx_rdec<B,U0,0>(raw0);out[1]=vx_rdec<B,U0,1>(raw0);out[2]=vx_rdec<B,U0+1,0>(raw1);out[3]=vx_rdec<B,U0+1,1>(raw1);}
    else{
        const int64_t bit=int64_t(64*first)+int64_t(16*U0*B);   // first digit of the window
        out[0]=vx_rdec_signed<B,U0,0>(raw0,vx_tprev_bit(a,count,bit-1));out[1]=vx_rdec_signed<B,U0,1>(raw0,vx_tprev_bit(a,count,bit+8*B-1));
        out[2]=vx_rdec_signed<B,U0+1,0>(raw1,vx_tprev_bit(a,count,bit+16*B-1));out[3]=vx_rdec_signed<B,U0+1,1>(raw1,vx_tprev_bit(a,count,bit+24*B-1));
    }
}
template<unsigned B,bool S>static inline void vx_rac_decode(sb_dvec out[4],const uint64_t *a,size_t count,size_t p0){
    const size_t first=(p0/64)*B;   // 64 digits = B limbs, block aligned
    // beyond the operand: zero digits; the signed codec's carry digit can start a block at limb == count
    if(first>count||(!S&&first==count)){for(unsigned u=0;u<4;++u)out[u]=sb_dzero();return;}
    if((p0>>5)&1)vx_rdec4<B,2,S>(out,a,count,first);else vx_rdec4<B,0,S>(out,a,count,first);
}
template<unsigned M,unsigned B,bool S=false>static inline void vx_rac_forward(double *data,const uint64_t *a,size_t count,const RacTables &t){
    const unsigned n=t.shape.branch,N=t.shape.nfull;const auto *pl=t.core;const size_t digits=(64*count)/B+(S?1:0); // whole digits available (+ carry digit)
    for(unsigned j=0;j<n;j+=32){
        sb_dvec re[M][4],im[M][4];bool has_im=false;
        for(unsigned s=0;s<M;++s){const size_t p=size_t(s)*n+j;vx_rac_decode<B,S>(re[s],a,count,p);
            if(size_t(N)+p<digits+64){vx_rac_decode<B,S>(im[s],a,count,size_t(N)+p);has_im=true;}else for(unsigned u=0;u<4;++u)im[s][u]=sb_dzero();}
        for(unsigned u=0;u<4;++u){qcv x[M],y[M];
            for(unsigned s=0;s<M;++s){x[s].re=re[s][u];x[s].im=has_im?im[s][u]:sb_dzero();x[s]=rac_rot_i(x[s],t.r*s);}
            q_pfa_bfly(x,y,M,1);
            for(unsigned b=0;b<M;++b)q_st(data+2*size_t(b)*n+2*j+16*u,q_mul(y[b],q_ld(t.tw+2*size_t(b)*n+2*j+16*u)));
        }
    }
    for(unsigned b=0;b<M;++b)pq16_odd_fwd(data+2*size_t(b)*n,n,pl);
}
// 64 consecutive digits held as the re (or im) lanes of eight staged tiles -> B limbs through vx_emit64
template<unsigned B>static inline void vx_rac_front64(const vx_target &t,size_t off,const double *stage,unsigned part,q_chain &ch,sb_vec rc){
    const sb_vec IE=sb_setr_64(0,2,4,6,8,10,12,14),IO=sb_setr_64(1,3,5,7,9,11,13,15);qcv v[4];
    for(unsigned q=0;q<4;++q){const sb_dvec a=load_dvec(stage+32*q+part),b=load_dvec(stage+32*q+16+part);
        v[q].re=sb__fn(permutex2var_pd)(a,IE,b);v[q].im=sb__fn(permutex2var_pd)(a,IO,b);}
    vx_emit64<B>(t,off,v,ch,rc);
}
template<unsigned M,unsigned B,bool S=false>static inline bool vx_rac_emit(uint64_t *r,size_t size,double *data,const RacTables &t,uint64_t *tail){
    const unsigned n=t.shape.branch,N=t.shape.nfull;q_chain ch[2*M]{};
    const size_t limbs_front=size_t(n)/64*B;vx_fronts<B,S> fr(r,size,tail,2*M,limbs_front,N);fr.init(ch[0]);
    for(unsigned j=0;j<n;j+=64){
        alignas(64) double stage[M][8][16];
        for(unsigned u=0;u<8;++u){qcv x[M],y[M];
            for(unsigned b=0;b<M;++b)x[b]=q_mulc(q_ld(data+2*size_t(b)*n+2*j+16*u),q_ld(t.tw+2*size_t(b)*n+2*j+16*u));
            q_pfa_bfly(x,y,M,0);
            for(unsigned s=0;s<M;++s)q_st(stage[s][u],rac_rot_i(y[s],4-((t.r*s)&3)));
        }
        for(unsigned s=0;s<M;++s){
            vx_rac_front64<B>(fr.target,((size_t(s)*n+j)/64)*B,&stage[s][0][0],0,ch[s],fr.rc(s));
            vx_rac_front64<B>(fr.target,((size_t(N)+size_t(s)*n+j)/64)*B,&stage[s][0][0],8,ch[M+s],fr.rc(M+s));
        }
    }
    return fr.close(ch,2*M,limbs_front);
}
static inline size_t vx_rac_tail_limbs(Shape s){return (size_t(s.branch)/64)*s.bits+s.bits+24;}
template<unsigned M,unsigned B,bool S=false>static inline bool vx_rac_multiply(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const RacTables &t,Frame &f,const double *cached=nullptr,bool cached_square=false){
    const bool square=cached?cached_square:a==b && an==bn;
    const unsigned n=t.shape.branch,N=t.shape.nfull;FrameMark mark(f);double *A=f.alloc<double>((cached||square?2:4)*size_t(N)+48),*D=cached?const_cast<double *>(cached):square?A:A+2*size_t(N)+32;
    uint64_t *tail=nullptr;if constexpr(S)tail=f.alloc<uint64_t>(vx_rac_tail_limbs(t.shape));
    if(cached){if(square)memcpy(A,cached,2*size_t(N)*8);else vx_rac_forward<M,B,S>(A,b,bn,t);}
    else{if(!square)vx_rac_forward<M,B,S>(D,b,bn,t);vx_rac_forward<M,B,S>(A,a,an,t);}rac_pointwise<M>(A,D,t);
    for(unsigned s=0;s<M;++s)pq16_odd_inv(A+2*size_t(s)*n,n,t.core);
    return vx_rac_emit<M,B,S>(r,an+bn,A,t,tail);
}
}
