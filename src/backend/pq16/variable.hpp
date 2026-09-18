#pragma once
// 17..20-bit codec. The transform is unchanged; 64 digits form B complete
// limbs. Two-digit carry slots have radix 2^(2B).
// S=false: unsigned digits d in [0,2^B).
// S=true (2026-09-08): locally balanced digits e_i = d_i - (t_i<<B) + t_{i-1},
// t_i = d_i>>(B-1), so |e_i| <= 2^(B-1); one carry digit t_{L-1} follows the
// operand. Coefficients are then signed; the emit adds K = m(2^B-1) to every
// coefficient of the fronts that reach the product (round constant), injects
// m into front 0 and expects the carry m at the end of the last such front
// (K sum over P slots = m(2^(BP)-1)). The slots beyond the stored product of
// that front are staged in a small tail so the fronts' carries can be
// resolved exactly and checked (vx_close_signed).
namespace sbn::v3::pq16 {
template<unsigned B,unsigned U,unsigned Odd>static inline __m256i vx_gather(sb_vec raw){ // 8 digits 2k+Odd of group U, unaligned inside u32 lanes
    alignas(64) static constexpr auto index=[] {struct X {unsigned char v[64]{};} x{};
        for(unsigned k=0;k<8;++k)for(unsigned q=0;q<4;++q)x.v[4*k+q]=((16*U*B)%64+(2*k+Odd)*B)/8+q;return x;}();
    return _mm512_castsi512_si256(_mm512_permutexvar_epi8(_mm512_load_si512(&index),raw));
}
template<unsigned B,unsigned U,unsigned Odd>static inline sb_dvec vx_dec(sb_vec raw){
    alignas(32) static constexpr auto shift=[] {struct X {unsigned v[8]{};} x{};
        for(unsigned k=0;k<8;++k)x.v[k]=((2*k+Odd)*B)%8;return x;}();
    auto v=vx_gather<B,U,Odd>(raw);
    v=_mm256_and_si256(_mm256_srlv_epi32(v,_mm256_load_si256((const __m256i *)&shift)),_mm256_set1_epi32((1u<<B)-1));
    return _mm512_cvtepu32_pd(v);
}
// sign-extended digit d - (t<<B) as i32 lanes: shift the digit's top bit to bit 31, then arithmetic shift back
template<unsigned B,unsigned U,unsigned Odd>static inline __m256i vx_sext(sb_vec raw){
    alignas(32) static constexpr auto shift=[] {struct X {unsigned v[8]{};} x{};
        for(unsigned k=0;k<8;++k)x.v[k]=32-B-((2*k+Odd)*B)%8;return x;}();
    return _mm256_srai_epi32(_mm256_sllv_epi32(vx_gather<B,U,Odd>(raw),_mm256_load_si256((const __m256i *)&shift)),32-B);
}
// top bit of the digit preceding group U (bit 64*first+16*U*B-1 of the operand), 0 before the operand start
template<unsigned B,unsigned U>static inline uint32_t vx_tprev(const uint64_t *a,size_t count,size_t first){
    const int64_t bit=int64_t(64*first)+int64_t(16*U*B)-1;if(bit<0)return 0;
    const size_t limb=size_t(bit)/64;return limb<count?uint32_t((a[limb]>>(bit%64))&1):0u;
}
template<unsigned B,unsigned U,bool S>static inline qcv vx_decode(const uint64_t *a,size_t count,size_t first){
    const size_t pos=first+(16*U*B)/64;const auto raw=pos<count?q_raw8(a+pos,count-pos):sb_zero();
    if constexpr(!S)return {vx_dec<B,U,0>(raw),vx_dec<B,U,1>(raw)};
    else{
        const auto e=vx_sext<B,U,0>(raw),o=vx_sext<B,U,1>(raw);
        const auto te=_mm256_srli_epi32(e,31),to=_mm256_srli_epi32(o,31);
        const auto prev=_mm256_insert_epi32(_mm256_setzero_si256(),int(vx_tprev<B,U>(a,count,first)),7);
        return {_mm512_cvtepi32_pd(_mm256_add_epi32(e,_mm256_alignr_epi32(to,prev,7))),_mm512_cvtepi32_pd(_mm256_add_epi32(o,te))};
    }
}
// Pair radix 2^(2B): split the odd coefficient at B first, so the
// pair's temporary sum fits one u64. Rounded coefficients below 2^51 give
// outgoing carry <2^(2B)+2^(2B), hence a one-bit generate/propagate chain is
// sufficient. rc = 0x4338000000000000 - K is the biased round constant.
template<unsigned B>static inline sb_vec vx_pair(qcv x,q_chain &c,sb_vec rc){
    static_assert(B>=16&&B<=21);   // 16 balanced for the large band (2026-09-09), 21 only for the envelope probes
    const sb_dvec magic=sb_as_dvec(sb_set1_64(0x4338000000000000LL));
    const auto even=sb_sub(sb_as_ivec(sb_add(x.re,magic)),rc),odd=sb_sub(sb_as_ivec(sb_add(x.im,magic)),rc),mask=sb_set1_64((1ull<<(2*B))-1);
    const auto raw=sb_add(even,sb_slli(sb_and(odd,sb_set1_64((1u<<B)-1)),B));
    const auto out=sb_add(sb_srli(odd,B),sb_srli(raw,2*B));
    auto sum=sb_add(sb_and(raw,mask),sb_alignr64(out,c.prev_hi,7));
    if constexpr(B==16){
        // 32-bit slots: biased coefficients up to 2^51 make the incoming carry up to 2^35, so a lane can overflow its slot
        // twice. The one-bit ripple below then loses a carry; resolve such (rare) steps, and steps entered with a
        // multi-bit carry-in, with an exact scalar pass.
        const __mmask8 twice=sb_gtu(sum,sb_set1_64(2*((1ull<<32)-1)+1));
        if(__builtin_expect(twice||c.cin>1,0)){
            alignas(64) uint64_t t[8];sb_store(t,sum);uint64_t carry=c.cin;
            for(unsigned l=0;l<8;++l){const uint64_t v=t[l]+carry;t[l]=v&((1ull<<32)-1);carry=v>>32;}
            c.prev_hi=out;c.cin=unsigned(carry);return sb__fn(load_si512)((const void *)t);
        }
    }
    const unsigned g=unsigned(sb_ltu(mask,sum)),pr=unsigned(sb_eq(sum,mask)),cn=pr+((g<<1)|c.cin),cy=cn^pr;
    sum=sb_add(sum,sb_set1_64(1),sb_k8(cy),sum);c.prev_hi=out;c.cin=(cn>>8)&1;return sb_and(sum,mask);
}
template<unsigned B,unsigned W>static inline sb_vec vx_pair_words(sb_vec a,sb_vec b){
    constexpr unsigned S=2*B,base=64*W/S;
    alignas(64) static constexpr auto map=[] {struct X {uint64_t g[8],right[8],left1[8],left2[8];} x{};
        for(unsigned k=0;k<8;++k){const unsigned bit=64*(W+k),s=bit%S;x.g[k]=bit/S-base;x.right[k]=s;x.left1[k]=S-s;x.left2[k]=2*S-s;}return x;}();
    const auto g=_mm512_load_si512(map.g);
    const auto x=_mm512_permutex2var_epi64(a,g,b),y=_mm512_permutex2var_epi64(a,sb_add(g,sb_set1_64(1)),b),z=_mm512_permutex2var_epi64(a,sb_add(g,sb_set1_64(2)),b);
    return sb_or(sb_or(_mm512_srlv_epi64(x,_mm512_load_si512(map.right)),_mm512_sllv_epi64(y,_mm512_load_si512(map.left1))),_mm512_sllv_epi64(z,_mm512_load_si512(map.left2)));
}
// Emit target: the product r[0,size); signed codec adds the tail staging of
// the last reaching front, limbs [tail_base,tail_end) (tail_base<size<=tail_end).
struct vx_target {uint64_t *r;size_t size;uint64_t *tail;size_t tail_base,tail_end;};
static inline void vx_store(const vx_target &t,size_t off,unsigned len,sb_vec v){
    if(off<t.size)q_st_tail(t.r+off,v,int64_t(std::min<size_t>(t.size-off,len)));
    if(t.tail&&off+len>t.size&&off>=t.tail_base&&off<t.tail_end)sb_store(t.tail+(off-t.tail_base),v,q_st_mask(std::min<size_t>(t.tail_end-off,len)));
}
template<unsigned B>static inline void vx_emit64(const vx_target &t,size_t off,const qcv *v,q_chain &chain,sb_vec rc){
#ifdef VX_COEFFICIENT_OBSERVER
    for(unsigned q=0;q<4;++q){alignas(64) double tmp[16];q_st(tmp,v[q]);for(unsigned k=0;k<8;++k){VX_COEFFICIENT_OBSERVER((off/B)*64+16*q+2*k,tmp[k]);VX_COEFFICIENT_OBSERVER((off/B)*64+16*q+2*k+1,tmp[k+8]);}}
#endif
    const auto a=vx_pair<B>(v[0],chain,rc),b=vx_pair<B>(v[1],chain,rc),c=vx_pair<B>(v[2],chain,rc),d=vx_pair<B>(v[3],chain,rc);
    static_assert(B>=16,"the 8-limb stores take two 8-slot vectors: 16 slots must cover 512 bits");
    constexpr unsigned mid=512/(2*B)-8,last=1024/(2*B)-24;   // B = 16: mid = 8 means "the next vector" (valignq keeps only three immediate bits)
    vx_store(t,off,8,vx_pair_words<B,0>(a,b));
    if constexpr(mid==8)vx_store(t,off+8,8,vx_pair_words<B,8>(c,d));
    else vx_store(t,off+8,8,vx_pair_words<B,8>(sb_alignr64(c,b,mid),sb_alignr64(d,c,mid)));
    if constexpr(B>16)vx_store(t,off+16,B-16,vx_pair_words<B,16>(sb_alignr64(sb_zero(),d,last),sb_zero()));
}
// Signed codec: bias m for a shape (K = m(2^B-1) > max |coefficient| + 1/2 for |digit| <= 2^(B-1), min digits <= N)
static inline uint64_t vx_bias_m(unsigned bits,unsigned nfull){
    const unsigned __int128 c=((unsigned __int128)nfull<<(2*bits-2))+2;return uint64_t((c+((1ull<<bits)-2))/((1ull<<bits)-1));
}
// Close the fronts of a signed emit: fronts 0..F carry the bias, front F holds
// limb size-1, its slots past the product are staged in the tail together with
// the first limbs of front F+1 (the carry digits of exact-length operands put
// their product coefficients up to two digits above the value's top digit,
// which can be the next front). Carries ripple through r then the tail; with S
// the (tiny, unbiased) value front F+1 holds, front F's carry is exactly m-S,
// so after rippling it into front F+1's staged limbs those must read m; front
// F's own limbs past the product must be zero; later fronts silent.
static inline bool vx_close_signed(const vx_target &t,q_chain *ch,unsigned fronts,size_t limbs_front,unsigned F,uint64_t m){
    const size_t end_F=size_t(F+1)*limbs_front;const bool next=F+1<fronts;
    auto trailing=[&](unsigned f){uint64_t hi7;memcpy(&hi7,(const char *)&ch[f].prev_hi+56,8);return (unsigned __int128)hi7+ch[f].cin;};
    auto ripple=[&](unsigned __int128 c,size_t from){   // add c at limb `from`, through r then the staged tail; returns what survives
        uint64_t *q=t.r+from,*qe=t.r+t.size;if(from<t.size){while(c&&q<qe){c+=*q;*q++=uint64_t(c);c>>=64;}from=t.size;}
        if(from<t.tail_base)return c;
        uint64_t *p=t.tail+(from-t.tail_base),*pe=t.tail+(t.tail_end-t.tail_base);
        while(c&&p<pe){c+=*p;*p++=uint64_t(c);c>>=64;}return c;};
    unsigned __int128 acc=0;   // carries past the end of the last front (all-bias fronts end one short of m and pass 1 on)
    for(unsigned f=0;f<F;++f)acc+=ripple(trailing(f),size_t(f+1)*limbs_front);
    for(size_t i=t.size;i<end_F;++i)if(t.tail[i-t.tail_base])return false;   // front F past the product: zero
    if(next){
        if(acc||ripple(trailing(F),end_F))return false;   // nothing may pass the 16-limb spill window
        if(t.tail[end_F-t.tail_base]!=m)return false;
        for(size_t i=end_F+1;i<t.tail_end;++i)if(t.tail[i-t.tail_base])return false;
        if(trailing(F+1))return false;
    }else if(trailing(F)+acc!=m)return false;
    for(unsigned f=F+2;f<fronts;++f)if(trailing(f))return false;
    return true;
}
template<unsigned B,unsigned U,bool S>static inline void vx_input_pair(double *data,const uint64_t *a,size_t count,unsigned n,unsigned j,const pq16_plan &pl){
    const unsigned l=n/4;const bool compact=pq16_twc(n);qcv x[4],y[4];
    for(unsigned s=0;s<4;++s){const size_t first=(size_t(s)*l+j)*B/32;x[s]=vx_decode<B,U,S>(a,count,first);y[s]=vx_decode<B,U+1,S>(a,count,first);}
    qcv w1,w2,w3,w4;const auto *tw=pl.tw22[__builtin_ctz(n)]+((j/16)+U/2)*pq16_tw_step(compact);
    pq16_tw22_get(tw,compact,&w1,&w2,&w3,&w4);q_bf4(x,w1,w2);q_bf4(y,w3,w4);
    for(unsigned s=0;s<4;++s){auto *p=data+2*size_t(s)*l+2*j+16*U;q_st(p,x[s]);q_st(p+16,y[s]);}
}
template<unsigned M,unsigned B,bool S=false>static inline void vx_forward(double *data,const uint64_t *a,size_t count,const CtTables &t){
    const unsigned n=t.shape.branch;const auto *pl=t.core;
    if constexpr(M==1){
        for(unsigned j=0;j<n/4;j+=32){vx_input_pair<B,0,S>(data,a,count,n,j,*pl);vx_input_pair<B,2,S>(data,a,count,n,j,*pl);}
    }else{
        for(unsigned j=0;j<n;j+=32){qcv v[4][M],y[M];
            for(unsigned s=0;s<M;++s){size_t first=(size_t(s)*n+j)*B/32;
                v[0][s]=vx_decode<B,0,S>(a,count,first);v[1][s]=vx_decode<B,1,S>(a,count,first);v[2][s]=vx_decode<B,2,S>(a,count,first);v[3][s]=vx_decode<B,3,S>(a,count,first);}
            for(unsigned u=0;u<4;++u){q_pfa_bfly(v[u],y,M,1);for(unsigned b=0;b<M;++b){if(b)y[b]=q_mul(y[b],q_ld(t.tw+2*size_t(b-1)*n+2*j+16*u));q_st(data+2*size_t(b)*n+2*j+16*u,y[b]);}}}
        for(unsigned s=0;s<M;++s)pq16_odd_fwd(data+2*size_t(s)*n,n,pl);
    }
    if constexpr(M==1)pq16_fwd_core(data,n,pl);
}
// Front bookkeeping shared by the CT/PQ and right-angle wide emits: per-front
// round constants, the injected bias and the tail window of front F.
template<unsigned B,bool S>struct vx_fronts {
    vx_target target;sb_vec rc_biased,rc_plain;unsigned F=0;uint64_t m=0;
    vx_fronts(uint64_t *r,size_t size,uint64_t *tail,unsigned fronts,size_t limbs_front,unsigned nfull){
        target={r,size,nullptr,0,0};rc_plain=sb_set1_64(0x4338000000000000LL);rc_biased=rc_plain;
        if constexpr(S){
            m=vx_bias_m(B,nfull);rc_biased=sb_sub(rc_plain,sb_set1_64(uint64_t(m*((1ull<<B)-1))));
            F=unsigned((size-1)/limbs_front);if(F>=fronts)F=fronts-1;
            const size_t front_start=size_t(F)*limbs_front,base=front_start+B*((size-1-front_start)/B);
            target.tail=tail;target.tail_base=base;target.tail_end=size_t(F+1)*limbs_front+(F+1<fronts?16:0);   // + the spill window of front F+1
        }
    }
    sb_vec rc(unsigned f)const{return S&&f<=F?rc_biased:rc_plain;}
    void init(q_chain &c0)const{if constexpr(S){c0.prev_hi=sb_set1_64(0);uint64_t v=m;memcpy((char *)&c0.prev_hi+56,&v,8);}}
    bool close(q_chain *ch,unsigned fronts,size_t limbs_front)const{
        if constexpr(S)return vx_close_signed(target,ch,fronts,limbs_front,F,m);
        else return q_chains_close(target.r,int64_t(target.size),ch,fronts,limbs_front)!=0;
    }
};
template<unsigned M,unsigned B,bool S=false,bool Cyclic=false,bool Prefix=false>static inline bool vx_emit(uint64_t *r,size_t size,double *data,const CtTables &t,uint64_t *tail){
    const unsigned n=t.shape.branch,fronts=M==1?4:M,len=M==1?n/4:n;const auto *pl=t.core;q_chain chains[fronts]{};
    const size_t limbs_front=size_t(len)*B/32;vx_fronts<B,S> fr(r,size,tail,fronts,limbs_front,t.shape.nfull);if constexpr(!Cyclic||Prefix)fr.init(chains[0]);
    const bool compact=pq16_twc(n);const double *tw=pl->tw22[__builtin_ctz(n)];
    for(unsigned j=0;j<len;j+=32){qcv v[fronts][4];
        if constexpr(M==1){for(unsigned u=0;u<4;u+=2){qcv w1,w2,w3,w4;pq16_tw22_get(tw,compact,&w1,&w2,&w3,&w4);tw+=pq16_tw_step(compact);
            for(unsigned q=0;q<2;++q){qcv x[4];for(unsigned s=0;s<4;++s)x[s]=q_ld(data+2*size_t(s)*len+2*j+16*(u+q));q_ibf4(x,q?w3:w1,q?w4:w2);for(unsigned s=0;s<4;++s)v[s][u+q]=x[s];}}}
        else{for(unsigned u=0;u<4;++u){qcv x[M],y[M];for(unsigned b=0;b<M;++b){x[b]=q_ld(data+2*size_t(b)*n+2*j+16*u);if(b)x[b]=q_mulc(x[b],q_ld(t.tw+2*size_t(b-1)*n+2*j+16*u));}q_pfa_bfly(x,y,M,0);for(unsigned s=0;s<M;++s)v[s][u]=y[s];}}
        for(unsigned s=0;s<fronts;++s)if(!Prefix||(size_t(s)*len+j)*B/32<size)vx_emit64<B>(fr.target,(size_t(s)*len+j)*B/32,v[s],chains[s],fr.rc(s));
    }
    // In a complete cyclic period the signed-digit bias sums to
    // m*(2^(64*size)-1), hence is already zero modulo the ring. Do not
    // inject the linear recipe's compensating m at front 0.
    if constexpr(Prefix)return q_chains_close_prefix(r,int64_t(size),chains,fronts,limbs_front)!=0;
    else if constexpr(Cyclic)return q_chains_close_cyc(r,int64_t(size),chains,fronts,limbs_front)!=0;
    else return fr.close(chains,fronts,limbs_front);
}
// tail staging limbs of the signed emit: one front, one 64-digit block, the next front's 16-limb spill window
static inline size_t vx_tail_limbs(Shape s){return (size_t(s.radix==1?s.branch/4:s.branch)*s.bits)/32+s.bits+24;}
template<unsigned M,unsigned B,bool S=false,bool Cyclic=false,bool Prefix=false>static inline bool vx_multiply(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const CtTables &t,Frame &f,const double *cached=nullptr,bool cached_square=false,size_t prefix=0){
    const bool square=Cyclic?cached_square:cached?cached_square:a==b && an==bn;
    const unsigned n=t.shape.branch,N=t.shape.nfull;FrameMark mark(f);double *A=f.alloc<double>((cached||square?2:4)*size_t(N)+48),*D=cached?const_cast<double *>(cached):square?A:A+2*size_t(N)+32;
    uint64_t *tail=nullptr;if constexpr(S&&!Cyclic&&!Prefix)tail=f.alloc<uint64_t>(vx_tail_limbs(t.shape));
    if(cached){if(square){if constexpr(!Cyclic)memcpy(A,cached,2*size_t(N)*8);}else vx_forward<M,B,S>(A,b,bn,t);}
    else{if(!square)vx_forward<M,B,S>(D,b,bn,t);vx_forward<M,B,S>(A,a,an,t);}
    if constexpr(Cyclic){if(cached&&square)ct_pointwise<M,true>(A,D,t);else ct_pointwise<M>(A,D,t);}else ct_pointwise<M>(A,D,t);
    for(unsigned s=0;s<M;++s){double *p=A+2*size_t(s)*n;if constexpr(M!=1)pq16_odd_inv(p,n,t.core);else pq16_inv_r8_only(p,n,t.core);}
    return vx_emit<M,B,S,Cyclic,Prefix>(r,Prefix?prefix:Cyclic?size_t(N)*B/32:an+bn,A,t,tail);
}
}
