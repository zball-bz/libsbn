#pragma once
#include "value/plus_ring.hpp"
// Bounded plus-ring kernel. Included after rac_wide.hpp. Inputs are
// shorter than the complete period; their balanced carry digit stays inside.
namespace sbn::v3::pq16 {
static inline void rac_window_front16(uint64_t *out,size_t ring,size_t off,const double *stage,unsigned part,
                                      q_chain &chain,uint64_t bias){
    const sb_vec even=sb_setr_64(0,2,4,6,8,10,12,14),odd=sb_setr_64(1,3,5,7,9,11,13,15);
    sb_dvec v[4];for(unsigned j=0;j<4;++j){v[j]=load_dvec(stage+16*j+part);
#ifdef VX_COEFFICIENT_OBSERVER
        alignas(64) double data[8];store_dvec(data,v[j]);for(unsigned k=0;k<8;++k)VX_COEFFICIENT_OBSERVER(4*off+8*j+k,data[k]);
#endif
    }
    // Each biased coefficient is <2*bias*(2^16-1). bias<=2^31 proves
    // a two-coefficient pair fits u64; the following quad uses exact lo/hi.
    const auto add=sb_set1_64(bias*((uint64_t(1)<<32)-1));
    const auto p=sb_add(rac_pairs16(v[0],v[1]),add),q=sb_add(rac_pairs16(v[2],v[3]),add);
    const auto lo=sb__fn(permutex2var_epi64)(p,even,q),hi=sb__fn(permutex2var_epi64)(p,odd,q);
    q_lohi packed;packed.lo=sb_add(lo,sb_slli(hi,32));packed.hi=sb_add(sb_srli(hi,32),sb_maskz(sb_ltu(packed.lo,lo),sb_set1_64(1)));
    q_st_tail(out+off,q_chain_step(&chain,packed),int64_t(ring-off));
}
template<unsigned M,unsigned B>static inline void rac_window_emit(uint64_t *out,double *data,const RacTables &t,unsigned terms){
    const unsigned n=t.shape.branch,N=t.shape.nfull,fronts=2*M;
    const size_t ring=size_t(N)*B/32,front_words=size_t(n)*B/64;
    q_chain chains[fronts]{};
    const uint64_t bias=vx_bias_m(B,terms);
    if constexpr(B==16)require(bias<=(uint64_t(1)<<31),SBN3_FATAL_MATH,"plus narrow pair capacity");
    const auto rc=sb_sub(sb_set1_64(0x4338000000000000ULL),sb_set1_64(bias*((uint64_t(1)<<B)-1)));
    const vx_target target{out,ring,nullptr,0,0};
    for(unsigned j=0;j<n;j+=64){alignas(64) double stage[M][8][16];
        for(unsigned u=0;u<8;++u){qcv x[M],y[M];
            for(unsigned b=0;b<M;++b)x[b]=q_mulc(q_ld(data+2*size_t(b)*n+2*j+16*u),q_ld(t.tw+2*size_t(b)*n+2*j+16*u));
            q_pfa_bfly(x,y,M,0);
            for(unsigned s=0;s<M;++s)q_st(stage[s][u],rac_rot_i(y[s],4-((t.r*s)&3)));
        }
        for(unsigned s=0;s<M;++s){
            if constexpr(B==16){
                for(unsigned half=0;half<2;++half){
                    rac_window_front16(out,ring,(size_t(s)*n+j+32*half)/4,&stage[s][4*half][0],0,chains[s],bias);
                    rac_window_front16(out,ring,(size_t(N)+size_t(s)*n+j+32*half)/4,&stage[s][4*half][0],8,chains[M+s],bias);
                }
            }else{
                vx_rac_front64<B>(target,((size_t(s)*n+j)/64)*B,&stage[s][0][0],0,chains[s],rc);
                vx_rac_front64<B>(target,((size_t(N)+size_t(s)*n+j)/64)*B,&stage[s][0][0],8,chains[M+s],rc);
            }
        }
    }
    out[ring]=0;
    for(unsigned f=0;f<fronts;++f){uint64_t high;memcpy(&high,reinterpret_cast<const char*>(&chains[f].prev_hi)+56,8);
        unsigned __int128 carry=(unsigned __int128)high+chains[f].cin;
        for(size_t j=size_t(f+1)*front_words;carry&&j<=ring;++j){carry+=out[j];out[j]=uint64_t(carry);carry>>=64;}
        require(!carry,SBN3_FATAL_MATH,"plus front carry capacity");
    }
    // Added coefficient bias sums to bias*(B_word^ring-1), congruent to
    // -2*bias modulo B_word^ring+1. Correct it together with the last carry.
    require(out[ring]<(uint64_t(1)<<62)&&bias<(uint64_t(1)<<61),SBN3_FATAL_MATH,"plus bias bound");
    const int64_t delta=int64_t(2*bias)-int64_t(out[ring]);
    limbs::plus_adjust(out,ring,delta);
}
template<unsigned M,unsigned B,bool Signed=true>static inline void rac_window_multiply(uint64_t *out,const uint64_t *a,size_t an,
    const uint64_t *b,size_t bn,const double *cached,const RacTables &t,Frame &space){
    const size_t N=t.shape.nfull,n=t.shape.branch,ring=N*B/32;
    require(an&&bn&&an<ring&&bn<ring,SBN3_FATAL_ARGUMENT,"bounded plus operands");
    FrameMark mark(space);auto *x=space.alloc<double>((cached?2:4)*N+48);
    auto *fresh=x+2*N+32;const double *other=cached;
    auto forward=[&](double *p,const uint64_t *v,size_t count){
        if constexpr(B==16&&!Signed)rac_forward<M>(p,v,count,t);
        else vx_rac_forward<M,B,Signed>(p,v,count,t);
    };
    if(!cached){forward(fresh,a,an);other=fresh;}
    forward(x,b,bn);rac_pointwise<M>(x,other,t);
    for(unsigned s=0;s<M;++s)pq16_odd_inv(x+2*size_t(s)*n,unsigned(n),t.core);
    rac_window_emit<M,B>(out,x,t,unsigned(((64*std::min(an,bn)+B-1)/B+1)*(Signed?1:4)));
}
}
