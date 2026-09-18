#pragma once
#include <new>
// CT/PQ layout: global frequency k=b+M*k2. Nonzero branch partners are
// (M-b,n-1-k2), a full reversal in bit-reversed/pi storage.
namespace sbn::v3::pq16 {
struct CtTables {
    const pq16_plan *core;Shape shape;
    double *tw;double phase_re[7],phase_im[7];
};
static inline size_t ct_table_bytes(Shape s){return sizeof(CtTables)+256+16*size_t(s.nfull-s.branch);}
static inline void ct_root_ratio(long double &c,long double &s,uint64_t k,uint64_t n){
    k%=n;const uint64_t quadrant=k/(n/4);k%=n/4;bool reflect=k>n/8;if(reflect)k=n/4-k;
    const uint64_t num=k<<PQ16_ROOT_GRID_LOG2,index=num/n,rem=num%n;
    const auto *a=PQ16_ROOT_COARSE[index>>8],*b=PQ16_ROOT_FINE[index&255];
    c=a[0]*b[0]-a[1]*b[1];s=a[0]*b[1]+a[1]*b[0];
    if(rem){
        // |d|<2*pi/2^19: omitted cosine term is below 9e-22.
        const long double d=0xc.90fdaa22168c235p-1L*(long double)rem/((long double)n*(1u<<PQ16_ROOT_GRID_LOG2));
        const long double dc=1-d*d/2,ds=d-d*d*d/6,x=c;c=x*dc-s*ds;s=s*dc+x*ds;
    }
    if(reflect){auto t=c;c=s;s=t;}const auto x=c,y=s;
    switch(quadrant){case 0:break;case 1:c=-y;s=x;break;case 2:c=-x;s=-y;break;default:c=y;s=-x;}
}
static inline CtTables *ct_prepare(Frame &f,const pq16_plan &core,Shape s){
    auto *p=new(f.allocate(sizeof(CtTables),128)) CtTables{};p->core=&core;p->shape=s;
    if(s.radix==1)return p;
    p->tw=static_cast<double *>(f.allocate(16*size_t(s.nfull-s.branch),128));
    for(unsigned b=1;b<s.radix;++b){long double cr,ci;ct_root_ratio(cr,ci,b,s.nfull);p->phase_re[b]=(double)cr;p->phase_im[b]=(double)ci;}
    // One accurate seed per j, then at most six extended-precision
    // products. There is no recurrence from one j to the next.
    for(unsigned j=0;j<s.branch;++j){
        long double wr,wi;ct_root_ratio(wr,wi,j,s.nfull);long double cr=wr,ci=wi;
        for(unsigned b=1;b<s.radix;++b){
            long double re=cr,im=ci;const uint64_t exponent=uint64_t(b)*j;
            if(exponent==0){re=1;im=0;}
            else if(exponent==s.nfull/4){re=0;im=1;}
            else if(exponent==s.nfull/2){re=-1;im=0;}
            else if(exponent==3*s.nfull/4){re=0;im=-1;}
            const size_t at=2*size_t(b-1)*s.branch+2*(j&~7u)+(j&7u);
            p->tw[at]=(double)re;p->tw[at+8]=(double)im;
            const long double next=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=next;
        }
    }
    return p;
}
template<unsigned M>static inline void ct_forward(double *data,const uint64_t *a,size_t count,const CtTables &t){
    const unsigned n=t.shape.branch;const auto *pl=t.core;
    for(unsigned j=0;j<n;j+=16){
        qcv x0[M],x1[M],y[M];
        for(unsigned s=0;s<M;++s){const size_t first=(size_t(s)*n+j)/2;
            const auto raw=first<count?q_raw8_center(a+first,count-first,0):sb_zero();q_dec2(x0+s,x1+s,raw,pl->dec_idx,0);}
        for(unsigned u=0;u<2;++u){q_pfa_bfly(u?x1:x0,y,M,1);
            for(unsigned b=0;b<M;++b){if(b)y[b]=q_mul(y[b],q_ld(t.tw+2*size_t(b-1)*n+2*j+16*u));q_st(data+2*size_t(b)*n+2*j+16*u,y[b]);}}
    }
    for(unsigned b=0;b<M;++b)pq16_odd_fwd(data+2*size_t(b)*n,n,pl);
}
// Preserve a cached spectrum while retiring its group copy immediately
// into pointwise + inverse leaf work; no complete read/write copy pass.
static inline void ct_square_first_branch(double *out,double *saved,unsigned n,double scale,unsigned leaf,const pq16_plan *pl){
    const auto sc=sb_set1_d(scale),qsc=sb_set1_d(.25*scale);
    memcpy(out,saved,256*sizeof(double));q_pw_head(out,saved,pl,sc,qsc);q_pw_groups(out,saved,1,1,pl,sc,qsc);
    if(leaf==128)pq16_ileaf128_run(out,128,pl);else{q_ileaf_group(out,0,leaf,pl);q_ileaf_group(out,1,leaf,pl);}
    for(unsigned base=128;base<n;base*=2){const unsigned g0=base/64;
        for(unsigned gl=g0;gl<g0+base/128;++gl){const unsigned gr=3*g0-1-gl;
            memcpy(out+128*size_t(gl),saved+128*size_t(gl),128*sizeof(double));
            memcpy(out+128*size_t(gr),saved+128*size_t(gr),128*sizeof(double));
            q_pw_groups(out,saved,gl,gr,pl,sc,qsc);
            if(leaf!=128){q_ileaf_group(out,gl,leaf,pl);q_ileaf_group(out,gr,leaf,pl);}
            else if(base==128)pq16_ileaf128_run(out+256*size_t(gl>>1),128,pl);
            else if(gl&1){pq16_ileaf128_run(out+256*size_t((gl-1)>>1),128,pl);pq16_ileaf128_run(out+256*size_t(gr>>1),128,pl);}
        }
    }
}
template<unsigned M,bool CopySquare=false>static inline void ct_pointwise(double *A,double *B,const CtTables &t){
    const unsigned n=t.shape.branch,N=t.shape.nfull,leaf=M==1?pq16_shape_of(n).leaf:pq16_odd_leaf(n);const auto *pl=t.core;
    if constexpr(CopySquare)ct_square_first_branch(A,B,n,1./N,leaf,pl);else pq16_pointwise_ileaves(A,B,n,1./N,leaf,pl);const auto sc=sb_set1_d(1./N),qsc=sb_set1_d(.25/N);
    for(unsigned b=1;b<=M/2;++b){auto *L=A+2*size_t(b)*n,*R=A+2*size_t(M-b)*n,*x=B+2*size_t(b)*n,*y=B+2*size_t(M-b)*n;
        for(unsigned gl=0;gl<n/64;++gl){unsigned gr=n/64-1-gl;
            if constexpr(CopySquare){memcpy(L+128*size_t(gl),x+128*size_t(gl),128*sizeof(double));memcpy(R+128*size_t(gr),y+128*size_t(gr),128*sizeof(double));}
            q_pw_groups_cross(L,R,x,y,gl,gr,pl,t.phase_re[b],t.phase_im[b],sc,qsc);
            if(leaf!=128){q_ileaf_group(L,gl,leaf,pl);q_ileaf_group(R,gr,leaf,pl);}
            else if(gl&1){pq16_ileaf128_run(L+128*size_t(gl-1),128,pl);pq16_ileaf128_run(R+128*size_t(gr),128,pl);}}
    }
}
template<unsigned M>static inline bool ct_emit(uint64_t *r,size_t size,double *data,const CtTables &t){
    const unsigned n=t.shape.branch;q_chain chains[M]{};
    for(unsigned j=0;j<n;j+=16){alignas(64) double stage[M][32];
        for(unsigned u=0;u<2;++u){qcv x[M],y[M];for(unsigned b=0;b<M;++b){
            x[b]=q_ld(data+2*size_t(b)*n+2*j+16*u);if(b)x[b]=q_mulc(x[b],q_ld(t.tw+2*size_t(b-1)*n+2*j+16*u));}
            q_pfa_bfly(x,y,M,0);for(unsigned s=0;s<M;++s)q_st(stage[s]+16*u,y[s]);}
        for(unsigned s=0;s<M;++s){const size_t off=(size_t(s)*n+j)/2;
            const auto q=q_pack2<true>(q_ld(stage[s]),q_ld(stage[s]+16),off);const auto out=q_chain_step(chains+s,q);
            if(off<size)q_st_tail(r+off,out,int64_t(size-off));}
    }
    return q_chains_close(r,size,chains,M,n/2);
}
template<unsigned M>static inline bool ct_multiply(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const CtTables &t,Frame &f,const double *cached=nullptr,bool cached_square=false){
    // Recipe query proves non-centered LIN and N<=32768, hence narrow pack.
    const bool square=cached?cached_square:a==b && an==bn;
    const unsigned n=t.shape.branch,N=t.shape.nfull;FrameMark mark(f);double *A=f.alloc<double>((cached||square?2:4)*size_t(N)+48),*B=cached?const_cast<double *>(cached):square?A:A+2*size_t(N)+32;
    if(cached){if(square)memcpy(A,cached,2*size_t(N)*8);else ct_forward<M>(A,b,bn,t);}
    else{if(!square)ct_forward<M>(B,b,bn,t);ct_forward<M>(A,a,an,t);}ct_pointwise<M>(A,B,t);
    for(unsigned s=0;s<M;++s)pq16_odd_inv(A+2*size_t(s)*n,n,t.core);
    return ct_emit<M>(r,an+bn,A,t);
}
}
