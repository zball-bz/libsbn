#include "backend/pq16/island.hpp"
#include <cstdio>
#include <initializer_list>
using namespace sbn::v3::pq16;
extern "C" void test_pq16_pack(){
    const long double tau=2*acosl(-1.L);
    for(unsigned M:{3u,5u,7u})for(unsigned n=64;n<=(M==7?2048u:4096u);n*=2){
        const auto *table=root_bank::rac_twiddle(M,n);assert(table);
        for(unsigned b=0;b<M;++b)for(unsigned j=0;j<n;++j){
            const uint64_t mod=4*uint64_t(M)*n,k=(uint64_t(4*b+(M%4)*M)*j)%mod;
            const long double angle=tau*k/mod;
            const size_t at=2*size_t(b)*n+2*(j&~7u)+(j&7u);
            assert(fabsl(table[at]-cosl(angle))<2e-16L&&fabsl(table[at+8]-sinl(angle))<2e-16L);
        }
    }
    for(unsigned lg=6;lg<=root_bank::pq_max_log2;++lg){
        const unsigned n=1u<<lg;
        for(bool r8:{false,true}){
            const bool compact=lg>=root_bank::compact_min_log2;
            const double *table=r8?root_bank::twiddle<true>(lg,compact):root_bank::twiddle<false>(lg,compact);
            if(!root_bank::has_stage(lg,compact,r8)){assert(!table);continue;}assert(table);
            assert(!(r8?root_bank::twiddle<true>(lg,!compact):root_bank::twiddle<false>(lg,!compact)));
            if(compact)for(unsigned g=0;g<n/64;++g){
                alignas(128) double runtime[32];
                pq16_twcv(runtime,g*(r8?8:16),1,n);
                pq16_twcv(runtime+16,r8?8*g:16*g+8,r8?2:1,n);
                assert(!memcmp(table+32*g,runtime,sizeof runtime));
            }
            else for(unsigned g=0;g<n/64;++g)for(unsigned block=0;block<4;++block)for(unsigned lane=0;lane<8;++lane){
                const unsigned j=g*(r8?8:16),scale=r8?(block<2?1:1u<<(block-1)):(block&1?2:1);
                const unsigned k=r8?(j+lane)*scale+(block==1?n/8:0):(j+lane+(block>=2?8:0))*scale;
                const auto w=root_bank::root(k,lg);const size_t at=64*g+16*block+lane;
                assert(!memcmp(table+at,&w.re,8)&&!memcmp(table+at+8,&w.im,8));
            }
        }
        for(unsigned g=0;g<n/64;++g)for(unsigned k=0;k<16;++k){
            const auto w=root_bank::root(root_bank::reverse(g*16+k,lg-2),lg);
            assert(!memcmp(root_bank::pq.data()+32*g+k,&w.re,8)&&!memcmp(root_bank::pq.data()+32*g+16+k,&w.im,8));
            double re,im;pq16_root(&re,&im,pq16_bitrev(g*16+k,lg-2),n);
            assert(!memcmp(root_bank::pq.data()+32*g+k,&re,8)&&!memcmp(root_bank::pq.data()+32*g+16+k,&im,8));
        }
    }
    puts("published compact stage roots: bitwise runtime-builder equivalence and layout guards PASS");
    // Tail windows near a page seam may sit in an allocation smaller than a
    // cache line. Poison every non-operand word, even though the pages exist.
    alignas(4096) uint64_t input[1024]{};
    alignas(64) uint64_t got[8];
    for(unsigned back=1;back<8;++back)for(unsigned count=1;count<8;++count){
        auto *p=input+512-back;
        SBN3_FRAME_POISON(input,sizeof input);SBN3_FRAME_UNPOISON(p,count*8);
        for(unsigned j=0;j<count;++j)p[j]=0x8172635445362718ULL+j;
        sb_store(got,q_raw8(p,count));
        for(unsigned j=0;j<8;++j)assert(got[j]==(j<count?0x8172635445362718ULL+j:0));
        SBN3_FRAME_UNPOISON(input,sizeof input);
    }
    constexpr uint64_t C=(1ull<<16)*65535*65535;
    alignas(64) uint64_t re0[8],im0[8],re1[8],im1[8],lo[8],hi[8],wlo[8],whi[8];
    uint64_t state=0x648f04938751ULL;
    for(unsigned t=0;t<4608;++t){
        for(unsigned i=0;i<8;++i)for(uint64_t *p:{re0,im0,re1,im1}){
            state^=state<<13;state^=state>>7;state^=state<<17;
            p[i]=t==0?C:t==1?0:t==2?C-(i&1):t==4096?UINT64_MAX:t>4096?state:state%(C+1);
        }
        const auto b=q_pack2i<false>(sb_load(re0),sb_load(im0),sb_load(re1),sb_load(im1));
        const auto a=t<4096?q_pack2i<true>(sb_load(re0),sb_load(im0),sb_load(re1),sb_load(im1)):b;
        sb_store(lo,a.lo);sb_store(hi,a.hi);sb_store(wlo,b.lo);sb_store(whi,b.hi);
        for(unsigned i=0;i<8;++i){
            const auto *re=i<4?re0:re1,*im=i<4?im0:im1;unsigned k=2*(i%4);
            const unsigned __int128 exact=(unsigned __int128)re[k]+((unsigned __int128)im[k]<<16)+
                ((unsigned __int128)re[k+1]<<32)+((unsigned __int128)im[k+1]<<48);
            assert(lo[i]==(uint64_t)exact && hi[i]==(uint64_t)(exact>>64));
            assert(lo[i]==wlo[i] && hi[i]==whi[i]);
        }
        if(t<4096){
            const auto delta=sb_set1_d((t&1)?-0.25:0.25);
            qcv x{sb_add(sb__fn(cvtepu64_pd)(sb_load(re0)),delta),sb_add(sb__fn(cvtepu64_pd)(sb_load(im0)),delta)};
            qcv y{sb_add(sb__fn(cvtepu64_pd)(sb_load(re1)),delta),sb_add(sb__fn(cvtepu64_pd)(sb_load(im1)),delta)};
            const auto fp=q_pack2<true>(x,y);sb_store(lo,fp.lo);sb_store(hi,fp.hi);
            for(unsigned i=0;i<8;++i)assert(lo[i]==wlo[i]&&hi[i]==whi[i]);
        }
    }
    puts("pq16 narrow pack: exact 128-bit coefficients/boundary/wide equivalence PASS");
}
