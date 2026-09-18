#include "backend/pq16/island.hpp"
#include <cstdio>
#include <initializer_list>
using namespace sbn::v3::pq16;
extern "C" void test_pq16_pack(){
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
