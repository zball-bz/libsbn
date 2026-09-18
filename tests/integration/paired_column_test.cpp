#define CR_NP 10
#define SBN3_P48_NS pair_writeback_gate
#include "product_support.hpp"
#include "backend/ntt_p48/engine.hpp"
#include <algorithm>
using namespace sbn::v3;
using namespace sbn::v3::pair_writeback_gate;
int main(){
    Fixture fixture(1,false);
    auto tables=fixture.allocate(120u<<20,128),work=fixture.allocate(8u<<20,128);
    Frame table_frame(*fixture.arena,tables);Primes primes{};
    primes.init(table_frame,19);primes.factors_for(table_frame,32,13);
    const size_t capacity=std::max(size_t(9)*(8192*TB*SLOT+64),
                                   size_t(8192)*padded_row_stride(9*TB*SLOT,1))+128;
    uint8_t *planes[6]{};
    for(auto &p:planes)p=static_cast<uint8_t *>(fixture.allocate(capacity,64).data);
    const auto pack=pk52_mk();size_t cases=0,checked_bytes=0;
    for(unsigned q=0;q<NP;++q)for(size_t C:{64u,256u,1024u,4096u,8192u})
      for(size_t blocks:{1u,7u,9u})for(unsigned row_major:{0u,1u})for(unsigned nts:{0u,1u}){
        Plan g{};g.C=g.nrows=C;g.lgC=__builtin_ctzll(C);g.M2=32;g.nblk=blocks;g.lbw=blocks*TB;
        g.bstride=g.bstride_o=C*TB*SLOT+64;g.plane_bytes=g.plane_bytes_o=blocks*g.bstride+128;
        const size_t rms=padded_row_stride(g.lbw*SLOT,1),product_bytes=row_major?rms*C+128:g.plane_bytes;
        assert(g.plane_bytes<=capacity && product_bytes<=capacity);
        for(auto p:planes)memset(p,0xc7,capacity);
        const auto &pv=primes.V[q];const uint64_t p=primes.P[q].p;
        for(size_t block=0;block<blocks;++block)for(size_t vr=0;vr<C;++vr)for(size_t slot=0;slot<TB;++slot){
            alignas(64) uint64_t a[8],b[8];
            for(unsigned lane=0;lane<8;++lane){a[lane]=lane==0?4*p-1:random_word()%(4*p);b[lane]=random_word()%(4*p);}
            const size_t offset=block*g.bstride+(vr*TB+slot)*SLOT;
            st52(planes[0]+offset,_mm512_load_si512(a),pack,pv);st52(planes[2]+offset,_mm512_load_si512(b),pack,pv);
        }
        memcpy(planes[1],planes[0],capacity);memcpy(planes[3],planes[2],capacity);
        Ctx ref{};ref.pl=&g;ref.PS=&primes;ref.hbs=ref.obs=g.bstride;ref.smalltw=1;ref.nts=nts;ref.rms=rms;
        ref.fp[q]=planes[0];ref.hp[q]=planes[0];
        {Frame frame(*fixture.arena,work);block_fn<8,0,1>(&ref,q*blocks,(q+1)*blocks,0,&frame);}
        ref.fp[q]=planes[2];ref.prm=row_major;ref.op[q]=row_major?planes[4]:nullptr;ref.frontier[0]=1;
        Ctx next=ref;next.hp[q]=planes[1];next.fp[q]=planes[3];next.op[q]=row_major?planes[5]:nullptr;
        next.frontier[0]=0;next.writeback_a=1;
        const size_t items=row_major?(blocks+7)&~size_t(7):blocks;
        allocation_watch_start();
        for(size_t k=0;k<items;k+=8){
            const size_t lo=q*items+k,hi=q*items+std::min(k+8,items);
            {Frame frame(*fixture.arena,work);block_fn4<8,0,0>(&ref,lo,hi,0,&frame);}
            {Frame frame(*fixture.arena,work);
                block_fn4<8,0,0,0,false,true>(&next,lo,hi,0,&frame);
            }
        }
        assert(!allocation_watch_stop());
        assert(!memcmp(planes[0],planes[1],capacity));
        assert(!memcmp(row_major?planes[4]:planes[2],row_major?planes[5]:planes[3],capacity));
        cases++;checked_bytes+=2*capacity;
    }
    printf("{\"status\":\"passed\",\"cases\":%zu,\"compared_bytes\":%zu,\"canonical_spectrum_and_product\":true,\"block_padding\":true,\"ctx_bytes\":%zu,\"rms_offset\":%zu}\n",
           cases,checked_bytes,sizeof(Ctx),offsetof(Ctx,rms));
}
