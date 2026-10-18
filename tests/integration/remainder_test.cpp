// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/divrem.h"
#include "sbn3/prepared_divisor.hpp"
#include "backend/u52/kernels.hpp"
#include "runtime/scratch.hpp"
#include "../oracle/oracle.h"
#include <cassert>
#include <cstdio>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
static uint64_t seed=439157231;static uint64_t random_word(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static size_t trim(const uint64_t*p,size_t n){while(n&&!p[n-1])--n;return n;}
static void one(size_t nn,size_t dn,unsigned pattern){
    std::vector<uint64_t>N(nn),D(dn);for(auto&w:N)w=random_word();for(auto&w:D)w=random_word();D.back()|=1;
    if(pattern==1){std::fill(N.begin(),N.end(),UINT64_MAX);std::fill(D.begin(),D.end(),UINT64_MAX);}
    if(pattern==2){D.back()=1;if(!N.empty())N.back()=0;}
    if(pattern==3){std::fill(N.begin(),N.end(),0);std::fill(D.begin(),D.end(),0);D.back()=1;}
    const auto savedN=N,savedD=D;const size_t qn=nn>=dn?nn-dn+1:0;
    const auto sb=sbn3_divrem_small_scratch_bytes(nn,dn),rb=sbn3_remainder_small_scratch_bytes(nn,dn);
    if(dn<=2)assert(!rb);assert(rb<=sb+8*qn+128);
    sbn::workspace full,rem;full.reserve(sb,64);rem.reserve(rb+64,64);
    std::vector<uint64_t>Q(qn),R(dn),got(dn+8,UINT64_C(0xf0e1d2c3b4a59687));
    allocation_watch_start();sbn3_divrem_small(Q.data(),R.data(),N.data(),nn,D.data(),dn,full.on().scratch.data());assert(!allocation_watch_stop());
    auto*work=rem.on().scratch.data();std::memset(work+rb,0xa7,64);
    allocation_watch_start();sbn3_remainder_small(got.data(),N.data(),nn,D.data(),dn,rb?work:nullptr);assert(!allocation_watch_stop());
    if(!std::equal(R.begin(),R.end(),got.begin()))std::fprintf(stderr,"remainder mismatch nn=%zu dn=%zu pattern=%u\n",nn,dn,pattern);
    assert(std::equal(R.begin(),R.end(),got.begin()));for(size_t j=dn;j<got.size();++j)assert(got[j]==UINT64_C(0xf0e1d2c3b4a59687));for(size_t j=0;j<64;++j)assert(work[rb+j]==std::byte(0xa7));
    assert(N==savedN&&D==savedD);
    if(nn<=256&&dn<=80){ref_int n,d,r;ref_inits(n,d,r,nullptr);ref_import(n,N.size(),-1,8,0,0,N.data());ref_import(d,D.size(),-1,8,0,0,D.data());ref_mod(r,n,d);ref_import(n,trim(got.data(),dn),-1,8,0,0,got.data());assert(!ref_cmp(n,r));ref_clears(n,d,r,nullptr);}
    // Force the native D&C family independently of the automatic local FFT
    // crossover. It must safely retire Q over N for both leaves and recursion.
    if(dn>=3){const auto bytes=sbn::v3::u52::remainder_scratch_bytes(nn,dn);rem.reserve(bytes+128,64);work=rem.on().scratch.data();std::memset(work+bytes,0xa7,64);
        {auto frame=sbn::v3::Frame::external(work,bytes);allocation_watch_start();sbn::v3::u52::remainder(got.data(),N.data(),nn,D.data(),dn,frame);assert(!allocation_watch_stop());}
        assert(std::equal(R.begin(),R.end(),got.begin()));for(size_t j=0;j<64;++j)assert(work[bytes+j]==std::byte(0xa7));}
}
int main(){
    for(size_t dn=1;dn<=80;++dn)for(size_t nn:{size_t(0),dn-1,dn,dn+1,dn+7,2*dn,3*dn+1,8*dn+3})for(unsigned p=0;p<4;++p)one(nn,dn,p);
    for(size_t dn:{127u,128u,129u,193u,255u,256u,257u,513u,1025u,4097u,8193u})for(size_t nn:{dn,dn+13,3*dn/2,2*dn,4*dn+1})for(unsigned p=0;p<3;++p)one(nn,dn,p);
    for(size_t dn:{1u,2u,17u,257u,8193u}){
        std::vector<uint64_t>dw(dn),nw(20*dn+17);for(auto&v:dw)v=random_word();for(auto&v:nw)v=random_word();dw.back()|=1;nw.back()|=1;
        sbn::integer d{sbn::integer_view{dw}},n{sbn::integer_view{nw,true}},r;r.reserve(dn);
        const auto need=sbn::divrem_requirements(std::nullopt,r.ref(),n.view(),d.view());assert(need.supported());if(dn<=2)assert(need.scratch_bytes==0);
        sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);allocation_watch_start();sbn::divrem_into(std::nullopt,r.ref(),n.view(),d.view(),work.on());assert(!allocation_watch_stop());assert(r==n%d);
        sbn::prepared_divisor prep(d,n.size(),3);const auto reused=prep.query(std::nullopt,r.ref(),n.view());assert(reused.supported()&&reused.scratch_bytes<=8*dn+128);
        work.reserve(reused.scratch_bytes,reused.scratch_alignment);allocation_watch_start();prep.execute(std::nullopt,r.ref(),n.view(),work.on());assert(!allocation_watch_stop());assert(r==n%d);
    }
    assert(sbn::try_exact_divide(-63,7)==sbn::integer(-9));assert(!sbn::try_exact_divide(64,7));
    puts("remainder-only scalar/IFMA/DC/FFT, dead-N quotient storage, guards/oracle, prepared and bounded scratch PASS");
}
