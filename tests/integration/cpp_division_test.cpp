// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/division.hpp"
#include "sbn3/prepared_divisor.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;using Mode=sbn::division_rounding;
static uint64_t rng=7410259863;static uint64_t next(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static bool adjust(Mode m,bool ns,bool ds,bool remainder){return remainder&&(m==Mode::floor?ns!=ds:m==Mode::ceil?ns==ds:m==Mode::euclidean?ns:false);}
int main(){
    for(int a=-65;a<=65;++a)for(int d=-17;d<=17;++d)if(d)
    for(auto mode:{Mode::trunc,Mode::floor,Mode::ceil,Mode::euclidean}){
        int q=a/d,r=a%d;if(adjust(mode,a<0,d<0,r!=0)){q+=(a<0)!=(d<0)?-1:1;r=a-q*d;}
        integer x=a,y=d;allocation_watch_start();const auto out=sbn::divrem(x,y,mode);
        assert(!allocation_watch_stop());assert(out.quotient==q&&out.remainder==r);
        assert(sbn::divide(x,y,mode)==q&&sbn::remainder(x,y,mode)==r);
    }
    assert(sbn::is_divisible(0,0)&&!sbn::is_divisible(1,0));assert(sbn::is_congruent(1,1,0)&&!sbn::is_congruent(1,2,0));
    assert(sbn::exact_divide(-63,7)==-9&&sbn::mod(-65,-7)==5);
    {integer n=~__uint128_t(0),d=(integer(1)<<64)+3;integer q,r;
        const auto need=sbn::divrem_requirements(q.ref(),r.ref(),n.view(),d.view());assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();sbn::divrem_into(q.ref(),r.ref(),n.view(),d.view());assert(!allocation_watch_stop());assert(q*d+r==n&&r<d);}
    for(std::size_t dn:{2u,3u,7u,17u,33u,161u,513u,1025u,8193u,32769u,65537u}){
        std::vector<uint64_t> dw(dn),qw(dn/2+1),rw(dn);
        for(auto&v:dw)v=next();dw.back()|=uint64_t(1)<<63;
        for(auto&v:qw)v=next();qw.back()|=1;
        for(auto&v:rw)v=next();rw.back()&=~(uint64_t(1)<<63);
        integer D{sbn::integer_view{dw}},Q{sbn::integer_view{qw}},R{sbn::integer_view{rw}};
        const integer N=D*Q+R;
        for(unsigned sign=0;sign<4;++sign){auto n=N,d=D;if(sign&1)n.negate();if(sign&2)d.negate();
            for(auto mode:{Mode::trunc,Mode::floor,Mode::ceil,Mode::euclidean}){
                auto expectq=Q,expectr=R;const bool bump=adjust(mode,n.negative(),d.negative(),!R.zero());
                if(bump){expectq+=1;expectr=D-R;}if(n.negative()!=d.negative())expectq.negate();if(n.negative()!=bump)expectr.negate();
                auto out=sbn::divrem(n,d,mode);assert(out.quotient==expectq&&out.remainder==expectr);
                if(dn>1025)continue;
                integer q,r;q.reserve(n.size()+1);r.reserve(d.size());
                auto need=sbn::divrem_requirements(q.ref(),r.ref(),n.view(),d.view(),mode);assert(need.supported());
                sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);
                allocation_watch_start();sbn::divrem_into(q.ref(),r.ref(),n.view(),d.view(),work.on(),mode);assert(!allocation_watch_stop());
                assert(q==expectq&&r==expectr);
                auto alias=n;alias.reserve(n.size()+1);
                need=sbn::divrem_requirements(alias.ref(),r.ref(),alias.view(),d.view(),mode);assert(need.supported());work.reserve(need.scratch_bytes,need.scratch_alignment);
                allocation_watch_start();sbn::divrem_into(alias.ref(),r.ref(),alias.view(),d.view(),work.on(),mode);assert(!allocation_watch_stop());
                assert(alias==expectq&&r==expectr);
            }
        }
        // A retained divisor snapshots its value, including its sign. No
        // source lifetime or value check is needed on subsequent execution.
        for(bool negative:{false,true}){
            auto source=negative?-D:D;sbn::prepared_divisor prepared(source,N.size(),4);source=0;
            auto moved=std::move(prepared);assert(!prepared.ready()&&moved.ready());
            for(auto mode:{Mode::trunc,Mode::floor,Mode::ceil,Mode::euclidean}){
                const auto want=sbn::divrem(-N,negative?-D:D,mode);auto result=moved.divrem(-N,mode);
                assert(result.quotient==want.quotient&&result.remainder==want.remainder);
                integer q,r;q.reserve(N.size()+1);r.reserve(D.size());auto numer=-N;
                allocation_watch_start();moved.execute(q.ref(),r.ref(),numer.view(),{},mode);assert(!allocation_watch_stop());
                assert(q==want.quotient&&r==want.remainder);
                assert(moved.divide(numer,mode)==q&&moved.remainder(numer,mode)==r);
                numer.reserve(N.size()+1);const auto needs=moved.query(numer.ref(),r.ref(),numer.view(),mode);assert(needs.supported());
                sbn::workspace w;w.reserve(needs.scratch_bytes,needs.scratch_alignment);
                allocation_watch_start();moved.execute(numer.ref(),r.ref(),numer.view(),w.on(),mode);assert(!allocation_watch_stop());assert(numer==q&&r==want.remainder);
            }
        }
        if(dn<=1025){const auto need=sbn::prepared_divisor::storage_requirements(D.size(),N.size());assert(need.supported());
            sbn::workspace store;store.reserve(need.scratch_bytes,need.scratch_alignment);
            allocation_watch_start();auto prepared=sbn::prepared_divisor::bind(D.view(),N.size(),store.on().scratch);assert(!allocation_watch_stop());assert(prepared.divide(N)==Q);}
        assert(N/D==Q&&N%D==R);assert(D/D==1&&D%D==0);
        assert(sbn::divrem(1,D,Mode::floor).quotient==0);
        assert(sbn::divrem(-1,D,Mode::floor).remainder==D-1);
    }
    // Rounded quotient can gain a limb when D is a whole radix power.
    const integer B=integer(1)<<128,N=B-1,D=integer(1)<<64;
    auto rounded=sbn::divrem(N,D,Mode::ceil);assert(rounded.quotient==D&&rounded.remainder==-1);
    auto x=N;x/=D;assert(x==D-1);x=N;x%=D;assert(x==D-1);
    sbn::fixed_integer<0> none;auto rejected=sbn::divrem_requirements(none.ref(),std::nullopt,N.view(),D.view());assert(!rejected.supported());
    puts("C++ signed division modes/paired-demand/aliases/exactness/local/general PASS");
}
