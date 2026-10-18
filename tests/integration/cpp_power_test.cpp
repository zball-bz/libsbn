// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/power.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;
int main(){
    assert(sbn::pow(0,0)==1&&sbn::pow(0,5)==0&&sbn::pow(-1,1001)==-1);
    for(int a=-17;a<=17;++a){integer reference=1;for(unsigned e=0;e<=80;++e){auto got=sbn::pow(a,e);assert(got==reference);reference*=a;}}
    for(std::size_t words:{3u,7u,33u,161u,513u,2049u}){
        std::vector<uint64_t>w(words);uint64_t state=749201563;for(auto&v:w){state^=state<<13;state^=state>>7;state^=state<<17;v=state;}w.back()|=1ULL<<63;
        integer a{sbn::integer_view{w,true}};
        for(unsigned e:{2u,3u,5u,8u,13u}){
            integer reference=1;for(unsigned j=0;j<e;++j)reference*=a;
            auto got=sbn::pow(a,e);assert(got==reference);
            integer out;out.reserve(words*e+1);const auto need=sbn::power_requirements(out.ref(),a.view(),e);assert(need.supported());
            sbn::workspace scratch;scratch.reserve(need.scratch_bytes,need.scratch_alignment);
            allocation_watch_start();sbn::power_into(out.ref(),a.view(),e,scratch.on());assert(!allocation_watch_stop());assert(out==reference);
            auto alias=a;alias.reserve(words*e+1);const auto aneed=sbn::power_requirements(alias.ref(),alias.view(),e);assert(aneed.supported());scratch.reserve(aneed.scratch_bytes,aneed.scratch_alignment);
            allocation_watch_start();sbn::power_into(alias.ref(),alias.view(),e,scratch.on());assert(!allocation_watch_stop());assert(alias==reference);
        }
    }
    for(unsigned shift:{0u,1u,63u,64u,129u,257u})for(unsigned e:{1u,2u,3u,129u})assert(sbn::pow(-(integer(1)<<shift),e)==((e&1)?-(integer(1)<<(shift*e)):(integer(1)<<(shift*e))));
    puts("C++ ordinary power fixed ladder/scalar leaves/squares/signs/alias and admitted storage PASS");
}
