// Original libsbn tests; LGPL-3.0-or-later.
#include "divrem_support.hpp"
#include "algorithms/small_division.hpp"
#include "runtime/scratch.hpp"
using namespace sbn::v3;
int main(){
    constexpr uint64_t guard=0x95bac7420176de38ULL;
    size_t count=0;
    for(size_t dn:{3u,65u,513u,5793u,8192u,32768u})for(unsigned pattern=0;pattern<3;++pattern){
        const auto p=local_division_plan_for_block(8*dn+7,dn,(dn+1)/2);
        const size_t bytes=local_division_remainder_bytes(p),offset=dn%8+1;
        std::vector<uint64_t>d(dn),memory((bytes+7)/8+offset+8,guard);
        for(auto &v:d)v=pattern==1?UINT64_MAX:random_word();
        d.back()=pattern==2?(d.back()&7)|1:d.back()|(uint64_t(1)<<63);
        for(size_t nn:{dn-1,dn,dn+1,2*dn-1,2*dn,8*dn+7}){
            std::vector<uint64_t>n(nn),q(nn>=dn?nn-dn+1:0),r(dn),discard(dn+2,guard);
            for(auto &v:n)v=pattern==1?UINT64_MAX:random_word();const auto saved=n,dd=d;
            {
                auto f=Frame::external(memory.data()+offset,p.storage_bytes);
                allocation_watch_start();local_division_execute(p,q.data(),r.data(),n.data(),nn,d.data(),f);assert(!allocation_watch_stop());
                assert(!f.used()&&f.peak()<=p.storage_bytes);
            }
            certify(n.data(),nn,d.data(),dn,q.data(),trim(q.data(),q.size()),r.data(),trim(r.data(),dn),dn<=513);
            {
                auto f=Frame::external(memory.data()+offset,bytes);
                allocation_watch_start();local_division_execute(p,nullptr,discard.data()+1,n.data(),nn,d.data(),f);assert(!allocation_watch_stop());
                assert(!f.used()&&f.peak()<=bytes);
            }
            assert(std::equal(r.begin(),r.end(),discard.begin()+1)&&n==saved&&d==dd);
            assert(discard.front()==guard&&discard.back()==guard);
            for(size_t j=0;j<offset;++j)assert(memory[j]==guard);
            for(size_t j=offset+(bytes+7)/8;j<memory.size();++j)assert(memory[j]==guard);
            ++count;
        }
    }
    for(size_t dn:{8192u,16384u}){
        const size_t nn=dn+dn/4;const auto p=local_fused_division_plan(nn,dn);
        assert(local_division_remainder_bytes(p)==p.storage_bytes);
        std::vector<uint64_t>d(dn),n(nn),q(nn-dn+1),r(dn),discard(dn),memory(p.storage_bytes/8+16);
        for(auto &v:d)v=random_word();for(auto &v:n)v=random_word();d.back()|=uint64_t(1)<<63;
        auto f=Frame::external(memory.data(),p.storage_bytes);
        local_division_execute(p,q.data(),r.data(),n.data(),nn,d.data(),f);
        certify(n.data(),nn,d.data(),dn,q.data(),trim(q.data(),q.size()),r.data(),trim(r.data(),dn),false);
        allocation_watch_start();local_division_execute(p,nullptr,discard.data(),n.data(),nn,d.data(),f);assert(!allocation_watch_stop());
        assert(r==discard&&!f.used());++count;
    }
    printf("local remainder %zu exact/certified division comparisons, bounds and no allocation PASS\n",count);
}
