#include "divrem_support.hpp"
#include "runtime/scratch.hpp"
#include "algorithms/small_division.hpp"
int main(){
    // Rounding an odd divisor must not add an entire fourth FFT block to
    // a 3/2-divisor-length quotient. Check both parities over the domain.
    for(size_t dn=64;dn<=sbn::v3::local_division_max_limbs;++dn){
        const size_t q=(3*dn+1)/2,m=sbn::v3::division_block_limbs(q,dn);
        assert(m<=(dn+1)/2 && (q+m-1)/m==3);
    }
    size_t count=0;
    constexpr uint64_t guard=0xab1234fe986743edULL;
    for(size_t dn=1;dn<=512;++dn){
        for(size_t nn:{size_t(0),dn-1,dn,dn+1,dn+3,dn+7,dn+15,2*dn-1,2*dn,2*dn+1}){
            for(unsigned pattern=0;pattern<4;++pattern){
                const size_t qcount=nn>=dn?nn-dn+1:0;
                std::vector<uint64_t> d(dn),n(nn);
                for(auto &v:d)v=random_word();for(auto &v:n)v=random_word();
                if(pattern==0)d.back()|=uint64_t(1)<<63;
                if(pattern==1)d.back()=(d.back()>>(dn%64))|1;
                if(pattern==2){std::fill(d.begin(),d.end(),UINT64_MAX);std::fill(n.begin(),n.end(),UINT64_MAX);}
                if(pattern==3)d.back()=(d.back()>>(dn%64))|1;
                if(nn && pattern==1)std::fill(n.begin()+nn/2,n.end(),0);
                const auto before_n=n,before_d=d;
                allocation_watch_start();const size_t bytes=sbn3_divrem_small_scratch_bytes(nn,dn);assert(!allocation_watch_stop());
                const size_t offset=1+dn%8,words=(bytes+7)/8;
                std::vector<uint64_t> scratch(words+offset+8,guard),q(qcount+2,guard),r(dn+2,guard);
                allocation_watch_start();const auto qn=sbn3_divrem_small(q.data()+1,r.data()+1,n.data(),nn,d.data(),dn,bytes?scratch.data()+offset:nullptr);
                assert(!allocation_watch_stop());const size_t rn=trim(r.data()+1,dn);
                certify(n.data(),nn,d.data(),dn,q.data()+1,qn,r.data()+1,rn,true);
                assert(qn==trim(q.data()+1,qcount)&&n==before_n&&d==before_d);
                assert(q.front()==guard&&q.back()==guard&&r.front()==guard&&r.back()==guard);
                for(size_t j=0;j<offset;++j)assert(scratch[j]==guard);
                for(size_t j=offset+words;j<scratch.size();++j)assert(scratch[j]==guard);
                ++count;
            }
        }
    }
    // Every possible first-quotient bit count, including the 52/64-bit
    // funnel seams, with both a general divisor and a power of two.
    for(unsigned bits=1;bits<832;++bits)for(unsigned pattern=0;pattern<2;++pattern){
        constexpr size_t dn=64;const size_t total=64*dn+832+bits,nn=(total+63)/64,qn=nn-dn+1;
        std::vector<uint64_t> d(dn),n(nn),q(qn),r(dn);
        if(!pattern)for(auto &word:d)word=random_word();d.back()|=uint64_t(1)<<63;
        for(auto &word:n)word=random_word();const unsigned last=unsigned(total%64);
        if(last)n.back()=(n.back()&((uint64_t(1)<<last)-1))|(uint64_t(1)<<(last-1));else n.back()|=uint64_t(1)<<63;
        std::vector<uint64_t> scratch((sbn3_divrem_small_scratch_bytes(nn,dn)+7)/8);
        allocation_watch_start();const auto size=sbn3_divrem_small(q.data(),r.data(),n.data(),nn,d.data(),dn,scratch.data());assert(!allocation_watch_stop());
        certify(n.data(),nn,d.data(),dn,q.data(),size,r.data(),trim(r.data(),dn),true);++count;
    }
    // External Frame descendants must remain valid without an arena lease.
    alignas(64) unsigned char storage[4096];
    {auto root=sbn::v3::Frame::external(storage,sizeof storage);auto mark=root.mark();
     {auto child=root.subframe(1024);auto*p=child.alloc<uint64_t>(8);p[7]=19;
      {auto nested=child.borrowed_view(p,64);assert(nested.data()==reinterpret_cast<uint8_t*>(p));}}
     root.rewind(mark);assert(!root.used());}
    printf("small division: %zu exact cases, caller scratch offsets, readonly inputs, guards and no allocations PASS\n",count);
}
