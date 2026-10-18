#include "product_support.hpp"
#include "product/word_window.hpp"
#include <algorithm>
using namespace sbn::v3;
using namespace sbn::v3::product;
int main(){
    unsigned checked=0;
    for(unsigned workers:{1u,4u}){
        Fixture f(workers,false);auto pool=f.allocate(size_t(64)<<20,size_t(1)<<21);
        auto run=[&](const std::vector<uint64_t>&a,const std::vector<uint64_t>&b,uint64_t bit,uint64_t bits,uint64_t expected){
            WordWindowPlan plan;
            allocation_watch_start();const auto status=word_window_query(a.size(),b.size(),bit,workers,plan);
            assert(!allocation_watch_stop()&&status==SBN3_SUPPORTED&&plan.bytes<=pool.bytes);
            auto scratch=Frame::borrow(*f.arena,pool,pool.data,plan.bytes);
            const auto aa=a,bb=b;
            allocation_watch_start();const auto got=word_window_execute(plan,{a.data(),a.size()},{b.data(),b.size()},bit,scratch,f.team,bits);
            assert(!allocation_watch_stop());assert(got==expected&&a==aa&&b==bb);++checked;
        };
        ref_int a,b,p;ref_inits(a,b,p,nullptr);
        for(size_t n:{1u,9u,31u,65u,127u,129u,511u,1025u,4097u})for(size_t bn:{n,n/2+1,size_t(3)}){
            std::vector<uint64_t>x(n),y(bn);for(auto &v:x)v=random_word();for(auto &v:y)v=random_word();
            for(uint64_t bits:{UINT64_MAX,64*n-13}){
                ref_import(a,n,-1,8,0,0,x.data());if(bits!=UINT64_MAX)ref_fdiv_r_2exp(a,a,bits);
                ref_import(b,bn,-1,8,0,0,y.data());ref_mul(p,a,b);
                for(uint64_t bit:{uint64_t(0),uint64_t(63),64*(n/2)+7,64*(n-1),64*n+31,64*(n+bn-1)+13,64*(n+bn)}){
                    ref_fdiv_q_2exp(a,p,bit);run(x,y,bit,bits,ref_get_ui(a));}
            }
        }
        // Large generic-program path, with a closed-form product rather
        // than a quadratic oracle. Also crosses a block boundary in bits.
        for(size_t n:{32771u,65537u}){
            const size_t m=n-31;std::vector<uint64_t>x(n,UINT64_MAX),y(m,UINT64_MAX);
            auto word=[&](size_t k){return k==0?uint64_t(1):k<m?uint64_t(0):k==n?UINT64_MAX-1:k<n+m?UINT64_MAX:uint64_t(0);};
            for(size_t k:{m-1,m,n,n+1,n+m-1})for(unsigned shift:{0u,17u})
                run(x,y,64*k+shift,UINT64_MAX,shift?(word(k)>>shift)|(word(k+1)<<(64-shift)):word(k));
        }
        ref_clears(a,b,p,nullptr);
    }
    printf("exact product word windows: %u masked/random/large structured cases, W1/W4 PASS\n",checked);
}
