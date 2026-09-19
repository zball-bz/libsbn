#include "product_support.hpp"
#include "algorithms/local_inverse.hpp"
#include "runtime/scratch.hpp"
#include <cmath>
#include <set>
using namespace sbn::v3;
int main(){
    std::set<size_t> sizes;
    for(size_t n=1;n<=64;++n)sizes.insert(n);
    for(unsigned j=24;j<=52;++j)sizes.insert(size_t(std::llround(std::exp2(j/4.))));
    unsigned checked=0;
    for(size_t n:sizes){
        Fixture f(1,false);
        const auto bytes=local_inverse_bytes(n);
        auto lease=f.allocate(bytes,64);
        Frame scratch(*f.arena,lease);
        auto *d=f.guarded(n),*u=f.guarded(n+1),*z=f.guarded(2*n+1),*limit=f.guarded(n+1);
        for(unsigned pattern=0;pattern<5;++pattern){
            for(size_t j=0;j<n;++j)d[j]=pattern==0?UINT64_MAX:pattern<3?0:random_word();
            d[n-1]|=uint64_t(1)<<63;
            if(pattern==2)d[0]|=1;
            memset(u,0xa5,(n+1)*8);
            allocation_watch_start();
            {ComputeLease computing(*f.arena);local_inverse(u,d,n,scratch);}
            assert(!allocation_watch_stop() && !scratch.used() && scratch.peak()<=bytes && u[n]==1);
            // Independent u64 schoolbook product: U*D <= all-ones numerator,
            // whose remainder must be strictly smaller than D.
            sbn3_mul_basecase(z,2*n+1,u,n+1,d,n);
            assert(!z[2*n]);
            bool below=false;
            for(size_t j=2*n;j-->0 && !below;){
                const auto r=~z[j],dv=j<n?d[j]:0;
                if(r!=dv){assert(r<dv);below=true;}
            }
            assert(below);++checked;
            const auto approximate_bytes=local_inverse_approximate_bytes(n);
            assert(approximate_bytes<=bytes);
            auto approximate_work=Frame::borrow(*f.arena,lease,lease.data,approximate_bytes);
            allocation_watch_start();
            {ComputeLease computing(*f.arena);local_inverse_approximate(u,d,n,approximate_work);}
            assert(!allocation_watch_stop() && !approximate_work.used() && approximate_work.peak()<=approximate_bytes && u[n]==1);
            // Independent full-width certificate, avoiding division or the
            // implementation's recurrence: |U*D-B^(2n)| < 3*D.
            sbn3_mul_basecase(z,2*n+1,u,n+1,d,n);
            if(z[2*n]){assert(z[2*n]==1);--z[2*n];}
            else{
                for(size_t j=0;j<2*n;++j)z[j]=~z[j];
                uint64_t carry=1;
                for(size_t j=0;j<2*n && carry;++j){++z[j];carry=z[j]==0;}
                assert(!carry);
            }
            for(size_t j=n+1;j<=2*n;++j)assert(!z[j]);
            uint64_t carry=0;
            for(size_t j=0;j<n;++j){unsigned __int128 v=(unsigned __int128)d[j]*3+carry;limit[j]=uint64_t(v);carry=uint64_t(v>>64);}
            limit[n]=carry;below=false;
            for(size_t j=n+1;j-->0 && !below;)if(z[j]!=limit[j]){assert(z[j]<limit[j]);below=true;}
            assert(below);
        }
    }
    printf("local inverse exact quotient/remainder and bounded 3-ulp certificates, framing, guards, allocation and scratch: %u cases each PASS\n",checked);
}
