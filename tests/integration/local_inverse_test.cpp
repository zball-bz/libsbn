#include "product_support.hpp"
#include "algorithms/local_inverse.hpp"
#include "value/limbs.hpp"
#include "runtime/scratch.hpp"
#include <algorithm>
#include <cmath>
#include <set>
using namespace sbn::v3;
// Keep the independent full-width certificate affordable in the extended
// FFT band. Reference multiplication remains outside allocation/timing gates.
void certificate_product(uint64_t *z,const uint64_t *u,const uint64_t *d,size_t n){
    if(n<=8192){sbn3_mul_basecase(z,2*n+1,u,n+1,d,n);return;}
    ref_int a,b,p;ref_inits(a,b,p,nullptr);
    ref_import(a,n+1,-1,8,0,0,u);ref_import(b,n,-1,8,0,0,d);ref_mul(p,a,b);
    memset(z,0,(2*n+1)*8);ref_export(z,nullptr,-1,8,0,0,p);
    ref_clears(a,b,p,nullptr);
}
void cyclic_lift_gate(){
    ref_int modulus,power,value,residue,actual;
    ref_inits(modulus,power,value,residue,actual,nullptr);
    size_t checked=0;
    for(size_t r:{2ul,3ul,4ul,8ul,17ul,33ul}){
        std::vector<uint64_t> magnitude(r),z(r);
        ref_set_ui(modulus,1);ref_mul_2exp(modulus,modulus,64*r);ref_sub_ui(modulus,modulus,1);
        for(size_t e=0;e<r;++e){
            ref_set_ui(power,1);ref_mul_2exp(power,power,64*e);
            for(unsigned pattern=0;pattern<6;++pattern)for(unsigned neg=0;neg<2;++neg){
                for(size_t j=0;j<r;++j)magnitude[j]=pattern==0?0:pattern<4?UINT64_MAX:random_word();
                magnitude[r-1]&=UINT64_MAX>>1;
                if(pattern==1){std::fill(magnitude.begin(),magnitude.end(),0);magnitude[0]=1;}
                if(pattern==2){std::fill(magnitude.begin(),magnitude.end(),0);magnitude[e]=1;}
                ref_import(value,r,-1,8,0,0,magnitude.data());
                if(neg)ref_neg(value,value);
                ref_add(residue,value,power);ref_mod(residue,residue,modulus);
                const bool zero=ref_sgn(residue)==0;
                for(unsigned representation=0;representation<1u+unsigned(zero);++representation){
                    std::fill(z.begin(),z.end(),0);
                    if(representation)std::fill(z.begin(),z.end(),UINT64_MAX);
                    else ref_export(z.data(),nullptr,-1,8,0,0,residue);
                    allocation_watch_start();
                    limbs::cyclic_sub_power(z.data(),r,e);
                    const bool negative=limbs::cyclic_absolute(z.data(),r);
                    assert(!allocation_watch_stop());
                    assert(negative==(ref_sgn(value)<0));
                    assert(z==magnitude);++checked;
                }
            }
        }
    }
    ref_clears(modulus,power,value,residue,actual,nullptr);
    printf("cyclic signed lift: %zu reference cases, borrow/sign/zero representations PASS\n",checked);
}
int main(){
    cyclic_lift_gate();
    std::set<size_t> sizes;
    for(size_t n=1;n<=64;++n)sizes.insert(n);
    for(unsigned j=24;j<=56;++j)sizes.insert(size_t(std::llround(std::exp2(j/4.))));
    for(size_t n:{767u,768u,769u,790u,825u,832u,1020u,1021u,1022u,1023u,1024u,
                  1277u,1278u,1279u,1533u,1534u,1535u,1660u,1661u,1662u,8193u,12289u})sizes.insert(n);
    unsigned checked=0;
    for(size_t n:sizes){
        Fixture f(1,false);
        const auto bytes=local_inverse_bytes(n),approximate_bytes=local_inverse_approximate_bytes(n);
        auto lease=f.allocate(std::max(bytes,approximate_bytes),64);
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
            certificate_product(z,u,d,n);
            assert(!z[2*n]);
            bool below=false;
            for(size_t j=2*n;j-->0 && !below;){
                const auto r=~z[j],dv=j<n?d[j]:0;
                if(r!=dv){assert(r<dv);below=true;}
            }
            assert(below);++checked;
            auto approximate_work=Frame::borrow(*f.arena,lease,lease.data,approximate_bytes);
            allocation_watch_start();
            {ComputeLease computing(*f.arena);local_inverse_approximate(u,d,n,approximate_work);}
            assert(!allocation_watch_stop() && !approximate_work.used() && approximate_work.peak()<=approximate_bytes && u[n]==1);
            // Independent full-width certificate, avoiding division or the
            // implementation's recurrence: |U*D-B^(2n)| < 3*D.
            certificate_product(z,u,d,n);
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
