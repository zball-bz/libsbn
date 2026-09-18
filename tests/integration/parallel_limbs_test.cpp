#include "product_support.hpp"
#include "value/parallel_limbs.hpp"
namespace pl=sbn::v3::parallel_limbs;
static void test(size_t n,unsigned workers){
    Fixture f(workers);auto *a=f.guarded(n),*b=f.guarded(n),*r=f.guarded(n),*ref=f.guarded(n);
    for(unsigned pattern=0;pattern<5;++pattern){
        for(size_t j=0;j<n;++j){a[j]=pattern==0?random_word():pattern==1?UINT64_MAX:pattern==2?0:pattern==3?UINT64_MAX:0;
            b[j]=pattern<=2?a[j]:(j==0?1:0);}
        for(size_t sn:{n,n-7}){
            memcpy(r,a,n*8);memcpy(ref,a,n*8);auto c=ref_add_n(ref,ref,b,sn);
            for(size_t j=sn;c&&j<n;++j)c=++ref[j]==0;
            allocation_watch_start();auto got=pl::add_to(f.team,r,n,b,sn);assert(!allocation_watch_stop());assert(got==c&&!memcmp(r,ref,n*8));
            memcpy(r,a,n*8);
            struct Scoped {uint64_t *r;const uint64_t *a;size_t n,an;uint64_t carry;} scoped{r,b,n,sn,0};
            allocation_watch_start();sbn3_team_run(f.team,[](void *p,sbn3_team_scope *s){auto &j=*static_cast<Scoped *>(p);j.carry=pl::add_to_scope(s,j.r,j.n,j.a,j.an);},&scoped);assert(!allocation_watch_stop());
            assert(scoped.carry==c&&!memcmp(r,ref,n*8));
            memcpy(r,a,n*8);memcpy(ref,a,n*8);c=ref_sub_n(ref,ref,b,sn);
            for(size_t j=sn;c&&j<n;++j){c=ref[j]==0;--ref[j];}
            allocation_watch_start();got=pl::sub_from(f.team,r,n,b,sn);assert(!allocation_watch_stop());assert(got==c&&!memcmp(r,ref,n*8));
        }
        for(uint64_t word:{uint64_t(0),uint64_t(1),uint64_t(2),uint64_t(3),UINT64_MAX}){
            const auto c=ref_mul_1(ref,a,n,word);
            allocation_watch_start();auto got=pl::mul_1(f.team,r,a,n,word);assert(!allocation_watch_stop());assert(got==c&&!memcmp(r,ref,n*8));
            memcpy(r,a,n*8);allocation_watch_start();got=pl::mul_1(f.team,r,r,n,word);assert(!allocation_watch_stop());assert(got==c&&!memcmp(r,ref,n*8));
        }
        memcpy(r,a,n*8);memcpy(ref,a,n*8);
        sbn::v3::limbs::cyclic_sub_shifted(ref,n,b,n-3,n/2+1);
        allocation_watch_start();pl::cyclic_sub_shifted(f.team,r,n,b,n-3,n/2+1);assert(!allocation_watch_stop());assert(!memcmp(r,ref,n*8));
        allocation_watch_start();pl::copy(f.team,r,a,n);pl::complement(f.team,r,n);assert(!allocation_watch_stop());
        for(size_t j=0;j<n;++j)assert(r[j]==~a[j]);
    }
    for(size_t input:{size_t(0),n-7,n})for(size_t output:{n-3,n})for(unsigned bits:{0u,1u,17u,63u}){
        memcpy(r,a,n*8);
        for(size_t j=0;j<output;++j)ref[j]=j<input?(!bits?a[j]:(a[j]>>bits)|((j+1<input?a[j+1]:0)<<(64-bits))):0;
        allocation_watch_start();pl::right_shift(f.team,r,output,input,bits);assert(!allocation_watch_stop());
        assert(!memcmp(r,ref,output*8));for(size_t j=output;j<n;++j)assert(r[j]==a[j]);
    }
    allocation_watch_start();pl::fill(f.team,r,n);assert(pl::zero(f.team,r,n));r[n-1]=1;assert(!pl::zero(f.team,r,n));assert(!allocation_watch_stop());
    printf("parallel limbs n=%zu W%u: reference/modular oracle carry/borrow, long seams, alias, wrap, zero, no allocation PASS\n",n,workers);
}
int main(){test(31,1);test((1u<<20)-1,16);for(unsigned w:{4u,16u}){test((1u<<18)+3,w);test((1u<<20)+15,w);}puts("parallel limbs gates PASS");}
