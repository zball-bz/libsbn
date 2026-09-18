#include <cstddef>
static void coefficient(size_t,double);
#define RAC_COEFFICIENT_OBSERVER ::coefficient
#include "backend/pq16/island.hpp"
#include "backend/pq16/ct.hpp"
#include "backend/pq16/rac.hpp"
#include "../integration/product_support.hpp"
#include <cmath>
using namespace sbn::v3;
using namespace sbn::v3::pq16;
static std::vector<int64_t> exact;
static std::vector<unsigned char> seen;
static double maximum;
static void coefficient(size_t at,double x){assert(at<exact.size()&&!seen[at]++&&std::isfinite(x));maximum=std::max(maximum,std::abs(x-double(exact[at])));}
template<unsigned M>static void check(unsigned n,unsigned pattern){
    const size_t N=M*n,ring=N/2;Fixture f(1);const Shape s{unsigned(N),n,M,false,Recipe::RightAngle,16,false};
    auto tl=f.allocate(table_bytes(s)+256,128),wl=f.allocate(48*N+2048,128);Frame tf(*f.arena,tl),work(*f.arena,wl);
    pq16_plan pl{};pl.builder=&tf;pq16_plan_ensure(&pl,n,true);auto *t=rac_prepare(tf,pl,s);
    auto *a=f.guarded(ring),*b=f.guarded(ring),*out=f.guarded(ring+1);
    for(unsigned side=0;side<2;++side)for(size_t j=0;j<ring;++j)(side?b:a)[j]=pattern==0?UINT64_MAX:pattern==1?(j%2?0:UINT64_MAX):pattern==2?UINT64_C(0x8000800080008000):random_word();
    std::vector<uint64_t> da(2*N),db(2*N),full(4*N);
    for(size_t j=0;j<2*N;++j){da[j]=(a[j/4]>>(16*(j%4)))&65535;db[j]=(b[j/4]>>(16*(j%4)))&65535;}
    ref_int A,B,P;ref_inits(A,B,P,nullptr);ref_import(A,da.size(),-1,8,0,0,da.data());ref_import(B,db.size(),-1,8,0,0,db.data());ref_mul(P,A,B);
    size_t count=0;ref_export(full.data(),&count,-1,8,0,0,P);ref_clears(A,B,P,nullptr);
    exact.assign(2*N,0);for(size_t j=0;j<4*N;++j)exact[j%(2*N)]+=(j<2*N?int64_t(full[j]):-int64_t(full[j]));seen.assign(2*N,0);maximum=0;
    allocation_watch_start();rac_plus_multiply<M>(out,a,ring,b,ring,false,nullptr,*t,work);assert(!allocation_watch_stop());
    for(auto v:seen)assert(v==1);printf("plus coefficient gate N=%zu M=%u pattern=%u max_error=%.9g\n",N,M,pattern,maximum);fflush(stdout);assert(maximum<.25);
}
int main(){for(unsigned pattern=0;pattern<4;++pattern){check<3>(2048,pattern);check<5>(2048,pattern);check<7>(1024,pattern);}puts("plus ring raw signed coefficient margin PASS");}
