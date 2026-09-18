// Right-angle recipe gate: every pre-rounding coefficient against the exact
// product (reference/modular oracle, 64-bit digit slots), full reference/modular oracle product, read-only tables, no
// implicit allocation, no workspace leak, guard pages. Patterns include the
// saturated, alternating, sparse, half-saturated and top-bit inputs plus
// unbalanced lengths and full-capacity shapes at every radix up to N=32768.
#include <cstddef>
static void rac_coefficient(size_t,double);
#define RAC_COEFFICIENT_OBSERVER ::rac_coefficient
#include "backend/pq16/island.hpp"
#include <new>
#include "backend/pq16/ct.hpp"
#include "backend/pq16/rac.hpp"
#include "../integration/product_support.hpp"
#include <sys/mman.h>
#include <cmath>
using namespace sbn::v3;using namespace sbn::v3::pq16;
static std::vector<uint64_t> rac_exact;static std::vector<unsigned char> rac_seen;static double rac_error;
static void rac_coefficient(size_t k,double x){
    assert(k<rac_exact.size()&&!rac_seen[k]++&&std::isfinite(x));const double e=std::abs(x-double(rac_exact[k]));rac_error=std::max(rac_error,e);assert(e<.5);
}
template<unsigned M>static void rac_one(unsigned n,size_t an,size_t bn,unsigned pattern){
    const unsigned N=M*n;Shape shape{N,n,M,false,Recipe::RightAngle,16};Fixture f(1);
    auto tl=f.allocate(table_bytes(shape),128),wl=f.allocate(scratch_bytes(shape,an,bn),128);Frame tf(*f.arena,tl),wf(*f.arena,wl);
    pq16_plan core{};core.builder=&tf;pq16_plan_ensure(&core,n,1);allocation_watch_start();auto *t=rac_prepare(tf,core,shape);assert(!allocation_watch_stop());
    auto *a=f.guarded(an),*b=f.guarded(bn),*r=f.guarded(an+bn);
    for(unsigned side=0;side<2;++side){auto *p=side?b:a;size_t len=side?bn:an;
        for(size_t j=0;j<len;++j)p[j]=pattern==0?UINT64_MAX:pattern==1?random_word():pattern==2?(j&1?0:UINT64_MAX):pattern==3?(j==0||j==len-1?UINT64_MAX:0):pattern==4?(j<len/2?UINT64_MAX:0):0x8000800080008000ull;}
    std::vector<uint64_t> da(4*an),db(4*bn);for(size_t j=0;j<da.size();++j)da[j]=(a[j/4]>>(16*(j%4)))&65535;for(size_t j=0;j<db.size();++j)db[j]=(b[j/4]>>(16*(j%4)))&65535;
    ref_int A,B,P;ref_inits(A,B,P,nullptr);ref_import(A,da.size(),-1,8,0,0,da.data());ref_import(B,db.size(),-1,8,0,0,db.data());ref_mul(P,A,B);
    rac_exact.assign(2*size_t(N),0);size_t words=0;ref_export(rac_exact.data(),&words,-1,8,0,0,P);ref_clears(A,B,P,nullptr);rac_seen.assign(rac_exact.size(),0);rac_error=0;
    SBN3_FRAME_UNPOISON(wf.data(),wf.capacity());memset(wf.data(),0xff,wf.capacity());SBN3_FRAME_POISON(wf.data(),wf.capacity());
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ));
    allocation_watch_start();const bool ok=rac_multiply<M>(r,a,an,b,bn,*t,wf);assert(ok&&!allocation_watch_stop()&&!wf.used());
    for(auto v:rac_seen)assert(v==1);verify_product(a,an,b,bn,r);
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ|PROT_WRITE));
    assert(tf.peak()<=table_bytes(shape)&&wf.peak()<=scratch_bytes(shape,an,bn));
    printf("RAC M%u N%u %zu x %zu pattern %u: %zu exact coefficients, max error %.9g PASS\n",M,N,an,bn,pattern,rac_exact.size(),rac_error);
}
template<unsigned M>static void rac_band(unsigned n){
    const size_t full=size_t(M)*n/4;
    for(unsigned p=0;p<6;++p)rac_one<M>(n,full,full,p);
    rac_one<M>(n,full+full/2,full/2,1);rac_one<M>(n,2*full-1,1,0);rac_one<M>(n,full/3,full/3,0);rac_one<M>(n,full-7,full-9,2);
}
extern "C" void test_pq16_rac(){
    rac_band<3>(128);rac_band<3>(1024);rac_band<3>(2048);rac_band<3>(4096);rac_band<3>(8192);
    rac_band<5>(128);rac_band<5>(512);rac_band<5>(1024);rac_band<5>(2048);rac_band<5>(4096);
    rac_band<7>(128);rac_band<7>(512);rac_band<7>(1024);rac_band<7>(2048);rac_band<7>(4096);
    // Recipe policy: 16-bit odd radix at W1 is right-angle; pow2, wider teams and the >32768 band are not.
    assert(execution_shape(query(609,609),1).recipe==Recipe::RightAngle);
    assert(execution_shape(query(4467,4467),1).recipe==Recipe::RightAngle);
    assert(execution_shape(query(6889,6889),1).recipe==Recipe::PfaPQ&&execution_shape(query(5547,5547),1).recipe==Recipe::PfaPQ);
    assert(execution_shape(query(609,609),2).recipe==Recipe::PfaPQ);
    assert(execution_shape(query(512,512),1).recipe==Recipe::PfaPQ);
    assert(execution_shape(query(9000,9000),1).recipe==Recipe::PfaPQ);
    assert(select(1218,1218,1).recipe==Recipe::RightAngle&&select(2435,2435,1).recipe==Recipe::RightAngle);
    {const Shape s=select(609,609,1);assert(s.bits==20&&s.balanced&&s.nfull==2048);}   // 2026-09-09: 20-bit balanced pow2 2048 beats 16-bit M5 2560 right-angle by 6.6 %
    puts("right-angle recipe gate PASS");
}
