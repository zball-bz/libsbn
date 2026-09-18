#include <cstddef>
static void ct_coefficient(size_t,const double *,const double *);
#define PQ16_COEFFICIENT_OBSERVER ::ct_coefficient
#include "backend/pq16/island.hpp"
#include <new>
#include "backend/pq16/ct.hpp"
#include "../integration/product_support.hpp"
#include <sys/mman.h>
#include <cmath>
#include <complex>
using namespace sbn::v3;using namespace sbn::v3::pq16;
static std::vector<uint64_t> ct_exact;static std::vector<unsigned char> ct_seen;static double ct_error;
static void ct_coefficient(size_t limb,const double *a,const double *b){
    for(unsigned t=0;t<2;++t)for(unsigned l=0;l<8;++l)for(unsigned p=0;p<2;++p){
        const size_t k=4*limb+16*t+2*l+p;const double x=(t?b:a)[l+8*p];
        assert(k<ct_exact.size()&&!ct_seen[k]++&&std::isfinite(x));
        const double err=std::abs(x-double(ct_exact[k]));ct_error=std::max(ct_error,err);assert(err<.5);
    }
}
static void ct_one(size_t an,size_t bn,unsigned pattern){
    auto shape=query(an,bn);assert(shape.nfull<=32768&&shape.radix!=1&&!shape.centered);shape.recipe=Recipe::CooleyTukeyPQ;Fixture f(1);
    auto tl=f.allocate(table_bytes(shape),128),wl=f.allocate(scratch_bytes(shape,an,bn),128);Frame tf(*f.arena,tl),wf(*f.arena,wl);
    pq16_plan core{};core.builder=&tf;pq16_plan_ensure(&core,shape.branch,1);auto *table=ct_prepare(tf,core,shape);
    auto *a=f.guarded(an),*b=f.guarded(bn),*r=f.guarded(an+bn);
    for(unsigned side=0;side<2;++side){auto *p=side?b:a;size_t len=side?bn:an;
        for(size_t j=0;j<len;++j)p[j]=pattern==0?UINT64_MAX:pattern==1?random_word():pattern==2?(j&1?0:UINT64_MAX):(j==0||j==len-1?UINT64_MAX:0);}
    std::vector<uint64_t> da(4*an),db(4*bn);for(size_t j=0;j<da.size();++j)da[j]=(a[j/4]>>(16*(j%4)))&65535;
    for(size_t j=0;j<db.size();++j)db[j]=(b[j/4]>>(16*(j%4)))&65535;
    ref_int A,B,P;ref_inits(A,B,P,nullptr);ref_import(A,da.size(),-1,8,0,0,da.data());ref_import(B,db.size(),-1,8,0,0,db.data());ref_mul(P,A,B);
    ct_exact.assign(2*shape.nfull,0);size_t words=0;ref_export(ct_exact.data(),&words,-1,8,0,0,P);ref_clears(A,B,P,nullptr);
    ct_seen.assign(ct_exact.size(),0);ct_error=0;
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ));
    allocation_watch_start();bool ok=false;
    switch(shape.radix){case 3:ok=ct_multiply<3>(r,a,an,b,bn,*table,wf);break;case 5:ok=ct_multiply<5>(r,a,an,b,bn,*table,wf);break;case 7:ok=ct_multiply<7>(r,a,an,b,bn,*table,wf);break;}
    assert(!allocation_watch_stop()&&ok&&!wf.used());for(auto n:ct_seen)assert(n==1);verify_product(a,an,b,bn,r);
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ|PROT_WRITE));
    printf("CT-PQ %zu x %zu M%u: %zu exact coefficients, max error %.9g PASS\n",an,bn,shape.radix,ct_exact.size(),ct_error);
}
template<unsigned M>static void ct_spectrum(unsigned n,unsigned digit){
    Fixture f(1);Shape s{M*n,n,M,false,Recipe::CooleyTukeyPQ};auto tl=f.allocate(table_bytes(s),128),dl=f.allocate(16*s.nfull,128);Frame tf(*f.arena,tl);
    pq16_plan core{};core.builder=&tf;pq16_plan_ensure(&core,n,1);auto *p=ct_prepare(tf,core,s);auto *out=static_cast<double *>(dl.data);
    std::vector<uint64_t> a((digit+4)/4);a[digit/4]=1ull<<(16*(digit%4));ct_forward<M>(out,a.data(),a.size(),*p);
    for(unsigned b=0;b<M;++b)for(unsigned r=0;r<n;++r){unsigned k=b+M*pq16_bitrev(r,__builtin_ctz(n));size_t at=2*size_t(b)*n+128*(r/64)+16*(r%8)+(r%64)/8;
        long double angle=0xc.90fdaa22168c235p-1L*k*(digit/2)/(M*n);std::complex<long double> want(cosl(angle),sinl(angle));if(digit&1)want*=std::complex<long double>(0,1);
        assert(std::abs(std::complex<long double>(out[at],out[at+8])-want)<1e-11L);
    }
}
extern "C" void test_pq16_ct(){
    for(unsigned d:{0u,1u,2u,3u,31u,511u}){ct_spectrum<3>(128,d);ct_spectrum<5>(128,d);ct_spectrum<7>(128,d);}
    ct_spectrum<5>(512,5101);
    for(size_t n:{558u,609u,664u,724u,790u,861u,1218u,1579u,2435u,2656u,3444u,4467u,4871u,5793u})for(unsigned p=0;p<4;++p)ct_one(n,n,p);
    ct_one(1024,1,0);ct_one(10239,1,1);ct_one(12287,1,0);ct_one(7167,1,0);
    ct_one(1024,255,1);ct_one(8192,2047,0);
    // Recipe policy moved to the right-angle gate (2026-09-08); CT/PQ stays selectable and wide-codec default.
    assert(execution_shape(query(512,512),1).recipe==Recipe::PfaPQ);
    assert(execution_shape(query(609,609),2).recipe==Recipe::PfaPQ);
    assert(execution_shape(query(9000,9000),1).recipe==Recipe::PfaPQ);
}
