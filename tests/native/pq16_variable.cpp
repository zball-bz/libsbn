#include <cstddef>
static void coefficient(size_t,double);
#define VX_COEFFICIENT_OBSERVER ::coefficient
#include "backend/pq16/island.hpp"
#include "backend/pq16/ct.hpp"
#include "backend/pq16/rac.hpp"
#include "backend/pq16/variable.hpp"
#include "backend/pq16/rac_wide.hpp"
#include "backend/pq16/numeric_domain.hpp"
#include "../integration/product_support.hpp"
#include <cmath>
#include <sys/mman.h>
using namespace sbn::v3;using namespace sbn::v3::pq16;
static std::vector<uint64_t> exact;static std::vector<unsigned char> seen;static double error;static size_t bad;
static void coefficient(size_t k,double x){if(exact.empty())return;assert(k<exact.size()&&!seen[k]++&&std::isfinite(x));double e=std::abs(x-double(exact[k]));error=std::max(error,e);bad+=e>=.5;}
template<unsigned B>static uint64_t digit(const uint64_t *a,size_t n,size_t j){size_t bit=j*B,k=bit/64;unsigned s=bit%64;if(k>=n)return 0;uint64_t v=a[k]>>s;if(s&&k+1<n)v|=a[k+1]<<(64-s);return v&((1u<<B)-1);}
template<unsigned B>static void codec(){
    uint64_t a[64];for(auto &x:a)x=random_word();
    for(unsigned count=0;count<=64;++count)for(unsigned base=0;base<=64;base+=B){
        qcv v[]={vx_decode<B,0,false>(a,count,base),vx_decode<B,1,false>(a,count,base),vx_decode<B,2,false>(a,count,base),vx_decode<B,3,false>(a,count,base)};
        for(unsigned u=0;u<4;++u){alignas(64) double d[16];q_st(d,v[u]);for(unsigned j=0;j<8;++j){assert(d[j]==digit<B>(a,count,(base/B)*64+u*16+2*j));assert(d[j+8]==digit<B>(a,count,(base/B)*64+u*16+2*j+1));}}
        // balanced digits: e_i = d_i - (t_i<<B) + t_{i-1}, t_i = d_i>>(B-1), t_{-1} = 0 at the operand start
        qcv w[]={vx_decode<B,0,true>(a,count,base),vx_decode<B,1,true>(a,count,base),vx_decode<B,2,true>(a,count,base),vx_decode<B,3,true>(a,count,base)};
        for(unsigned u=0;u<4;++u){alignas(64) double d[16];q_st(d,w[u]);for(unsigned j=0;j<16;++j){const size_t i=(base/B)*64+u*16+j;const int64_t di=int64_t(digit<B>(a,count,i)),ti=di>>(B-1),tp=i?int64_t(digit<B>(a,count,i-1)>>(B-1)):0;
            assert(d[j%2?j/2+8:j/2]==double(di-(ti<<B)+tp));}}
    }
    for(unsigned mode=0;mode<4;++mode){Fixture f(1);auto *r=f.guarded(4*B+4);memset(r,0,(4*B+4)*8);q_chain c{};ref_int gold,v,got;ref_inits(gold,v,got,nullptr);
        for(unsigned block=0;block<4;++block){qcv z[4];alignas(64) double d[64];
            for(unsigned k=0;k<64;++k){uint64_t x=mode==0?0:mode==1?(1ull<<49)-1:mode==2?random_word()&((1ull<<49)-1):(k?((1u<<B)-1):(1ull<<40));d[16*(k/16)+(k%16)/2+8*(k%2)]=double(x);ref_set_ui(v,x);ref_mul_2exp(v,v,B*(64*block+k));ref_add(gold,gold,v);}
            for(unsigned k=0;k<4;++k)z[k]=q_ld(d+16*k);const vx_target t{r,4*B+4,nullptr,0,0};vx_emit64<B>(t,block*B,z,c,sb_set1_64(0x4338000000000000LL));}
        assert(q_chains_close(r,4*B+4,&c,1,4*B));ref_import(got,4*B+4,-1,8,0,0,r);assert(!ref_cmp(gold,got));ref_clears(gold,v,got,nullptr);}
    printf("{\"kind\":\"codec\",\"bits\":%u,\"success\":true}\n",B);fflush(stdout);
}
template<unsigned M,unsigned B,bool Rac=false>static void product(unsigned n,unsigned pattern,bool unbalanced=false,unsigned trim=1){
    const unsigned N=M*n;size_t an=size_t(N)*B/64-trim,bn=an;if(unbalanced){an=2*an-1;bn=1;}
    Shape shape{N,n,M,false,Rac?Recipe::RightAngle:Recipe::CooleyTukeyPQ,B};Fixture f(1);auto tl=f.allocate(table_bytes(shape),128),wl=f.allocate(scratch_bytes(shape,an,bn),128);Frame tf(*f.arena,tl),wf(*f.arena,wl);
    pq16_plan core{};core.builder=&tf;pq16_plan_ensure(&core,n,M!=1);const CtTables *t=nullptr;const RacTables *rt=nullptr;
    if constexpr(Rac)rt=rac_prepare(tf,core,shape);else t=ct_prepare(tf,core,shape);
    auto *a=f.guarded(an),*b=f.guarded(bn),*r=f.guarded(an+bn);
    for(unsigned side=0;side<2;++side){auto *p=side?b:a;size_t len=side?bn:an;for(size_t j=0;j<len;++j)p[j]=pattern==0?UINT64_MAX:pattern==1?random_word():pattern==2?(j&1?0:UINT64_MAX):(j==0||j==len-1?UINT64_MAX:0);}
    const size_t na=(64*an+B-1)/B,nb=(64*bn+B-1)/B;std::vector<uint64_t> da(na),db(nb);for(size_t j=0;j<na;++j)da[j]=digit<B>(a,an,j);for(size_t j=0;j<nb;++j)db[j]=digit<B>(b,bn,j);
    ref_int A,D,P,Q;ref_inits(A,D,P,Q,nullptr);ref_import(A,na,-1,8,0,0,da.data());ref_import(D,nb,-1,8,0,0,db.data());ref_mul(P,A,D);exact.assign(2*N,0);size_t count=0;ref_export(exact.data(),&count,-1,8,0,0,P);seen.assign(exact.size(),0);error=0;bad=0;
    SBN3_FRAME_UNPOISON(wf.data(),wf.capacity());memset(wf.data(),0xff,wf.capacity());SBN3_FRAME_POISON(wf.data(),wf.capacity());assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ));
    allocation_watch_start();bool ok;if constexpr(Rac)ok=vx_rac_multiply<M,B>(r,a,an,b,bn,*rt,wf);else ok=vx_multiply<M,B>(r,a,an,b,bn,*t,wf);
    assert(!allocation_watch_stop()&&!wf.used());for(auto v:seen)assert(v==1);
    ref_import(A,an,-1,8,0,0,a);ref_import(D,bn,-1,8,0,0,b);ref_mul(P,A,D);ref_import(Q,an+bn,-1,8,0,0,r);bool eq=!ref_cmp(P,Q);ref_clears(A,D,P,Q,nullptr);
    if(variable_supported(shape,an,bn))assert(ok&&eq&&!bad&&error<=.25);
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ|PROT_WRITE));
    printf("{\"kind\":\"product\",\"recipe\":\"%s\",\"bits\":%u,\"M\":%u,\"branch\":%u,\"N\":%u,\"an\":%zu,\"bn\":%zu,\"pattern\":%u,\"coefficients\":%zu,\"max_error\":%.17g,\"bad\":%zu,\"emit\":%s,\"equal\":%s}\n",Rac?"rac":"ctpq",B,M,n,N,an,bn,pattern,exact.size(),error,bad,ok?"true":"false",eq?"true":"false");fflush(stdout);exact.clear();
}
template<unsigned B>static void band(){
    codec<B>();
    for(unsigned n:{256u,512u,1024u,2048u,4096u,8192u}){if(B==20&&n>256)break;if(B==19&&n>512)break;if(B==18&&n>2048)break;
        for(unsigned p=0;p<4;++p){product<1,B>(4*n,p);product<3,B>(n,p);if(n<8192){product<5,B>(n,p);product<7,B>(n,p);}}}
    for(unsigned trim:{0u,2u}){
        product<1,B>(B==17?32768:B==18?8192:B==19?2048:512,0,false,trim);
        product<3,B>(B==17?8192:B==18?2048:B==19?512:256,0,false,trim);
        product<5,B>(B==17?4096:B==18?1024:B==19?256:128,0,false,trim);
        product<7,B>(B==17?4096:B==18?1024:B==19?256:128,0,false,trim);
    }
    product<1,B>(2048,0,true);product<3,B>(512,1,true);product<5,B>(512,0,true);product<7,B>(512,1,true);
    // right-angle wide recipe: inside its (smaller) envelope and at the first shape beyond it
    for(unsigned p=0;p<4;++p){
        if(B==17){product<3,B,true>(4096,p);product<5,B,true>(2048,p);product<7,B,true>(2048,p);product<3,B,true>(8192,p);}
        if(B==18){product<3,B,true>(1024,p);product<5,B,true>(1024,p);product<7,B,true>(512,p);product<3,B,true>(2048,p);}
        if(B==19){product<3,B,true>(512,p);product<5,B,true>(256,p);product<3,B,true>(1024,p);}
        if(B==20){product<3,B,true>(256,p);product<5,B,true>(128,p);product<7,B,true>(128,p);}
        if(B>=17){product<5,B,true>(64,p);product<7,B,true>(64,p);}   // branch 64 (2026-09-09): right-angle only
    }
    product<5,B,true>(B>=19?256:B==18?1024:2048,1,true);product<3,B,true>(B>=19?512:B==18?1024:4096,0,true);
}
extern "C" void test_pq16_variable(){band<17>();band<18>();band<19>();band<20>();
    // envelope policy: right-angle wide caps are the measured ones, CT/PQ caps unchanged
    assert(variable_supported(Shape{12288,4096,3,false,Recipe::RightAngle,17},3000,3000)&&!variable_supported(Shape{24576,8192,3,false,Recipe::RightAngle,17},6000,6000));
    assert(variable_supported(Shape{24576,8192,3,false,Recipe::CooleyTukeyPQ,17},6000,6000)&&!variable_supported(Shape{3584,512,7,false,Recipe::RightAngle,19},800,800));
    assert(select(1328,1328,1).bits>16||select(1328,1328,1).recipe==Recipe::RightAngle);
    puts("variable FFT codec/error envelopes PASS");}
