// Balanced (signed) wide-codec gate: production vx_multiply<M,B,true> / vx_rac_multiply<M,B,true> on adversarial
// operands (limb patterns, digit patterns alternating/constant around 2^(B-1), square wave, full top digits with
// carry, unbalanced lengths); every pre-rounding coefficient against the exact signed convolution of the locally
// balanced digit sequences (reference/modular oracle, sign split); full reference/modular oracle product; emit invariant (front carries, m at the last front,
// zero tail); read-only tables; no implicit allocation; no workspace leak; envelope and selection policy.
#include <cstddef>
static void coefficient(size_t,double);
#define VX_COEFFICIENT_OBSERVER ::coefficient
#include "backend/pq16/island.hpp"
#include <new>
#include <initializer_list>
#include "backend/pq16/ct.hpp"
#include "backend/pq16/rac.hpp"
#include "backend/pq16/variable.hpp"
#include "backend/pq16/rac_wide.hpp"
#include "backend/pq16/numeric_domain.hpp"
#include "../integration/product_support.hpp"
#include <cmath>
#include <numeric>
#include <sys/mman.h>
using namespace sbn::v3;using namespace sbn::v3::pq16;

namespace {
std::vector<int64_t> exact;std::vector<unsigned char> seen;double error;size_t bad;
uint64_t digit(const uint64_t *a,size_t n,size_t j,unsigned B){size_t bit=j*B,k=bit/64;unsigned s=bit%64;if(k>=n)return 0;uint64_t v=a[k]>>s;if(s&&k+1<n)v|=a[k+1]<<(64-s);return v&((uint64_t(1)<<B)-1);}
std::vector<int64_t> balanced_digits(const uint64_t *a,size_t n,unsigned B){   // whole + partial top digit, then the carry digit
    const size_t L=(64*n+B-1)/B;std::vector<int64_t> e(L+1);int64_t prev=0;
    for(size_t i=0;i<L;++i){const int64_t d=int64_t(digit(a,n,i,B));const int64_t t=d>>(B-1);e[i]=d-(t<<B)+prev;prev=t;}
    e[L]=prev;if(!prev)e.pop_back();return e;
}
std::vector<int64_t> signed_conv(const std::vector<int64_t> &a,const std::vector<int64_t> &b,size_t len){
    auto part=[](const std::vector<int64_t> &v,bool neg){std::vector<uint64_t> r(v.size());for(size_t i=0;i<v.size();++i)r[i]=uint64_t(neg?(v[i]<0?-v[i]:0):(v[i]>0?v[i]:0));return r;};
    auto ap=part(a,false),an=part(a,true),bp=part(b,false),bn=part(b,true);std::vector<int64_t> c(len,0);
    auto acc=[&](const std::vector<uint64_t> &x,const std::vector<uint64_t> &y,int sign){
        ref_int X,Y,P;ref_inits(X,Y,P,nullptr);ref_import(X,x.size(),-1,8,0,0,x.data());ref_import(Y,y.size(),-1,8,0,0,y.data());ref_mul(P,X,Y);
        std::vector<uint64_t> out(len,0);size_t cnt=0;assert(ref_sizeinbase(P,2)<=64*len);ref_export(out.data(),&cnt,-1,8,0,0,P);
        for(size_t i=0;i<len;++i){assert(out[i]<(1ull<<62));c[i]+=sign*int64_t(out[i]);}ref_clears(X,Y,P,nullptr);};
    acc(ap,bp,+1);acc(an,bn,+1);acc(ap,bn,-1);acc(an,bp,-1);return c;
}
constexpr unsigned PATTERNS=12;
void fill(uint64_t *p,size_t len,unsigned pattern,unsigned B){
    memset(p,0,len*8);const size_t nd=(64*len)/B;const uint64_t half=uint64_t(1)<<(B-1),max=(uint64_t(1)<<B)-1;
    auto put=[&](size_t j,uint64_t d){const size_t bit=j*B,k=bit/64;const unsigned s=bit%64;p[k]|=d<<s;if(s+B>64&&k+1<len)p[k+1]|=d>>(64-s);};
    switch(pattern){
    case 0:for(size_t j=0;j<len;++j)p[j]=UINT64_MAX;break;
    case 1:for(size_t j=0;j<len;++j)p[j]=random_word();break;
    case 2:for(size_t j=0;j<len;++j)p[j]=(j&1)?0:UINT64_MAX;break;
    case 3:p[0]=UINT64_MAX;p[len-1]=UINT64_MAX;break;
    case 4:for(size_t j=0;j<len/2;++j)p[j]=UINT64_MAX;break;
    case 5:for(size_t j=0;j<len;++j)p[j]=0x8000800080008000ull;break;
    case 6:for(size_t j=0;j<nd;++j)put(j,(j&1)?half-1:half);break;   // balanced +-2^(B-1)
    case 7:for(size_t j=0;j<nd;++j)put(j,half);break;                 // balanced -2^(B-1)+1 constant
    case 8:for(size_t j=0;j<nd;++j)put(j,half-1);break;               // balanced +2^(B-1)-1 constant
    case 9:for(size_t j=0;j<nd;++j)put(j,((j/5)&1)?half-1:half);break; // square wave
    case 10:for(size_t j=0;j<nd;++j)put(j,(random_word()&1)?half-1:half);break;
    default:for(size_t j=0;j<nd;++j)put(j,max);break;                 // full top digit >= 2^(B-1): carry digit
    }
}
void coefficient_hook(size_t k,double x){if(exact.empty())return;assert(k<exact.size()&&!seen[k]++&&std::isfinite(x));double e=std::abs(x-double(exact[k]));error=std::max(error,e);bad+=e>=.5;}
template<unsigned M,unsigned B,bool Rac>double product(unsigned n,unsigned pattern,size_t an,size_t bn){
    const unsigned N=M*n;Shape shape{N,n,M,false,Rac?Recipe::RightAngle:Recipe::CooleyTukeyPQ,B,true};Fixture f(1);
    auto tl=f.allocate(table_bytes(shape),128),wl=f.allocate(scratch_bytes(shape,an,bn),128);Frame tf(*f.arena,tl),wf(*f.arena,wl);
    pq16_plan core{};core.builder=&tf;pq16_plan_ensure(&core,n,shape.radix!=1);const CtTables *ct=nullptr;const RacTables *rt=nullptr;
    if constexpr(Rac)rt=rac_prepare(tf,core,shape);else ct=ct_prepare(tf,core,shape);
    auto *a=f.guarded(an),*b=f.guarded(bn),*r=f.guarded(an+bn);fill(a,an,pattern,B);fill(b,bn,pattern,B);
    const auto ea=balanced_digits(a,an,B),eb=balanced_digits(b,bn,B);assert(ea.size()+eb.size()-1<=2*size_t(N));
    exact=signed_conv(ea,eb,2*size_t(N));seen.assign(exact.size(),0);error=0;bad=0;
    SBN3_FRAME_UNPOISON(wf.data(),wf.capacity());memset(wf.data(),0xff,wf.capacity());SBN3_FRAME_POISON(wf.data(),wf.capacity());assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ));
    allocation_watch_start();bool ok;if constexpr(Rac)ok=vx_rac_multiply<M,B,true>(r,a,an,b,bn,*rt,wf);else ok=vx_multiply<M,B,true>(r,a,an,b,bn,*ct,wf);
    assert(!allocation_watch_stop()&&ok&&!wf.used());for(auto v:seen)assert(v==1);assert(bad==0);verify_product(a,an,b,bn,r);
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ|PROT_WRITE));exact.clear();return error;
}
template<unsigned M,unsigned B>void shape(unsigned n,bool cap){
    const unsigned N=M*n;const size_t full=size_t(N)*B/32,an=full/2-1;double worst=0;
    const bool rac=M!=1&&N<=balanced_cap(Recipe::RightAngle,B,M),ct=n>=128;   // the right-angle kernels run inside their own (smaller) cap; branch 64 is right-angle only
    auto both=[&](unsigned p,size_t x,size_t y){if(ct)worst=std::max(worst,product<M,B,false>(n,p,x,y));if constexpr(M!=1){if(rac)worst=std::max(worst,product<M,B,true>(n,p,x,y));}};
    for(unsigned p=0;p<PATTERNS;++p)if(!cap||p>=6)both(p,an,an);
    // exact capacity with complete top digits (carry digits on both operands), then unbalanced lengths
    const size_t step=B/std::gcd(B,64u);size_t ac=((full/2)/step)*step;while(2*((64*ac)/B)+2>2*size_t(N))ac-=step;
    both(11,ac,ac);if(!cap){both(6,2*an-1,1);both(0,2*an-1,1);}
    assert(worst<.25);printf("balanced %u-bit M%u N%u%s: worst coefficient error %.4g PASS\n",B,M,N,rac?" (+right-angle)":"",worst);
}
}
void coefficient(size_t k,double x){coefficient_hook(k,x);}
extern "C" void test_pq16_balanced(){
    shape<1,17>(8192,false);shape<3,17>(2048,false);shape<5,17>(1024,false);shape<7,17>(1024,false);
    shape<1,18>(4096,false);shape<3,18>(2048,false);shape<5,18>(1024,false);shape<7,18>(1024,false);
    shape<1,19>(2048,false);shape<3,19>(1024,false);shape<5,19>(512,false);shape<7,19>(512,false);
    // 20-bit (2026-09-09 small band): smallest shapes and the caps; the spill case (carry digits whose product lands in the next front)
    shape<1,20>(512,false);shape<3,20>(256,false);shape<5,20>(128,false);shape<7,20>(128,false);
    shape<1,20>(2048,true);shape<3,20>(1536/3,true);shape<5,20>(256,true);shape<7,20>(256,true);
    shape<1,17>(512,false);shape<5,17>(128,false);shape<7,17>(128,false);
    shape<5,19>(64,false);shape<7,19>(64,false);shape<5,20>(64,false);shape<7,20>(64,false);   // branch 64: M5 320 / M7 448 (right-angle inside its cap)
    // 16-bit balanced digits for the large band (2026-09-09): the same kernels with 32-bit pair slots (two-bit carry path
    // from N 2^17 on); smallest shapes, then the measured caps (CT/PQ M1 131072, M3 196608, M5 163840, M7 229376)
    shape<3,16>(32768,false);shape<5,16>(16384,false);shape<7,16>(16384,false);
    shape<1,16>(131072,true);shape<3,16>(65536,true);shape<5,16>(32768,true);shape<7,16>(32768,true);
    assert(variable_supported(Shape{229376,32768,7,false,Recipe::CooleyTukeyPQ,16,true},55109,55109)&&!variable_supported(Shape{458752,65536,7,false,Recipe::CooleyTukeyPQ,16,true},110218,110218));
    assert(!variable_supported(Shape{131072,131072,1,false,Recipe::CooleyTukeyPQ,16,false},32000,32000));   // unsigned 16-bit stays on the classic pipeline
    assert(!variable_supported(Shape{448,64,7,false,Recipe::CooleyTukeyPQ,19,false},128,128)&&variable_supported(Shape{448,64,7,false,Recipe::RightAngle,19,false},128,128));
    // envelope caps: CT/PQ caps, plus the right-angle caps that differ
    shape<1,18>(16384,true);shape<3,18>(4096,true);shape<5,18>(4096,true);shape<7,18>(2048,true);shape<5,18>(2048,true);
    shape<1,19>(4096,true);shape<3,19>(2048,true);shape<5,19>(1024,true);shape<7,19>(512,true);shape<7,19>(256,true);
    // policy: caps, carry-digit capacity, selection
    assert(variable_supported(Shape{12288,4096,3,false,Recipe::CooleyTukeyPQ,18,true},3300,3300)&&!variable_supported(Shape{24576,8192,3,false,Recipe::CooleyTukeyPQ,18,true},6600,6600));
    assert(variable_supported(Shape{12288,4096,3,false,Recipe::RightAngle,18,true},3300,3300)&&!variable_supported(Shape{6144,2048,3,false,Recipe::RightAngle,19,true},1700,1700));
    assert(variable_supported(Shape{2560,512,5,false,Recipe::RightAngle,19,true},756,756)&&variable_supported(Shape{5120,1024,5,false,Recipe::CooleyTukeyPQ,19,true},1500,1500));
    assert(!variable_supported(Shape{2048,2048,1,false,Recipe::CooleyTukeyPQ,17,true},544,544)&&variable_supported(Shape{2048,2048,1,false,Recipe::CooleyTukeyPQ,17,false},544,544)); // 544*64 = 2048 digits each: carry digits need the next size
    const Shape s899=select(899,899,1),s3298=select(3298,3298,1),s4467=select(4467,4467,1),s512=select(512,512,1),s3597=select(3597,3597,1);
    assert(s899.balanced&&s899.bits==19&&s899.radix==3&&s899.nfull==3072&&s899.recipe==Recipe::RightAngle);
    assert(!s3597.balanced&&s3597.bits==17&&s3597.radix==7&&s3597.nfull==14336&&s3597.recipe==Recipe::RightAngle); // 2026-09-08: beats 16-bit pow2 16384 by 3-5%; right-angle 1-4% ahead of CT/PQ with the odd-radix branch plan
    assert(s3298.balanced&&s3298.bits==18&&s3298.radix==3&&s3298.nfull==12288);
    assert(s4467.balanced&&s4467.bits==18&&s4467.radix==1&&s4467.nfull==16384);
    assert(!s512.balanced&&s512.bits==16);
    // small band (2026-09-09): the codec opens at 128-limb operands, 20-bit digits inside their caps, 21-bit never
    assert(variable_supported(Shape{512,512,1,false,Recipe::CooleyTukeyPQ,20,true},159,159)&&variable_supported(Shape{512,512,1,false,Recipe::CooleyTukeyPQ,20,false},159,159));
    assert(!variable_supported(Shape{1024,1024,1,false,Recipe::CooleyTukeyPQ,20,false},300,300)&&variable_supported(Shape{1024,1024,1,false,Recipe::CooleyTukeyPQ,20,true},300,300));
    assert(!variable_supported(Shape{512,512,1,false,Recipe::CooleyTukeyPQ,21,true},150,150)&&!variable_supported(Shape{768,256,3,false,Recipe::RightAngle,20,false},220,220)&&variable_supported(Shape{768,256,3,false,Recipe::RightAngle,20,true},220,220));
    assert(variable_supported(Shape{640,128,5,false,Recipe::CooleyTukeyPQ,17,true},150,150)&&!variable_supported(Shape{640,128,5,false,Recipe::CooleyTukeyPQ,17,true},100,100));
    const Shape s159=select(159,159,1);assert(s159.bits==20&&s159.nfull==512);
    puts("balanced wide codec gate PASS");
}
