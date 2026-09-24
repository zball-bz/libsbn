#include <cstddef>
static void coefficient(size_t,double);
#define VX_COEFFICIENT_OBSERVER ::coefficient
#include "backend/pq16/island.hpp"
#include "backend/pq16/ct.hpp"
#include "backend/pq16/rac.hpp"
#include "backend/pq16/variable.hpp"
#include "backend/pq16/rac_wide.hpp"
#include "backend/pq16/rac_window.hpp"
#include "../integration/product_support.hpp"
#include "value/plus_ring.hpp"
#include <cmath>
using namespace sbn::v3;
using namespace sbn::v3::pq16;
static std::vector<int64_t> exact;
static std::vector<unsigned char> seen;
static double maximum;
static void coefficient(size_t at,double x){assert(at<exact.size()&&!seen[at]++&&std::isfinite(x));maximum=std::max(maximum,std::abs(x-double(exact[at])));}
static void fill(uint64_t *a,size_t n,unsigned bits,unsigned pattern){
    memset(a,0,n*8);
    if(pattern==0||pattern==1||pattern==4){for(size_t j=0;j<n;++j)a[j]=pattern==0?UINT64_MAX:pattern==1?random_word():j%2?UINT64_MAX:0;return;}
    for(size_t j=0;j<(64*n+bits-1)/bits;++j){const uint64_t d=pattern==2?(1u<<(bits-1))-1:pattern==3?1u<<(bits-1):(1u<<(bits-1))-(j%2);
        const size_t word=j*bits/64;const unsigned shift=j*bits%64;a[word]|=d<<shift;
        if(shift+bits>64&&word+1<n)a[word+1]|=d>>(64-shift);}
}
static std::vector<int64_t> decode(const uint64_t *a,size_t n,unsigned b,bool signed_digits){
    const size_t count=(64*n+b-1)/b;std::vector<int64_t> v(count+1);uint64_t previous=0;
    for(size_t j=0;j<count;++j){size_t word=j*b/64;unsigned shift=j*b%64;uint64_t d=a[word]>>shift;
        if(shift&&word+1<n)d|=a[word+1]<<(64-shift);d&=(1u<<b)-1;
        const uint64_t top=signed_digits?d>>(b-1):0;v[j]=int64_t(d)-int64_t(top<<b)+int64_t(previous);previous=top;}
    v[count]=previous;return v;
}
template<unsigned M,unsigned B,bool Signed=true>static void check(unsigned branch,unsigned parts,unsigned pattern,bool explore){
    const size_t N=M*branch,ring=N*B/32,an=ring*parts/16-1,bn=ring-1;
    Fixture f(1);Shape shape{unsigned(N),branch,M,false,Recipe::RightAngle,B,Signed};
    auto tl=f.allocate(table_bytes(shape)+256,128),wl=f.allocate(32*N+2048,128),cl=f.allocate(16*N+128,128);
    Frame tables(*f.arena,tl),work(*f.arena,wl);pq16_plan core{};core.builder=&tables;pq16_plan_ensure(&core,branch,true);
    auto *t=rac_prepare(tables,core,shape);auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(ring+1);
    fill(a,an,B,pattern==6?2:pattern==7?3:pattern);fill(b,bn,B,pattern==6?3:pattern==7?2:pattern);
    if(pattern==8){memset(a,0,8*an);memset(b,0,8*bn);a[an-1]=1;b[ring-an+1]=1;}
    if(pattern==9)memset(a,0,8*an);
    const auto da=decode(a,an,B,Signed),db=decode(b,bn,B,Signed);const uint64_t lift=Signed?uint64_t(1)<<(B-1):0;
    std::vector<uint64_t> aa(da.size()),bb(db.size()),pa(da.size()+1),pb(db.size()+1),full(da.size()+db.size());
    for(size_t j=0;j<aa.size();++j){aa[j]=uint64_t(da[j]+int64_t(lift));pa[j+1]=pa[j]+aa[j];}
    for(size_t j=0;j<bb.size();++j){bb[j]=uint64_t(db[j]+int64_t(lift));pb[j+1]=pb[j]+bb[j];}
    ref_int A,C,P,Mod;ref_inits(A,C,P,Mod,nullptr);
    ref_import(A,aa.size(),-1,8,0,0,aa.data());ref_import(C,bb.size(),-1,8,0,0,bb.data());ref_mul(P,A,C);
    size_t count=0;ref_export(full.data(),&count,-1,8,0,0,P);exact.assign(2*N,0);
    for(size_t k=0;k+1<full.size();++k){const size_t lo=k>=bb.size()?k-bb.size()+1:0,hi=std::min(k+1,aa.size());
        const size_t other_lo=k+1-hi,other_hi=k-lo+1;
        const __int128 c=(__int128)full[k]-(__int128)lift*((pa[hi]-pa[lo])+(pb[other_hi]-pb[other_lo]))+(__int128)lift*lift*(hi-lo);
        exact[k%(2*N)]+=(k<2*N?1:-1)*int64_t(c);}
    ref_import(A,an,-1,8,0,0,a);ref_import(C,bn,-1,8,0,0,b);ref_mul(P,A,C);
    ref_set_ui(Mod,1);ref_mul_2exp(Mod,Mod,64*ring);ref_add_ui(Mod,Mod,1);ref_mod(P,P,Mod);
    auto *cache=static_cast<double*>(cl.data);if constexpr(B==16&&!Signed)rac_forward<M>(cache,a,an,*t);else vx_rac_forward<M,B,Signed>(cache,a,an,*t);
    const std::vector<double> saved(cache,cache+2*N);
    for(bool cached:{false,true}){
        maximum=0;seen.assign(2*N,0);
        allocation_watch_start();rac_window_multiply<M,B,Signed>(out,a,an,b,bn,cached?cache:nullptr,*t,work);assert(!allocation_watch_stop());
        for(auto v:seen)assert(v==1);ref_import(A,ring+1,-1,8,0,0,out);const bool equal=ref_cmp(A,P)==0;
        assert(!memcmp(cache,saved.data(),16*N)&&!work.used());
        printf("{\"B\":%u,\"M\":%u,\"N\":%zu,\"parts\":%u,\"pattern\":%u,\"signed\":%s,\"cached\":%s,\"max_error\":%.9g,\"equal\":%s}\n",B,M,N,parts,pattern,Signed?"true":"false",cached?"true":"false",maximum,equal?"true":"false");fflush(stdout);
        if(!explore&&bounded_plus_supported(shape,an,bn))assert(equal&&maximum<.25);
    }
    ref_clears(A,C,P,Mod,nullptr);
}
static void arithmetic(){
    unsigned cases=0;
    for(size_t n:{3u,9u,64u})for(unsigned pattern=0;pattern<5;++pattern){
        std::vector<uint64_t> z(n+1),a(n+1),out(n+1);
        for(size_t j=0;j<n;++j)z[j]=pattern==0?0:pattern==1?UINT64_MAX:random_word();
        if(pattern==3){std::fill(z.begin(),z.end(),0);z[n]=1;}
        if(pattern==4){std::fill(z.begin(),z.end(),0);z[n-1]=uint64_t(1)<<63;}
        for(auto &v:a)v=random_word();
        ref_int Z,A,P,E,R;ref_inits(Z,A,P,E,R,nullptr);ref_import(Z,n+1,-1,8,0,0,z.data());ref_import(A,n+1,-1,8,0,0,a.data());
        ref_set_ui(P,1);ref_mul_2exp(P,P,64*n);ref_add_ui(P,P,1);
        for(size_t shift:{size_t(0),n/2,n-1}){
            out=z;ref_mul_2exp(E,A,64*shift);ref_sub(E,Z,E);ref_mod(E,E,P);
            limbs::plus_sub_shifted(out.data(),n,a.data(),a.size(),shift);
            ref_import(R,n+1,-1,8,0,0,out.data());assert(ref_cmp(R,E)==0);
            ref_mul_2exp(R,E,1);const bool want=ref_cmp(R,P)>0;
            if(want)ref_sub(E,P,E);
            assert(limbs::plus_absolute(out.data(),n)==want);ref_import(R,n+1,-1,8,0,0,out.data());assert(ref_cmp(R,E)==0);++cases;
        }
        for(size_t exponent:{size_t(0),n-1,n,n+1,2*n-1}){
            out=z;ref_set_ui(E,1);ref_mul_2exp(E,E,64*exponent);ref_sub(E,Z,E);ref_mod(E,E,P);
            limbs::plus_sub_power(out.data(),n,exponent);ref_import(R,n+1,-1,8,0,0,out.data());assert(ref_cmp(R,E)==0);++cases;
        }
        ref_clears(Z,A,P,E,R,nullptr);
    }
    printf("plus arithmetic: %u independent integer cases PASS\n",cases);
}
int main(int argc,char **argv){
    const bool explore=argc==2&&!strcmp(argv[1],"--explore");assert(argc==1||explore);
    arithmetic();
    for(unsigned parts:{8u,9u})for(unsigned pattern=0;pattern<10;++pattern){
        check<3,16>(2048,parts,pattern,explore);check<3,16>(4096,parts,pattern,explore);
        check<5,16>(2048,parts,pattern,explore);check<7,16>(4096,parts,pattern,explore);
        check<3,17>(4096,parts,pattern,explore);check<5,17>(4096,parts,pattern,explore);
        check<3,18>(2048,parts,pattern,explore);check<3,18>(4096,parts,pattern,explore);
        check<5,16>(4096,parts,pattern,explore);check<7,17>(2048,parts,pattern,explore);
        check<5,18>(1024,parts,pattern,explore);check<7,18>(512,parts,pattern,explore);
        check<3,16,false>(4096,parts,pattern,explore);check<5,16,false>(4096,parts,pattern,explore);check<7,16,false>(4096,parts,pattern,explore);
    }
    puts(explore?"RAC window envelope exploration complete":"RAC window coefficient/integer/cache gates PASS");
}
