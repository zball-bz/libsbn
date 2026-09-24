#include <cstddef>
static void coefficient(size_t,double);
#define VX_COEFFICIENT_OBSERVER ::coefficient
#include "backend/pq16/island.hpp"
#include "backend/pq16/ct.hpp"
#include "backend/pq16/variable.hpp"
#include "../integration/product_support.hpp"
#include <cmath>
using namespace sbn::v3;
using namespace sbn::v3::pq16;
static std::vector<int64_t> exact;
static std::vector<unsigned char> seen;
static double maximum;
static void pair_carry_boundary(){
    constexpr uint64_t radix=uint64_t(1)<<32,mask=radix-1;
    for(unsigned incoming=0;incoming<3;++incoming)for(unsigned boundary=0;boundary<8;++boundary){
        alignas(64) double re[8],im[8];alignas(64) uint64_t got[8],hi[8];
        for(unsigned j=0;j<8;++j){re[j]=double(radix-2+(j>=boundary));im[j]=double(radix<<16);}
        q_chain chain{};chain.prev_hi=sb_set1_64(radix);chain.cin=incoming;
        sb_store(got,vx_pair<16>({_mm512_load_pd(re),_mm512_load_pd(im)},chain,sb_set1_64(0x4338000000000000LL)));
        unsigned __int128 carry=radix+incoming;
        for(unsigned j=0;j<8;++j){const unsigned __int128 v=uint64_t(re[j])+((unsigned __int128)uint64_t(im[j])<<16)+carry;
            assert(got[j]==uint64_t(v&mask));carry=v>>32;}
        sb_store(hi,chain.prev_hi);assert(carry==hi[7]+chain.cin);
    }
    puts("32-bit pair double-carry boundary: exact 128-bit recurrence PASS");
}
static void coefficient(size_t at,double x){assert(at<exact.size()&&!seen[at]++&&std::isfinite(x));maximum=std::max(maximum,std::abs(x-double(exact[at])));}
static void fill(uint64_t *a,size_t n,unsigned bits,unsigned pattern){
    memset(a,0,n*8);
    if(pattern==0||pattern==1||pattern==4){for(size_t j=0;j<n;++j)a[j]=pattern==0?UINT64_MAX:pattern==1?random_word():j%2?UINT64_MAX:0;return;}
    for(size_t j=0;j<(64*n+bits-1)/bits;++j){const uint64_t d=pattern==2?(1u<<(bits-1))-1:pattern==3?1u<<(bits-1):(1u<<(bits-1))-(j%2);
        size_t word=j*bits/64;unsigned shift=j*bits%64;a[word]|=d<<shift;if(shift+bits>64&&word+1<n)a[word+1]|=d>>(64-shift);}
}
static std::vector<int64_t> decode(const uint64_t *a,size_t n,unsigned b,bool balanced){
    size_t count=(64*n+b-1)/b;std::vector<int64_t> v(count+1);uint64_t previous=0;
    for(size_t j=0;j<count;++j){size_t word=j*b/64;unsigned shift=j*b%64;uint64_t d=a[word]>>shift;if(shift&&word+1<n)d|=a[word+1]<<(64-shift);d&=(1u<<b)-1;
        const uint64_t top=balanced?d>>(b-1):0;v[j]=int64_t(d)-int64_t(top<<b)+int64_t(previous);previous=top;}
    v[count]=previous;return v;
}
template<unsigned M,unsigned B,bool S>static void check(unsigned branch,unsigned pattern,unsigned parts){
    const size_t N=M*branch,ring=N*B/32,an=ring*parts/16-1,bn=ring;Fixture f(1);Shape shape{unsigned(N),branch,M,false,Recipe::CooleyTukeyPQ,B,S};
    auto tl=f.allocate(table_bytes(shape)+256,128),wl=f.allocate(32*N+2048,128);Frame tf(*f.arena,tl),work(*f.arena,wl);
    pq16_plan plan{};plan.builder=&tf;pq16_plan_ensure(&plan,branch,M!=1);auto *t=ct_prepare(tf,plan,shape);
    auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(ring);fill(a,an,B,pattern==6?2:pattern==7?3:pattern);fill(b,bn,B,pattern==6?3:pattern==7?2:pattern);
    auto da=decode(a,an,B,S),db=decode(b,bn,B,S);const uint64_t lift=S?uint64_t(1)<<(B-1):0;
    std::vector<uint64_t> aa(da.size()),bb(db.size()),pa(da.size()+1),pb(db.size()+1),full(da.size()+db.size());
    for(size_t j=0;j<aa.size();++j){aa[j]=uint64_t(da[j]+int64_t(lift));pa[j+1]=pa[j]+aa[j];}
    for(size_t j=0;j<bb.size();++j){bb[j]=uint64_t(db[j]+int64_t(lift));pb[j+1]=pb[j]+bb[j];}
    ref_int A,C,P;ref_inits(A,C,P,nullptr);ref_import(A,aa.size(),-1,8,0,0,aa.data());ref_import(C,bb.size(),-1,8,0,0,bb.data());ref_mul(P,A,C);
    size_t count=0;ref_export(full.data(),&count,-1,8,0,0,P);ref_clears(A,C,P,nullptr);
    exact.assign(2*N,0);
    for(size_t k=0;k+1<full.size();++k){const size_t lo=k>=bb.size()?k-bb.size()+1:0,hi=std::min(k+1,aa.size()),other_lo=k+1-hi,other_hi=k-lo+1;
        const __int128 c=(__int128)full[k]-(__int128)lift*((pa[hi]-pa[lo])+(pb[other_hi]-pb[other_lo]))+(__int128)lift*lift*(hi-lo);
        exact[k%(2*N)]+=int64_t(c);}
    maximum=0;seen.assign(2*N,0);allocation_watch_start();const bool ok=vx_multiply<M,B,S,true>(out,a,an,b,bn,*t,work,nullptr,false);assert(!allocation_watch_stop()&&ok);
    for(auto v:seen)assert(v==1);
    // Coefficient accuracy alone does not establish integer carry correctness.
    // Opposite balanced-digit signs exercise the bias when short support>N.
    ref_int X,Y,Z,Mod;ref_inits(X,Y,Z,Mod,nullptr);ref_import(X,an,-1,8,0,0,a);ref_import(Y,bn,-1,8,0,0,b);
    ref_mul(Z,X,Y);ref_set_ui(Mod,1);ref_mul_2exp(Mod,Mod,64*ring);ref_sub_ui(Mod,Mod,1);ref_mod(Z,Z,Mod);
    ref_import(X,ring,-1,8,0,0,out);ref_mod(X,X,Mod);assert(ref_cmp(X,Z)==0);ref_clears(X,Y,Z,Mod,nullptr);
    printf("wide cyclic precision B%u M%u N%zu signed=%u parts=%u pattern=%u max_error=%.9g\n",B,M,N,S,parts,pattern,maximum);fflush(stdout);assert(maximum<.25);
}
template<unsigned M,unsigned B,bool S>static void band(){
    unsigned branch=128;Shape s{M*branch,branch,M,false,Recipe::CooleyTukeyPQ,B,S};if(!cyclic_supported(s,1,1))return;
    for(;;){Shape next{M*branch*2,branch*2,M,false,Recipe::CooleyTukeyPQ,B,S};if(!cyclic_supported(next,1,1))break;branch*=2;}
    for(unsigned parts:{8u,9u,10u}){if(parts==10&&!(B==16&&S))continue;for(unsigned pattern=0;pattern<8;++pattern)check<M,B,S>(branch,pattern,parts);}
}
template<unsigned B>static void bits(){band<1,B,false>();band<3,B,false>();band<5,B,false>();band<7,B,false>();band<1,B,true>();band<3,B,true>();band<5,B,true>();band<7,B,true>();}
int main(){pair_carry_boundary();bits<16>();bits<17>();bits<18>();bits<19>();bits<20>();puts("wide cyclic raw signed coefficient precision margin PASS");}
