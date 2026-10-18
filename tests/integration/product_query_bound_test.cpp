#include "product/short_query_bounds.hpp"
#include "backend/pq16/kernels.hpp"
#include "sbn3/product.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace sbn::v3;
// W1 small domain (an+bn in [256, 16384]): the floor against every legal shape, not only the
// selected plan (all radices, branches, widths, recipes, centering and balanced digits that
// pq16::supported accepts), and the AUTO choice against the forced PQ16 candidate, so that a
// skipped FFT search could not have won.
static unsigned every_shape(size_t an,size_t bn) {
    // fft_score is max(relaxed, per-radix floor): both must hold; the staged predicate must agree
    const double lower=std::max(query_bounds::fft_score(an,bn,1),query_bounds::relaxed_score(an,bn,1));unsigned n=0;
    for(double best:{lower*0.5,lower,lower*(1+1e-6),lower*2})
        assert(query_bounds::fft_may_beat(an,bn,1,best)==!(best<=query_bounds::fft_score(an,bn,1)));
    for(uint32_t radix:{1u,3u,5u,7u})for(uint32_t branch=1;branch<=(1u<<15);branch*=2)
        for(unsigned bits=16;bits<=20;++bits)
            for(auto recipe:{pq16::Recipe::PfaPQ,pq16::Recipe::CooleyTukeyPQ,pq16::Recipe::RightAngle})
                for(bool centered:{false,true})for(bool balanced:{false,true}){
                    const pq16::Shape s{radix*branch,branch,radix,centered,recipe,bits,balanced};
                    if(!pq16::supported(s,an,bn,1))continue;
                    const double cost=pq16::native_cost(s,an,bn);
                    if(!(lower<=cost))fprintf(stderr,"floor %zu x %zu: M%u N%u %ub recipe %u c%d b%d: %.9g > %.9g\n",an,bn,radix,
                                              radix*branch,bits,unsigned(recipe),centered,balanced,lower,cost);
                    assert(lower<=cost);++n;
                }
    return n;
}
static void choice(size_t an,size_t bn,unsigned &checked) {
    sbn3_product_spec s{an,bn};sbn3_mul_plan p{};sbn3_mul_info automatic{},fft{};
    sbn3_mul_options o{};o.workers=1;
    if(sbn3_mul_query(&s,&o,&p,&automatic)!=SBN3_SUPPORTED)return;
    o.algorithm=SBN3_MUL_PQ16;
    if(sbn3_mul_query(&s,&o,&p,&fft)!=SBN3_SUPPORTED || automatic.algorithm==SBN3_MUL_PQ16)return;
    const double chosen=automatic.algorithm==SBN3_MUL_U52?u52_product_cost(an,bn,1):
                        automatic.algorithm==SBN3_MUL_SCALAR?scalar_product_cost(an,bn,1):-1.;
    if(chosen<0)return;
    const double pq=pq16_product_cost(fft,an,bn);
    if(!(chosen<=pq))fprintf(stderr,"AUTO %zu x %zu chose %u at %.9g, PQ16 %.9g\n",an,bn,automatic.algorithm,chosen,pq);
    assert(chosen<=pq);++checked;
}
// The W1 direct U52 entries of AUTO (api.cpp select_family). Squares: U52 through 292 and on
// 320..349, the cost comparison (PQ16) on 293..319, each edge +-1; a workspace budget below U52's
// own falls back to the comparison, which then cannot answer U52. Products to 16 limbs: scalar or
// U52, and where U52 the comparison would take it too (cheaper than the scalar model and below
// every FFT score), both operand orders and the side beyond 16.
static unsigned dispatch() {
    unsigned n_checked=0;
    auto sqr=[](size_t n,size_t budget,sbn3_product_info &i){
        sbn3_product_request r{};r.kind=SBN3_PRODUCT_SQR;r.a_limbs=n;
        sbn3_mul_options o{};o.workers=1;o.workspace_budget=budget;sbn3_mul_plan p{};i={};
        return sbn3_product_query(&r,&o,&p,&i);
    };
    for(size_t n=280;n<=360;++n){
        sbn3_product_info i{};
        assert(sqr(n,0,i)==SBN3_SUPPORTED);
        const bool u52=n<=292 || (n>=320 && n<=349);
        if(u52 || n<=319)assert(i.mul.algorithm==(u52?SBN3_MUL_U52:SBN3_MUL_PQ16));
        if(u52){
            const size_t need=i.mul.workspace_bytes;sbn3_product_info j{};
            assert(sqr(n,need,j)==SBN3_SUPPORTED && j.mul.algorithm==SBN3_MUL_U52);
            const auto rc=sqr(n,need-1,j);
            assert(rc==SBN3_QUERY_CAPACITY || (rc==SBN3_SUPPORTED && j.mul.algorithm!=SBN3_MUL_U52 && j.mul.workspace_bytes<need));
        }
        ++n_checked;
    }
    for(size_t a=1;a<=17;++a)for(size_t b=1;b<=17;++b){
        sbn3_product_spec s{a,b};sbn3_mul_options o{};o.workers=1;sbn3_mul_plan p{};sbn3_mul_info i{};
        assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_SUPPORTED);
        if(std::max(a,b)<=16)assert(i.algorithm==SBN3_MUL_SCALAR || i.algorithm==SBN3_MUL_U52);
        if(i.algorithm==SBN3_MUL_U52){
            const double u=u52_product_cost(a,b,1);
            assert(u<scalar_product_cost(a,b,1) && (std::max(a,b)>16 || !query_bounds::fft_may_beat(a,b,1,u)));
        }
        ++n_checked;
    }
    return n_checked;
}
// The compile-time floor table equals native_cost evaluated at run time, entry by entry.
static unsigned floor_table() {
    unsigned n=0;volatile unsigned bump=0;
    for(unsigned c=0;c<2;++c)for(unsigned r=0;r<4;++r)for(unsigned k=0;k<query_bounds::floor_detail::logs;++k){
        const double t=query_bounds::floor_detail::table.v[c][r][k];
        const uint32_t radix=query_bounds::floor_detail::radices[r]+bump,branch=(uint32_t(1)<<k)+bump;
        if(radix*branch>8192){assert(t<0);continue;}
        const double v=query_bounds::floor_detail::value(c?20:16,radix,branch);
        if(!(t==v))fprintf(stderr,"floor table c%u M%u k%u: %.17g != %.17g\n",c,radix,k,t,v);
        assert(t==v);++n;
    }
    return n;
}
int main(){
    unsigned checked=0;
    const unsigned entries=floor_table();
    for(size_t n:{1u,2u,7u,8u,17u,33u,63u,65u,127u,128u,129u,256u,257u,511u,512u,1023u,1024u,2047u,4097u,8192u,16384u,32768u,65536u,131072u,262144u,524288u,1048576u})
        for(unsigned ratio:{1u,3u,7u,64u})for(unsigned workers:{1u,2u,3u,4u,8u,16u,32u})for(int bits:{0,16,17,18,19,20}){
            sbn3_product_spec s{n,std::max<size_t>(1,n/ratio)};
            sbn3_mul_options o{};o.algorithm=SBN3_MUL_PQ16;o.workers=workers;o.trunk_bits=bits;
            sbn3_mul_plan p{};sbn3_mul_info i{};
            if(sbn3_mul_query(&s,&o,&p,&i)!=SBN3_SUPPORTED)continue;
            const double actual=pq16_product_cost(i,s.a_limbs,s.b_limbs);
            const double lower=query_bounds::fft_score(s.a_limbs,s.b_limbs,workers);
            if(!(lower<=actual))fprintf(stderr,"FFT floor %zu x %zu W%u bits%d: %.9g > %.9g\n",s.a_limbs,s.b_limbs,workers,bits,lower,actual);
            assert(lower<=actual);++checked;
        }
    std::vector<std::pair<size_t,size_t>> domain;
    for(size_t n=128;n<=1100;++n)domain.push_back({n,n});
    for(size_t n:{1500u,2047u,2048u,3000u,4096u,6000u,8191u,8192u})domain.push_back({n,n});
    for(size_t a=256;a<=12000;a+=a/7+1)for(unsigned ratio:{2u,3u,7u,64u})if(a+a/ratio>=256 && a+a/ratio<=16384)domain.push_back({a,a/ratio});
    unsigned shapes=0,choices=0;
    for(auto [a,b]:domain){shapes+=every_shape(a,b);choice(a,b,choices);}
    const unsigned direct=dispatch();
    printf("FFT score lower bound against %u actual plans, %u legal W1 shapes over %zu small-domain products, %u AUTO choices "
           "against forced PQ16, %u compile-time floor entries, %u W1 direct-entry dispatches, widths/worker counts/rectangular shapes PASS\n",
           checked,shapes,domain.size(),choices,entries,direct);
}
