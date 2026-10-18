// Original libsbn code; LGPL-3.0-or-later.
#include "gcd/internal.hpp"
#include "sbn3/jacobi.h"
#include "common/small_checks.h"
namespace sbn::v3::gcd {
struct SymbolCall {Span a,b;void*scratch;size_t bytes;int result;};
static void symbol_body(void*ptr,sbn3_team_scope*scope){
    auto &c=*static_cast<SymbolCall*>(ptr);auto frame=Frame::external(c.scratch,c.bytes);
    JacobiState symbol{};SymbolEngine e{{frame,scope,default_policy(scope?scope->width:1)},&symbol};
    const size_t n=std::min(c.a.n,c.b.n);
    Pair p{allocate(frame,round_words(n+3)),allocate(frame,round_words(n+3))};unsigned sign=0;
    if(c.a.n>c.b.n){assign(p.b,c.b);stream_remainder(p.a,c.a,c.b,e);}
    else if(c.b.n>c.a.n){
        // Strip numerator twos, then reciprocity turns a long denominator
        // into a streaming remainder. No original-size quotient is stored.
        size_t skip=0;while(!c.a.p[skip])++skip;
        const unsigned shift=__builtin_ctzll(c.a.p[skip]);
        sign=(shift*((c.b.p[0]>>1)^(c.b.p[0]>>2)))&1;
        p.b.n=c.a.n-skip;
        for(size_t j=0;j<p.b.n;++j)p.b.p[j]=shift?(c.a.p[skip+j]>>shift)|(skip+j+1<c.a.n?c.a.p[skip+j+1]<<(64-shift):0):c.a.p[skip+j];
        p.b.trim();sign^=((p.b.p[0]&c.b.p[0])>>1)&1;
        stream_remainder(p.a,c.b,p.b,e);
    }else {assign(p.a,c.a);assign(p.b,c.b);}
    symbol=JacobiState::initial(p.a.n?p.a.p[0]:0,p.b.p[0],sign);
    c.result=symbol_execution(p,e);
}
}
using namespace sbn::v3;
extern "C" size_t sbn3_jacobi_scratch_bytes(size_t an,size_t nn,unsigned w){return nn?sbn3_gcd_scratch_bytes(an,nn,w):SIZE_MAX;}
extern "C" sbn3_query_result sbn3_jacobi_query(size_t an,size_t nn,const sbn3_jacobi_options*o,sbn3_jacobi_info*i){
    require(i,SBN3_FATAL_ARGUMENT,"Jacobi query output");*i={};if(!nn)return SBN3_UNSUPPORTED;
    const sbn3_gcd_options opt{o?o->workers:1};sbn3_gcd_info g{};auto result=sbn3_gcd_query(an,nn,&opt,&g);
    if(result==SBN3_SUPPORTED)*i={g.scratch_bytes,g.scratch_alignment,g.value_bytes,g.temporary_bytes,g.arithmetic_bytes,g.published_constants_bytes+sizeof(gcd::jacobi_transitions),g.workers};
    return result;
}
[[gnu::always_inline]] static inline int jacobi_magnitude(const uint64_t*a,size_t an,const uint64_t*b,size_t bn,void*scratch,size_t bytes,sbn3_team*team){
    if constexpr(SBN3_CHECK_SMALL){
        const size_t need=sbn3_jacobi_scratch_bytes(an,bn,team?team->width:1);
        require(need!=SIZE_MAX&&bytes>=need,SBN3_FATAL_WORKSPACE,"Jacobi scratch",need,bytes);
        valid_span(a,an*8,"Jacobi numerator");valid_span(b,bn*8,"Jacobi denominator");valid_span(scratch,need,"Jacobi scratch");
        require(bn&&(b[0]&1),SBN3_FATAL_ARGUMENT,"Jacobi positive odd denominator");
        require(!need||(scratch&&!(uintptr_t(scratch)&(std::max(an,bn)<=SBN3_GCD_SMALL_LIMBS?7:63))),SBN3_FATAL_ARGUMENT,"Jacobi scratch alignment");
        require(!overlaps(scratch,need,a,an*8)&&!overlaps(scratch,need,b,bn*8),SBN3_FATAL_ARGUMENT,"Jacobi input alias");
    }
    if(an<=1&&bn==1)return sbn3_jacobi_u64(an?a[0]:0,b[0]);
    // The native preserving entry normalizes its inputs once. Avoid a
    // duplicate scan and another dispatch chain for ordinary small calls.
    if(std::max(an,bn)<=SBN3_GCD_SMALL_LIMBS||std::min(an,bn)<=2)
        return gcd::symbol_small(a,an,b,bn,scratch);
    auto av=gcd::normalized({a,an}),bv=gcd::normalized({b,bn});
    if(!av.n)return bv.n==1&&bv.p[0]==1;
    if(std::max(av.n,bv.n)<=SBN3_GCD_SMALL_LIMBS||std::min(av.n,bv.n)<=2)
        return gcd::symbol_small(av.p,av.n,bv.p,bv.n,scratch);
    const size_t need=sbn3_jacobi_scratch_bytes(an,bn,team?team->width:1);
    require(need!=SIZE_MAX&&bytes>=need,SBN3_FATAL_WORKSPACE,"Jacobi scratch",need,bytes);
    gcd::SymbolCall c{av,bv,scratch,need,0};
    if(team)sbn3_team_run(team,gcd::symbol_body,&c);else serial_team_run(gcd::symbol_body,&c);return c.result;
}
extern "C" int sbn3_jacobi(const uint64_t*a,size_t an,const uint64_t*b,size_t bn,void*scratch,size_t bytes,sbn3_team*team){
    return jacobi_magnitude(a,an,b,bn,scratch,bytes,team);
}
extern "C" int sbn3_int_jacobi(sbn3_int_view a,sbn3_int_view n,void*scratch,size_t bytes,sbn3_team*team){
    if constexpr(SBN3_CHECK_SMALL)require(!n.negative,SBN3_FATAL_ARGUMENT,"Jacobi negative denominator");
    const int result=jacobi_magnitude(a.data,a.size,n.data,n.size,scratch,bytes,team);
    return a.negative&&(n.data[0]&2)?-result:result;
}
