// Original libsbn code; LGPL-3.0-or-later.
#include "gcd/internal.hpp"
#include "common/small_checks.h"
using namespace sbn::v3;
namespace sbn::v3::gcd {
static constexpr size_t maximum=size_t(1)<<20;
static sbn3_query_result query(size_t an,size_t bn,unsigned mask,const sbn3_gcd_options *o,sbn3_gcd_info &i,bool general=false) {
    i={};i.workers=o&&o->workers?o->workers:1;i.cofactor_mask=mask;
    if(i.workers>32||mask>3||an>maximum||bn>maximum)return SBN3_UNSUPPORTED;
    i.gcd_capacity=std::max(an,bn);i.x_capacity=mask&SBN3_EXGCD_X?std::max<size_t>(bn,1):0;
    i.y_capacity=mask&SBN3_EXGCD_Y?std::max<size_t>(an,1):0;i.scratch_alignment=64;
    if(!an||!bn)return SBN3_SUPPORTED;
    if(!general&&!mask&&std::max(an,bn)<=SBN3_GCD_SMALL_LIMBS) {
        i.scratch_alignment=8;i.scratch_bytes=sbn3_gcd_small_scratch_bytes(an,bn);i.value_bytes=i.scratch_bytes;return SBN3_SUPPORTED;
    }
    if(!general&&!mask&&std::min(an,bn)<=2)return SBN3_SUPPORTED;
    if(mask&&an<=1&&bn<=1)return SBN3_SUPPORTED;
    const size_t n=std::min(an,bn),largest=mask?std::max(an,bn):n;
    i.value_bytes=16*round_words(n+3);
    i.matrix_bytes=(mask&SBN3_EXGCD_X?16*round_words(bn+3):0)+(mask&SBN3_EXGCD_Y?16*round_words(an+3):0);
    i.temporary_bytes=8*((general?8:4)*largest+4096);
    i.arithmetic_bytes=arithmetic_envelope(largest,i.workers);
    i.scratch_bytes=i.value_bytes+i.matrix_bytes+i.temporary_bytes+i.arithmetic_bytes+256;
    i.published_constants_bytes=7771648+1235920;
    return SBN3_SUPPORTED;
}
static void value(sbn3_int *out,Span a,bool negative) {
    if(!out)return;
    require(a.n<=out->capacity&&(!a.n||out->data),SBN3_FATAL_ARGUMENT,"extended GCD output capacity",a.n,out->capacity);
    if(a.n)memcpy(out->data,a.p,a.n*8);out->size=a.n;out->negative=a.n&&negative;
}
static void small_extended(sbn3_int *g,sbn3_int *x,sbn3_int *y,U a,U b,unsigned sa,unsigned sb) {
    gcd_small::Matrix m;const U result=word_matrix(a,b,m);
    const U xv=m.u[1][1],yv=m.u[0][1];value(g,{&result,1},false);value(x,{&xv,bool(xv)},sa);value(y,{&yv,bool(yv)},!sb);
}
struct Call {
    Span a,b;unsigned sa=0,sb=0;Nat g;sbn3_int *x=nullptr,*y=nullptr;
    void *scratch;size_t bytes;unsigned rows=0;Metrics *metrics=nullptr;Policy policy{};
};
Policy baseline_policy(unsigned workers,bool extended){
    Policy p;p.word32=true;p.leaf_pipeline=workers>1;p.word_ifma=true;p.partial_tail=true;
    p.fft_matrix=p.fft_vector=p.joint_vector=true;
    // Both serial and parallel ordinary reductions use the same certified
    // middle recurrence. Extended W1 retains its measured serial route.
    if(!extended||workers>1)p.middle_recursive=p.middle_cache=true;
    // Small independent products use serial backend recipes in a bounded
    // batch; large products retain their own team. No extra public scratch.
    p.parallel_seven=p.parallel_adjust=workers>1;
    return p;
}
Policy default_policy(unsigned workers,bool extended){
    Policy p=baseline_policy(workers,extended);p.compact_forward=true;
    // Serial coefficients benefit from fewer outer rounds. Parallel work
    // gains little from that wider prefix, while the half uses less live
    // storage. These are fixed worker classes, never an input-size search.
    // The ordinary terminal retains its existing partial-tail reduction.
    p.outer=extended&&workers==1?2:1;
    if(extended){p.deferred=p.column_fft=p.middle_recursive=p.middle_cache=true;}
    return p;
}
static void body(void *ptr,sbn3_team_scope *scope) {
    auto &c=*static_cast<Call*>(ptr);
    // A common radix power cancels from both operands without changing
    // either Bezout coefficient. Use views of the nonzero suffixes; restore
    // that power only in the GCD output. Random operands exit after one read.
    size_t low=0;const size_t common=std::min(c.a.n,c.b.n);
    while(low<common&&!(c.a.p[low]|c.b.p[low]))++low;
    struct RestoreLow {
        Nat &g;size_t words;
        ~RestoreLow(){if(words){g.p-=words;g.cap+=words;memset(g.p,0,words*8);g.n+=words;}}
    } restore{c.g,low};
    if(low){
        require(low<common&&low<c.g.cap,SBN3_FATAL_ARGUMENT,"GCD scaled output capacity");
        c.a.p+=low;c.a.n-=low;c.b.p+=low;c.b.n-=low;c.g.p+=low;c.g.cap-=low;
    }
    auto frame=Frame::external(c.scratch,c.bytes);Engine e{frame,scope,c.policy,c.metrics};
    const size_t n=std::min(c.a.n,c.b.n);Pair pair{allocate(frame,round_words(n+3)),allocate(frame,round_words(n+3))};
    // Preserve optional-output storage bounds. A lone coefficient can use
    // a one-row eager fallback at the root unless an initial quotient mixes
    // the two post-remainder components needed for that requested output.
    const bool a_long=c.a.n>c.b.n,b_long=c.b.n>c.a.n;
    const bool need_quotient=(a_long&&c.y)||(b_long&&c.x);
    const unsigned column_rows=need_quotient?3:c.rows;
    const size_t quotient_capacity=need_quotient?round_words(std::max(c.a.n,c.b.n)-n+2):0;
    if(c.policy.deferred&&c.rows&&n>e.policy.leaf&&
       deferred_suffix_bytes(n,scope?scope->width:1,column_rows)+(quotient_capacity?8*quotient_capacity+63:0)<=frame.capacity()-frame.used()){
        Nat quotient{};if(need_quotient)quotient=allocate(frame,quotient_capacity);
        if(a_long){assign(pair.b,c.b);stream_remainder(pair.a,c.a,c.b,e,need_quotient?&quotient:nullptr);
            if(!pair.a.n){assign(pair.a,pair.b);if(need_quotient){U one=1;require(!limbs::sub_from(quotient.p,quotient.n,&one,1),SBN3_FATAL_MATH,"GCD deferred initial quotient");quotient.trim();}}}
        else if(b_long){assign(pair.a,c.a);stream_remainder(pair.b,c.b,c.a,e,need_quotient?&quotient:nullptr);
            if(!pair.b.n){assign(pair.b,pair.a);if(need_quotient){U one=1;require(!limbs::sub_from(quotient.p,quotient.n,&one,1),SBN3_FATAL_MATH,"GCD deferred initial quotient");quotient.trim();}}}
        else {assign(pair.a,c.a);assign(pair.b,c.b);}
        deferred_column(c.g,pair,e,column_rows);
        if(a_long&&c.y){FrameMark lifetime(frame);auto t=allocate(frame,quotient.n+pair.b.n+2);e.multiply(t,quotient,pair.b);add(t,t,pair.a);
            value(c.y,t,!c.sb);value(c.x,pair.b,c.sa);}
        else if(b_long&&c.x){FrameMark lifetime(frame);auto t=allocate(frame,quotient.n+pair.a.n+2);e.multiply(t,quotient,pair.a);add(t,t,pair.b);
            value(c.x,t,c.sa);value(c.y,pair.a,!c.sb);}
        else{value(c.x,pair.b,c.sa);value(c.y,pair.a,!c.sb);}
        if(c.metrics)c.metrics->peak=frame.peak();return;
    }
    Matrix cof{};Matrix *cofptr=nullptr;
    if(c.rows){cof=make_matrix(frame,round_words(c.a.n+3),round_words(c.b.n+3),c.rows);cofptr=&cof;}
    if(c.a.n>c.b.n) {
        assign(pair.b,c.b);Nat *q=c.rows&1?&cof.e[0][1]:nullptr;
        stream_remainder(pair.a,c.a,c.b,e,q);
        if(!pair.a.n){if(!cofptr){assign(c.g,pair.b);return;}
            assign(pair.a,pair.b);if(q){U one=1;require(!limbs::sub_from(q->p,q->n,&one,1),SBN3_FATAL_MATH,"GCD initial quotient");q->trim();}}
    } else if(c.b.n>c.a.n) {
        assign(pair.a,c.a);Nat *q=c.rows&2?&cof.e[1][0]:nullptr;
        stream_remainder(pair.b,c.b,c.a,e,q);
        if(!pair.b.n){if(!cofptr){assign(c.g,pair.a);return;}
            assign(pair.b,pair.a);if(q){U one=1;require(!limbs::sub_from(q->p,q->n,&one,1),SBN3_FATAL_MATH,"GCD initial quotient");q->trim();}}
    } else {assign(pair.a,c.a);assign(pair.b,c.b);}
    execution(c.g,cofptr,pair,e);
    if(c.x)value(c.x,cof.e[1][1],c.sa);
    if(c.y)value(c.y,cof.e[0][1],!c.sb);
    if(c.metrics)c.metrics->peak=frame.peak();
}
void policy_query(size_t an,size_t bn,unsigned workers,sbn3_gcd_info &i) {
    const sbn3_gcd_options o{workers};require(query(an,bn,0,&o,i,true)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"GCD policy domain");
}
size_t policy_gcd(U *out,Span a,Span b,void *scratch,size_t bytes,const Policy &policy,Metrics *metrics,sbn3_team *team,bool public_storage) {
    sbn3_gcd_info quoted{};
    if(public_storage){const sbn3_gcd_options o{team?team->width:1};
        require(query(a.n,b.n,0,&o,quoted)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"GCD bounded policy domain");
    }else policy_query(a.n,b.n,team?team->width:1,quoted);
    require(bytes>=quoted.scratch_bytes,SBN3_FATAL_WORKSPACE,"GCD policy scratch",quoted.scratch_bytes,bytes);
    if(public_storage&&std::max(a.n,b.n)<=SBN3_GCD_SMALL_LIMBS)
        return sbn3_gcd_small(out,a.p,a.n,b.p,b.n,scratch);
    a=normalized(a);b=normalized(b);
    if(!a.n||!b.n){const auto v=a.n?a:b;if(v.n)memcpy(out,v.p,v.n*8);return v.n;}
    Call c{a,b,0,0,{out,0,std::min(a.n,b.n)},nullptr,nullptr,scratch,quoted.scratch_bytes,0,metrics,policy};
    if(team)sbn3_team_run(team,body,&c);else if(std::max(a.n,b.n)<direct_product_limit)body(&c,nullptr);else serial_team_run(body,&c);return c.g.n;
}
static void spans(const sbn3_gcd_info &i,Span a,Span b,void *g,size_t gn,sbn3_int *x,sbn3_int *y,void *scratch,size_t bytes) {
    require(bytes>=i.scratch_bytes,SBN3_FATAL_WORKSPACE,"GCD scratch",i.scratch_bytes,bytes);
    if constexpr(SBN3_CHECK_SMALL) {
        require(!i.scratch_bytes||(scratch&&!(uintptr_t(scratch)&(i.scratch_alignment-1))),SBN3_FATAL_ARGUMENT,"GCD scratch alignment");
        const void *w[]{g,x?x->data:nullptr,y?y->data:nullptr,scratch};
        const size_t sizes[]{gn*8,x?x->capacity*8:0,y?y->capacity*8:0,bytes};
        valid_span(a.p,a.n*8,"GCD input A");valid_span(b.p,b.n*8,"GCD input B");
        for(unsigned j=0;j<4;++j){valid_span(w[j],sizes[j],"GCD writable span");
            require(!overlaps(w[j],sizes[j],a.p,a.n*8)&&!overlaps(w[j],sizes[j],b.p,b.n*8),SBN3_FATAL_ARGUMENT,"GCD input alias");
            for(unsigned k=0;k<j;++k)require(!overlaps(w[j],sizes[j],w[k],sizes[k]),SBN3_FATAL_ARGUMENT,"GCD output alias");}
    }
}
}
extern "C" size_t sbn3_gcd_scratch_bytes(size_t a,size_t b,unsigned workers) {
    if(workers>32)return SIZE_MAX;
    if(std::max(a,b)<=SBN3_GCD_SMALL_LIMBS)return sbn3_gcd_small_scratch_bytes(a,b);
    sbn3_gcd_info i{};const sbn3_gcd_options o{workers};
    return gcd::query(a,b,0,&o,i)==SBN3_SUPPORTED?i.scratch_bytes:SIZE_MAX;
}
extern "C" size_t sbn3_exgcd_scratch_bytes(size_t a,size_t b,unsigned mask,unsigned workers) {
    if(!mask)return sbn3_gcd_scratch_bytes(a,b,workers);
    sbn3_gcd_info i{};const sbn3_gcd_options o{workers};
    return gcd::query(a,b,mask,&o,i)==SBN3_SUPPORTED?i.scratch_bytes:SIZE_MAX;
}
extern "C" sbn3_query_result sbn3_gcd_query(size_t a,size_t b,const sbn3_gcd_options *o,sbn3_gcd_info *i) {
    require(i,SBN3_FATAL_ARGUMENT,"GCD query output");return gcd::query(a,b,0,o,*i);
}
extern "C" sbn3_query_result sbn3_exgcd_query(size_t a,size_t b,unsigned mask,const sbn3_gcd_options *o,sbn3_gcd_info *i) {
    require(i,SBN3_FATAL_ARGUMENT,"extended GCD query output");return gcd::query(a,b,mask,o,*i);
}
extern "C" size_t sbn3_gcd(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,
                           void *scratch,size_t bytes,sbn3_team *team) {
    if(std::max(an,bn)<=SBN3_GCD_SMALL_LIMBS){
        if constexpr(SBN3_CHECK_SMALL)require(bytes>=sbn3_gcd_small_scratch_bytes(an,bn),SBN3_FATAL_WORKSPACE,"small general GCD scratch");
        return sbn3_gcd_small(out,a,an,b,bn,scratch);
    }
    const sbn3_gcd_options o{team?team->width:1};sbn3_gcd_info i{};
    require(gcd::query(an,bn,0,&o,i)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"GCD domain");
    auto av=gcd::normalized({a,an}),bv=gcd::normalized({b,bn});
    const size_t bound=av.n&&bv.n?std::min(an,bn):av.n||bv.n?std::max(an,bn):0;
    gcd::spans(i,{a,an},{b,bn},out,bound,nullptr,nullptr,scratch,bytes);
    if(!av.n||!bv.n){const auto v=av.n?av:bv;if(v.n)memcpy(out,v.p,v.n*8);return v.n;}
    if(std::min(av.n,bv.n)<=2)return gcd::finish(out,av,bv,nullptr);
    gcd::Call call{av,bv,0,0,{out,0,bound},nullptr,nullptr,scratch,i.scratch_bytes};
    call.policy=gcd::default_policy(o.workers);
    if(team)sbn3_team_run(team,gcd::body,&call);else serial_team_run(gcd::body,&call);return call.g.n;
}
void gcd::policy_exgcd(sbn3_int *g,sbn3_int *x,sbn3_int *y,sbn3_int_view a,sbn3_int_view b,
                               void *scratch,size_t bytes,sbn3_team *team,const Policy &policy,Metrics *metrics) {
    require(g,SBN3_FATAL_ARGUMENT,"extended GCD result");
    const unsigned mask=(x?SBN3_EXGCD_X:0)|(y?SBN3_EXGCD_Y:0);
    const sbn3_gcd_options o{team?team->width:1};sbn3_gcd_info i{};
    require(gcd::query(a.size,b.size,mask,&o,i)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"extended GCD domain");
    auto av=gcd::normalized({a.data,a.size}),bv=gcd::normalized({b.data,b.size});
    gcd::spans(i,{a.data,a.size},{b.data,b.size},g->data,g->capacity,x,y,scratch,bytes);
    if(!av.n||!bv.n){const auto v=av.n?av:bv;const uint64_t one=1;
        gcd::value(g,v,false);gcd::value(x,{&one,size_t(bool(av.n))},a.negative);
        gcd::value(y,{&one,size_t(bool(bv.n))},b.negative);return;}
    if(!mask){g->size=sbn3_gcd(g->data,a.data,a.size,b.data,b.size,scratch,bytes,team);g->negative=0;return;}
    if(av.n==1&&bv.n==1){gcd::small_extended(g,x,y,av.p[0],bv.p[0],a.negative,b.negative);return;}
    const unsigned rows=(x?2:0)|(y?1:0);
    gcd::Call call{av,bv,a.negative,b.negative,{g->data,0,g->capacity},x,y,scratch,i.scratch_bytes,rows};
    call.policy=policy;call.metrics=metrics;
    if(team)sbn3_team_run(team,gcd::body,&call);else if(std::max(av.n,bv.n)<gcd::direct_product_limit)gcd::body(&call,nullptr);else serial_team_run(gcd::body,&call);
    g->size=call.g.n;g->negative=0;
}

extern "C" void sbn3_int_exgcd(sbn3_int *g,sbn3_int *x,sbn3_int *y,sbn3_int_view a,sbn3_int_view b,
                               void *scratch,size_t bytes,sbn3_team *team) {
    gcd::policy_exgcd(g,x,y,a,b,scratch,bytes,team,gcd::default_policy(team?team->width:1,true));
}
