// Original libsbn code; LGPL-3.0-or-later.
#include "gcd/internal.hpp"
#include "product/program.hpp"
#include "product/local_product.hpp"
#include "backend/u52/kernels.hpp"
#include "algorithms/small_division.hpp"
#include "sbn3/divrem.h"
namespace sbn::v3::gcd {
size_t arithmetic_envelope(size_t n,unsigned workers) {
    // A matrix term is bounded by its nonnegative result coefficient, or is
    // a tail times a half-reduction coefficient. Consequently its shorter
    // argument is at most ceil(n/2)+2, its longer at most n+2. Conversion
    // guards are included. Other episodes must fit before preparation.
    const size_t m=std::min<size_t>(size_t(1)<<20,std::max<size_t>(n+2,16));
    return std::max(u52::scratch_bytes(m,std::min(m,(n+1)/2+2)),u52::divide_scratch_bytes(2*m,m))+
        16384+4096*workers;
}
void Engine::multiply(Nat &out,Span a,Span b) {
    a=normalized(a);b=normalized(b);
    if(!a.n||!b.n){if(metrics)++metrics->products;out.n=0;return;}
    require(a.n+b.n<=out.cap,SBN3_FATAL_WORKSPACE,"HGCD product output",a.n+b.n,out.cap);
    if(a.n<b.n)std::swap(a,b);
    if(metrics)++metrics->products;
    if(b.n==1){const U c=sbn3i_mul_1(out.p,a.p,long(a.n),b.p[0]);out.p[a.n]=c;out.n=a.n+bool(c);return;}
    FrameMark mark(space);
    if(b.n<direct_product_limit){u52::multiply(out.p,a.p,a.n,b.p,b.n,space);}
    else {
        ProductProgramPlan p{};sbn3_mul_options o{};o.borrow_output=1;
        o.workers=a.n+b.n>=policy.parallel?scope->width:1;
        const bool supported=product_program_query(a.n,b.n,o,p)==SBN3_SUPPORTED;
        const size_t staging=supported&&p.info.output_limbs>out.cap?p.info.output_limbs*8+64:0;
        const size_t need=supported?p.prepared_bytes+p.info.workspace_alignment+p.info.workspace_bytes+staging+256:SIZE_MAX;
        if(supported&&need<=space.capacity()-space.used()) {
            if(metrics)++metrics->product_width[p.info.workers];
            auto tables=space.subframe(p.prepared_bytes+128,128);
            const auto program=product_program_prepare(p,tables);
            auto work=space.subframe(p.info.workspace_bytes,p.info.workspace_alignment);
            U *dest=staging?space.alloc<U>(p.info.output_limbs):out.p;
            product_program_execute(program,work,scope,{a.p,a.n},{b.p,b.n},{dest,p.info.output_limbs});
            if(staging)memcpy(out.p,dest,(a.n+b.n)*8);
        } else {if(metrics)++metrics->product_fallback;u52::multiply(out.p,a.p,a.n,b.p,b.n,space);}
    }
    out.n=a.n+b.n;out.trim();
}
void Engine::multiply_pair(Nat &x,Nat &y,Span common,Span a,Span b) {
    common=normalized(common);a=normalized(a);b=normalized(b);
    if(!policy.pairs||!common.n||!a.n||!b.n||std::min(common.n,std::max(a.n,b.n))<128) {
        multiply(x,common,a);multiply(y,common,b);return;
    }
    FrameMark mark(space);ProductProgramPlan p{};sbn3_mul_options o{};
    o.workers=common.n+std::max(a.n,b.n)>=policy.parallel?scope->width:1;
    const bool supported=product_program_query(common.n,std::max(a.n,b.n),o,p)==SBN3_SUPPORTED;
    const bool cached=supported&&p.pair_workspace_bytes&&
        (policy.retain_pair_cache||p.pair_workspace_bytes<=p.info.workspace_bytes);
    const size_t workspace=cached?p.pair_workspace_bytes:p.info.workspace_bytes;
    if(supported&&(cached||(p.contract&program_bounded_inputs))&&p.an+p.bn<=x.cap&&p.an+p.bn<=y.cap&&
       p.prepared_bytes+workspace+p.info.workspace_alignment+256<=space.capacity()-space.used()) {
        auto tables=space.subframe(p.prepared_bytes+128,128);const auto program=product_program_prepare(p,tables);
        auto work=space.subframe(workspace,p.info.workspace_alignment);
        if(cached)product_program_pair(program,work,scope,{common.p,common.n},{a.p,a.n},{b.p,b.n},
                                      {x.p,x.cap},{y.p,y.cap});
        else {
            product_program_bounded(program,work,scope,{common.p,common.n},{a.p,a.n},{x.p,x.cap});
            product_program_bounded(program,work,scope,{common.p,common.n},{b.p,b.n},{y.p,y.cap});
        }
        x.n=common.n+a.n;y.n=common.n+b.n;x.trim();y.trim();
        if(metrics){metrics->pairs+=cached;++metrics->pair_preparations;metrics->products+=2;}
    } else {multiply(x,common,a);multiply(y,common,b);}
}
void Engine::divide(Nat &q,Nat &r,Span a,Span b) {
    a=normalized(a);b=normalized(b);require(b.n,SBN3_FATAL_MATH,"HGCD zero divisor");
    require(r.cap>=b.n&&q.cap>=(a.n>=b.n?a.n-b.n+1:0),SBN3_FATAL_WORKSPACE,"HGCD division outputs");
    if(a.n<b.n){q.n=0;assign(r,a);return;}
    FrameMark mark(space);if(metrics)++metrics->divisions;
    bool local=false;
    if(b.n<=local_division_max_limbs) {
        const auto bytes=sbn3_divrem_small_scratch_bytes(a.n,b.n);
        // The optional local FFT recipe is not the u52 fallback covered by
        // arithmetic_envelope. Admit its complete quote before selecting it.
        if(bytes+63<=space.capacity()-space.used()){
            q.n=sbn3_divrem_small(q.p,r.p,a.p,a.n,b.p,b.n,space.allocate(bytes,64));local=true;
        }
    }
    if(!local&&a.n-b.n<6) {
        q.n=divrem_words::schoolbook(q.p,r.p,a.p,a.n,b.p,b.n,space.alloc<U>(a.n+b.n+2));
    } else if(!local) {
        u52::divide(q.p,r.p,a.p,a.n,b.p,b.n,space);q.n=a.n-b.n+1;q.trim();
    }
    r.n=b.n;r.trim();
}
void stream_remainder(Nat &out,Span input,Span divisor,Engine &e,Nat *quotient) {
    divisor=normalized(divisor);require(divisor.n,SBN3_FATAL_MATH,"GCD stream divisor");
    require(out.cap>=divisor.n&&(!quotient||quotient->cap>=(input.n>=divisor.n?input.n-divisor.n+1:0)),
            SBN3_FATAL_WORKSPACE,"GCD streamed division outputs");
    if(input.n<divisor.n){assign(out,input);if(quotient)quotient->n=0;return;}
    if(divisor.n>=3&&divisor.n<=local_division_max_limbs){
        if(const size_t block=local_division_block(input.n,divisor.n)){
            const auto plan=local_division_auto_plan(input.n,divisor.n,block);
            const size_t bytes=quotient?plan.storage_bytes:local_division_remainder_bytes(plan);
            if(bytes+63<=e.space.capacity()-e.space.used()){
                FrameMark mark(e.space);auto work=e.space.subframe(bytes);
                local_division_execute(plan,quotient?quotient->p:nullptr,out.p,input.p,input.n,divisor.p,work);
                out.n=divisor.n;out.trim();
                if(quotient){quotient->n=input.n-divisor.n+1;quotient->trim();}
                if(e.metrics)++e.metrics->divisions;return;
            }
        }
    }
    FrameMark mark(e.space);const size_t block=divisor.n;
    auto chunk=allocate(e.space,2*block+2),q=allocate(e.space,block+3);
    const size_t quotient_words=input.n-divisor.n+1;
    if(quotient)quotient->n=quotient_words;
    out.n=0;
    for(size_t end=input.n;end;) {
        const size_t begin=end>block?end-block:0,take=end-begin;
        memcpy(chunk.p,input.p+begin,take*8);
        if(out.n)memcpy(chunk.p+take,out.p,out.n*8);
        chunk.n=take+out.n;chunk.trim();e.divide(q,out,chunk,divisor);
        if(quotient){const size_t stored=begin<quotient_words?std::min(take,quotient_words-begin):0;
            require(q.n<=stored,SBN3_FATAL_MATH,"GCD quotient block bound");
            if(q.n)memcpy(quotient->p+begin,q.p,q.n*8);
            if(q.n<stored)memset(quotient->p+begin+q.n,0,(stored-q.n)*8);}
        end=begin;
    }
    if(quotient)quotient->trim();
}
}
