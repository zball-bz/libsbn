/* hgcd.c and hgcd_appr.c (matrix-only second-child control).

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2003-2005, 2008, 2011, 2012 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

// Adapted control/stop structure; libsbn views, kernels and planned scratch.
#include "gcd/internal.hpp"
#include "runtime/linux/sync.hpp"
namespace sbn::v3::gcd {
// Bounded SPSC stream. Producer alone owns residuals/scratch;
// consumer alone owns the matrix until flush joins all published heads.
struct HeadStream {
    static constexpr unsigned capacity=64,batch=4;
    alignas(64) uint64_t published=0;
    alignas(64) uint64_t consumed=0;
    alignas(64) uint32_t closed=0;
    alignas(64) gcd_small::Matrix records[capacity];
    uint64_t produced=0,available=capacity,wraps=0,flushes=0;
};
void head_stream_push(HeadStream*s,const gcd_small::Matrix&h){
    if(s->produced==s->available){++s->wraps;
        do{s->available=__atomic_load_n(&s->consumed,__ATOMIC_ACQUIRE)+s->capacity;if(s->produced==s->available)os::pause();}while(s->produced==s->available);
    }
    s->records[s->produced%s->capacity]=h;++s->produced;
    if(!(s->produced%s->batch))__atomic_store_n(&s->published,s->produced,__ATOMIC_RELEASE);
}
void head_stream_flush(HeadStream*s){
    ++s->flushes;
    __atomic_store_n(&s->published,s->produced,__ATOMIC_RELEASE);
    while(__atomic_load_n(&s->consumed,__ATOMIC_ACQUIRE)!=s->produced)os::pause();
}
template<class E>static bool pipelined_leaf(Pair&p,size_t stop,Matrix&m,E&e){
    // Establish actual progress before recruiting a helper. Powers of two
    // and short reductions can finish in one step; their initial matrix is
    // tiny and does not justify starting/draining a stream.
    if(!step(p,stop,&m,e))return false;
    if(!compare(p.a,p.b))return true;
    HeadStream stream;struct Call{Pair*p;size_t stop;Matrix*m;E*e;HeadStream*s;}c{&p,stop,&m,&e,&stream};
    sbn3_team_scope two{e.scope->team,e.scope->first,2,false,e.scope->epoch};
    sbn3_team_invoke2(&two,SBN3_PARALLEL_CHILDREN,1,[](void*ptr,sbn3_team_scope*scope){
        auto&c=*static_cast<Call*>(ptr);auto producer=rebind(*c.e,c.e->space,scope);producer.stream=c.s;
        while(step(*c.p,c.stop,c.m,producer)){}
        head_stream_flush(c.s);__atomic_store_n(&c.s->closed,1,__ATOMIC_RELEASE);
    },&c,[](void*ptr,sbn3_team_scope*scope){
        auto&c=*static_cast<Call*>(ptr);Engine consumer{c.e->space,scope,c.e->policy,nullptr};uint64_t done=0;
        for(;;){const uint64_t end=__atomic_load_n(&c.s->published,__ATOMIC_ACQUIRE);
            while(done<end){matrix_word(*c.m,c.s->records[done%c.s->capacity],consumer);++done;}
            __atomic_store_n(&c.s->consumed,done,__ATOMIC_RELEASE);
            if(__atomic_load_n(&c.s->closed,__ATOMIC_ACQUIRE))break;
            while(__atomic_load_n(&c.s->published,__ATOMIC_ACQUIRE)==done&&!__atomic_load_n(&c.s->closed,__ATOMIC_ACQUIRE))os::pause();
        }
    },&c);
    if(e.metrics){++e.metrics->leaf_streams;e.metrics->leaf_stream_wraps+=stream.wraps;e.metrics->leaf_stream_flushes+=stream.flushes;}
    return true;
}
template<class E>static bool run(Pair &p,Matrix *matrix,E &e,bool matrix_only=false,ResidualBound *bound=nullptr) {
    const size_t original=p.size();
    // A value-only root may aim directly at its terminal. For target s,
    // two children of about n-s limbs each remove (n-s)/2 limbs. This
    // avoids another complete half reduction for a small terminal excess.
    const bool tail=!matrix&&e.policy.partial_tail&&original<=2*e.policy.finish;
    const size_t stop=tail?std::max(original/2+e.policy.stop_guard,e.policy.finish-2):original/2+e.policy.stop_guard;
    auto done=[&](bool ok){if(bound)*bound=p.a.n&&p.b.n?residual_bound(p):ResidualBound{};return ok;};
    if(std::min(p.a.n,p.b.n)<=stop||original<=2)return done(false);
    if(e.metrics)++e.metrics->recursive_calls;
    bool success=false;
    if(original>e.policy.leaf) {
        const size_t intermediate=tail?(original+stop)/2+1:3*original/4+e.policy.stop_guard,cut=tail?stop:original/2;
        {
            FrameMark mark(e.space);Matrix local{};
            Matrix *first=matrix;
            if(!first){local=make_matrix(e.space,original-cut);first=&local;}
            if(reduce_high(p,cut,*first,e))success=true;
        }
        while(p.size()>intermediate) {
            if(!step(p,stop,matrix,e))return done(success);
            success=true;
        }
        const size_t n=p.size();
        if(n>stop+2) {
            // A child with guard g stops at floor(child_n/2)+g.
            // Choose its offset so that this target lifts to stop+1,
            // independent of g. The usual +1 formula assumes g==1 and
            // leaves 2*(g-1) unnecessary source words at every right spine.
            const size_t second_cut=2*stop-n+3-2*e.policy.stop_guard;
            FrameMark mark(e.space);auto second=make_matrix(e.space,n-second_cut);
            if(matrix_only){
                // The caller will apply the composed matrix to its original
                // operands. Computing these two child remainders would be
                // dead work. Integer lifting keeps the two guard limbs in
                // second_cut, including abnormal (large) quotient steps.
                auto head=high(p,second_cut);
                ResidualBound child{};
                if(hgcd_matrix_only(head,second,e,bound?&child:nullptr)){
                    if(bound)require(lift_bound(child,second,64*second_cut,0,*bound),
                                     SBN3_FATAL_MATH,"HGCD matrix-only lift certificate");
                    compose(*matrix,second,e);return true;
                }
            }else
            if(reduce_high(p,second_cut,second,e)) {
                if(matrix)compose(*matrix,second,e);
                success=true;
            }
        }
    }
    if(original<=e.policy.leaf&&matrix&&e.policy.leaf_pipeline&&e.scope&&e.scope->width>=2&&
       (!e.metrics||!e.metrics->leaf_observer)){
        // Equal prefixes need no head or matrix update. Avoid starting a
        // consumer that would only drain an empty stream.
        if(compare(p.a,p.b))success=pipelined_leaf(p,stop,*matrix,e)||success;
    }
    else while(step(p,stop,matrix,e))success=true;
    return done(success);
}
bool hgcd(Pair &p,Matrix &m,Engine &e){return run(p,&m,e);}
template<class E>static bool matrix_only_impl(Pair &p,Matrix &m,E &e,ResidualBound *bound){
    if(e.policy.middle_recursive){
        ResidualBound local{};
        if(recursive_middle_matrix(p,m,e,local)){if(bound)*bound=local;return true;}
        // A declined/uncertified window leaves p and the identity matrix
        // intact. Use one conventional subtree, rather than returning a
        // quarter reduction and forcing linear parent correction steps.
        auto fallback=rebind(e,e.space,e.scope);fallback.policy.middle_recursive=false;
        return run(p,&m,fallback,true,bound);
    }
    // The earlier middle prototype returns exact child values, but has no
    // parent certificate interface. Keep it as an independent diagnostic.
    if constexpr(!E::symbolic)if(!bound&&e.policy.middle&&middle_matrix(p,m,e))return true;
    return run(p,&m,e,true,bound);
}
static void compose_outer(Matrix &m,const Matrix &r,Engine &e){
    // The original-size Bezout accumulator is not an internal balanced
    // HGCD matrix. Retain multiplication's own parallelism for this chain.
    Engine outer{e.space,e.scope,e.policy,e.metrics};
    outer.policy.parallel_seven=false;
    compose(m,r,outer);
}
size_t outer_prefix(size_t n,unsigned policy,bool initial){
    if(policy==3)return n-(2*n)/3;
    if(policy==2||(policy==4&&!initial))return (2*n+2)/3;
    return n-n/2;
}
template<class E>static bool reduce_impl(Pair &p,Matrix *accumulate,E &e,bool initial) {
    const size_t n=p.size(),before=e.metrics?std::max(bit_length(p.a),bit_length(p.b)):0;
    auto result=[&](bool ok,size_t prefix){if(e.metrics){++e.metrics->outer_calls;e.metrics->outer_input_words+=n;e.metrics->outer_prefix_words+=prefix;
        const size_t after=std::max(bit_length(p.a),bit_length(p.b));e.metrics->outer_drop_bits+=before>after?before-after:0;}return ok;};
    if(!accumulate&&(!e.policy.outer||(e.policy.partial_tail&&n<=2*e.policy.finish)))
        return result(run(p,nullptr,e),0);
    const size_t prefix=outer_prefix(n,e.policy.outer,initial),cut=n-prefix;
    FrameMark mark(e.space);auto m=make_matrix(e.space,prefix);
    if(!reduce_high(p,cut,m,e))return result(false,prefix);
    if(accumulate)compose_outer(*accumulate,m,e);return result(true,prefix);
}
static size_t finish_owned(Nat &out,Pair &p,Engine &e) {
    PhaseMeter phase(e.metrics,Phase::finish,p.size());
    if(std::min(p.a.n,p.b.n)<=2)return out.n=finish(out.p,p.a,p.b,nullptr);
    const size_t bytes=sbn3_gcd_small_scratch_bytes(p.a.n,p.b.n);
    const uintptr_t base=uintptr_t(e.space.data()),begin=uintptr_t(p.a.p),end=uintptr_t(p.b.p+p.b.cap);
    if(p.a.p+p.a.cap==p.b.p&&begin>=base&&end==base+e.space.used()) {
        const size_t extent=end-begin;
        if(bytes>extent)e.space.allocate(bytes-extent,8);
        auto *b=p.a.p+bytes/16;
        if(b!=p.b.p)memmove(b,p.b.p,p.b.n*8);
        // The private bridge accepts this owned layout; the public preserving
        // entry retains its disjoint input/workspace contract.
        return out.n=finish(out.p,{p.a.p,p.a.n},{b,p.b.n},p.a.p);
    }
    FrameMark mark(e.space);return out.n=finish(out.p,p.a,p.b,e.space.allocate(bytes,64));
}
size_t execution(Nat &out,Matrix *cofactors,Pair &p,Engine &e) {
    // A cofactor-bearing terminal is a different leaf from plain GCD.
    // Reducing 768 words while touching original-size cofactors at every
    // word step defeats the recursive batching. Keep those updates local.
    const size_t terminal=cofactors?e.policy.leaf:e.policy.finish;
    bool initial=true;
    while(p.size()>terminal) {
        if(!cofactors&&std::min(p.a.n,p.b.n)<=2)return out.n=finish(out.p,p.a,p.b,nullptr);
        const bool first=initial;initial=false;
        if(!reduce(p,cofactors,e,first)) {
            if(!step(p,0,cofactors,e))break;
        }
    }
    if(cofactors) {
        size_t cn=0;for(unsigned i=0;i<2;++i)if(cofactors->rows&(1u<<i))for(unsigned j=0;j<2;++j)cn=std::max(cn,cofactors->e[i][j].n);
        if(cn>1&&compare(p.a,p.b)!=0) {
            FrameMark mark(e.space);const size_t cap=round_words(p.size()+3);
            auto last=make_matrix(e.space,cap,cap,3);
            while(step(p,0,&last,e)){}
            compose_outer(*cofactors,last,e);
        } else while(step(p,0,cofactors,e)){}
        require(compare(p.a,p.b)==0,SBN3_FATAL_MATH,"extended GCD terminal equality");assign(out,p.a);
    } else {
        require(p.size()<=SBN3_GCD_SMALL_LIMBS||compare(p.a,p.b)==0,SBN3_FATAL_MATH,"GCD terminal domain");
        if(compare(p.a,p.b)==0)assign(out,p.a);
        else finish_owned(out,p,e);
    }
    return out.n;
}
bool hgcd_matrix_only(Pair&p,Matrix&m,Engine&e,ResidualBound*b){return matrix_only_impl(p,m,e,b);}
bool hgcd_matrix_only(Pair&p,Matrix&m,SymbolEngine&e,ResidualBound*b){SymbolTransaction tx(e);bool ok=matrix_only_impl(p,m,e,b);if(ok)tx.commit();return ok;}
bool hgcd(Pair&p,Matrix&m,SymbolEngine&e){return run(p,&m,e);}
bool reduce(Pair&p,Matrix*m,Engine&e,bool i){return reduce_impl(p,m,e,i);}
bool reduce(Pair&p,Matrix*m,SymbolEngine&e,bool i){return reduce_impl(p,m,e,i);}
int symbol_execution(Pair &p,SymbolEngine &e){
    bool initial=true;
    while(p.size()>e.policy.finish&&std::min(p.a.n,p.b.n)>2){
        bool first=initial;initial=false;
        if(!reduce(p,nullptr,e,first)&&!step(p,0,nullptr,e))break;
    }
    if(compare(p.a,p.b)==0)return p.a.n==1&&p.a.p[0]==1?e.symbol->sign():0;
    if(std::min(p.a.n,p.b.n)<=2)return symbol_finish(p.a.p,p.a.n,p.b.p,p.b.n,nullptr,*e.symbol);
    const size_t bytes=sbn3_gcd_small_scratch_bytes(p.a.n,p.b.n);
    const uintptr_t begin=uintptr_t(p.a.p),end=uintptr_t(p.b.p+p.b.cap);
    if(p.a.p+p.a.cap==p.b.p&&end==uintptr_t(e.space.data())+e.space.used()){
        if(bytes>end-begin)e.space.allocate(bytes-(end-begin),8);
        auto*b=p.a.p+bytes/16;if(b!=p.b.p)memmove(b,p.b.p,p.b.n*8);
        return symbol_finish(p.a.p,p.a.n,b,p.b.n,p.a.p,*e.symbol);
    }
    FrameMark mark(e.space);return symbol_finish(p.a.p,p.a.n,p.b.p,p.b.n,e.space.allocate(bytes,64),*e.symbol);
}
}
