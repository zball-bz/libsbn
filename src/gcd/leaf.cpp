/* hgcd_step.c.

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
namespace sbn::v3::gcd {
template<class Observer>static U word_matrix_impl(U a,U b,gcd_small::Matrix &m,Observer observe) {
    require(a&&b,SBN3_FATAL_ARGUMENT,"word extended GCD inputs");m={{{1,0},{0,1}}};
    while(a!=b) {
        const unsigned col=a>b?1:0;U &v=a>b?a:b;const U other=a>b?b:a;
        const U q=(v-1)/other;v-=q*other;observe(col,q);
        for(unsigned row=0;row<2;++row){const W value=W(q)*m.u[row][1-col]+m.u[row][col];
            require(!(value>>64),SBN3_FATAL_MATH,"word Bezout coefficient bound");m.u[row][col]=U(value);}
    }
    return a;
}
U word_matrix(U a,U b,gcd_small::Matrix &m){return word_matrix_impl(a,b,m,NoSymbol{});}
template<class E>static bool step_impl(Pair &p,size_t stop,Matrix *matrix,E &e) {
    PhaseMeter phase(e.metrics,Phase::leaf,p.size());
    const size_t n=p.size();if(std::min(p.a.n,p.b.n)<=stop)return false;
    if(n==1&&!stop) {
        if(p.a.p[0]==p.b.p[0])return false;
        gcd_small::Matrix words;const U g=word_matrix_impl(p.a.p[0],p.b.p[0],words,observer(e));
        if(matrix){Matrix last{};for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)last.e[i][j]={&words.u[i][j],size_t(bool(words.u[i][j])),1};
            compose(*matrix,last,e);}
        word(p.a,g);word(p.b,g);return true;
    }
    if(n>=2&&std::min(p.a.n,p.b.n)+1>=n) {
        memset(p.a.p+p.a.n,0,(n-p.a.n)*8);memset(p.b.p+p.b.n,0,(n-p.b.n)*8);
        const U mask=p.a.p[n-1]|p.b.p[n-1];
        if(n>stop+1||mask>=4) {
            unsigned shift=n==stop+1?0:unsigned(__builtin_clzll(mask));
            auto upper=[&](const U *a,size_t at){return shift?(a[at]<<shift)|(at?a[at-1]>>(64-shift):0):a[at];};
            gcd_small::Matrix h;
            if(gcd_small::head_matrix<33,true>(upper(p.a.p,n-1),upper(p.a.p,n-2),
                                              upper(p.b.p,n-1),upper(p.b.p,n-2),&h,observer(e))) {
                if(e.metrics&&e.metrics->leaf_observer)e.metrics->leaf_observer(e.metrics->leaf_context,p,matrix,stop,h);
                if(matrix){if(e.stream)head_stream_push(e.stream,h);else matrix_word(*matrix,h,e);}
                if(e.policy.word32)apply_head_word32(p.a.p,p.b.p,n,h);
                else apply_head(p.a.p,p.b.p,n,h);
                p.a.n=p.b.n=n;p.a.trim();p.b.trim();
                require(p.a.n>stop&&p.b.n>stop,SBN3_FATAL_MATH,"HGCD head crossed stop");
                if(e.metrics)++e.metrics->heads;return true;
            }
        }
    }
    if(e.stream)head_stream_flush(e.stream);
    const int order=compare(p.a,p.b);if(!order)return false;
    Nat &large=order>0?p.a:p.b,&small=order>0?p.b:p.a;
    FrameMark mark(e.space);auto r=allocate(e.space,large.n+2);
    subtract(r,large,small);if(r.n<=stop)return false;
    auto q=allocate(e.space,large.n-small.n+2);
    if(compare(r,small)<0)word(q,1);
    else {
        e.divide(q,r,large,small);
        if(r.n<=stop) {
            U one=1;require(!limbs::sub_from(q.p,q.n,&one,1),SBN3_FATAL_MATH,"HGCD quotient decrement");q.trim();
            add(r,r,small);
        }
    }
    require(q.n&&r.n>stop,SBN3_FATAL_MATH,"HGCD bounded quotient");
    observer(e)(order>0?1:0,q.p[0]);
    if(matrix)matrix_shear(*matrix,order>0?1:0,q,e);
    assign(large,r);return true;
}
bool step(Pair &p,size_t s,Matrix*m,Engine&e){return step_impl(p,s,m,e);}
bool step(Pair &p,size_t s,Matrix*m,SymbolEngine&e){return step_impl(p,s,m,e);}
}
