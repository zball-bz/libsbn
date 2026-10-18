/* matrix22_mul.c.

   Contributed by Niels Möller and Marco Bodrato.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2003-2005, 2008, 2009 Free Software Foundation, Inc.

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

// Bodrato/GNU MP seven-product schedule, adapted to normalized signed views.
#include "gcd/internal.hpp"
#include "product/independent.hpp"
namespace sbn::v3::gcd {
static bool signed_sum(Nat &out,Span a,bool an,Span b,bool bn) {
    bool sign=an;
    if(an==bn)add(out,a,b);
    else if(compare(a,b)>=0)subtract(out,a,b);
    else{subtract(out,b,a);sign=bn;}
    return out.n&&sign;
}
// Original bounded parallel schedule. Unlike the in-place schedule below,
// all seven bilinear inputs remain immutable until every product completes.
bool compose_seven_parallel(Matrix &r,const Matrix &m,Engine &e){
    if(r.rows!=3||!e.scope||e.scope->width<2)return false;
    size_t rn=0,mn=0,cap=SIZE_MAX;
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){rn=std::max(rn,r.e[i][j].n);mn=std::max(mn,m.e[i][j].n);cap=std::min(cap,r.e[i][j].cap);}
    if(std::min(rn,mn)<96||rn+mn<e.policy.batch_min||rn+mn+3>cap)return false;
    const size_t values=8*(5*(rn+3)+5*(mn+3)+7*(rn+mn+6))+2048;
    if(values>e.space.capacity()-e.space.used())return false;
    FrameMark lifetime(e.space);Nat s[5],t[5],p[7];bool ss[5]{},ts[5]{},ps[7]{};
    for(auto &v:s)v=allocate(e.space,rn+3);for(auto &v:t)v=allocate(e.space,mn+3);
    for(auto &v:p)v=allocate(e.space,rn+mn+6);
    const auto &a=r.e[0][0],&b=r.e[0][1],&c=r.e[1][0],&d=r.e[1][1];
    const auto &ee=m.e[0][0],&f=m.e[0][1],&g=m.e[1][0],&h=m.e[1][1];
    add(s[0],a,d);add(s[1],c,d);add(s[2],a,b);
    ss[3]=signed_sum(s[3],c,false,a,true);ss[4]=signed_sum(s[4],b,false,d,true);
    add(t[0],ee,h);ts[1]=signed_sum(t[1],f,false,h,true);ts[2]=signed_sum(t[2],g,false,ee,true);
    add(t[3],ee,f);add(t[4],g,h);
    const Span left[]{s[0],s[1],a,d,s[2],s[3],s[4]},right[]{t[0],ee,t[1],t[2],h,t[3],t[4]};
    ps[2]=ts[1];ps[3]=ts[2];ps[5]=ss[3];ps[6]=ss[4];
    product::IndependentProduct jobs[7];
    for(unsigned j=0;j<7;++j)jobs[j]={{left[j].p,left[j].n},{right[j].p,right[j].n},{p[j].p,p[j].cap}};
    if(!product::independent_products(jobs,7,e.space,e.scope))return false;
    for(unsigned j=0;j<7;++j){p[j].n=left[j].n+right[j].n;p[j].trim();if(!p[j].n)ps[j]=false;}
    auto &o00=r.e[0][0],&o01=r.e[0][1],&o10=r.e[1][0],&o11=r.e[1][1];
    bool s00=signed_sum(o00,p[0],false,p[3],ps[3]);
    s00=signed_sum(o00,o00,s00,p[4],true);s00=signed_sum(o00,o00,s00,p[6],ps[6]);
    const bool s01=signed_sum(o01,p[2],ps[2],p[4],false),s10=signed_sum(o10,p[1],false,p[3],ps[3]);
    bool s11=signed_sum(o11,p[0],false,p[1],true);s11=signed_sum(o11,o11,s11,p[2],ps[2]);s11=signed_sum(o11,o11,s11,p[5],ps[5]);
    require(!s00&&!s01&&!s10&&!s11,SBN3_FATAL_MATH,"parallel Strassen result sign");
    if(e.metrics){e.metrics->products+=7;++e.metrics->batch_compositions;}return true;
}
bool compose_seven(Matrix &r,const Matrix &m,Engine &e) {
    if(r.rows!=3)return false;
    size_t rn=0,mn=0,cap=SIZE_MAX;
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){rn=std::max(rn,r.e[i][j].n);mn=std::max(mn,m.e[i][j].n);cap=std::min(cap,r.e[i][j].cap);}
    if(std::min(rn,mn)<96||rn+mn+3>cap)return false;
    FrameMark mark(e.space);auto s=allocate(e.space,rn+3),t=allocate(e.space,mn+3);
    auto u=allocate(e.space,rn+mn+5),v=allocate(e.space,rn+mn+5);
    auto &a=r.e[0][0],&b=r.e[0][1],&c=r.e[1][0],&d=r.e[1][1];
    const auto &ee=m.e[0][0],&f=m.e[0][1],&g=m.e[1][0],&h=m.e[1][1];
    e.multiply(u,b,g);                                      // b*g
    bool ds=signed_sum(d,d,false,c,true);                   // d-c
    bool bs=signed_sum(b,b,false,d,ds);                     // b-c+d
    bool ss=signed_sum(s,b,bs,a,true);                      // -a+b-c+d
    e.multiply(v,a,ee);add(a,u,v);                          // a*e+b*g
    bool ts=signed_sum(t,h,false,g,true);                   // h-g
    e.multiply(v,d,t);bool vs=ds^ts^true;                   // -(d-c)*(h-g)
    ts=signed_sum(t,t,ts,f,false);                          // f-g+h
    e.multiply(d,b,t);ds=bs^ts;
    ds=signed_sum(d,d,ds,u,false);                          // u3+u5
    ts=signed_sum(t,t,ts,ee,true);                          // -e+f-g+h
    e.multiply(u,c,t);bool us=ts;                          // u6
    bs=signed_sum(b,b,bs,c,false);                          // b+d
    require(!bs,SBN3_FATAL_MATH,"HGCD Strassen positive sum");
    bool cs=signed_sum(c,d,ds,u,us);                        // u3+u5+u6
    ds=signed_sum(d,d,ds,v,vs);                             // u3+u5-u2
    e.multiply(u,s,f);us=ss;                               // u4
    add(t,f,h);e.multiply(v,b,t);                          // u1
    bs=signed_sum(b,d,ds,u,!us);                           // u3+u5-u2-u4
    ds=signed_sum(d,v,false,d,!ds);                        // u1+u2-u3-u5
    cs=signed_sum(c,v,false,c,!cs);                        // u1-u3-u5-u6
    require(!bs&&!cs&&!ds,SBN3_FATAL_MATH,"HGCD Strassen result sign");
    return true;
}
}
