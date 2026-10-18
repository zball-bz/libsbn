// Original libsbn code; LGPL-3.0-or-later.
#include "gcd_small_core.hpp"
#include "jacobi_word.hpp"
#include "gcd/jacobi_state.hpp"
#include "sbn3/jacobi.h"
namespace sbn::v3::gcd_small {
using State=sbn::v3::gcd::JacobiState;
struct Observe {State *s;void operator()(unsigned d,U q)const {(*s)(d,q);}};
// Canonical two-limb reduction and a sign-corrected odd word reduction.
static __uint128_t residue(const U*a,size_t n,const U*d,size_t dn){
    if(!n)return 0;
    if(dn==1)return remainder_word(a,n,d[0]);
    return remainder_two(a,n,d);
}
static __uint128_t wide(const U*a,size_t n){return n?__uint128_t(a[0])|(n>1?__uint128_t(a[1])<<64:0):0;}
static int reduced_symbol(const U*a,size_t n,const U*d,size_t dn){
    if(dn==1){
        if(d[0]==1)return 1;
        // c == -a * (2^64)^(-j) (mod d). The radix factor is a
        // square modulo odd d, so only (-1/d) changes the symbol.
        const U c=odd_gcd_residue(a,n,d[0]);
        return (d[0]&2?-1:1)*jacobi_binary(c,d[0]);
    }
    return jacobi128(residue(a,n,d,dn),wide(d,dn));
}
static int short_symbol(const U*a,size_t an,const U*b,size_t bn){
    if(bn==1&&b[0]==1)return 1;
    if(!an)return 0;
    if(bn<=2)return an<=2?jacobi128(wide(a,an),wide(b,bn)):reduced_symbol(a,an,b,bn);
    require(an<=2,SBN3_FATAL_ARGUMENT,"Jacobi short shape");
    auto av=wide(a,an);unsigned t=zeros128(av),e=(t*((b[0]>>1)^(b[0]>>2)))&1;av>>=t;
    if(av==1)return 1-2*int(e);
    e^=((U(av)&b[0])>>1)&1;
    U d[2]{U(av),U(av>>64)};
    return (1-2*int(e))*reduced_symbol(b,bn,d,1+bool(d[1]));
}
static int short_state(const U*a,size_t an,const U*b,size_t bn,State s){
    return s.sign()*(s.numerator_is_a()?short_symbol(a,an,b,bn):short_symbol(b,bn,a,an));
}
static int symbol_compute(const U*aa,size_t an,const U*bb,size_t bn,void*work,State state){
    an=norm(aa,an);bn=norm(bb,bn);
    if(std::min(an,bn)<=2)return short_state(aa,an,bb,bn,state);
    const size_t cap=sbn3_gcd_small_scratch_bytes(an,bn)/16;
    U*a=static_cast<U*>(work),*b=a+cap;
    // Owned terminal layout may already be in these two buffers.
    if(a!=aa)memmove(a,aa,an*8);if(b!=bb)memmove(b,bb,bn*8);
    bool packed=false;
    for(;;){
        size_t n=std::max(an,bn);
        memset(a+an,0,(n-an+1)*8);memset(b+bn,0,(n-bn+1)*8);
        if(!packed&&std::min(an,bn)<=2)return short_state(a,an,b,bn,state);
        if(packed){
            const size_t bits=52*(n-1)+64-unsigned(__builtin_clzll(a[n-1]|b[n-1]));
            if(bits<=128){U x[2]{read52(a,n,0),read52(a,n,64)},y[2]{read52(b,n,0),read52(b,n,64)};
                return short_state(x,norm(x,2),y,norm(y,2),state);}
        }
        Matrix m;bool got;
        const bool convert=!packed&&n>=persistent_from;
        if(packed){
            const size_t bits=52*(n-1)+64-unsigned(__builtin_clzll(a[n-1]|b[n-1]));
            got=head_matrix<44,true>(read52(a,n,bits-64),read52(a,n,bits-128),read52(b,n,bits-64),read52(b,n,bits-128),&m,Observe{&state});
        }else{
            const unsigned sh=__builtin_clzll(a[n-1]|b[n-1]);
            auto top=[&](const U*x,size_t j){return sh?(x[j]<<sh)|(x[j-1]>>(64-sh)):x[j];};
            got=convert?head_matrix<44,true>(top(a,n-1),top(a,n-2),top(b,n-1),top(b,n-2),&m,Observe{&state}):
                        head_matrix<33,true>(top(a,n-1),top(a,n-2),top(b,n-1),top(b,n-2),&m,Observe{&state});
        }
        if(got){
            if(convert){an=expand52(a,an);bn=expand52(b,bn);n=std::max(an,bn);
                memset(a+an,0,(n-an)*8);memset(b+bn,0,(n-bn)*8);packed=true;}
            if(packed)n=matrix52(a,b,n,m);
            else n=n>=vector_from?ifma_matrix(a,b,a,b,n,m):scalar_matrix(a,b,a,b,n,m);
            an=norm(a,n);bn=norm(b,n);continue;
        }
        if(packed){an=compact52(a,an);bn=compact52(b,bn);packed=false;n=std::max(an,bn);
            memset(a+an,0,(n-an+1)*8);memset(b+bn,0,(n-bn+1)*8);
            if(std::min(an,bn)<=2)return short_state(a,an,b,bn,state);}
        const int cmp=sbn::v3::divrem_words::compare(a,b,n);if(!cmp)return 0;
        U *large=cmp>0?a:b,*small=cmp>0?b:a;size_t &ln=cmp>0?an:bn,&sn=cmp>0?bn:an;
        sbn3i_sub_n(large,large,small,long(n));ln=norm(large,n);state(cmp>0?1:0,1);
        if(std::min(an,bn)<=2)return short_state(a,an,b,bn,state);
        if(ln>=sn){U q;ln=remainder<true>(large,ln,small,sn,&q);state(cmp>0?1:0,q);}
    }
}
}
namespace sbn::v3::gcd {
int symbol_finish(const uint64_t*a,size_t an,const uint64_t*b,size_t bn,void*w,JacobiState s){return gcd_small::symbol_compute(a,an,b,bn,w,s);}
int symbol_small(const uint64_t*a,size_t an,const uint64_t*b,size_t bn,void*w){return symbol_finish(a,an,b,bn,w,JacobiState::initial(an?a[0]:0,b[0]));}
}
extern "C" int sbn3_jacobi_u64(uint64_t a,uint64_t b){
    if constexpr(SBN3_CHECK_SMALL)sbn::v3::require(b&1,SBN3_FATAL_ARGUMENT,"Jacobi odd denominator");
    return sbn::v3::gcd_small::jacobi_binary(a,b);
}
