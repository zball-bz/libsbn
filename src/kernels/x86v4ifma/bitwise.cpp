// Original libsbn code; LGPL-3.0-or-later.
#include "bitwise_impl.hpp"
namespace sbn::v3::bits {
size_t normalized_size(const U*a,size_t n) noexcept {
    if(n&&a[n-1])return n;
    while(n>=8){auto v=_mm512_loadu_si512(a+n-8);unsigned m=_mm512_cmpneq_epi64_mask(v,_mm512_setzero_si512());if(m)return n-8+32-unsigned(__builtin_clz(m));n-=8;}
    while(n&&!a[n-1])--n;return n;
}
size_t first_nonzero(const U*a,size_t n) noexcept {
    if(n&&a[0])return 0;
    size_t i=0;
    for(;n-i>=8;i+=8){auto v=_mm512_loadu_si512(a+i);unsigned m=_mm512_cmpneq_epi64_mask(v,_mm512_setzero_si512());if(m)return i+unsigned(__builtin_ctz(m));}
    while(i<n&&!a[i])++i;return i;
}
size_t first_not_ones(const U*a,size_t n) noexcept {
    if(n&&a[0]!=UINT64_MAX)return 0;
    size_t i=0;
    for(;n-i>=8;i+=8){auto v=_mm512_loadu_si512(a+i);unsigned m=_mm512_cmpneq_epi64_mask(v,_mm512_set1_epi64(-1));if(m)return i+unsigned(__builtin_ctz(m));}
    while(i<n&&a[i]==UINT64_MAX)++i;return i;
}
size_t population(const U*a,size_t n) noexcept {
    auto s0=_mm512_setzero_si512(),s1=s0,s2=s0,s3=s0;size_t i=0,sum=0;
    for(;n-i>=32;i+=32){s0=_mm512_add_epi64(s0,_mm512_popcnt_epi64(_mm512_loadu_si512(a+i)));
        s1=_mm512_add_epi64(s1,_mm512_popcnt_epi64(_mm512_loadu_si512(a+i+8)));
        s2=_mm512_add_epi64(s2,_mm512_popcnt_epi64(_mm512_loadu_si512(a+i+16)));
        s3=_mm512_add_epi64(s3,_mm512_popcnt_epi64(_mm512_loadu_si512(a+i+24)));}
    if(i)sum=_mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(s0,s1),_mm512_add_epi64(s2,s3)));
    for(;i<n;++i)sum+=__builtin_popcountll(a[i]);return sum;
}
size_t binary(U*r,const U*a,size_t an,bool sa,const U*b,size_t bn,bool sb,sbn3_bit_operation op) noexcept {
    return binary_width<8,4>(r,a,an,sa,b,bn,sb,op);
}
size_t adjust_copy(U*r,const U*a,size_t n,bool inc) noexcept {
    if(!n){r[0]=1;return 1;}
    const size_t pivot=inc?first_not_ones(a,n):first_nonzero(a,n);
    // Prefix words become all zero (increment) or all one (decrement).
    if(pivot)memset(r,inc?0:255,pivot*8);
    if(pivot==n){if(inc){r[n]=1;return n+1;}return 0;}
    const U value=a[pivot]+(inc?U(1):U(-1));r[pivot]=value;
    if(pivot+1<n&&r!=a)memmove(r+pivot+1,a+pivot+1,(n-pivot-1)*8);
    return normalized_size(r,n);
}
// Popcount of finite A'/B' xor over full word slots. A'=A-1 when
// ap!=SIZE_MAX, otherwise A. No output allocation and no sign extension.
size_t population_xor(const U*a,size_t an,size_t ap,const U*b,size_t bn,size_t bp,size_t first,size_t n) noexcept {
    size_t i=first,end=first+n,sum=0;auto acc=_mm512_setzero_si512();
    for(;end-i>=8;i+=8){auto av=prefix_load<8,true>(a,an,i,ap==SIZE_MAX?0:ap),bv=prefix_load<8,true>(b,bn,i,bp==SIZE_MAX?0:bp);
        // No decrement for nonnegative operands. The ordinary body below
        // uses raw loads once the prefix is known dead.
        if(ap==SIZE_MAX)av=prefix_load<8,false>(a,an,i,0);
        if(bp==SIZE_MAX)bv=prefix_load<8,false>(b,bn,i,0);
        acc=_mm512_add_epi64(acc,_mm512_popcnt_epi64(_mm512_xor_si512(av,bv)));}
    if(i!=first)sum=_mm512_reduce_add_epi64(acc);
    for(;i<end;++i){U x=i<an?a[i]:0,y=i<bn?b[i]:0;if(ap!=SIZE_MAX&&i<=ap)--x;if(bp!=SIZE_MAX&&i<=bp)--y;sum+=__builtin_popcountll(x^y);}
    return sum;
}
}

#include "bitwise_api.hpp"

#include "sbn3/detail/bitwise.hpp"
namespace sbn::detail {
template<int Op>[[gnu::noinline]]static size_t canonical_large(uint64_t*r,const uint64_t*a,size_t sa,const uint64_t*b,size_t sb)noexcept{
    constexpr size_t high=size_t(1)<<63;const size_t an=sa&~high,bn=sb&~high;const bool na=sa>>63,nb=sb>>63;
    const auto shape=sbn::v3::bits::shape(sbn3_bit_operation(Op),an,na,bn,nb);
    if(a==b&&sa==sb){if constexpr(Op==SBN3_BIT_XOR)return 0;else{if(an&&r!=a)memmove(r,a,an*8);return sa;}}
    const size_t n=sbn::v3::bits::dispatch_signs<8,4,Op>(r,a,an,na,b,bn,nb);
    return n|((n&&shape.negative)?high:0);
}
template<int Op,bool A,bool B>[[gnu::always_inline]]static inline size_t pair_quick(uint64_t*r,const uint64_t*a,size_t an,const uint64_t*b,size_t bn,size_t n)noexcept{
    using namespace sbn::v3::bits;using V=Vec<8>;
    constexpr bool negative=Op==SBN3_BIT_AND?A&&B:Op==SBN3_BIT_OR?A||B:A!=B;
    if constexpr(A||B){const U lo_a=an?a[0]:0,lo_b=bn?b[0]:0;
        if((A&&!lo_a)||(B&&!lo_b))return SIZE_MAX;
        if constexpr(negative)if(boolean<Op,A,B>(lo_a-U(A),lo_b-U(B))==UINT64_MAX)return SIZE_MAX;
    }
    const unsigned high_mask=_bzhi_u32(255,unsigned(n-8));
    auto a0=V::load(a,_bzhi_u32(255,unsigned(std::min(an,size_t(8))))),b0=V::load(b,_bzhi_u32(255,unsigned(std::min(bn,size_t(8)))));
    auto a1=an>8?V::load(a+8,_bzhi_u32(255,unsigned(std::min(an,n)-8))):V::zero();
    auto b1=bn>8?V::load(b+8,_bzhi_u32(255,unsigned(std::min(bn,n)-8))):V::zero();
    if constexpr(A)a0=V::dec(a0,1);if constexpr(B)b0=V::dec(b0,1);
    auto lo=boolean<8,Op,A,B>(a0,b0),hi=boolean<8,Op,A,B>(a1,b1);if constexpr(negative)lo=V::inc(lo,1);
    const unsigned high_bits=V::nonzero(hi)&high_mask;
    size_t size;if(high_bits)size=8+32-unsigned(__builtin_clz(high_bits));else{const auto low_bits=V::nonzero(lo);size=low_bits?32-unsigned(__builtin_clz(low_bits)):0;}
    V::store(r,lo);V::store(r+8,hi,high_mask);
    return size|(negative&&size?size_t(1)<<63:0);
}
template<int Op>[[gnu::noinline]]static size_t canonical_medium(uint64_t*r,const uint64_t*a,size_t sa,const uint64_t*b,size_t sb)noexcept{
    constexpr size_t high=size_t(1)<<63;const auto an=sa&~high,bn=sb&~high;const bool na=sa>>63,nb=sb>>63;
    const auto n=sbn::v3::bits::shape(sbn3_bit_operation(Op),an,na,bn,nb).words;
    if(n>8&&n<=16){size_t result;
        if(na){if(nb)result=pair_quick<Op,true,true>(r,a,an,b,bn,n);else result=pair_quick<Op,true,false>(r,a,an,b,bn,n);}
        else{if(nb)result=pair_quick<Op,false,true>(r,a,an,b,bn,n);else result=pair_quick<Op,false,false>(r,a,an,b,bn,n);}
        if(result!=SIZE_MAX)return result;
    }
    return canonical_large<Op>(r,a,sa,b,sb);
}
template<int Op>[[gnu::always_inline]]static inline size_t canonical_bits(uint64_t*r,const uint64_t*a,size_t sa,const uint64_t*b,size_t sb)noexcept{
    namespace bits=sbn::v3::bits;constexpr size_t high=size_t(1)<<63;
    if(!((sa|sb)&high)){
        const size_t n=Op==SBN3_BIT_AND?std::min(sa,sb):std::max(sa,sb);
        if(n<=8)return bits::single_vector<8,Op,false,false,true>(r,a,sa,b,sb,n);
        if(n>16)return canonical_large<Op>(r,a,sa,b,sb);
    }else{
        const size_t an=sa&~high,bn=sb&~high;const bool na=sa>>63,nb=sb>>63;
        const auto n=bits::shape(sbn3_bit_operation(Op),an,na,bn,nb).words;
        if(n>16)return canonical_large<Op>(r,a,sa,b,sb);
        if(n<=8){if(na){if(nb)return bits::single_vector<8,Op,true,true,true>(r,a,an,b,bn,n);
                      return bits::single_vector<8,Op,true,false,true>(r,a,an,b,bn,n);}
            return bits::single_vector<8,Op,false,true,true>(r,a,an,b,bn,n);}
    }
    return canonical_medium<Op>(r,a,sa,b,sb);
}
size_t bit_and(uint64_t*r,const uint64_t*a,size_t an,const uint64_t*b,size_t bn)noexcept{return canonical_bits<SBN3_BIT_AND>(r,a,an,b,bn);}
size_t bit_or(uint64_t*r,const uint64_t*a,size_t an,const uint64_t*b,size_t bn)noexcept{return canonical_bits<SBN3_BIT_OR>(r,a,an,b,bn);}
size_t bit_xor(uint64_t*r,const uint64_t*a,size_t an,const uint64_t*b,size_t bn)noexcept{return canonical_bits<SBN3_BIT_XOR>(r,a,an,b,bn);}
size_t word_population(const uint64_t*a,size_t n)noexcept{
    if(n<=2)return (n?__builtin_popcountll(a[0]):0)+(n>1?__builtin_popcountll(a[1]):0);
    return sbn::v3::bits::population(a,n);
}
}
