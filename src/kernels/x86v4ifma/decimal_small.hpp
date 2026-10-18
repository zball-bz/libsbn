#pragma once
#include "decimal_tiny.hpp"
#include <cstring>

namespace sbn::v3::tiny_decimal {
// These recurrences use only caller/stack storage. The radix service selects
// them by shape before constructing any large power, reciprocal or tree.
struct DivWord {uint64_t d,inv;unsigned shift;};
inline constexpr auto decimal_divisors=[] {
    std::array<DivWord,20> a{};
    for(unsigned k=1;k<=19;++k){
        const auto p=uint64_t(powers[k]);const unsigned s=unsigned(__builtin_clzll(p));
        const auto d=p<<s;a[k]={d,uint64_t(~Wide(0)/d),s};
    }return a;
}();
/* GMP normalized preinverse 2/1 identity, as in split16 in decimal_tiny.hpp.
 * Copyright 1991-2016 Free Software Foundation, Inc.
 * Retained terms: src/core/x86_64/scalar-donor-notice.txt (LGPLv3+ / GPLv2+).
 * Here r remains normalized across the loop. */
[[gnu::always_inline]] inline uint64_t quotient_word(uint64_t &r,uint64_t low,const DivWord &v) noexcept {
    Wide z=Wide(r)*v.inv+(Wide(r+1)<<64)+low;
    uint64_t q=uint64_t(z>>64),rem=low-q*v.d;
    const uint64_t mask=0-uint64_t(rem>uint64_t(z));q+=mask;rem+=mask&v.d;
    const bool fix=rem>=v.d;q+=fix;rem-=fix?v.d:0;r=rem;return q;
}
// a <- floor((head*2^(64*n)+a)/10^k), return the exact remainder.
// head < 10^k; the quotient has exactly n words including high zeros.
[[gnu::always_inline]] inline uint64_t divide_words(uint64_t *a,size_t n,uint64_t head,unsigned k=16) noexcept {
    const auto v=decimal_divisors[k];const unsigned s=v.shift;
    if(!s){uint64_t r=head;for(size_t j=n;j-->0;)a[j]=quotient_word(r,a[j],v);return r;}
    uint64_t r=(head<<s)|(a[n-1]>>(64-s));
    for(size_t j=n;j-->0;){const uint64_t low=(a[j]<<s)|(j?a[j-1]>>(64-s):0);
        a[j]=quotient_word(r,low,v);}
    return r>>s;
}
[[gnu::always_inline]] inline void write16(char *out,uint64_t x) noexcept {
    const uint64_t hi=x/d8;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),_mm_unpacklo_epi64(digits8(hi),digits8(x-hi*d8)));
}
[[gnu::always_inline]] inline void write_padded(char *out,uint64_t x,unsigned n) noexcept {
    const uint64_t hi=x/d8;
    auto bytes=_mm_unpacklo_epi64(digits8(hi),digits8(x-hi*d8));
    bytes=_mm_shuffle_epi8(bytes,_mm_sub_epi8(_mm_add_epi8(indices16(),_mm_set1_epi8(16)),_mm_set1_epi8(n)));
    _mm_mask_storeu_epi8(out,__mmask16((uint32_t(1)<<n)-1),bytes);
}
[[gnu::always_inline]] inline void write19(char *out,uint64_t x,unsigned n=19) noexcept {
    const uint64_t q8=x/d8,q16=x/d16;
    auto text=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi64(digits8(q16),digits8(q8-q16*d8))),digits8(x-q8*d8),1);
    text=_mm256_permutexvar_epi8(_mm256_sub_epi8(_mm256_add_epi8(indices32(),_mm256_set1_epi8(24)),_mm256_set1_epi8(n)),text);
    _mm256_mask_storeu_epi8(out,__mmask32((uint32_t(1)<<n)-1),text);
}
template<unsigned K=19>[[gnu::always_inline]] inline void emit_block(char *out,uint64_t x,unsigned k=K) noexcept {
    if constexpr(K==19){if(k>16){write19(out,x,k);return;}}
    if(k==16)write16(out,x);else write_padded(out,x,k);
}
template<unsigned K=19>inline char *write_words(char *out,const uint64_t *in,size_t n,uint64_t *work,uint64_t *chunks) noexcept {
    while(n>1&&!in[n-1])--n;
    if(n==1)return write64(out,*in);
    if(n==2)return write128(out,{in[0],in[1]});
    memcpy(work,in,n*8);size_t count=0;
    while(n>2){chunks[count++]=divide_words(work,n,0,K);while(n>1&&!work[n-1])--n;}
    out=n==1?write64(out,*work):write128(out,{work[0],work[1]});
    while(count){emit_block<K>(out,chunks[--count]);out+=K;}return out;
}
inline sbn3_decimal_status read_words(uint64_t *out,size_t n,const char *in,size_t count,uint64_t *work) noexcept {
    if(!count)return SBN3_DECIMAL_INVALID;
    if(n==1)return read64(out,in,count);
    if(count<=39){sbn3_uint128 v;const auto ok=read128(&v,in,count);
        if(ok==SBN3_DECIMAL_OK){out[0]=v.lo;out[1]=v.hi;memset(out+2,0,(n-2)*8);return ok;}
        if(ok==SBN3_DECIMAL_INVALID||n==2)return ok;
        // A 39-digit integer may need a third word (e.g. 2^128).
    }
    // Trim only excess zeroes. A decimal length is a conservative overflow
    // bound; still validate every character before reporting overflow.
    const size_t limit=(n*64*1234>>12)+1;
    if(trim(in,count,unsigned(limit)))return too_long(in,count);
    unsigned first=unsigned((count-1)%16+1);uint64_t bad=0;
    work[0]=parse16(in,first,bad);size_t used=1;bool overflow=false;
    for(size_t at=first;at<count;at+=16){uint64_t carry=parse16(in+at,16,bad);
        for(size_t j=0;j<used;++j){const Wide v=Wide(work[j])*d16+carry;work[j]=uint64_t(v);carry=uint64_t(v>>64);}
        if(carry){if(used<n)work[used++]=carry;else overflow=true;}}
    if(bad)return SBN3_DECIMAL_INVALID;
    if(overflow)return SBN3_DECIMAL_OVERFLOW;
    memcpy(out,work,used*8);memset(out+used,0,(n-used)*8);return SBN3_DECIMAL_OK;
}
template<unsigned K=19>inline char *write_fraction_words(char *out,const uint64_t *in,size_t n,size_t count,uint64_t *work) noexcept {
    memcpy(work,in,n*8);size_t low=0,zero_bits=0;
    while(low<n&&!work[low])++low;zero_bits=64*low;
    if(low<n)zero_bits+=unsigned(__builtin_ctzll(work[low]));
    while(count){if(low==n){memset(out,'0',count);return out+count;}
        const unsigned k=count>=K?K:unsigned(count);const uint64_t d=uint64_t(powers[k]);
        uint64_t carry=0;
        for(size_t j=low;j<n;++j){const Wide z=Wide(work[j])*d+carry;work[j]=uint64_t(z);carry=uint64_t(z>>64);}
        emit_block<K>(out,carry,k);
        out+=k;count-=k;zero_bits+=k;low=zero_bits/64<n?zero_bits/64:n;
    }return out;
}
template<unsigned N>[[gnu::always_inline]] inline char *write_fraction_fixed(char *out,const uint64_t *in,size_t count) noexcept {
    static_assert(N==1||N==2);
    uint64_t lo=in[0],hi=N==2?in[1]:0;
    while(count){const unsigned k=count>=19?19:unsigned(count);const auto d=uint64_t(powers[k]);
        const Wide a=Wide(lo)*d;lo=uint64_t(a);uint64_t word=uint64_t(a>>64);
        if constexpr(N==2){const Wide b=Wide(hi)*d+word;hi=uint64_t(b);word=uint64_t(b>>64);}
        emit_block<19>(out,word,k);
        out+=k;count-=k;
    }return out;
}
template<unsigned K=19>inline sbn3_decimal_status read_fraction_words(uint64_t *out,size_t n,const char *in,size_t count,uint64_t *work) noexcept {
    memset(work,0,n*8);uint64_t bad=0;
    if(count){unsigned k=unsigned((count-1)%K+1);size_t at=count-k;
        for(;;){const uint64_t head=k<=16?parse16(in+at,k,bad):uint64_t(parse20(in+at,k,bad));
            // Invalid bytes can make head exceed 10^k. Do not feed them to
            // a division with a strict head bound; valid-path values are unchanged.
            divide_words(work,n,bad?0:head,k);
            if(!at)break;at-=K;k=K;}
    }
    if(bad)return SBN3_DECIMAL_INVALID;
    memcpy(out,work,n*8);return SBN3_DECIMAL_OK;
}
template<unsigned N>[[gnu::always_inline]] inline sbn3_decimal_status read_fraction_fixed(uint64_t *out,const char *in,size_t count) noexcept {
    static_assert(N==1||N==2);
    uint64_t value[N]{};uint64_t bad=0;
    if(count){unsigned k=unsigned((count-1)%19+1);size_t at=count-k;
        for(;;){const uint64_t head=k<=16?parse16(in+at,k,bad):uint64_t(parse20(in+at,k,bad));
            divide_words(value,N,bad?0:head,k);
            if(!at)break;at-=19;k=19;}}
    if(bad)return SBN3_DECIMAL_INVALID;
    memcpy(out,value,N*8);return SBN3_DECIMAL_OK;
}
}
