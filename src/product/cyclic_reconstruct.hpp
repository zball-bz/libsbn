#pragma once
#include "value/limbs.hpp"
namespace sbn::v3 {
// Independently exact low words. Each multiply-add fits in 128 bits.
template<size_t Words>
inline void product_low_words(uint64_t (&out)[Words],const uint64_t *a,const uint64_t *b) noexcept {
    for(auto &word:out)word=0;
    for(size_t j=0;j<Words;++j){uint64_t carry=0;
        for(size_t i=0;i+j<Words;++i){
            const unsigned __int128 v=(unsigned __int128)a[i]*b[j]+out[i+j]+carry;
            out[i+j]=uint64_t(v);carry=uint64_t(v>>64);
        }
    }
}
// E=z+t*(B^n-1). Knowing E mod B^Words identifies t mod B^Words.
// Require |t|<B^Words/2 and |E|<B^(n+Words)/2; n>=Words.
// z has n words (including redundant zero B^n-1); n+Words are writable.
// Dynamic witnesses consume caller-provided low words; the bounded fixed
// overload retains its original input-preserving contract.
inline bool reconstruct_cyclic(uint64_t *z,size_t n,uint64_t *wraps,size_t words) noexcept {
    require(words&&n>=words,SBN3_FATAL_ARGUMENT,"cyclic reconstruction span");
    uint64_t borrow=0;
    for(size_t j=0;j<words;++j){const unsigned __int128 v=(unsigned __int128)wraps[j]+borrow;
        wraps[j]=z[j]-uint64_t(v);borrow=(unsigned __int128)z[j]<v;}
    for(size_t j=0;j<words;++j)z[n+j]=wraps[j];
    if(wraps[words-1]>>63){
        for(size_t j=0;j<words;++j)wraps[j]=~wraps[j];
        const uint64_t one=1;limbs::add_to(wraps,words,&one,1);
        limbs::add_to(z,n+words,wraps,words);
    }else limbs::sub_from(z,n+words,wraps,words);
    const bool negative=z[n+words-1]>>63;
    if(negative){for(size_t j=0;j<n+words;++j)z[j]=~z[j];const uint64_t one=1;limbs::add_to(z,n+words,&one,1);}
    return negative;
}
template<size_t Words>
inline bool reconstruct_cyclic(uint64_t *z,size_t n,const uint64_t (&low)[Words]) noexcept {
    static_assert(Words>=1&&Words<=8);
    uint64_t wraps[Words];memcpy(wraps,low,sizeof wraps);
    return reconstruct_cyclic(z,n,wraps,Words);
}
// The independent low witness may occupy the otherwise unwritten high tail
// z[n,n+words). Resolve low carries before adjusting that tail, so no second
// witness buffer is needed. The bounds are identical to reconstruct_cyclic.
inline bool reconstruct_cyclic_tail(uint64_t *z,size_t n,size_t words) noexcept {
    require(words&&n>=words,SBN3_FATAL_ARGUMENT,"cyclic tail reconstruction span");
    auto *t=z+n;uint64_t borrow=0;const uint64_t one=1;
    for(size_t j=0;j<words;++j){const unsigned __int128 amount=(unsigned __int128)t[j]+borrow;
        t[j]=z[j]-uint64_t(amount);borrow=(unsigned __int128)z[j]<amount;}
    auto negate=[&]{for(size_t j=0;j<words;++j)t[j]=~t[j];limbs::add_to(t,words,&one,1);};
    if(t[words-1]>>63){
        negate();const auto carry=limbs::add_to(z,n,t,words);negate();
        if(carry)limbs::add_to(t,words,&one,1);
    }else if(limbs::sub_from(z,n,t,words))limbs::sub_from(t,words,&one,1);
    const bool negative=t[words-1]>>63;
    if(negative){for(size_t j=0;j<n+words;++j)z[j]=~z[j];limbs::add_to(z,n+words,&one,1);}
    return negative;
}
}
