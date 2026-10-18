#pragma once
#include "radix/leaf.hpp"
#include "radix/bits.hpp"
#include "common/checked.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstring>
namespace sbn::v3::radix::bit_detail {
struct alignas(64) Maps {uint8_t compress[64],expand[64],shifts[64];};
consteval auto make_maps(){
    struct Bank {Maps row[7]{};} b;
    for(unsigned s=1;s<=6;++s)for(unsigned k=0;k<64;++k){
        b.row[s].compress[k]=k<8*s?uint8_t((7-k/s)*8+k%s):0;
        b.row[s].expand[k]=uint8_t((7-k/8)*s+k%8);
        b.row[s].shifts[k]=uint8_t(s*(7-k%8));
    }
    return b;
}
inline constexpr auto maps=make_maps();
struct Decode {
    const uint8_t *table;
    __m512i t[4];
    explicit Decode(const uint8_t *p):table(p){if(p)for(unsigned j=0;j<4;++j)t[j]=_mm512_loadu_si512(p+64*j);}
    __m512i operator()(__m512i v)const {
        if(!table)return v;
        return _mm512_mask_blend_epi8(_mm512_movepi8_mask(v),
            _mm512_permutex2var_epi8(t[0],v,t[1]),_mm512_permutex2var_epi8(t[2],v,t[3]));
    }
};
inline __mmask64 byte_mask(unsigned n){return n==64?~__mmask64(0):(__mmask64(1)<<n)-1;}
template<unsigned S>void emit(uint8_t *out,size_t count,const uint64_t *a,size_t an,int64_t first,const uint8_t *alphabet){
    const auto expand=_mm512_load_si512(maps.row[S].expand),shifts=_mm512_load_si512(maps.row[S].shifts);
    const auto mask=_mm512_set1_epi8((1<<S)-1),encode=_mm512_loadu_si512(alphabet);
    const auto next=_mm512_setr_epi64(1,2,3,4,5,6,7,7);
    for(size_t done=0;done<count;done+=64,first-=64*S){
        const int64_t low=first-int64_t(63*S);
        __m512i packed;
        if(low>=0&&uint64_t(low)/64<=an&&an-size_t(low/64)>=S+unsigned(low%64!=0)){
            const unsigned r=unsigned(low%64);
            packed=_mm512_maskz_loadu_epi64(__mmask8((1u<<(S+unsigned(r!=0)))-1),a+size_t(low/64));
            packed=_mm512_or_si512(_mm512_srlv_epi64(packed,_mm512_set1_epi64(r)),
                _mm512_sllv_epi64(_mm512_permutexvar_epi64(next,packed),_mm512_set1_epi64(64-r)));
        }else{
            alignas(64) uint64_t words[8]{};
            for(unsigned j=0;j<S;++j)words[j]=bits_at(a,an,low+int64_t(64*j));
            packed=_mm512_load_si512(words);
        }
        const auto chunks=_mm512_permutexvar_epi8(expand,packed);
        const auto digits=_mm512_and_si512(_mm512_multishift_epi64_epi8(shifts,chunks),mask);
        _mm512_mask_storeu_epi8(out+done,byte_mask(unsigned(std::min<size_t>(64,count-done))),
                              _mm512_permutexvar_epi8(digits,encode));
    }
}
template<unsigned S>void parse(uint64_t *out,size_t words,size_t begin,size_t end,const uint8_t *text,size_t count,
                              int64_t origin,const uint8_t *table){
    const Decode decode(table);
    const auto c1=_mm512_set1_epi16(short(0x0100|(1<<S))),c2=_mm512_set1_epi32(0x00010000|(1<<(2*S)));
    const auto b4=_mm512_set1_epi64(uint64_t(1)<<(4*S)),compress=_mm512_load_si512(maps.row[S].compress);
    const auto next=_mm512_setr_epi64(1,2,3,4,5,6,7,7);
    const int64_t bit=int64_t(64)*int64_t(begin)*S-origin;
    int64_t digit=bit>=0?bit/S:-((-bit+S-1)/S);
    const unsigned shift=unsigned(bit-digit*S);
    for(size_t block=begin;block<end;++block,digit+=64){
        const int64_t start=int64_t(count)-64-digit;
        __m512i d;
        if(start>=0&&uint64_t(start)+64<=count)d=decode(_mm512_loadu_si512(text+size_t(start)));
        else if(start>=int64_t(count)||start+64<=0)d=_mm512_setzero_si512();
        else{
            alignas(64) uint8_t raw[64]{};
            const unsigned first=unsigned(std::max<int64_t>(0,-start)),last=unsigned(std::min<int64_t>(64,int64_t(count)-start));
            std::memcpy(raw+first,text+size_t(start+first),last-first);
            d=_mm512_maskz_mov_epi8(byte_mask(last)^byte_mask(first),decode(_mm512_load_si512(raw)));
        }
        const auto pairs=_mm512_maddubs_epi16(d,c1),quads=_mm512_madd_epi16(pairs,c2);
        auto packed=_mm512_permutexvar_epi8(compress,_mm512_add_epi64(_mm512_mul_epu32(quads,b4),_mm512_srli_epi64(quads,32)));
        if(shift){
            const int64_t at=start-1;const uint64_t high=at>=0&&uint64_t(at)<count?(table?table[text[at]]:text[at]):0;
            packed=_mm512_mask_mov_epi64(packed,__mmask8(1u<<S),_mm512_set1_epi64(high));
            packed=_mm512_or_si512(_mm512_srlv_epi64(packed,_mm512_set1_epi64(shift)),
                _mm512_sllv_epi64(_mm512_permutexvar_epi64(next,packed),_mm512_set1_epi64(64-shift)));
        }
        const size_t at=block*S,n=std::min<size_t>(S,words-at);
        _mm512_mask_storeu_epi64(out+at,__mmask8((1u<<n)-1),packed);
    }
}
}
namespace sbn::v3::radix {
size_t validate_digits(const uint8_t *p,size_t n,unsigned base,const uint8_t *table) noexcept {
    const bit_detail::Decode decode(table);const auto top=_mm512_set1_epi8(char(base-1));
    for(size_t j=0;j<n;j+=64){const auto live=bit_detail::byte_mask(unsigned(std::min<size_t>(64,n-j)));
        const auto v=decode(_mm512_maskz_loadu_epi8(live,p+j));
        const uint64_t bad=live&_mm512_cmpgt_epu8_mask(v,top);
        if(bad)return j+unsigned(__builtin_ctzll(bad));
    }
    return n;
}
void format_bits(uint8_t *out,size_t digits,const uint64_t *a,size_t an,unsigned s,int64_t first,const uint8_t *alphabet) noexcept {
    switch(s){
#define BIT_CASE(S) case S:bit_detail::emit<S>(out,digits,a,an,first,alphabet);return
        BIT_CASE(1);BIT_CASE(2);BIT_CASE(3);BIT_CASE(4);BIT_CASE(5);BIT_CASE(6);
#undef BIT_CASE
    }
    fatal(SBN3_FATAL_ARGUMENT,"radix bit width");
}
void parse_bits_range(uint64_t *out,size_t words,size_t begin,size_t end,const uint8_t *digits,size_t count,
                      unsigned s,int64_t origin,const uint8_t *decode) noexcept {
    switch(s){
#define BIT_CASE(S) case S:bit_detail::parse<S>(out,words,begin,end,digits,count,origin,decode);return
        BIT_CASE(1);BIT_CASE(2);BIT_CASE(3);BIT_CASE(4);BIT_CASE(5);BIT_CASE(6);
#undef BIT_CASE
    }
    fatal(SBN3_FATAL_ARGUMENT,"radix bit width");
}
}
