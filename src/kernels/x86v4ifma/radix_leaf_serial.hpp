#pragma once
#include <array>
namespace sbn::v3::radix::leaf_detail {
inline constexpr auto &serial_radices=fraction_bases;
struct alignas(64) SerialMap {uint8_t index[4][64]{};uint64_t high[4]{};};
constexpr auto make_serial_maps(){std::array<SerialMap,33> out{};
    for(unsigned q=8;q<=32;++q){const unsigned groups=(q+7)/8,pad=8*groups-q;
        for(unsigned b=0;b<groups;++b)for(unsigned j=0;j<64;++j){const unsigned pos=b*64+j,word=pos/q,digit=pos%q+pad,g=digit/8;
            out[q].index[b][j]=uint8_t(64*g+8*word+digit%8);if(g>=2)out[q].high[b]|=uint64_t(1)<<j;}}
    return out;}
inline constexpr auto serial_maps=make_serial_maps();
inline void emit_serial_words(uint8_t *out,const uint64_t *words,unsigned count,const DigitPlan &p){
    const auto r=serial_radices[p.base];const unsigned q=r.digits,groups=(q+7)/8;
    const Emitter emit(p);const auto zero=_mm512_setzero_si512(),one=_mm512_set1_epi64(1),
        radix=_mm512_set1_epi64(p.b8),inverse=_mm512_set1_epi64(r.reciprocal8);
    auto value=_mm512_maskz_loadu_epi64(__mmask8((1u<<count)-1),words);__m512i digits[4]{zero,zero,zero,zero};
    for(unsigned g=groups;g-->0;){auto rem=value;
        if(g){auto quotient=_mm512_madd52hi_epu64(zero,value,inverse);
            rem=_mm512_sub_epi64(value,_mm512_mullo_epi64(quotient,radix));
            const auto fix=_mm512_cmp_epu64_mask(rem,radix,_MM_CMPINT_NLT);
            value=_mm512_mask_add_epi64(quotient,fix,quotient,one);rem=_mm512_mask_sub_epi64(rem,fix,rem,radix);}
        digits[g]=emit(rem);}
    for(unsigned g=0;g<groups&&64*g<count*q;++g){const auto index=_mm512_load_si512(serial_maps[q].index[g]);
        auto packed=_mm512_permutex2var_epi8(digits[0],index,digits[1]);
        if(groups>2)packed=_mm512_mask_mov_epi8(packed,__mmask64(serial_maps[q].high[g]),_mm512_permutex2var_epi8(digits[2],index,digits[3]));
        const unsigned bytes=std::min(64u,count*q-64*g);
        _mm512_mask_storeu_epi8(out+64*g,bytes==64?~uint64_t(0):(uint64_t(1)<<bytes)-1,packed);}
}
}
namespace sbn::v3::radix {
size_t serial_fraction_words(size_t n,uint64_t digits,unsigned base) noexcept {
    require(n&&base>=3&&base<64&&(base&(base-1)),SBN3_FATAL_ARGUMENT,"serial fraction shape");
    const unsigned q=leaf_detail::serial_radices[base].digits;
    return 8*((64*n+407)/408)+size_t((digits+q-1)/q);
}
void serial_fraction_emit(uint8_t *out,uint8_t *guard,const uint64_t *input,size_t n,uint64_t digits,unsigned guards,
                          const DigitPlan &p,uint64_t *work) noexcept {
    const auto radix=leaf_detail::serial_radices[p.base];const uint64_t total=digits+guards;
    const size_t N=8*((64*n+407)/408),rounds=size_t((total+radix.digits-1)/radix.digits);
    auto *words=work+N;
    // Top-aligned radix-2^51 state. One spare bit admits redundant digits
    // below 2*B inside IFMA. D<=B/2 keeps this range invariant.
    constexpr uint64_t mask=(uint64_t(1)<<51)-1;
    for(size_t j=0;j<N;++j){const int64_t bit=int64_t(64*n)-int64_t(51*(N-j));
        if(bit<=-51){work[j]=0;continue;}
        const size_t lo=bit<0?0:size_t(bit),skip=lo-bit,k=lo/64;const unsigned shift=lo%64;
        uint64_t v=input[k]>>shift;if(shift&&k+1<n)v|=input[k+1]<<(64-shift);
        work[j]=(v<<skip)&mask;
    }
    const auto zero=_mm512_setzero_si512(),D=_mm512_set1_epi64(radix.power),M=_mm512_set1_epi64(mask);
    size_t low=0;
    for(size_t r=0;r<rounds;++r){while(low<N&&!_mm512_test_epi64_mask(_mm512_loadu_si512(work+low),_mm512_loadu_si512(work+low)))low+=8;
        if(low==N){memset(words+r,0,(rounds-r)*8);break;}
        auto previous=zero;
        for(size_t j=low;j<N;j+=8){const auto x=_mm512_loadu_si512(work+j);
            const auto lo=_mm512_madd52lo_epu64(zero,x,D),hi=_mm512_madd52hi_epu64(zero,x,D);
            const auto high=_mm512_or_si512(_mm512_slli_epi64(hi,1),_mm512_srli_epi64(lo,51));
            _mm512_storeu_si512(work+j,_mm512_add_epi64(_mm512_and_si512(lo,M),_mm512_alignr_epi64(high,previous,7)));
            previous=high;
        }
        words[r]=uint64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(previous,3),1));
    }
    // D*V = q*B^N + V', exactly, with 0<=V'<2*B^N. Delay all
    // carry propagation until the end, including the redundant base-D words.
    uint64_t carry=0;for(size_t j=low;j<N;++j)carry=(work[j]+carry)>>51;
    for(size_t j=rounds;j-->0;){const uint64_t value=words[j]+carry;carry=value>=radix.power;
        words[j]=value-carry*radix.power;}
    require(!carry,SBN3_FATAL_MATH,"serial fraction leading carry");
    for(size_t at=0;at<rounds;at+=8){uint8_t bytes[256];const unsigned count=unsigned(std::min<size_t>(8,rounds-at));
        leaf_detail::emit_serial_words(bytes,words+at,count,p);
        const uint64_t begin=uint64_t(at)*radix.digits,end=begin+count*radix.digits;
        if(begin<digits)memcpy(out+begin,bytes,size_t(std::min(end,digits)-begin));
        const uint64_t g0=std::max(begin,digits),g1=std::min(end,total);
        if(g1>g0)memcpy(guard+g0-digits,bytes+g0-begin,size_t(g1-g0));
    }
}
}
