// Radix leaf kernels (AVX-512 IFMA/VBMI). Emission is the radix-split cascade
// of DigitViewer with the 64-bit level on madd52hi: word / b^4, halves / b^2,
// pairs / b, then one vpermb through the output table: 8 words -> 64 bytes in
// one store. Parsing is maddubs [b,1] -> madd [b^2,1] -> mul_epu32 [b^4] with a
// branch-free validity mask. Magics satisfy the exact Granlund-Montgomery
// criterion and are re-verified by the leaf gate against the scalar reference.
#include "radix/leaf.hpp"
#include "radix/word_base.hpp"
#include "radix/digit_bounds.hpp"
#include "common/checked.hpp"
#include "radix_leaf_group.hpp"
#include "radix_digit_constants.hpp"
#include "radix_leaf_tree.hpp"
#include "radix_leaf_serial.hpp"
#include "radix_leaf_parse.hpp"
#include "radix_limb.hpp"
#include "radix_bits.hpp"
#include "radix_decimal.hpp"
#include <immintrin.h>
#include <cstring>
#include "radix/decimal_small.hpp"
namespace sbn::v3::radix {
namespace {
consteval auto make_identity_plans(){
    std::array<DigitPlan,65> out{};
    for(unsigned b=2;b<=64;++b){auto &p=out[b];p.base=b;
        const auto &c=leaf_detail::digit_constants[b];
        p.b2=c.b2;p.b4=c.b4;p.b8=c.b8;p.m4=c.m4;p.m2=c.m2;
        p.s4=c.s4;p.s2=c.s2;p.m1=c.m1;p.s1=c.s1;
        for(unsigned j=0;j<64;++j)p.encode[j]=uint8_t(j);
        for(unsigned j=0;j<256;++j)p.decode[j]=j<b?uint8_t(j):uint8_t(0xff);
    }
    return out;
}
constexpr auto identity_plans=make_identity_plans();
constexpr auto ascii_plan=[] {
    auto p=identity_plans[10];p.identity=false;p.decimal_ascii=true;
    for(auto &v:p.decode)v=0xff;
    for(unsigned j=0;j<10;++j){p.encode[j]=uint8_t('0'+j);p.decode['0'+j]=uint8_t(j);}
    return p;
}();
}
const DigitPlan &identity_digit_plan(unsigned base) noexcept {return identity_plans[base];}
const DigitPlan &decimal_ascii_plan() noexcept {return ascii_plan;}
size_t first_nonzero_digit(const uint8_t *digits,size_t count,uint8_t zero) noexcept {
    const auto z=_mm512_set1_epi8(char(zero));
    for(size_t at=0;at<count;at+=64){
        const uint64_t mask=_mm512_cmpneq_epi8_mask(_mm512_loadu_si512(digits+at),z);
        if(mask)return at+unsigned(__builtin_ctzll(mask));
    }
    return count;
}
size_t digit_constant_bytes() noexcept {return sizeof(leaf_detail::digit_constants)+sizeof(word_bases)+leaf_detail::block_constant_bytes+sizeof(leaf_detail::serial_radices)+sizeof(leaf_detail::serial_maps)+sizeof(bit_detail::maps)+sizeof(leaf_detail::parse_radices)+sizeof(inverse_logs)+sizeof(identity_plans)+sizeof(ascii_plan)+sizeof(leaf_detail::limb_radices)+4*sizeof(leaf_detail::IntegerPackMap)+decimal_constant_bound_bytes;}
bool digit_plan_init(DigitPlan &p, unsigned base, const uint8_t *table) noexcept {
    if(base==10&&table&&decimal_alphabet(table)){p=ascii_plan;return true;}
    p = {};
    if (base < 3 || base > 63 || !(base & (base - 1)))
        return false;
    p.base = base;
    const auto &c=leaf_detail::digit_constants[base];
    p.b2=c.b2;p.b4=c.b4;p.b8=c.b8;
    p.m4=c.m4;p.m2=c.m2;p.s4=c.s4;p.s2=c.s2;p.m1=c.m1;p.s1=c.s1;
    std::memset(p.decode, 0xff, sizeof p.decode);
    if(!table){
        const auto bytes=_mm512_setr_epi64(0x0706050403020100LL,0x0f0e0d0c0b0a0908LL,
            0x1716151413121110LL,0x1f1e1d1c1b1a1918LL,0x2726252423222120LL,0x2f2e2d2c2b2a2928LL,
            0x3736353433323130LL,0x3f3e3d3c3b3a3938LL);
        _mm512_storeu_si512(p.encode,bytes);
        _mm512_storeu_si512(p.decode,_mm512_mask_blend_epi8((__mmask64(1)<<base)-1,
                                                            _mm512_set1_epi8(-1),bytes));
        return true;
    }
    for (unsigned v = 0; v < 64; ++v) {
        p.encode[v] = table && v < base ? table[v] : uint8_t(v);
        p.identity = p.identity && p.encode[v] == v;
    }
    for (unsigned v = 0; v < base; ++v) {
        if (p.decode[p.encode[v]] != 0xff)
            return false; // repeated alphabet byte
        p.decode[p.encode[v]] = uint8_t(v);
    }
    return true;
}
namespace {
// Keep the digit cascade in registers when its producer already has eight
// residues. The word-chunk codec shares these constants across its groups.
using WordEmitter=leaf_detail::Emitter;
}
void emit_words(uint8_t *out,const uint64_t *words,size_t count,const DigitPlan &p) noexcept {
    require(!(count&7),SBN3_FATAL_ARGUMENT,"radix emit word count");
    const WordEmitter emit(p);
    for(size_t i=0;i<count;i+=8)_mm512_storeu_si512(out+i*8,emit(_mm512_loadu_si512(words+i)));
}
void emit_word_chunks(uint8_t *out,const uint64_t *words,unsigned count,const DigitPlan &p) noexcept {
    require(count<=8,SBN3_FATAL_ARGUMENT,"radix chunk count");
    const auto wb=word_bases[p.base];
    const unsigned groups=(wb.digits+7)/8,prefix=groups*8-wb.digits;
    if(count*groups<=8){
        // A short final batch otherwise decodes one mostly empty vector per
        // group. Pack its scalar residues into one full digit-cascade call.
        alignas(64) uint64_t chunks[8]{};
        alignas(64) uint8_t bytes[64];
        for(unsigned lane=0;lane<count;++lane){uint64_t value=words[lane];
            for(unsigned g=groups;g-- >0;)
                chunks[lane*groups+g]=g?split_radix_word(value,p.b8,wb.reciprocal8):value;
        }
        emit_words(bytes,chunks,8,p);
        for(unsigned lane=0;lane<count;++lane)
            memcpy(out+size_t(lane)*wb.digits,bytes+8*lane*groups+prefix,wb.digits);
        return;
    }
    const __m512i zero=_mm512_setzero_si512(),M=_mm512_set1_epi64((long long)wb.reciprocal8),
                  B=_mm512_set1_epi64((long long)p.b8),one=_mm512_set1_epi64(1);
    __m512i q=_mm512_maskz_loadu_epi64(__mmask8((1u<<count)-1),words);
    const WordEmitter emit(p);
    alignas(64) uint8_t digits[8][64];
    for(unsigned j=groups;j-->0;){
        __m512i r=q;
        if(j){
            // b>=3 => floor(2^64/b^8)<2^52. Only the first numerator
            // can exceed 52 bits; every following quotient also fits.
            __m512i lo=_mm512_madd52hi_epu64(zero,q,M),hi=zero;
            if(j+1==groups){
                const __m512i qhi=_mm512_srli_epi64(q,52);
                lo=_mm512_madd52lo_epu64(lo,qhi,M);hi=_mm512_madd52hi_epu64(zero,qhi,M);
            }
            __m512i quotient=_mm512_add_epi64(_mm512_srli_epi64(lo,12),_mm512_slli_epi64(hi,40));
            r=_mm512_sub_epi64(q,_mm512_mullo_epi64(quotient,B));
            const __mmask8 correction=_mm512_cmp_epu64_mask(r,B,_MM_CMPINT_NLT);
            q=_mm512_mask_add_epi64(quotient,correction,quotient,one);
            r=_mm512_mask_sub_epi64(r,correction,r,B);
        }
        _mm512_store_si512(digits[j],emit(r));
    }
    for(unsigned lane=0;lane<count;++lane){
        auto *dst=out+size_t(lane)*wb.digits;
        uint64_t head;memcpy(&head,digits[0]+8*lane,8);head>>=8*prefix;
        // K>=10: this eight-byte store stays inside the output chunk.
        // The next complete group overwrites the extra bytes of the prefix.
        memcpy(dst,&head,8);dst+=8-prefix;
        for(unsigned j=1;j<groups;++j){memcpy(dst,digits[j]+8*lane,8);dst+=8;}
    }
}
unsigned format_limb(uint8_t *out,uint64_t value,const DigitPlan &p) noexcept {
    if(p.base==10&&(p.identity||p.decimal_ascii))return decimal_detail::format_limb(out,value,p.decimal_ascii);
    const auto &r=leaf_detail::limb_radices[p.base];
    const auto zero=_mm512_setzero_si512(),x=_mm512_set1_epi64((long long)value);
    const auto d=_mm512_load_si512(r.divisor),m=_mm512_load_si512(r.reciprocal);
    auto lo=_mm512_madd52hi_epu64(zero,x,m);
    const auto upper=_mm512_srli_epi64(x,52);
    lo=_mm512_madd52lo_epu64(lo,upper,m);
    const auto hi=_mm512_madd52hi_epu64(zero,upper,m);
    auto q=_mm512_add_epi64(_mm512_srli_epi64(lo,12),_mm512_slli_epi64(hi,40));
    const auto rem=_mm512_sub_epi64(x,_mm512_mullo_epi64(q,d));
    const auto correct=__mmask8(r.active)&_mm512_cmp_epu64_mask(rem,d,_MM_CMPINT_NLT);
    q=_mm512_mask_add_epi64(q,correct,q,_mm512_set1_epi64(1));q=_mm512_mask_mov_epi64(q,1,x);
    const auto next=_mm512_maskz_permutexvar_epi64(0x7f,_mm512_setr_epi64(1,2,3,4,5,6,7,0),q);
    const auto residues=_mm512_sub_epi64(q,_mm512_mullo_epi64(next,_mm512_set1_epi64((long long)p.b8)));
    WordEmitter emit(p);const auto bytes=emit(_mm512_permutexvar_epi64(_mm512_setr_epi64(7,6,5,4,3,2,1,0),residues));
    _mm512_storeu_si512(out,bytes);
    const uint64_t nonzero=_mm512_cmpneq_epi8_mask(bytes,_mm512_set1_epi8(char(p.encode[0])));
    return nonzero?unsigned(__builtin_ctzll(nonzero)):64;
}
bool parse_words(uint64_t *words, const uint8_t *digits, size_t count, const DigitPlan &p) noexcept {
    require(!(count & 7), SBN3_FATAL_ARGUMENT, "radix parse word count");
    const bool identity = p.identity;
    const bit_detail::Decode decode(identity?nullptr:p.decode);
    const __m512i top = _mm512_set1_epi8(char(p.base - 1));
    const __m512i C1 = _mm512_set1_epi16(short(0x0100 | p.base));       // [b, 1] per byte pair
    const __m512i C2 = _mm512_set1_epi32(int(0x00010000u | uint32_t(p.b2))); // [b^2, 1] per u16 pair
    const __m512i B4 = _mm512_set1_epi64((long long)p.b4);
    uint64_t bad = 0;
    for (size_t i = 0; i < count; i += 8) {
        __m512i d = _mm512_loadu_si512(digits + i * 8);
        d=decode(d);
        bad |= uint64_t(_mm512_cmpgt_epu8_mask(d, top));
        const __m512i v16 = _mm512_maddubs_epi16(d, C1); // two digits per u16
        const __m512i v32 = _mm512_madd_epi16(v16, C2);  // four digits per u32
        const __m512i high = _mm512_srli_epi64(v32, 32); // digits 3..0 of the word (later bytes)
        const __m512i low = _mm512_mul_epu32(v32, B4);   // digits 7..4 (earlier bytes) * b^4
        _mm512_storeu_si512(words + i, _mm512_add_epi64(low, high));
    }
    return !bad;
}
namespace {
// Constant bit geometry for eight u64 outputs. No scalar division by 52,
// temporary digit array or data-dependent carry is needed for the packing.
template<unsigned Block,unsigned Digits> __m512i pack_integer(__m512i lo,__m512i hi) noexcept {
    static constexpr auto map=[] {leaf_detail::IntegerPackMap m;
        for(unsigned lane=0;lane<8;++lane){const unsigned bit=64*(8*Block+lane),index=bit/52,shift=bit%52;
            m.i[lane]=index;m.j[lane]=index+1;m.k[lane]=index+2;
            m.right[lane]=shift;m.left[lane]=52-shift;m.extra[lane]=104-shift;
            if(index<Digits)m.m0|=1u<<lane;
            if(index+1<Digits)m.m1|=1u<<lane;
            if(index+2<Digits&&shift>40)m.m2|=1u<<lane;
        }return m;}();
    const auto a=_mm512_maskz_permutex2var_epi64(map.m0,lo,_mm512_loadu_si512(map.i.data()),hi);
    const auto b=_mm512_maskz_permutex2var_epi64(map.m1,lo,_mm512_loadu_si512(map.j.data()),hi);
    const auto c=_mm512_maskz_permutex2var_epi64(map.m2,lo,_mm512_loadu_si512(map.k.data()),hi);
    return _mm512_ternarylogic_epi64(_mm512_srlv_epi64(a,_mm512_loadu_si512(map.right.data())),
        _mm512_sllv_epi64(b,_mm512_loadu_si512(map.left.data())),
        _mm512_sllv_epi64(c,_mm512_loadu_si512(map.extra.data())),0xfe);
}
template<unsigned N> void assemble_integer(uint64_t *out,size_t stride,const uint64_t *words,unsigned groups,
                                           const leaf_detail::ParseRadix &table) noexcept {
    const auto zero=_mm512_setzero_si512(),mask=_mm512_set1_epi64((uint64_t(1)<<52)-1);
    const auto previous=_mm512_setr_epi64(0,0,1,2,3,4,5,6);
    __m512i low[N][4],high[N][4],lo[N],hi[N];
    for(unsigned k=0;k<N;++k)for(unsigned part=0;part<4;++part)low[k][part]=high[k][part]=zero;
    low[0][0]=_mm512_maskz_set1_epi64(1,(long long)words[15]);
    auto term=[&]<unsigned Part>(unsigned j){
        const auto value=_mm512_set1_epi64((long long)words[15-j]);
        for(unsigned k=0;k<N;++k){const auto coefficient=_mm512_load_si512(table.power[j]+8*k);
            low[k][Part]=_mm512_madd52lo_epu64(low[k][Part],value,coefficient);
            high[k][Part]=_mm512_madd52hi_epu64(high[k][Part],value,coefficient);}
    };
    // Independent accumulators hide IFMA latency; the decomposition is by
    // instruction scheduling, not a length-dependent arithmetic recipe.
    unsigned j=1;
    for(;j+3<groups;j+=4){term.template operator()<0>(j);term.template operator()<1>(j+1);
        term.template operator()<2>(j+2);term.template operator()<3>(j+3);}
    if(j<groups)term.template operator()<0>(j++);
    if(j<groups)term.template operator()<1>(j++);
    if(j<groups)term.template operator()<2>(j);
    for(unsigned k=0;k<N;++k){
        lo[k]=_mm512_add_epi64(_mm512_add_epi64(low[k][0],low[k][1]),_mm512_add_epi64(low[k][2],low[k][3]));
        hi[k]=_mm512_add_epi64(_mm512_add_epi64(high[k][0],high[k][1]),_mm512_add_epi64(high[k][2],high[k][3]));}
    uint64_t overflow=0;
    if constexpr(N==1)if(stride>6)overflow=uint64_t(_mm256_extract_epi64(_mm512_extracti64x4_epi64(hi[0],1),3));
    __m512i value[N],carry[N];
    for(unsigned k=0;k<N;++k){auto shifted=_mm512_maskz_permutexvar_epi64(0xfe,previous,hi[k]);
        if(k)shifted=_mm512_mask_permutexvar_epi64(shifted,1,_mm512_set1_epi64(7),hi[k-1]);
        value[k]=_mm512_add_epi64(lo[k],shifted);carry[k]=_mm512_srli_epi64(value[k],52);
        value[k]=_mm512_and_si512(value[k],mask);}
    while(_mm512_test_epi64_mask(carry[0],carry[0])||(N==2&&_mm512_test_epi64_mask(carry[N-1],carry[N-1]))){
        if constexpr(N==1)if(stride>6)overflow+=uint64_t(_mm256_extract_epi64(_mm512_extracti64x4_epi64(carry[0],1),3));
        for(unsigned k=N;k-->0;){auto shifted=_mm512_maskz_permutexvar_epi64(0xfe,previous,carry[k]);
            if(k)shifted=_mm512_mask_permutexvar_epi64(shifted,1,_mm512_set1_epi64(7),carry[k-1]);
            value[k]=_mm512_add_epi64(value[k],shifted);carry[k]=_mm512_srli_epi64(value[k],52);
            value[k]=_mm512_and_si512(value[k],mask);}
    }
    const auto upper=N==2?value[N-1]:_mm512_maskz_set1_epi64(1,(long long)overflow);
    constexpr unsigned digits=N==2?16:9;
    const unsigned first=unsigned(std::min<size_t>(8,stride));
    _mm512_mask_storeu_epi64(out,__mmask8((1u<<first)-1),pack_integer<0,digits>(value[0],upper));
    if(stride>8)_mm512_mask_storeu_epi64(out+8,__mmask8((1u<<(stride-8))-1),pack_integer<1,digits>(value[0],upper));
}
}
namespace {
template<bool Decimal> [[gnu::noinline]] bool parse_integer_blocks_impl(uint64_t *out,size_t stride,const uint8_t *digits,size_t count,
                          unsigned block_digits,const DigitPlan &p) noexcept {
    const bit_detail::Decode decode(Decimal||p.identity?nullptr:p.decode);
    const auto offset=_mm512_set1_epi8(p.decimal_ascii?'0':0);
    const auto top=_mm512_set1_epi8(Decimal?9:char(p.base-1));
    const auto C1=_mm512_set1_epi16(short(0x0100|(Decimal?10:p.base)));
    const auto C2=_mm512_set1_epi32(int(0x00010000u|(Decimal?100:uint32_t(p.b2))));
    const auto B4=_mm512_set1_epi64(Decimal?10000:(long long)p.b4);
    const auto indices=_mm512_setr_epi64(0x0706050403020100LL,0x0f0e0d0c0b0a0908LL,
        0x1716151413121110LL,0x1f1e1d1c1b1a1918LL,0x2726252423222120LL,0x2f2e2d2c2b2a2928LL,
        0x3736353433323130LL,0x3f3e3d3c3b3a3938LL);
    uint64_t bad=0;size_t length=((count-1)&(block_digits-1))+1;
    while(count){
        alignas(64) uint64_t words[16];
        size_t first=(length-1)%64+1,at=0,index=length>64?0:8;
        while(at<length){
            const auto active=__mmask64(~uint64_t(0)>>(64-first));
            auto d=decode(_mm512_maskz_loadu_epi8(active,digits+at));
            if constexpr(Decimal)d=_mm512_sub_epi8(d,offset);
            bad|=uint64_t(_mm512_cmpgt_epu8_mask(d,top)&active);
            if(first<64){const unsigned missing=64-unsigned(first);
                d=_mm512_maskz_permutexvar_epi8(__mmask64(~uint64_t(0)<<missing),
                    _mm512_sub_epi8(indices,_mm512_set1_epi8(char(missing))),d);}
            const auto pairs=_mm512_maddubs_epi16(d,C1),quads=_mm512_madd_epi16(pairs,C2);
            _mm512_store_si512(words+index,_mm512_add_epi64(_mm512_mul_epu32(quads,B4),_mm512_srli_epi64(quads,32)));
            at+=first;index+=8;first=64;
        }
        const unsigned groups=unsigned((length+7)/8);
        if(length<=word_bases[p.base].digits){
            uint64_t value=words[16-groups];
            for(unsigned j=17-groups;j<16;++j)value=value*p.b8+words[j];
            out[0]=value;for(size_t j=1;j<stride;++j)out[j]=0;
        }else if(groups<=leaf_detail::parse_radices[p.base].one_vector_words)assemble_integer<1>(out,stride,words,groups,leaf_detail::parse_radices[p.base]);
        else assemble_integer<2>(out,stride,words,groups,leaf_detail::parse_radices[p.base]);
        digits+=length;count-=length;out+=stride;length=block_digits;
    }
    return !bad;
}
}
bool parse_integer_blocks(uint64_t *out,size_t stride,const uint8_t *digits,size_t count,
                          unsigned block_digits,const DigitPlan &p) noexcept {
    require(count&&count<=512&&stride<=16&&(block_digits==64||block_digits==128),SBN3_FATAL_ARGUMENT,"integer block shape");
    if(p.base==10&&(p.identity||p.decimal_ascii)){
        if(count<=38)return decimal_detail::parse_small(out,stride,digits,unsigned(count),p.decimal_ascii);
        return parse_integer_blocks_impl<true>(out,stride,digits,count,block_digits,p);
    }
    return parse_integer_blocks_impl<false>(out,stride,digits,count,block_digits,p);
}
bool parse_integer_fragments(uint64_t *out,size_t stride,const uint8_t *digits,size_t count,const DigitPlan &p) noexcept {
    return parse_integer_blocks(out,stride,digits,count,64,p);
}
void extract_words(uint64_t *words, const uint64_t *const fraction[8], unsigned limbs, unsigned u52_digits,
                   unsigned rounds, const DigitPlan &p) noexcept {
    require(limbs && limbs <= max_fragment_limbs && u52_digits && u52_digits <= max_fragment_u52 &&
                (rounds == fragment_words || rounds == fragment_words + 1),
            SBN3_FATAL_ARGUMENT, "radix fragment extraction shape");
    // u52 digit j (0 = lowest) of lane u holds bits [top - 52 (u52_digits - j), ...) of the fraction,
    // top = 64 * limbs; bits below the fraction's lowest limb are zero.
    require(uint64_t(52) * u52_digits < uint64_t(64) * limbs + 52, SBN3_FATAL_ARGUMENT, "radix fragment u52 digits");
    alignas(64) uint64_t lanes[max_fragment_u52][8];
    const int64_t top = int64_t(64) * limbs;
    for (unsigned j = 0; j < u52_digits; ++j) {
        const int64_t bit = top - int64_t(52) * (u52_digits - j);
        for (unsigned u = 0; u < 8; ++u) {
            uint64_t v = 0;
            if (const uint64_t *y = fraction[u]) {
                // value = bits [bit, bit + 52) of y, bits below 0 are zero
                const int64_t lo = bit < 0 ? 0 : bit;
                const unsigned skip = unsigned(lo - bit); // zero bits at the bottom of this digit
                const size_t q = size_t(lo >> 6);
                const unsigned r = unsigned(lo & 63);
                uint64_t w = q < limbs ? y[q] >> r : 0;
                if (r && q + 1 < limbs)
                    w |= y[q + 1] << (64 - r);
                v = (w << skip) & ((uint64_t(1) << 52) - 1);
                if (skip >= 52)
                    v = 0;
            }
            lanes[j][u] = v;
        }
    }
    const __m512i zero = _mm512_setzero_si512();
    const __m512i B8 = _mm512_set1_epi64((long long)p.b8);
    const __m512i mask = _mm512_set1_epi64((long long)((uint64_t(1) << 52) - 1));
    __m512i x[max_fragment_u52];
    for (unsigned j = 0; j < u52_digits; ++j)
        x[j] = _mm512_load_si512(lanes[j]);
    alignas(64) uint64_t round_words[fragment_words + 1][8];
    for (unsigned r = 0; r < rounds; ++r) {
        __m512i carry = zero;
        for (unsigned j = 0; j < u52_digits; ++j) {
            const __m512i lo = _mm512_madd52lo_epu64(zero, x[j], B8);
            const __m512i hi = _mm512_madd52hi_epu64(zero, x[j], B8);
            const __m512i t = _mm512_add_epi64(lo, carry);
            carry = _mm512_add_epi64(hi, _mm512_srli_epi64(t, 52));
            x[j] = _mm512_and_si512(t, mask);
        }
        _mm512_store_si512(round_words[r], carry); // the integer part: the next word of every lane
    }
    for (unsigned u = 0; u < 8; ++u)
        for (unsigned r = 0; r < rounds; ++r)
            words[u * (fragment_words + 1) + r] = round_words[r][u];
}
void emit_fragments(uint8_t *out,uint64_t *first,uint64_t *overlap,const uint64_t *const fraction[8],
                    unsigned limbs,unsigned u52_digits,unsigned count,const DigitPlan &p) noexcept {
    require(count && count<=8 && limbs && limbs<=max_fragment_limbs && u52_digits &&
            u52_digits<=max_fragment_u52 && 52*u52_digits<64*limbs+52,
            SBN3_FATAL_ARGUMENT,"radix fragment group shape");
    // Partial groups do not pay for emitting eight inactive SIMD lanes.
    if(count<8){
        alignas(64) uint64_t words[8*9];
        extract_words(words,fraction,limbs,u52_digits,9,p);
        for(unsigned u=0;u<count;++u){
            emit_words(out+64*u,words+9*u,8,p);first[u]=words[9*u];overlap[u]=words[9*u+8];
        }
        return;
    }
    #define CASE(N) case N:leaf_detail::emit_full_group<N>(out,first,overlap,fraction,limbs,p);break;
    switch(u52_digits){CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10)}
    #undef CASE
}
void emit_words_reference(uint8_t *out, const uint64_t *words, size_t count, const DigitPlan &p) noexcept {
    for (size_t i = 0; i < count; ++i) {
        uint64_t x = words[i];
        for (unsigned j = word_digits; j-- > 0;) {
            out[i * 8 + j] = p.encode[x % p.base];
            x /= p.base;
        }
    }
}
bool parse_words_reference(uint64_t *words, const uint8_t *digits, size_t count, const DigitPlan &p) noexcept {
    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
        uint64_t x = 0;
        for (unsigned j = 0; j < word_digits; ++j) {
            const uint8_t v = p.decode[digits[i * 8 + j]];
            ok = ok && v != 0xff;
            x = x * p.base + (v == 0xff ? 0 : v);
        }
        words[i] = x;
    }
    return ok;
}
} // namespace sbn::v3::radix
