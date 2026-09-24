// Radix leaf kernels (AVX-512 IFMA/VBMI). Emission is the radix-split cascade
// of DigitViewer with the 64-bit level on madd52hi: word / b^4, halves / b^2,
// pairs / b, then one vpermb through the output table: 8 words -> 64 bytes in
// one store. Parsing is maddubs [b,1] -> madd [b^2,1] -> mul_epu32 [b^4] with a
// branch-free validity mask. Magics satisfy the exact Granlund-Montgomery
// criterion and are re-verified by the leaf gate against the scalar reference.
#include "radix/leaf.hpp"
#include "radix/word_base.hpp"
#include "common/checked.hpp"
#include "radix_leaf_group.hpp"
#include <immintrin.h>
#include <cstring>
namespace sbn::v3::radix {
namespace {
using u128 = __uint128_t;
// Largest shift with an exact magic: floor(x * m / 2^(W+s)) == x / d for all x < limit.
bool magic(uint64_t d, uint64_t limit, unsigned W, uint64_t cap, uint64_t &m_out, unsigned &s_out) {
    for (unsigned s = W; s-- > 0;) {
        if (s < 63 && d <= (uint64_t(1) << s))
            continue; // m would reach the cap
        const u128 k = u128(1) << (W + s);
        const uint64_t m = uint64_t((k + d - 1) / d);
        if (m >= cap)
            continue;
        if ((u128(m) * d - k) * (limit - 1) < k) {
            m_out = m;
            s_out = s;
            return true;
        }
    }
    return false;
}
} // namespace
bool digit_plan_init(DigitPlan &p, unsigned base, const uint8_t *table) noexcept {
    p = {};
    if (base < 3 || base > 63 || !(base & (base - 1)))
        return false;
    p.base = base;
    p.b2 = uint64_t(base) * base;
    p.b4 = p.b2 * p.b2;
    p.b8 = p.b4 * p.b4;
    uint64_t m = 0;
    unsigned s = 0;
    if (!magic(p.b4, p.b8, 52, uint64_t(1) << 52, m, s))
        return false;
    p.m4 = m;
    p.s4 = s;
    if (!magic(p.b2, p.b4, 52, uint64_t(1) << 52, m, s))
        return false;
    p.m2 = m;
    p.s2 = s;
    if (!magic(base, p.b2, 16, uint64_t(1) << 16, m, s))
        return false;
    p.m1 = uint16_t(m);
    p.s1 = s;
    std::memset(p.decode, 0xff, sizeof p.decode);
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
void emit_words(uint8_t *out, const uint64_t *words, size_t count, const DigitPlan &p) noexcept {
    require(!(count & 7), SBN3_FATAL_ARGUMENT, "radix emit word count");
    const __m512i zero = _mm512_setzero_si512();
    const __m512i M4 = _mm512_set1_epi64((long long)p.m4), M2 = _mm512_set1_epi64((long long)p.m2);
    const __m512i B4 = _mm512_set1_epi64((long long)p.b4), B2 = _mm512_set1_epi64((long long)p.b2);
    const __m512i M1 = _mm512_set1_epi16(short(p.m1)), B1 = _mm512_set1_epi16(short(p.base));
    const __m512i table = _mm512_loadu_si512(p.encode);
    const __m128i s4 = _mm_cvtsi32_si128(int(p.s4)), s2 = _mm_cvtsi32_si128(int(p.s2)),
                  s1 = _mm_cvtsi32_si128(int(p.s1));
    for (size_t i = 0; i < count; i += 8) {
        const __m512i x = _mm512_loadu_si512(words + i);
        // level 1: a = x / b^4 (digits 7..4), b = x mod b^4 (digits 3..0)
        const __m512i a = _mm512_srl_epi64(_mm512_madd52hi_epu64(zero, x, M4), s4);
        const __m512i b = _mm512_sub_epi64(x, _mm512_mullo_epi64(a, B4));
        // level 2: quarters, each below b^2
        const __m512i qa = _mm512_srl_epi64(_mm512_madd52hi_epu64(zero, a, M2), s2);
        const __m512i ra = _mm512_sub_epi64(a, _mm512_mullo_epi64(qa, B2));
        const __m512i qb = _mm512_srl_epi64(_mm512_madd52hi_epu64(zero, b, M2), s2);
        const __m512i rb = _mm512_sub_epi64(b, _mm512_mullo_epi64(qb, B2));
        // per lane, u16s [qa, ra, qb, rb] are already in memory (MSD) order
        const __m512i y = _mm512_or_si512(_mm512_or_si512(qa, _mm512_slli_epi64(ra, 16)),
                                          _mm512_or_si512(_mm512_slli_epi64(qb, 32), _mm512_slli_epi64(rb, 48)));
        // level 3: every u16 pair into two digit bytes
        const __m512i qy = _mm512_srl_epi16(_mm512_mulhi_epu16(y, M1), s1);
        const __m512i ry = _mm512_sub_epi16(y, _mm512_mullo_epi16(qy, B1));
        const __m512i digits = _mm512_or_si512(qy, _mm512_slli_epi16(ry, 8));
        _mm512_storeu_si512(out + i * 8, _mm512_permutexvar_epi8(digits, table));
    }
}
void emit_word_chunks(uint8_t *out,const uint64_t *words,unsigned count,const DigitPlan &p) noexcept {
    require(count<=8,SBN3_FATAL_ARGUMENT,"radix chunk count");
    const auto wb=word_bases[p.base];
    const unsigned groups=(wb.digits+7)/8,prefix=groups*8-wb.digits;
    const __m512i zero=_mm512_setzero_si512(),M=_mm512_set1_epi64((long long)wb.reciprocal8),
                  B=_mm512_set1_epi64((long long)p.b8),one=_mm512_set1_epi64(1);
    __m512i q=_mm512_maskz_loadu_epi64(__mmask8((1u<<count)-1),words);
    alignas(64) uint64_t remainders[8];
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
        _mm512_store_si512(remainders,r);emit_words(digits[j],remainders,8,p);
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
bool parse_words(uint64_t *words, const uint8_t *digits, size_t count, const DigitPlan &p) noexcept {
    require(!(count & 7), SBN3_FATAL_ARGUMENT, "radix parse word count");
    const bool identity = p.identity;
    const __m512i top = _mm512_set1_epi8(char(p.base - 1));
    const __m512i C1 = _mm512_set1_epi16(short(0x0100 | p.base));       // [b, 1] per byte pair
    const __m512i C2 = _mm512_set1_epi32(int(0x00010000u | uint32_t(p.b2))); // [b^2, 1] per u16 pair
    const __m512i B4 = _mm512_set1_epi64((long long)p.b4);
    uint64_t bad = 0;
    for (size_t i = 0; i < count; i += 8) {
        __m512i d = _mm512_loadu_si512(digits + i * 8);
        if (!identity) {
            // Alphabet bytes -> values through the 256-entry table (rare path: scalar).
            alignas(64) uint8_t raw[64], value[64];
            _mm512_store_si512(raw, d);
            for (unsigned j = 0; j < 64; ++j)
                value[j] = p.decode[raw[j]];
            d = _mm512_load_si512(value);
        }
        bad |= uint64_t(_mm512_cmpgt_epu8_mask(d, top));
        const __m512i v16 = _mm512_maddubs_epi16(d, C1); // two digits per u16
        const __m512i v32 = _mm512_madd_epi16(v16, C2);  // four digits per u32
        const __m512i high = _mm512_srli_epi64(v32, 32); // digits 3..0 of the word (later bytes)
        const __m512i low = _mm512_mul_epu32(v32, B4);   // digits 7..4 (earlier bytes) * b^4
        _mm512_storeu_si512(words + i, _mm512_add_epi64(low, high));
    }
    return !bad;
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
