/* Imported from libsbn/include/sbn/detail/u52/cvt.h; arithmetic preserved, private namespace and prepared Frame adapter. */
#pragma once
namespace sbn::v3::u52 {

// u64 <-> u52 boundary layer and the public multiply entry point.
//
// Layout fact both directions exploit: 8 u52 digits = 416 bits = exactly 52
// bytes of the packed u64 representation, so block j of digits maps to byte
// offset 52*j with byte-aligned blocks. Within a block, digit i starts at byte
// floor(6.5*i) with a nibble shift of (i&1)*4.
//
// decode (u64 -> u52): one (masked) 64-byte load, one vpermb gathering each
// digit's 8-byte window, one variable shift, one mask -- 8 digits per ~4 ops.
// encode (u52 -> u64): fused with the signed canonicalization of the redundant
// product; the canonical vector is byte-permuted into the packed layout with
// two vpermb (byte 7 of a canonical digit is always zero and serves as the
// zero filler -- scheme ported from reference/ifma52_mul.cpp pack52_vec) and
// byte-masked stores. Exact u52 digit counts come from __builtin_clzll on the
// top limb.

// ---- decode -----------------------------------------------------------

// digit i gathers input bytes floor(6.5 i) .. +7 of its 52-byte block
SB_ALIGN64 static const uint8_t u52_dec_perm[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,
     6,  7,  8,  9, 10, 11, 12, 13,
    13, 14, 15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24, 25, 26,
    26, 27, 28, 29, 30, 31, 32, 33,
    32, 33, 34, 35, 36, 37, 38, 39,
    39, 40, 41, 42, 43, 44, 45, 46,
    45, 46, 47, 48, 49, 50, 51, 52,
};

// bit length of {ap, an}; tolerates zero high limbs
static inline uint64_t u64_bit_length(const uint64_t *ap, uint64_t an){
    while(an && ap[an - 1] == 0) --an;
    if(!an) return 0;
    return an * 64 - (uint64_t)__builtin_clzll(ap[an - 1]);
}

// unpack {ap, an} u64 limbs into n52 = ceil(bits/52) canonical u52 digits at
// r (vector-padded: full-vector stores, zero lanes beyond n52). Returns n52.
// EDGE: the input ends in a page's last 64 bytes; tail loads go page-safe.
template<bool EDGE>
static inline uint64_t u52_from_u64_(sb_pvec r, const uint64_t *ap, uint64_t an){
    const uint64_t bits = u64_bit_length(ap, an);
    const uint64_t n52 = (bits + 51) / 52;
    const uint8_t *p = (const uint8_t *)ap;
    int64_t rem = (int64_t)(an * 8);
    const sb_vec perm = sb_load((sb_cpvec)u52_dec_perm);
    const sb_vec sh = sb_setr_64(0, 4, 0, 4, 0, 4, 0, 4);
    uint64_t blocks = (n52 + 7) >> 3;
    for(; rem >= 64 && blocks > 1; --blocks, p += 52, rem -= 52, ++r)
        sb_store(r, sb_and(sb_srlv(sb_permb(perm, sb_load((sb_cpvec)p)), sh), SB_MASK52()));
    for(; blocks; --blocks, p += 52, rem -= 52, ++r){    // at most two masked
        const sb_vec w = !EDGE ? sb__fn(maskz_loadu_epi8)(rem >= 64 ? ~0ull : (rem > 0 ? (~0ull >> (64 - rem)) : 0), p)
                         : rem >= 64 ? sb_load((sb_cpvec)p) : rem > 0 ? u52_tail_loadu8(p, rem) : sb_zero();
        sb_store(r, sb_and(sb_srlv(sb_permb(perm, w), sh), SB_MASK52()));
    }
    return n52;
}
__attribute__((noinline, cold)) static uint64_t u52_from_u64_edge(sb_pvec r, const uint64_t *ap, uint64_t an){
    return u52_from_u64_<true>(r, ap, an);
}
static inline uint64_t u52_from_u64(sb_pvec r, const uint64_t *ap, uint64_t an){
    if(__builtin_expect(u52_page_edge(ap + an), 0)) return u52_from_u64_edge(r, ap, an);
    return u52_from_u64_<false>(r, ap, an);
}


// ---- packed u64 output windows ------------------------------------------

// Positive carry chain for redundant digits; canonical digits bypass it.
#define high_canon(vec, carry, sigcarry) do { \
    const sb_vec M_ = SB_MASK52(); \
    sb_vec hi_ = sb_srli(vec, 52); \
    sb_vec his_ = sb_alignr64(hi_, carry, 7); \
    sb_vec t_ = sb_add(sb_and(vec, M_), his_); \
    carry = hi_; \
    unsigned g_ = (unsigned)sb_gtu(t_, M_), p_ = (unsigned)sb_eq(t_, M_); \
    unsigned chain_ = p_ + ((g_ << 1) | sigcarry); \
    sigcarry = chain_ >> 8; \
    vec = sb_and(sb_sub(t_, M_, (__mmask8)(p_ ^ chain_), t_), M_); \
} while(0)

// Output windows: a 64-byte window starting x = 4w bytes into packed block bk
// takes bytes x..51 of bk, then bk+1, then (x > 40) the first x-40 bytes of
// bk+2; packed blocks carry zeros in bytes 52..63.
struct U52HighWindows { uint8_t i1[13][64], i2[13][64]; uint64_t m2[13]; };
static constexpr U52HighWindows u52_high_windows(){
    U52HighWindows t{};
    for(int w = 0; w < 13; ++w){
        const int x = 4 * w;
        t.m2[w] = 0;
        for(int b = 0; b < 64; ++b){
            if(b < 52 - x) t.i1[w][b] = (uint8_t)(x + b);
            else if(b < 104 - x) t.i1[w][b] = (uint8_t)(64 + b - (52 - x));
            else { t.i1[w][b] = 0; t.i2[w][b] = (uint8_t)(b - (104 - x)); t.m2[w] |= 1ull << b; }
        }
    }
    return t;
}
SB_ALIGN64 static constexpr U52HighWindows u52_high_win = u52_high_windows();

// Emit n u64 limbs from bit 64*cut_limbs; cut_limbs*8 >= 52*kv.
// canonical requires every input digit to lie in [0, 2^52); otherwise resolve carries.
// Output and digit storage must be disjoint; only the declared digits are read.
// R[kv..rv), rv = ceil(digits/8); lanes >= digits of R[rv-1] are not read.
static inline void u52_emit_window(uint64_t *out, sb_cpvec R, uint64_t kv, uint64_t digits, uint64_t n, uint64_t cut_limbs, bool canonical=false){
    const uint64_t rv = (digits + 7) / 8, nvec = (n + 7) / 8;
    sb_vec P0 = sb_zero(), P1 = sb_zero(), P2 = sb_zero(), cc = sb_zero();
    unsigned sig = 0;
    uint64_t pc = 0, j = 0, ws = 8 * cut_limbs - 52 * kv;
    auto put = [&]() __attribute__((always_inline)) {
        const unsigned w = (unsigned)(ws % 52) / 4;
        sb_vec x;
        if(w > 10){                                   // three blocks: pc-3, pc-2, pc-1
            x = sb__fn(permutex2var_epi8)(P0, sb_load((sb_cpvec)u52_high_win.i1[w]), P1);
            x = sb__fn(mask_permutexvar_epi8)(x, u52_high_win.m2[w], sb_load((sb_cpvec)u52_high_win.i2[w]), P2);
        }else x = sb__fn(permutex2var_epi8)(P1, sb_load((sb_cpvec)u52_high_win.i1[w]), P2);
        const int64_t left = (int64_t)(8 * (n - 8 * j));
        if(left >= 64) sb_store(out + 8 * j, x);
        else u52_tail_storeu8((uint8_t *)(out + 8 * j), x, left);
        ++j; ws += 64;
    };
    for(uint64_t v = kv;; ++v){
        sb_vec t = v + 1 < rv ? sb_load(R + v) : v + 1 == rv ? sb_load(R + v, u52_limb_mask(digits)) : sb_zero();
        if(!canonical) high_canon(t, cc, sig);
        P0 = P1; P1 = P2; P2 = u52_pack52(t); ++pc;
        if((ws + 63) / 52 + 1 == pc){                 // the window's last block just arrived
            put();
            if(j == nvec) return;
        }
    }
}
#undef high_canon

} // namespace
