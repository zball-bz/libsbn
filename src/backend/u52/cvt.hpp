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
static inline uint64_t u52_from_u64(sb_pvec r, const uint64_t *ap, uint64_t an){
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
        const sb_vec w = sb__fn(maskz_loadu_epi8)(
            rem >= 64 ? ~0ull : (rem > 0 ? (~0ull >> (64 - rem)) : 0), p);
        sb_store(r, sb_and(sb_srlv(sb_permb(perm, w), sh), SB_MASK52()));
    }
    return n52;
}

} // namespace
