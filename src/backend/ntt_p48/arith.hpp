/* Imported from labs/cr/cr_arith.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
#include "backend/ntt_p48/build_contract.hpp"
#ifndef SBN3_P48_NS
#define SBN3_P48_NS p48_np6
#endif
/* cr_arith.hpp — clean-room arithmetic layer: the prime set (NP primes < 2^48, prod 2^191.996 at
 * NP = 4), scalar setup helpers, Shoup (c, rec) pairs, and the AVX-512 IFMA vector primitives with
 * the lazy [0, 4p) discipline (16p < 2^52 gives the fold-free forward chain and the per-lane
 * inverse folds, cr_rows.hpp).
 *
 * Written from the algebra (docs/p50_ntt_design.md §1), not copied:
 *   b52(a, c, rec) = a·c − hi52(a·rec)·p  ∈ [0, 2p)  for any a < 2^52
 *   redc(lo, hi)   = (lo + hi·2^52)·2^-52 mod p, lazy < ~2.1p
 *   sh2(x)         = x < 2p ? x : x − 2p      (fold [0,4p) → [0,2p))
 *   shp(x)         = x < p  ? x : x − p
 * All prime-dependent vector constants are hoisted once into PrimeV. */
#include <immintrin.h>
#include "common/checked.hpp"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

namespace sbn::v3::SBN3_P48_NS {

typedef unsigned __int128 u128;
typedef __m512i V;

/* The prime set (NOTES §24.2): primes a·2^32 + 1 just below 2^48 (2-adicity 32). In 52-bit IFMA
 * lanes they leave 16p of headroom, which lets the radix-8/16 moths run their levels without
 * per-pair folds (the [0,4p) contract holds at every moth boundary; the fold schedule is derived
 * per lane in cr_rows.hpp). NP of them are used (-DCR_NP, default 4); the CRT capacity is
 * prod(PR[0..NP)) = 2^191.996 at NP = 4 (T = 80 to 2^31 trunks), 2^(48·NP − 0.001·NP) in general
 * (cr_geom.hpp plan_T_auto picks T from it). The p50 configuration (four primes < 2^50, 52-bit
 * slots) was removed on 2026-09-04 (NOTES §31). */
enum { NP = CR_NP, W8 = 8, NPMAX = 10 };
static_assert(NP >= 2 && NP <= NPMAX, "prime count");
static constexpr uint64_t PR[NPMAX] = {
    281346127691777ull,    /* 0xffe2·2^32 + 1 */
    281333242789889ull,    /* 0xffdf·2^32 + 1 */
    281255933378561ull,    /* 0xffcd·2^32 + 1 */
    281212983705601ull,    /* 0xffc3·2^32 + 1 */
    281204393771009ull,    /* 0xffc1·2^32 + 1 */
    281161444098049ull,    /* 0xffb7·2^32 + 1 */
    281135674294273ull,    /* 0xffb1·2^32 + 1 */
    281088429654017ull,    /* 0xffa6·2^32 + 1 */
    281019710177281ull,    /* 0xff96·2^32 + 1 */
    280710472531969ull     /* 0xff4e·2^32 + 1 */
};
static const uint64_t M52 = (1ull << 52) - 1;

/* ---- scalar (setup only) ---------------------------------------------- */
inline constexpr uint64_t mulm(uint64_t a, uint64_t b, uint64_t p){
    return (uint64_t)((u128)a * b % p);
}
inline constexpr uint64_t powm(uint64_t a, uint64_t e, uint64_t p){
    uint64_t r = 1;
    for(; e; e >>= 1, a = mulm(a, a, p)) if(e & 1) r = mulm(r, a, p);
    return r;
}
inline constexpr uint64_t invm(uint64_t a, uint64_t p){ return powm(a, p - 2, p); }

struct cc { uint64_t c, rec; };
inline constexpr cc cc_of(uint64_t c, uint64_t p){
    cc t = { c, (uint64_t)(((u128)c << 52) / p) };
    return t;
}
inline constexpr uint64_t primitive_root(uint64_t p){
    uint64_t f[16]{}; int nf = 0; uint64_t n = p - 1;
    for(uint64_t d = 2; (u128)d * d <= n; d += (d == 2 ? 1 : 2))
        if(n % d == 0){ f[nf++] = d; while(n % d == 0) n /= d; }
    if(n > 1) f[nf++] = n;
    for(uint64_t g = 2;; ++g){
        int ok = 1;
        for(int i = 0; i < nf && ok; ++i)
            if(powm(g, (p - 1) / f[i], p) == 1) ok = 0;
        if(ok) return g;
    }
}

// Every entry depends only on the fixed prime family and this NP instance.
// Constant initialization; no first-use guard, heap, or run-time modular inverse.
struct FixedPrimeSetup {
    uint64_t cofactor, cofactor_inverse, r52_inverse, decode[5];
    cc prefix_inverse;
    // z_32 = g^((p-1)/2^32) for the smallest primitive root g, and its inverse:
    // every tower level root is z_t = z_32^(2^(32-t)) (tables.hpp).
    uint64_t root32, root32_inverse;
};
struct FixedSetup { FixedPrimeSetup prime[NP]; };
inline constexpr FixedSetup fixed_setup = [] {
    FixedSetup result{};
    for (int q = 0; q < NP; ++q) {
        const uint64_t p = PR[q], r52 = (uint64_t)((u128(1) << 52) % p);
        auto &s = result.prime[q];
        s.cofactor = 1;
        for (int k = 0; k < NP; ++k) if (k != q) s.cofactor = mulm(s.cofactor, PR[k], p);
        s.cofactor_inverse = invm(s.cofactor, p);
        s.r52_inverse = invm(r52, p);
        uint64_t prefix = 1;
        for (int k = 0; k < q; ++k) prefix = mulm(prefix, PR[k], p);
        s.prefix_inverse = cc_of(invm(prefix, p), p);
        s.decode[0] = r52;
        for (int k = 1; k < 5; ++k) s.decode[k] = mulm(s.decode[k - 1], (uint64_t(1) << 48) % p, p);
        s.root32 = powm(primitive_root(p), (p - 1) >> 32, p);
        s.root32_inverse = invm(s.root32, p);
    }
    return result;
}();
inline constexpr uint64_t inverse_power2_or_general(uint64_t x, uint64_t p) {
    // x divides p-1: x * (p - (p-1)/x) == 1 mod p. Includes x=1.
    if (x && !(x & (x - 1)) && !((p - 1) & (x - 1)))
        return p - ((p - 1) >> __builtin_ctzll(x));
    return invm(x, p);
}
/* rec-only tables: c = floor(rec·p / 2^52) + 1 (exact, p odd, 0<c<p) */
inline cc cc_ld(uint64_t rec, uint64_t p){
    cc t = { (uint64_t)(((u128)rec * p) >> 52) + 1, rec };
    return t;
}
inline void inverse_powers_build(cc *out,uint64_t p){
    uint64_t inverse=1;
    for(unsigned k=0;k<=32;++k){out[k]=cc_of(inverse,p);inverse=(inverse+(inverse&1?p:0))>>1;}
}

/* ---- vector primitives -------------------------------------------------- */
inline V vset(uint64_t x){ return _mm512_set1_epi64((long long)x); }

/* Barrett-by-constant, no pre-reduction: a < 2^52 → [0, 2p) */
inline V b52(V a, V c, V rec, V pn){
    V z = _mm512_setzero_si512();
    V q = _mm512_madd52hi_epu64(z, a, rec);
    V s = _mm512_madd52lo_epu64(z, a, c);
    s = _mm512_madd52lo_epu64(s, q, pn);
    return _mm512_and_si512(s, vset(M52));
}
inline V sh2(V x, V p2){ return _mm512_min_epu64(x, _mm512_sub_epi64(x, p2)); }
inline V shp(V x, V p){ return _mm512_min_epu64(x, _mm512_sub_epi64(x, p)); }
/* wide REDC52 of a (lo, hi) 104-bit column pair: t·2^-52 mod p, lazy */
inline V redc(V lo, V hi, V J, V p, V M){
    V lom = _mm512_and_si512(lo, M);
    V m = _mm512_and_si512(_mm512_madd52lo_epu64(_mm512_setzero_si512(), lom, J), M);
    V r = _mm512_madd52hi_epu64(_mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52)), m, p);
    __mmask8 cy = _mm512_test_epi64_mask(lom, M);
    return _mm512_mask_add_epi64(r, cy, r, vset(1));
}
/* single-product REDC: a·c·2^-52 mod p (a, c < 2^52) — Montgomery form */
inline V redc1(V a, V c, V J, V p, V M){
    V z = _mm512_setzero_si512();
    V xl = _mm512_madd52lo_epu64(z, a, c);
    V xh = _mm512_madd52hi_epu64(z, a, c);
    V xm = _mm512_and_si512(xl, M);
    V m = _mm512_and_si512(_mm512_madd52lo_epu64(z, xm, J), M);
    V r = _mm512_madd52hi_epu64(xh, m, p);
    __mmask8 cy = _mm512_test_epi64_mask(xm, M);
    return _mm512_mask_add_epi64(r, cy, r, vset(1));
}
/* x·inv2 mod p by halving: x in [0,4p) → sh2 → [0,2p) → +p·odd → >>1 → [0,1.5p) */
inline V halve(V x, V p, V p2){
    V y = sh2(x, p2);
    V odd = _mm512_and_si512(y, vset(1));
    V add = _mm512_and_si512(_mm512_sub_epi64(_mm512_setzero_si512(), odd), p);
    return _mm512_srli_epi64(_mm512_add_epi64(y, add), 1);
}

struct vcc { V c, rec; };
inline vcc vcc_of(cc t){ vcc v = { vset(t.c), vset(t.rec) }; return v; }
/* rec-only → pair, in the vector domain (one IFMA off the broadcast) */
inline vcc vcc_ofr(uint64_t rec, V vp){
    vcc v;
    v.rec = vset(rec);
    v.c = _mm512_madd52hi_epu64(vset(1), v.rec, vp);
    return v;
}
inline vcc vcc_ofr8(const uint64_t *rec8, V vp){   /* 8 lane-distinct */
    vcc v;
    v.rec = _mm512_loadu_si512(rec8);
    v.c = _mm512_madd52hi_epu64(vset(1), v.rec, vp);
    return v;
}

/* ---- prime context ------------------------------------------------------ */
struct Prime {
    uint64_t p, p2, J;          /* J = −p^-1 mod 2^52                    */
    uint64_t r1;                /* floor(2^52 / p): the rec of c = 1 (foldz) */
    cc inv2;
    const cc *invpow2=nullptr;   /* plan-owned normalization constants, k=0..32 */
    /* Compressed prefix-stable BR tower (tables.hpp): e[n] = rec(w[2n]), n < 2^(lg-1). The odd
     * entries, the inverse tower and the halved inverse twiddles are derived on access; ez is the
     * 64-slot image of the first inverse block under the same mirrored indexing. */
    const uint64_t *e = nullptr, *ez = nullptr;
    const uint64_t *heap = nullptr;   /* H = true kernels: the materialized per-column table (col.hpp) */
    size_t lg;
    /* small-table columns (NOTES §10): F[d][k·M2f + b] = rec(w_d[2^k·b]),
     * k <= lgF, b < M2f — the per-level/leaf factors of column b; with the
     * shared prefix w_d[0..2C) they replace the full tower for po2 plans */
    uint64_t *F[2]; size_t lgF, M2f;
};
struct PrimeV {                 /* hoisted vector constants               */
    V p, p2, pn, J, M;
    V p4, p8, r1;               /* 4p, 8p, rec(1): the p48 fold discipline (sh4/sh8/foldz) */
    V r2h, r2l;                 /* R2 = floor(2^104/p) split 52: exact rec-from-c (vrec_of) */
    inline void init(const Prime &P){
        p = vset(P.p); p2 = vset(P.p2); pn = vset((1ull << 52) - P.p);
        J = vset(P.J); M = vset(M52);
        p4 = vset(4 * P.p); p8 = vset(8 * P.p); r1 = vset(P.r1);
        const uint64_t R2 = (uint64_t)(((u128)1 << 104) / P.p);
        r2h = vset(R2 >> 52); r2l = vset(R2 & M52);
    }
    /* the kernels that carry (p, 2p, 2^52 − p) as vectors rebuild the context from them */
    inline void from3(V p_, V p2_, V pn_, const Prime &P){
        p = p_; p2 = p2_; pn = pn_; J = vset(P.J); M = vset(M52);
        p4 = _mm512_add_epi64(p2_, p2_); p8 = _mm512_add_epi64(p4, p4); r1 = vset(P.r1);
    }
};
/* register-resident copy of the context: the plain struct copy (10 vectors, 640 B) is a memcpy
 * call per kernel call — visible on the per-block leaves (b16, leaf8x2) */
__attribute__((always_inline)) inline PrimeV pv_regs(const PrimeV &s){
    PrimeV r; r.p = s.p; r.p2 = s.p2; r.pn = s.pn; r.J = s.J; r.M = s.M; r.p4 = s.p4; r.p8 = s.p8; r.r1 = s.r1; r.r2h = s.r2h; r.r2l = s.r2l; return r;
}
/* p48 folds (NOTES §24.2): sh4/sh8 halve a range by a fixed multiple of p; foldz takes any
 * x < 2^52 to [0, 2p) with rec(1) = floor(2^52/p): q = hi52(x·r1), x − q·p (mod 2^52) */
inline V sh4(V x, V p4){ return _mm512_min_epu64(x, _mm512_sub_epi64(x, p4)); }
inline V sh8(V x, V p8){ return _mm512_min_epu64(x, _mm512_sub_epi64(x, p8)); }
inline V foldz(V x, V r1, V pn){
    V q = _mm512_madd52hi_epu64(_mm512_setzero_si512(), x, r1);
    return _mm512_and_si512(_mm512_madd52lo_epu64(x, q, pn), vset(M52));
}
/* rec = floor(c·2^52/p) for canonical c in [1, p): rec' = c·R2h + hi52(c·R2l)
 * is rec or rec−1 (R2 = floor(2^104/p) truncation < 1 unit), one fix-up by
 * hi52((rec'+1)·p) < c. 3 IFMA + 4 ops. */
inline V vrec_of(V c, const PrimeV &pv){
    const V z = _mm512_setzero_si512(), one = vset(1);
    V r = _mm512_madd52lo_epu64(z, c, pv.r2h);
    r = _mm512_add_epi64(r, _mm512_madd52hi_epu64(z, c, pv.r2l));
    const V r1 = _mm512_add_epi64(r, one);
    const V t = _mm512_madd52hi_epu64(z, r1, pv.p);
    return _mm512_mask_mov_epi64(r, _mm512_cmplt_epu64_mask(t, c), r1);
}

/* 8x8 u64 transpose, in place */
inline void tr8(V x[8]){
    V u[8], v[8];
    for(int i = 0; i < 4; ++i){
        u[2*i]   = _mm512_unpacklo_epi64(x[2*i], x[2*i+1]);
        u[2*i+1] = _mm512_unpackhi_epi64(x[2*i], x[2*i+1]);
    }
    v[0] = _mm512_shuffle_i64x2(u[0], u[2], 0x88);
    v[1] = _mm512_shuffle_i64x2(u[1], u[3], 0x88);
    v[2] = _mm512_shuffle_i64x2(u[0], u[2], 0xdd);
    v[3] = _mm512_shuffle_i64x2(u[1], u[3], 0xdd);
    v[4] = _mm512_shuffle_i64x2(u[4], u[6], 0x88);
    v[5] = _mm512_shuffle_i64x2(u[5], u[7], 0x88);
    v[6] = _mm512_shuffle_i64x2(u[4], u[6], 0xdd);
    v[7] = _mm512_shuffle_i64x2(u[5], u[7], 0xdd);
    x[0] = _mm512_shuffle_i64x2(v[0], v[4], 0x88);
    x[1] = _mm512_shuffle_i64x2(v[1], v[5], 0x88);
    x[2] = _mm512_shuffle_i64x2(v[2], v[6], 0x88);
    x[3] = _mm512_shuffle_i64x2(v[3], v[7], 0x88);
    x[4] = _mm512_shuffle_i64x2(v[0], v[4], 0xdd);
    x[5] = _mm512_shuffle_i64x2(v[1], v[5], 0xdd);
    x[6] = _mm512_shuffle_i64x2(v[2], v[6], 0xdd);
    x[7] = _mm512_shuffle_i64x2(v[3], v[7], 0xdd);
}

} // namespace sbn::v3::SBN3_P48_NS
