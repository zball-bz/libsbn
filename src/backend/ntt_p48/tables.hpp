/* Imported from labs/cr/cr_tables.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_tables.hpp — prime contexts + the prefix-stable bit-reversed twiddle
 * tower (rec-only), stored compressed.
 *
 * Tower law (FLINT/p50 convention, re-derived): w[0] = 1; for level t
 * (1..lg) with z_t = g^((p-1)/2^t) (order 2^t):
 *     w[2^(t-1) + s] = z_t · w[s],  s < 2^(t-1)
 * so w[n] = prod over set bits b of n of z_{b+1}; hence
 *     w[2n]^2 = w[n]  and  w[i | j] = w[i]·w[j] for bit-disjoint i, j.
 * A tree node (m, j) has modulus x^m − w[2j]^2 = x^m − w[j] and splits
 * with the twiddle w[2j] into children (m/2, 2j) [x^(m/2) − w[2j]] and
 * (m/2, 2j+1) [x^(m/2) + w[2j]]; leaf k carries x^R − w[k].
 *
 * Storage (2026-09-17): only e[n] = rec(w[2n]), n < 2^(lg-1) — 8 B per TWO
 * leaves instead of 40 B (w, winv and the halved inverse table). Everything
 * else is derived on access, exactly:
 *   odd entries     w[2n+1] = −w[2n]           and rec(p − c) = M52 − rec(c)
 *                   (c·2^52/p is never an integer for 0 < c < p)
 *   inverse tower   w[n] = ζ^bitrev(n) (ζ = z_lg). Inside a level block
 *                   [2^t, 2^(t+1)) the bit reversals of k and of its mirror
 *                   3·2^t − 1 − k add up to the full order, so
 *                       winv[k] = w[3·2^t − 1 − k],  winv[0] = 1,
 *                   and for even entries (n >= 1, b = bit_floor(n))
 *                       rec(winv[2n]) = M52 − e[3b − 1 − n].
 *   halved inverse  rec(c/2 mod p) = (rec(c) >> 1) + (c odd ? 2^51 : 0).
 * The level-k children k·j + c (c < k) of a tree node j >= 1 mirror to
 * e[k·(3b − j) − 1 − c], b = bit_floor(j): one bit_floor per kernel block
 * (TwCur). Node 0 has no enclosing block; it reads the 64-slot image ez under
 * the same indexing with (3b − j) := 8:
 *     ez[8k − 1 − c] = M52 − rec(winv[2c]),  k in {1, 2, 4, 8}, c < k.
 * The level roots come from the constexpr z_32 of arith.hpp; tables are built
 * eight entries at a time with the exact rec derivation (vrec_of). */
#include "arith.hpp"
#include "scratch_adapter.hpp"
#include <stdio.h>

namespace sbn::v3::SBN3_P48_NS {

/* z[t] = z_t (order 2^t) or its inverse, t = 1..32; z[0] and z[33..] are zero (never a valid level) */
constexpr unsigned tower_root_slots = 80;
inline void tower_roots(uint64_t z[tower_root_slots], int q, bool inverse){
    const uint64_t p = PR[q];
    for(unsigned t = 0; t < tower_root_slots; ++t) z[t] = 0;
    z[32] = inverse ? fixed_setup.prime[q].root32_inverse : fixed_setup.prime[q].root32;
    for(unsigned t = 32; t > 1; --t) z[t - 1] = mulm(z[t], z[t], p);
}
/* out[0..n): u[0] = 1, u[2^(t-1) + s] = z[t]·u[s] (t >= 1), rec-only and exact. Entries below 8 are
 * scalar; the vector loop may write up to 7 entries past n (rows are padded or rewritten later). */
inline void tower_fill(uint64_t *out, size_t n, const uint64_t *z, const Prime &pr, const PrimeV &pv_){
    const PrimeV pv = pv_regs(pv_);
    const uint64_t p = pr.p;
    if(!n) return;
    out[0] = pr.r1;
    size_t t = 1, blk = 1;
    for(; blk < 8 && blk < n; blk <<= 1, ++t)
        for(size_t s = 0; s < blk && blk + s < n; ++s)
            out[blk + s] = cc_of(mulm(z[t], cc_ld(out[s], p).c, p), p).rec;
    const V one = vset(1);
    for(; blk < n; blk <<= 1, ++t){
        const vcc zv = vcc_of(cc_of(z[t], p));
        const size_t count = blk < n - blk ? blk : n - blk;
        for(size_t s = 0; s < count; s += 8){
            const V c = _mm512_madd52hi_epu64(one, _mm512_loadu_si512(out + s), pv.p);
            _mm512_storeu_si512(out + blk + s, vrec_of(shp(b52(c, zv.c, zv.rec, pv.pn), pv.p), pv));
        }
    }
}
/* stored even nodes for a tower of 2^lg entries (at least 16: the first-block image reads e[0..8)) */
inline size_t tower_nodes(size_t lg){
    const size_t n = lg ? (size_t)1 << (lg - 1) : 1;
    return n < 16 ? 16 : n;
}
constexpr size_t tower_zero_slots = 64;
inline size_t tower_bytes(size_t lg){ return tower_nodes(lg) * sizeof(uint64_t) + 64; }

struct Primes {
    Prime P[NP];
    PrimeV V[NP];
    cc inverse_powers[NP][33];
    size_t lg;
    /* compressed tower covering 2^lg entries per direction */
    void init(scratch &storage, size_t lg_){
        lg = lg_;
        ::sbn::v3::require(lg <= 32, SBN3_FATAL_MATH, "p48 tower order");
        for(int q = 0; q < NP; ++q){
            Prime &pr = P[q];
            const uint64_t p = PR[q];
            pr.p = p; pr.p2 = 2 * p;
            if(!pr.e){ pr.F[0] = pr.F[1] = NULL; pr.M2f = 0; pr.lgF = 0; }   /* first init: no factor tables yet */
            uint64_t pi = 1;
            for(int i = 0; i < 6; ++i) pi *= 2 - p * pi;     /* p^-1 mod 2^64 */
            pr.J = (0 - pi) & M52;
            pr.r1 = (uint64_t)(((u128)1 << 52) / p);
            pr.inv2 = cc_of((p + 1) >> 1, p);
            inverse_powers_build(inverse_powers[q],p);
            pr.invpow2=inverse_powers[q];
            pr.lg = lg; pr.heap = nullptr;
            V[q].init(pr);
            const size_t n = tower_nodes(lg);
            uint64_t *e = (uint64_t *)storage.allocate(tower_bytes(lg));
            uint64_t *ez = (uint64_t *)storage.allocate(tower_zero_slots * sizeof(uint64_t));
            uint64_t z[tower_root_slots];
            tower_roots(z, q, false);
            tower_fill(e, n, z + 1, pr, V[q]);               /* e[2^(t-1) + s] = z_{t+1}·e[s] */
            for(size_t s = 0; s < tower_zero_slots; ++s) ez[s] = 0;
            for(size_t k = 1; k <= 8; k <<= 1)
                for(size_t c = 0; c < k; ++c){
                    const size_t b = (size_t)1 << (63 - __builtin_clzll(c | 1));
                    ez[8 * k - 1 - c] = c ? e[3 * b - 1 - c] : M52 - e[0];
                }
            pr.e = e; pr.ez = ez;
        }
    }
    /* factor tables for column materialization: w[2^k·b] = prod over the
     * set bits j of b of z_{j+k+1} (tower law): row k is the same doubling
     * tower with the level roots shifted by k */
    void factors_for(scratch &storage, size_t M2, size_t lgC){
        for(int q = 0; q < NP; ++q){
            Prime &pr = P[q];
            if(pr.F[0] && pr.M2f >= M2) continue;
            const size_t lgF = lgC;                                /* lgC cap */
            size_t lgM2 = 0; while(((size_t)1 << lgM2) < M2) ++lgM2;
            ::sbn::v3::require(lgF + lgM2 <= 32, SBN3_FATAL_MATH, "p48 factor order");
            for(int d = 0; d < 2; ++d){
                uint64_t z[tower_root_slots];
                tower_roots(z, q, d != 0);
                uint64_t *F = (uint64_t *)storage.allocate(sizeof(uint64_t) * (lgF + 1) * M2 + 64);
                for(size_t k = 0; k <= lgF; ++k) tower_fill(F + k * M2, M2, z + k, pr, V[q]);
                pr.F[d] = F;
            }
            pr.lgF = lgF; pr.M2f = M2;
        }
    }

};

/* ---- compressed-tower access ----------------------------------------------------------------
 * TwCur is the per-block cursor of tree node j: the level-k child k·j + c (c < k; k <= 8 at j = 0)
 * reads one stored word. H = true: the materialized heap table of col.hpp (entered at j = 1). */
struct TwCur { const uint64_t *tb; size_t m; };
template<bool WI, bool H> __attribute__((always_inline)) inline TwCur twcur(const Prime &P, size_t j){
    if constexpr(H){ TwCur t = { P.heap, j }; return t; }
    else if constexpr(!WI){ TwCur t = { P.e, j }; return t; }
    else{
        const size_t b = (size_t)1 << (63 - __builtin_clzll(j | 1));
        TwCur t = { j ? P.e : P.ez, j ? 3 * b - j : 8 };
        return t;
    }
}
template<bool WI, bool H> __attribute__((always_inline)) inline uint64_t twk(const TwCur &t, size_t k, size_t c){
    if constexpr(H || !WI) return t.tb[k * t.m + c];
    else return M52 - t.tb[k * t.m - 1 - c];
}
/* the same twiddle as a broadcast Shoup pair; the mirrored inverse is negated in the vector domain */
template<bool WI, bool H> __attribute__((always_inline)) inline vcc twv(const TwCur &t, size_t k, size_t c, const PrimeV &pv){
    vcc v;
    if constexpr(H || !WI) v.rec = vset(t.tb[k * t.m + c]);
    else v.rec = _mm512_xor_si512(vset(t.tb[k * t.m - 1 - c]), pv.M);
    v.c = _mm512_madd52hi_epu64(vset(1), v.rec, pv.p);
    return v;
}
/* rec of the twiddle of tree node `node` = rec(w_dir[2·node]) */
template<bool WI> __attribute__((always_inline)) inline uint64_t twr(const Prime &P, size_t node){
    return twk<WI, false>(twcur<WI, false>(P, node), 1, 0);
}
/* rec(w_dir[k0 .. k0+8)), k0 a multiple of 8: the 8 consecutive leaf roots */
template<bool WI> __attribute__((always_inline)) inline V tower8(const Prime &P, size_t k0){
    if constexpr(!WI){
        const V v = _mm512_maskz_loadu_epi64(0x0F, P.e + k0 / 2);
        const V x = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3), v);
        return _mm512_mask_xor_epi64(x, 0xAA, x, vset(M52));
    }else{
        const TwCur t = twcur<true, false>(P, k0 / 8);
        const V v = _mm512_maskz_loadu_epi64(0x0F, t.tb + 4 * t.m - 4);
        const V x = _mm512_permutexvar_epi64(_mm512_setr_epi64(3, 3, 2, 2, 1, 1, 0, 0), v);
        return _mm512_mask_xor_epi64(x, 0x55, x, vset(M52));
    }
}
/* rec(w_dir[2(8·j8 + c)]), c < 8: the eight level-8 children of node j8 */
template<bool WI> __attribute__((always_inline)) inline V tower_even8(const Prime &P, size_t j8){
    if constexpr(!WI) return _mm512_loadu_si512(P.e + 8 * j8);
    else{
        const TwCur t = twcur<true, false>(P, j8);
        const V v = _mm512_loadu_si512(t.tb + 8 * t.m - 8);
        return _mm512_xor_si512(_mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), v), vset(M52));
    }
}
/* winv[2j]·inv2 as a Shoup pair (partial nodes of the normalized truncated inverses): exact and
 * table-free, in scalar arithmetic (one widening multiply) so the vector units see two broadcasts */
__attribute__((always_inline)) inline vcc wi2_of(const Prime &P, size_t j, V){
    const uint64_t rec = twr<true>(P, j), c = (uint64_t)(((u128)rec * P.p) >> 52) + 1, odd = c & 1;
    cc r = { (c + (odd ? P.p : 0)) >> 1, (rec >> 1) + (odd << 51) };
    return vcc_of(r);
}

/* plain value of tower entry k (setup/reference use) */
inline uint64_t wval(const Prime &P, size_t k){ const uint64_t c = cc_ld(twr<false>(P, k >> 1), P.p).c; return k & 1 ? P.p - c : c; }
inline uint64_t wival(const Prime &P, size_t k){ const uint64_t c = cc_ld(twr<true>(P, k >> 1), P.p).c; return k & 1 ? P.p - c : c; }

} // namespace sbn::v3::SBN3_P48_NS
