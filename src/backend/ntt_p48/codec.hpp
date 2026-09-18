/* Imported from labs/cr/cr_codec.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_codec.hpp — integer ↔ residue endpoints and the packed-48 planes.
 *
 * Decode, T ≤ 88 (2-piece funnel, BT = 8 trunks per vector): trunk t of vector v
 *   sits at bit 8T·v + T·t = limb ((8T·v)>>6) + ((off + T·t) >> 6) with the
 *   intra-limb offset off = (8T·v) & 63 ∈ {0, 32} (T ≡ 0 mod 4: two index
 *   and shift variants); lo = low LW bits, hi = top 52 bits. Per prime:
 *   r = REDC(hi · 2^T mod p) + lo = hi·2^LW + lo mod p  (then canonical).
 *   T = 80 has a byte-permute funnel (t80); T > 88 (NP ≥ 5) the wide codec (Dec::t8, §32).
 *
 * Leaf-L layout picks: a group of 8 leaves = 8L consecutive trunks = L
 * natural vectors; group vector s, lane t = trunk (8g + t)·L + s =
 * natural vector (tL + s)/8, lane (tL + s) % 8.
 *
 * Packed planes: 8 canonical residues < 2^48 → 48 bytes (SLOT = 48); blocked layout (cr_geom). */
#include "geom.hpp"
#include "scratch_adapter.hpp"

namespace sbn::v3::SBN3_P48_NS {

/* the product's plane scale per prime: inv(C · rowscale · lanescale) · 2^52 (the REDC stray of the
 * leaf products). Applied in Garner (cr_emit.hpp) — or, with the wide codec (T > 88, packed
 * canonical planes), folded for free into A's decode constants (§32: fold_scale_on) */
inline uint64_t plane_scale(const Plan &pl, uint64_t p){
    uint64_t s = mulm(inverse_power2_or_general((pl.C * pl.rowscale) % p, p), (uint64_t)(((u128)1 << 52) % p), p);
    if(pl.Lg > 1) s = mulm(s, inverse_power2_or_general(8, p), p);
    return s;
}
/* CRT cofactor inverse, scalar plan/setup only. */
inline uint64_t cofactor_inverse(int q){
    return fixed_setup.prime[q].cofactor_inverse;
}
inline int fold_scale_on(const Plan &pl){
    return pl.T > 88 && SLOT == 48 && SLOTO == 48;
}

inline constexpr int DECODE_PIECES=NP<=8?4:5;
template<bool> struct DecodeFourthWindow {};
template<> struct DecodeFourthWindow<true> { V index[DECODE_PIECES]; __mmask64 mask[DECODE_PIECES]; bool load; };
struct Dec {
    V vMLO;
    V iA_lo[2], sA_lo[2], iA_hi[2], sA_hi[2];   /* variant = intra-limb offset 0 / 32 */
    V vC[NP], vJ[NP], vP[NP];
    V vCl[NP], vCr[NP], vPn[NP];                 /* (c, rec) of 2^LW mod p, 2^52 − p: the lazy b52 decode (K3', NOTES §17) */
    int lazy;                                    /* residues left in [0, 4p): the row transforms take lazy inputs */
    const uint64_t *ab;
    size_t TB8;                                  /* bits per vector = 8T */
    /* T = 80 (word-aligned trunks, NOTES §25): lo = bytes [10l, 10l+4), hi = bytes [10l+4, 10l+10) of
     * the vector's 80-byte window, each by one 2-source zeroing byte permute over the two 64-B loads;
     * r = b52(hi, 2^32 mod p) + lo < 2p + 2^32 (lazy) — 9 ops per (vector, prime) instead of 15 */
    int t80;
    V bl_idx, bh_idx; __mmask64 bl_msk, bh_msk;
    V vC32c[NP], vC32r[NP];
    /* T ≡ 0 mod 8, T > 88 (NP ≥ 5, NOTES §32): byte-aligned trunks of T/8 bytes; a vector's window is
     * T ≤ 232 bytes; up to four 64-B loads (fourth window at NP9/10); piece k of lane l = bytes
     * [l·T/8 + 6k, +6) ∩ the trunk (npc = ceil(T/48) pieces), each one zeroing byte permute over the
     * window (L0,L1) (pieces ending before byte 128) or (L1,L2) (the rest); per prime
     * r = REDC(Σ_k d_k·C_k), C_k = 2^(48k+52) mod p: 2·npc IFMA + 8 ops, lazy < 1.313p (at most five 48-bit slices). */
    int t8, npc, ld3;
    V pa_idx[DECODE_PIECES], pb_idx[DECODE_PIECES]; __mmask64 pa_msk[DECODE_PIECES], pb_msk[DECODE_PIECES];
    V vCw[DECODE_PIECES][NP];
    [[no_unique_address]] DecodeFourthWindow<(NP>8)> fourth;
};

/* sc: per-prime multipliers folded into the wide codec's constants (NULL = none; T ≤ 88 ignores it) */
inline void dec_init(Dec &d, const uint64_t *a, const Plan &pl, const Primes &PS, const uint64_t *sc = NULL){
    memset(&d, 0, sizeof d);
    d.ab = a;
    d.TB8 = 8 * (size_t)pl.T;
    d.lazy = 1;
    for(int q = 0; q < NP; ++q){
        const uint64_t p = PS.P[q].p;
        d.vJ[q] = vset(PS.P[q].J); d.vP[q] = vset(p); d.vPn[q] = vset((1ull << 52) - p);
    }
    d.t8 = pl.T > 88;
    if(!d.t8){
        for(int var = 0; var < 2; ++var){
            alignas(64) uint64_t li[8], ls[8], hi[8], hs[8];
            for(int l = 0; l < 8; ++l){
                const long long bp = (long long)pl.T * l + 32 * var;
                li[l] = (uint64_t)(bp >> 6); ls[l] = (uint64_t)(bp & 63);
                const long long hp = bp + pl.LW;
                hi[l] = (uint64_t)(hp >> 6); hs[l] = (uint64_t)(hp & 63);
            }
            d.iA_lo[var] = _mm512_load_si512(li); d.sA_lo[var] = _mm512_load_si512(ls);
            d.iA_hi[var] = _mm512_load_si512(hi); d.sA_hi[var] = _mm512_load_si512(hs);
        }
        d.vMLO = vset((1ull << pl.LW) - 1);
        for(int q = 0; q < NP; ++q){
            const uint64_t p = PS.P[q].p;
            d.vC[q] = vset(powm(2, (uint64_t)pl.T, p));
            /* the lazy path multiplies plainly (b52), so its constant is 2^LW mod p — the
             * Montgomery path's 2^T mod p carries the 2^-52 of REDC (T − 52 = LW) */
            const uint64_t Cl = (uint64_t)(((u128)1 << pl.LW) % p);
            d.vCl[q] = vset(Cl);
            d.vCr[q] = vset(cc_of(Cl, p).rec);
        }
        d.t80 = pl.T == 80;
        alignas(64) uint8_t bl[64], bh[64]; uint64_t ml = 0, mh = 0;
        for(int l = 0; l < 8; ++l)
            for(int b = 0; b < 8; ++b){
                bl[8 * l + b] = (uint8_t)(b < 4 ? 10 * l + b : 0);     if(b < 4) ml |= 1ull << (8 * l + b);
                bh[8 * l + b] = (uint8_t)(b < 6 ? 10 * l + 4 + b : 0); if(b < 6) mh |= 1ull << (8 * l + b);
            }
        d.bl_idx = _mm512_load_si512(bl); d.bh_idx = _mm512_load_si512(bh); d.bl_msk = ml; d.bh_msk = mh;
        for(int q = 0; q < NP; ++q){
            const uint64_t p = PS.P[q].p, C32 = (uint64_t)(((u128)1 << 32) % p);
            d.vC32c[q] = vset(C32); d.vC32r[q] = vset(cc_of(C32, p).rec);
        }
        return;
    }
    /* the wide codec (§32) */
    const int tb = pl.T / 8;
    d.npc = (pl.T + 47) / 48; d.ld3 = pl.T > 128;
#if CR_NP>8
    d.fourth.load=pl.T>192;
#endif
    for(int k = 0; k < d.npc; ++k){
        alignas(64) uint8_t ia[64], ib[64], ic[64]{}; uint64_t ma = 0, mb = 0;[[maybe_unused]] uint64_t mc=0;
        memset(ia, 0, 64); memset(ib, 0, 64);
        const int len = (k + 1) * 6 <= tb ? 6 : tb - 6 * k;
        for(int l = 0; l < 8; ++l){
            const int b0 = l * tb + 6 * k;
            const int winA = b0 + len <= 128;
            for(int b = 0; b < len; ++b){
                const int dst = 8 * l + b, src = b0 + b;
                if(winA){ ia[dst] = (uint8_t)src; ma |= 1ull << dst; }
                else if(b0+len<=192){ ib[dst] = (uint8_t)(src - 64); mb |= 1ull << dst; }
                else {ic[dst]=(uint8_t)(src-128);mc|=1ull<<dst;}
            }
        }
        d.pa_idx[k] = _mm512_load_si512(ia); d.pb_idx[k] = _mm512_load_si512(ib); d.pa_msk[k] = ma; d.pb_msk[k] = mb;
#if CR_NP>8
        {d.fourth.index[k]=_mm512_load_si512(ic);d.fourth.mask[k]=mc;}
#endif
        for(int q = 0; q < NP; ++q){
            const uint64_t p = PS.P[q].p;
            uint64_t C = fixed_setup.prime[q].decode[k];
            if(sc) C = mulm(C, sc[q] % p, p);
            d.vCw[k][q] = vset(C);
        }
    }
}
/* vector u of an operand: first limb and decode variant (bit 8T·u) */
__attribute__((always_inline)) inline size_t dec_limb(const Dec &d, size_t u){ return (d.TB8 * u) >> 6; }
__attribute__((always_inline)) inline int dec_var(const Dec &d, size_t u){ return (int)(((d.TB8 * u) & 63) >> 5); }

/* the wide codec's funnel: piece k of the 8 trunks (zero-extended bytes) */
__attribute__((always_inline)) inline
V dec8_piece(const Dec &d, int k, V L0, V L1, V L2, V L3=_mm512_setzero_si512()){
    V pc = _mm512_maskz_permutex2var_epi8(d.pa_msk[k], L0, d.pa_idx[k], L1);
    if(d.pb_msk[k]) pc = _mm512_or_si512(pc, _mm512_maskz_permutex2var_epi8(d.pb_msk[k], L1, d.pb_idx[k], L2));
#if CR_NP>8
    if(d.fourth.mask[k])pc=_mm512_or_si512(pc,_mm512_maskz_permutex2var_epi8(d.fourth.mask[k],L2,d.fourth.index[k],L3));
#endif
    return pc;
}
/* Expanded 48-bit slices can be shared by every prime of a row. */
__attribute__((always_inline)) inline
V dec8_prepared(const Dec &d,const V *pieces,int q){
    const V z=_mm512_setzero_si512();V lo=z,hi=z;
    for(int k=0;k<d.npc;++k){lo=_mm512_madd52lo_epu64(lo,pieces[k],d.vCw[k][q]);hi=_mm512_madd52hi_epu64(hi,pieces[k],d.vCw[k][q]);}
    V result=redc(lo,hi,d.vJ[q],d.vP[q],vset(M52));
    return d.lazy?result:shp(result,d.vP[q]);
}
/* one prime of one vector from a local piece (readable 16 limbs, 24 when T>128, 32 when T>192), variant var */
__attribute__((always_inline)) inline
V dec1q(const Dec &d, const uint64_t *base, int q, int var){
    const V L0 = _mm512_loadu_si512(base);
    const V L1 = _mm512_loadu_si512(base + 8);
    if(d.t8){
        const V L2 = d.ld3 ? _mm512_loadu_si512(base + 16) : L0;
        V L3=L0;
#if CR_NP>8
        if(d.fourth.load)L3=_mm512_loadu_si512(base+24);
#endif
        const V z = _mm512_setzero_si512();
        V lo = z, hi = z;
        for(int k = 0; k < d.npc; ++k){
            const V pc = dec8_piece(d, k, L0, L1, L2,L3);
            lo = _mm512_madd52lo_epu64(lo, pc, d.vCw[k][q]);
            hi = _mm512_madd52hi_epu64(hi, pc, d.vCw[k][q]);
        }
        V r = redc(lo, hi, d.vJ[q], d.vP[q], vset(M52));       /* < p + 2^46 + 4: lazy */
        if(!d.lazy) r = shp(r, d.vP[q]);
        return r;
    }
    if(d.t80){
        const V lo = _mm512_maskz_permutex2var_epi8(d.bl_msk, L0, d.bl_idx, L1);
        const V hi = _mm512_maskz_permutex2var_epi8(d.bh_msk, L0, d.bh_idx, L1);
        V r = _mm512_add_epi64(b52(hi, d.vC32c[q], d.vC32r[q], d.vPn[q]), lo);
        if(!d.lazy){ r = shp(r, d.vP[q]); r = shp(r, d.vP[q]); }
        return r;
    }
    const V one = vset(1);
    V A = _mm512_permutex2var_epi64(L0, d.iA_lo[var], L1);
    V B = _mm512_permutex2var_epi64(L0, _mm512_add_epi64(d.iA_lo[var], one), L1);
    const V lo = _mm512_and_si512(_mm512_shrdv_epi64(A, B, d.sA_lo[var]), d.vMLO);
    A = _mm512_permutex2var_epi64(L0, d.iA_hi[var], L1);
    B = _mm512_permutex2var_epi64(L0, _mm512_add_epi64(d.iA_hi[var], one), L1);
    const V M = vset(M52);
    const V hi = _mm512_and_si512(_mm512_shrdv_epi64(A, B, d.sA_hi[var]), M);
    if(d.lazy)                                   /* b52: hi·2^T mod p in [0, 2p), + lo < 2^LW → < 4p (5 ops vs 13) */
        return _mm512_add_epi64(b52(hi, d.vCl[q], d.vCr[q], d.vPn[q]), lo);
    V r = _mm512_add_epi64(redc1(hi, d.vC[q], d.vJ[q], d.vP[q], M), lo);
    r = shp(r, d.vP[q]);
    return shp(r, d.vP[q]);
}
/* two primes from one funnel (rows pass pair mode, §21: the piece buffer is read twice per row
 * instead of four times and the two permute funnels are shared) */
__attribute__((always_inline)) inline
void dec2q(const Dec &d, const uint64_t *base, int q0, int var, V &o0, V &o1){
    const V L0 = _mm512_loadu_si512(base);
    const V L1 = _mm512_loadu_si512(base + 8);
    if(d.t8){
        const V L2 = d.ld3 ? _mm512_loadu_si512(base + 16) : L0;
        V L3=L0;
#if CR_NP>8
        if(d.fourth.load)L3=_mm512_loadu_si512(base+24);
#endif
        const V z = _mm512_setzero_si512();
        V lo0 = z, hi0 = z, lo1 = z, hi1 = z;
        for(int k = 0; k < d.npc; ++k){
            const V pc = dec8_piece(d, k, L0, L1, L2,L3);
            lo0 = _mm512_madd52lo_epu64(lo0, pc, d.vCw[k][q0]);     hi0 = _mm512_madd52hi_epu64(hi0, pc, d.vCw[k][q0]);
            lo1 = _mm512_madd52lo_epu64(lo1, pc, d.vCw[k][q0 + 1]); hi1 = _mm512_madd52hi_epu64(hi1, pc, d.vCw[k][q0 + 1]);
        }
        const V M = vset(M52);
        o0 = redc(lo0, hi0, d.vJ[q0], d.vP[q0], M); o1 = redc(lo1, hi1, d.vJ[q0 + 1], d.vP[q0 + 1], M);
        if(!d.lazy){ o0 = shp(o0, d.vP[q0]); o1 = shp(o1, d.vP[q0 + 1]); }
        return;
    }
    if(d.t80){
        const V lo = _mm512_maskz_permutex2var_epi8(d.bl_msk, L0, d.bl_idx, L1);
        const V hi = _mm512_maskz_permutex2var_epi8(d.bh_msk, L0, d.bh_idx, L1);
        o0 = _mm512_add_epi64(b52(hi, d.vC32c[q0], d.vC32r[q0], d.vPn[q0]), lo);
        o1 = _mm512_add_epi64(b52(hi, d.vC32c[q0 + 1], d.vC32r[q0 + 1], d.vPn[q0 + 1]), lo);
        if(!d.lazy){ o0 = shp(shp(o0, d.vP[q0]), d.vP[q0]); o1 = shp(shp(o1, d.vP[q0 + 1]), d.vP[q0 + 1]); }
        return;
    }
    const V one = vset(1);
    V A = _mm512_permutex2var_epi64(L0, d.iA_lo[var], L1);
    V B = _mm512_permutex2var_epi64(L0, _mm512_add_epi64(d.iA_lo[var], one), L1);
    const V lo = _mm512_and_si512(_mm512_shrdv_epi64(A, B, d.sA_lo[var]), d.vMLO);
    A = _mm512_permutex2var_epi64(L0, d.iA_hi[var], L1);
    B = _mm512_permutex2var_epi64(L0, _mm512_add_epi64(d.iA_hi[var], one), L1);
    const V M = vset(M52);
    const V hi = _mm512_and_si512(_mm512_shrdv_epi64(A, B, d.sA_hi[var]), M);
    if(d.lazy){
        o0 = _mm512_add_epi64(b52(hi, d.vCl[q0], d.vCr[q0], d.vPn[q0]), lo);
        o1 = _mm512_add_epi64(b52(hi, d.vCl[q0 + 1], d.vCr[q0 + 1], d.vPn[q0 + 1]), lo);
    }else{
        V r0 = _mm512_add_epi64(redc1(hi, d.vC[q0], d.vJ[q0], d.vP[q0], M), lo); r0 = shp(r0, d.vP[q0]); o0 = shp(r0, d.vP[q0]);
        V r1 = _mm512_add_epi64(redc1(hi, d.vC[q0 + 1], d.vJ[q0 + 1], d.vP[q0 + 1], M), lo); r1 = shp(r1, d.vP[q0 + 1]); o1 = shp(r1, d.vP[q0 + 1]);
    }
}
/* all primes of natural vector v straight from the operand (setup/reference use; the
 * caller guarantees 16 readable limbs past the vector's first limb) */
__attribute__((always_inline)) inline
void dec1(const Dec &d, size_t v, V out[NP]){
    const uint64_t *base = d.ab + dec_limb(d, v);
    const int var = dec_var(d, v);
    for(int q = 0; q < NP; ++q) out[q] = dec1q(d, base, q, var);
}

/* ---- leaf-L lane picks ------------------------------------------------ */
/* group vector s from the L decoded natural vectors nv[0..L) */
template<int L>
struct Pick {
    V idx[L][4];
    __mmask8 msk[L][4];
    int npair;
    Pick(){
        npair = (L + 1) / 2;
        for(int s = 0; s < L; ++s)
            for(int k = 0; k < 4; ++k){
                alignas(64) uint64_t ix[8]; __mmask8 m = 0;
                for(int t = 0; t < 8; ++t){
                    const int e = t * L + s, src = e / 8, lane = e % 8;
                    ix[t] = 0;
                    if(src == 2 * k){ ix[t] = (uint64_t)lane; m |= (__mmask8)(1 << t); }
                    else if(src == 2 * k + 1){ ix[t] = (uint64_t)(8 + lane); m |= (__mmask8)(1 << t); }
                }
                idx[s][k] = _mm512_load_si512(ix); msk[s][k] = m;
            }
    }
    __attribute__((always_inline)) inline V get(const V *nv, int s) const {
        V out = _mm512_setzero_si512();
        for(int k = 0; k < npair; ++k){
            if(!msk[s][k]) continue;
            const V b = (2 * k + 1 < L) ? nv[2 * k + 1] : _mm512_setzero_si512();
            out = _mm512_or_si512(out, _mm512_maskz_permutex2var_epi64(msk[s][k], nv[2 * k], idx[s][k], b));
        }
        return out;
    }
};
/* inverse pick: natural vector j (of a group) from the L group vectors gv[s]:
 * lane l = trunk 8j + l = (t = (8j+l)/L, s = (8j+l)%L) → gv[s] lane t */
template<int L>
struct Unpick {
    V idx[L][4];
    __mmask8 msk[L][4];
    int npair;
    Unpick(){
        npair = (L + 1) / 2;
        for(int j = 0; j < L; ++j)
            for(int k = 0; k < 4; ++k){
                alignas(64) uint64_t ix[8]; __mmask8 m = 0;
                for(int l = 0; l < 8; ++l){
                    const int e = 8 * j + l, t = e / L, s = e % L;
                    ix[l] = 0;
                    if(s == 2 * k){ ix[l] = (uint64_t)t; m |= (__mmask8)(1 << l); }
                    else if(s == 2 * k + 1){ ix[l] = (uint64_t)(8 + t); m |= (__mmask8)(1 << l); }
                }
                idx[j][k] = _mm512_load_si512(ix); msk[j][k] = m;
            }
    }
    __attribute__((always_inline)) inline V get(const V *gv, int j) const {
        V out = _mm512_setzero_si512();
        for(int k = 0; k < npair; ++k){
            if(!msk[j][k]) continue;
            const V b = (2 * k + 1 < L) ? gv[2 * k + 1] : _mm512_setzero_si512();
            out = _mm512_or_si512(out, _mm512_maskz_permutex2var_epi64(msk[j][k], gv[2 * k], idx[j][k], b));
        }
        return out;
    }
};

/* ---- packed 48-bit planes ------------------------------------------------
 * plane slot: SLOT = 48 (the engine: 8 canonical residues < 2^48 in 48 bytes; the load is one
 * zeroing byte permute, the store folds [0,4p) → [0,p) (4 ops) and packs with one byte permute)
 * or 64 (-DCR_SLOT=64: one residue per 64-bit lane, one vector per cache line, +33 % bytes,
 * line-exact corner turns — NOTES §8). The 52-bit packing of the p50 configuration was removed
 * (§31); the struct keeps its name. */
struct Pk52 {
    V g48, s48;                 /* unpack / pack byte permutes */
    __mmask64 m48, st48;
};
inline Pk52 pk52_mk(void){
    alignas(64) uint8_t g48[64], s48[64];
    memset(s48, 0, 64);
    for(int l = 0; l < 8; ++l)
        for(int b = 0; b < 8; ++b){
            g48[8 * l + b] = (uint8_t)(b < 6 ? 6 * l + b : 0);       /* unpack: lane l bytes 0..5 ← packed bytes 6l.. */
            if(b < 6) s48[6 * l + b] = (uint8_t)(8 * l + b);         /* pack: packed byte 6l+b ← lane l byte b */
        }
    Pk52 K;
    K.g48 = _mm512_load_si512(g48); K.s48 = _mm512_load_si512(s48);
    K.m48 = 0x3F3F3F3F3F3F3F3Full; K.st48 = 0x0000FFFFFFFFFFFFull;
    return K;
}
static_assert(SLOT == 48 || SLOT == 64, "slot size");
/* a packed slot's bytes: one unmasked 64-B load (reads up to 16 B past the slot: every plane and
 * image carries ≥ 64 B of slack). The masked forms cost ~40 % on the latency-bound gathers (§25). */
template<int S>
__attribute__((always_inline)) inline
V ldraw(const uint8_t *src, const Pk52 &K){ (void)K; return _mm512_loadu_si512(src); }
template<int S> __attribute__((always_inline)) inline
V ldslot(const uint8_t *src, const Pk52 &K){
    if constexpr(S == 64) return _mm512_loadu_si512(src);
    else { V raw = ldraw<48>(src, K); return _mm512_maskz_permutexvar_epi8(K.m48, K.g48, raw); }
}
template<int S> __attribute__((always_inline)) inline
void stslot(uint8_t *dst, V x, const Pk52 &K, const PrimeV &pv){
    if constexpr(S == 64){ _mm512_storeu_si512(dst, x); return; }
    else{
        const V c = shp(sh2(x, pv.p2), pv.p);                       /* [0,4p) → [0,p) < 2^48 */
        _mm512_mask_storeu_epi8(dst, K.st48, _mm512_permutexvar_epi8(K.s48, c));
    }
}
__attribute__((always_inline)) inline V ld52(const uint8_t *src, const Pk52 &K){ return ldslot<SLOT>(src, K); }
__attribute__((always_inline)) inline void st52(uint8_t *dst, V x, const Pk52 &K, const PrimeV &pv){ stslot<SLOT>(dst, x, K, pv); }
/* slot-size-parametrized forms (the product planes may use SLOTO != SLOT) */
static_assert(SLOTO == 48 || SLOTO == 64, "product slot size");
/* store n bytes (multiple of 16; src 64-B aligned, readable to a 64-B
 * multiple) at dst (16-B aligned): the whole lines NT-capable, the head /
 * tail partial lines by masked stores (they complete a line the same
 * thread's neighbouring piece also writes: one RFO per line) */
__attribute__((always_inline)) inline
void st_lines(uint8_t *dst, const uint8_t *src, size_t n, int nts){
    size_t o = 0;
    const size_t head = (64 - ((uintptr_t)dst & 63)) & 63;
    if(head){
        const size_t h = head < n ? head : n;
        _mm512_mask_storeu_epi8(dst, (__mmask64)((1ull << h) - 1), _mm512_loadu_si512(src));
        o = h;
    }
    for(; o + 64 <= n; o += 64){
        const V v = _mm512_loadu_si512(src + o);
        if(nts) _mm512_stream_si512((__m512i *)(dst + o), v);
        else _mm512_storeu_si512(dst + o, v);
    }
    if(o < n) _mm512_mask_storeu_epi8(dst + o, (__mmask64)((1ull << (n - o)) - 1), _mm512_loadu_si512(src + o));
}
/* ---- NT line assembler ----------------------------------------------------
 * Adjacent pieces of one block arrive in address order (the rows pass:
 * virtual row vr's piece at blk·bstride + vr·TB·SLOT, the next row's piece
 * right behind it). Whole lines are streamed as they complete; the
 * trailing partial line waits in img[blk] (64-B image + address + fill)
 * until the next piece completes it — no RFO, no write-back. flush()
 * writes what is still pending with masked stores (task boundaries only). */
struct LineAsm { uint8_t *img; uint8_t **addr; uint8_t *fill; size_t n; };
inline void lasm_init(LineAsm &la, scratch *ws, size_t nblk){
    la.n = nblk;
    la.img = SALLOC(ws, uint8_t, 64 * nblk + 64);
    la.img = (uint8_t *)(((uintptr_t)la.img + 63) & ~(uintptr_t)63);
    la.addr = SALLOC(ws, uint8_t *, nblk);
    la.fill = SALLOC(ws, uint8_t, nblk + 64);
    memset(la.addr, 0, sizeof(uint8_t *) * nblk);
    memset(la.fill, 0, nblk);
}
__attribute__((always_inline)) inline
void line_out(uint8_t *dst, const uint8_t *src, int nts){
    const V v = _mm512_loadu_si512(src);
    if(nts) _mm512_stream_si512((__m512i *)dst, v); else _mm512_storeu_si512(dst, v);
}
/* Piece of n bytes at dst. Source must be readable through n+64 bytes;
 * masked stores do not suppress the preceding full-vector source loads. */
__attribute__((always_inline)) inline
void lasm_put(LineAsm &la, size_t blk, uint8_t *dst, const uint8_t *src, size_t n, int nts){
    uint8_t *img = la.img + 64 * blk;
    uint8_t *pa = la.addr[blk];
    size_t pf = la.fill[blk], o = 0;
    if(pa && dst == pa + pf){                                   /* continues the pending line */
        const size_t room = 64 - pf, h = n < room ? n : room;
        _mm512_mask_storeu_epi8(img + pf, (__mmask64)((1ull << h) - 1), _mm512_loadu_si512(src));
        pf += h; o = h;
        if(pf < 64){ la.fill[blk] = (uint8_t)pf; return; }
        line_out(pa, img, nts);
        pa = NULL; pf = 0;
    }else{
        if(pa){                                                 /* out-of-order arrival: write what is pending */
            _mm512_mask_storeu_epi8(pa, (__mmask64)((1ull << pf) - 1), _mm512_load_si512(img));
            pa = NULL; pf = 0;
        }
        const size_t head = (64 - ((uintptr_t)dst & 63)) & 63;
        if(head){                                               /* task start inside a line: one RFO */
            const size_t h = head < n ? head : n;
            _mm512_mask_storeu_epi8(dst, (__mmask64)((1ull << h) - 1), _mm512_loadu_si512(src));
            o = h;
        }
    }
    for(; o + 64 <= n; o += 64) line_out(dst + o, src + o, nts);
    if(o < n){
        pa = dst + o; pf = n - o;
        _mm512_mask_storeu_epi8(img, (__mmask64)((1ull << pf) - 1), _mm512_loadu_si512(src + o));
    }
    la.addr[blk] = pa; la.fill[blk] = (uint8_t)pf;
}
inline void lasm_flush(LineAsm &la){
    for(size_t blk = 0; blk < la.n; ++blk){
        if(!la.addr[blk]) continue;
        const size_t pf = la.fill[blk];
        _mm512_mask_storeu_epi8(la.addr[blk], (__mmask64)((1ull << pf) - 1), _mm512_load_si512(la.img + 64 * blk));
        la.addr[blk] = NULL; la.fill[blk] = 0;
    }
}
/* a transformed row (lbw slots, lbw multiple of TB) → its TB-slot pieces
 * at virtual row vr of every block, through the line assembler */
inline void st52blk(LineAsm &la, uint8_t *plane, const Plan &pl, size_t vr, const V *row, size_t rstr, int nts, const Pk52 &K, const PrimeV &pv){
    alignas(64) uint8_t buf[TB * SLOT + 64];
    for(size_t blk = 0; blk < pl.nblk; ++blk){
        for(size_t sl = 0; sl < TB; ++sl) st52(buf + sl * SLOT, row[(blk * TB + sl) * rstr], K, pv);
        lasm_put(la, blk, plane + blk * pl.bstride + vr * (TB * SLOT), buf, TB * SLOT, nts);
    }
}

} // namespace sbn::v3::SBN3_P48_NS
