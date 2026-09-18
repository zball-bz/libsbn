/* Imported from labs/cr/cr_col.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_col.hpp — column-axis kernels of the Bailey tile + leaf products.
 *
 * colpass<IS,WI>(x, C, b): the C-point transform at tower node b over a
 * column buffer of C vectors (one per row). Level with block size sz
 * has nb = C/sz blocks; block tn is node b·nb + tn (twiddle tab[2·node]).
 * Same four transforms as the row kernels via (IS, WI):
 *   <false,false> CF   <true,true> CI (unnormalized, ×C)
 *   <true,false>  CF^T <false,true> CI^T
 *
 * Leaf products (the pointwise stage), fresh operand y updated in place
 * against the cached kernel a:
 *   conv8 (native, in-lane ring x^8 − w[k])         : y ← a·y
 *   conv8T (its transpose, derived in DESIGN.md §1)  : y ← M_a^T y =
 *        conv in ring x^8 − w^-1 with kernel a''=(a0, w·a7, …, w·a1)
 *   convL / convLT<L> (leaf-L SoA rings x^L − w_t per lane t)
 *   lane levels (leaf-L): the 3 in-vector tree levels rooted at node
 *   v' (8M-leaf tower), lane_lv<IS,WI>. */
#include "rows.hpp"

namespace sbn::v3::SBN3_P48_NS {

/* ---- small-table column pass (NOTES §10) --------------------------------
 * Node (level nb, tn) of the column at slot b has twiddle w[2(b·nb + tn)] =
 * w[2b·nb] · w[2tn] (bit-disjoint indices), leaf k = C·b + a has w[C·b]·w[a].
 * So a column needs only the SHARED tower prefix (w[0..2C)) plus one factor
 * per level; the per-column rec-only tables are materialized on the fly:
 *   T[nb + tn] (nb = 1..C/2)   and   TL[a] (a < C)
 * with one b52 + one exact rec derivation per 8 nodes (~0.9 IFMA/node). */
template<bool WI>
inline void col_tw_build(uint64_t *T, uint64_t *TL, const Prime &P, size_t lgC, size_t C, size_t b, const PrimeV &pv_){
    const PrimeV pv = pv_regs(pv_);
    const V one = vset(1);
    const uint64_t *F = P.F[WI]; const size_t M2f = P.M2f;
    size_t k = 1;
    for(size_t nb = 1; nb < C; nb <<= 1, ++k){
        const vcc f = vcc_ofr(F[k * M2f + b], pv.p);             /* level factor w[2b·nb] = w[2^k·b] */
        for(size_t t0 = 0; t0 < nb; t0 += 8){
            const V rs = tower_even8<WI>(P, t0 / 8);      /* shared twiddles w[2(t0 + i)]; the tower always holds its first 16 nodes */
            const V cs = _mm512_madd52hi_epu64(one, rs, pv.p);      /* c of the shared twiddle */
            const V cp = shp(b52(cs, f.c, f.rec, pv.pn), pv.p);     /* product, canonical */
            const V rp = vrec_of(cp, pv);
            if(nb >= 8) _mm512_storeu_si512(T + nb + t0, rp);
            else _mm512_mask_storeu_epi64(T + nb + t0, (__mmask8)((1u << nb) - 1), rp);
        }
    }
    const vcc fl = vcc_ofr(F[lgC * M2f + b], pv.p);                  /* leaf factor w[C·b] = w[2^lgC·b] */
    for(size_t a0 = 0; a0 < C; a0 += 8){
        const V rs = tower8<WI>(P, a0);                                 /* shared leaf roots w[a0 + i] */
        const V cs = _mm512_madd52hi_epu64(one, rs, pv.p);
        const V cp = shp(b52(cs, fl.c, fl.rec, pv.pn), pv.p);
        _mm512_storeu_si512(TL + a0, vrec_of(cp, pv));
    }
}
/* colpass_h: the column transform on the materialized table T (heap
 * indexed, T[nb + tn]) through the radix-8 DFS row kernel `full` entered at
 * the heap root (3 levels per load/store instead of 1): IS=false forward
 * (BFR top-down), IS=true unnormalized inverse (IBFR bottom-up). */
template<bool IS>
inline void colpass_h(uint64_t *x, size_t C, const uint64_t *T, const Prime &P, const PrimeV &pv, PfCur *pf){
    Prime Pc = P; Pc.heap = T;
    full<IS, false, 1, true>(x, C, 1, Pc, pv, pf);
}
template<bool IS>
void colpass_t2(uint64_t *x, size_t C, const uint64_t *T, const PrimeV &pv_);
/* codec-fused column pass on the materialized table T (tile v5): PIN = the
 * forward reads its column packed from pin (piece stride pstr), POUT = the
 * inverse writes it packed to pout; leaf(o, n) runs before every inverse
 * leaf block (the products) */
template<bool IS, int PIN, int POUT, class Leaf>
inline void colpass_x(uint64_t *x, size_t C, const uint64_t *T, const Prime &P, const PrimeV &pv, PfCur *pf,
                      const uint8_t *pin, uint8_t *pout, size_t pstr, const Pk52 &K, const Leaf &leaf){
    Prime Pc = P; Pc.heap = T;
    full_x<IS, false, PIN, POUT, Leaf>(x, C, 1, Pc, pv, pf, pin, pout, pstr, K, leaf);
}
template<bool IS>
inline void colpass_t(uint64_t *x, size_t C, const uint64_t *T, const PrimeV &pv, const Prime *P = NULL, PfCur *pf = NULL){
    if(P) colpass_h<IS>(x, C, T, *P, pv, pf);
    else colpass_t2<IS>(x, C, T, pv);
}
template<bool IS>
void colpass_t2(uint64_t *x, size_t C, const uint64_t *T, const PrimeV &pv_){
    const PrimeV pv = pv_regs(pv_);
    if constexpr(!IS){
        size_t nb = 1;
        for(size_t sz = C; sz >= 2; sz >>= 1, nb <<= 1){
            const size_t half = sz >> 1;
            for(size_t tn = 0; tn < nb; ++tn){
                const vcc w = vcc_ofr(T[nb + tn], pv.p);
                uint64_t *lo = x + tn * sz * 8, *hi = lo + half * 8;
                for(size_t i = 0; i < half; ++i){
                    V a = _mm512_load_si512(lo + i * 8), c = _mm512_load_si512(hi + i * 8);
                    bf<false>(a, c, w.c, w.rec, pv);
                    _mm512_store_si512(lo + i * 8, a);
                    _mm512_store_si512(hi + i * 8, c);
                }
            }
        }
    }else{
        size_t nb = C >> 1;
        for(size_t sz = 2; sz <= C; sz <<= 1, nb >>= 1){
            const size_t half = sz >> 1;
            for(size_t tn = 0; tn < nb; ++tn){
                const vcc w = vcc_ofr(T[nb + tn], pv.p);
                uint64_t *lo = x + tn * sz * 8, *hi = lo + half * 8;
                for(size_t i = 0; i < half; ++i){
                    V a = _mm512_load_si512(lo + i * 8), c = _mm512_load_si512(hi + i * 8);
                    bf<true>(a, c, w.c, w.rec, pv);
                    _mm512_store_si512(lo + i * 8, a);
                    _mm512_store_si512(hi + i * 8, c);
                }
            }
        }
    }
}

template<bool IS, bool WI>
void colpass(uint64_t *x, size_t C, size_t b, const Prime &P, const PrimeV &pv_){
    const PrimeV pv = pv_regs(pv_);
    if constexpr(!IS){
        size_t nb = 1;
        for(size_t sz = C; sz >= 2; sz >>= 1, nb <<= 1){
            const size_t half = sz >> 1;
            for(size_t tn = 0; tn < nb; ++tn){
                const vcc w = vcc_ofr(twr<WI>(P, b * nb + tn), pv.p);
                uint64_t *lo = x + tn * sz * 8, *hi = lo + half * 8;
                for(size_t i = 0; i < half; ++i){
                    V a = _mm512_load_si512(lo + i * 8), c = _mm512_load_si512(hi + i * 8);
                    bf<false>(a, c, w.c, w.rec, pv);
                    _mm512_store_si512(lo + i * 8, a);
                    _mm512_store_si512(hi + i * 8, c);
                }
            }
        }
    }else{
        size_t nb = C >> 1;
        for(size_t sz = 2; sz <= C; sz <<= 1, nb >>= 1){
            const size_t half = sz >> 1;
            for(size_t tn = 0; tn < nb; ++tn){
                const vcc w = vcc_ofr(twr<WI>(P, b * nb + tn), pv.p);
                uint64_t *lo = x + tn * sz * 8, *hi = lo + half * 8;
                for(size_t i = 0; i < half; ++i){
                    V a = _mm512_load_si512(lo + i * 8), c = _mm512_load_si512(hi + i * 8);
                    bf<true>(a, c, w.c, w.rec, pv);
                    _mm512_store_si512(lo + i * 8, a);
                    _mm512_store_si512(hi + i * 8, c);
                }
            }
        }
    }
}

/* ---- native leaf: ring x^8 − w, in-lane schoolbook ------------------- */
/* core: out = kernel(a, aw) ⊛ y, column accumulate < 8·p·2p, one REDC */
/* the 8 column products accumulate in 4 independent (lo, hi) pairs
 * (chains of 2 IFMAs instead of 8: the single-chain form ran at ~26
 * cycles/vector, latency-bound; CR_CONV8_SPLIT=0 restores it). The
 * partial sums are exact, so the result is bit-identical. */
__attribute__((always_inline)) inline
V conv8_core(V a, V aw, V y2p, const PrimeV &pv){
    const V z = _mm512_setzero_si512();
#define CR_CSTEP(i, lo, hi) do{                                          \
        V b_ = _mm512_permutexvar_epi64(_mm512_set1_epi64(i), y2p);      \
        V w_ = (i) ? _mm512_alignr_epi64(a, aw, 8 - (i)) : a;            \
        lo = _mm512_madd52lo_epu64(lo, w_, b_);                          \
        hi = _mm512_madd52hi_epu64(hi, w_, b_);                          \
    }while(0)
#if CR_CONV8_SPLIT
    V lo0 = z, hi0 = z, lo1 = z, hi1 = z, lo2 = z, hi2 = z, lo3 = z, hi3 = z;
    CR_CSTEP(0, lo0, hi0); CR_CSTEP(1, lo1, hi1); CR_CSTEP(2, lo2, hi2); CR_CSTEP(3, lo3, hi3);
    CR_CSTEP(4, lo0, hi0); CR_CSTEP(5, lo1, hi1); CR_CSTEP(6, lo2, hi2); CR_CSTEP(7, lo3, hi3);
    const V lo = _mm512_add_epi64(_mm512_add_epi64(lo0, lo1), _mm512_add_epi64(lo2, lo3));
    const V hi = _mm512_add_epi64(_mm512_add_epi64(hi0, hi1), _mm512_add_epi64(hi2, hi3));
#else
    V lo = z, hi = z;
    CR_CSTEP(0, lo, hi); CR_CSTEP(1, lo, hi); CR_CSTEP(2, lo, hi); CR_CSTEP(3, lo, hi);
    CR_CSTEP(4, lo, hi); CR_CSTEP(5, lo, hi); CR_CSTEP(6, lo, hi); CR_CSTEP(7, lo, hi);
#endif
#undef CR_CSTEP
    return sh2(redc(lo, hi, pv.J, pv.p, pv.M), pv.p2);   /* < 2.86p → fold */
}
/* y ← a·y mod (x^8 − w[k]); a lazy [0,4p) cached, y lazy [0,4p) */
inline void conv8(V *y, const V *a, cc wk, const PrimeV &pv){
    V ac = shp(sh2(*a, pv.p2), pv.p);                    /* [0,p)     */
    V aw = shp(b52(ac, vset(wk.c), vset(wk.rec), pv.pn), pv.p);
    *y = conv8_core(ac, aw, sh2(*y, pv.p2), pv);
}
/* y ← M_a^T y : ring x^8 − winv[k] with a'' = (a0, w·a7, …, w·a1) */
inline void conv8T(V *y, const V *a, cc wk, cc wik, const PrimeV &pv){
    const V RG = _mm512_setr_epi64(0, 7, 6, 5, 4, 3, 2, 1);
    V ar = _mm512_permutexvar_epi64(RG, shp(sh2(*a, pv.p2), pv.p));
    V arw = shp(b52(ar, vset(wk.c), vset(wk.rec), pv.pn), pv.p);
    V arwi = shp(b52(ar, vset(wik.c), vset(wik.rec), pv.pn), pv.p);
    V a2 = _mm512_mask_mov_epi64(arw, 0x01, ar);         /* a''       */
    V aw2 = _mm512_mask_mov_epi64(ar, 0x01, arwi);       /* a''·winv  */
    *y = conv8_core(a2, aw2, sh2(*y, pv.p2), pv);
}

/* ---- leaf-L: lane levels ------------------------------------------------ */
struct LaneLv { V swp; __mmask8 hi; V wc, wrec; };
/* constants of the 3 in-vector levels at node vp (8M-leaf tower);
 * table = w (WI=false) or winv (WI=true) */
template<bool WI>
inline void lane_build(LaneLv lv[3], size_t vp, const Prime &P, const PrimeV &pv){
    static const uint64_t s4[8] = {4,5,6,7,0,1,2,3};
    static const uint64_t s2[8] = {2,3,0,1,6,7,4,5};
    static const uint64_t s1[8] = {1,0,3,2,5,4,7,6};
    const TwCur tc = twcur<WI, false>(P, vp);
    alignas(64) uint64_t rec[3][8];
    for(int t = 0; t < 8; ++t){
        rec[0][t] = twk<WI, false>(tc, 1, 0);
        rec[1][t] = twk<WI, false>(tc, 2, (size_t)(t >> 2));
        rec[2][t] = twk<WI, false>(tc, 4, (size_t)(t >> 1));
    }
    lv[0].swp = _mm512_loadu_si512(s4); lv[0].hi = 0xF0;
    lv[1].swp = _mm512_loadu_si512(s2); lv[1].hi = 0xCC;
    lv[2].swp = _mm512_loadu_si512(s1); lv[2].hi = 0xAA;
    for(int l = 0; l < 3; ++l){
        lv[l].wrec = _mm512_load_si512(rec[l]);
        lv[l].wc = _mm512_madd52hi_epu64(vset(1), lv[l].wrec, pv.p);
    }
}
template<bool IS>
__attribute__((always_inline)) inline
V lane_bf(V x, const LaneLv &L, const PrimeV &pv){
    V xs = _mm512_permutexvar_epi64(L.swp, x);
    V vlo = _mm512_mask_blend_epi64(L.hi, x, xs);       /* "a" in all lanes */
    V vhi = _mm512_mask_blend_epi64(L.hi, xs, x);       /* "b" in all lanes */
    if constexpr(!IS){
        V wb = b52(vhi, L.wc, L.wrec, pv.pn);
        V s = sh2(vlo, pv.p2);
        return _mm512_add_epi64(s, _mm512_mask_sub_epi64(wb, L.hi, pv.p2, wb));
    }else{
        V a = sh2(vlo, pv.p2), b = sh2(vhi, pv.p2);
        V s = _mm512_add_epi64(a, b);
        V d = b52(_mm512_add_epi64(_mm512_sub_epi64(a, b), pv.p2), L.wc, L.wrec, pv.pn);
        return _mm512_mask_blend_epi64(L.hi, s, d);
    }
}
/* apply the 3 levels (top-down for BFR, bottom-up for IBFR) to one vector */
template<bool IS>
__attribute__((always_inline)) inline
V lane_lv(V x, const LaneLv lv[3], const PrimeV &pv){
    if constexpr(!IS){ x = lane_bf<IS>(x, lv[0], pv); x = lane_bf<IS>(x, lv[1], pv); x = lane_bf<IS>(x, lv[2], pv); }
    else             { x = lane_bf<IS>(x, lv[2], pv); x = lane_bf<IS>(x, lv[1], pv); x = lane_bf<IS>(x, lv[0], pv); }
    return x;
}

/* ---- leaf-L SoA ring product: L block vectors at stride str (in V) ---- */
/* y[s·str] ← a·y in ring x^L − W_t (per-lane roots Wc/Wrec) */
template<int L, bool Scaled=false, bool CanonicalA=false>
__attribute__((always_inline)) inline
void convL(V *y, size_t ystr, const V *a, size_t astr, V Wc, V Wrec, const PrimeV &pv_, V kc={}, V kr={}){
    const PrimeV pv = pv_regs(pv_);
    V av[8], aw[8], bs[8];
    for(int j = 0; j < L; ++j){
        if constexpr(Scaled) av[j] = shp(b52(a[j * astr], kc, kr, pv.pn), pv.p);
        else if constexpr(CanonicalA){P48_CHK(a[j*astr],pv.p,"packed A canonical");av[j]=a[j*astr];}
        else av[j] = shp(sh2(a[j * astr], pv.p2), pv.p);
        bs[j] = sh2(y[j * ystr], pv.p2);
    }
    for(int j = 1; j < L; ++j) aw[j] = shp(b52(av[j], Wc, Wrec, pv.pn), pv.p);
    for(int k = 0; k < L; ++k){
        V lo = _mm512_setzero_si512(), hi = lo;
        for(int i = 0; i < L; ++i){
            V t = (k >= i) ? av[k - i] : aw[k - i + L];
            lo = _mm512_madd52lo_epu64(lo, t, bs[i]);
            hi = _mm512_madd52hi_epu64(hi, t, bs[i]);
        }
        y[k * ystr] = sh2(redc(lo, hi, pv.J, pv.p, pv.M), pv.p2);
    }
}
/* transposed: ring x^L − W^-1 with a''_0 = a_0, a''_j = W·a_{L−j};
 * wrapped kernel aw''_0 = W^-1·a_0, aw''_j = a_{L−j} */
template<int L, bool Scaled=false>
__attribute__((always_inline)) inline
void convLT(V *y, size_t ystr, const V *a, size_t astr, V Wc, V Wrec, V Wic, V Wirec, const PrimeV &pv_, V kc={}, V kr={}){
    const PrimeV pv = pv_regs(pv_);
    V av[8], aw[8], bs[8], ac[8];
    for(int j = 0; j < L; ++j){
        if constexpr(Scaled) ac[j] = shp(b52(a[j * astr], kc, kr, pv.pn), pv.p);
        else ac[j] = shp(sh2(a[j * astr], pv.p2), pv.p);
        bs[j] = sh2(y[j * ystr], pv.p2);
    }
    av[0] = ac[0];
    aw[0] = shp(b52(ac[0], Wic, Wirec, pv.pn), pv.p);
    for(int j = 1; j < L; ++j){
        av[j] = shp(b52(ac[L - j], Wc, Wrec, pv.pn), pv.p);
        aw[j] = ac[L - j];
    }
    for(int k = 0; k < L; ++k){
        V lo = _mm512_setzero_si512(), hi = lo;
        for(int i = 0; i < L; ++i){
            V t = (k >= i) ? av[k - i] : aw[k - i + L];
            lo = _mm512_madd52lo_epu64(lo, t, bs[i]);
            hi = _mm512_madd52hi_epu64(hi, t, bs[i]);
        }
        y[k * ystr] = sh2(redc(lo, hi, pv.J, pv.p, pv.M), pv.p2);
    }
}


/* ---- native leaf, transposed 8×8 form ----------------------------------
 * 8 consecutive columns a0..a0+7 (native L = 8): transpose the 8 column
 * vectors (vector t = lane t of the 8 columns), run the ring product over
 * the lane index as convL<8> with the 8 columns' roots as a lane vector
 * (no in-lane shuffles: the register form spends 15 shuffles per vector
 * and runs shuffle-bound at ~17 TSC/vector), transpose back. Same exact
 * integer sums → bit-identical to conv8. rec8: the 8 rec-only roots
 * w[C·b + a0 + k]; MID: rec8 (w) and irec8 (w^-1). */
template<bool Scaled=false, bool CanonicalA=false>
__attribute__((always_inline)) inline
void conv8x8(V *y, const V *a, V wr, const PrimeV &pv, V kc={}, V kr={}){
    V ya[8], aa[8];
    for(int k = 0; k < 8; ++k){ ya[k] = y[k]; aa[k] = a[k]; }
    tr8(ya); tr8(aa);
    const V wc = _mm512_madd52hi_epu64(vset(1), wr, pv.p);
    convL<8,Scaled,CanonicalA>(ya, 1, aa, 1, wc, wr, pv, kc, kr);
    tr8(ya);
    for(int k = 0; k < 8; ++k) y[k] = ya[k];
}
template<bool Scaled=false>
__attribute__((always_inline)) inline
void conv8x8T(V *y, const V *a, V wr, V wir, const PrimeV &pv, V kc={}, V kr={}){
    V ya[8], aa[8];
    for(int k = 0; k < 8; ++k){ ya[k] = y[k]; aa[k] = a[k]; }
    tr8(ya); tr8(aa);
    const V wc = _mm512_madd52hi_epu64(vset(1), wr, pv.p), wic = _mm512_madd52hi_epu64(vset(1), wir, pv.p);
    convLT<8,Scaled>(ya, 1, aa, 1, wc, wr, wic, wir, pv, kc, kr);
    tr8(ya);
    for(int k = 0; k < 8; ++k) y[k] = ya[k];
}

/* rec8 / irec8: eight materialized rec-only roots (the TL tables of col_tw_build) */
template<bool Scaled=false, bool CanonicalA=false>
__attribute__((always_inline)) inline
void conv8x8(V *y, const V *a, const uint64_t *rec8, const PrimeV &pv, V kc={}, V kr={}){
    conv8x8<Scaled, CanonicalA>(y, a, _mm512_loadu_si512(rec8), pv, kc, kr);
}
template<bool Scaled=false>
__attribute__((always_inline)) inline
void conv8x8T(V *y, const V *a, const uint64_t *rec8, const uint64_t *irec8, const PrimeV &pv, V kc={}, V kr={}){
    conv8x8T<Scaled>(y, a, _mm512_loadu_si512(rec8), _mm512_loadu_si512(irec8), pv, kc, kr);
}

/* strided forms (tile K2, NOTES §17): y and a read at strides ys/as (the
 * W = 2 interleaved pair F[2r] = Y[r], F[2r+1] = A[r]), result to yo[0..8) */
__attribute__((always_inline)) inline
void conv8x8_s(V *yo, const V *y, size_t ys, const V *a, size_t as, const uint64_t *rec8, const PrimeV &pv){
    V ya[8], aa[8];
    for(int k = 0; k < 8; ++k){ ya[k] = y[k * ys]; aa[k] = a[k * as]; }
    tr8(ya); tr8(aa);
    const V wr = _mm512_loadu_si512(rec8);
    const V wc = _mm512_madd52hi_epu64(vset(1), wr, pv.p);
    convL<8>(ya, 1, aa, 1, wc, wr, pv);
    tr8(ya);
    for(int k = 0; k < 8; ++k) yo[k] = ya[k];
}
__attribute__((always_inline)) inline
void conv8x8T_s(V *yo, const V *y, size_t ys, const V *a, size_t as, const uint64_t *rec8, const uint64_t *irec8, const PrimeV &pv){
    V ya[8], aa[8];
    for(int k = 0; k < 8; ++k){ ya[k] = y[k * ys]; aa[k] = a[k * as]; }
    tr8(ya); tr8(aa);
    const V wr = _mm512_loadu_si512(rec8), wir = _mm512_loadu_si512(irec8);
    const V wc = _mm512_madd52hi_epu64(vset(1), wr, pv.p), wic = _mm512_madd52hi_epu64(vset(1), wir, pv.p);
    convLT<8>(ya, 1, aa, 1, wc, wr, wic, wir, pv);
    tr8(ya);
    for(int k = 0; k < 8; ++k) yo[k] = ya[k];
}
/* W = 2 column pass on the materialized table: two columns with the SAME
 * twiddles (Y_s and A_s of one slot) interleaved x[2r], x[2r+1] */
template<bool IS>
inline void colpass_h2(uint64_t *x, size_t C, const uint64_t *T, const Prime &P, const PrimeV &pv, PfCur *pf){
    Prime Pc = P; Pc.heap = T;
    full<IS, false, 2, true>(x, C, 1, Pc, pv, pf);
}

/* ---- batched lane levels: 8 consecutive columns a0..a0+7 (same s) -----
 * tr8 turns the lane axis into the vector axis: y[t] = lane t of the 8
 * columns (lane k ↔ column a0+k). The 3 levels are then plain vector
 * butterflies with PER-LANE twiddle vectors (node vp = C·b + a0 + k):
 *   level 1: (t, t+4)  w[2vp]            level 2: (t, t+2) w[4vp + 2(t>>2)]
 *   level 3: (t, t+1)  w[8vp + 2(t>>1)]
 * (the same BFR/IBFR forms as the row kernels; transposes = IS flip). */
struct LaneT { V c1, r1, c2[2], r2[2], c3[4], r3[4]; };
template<bool WI>
inline void lane8_build(LaneT &lt, size_t vp0, const Prime &P, const PrimeV &pv){
    TwCur tc[8];
    for(int k = 0; k < 8; ++k) tc[k] = twcur<WI, false>(P, vp0 + (size_t)k);
    alignas(64) uint64_t r[8];
    for(int k = 0; k < 8; ++k) r[k] = twk<WI, false>(tc[k], 1, 0);
    lt.r1 = _mm512_load_si512(r); lt.c1 = _mm512_madd52hi_epu64(vset(1), lt.r1, pv.p);
    for(int h = 0; h < 2; ++h){
        for(int k = 0; k < 8; ++k) r[k] = twk<WI, false>(tc[k], 2, (size_t)h);
        lt.r2[h] = _mm512_load_si512(r); lt.c2[h] = _mm512_madd52hi_epu64(vset(1), lt.r2[h], pv.p);
    }
    for(int h = 0; h < 4; ++h){
        for(int k = 0; k < 8; ++k) r[k] = twk<WI, false>(tc[k], 4, (size_t)h);
        lt.r3[h] = _mm512_load_si512(r); lt.c3[h] = _mm512_madd52hi_epu64(vset(1), lt.r3[h], pv.p);
    }
}
template<bool IS>
__attribute__((always_inline)) inline
void lane8(V x[8], const LaneT &lt, const PrimeV &pv_){
    const PrimeV pv = pv_regs(pv_);
    tr8(x);
#define CR_LN_L1 do{ for(int t = 0; t < 4; ++t) bf<IS>(x[t], x[t + 4], lt.c1, lt.r1, pv); }while(0)
#define CR_LN_L2 do{ for(int h = 0; h < 2; ++h) for(int t = 0; t < 2; ++t) bf<IS>(x[4*h + t], x[4*h + t + 2], lt.c2[h], lt.r2[h], pv); }while(0)
#define CR_LN_L3 do{ for(int h = 0; h < 4; ++h) bf<IS>(x[2*h], x[2*h + 1], lt.c3[h], lt.r3[h], pv); }while(0)
    if constexpr(!IS){ CR_LN_L1; CR_LN_L2; CR_LN_L3; } else { CR_LN_L3; CR_LN_L2; CR_LN_L1; }
#undef CR_LN_L1
#undef CR_LN_L2
#undef CR_LN_L3
    tr8(x);
}


/* split forms: lane8_in = tr8 + levels (output stays TRANSPOSED: vector t
 * = lane-slot t of the 8 columns); lane8_out = levels + tr8 (input
 * transposed). roots8: the leaf roots for (8 columns a0.., slot t) as 8
 * vectors r[t] (lane k = w[8(vp0+k) + t]) via 8 loads + one transpose. */
template<bool IS>
__attribute__((always_inline)) inline
void lane8_in(V x[8], const LaneT &lt, const PrimeV &pv_){
    const PrimeV pv = pv_regs(pv_);
    tr8(x);
#define CR_LN_L1 do{ for(int t = 0; t < 4; ++t) bf<IS>(x[t], x[t + 4], lt.c1, lt.r1, pv); }while(0)
#define CR_LN_L2 do{ for(int h = 0; h < 2; ++h) for(int t = 0; t < 2; ++t) bf<IS>(x[4*h + t], x[4*h + t + 2], lt.c2[h], lt.r2[h], pv); }while(0)
#define CR_LN_L3 do{ for(int h = 0; h < 4; ++h) bf<IS>(x[2*h], x[2*h + 1], lt.c3[h], lt.r3[h], pv); }while(0)
    if constexpr(!IS){ CR_LN_L1; CR_LN_L2; CR_LN_L3; } else { CR_LN_L3; CR_LN_L2; CR_LN_L1; }
#undef CR_LN_L1
#undef CR_LN_L2
#undef CR_LN_L3
}
template<bool IS>
__attribute__((always_inline)) inline
void lane8_out(V x[8], const LaneT &lt, const PrimeV &pv_){
    const PrimeV pv = pv_regs(pv_);
#define CR_LN_L1 do{ for(int t = 0; t < 4; ++t) bf<IS>(x[t], x[t + 4], lt.c1, lt.r1, pv); }while(0)
#define CR_LN_L2 do{ for(int h = 0; h < 2; ++h) for(int t = 0; t < 2; ++t) bf<IS>(x[4*h + t], x[4*h + t + 2], lt.c2[h], lt.r2[h], pv); }while(0)
#define CR_LN_L3 do{ for(int h = 0; h < 4; ++h) bf<IS>(x[2*h], x[2*h + 1], lt.c3[h], lt.r3[h], pv); }while(0)
    if constexpr(!IS){ CR_LN_L1; CR_LN_L2; CR_LN_L3; } else { CR_LN_L3; CR_LN_L2; CR_LN_L1; }
#undef CR_LN_L1
#undef CR_LN_L2
#undef CR_LN_L3
    tr8(x);
}
template<bool WI>
__attribute__((always_inline)) inline
void roots8(V rec[8], V c[8], size_t vp0, const Prime &P, const PrimeV &pv){
    for(int k = 0; k < 8; ++k) rec[k] = tower8<WI>(P, 8 * (vp0 + (size_t)k));   /* row k = column k */
    tr8(rec);                                                              /* now rec[t] lane k */
    for(int t = 0; t < 8; ++t) c[t] = _mm512_madd52hi_epu64(vset(1), rec[t], pv.p);
}

} // namespace sbn::v3::SBN3_P48_NS
