/* Imported from labs/cr/cr_emit.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_emit.hpp — output endpoint: plane gather (view) → per-prime scale → Garner mixed-radix
 * digits → radix-2^52 compose → the trunk value's radix-2^T digits (b0, b1 of T bits = WD 64-bit
 * words each, b2 = the top 48·NP − 2T ≤ 64 bits) → placement at bit T·k → limbs, chunked (4096
 * trunks per task, L2-resident buffers) with a per-chunk spill journal joined serially.
 *
 * Placement: the vector radix-2^T lag chain (lag_add: this vector's b0 + the previous vector's b1
 * one lane up + its b2 two lanes up, the per-lane carries one lane up — NOTES §20 E3, generalized
 * to WD words in §32) and Regroup (T-bit digit stream → 64-bit limbs). The unfused emit_chunk path
 * serves the non-fused arms and M2 > 8192; the LIN default emits inside cr_fuse.hpp with the same
 * digits + Pack8. NP = 4 keeps its hand-written compose (emit_digits), bit-identical to the
 * general emit_digits_gen (gated in cr_test.cpp). */
#include "codec.hpp"
#include "scratch_adapter.hpp"


namespace sbn::v3::SBN3_P48_NS {

enum { OCH = CR_OCH };
/* radix-2^52 digits of a trunk value (< prod(PR) < 2^(48·NP)) plus one for the MAC's top carries;
 * words per T-bit digit (T ≤ 232) */
enum { NC = (48 * NP + 51) / 52 + 1, WDMAX = NP<=8?3:4, PD = (48 * NP + 51) / 52, QD = (48 * (NP - 1) + 51) / 52 };
constexpr int cr_ndig(int i){ return (48 * i + 51) / 52; }   /* digits of prod_{k<i} p_k (bit length 48i) */
#ifdef CR_EMIT_PROF
#include <x86intrin.h>
struct EpCnt { uint64_t gather, dig, chain, regroup, vec, total; uint64_t pad[2]; };
static EpCnt g_ep[64];
static int g_ep_skip;
__attribute__((always_inline)) inline uint64_t cr_tsc_e(void){ unsigned lo, hi; __asm__ __volatile__("lfence\n\trdtsc\n\tlfence" : "=a"(lo), "=d"(hi) :: "memory"); return ((uint64_t)hi << 32) | lo; }
#define EP_T(v) uint64_t v = cr_tsc_e()
#define EP_ADD(acc, a, b) g_ep[wid].acc += (b) - (a)
#else
#define EP_T(v)
#define EP_ADD(acc, a, b)
#endif

struct Garner {
    cc ppm[NP][NP];             /* prod_{l<k} p_l mod p_i                 */
    cc inv[NP];                 /* (prod_{l<i} p_l)^-1 mod p_i            */
    cc sc[NP];                  /* per-prime plane scale                  */
    uint64_t w1[1], w2[2], w3[3];   /* the NP = 4 compose constants (bit-exact baseline path) */
    uint64_t W[NP][NC];         /* W_i = prod_{k<i} p_k in cr_ndig(i) radix-2^52 digits (i >= 1) */
    uint64_t Wm[NP][NP];        /* (prod_{l<k} p_l mod p_i)·2^52 mod p_i: the dot-product Garner (§32) */
    int direct;
    int nosc;                   /* the plane scale is folded into A's decode (fold_scale_on): the gathered
                                 * residues are canonical and pre-scaled — no per-prime scale step */
    void init(const Primes &PS, const Plan &pl, int mode=1){
        direct=mode==2;
        nosc = fold_scale_on(pl);
        for(int i = 0; i < NP; ++i){
            const uint64_t p = PS.P[i].p;
            const uint64_t r52 = (uint64_t)(((u128)1 << 52) % p);
            uint64_t pp = 1;
            for(int k = 0; k < i; ++k){
                ppm[i][k] = cc_of(pp, p);
                Wm[i][k] = mulm(pp, r52, p);
                pp = mulm(pp, PR[k] % p, p);
            }
            inv[i] = fixed_setup.prime[i].prefix_inverse;
            sc[i] = cc_of(nosc ? 1 : mulm(plane_scale(pl,p),direct?cofactor_inverse(i):1,p), p);
        }
        w1[0] = PR[0] & M52;
        {
            u128 w = (u128)PR[0] * PR[1];
            w2[0] = (uint64_t)(w & M52); w2[1] = (uint64_t)(w >> 52);
            const uint64_t a0 = (uint64_t)(w & M52), a1 = (uint64_t)((w >> 52) & M52), a2 = (uint64_t)(w >> 104);
            u128 c0 = (u128)a0 * PR[2];
            u128 c1 = (u128)a1 * PR[2] + (uint64_t)(c0 >> 52);
            u128 c2 = (u128)a2 * PR[2] + (uint64_t)(c1 >> 52);
            w3[0] = (uint64_t)(c0 & M52); w3[1] = (uint64_t)(c1 & M52); w3[2] = (uint64_t)(c2 & M52);
        }
        /* general compose: W_i by repeated multiplication in radix 2^52 */
        memset(W, 0, sizeof W);
        for(int i = 1; i < NP; ++i){
            uint64_t dg[NC + 2]; memset(dg, 0, sizeof dg); dg[0] = 1; int n = 1;
            for(int k = 0; k < i; ++k){
                u128 cy = 0;
                for(int t = 0; t < n; ++t){ const u128 x = (u128)dg[t] * PR[k] + cy; dg[t] = (uint64_t)x & M52; cy = x >> 52; }
                while(cy){ dg[n++] = (uint64_t)cy & M52; cy >>= 52; }
            }
            if(n != cr_ndig(i)) ::sbn::v3::fatal(SBN3_FATAL_MATH, "p48 Garner digits", n, cr_ndig(i));
            for(int t = 0; t < n; ++t) W[i][t] = dg[t];
        }
    }
};

struct EmitCtx {
    const Plan *pl;
    const Primes *PS;
    const Garner *G;
    const uint8_t *plane[NP];   /* result planes (packed)                 */
    int reversed;               /* TMP: view vector s ← plane Zw−1−s, REV8 */
    size_t vlim;                /* reversed: Zw                           */
    uint64_t *rp;               /* [0, outcap)                            */
    uint64_t *tail;             /* [outcap, nl)                           */
    uint64_t (*spill)[8];       /* per-chunk seam journal                 */
    int nch;
    int nt_out;                 /* NT stores into rp (64B-aligned rp)     */
    const void *EK;             /* hoisted per-call constants (EmitK*)    */
    const void *RG;             /* hoisted regroup tables (Regroup*)      */
    const void *map;            /* task → chunk order (EmitMap*)          */
};


__attribute__((always_inline)) inline
uint64_t *emit_sink(const EmitCtx &c, size_t i){
    return i < c.pl->outcap ? c.rp + i : c.tail + (i - c.pl->outcap);
}

/* plane element address of natural vector v (native) */
__attribute__((always_inline)) inline
const uint8_t *plane_at(const Plan &pl, const uint8_t *plane, size_t v, int *ok){
    const size_t r = v & (pl.C - 1), cc = v >> pl.lgC;
    *ok = cc < pl.lbv;
    return plane + plane_off(pl, r, cc);
}

/* gather view vector s of prime q (lazy residues, zero beyond) */
template<int L>
__attribute__((always_inline)) inline
V emit_gather(const EmitCtx &c, int q, size_t s, const Pk52 &K, const Unpick<L> *up){
    const Plan &pl = *c.pl;
    size_t v;
    if(c.reversed){
        if(s >= c.vlim) return _mm512_setzero_si512();
        v = c.vlim - 1 - s;
    }else v = pl.voff / 8 + s;
    V x;
    if constexpr(L == 8){
        int ok;
        const uint8_t *e = plane_at(pl, c.plane[q], v, &ok);
        x = ok ? ld52(e, K) : _mm512_setzero_si512();
    }else{
        const size_t g = v / L, j = v % L;
        const size_t r = g & (pl.C - 1), cc = g >> pl.lgC;
        if(cc >= pl.lbv) x = _mm512_setzero_si512();
        else{
            V gv[L];
            for(int sidx = 0; sidx < L; ++sidx)
                gv[sidx] = ld52(c.plane[q] + plane_off(pl, r * L + sidx, cc), K);
            x = up->get(gv, (int)j);
        }
    }
    if(c.reversed) x = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), x);
    return x;
}

/* ---- radix-2^T digit stream: Garner → compose → (b0[WD], b1[WD], b2) ----------------------
 * b0 = bits [0, T) of the trunk value, b1 = [T, 2T), b2 = [2T, 2T + 64) ⊇ the rest (the planner
 * keeps 48·NP − 2T ≤ 64); word j of a T-digit = its bits [64j, 64j + 64), the top word HB =
 * T − 64(WD − 1) bits (HB = 64 at T = 128/192). */
template<int WD> struct DigW { V b0[WD], b1[WD], b2; };
typedef DigW<2> Dig;

struct EmitK {                      /* hoisted per-chunk constants          */
    V vp[NP], vpn[NP], vp2[NP], vsc[NP], vsr[NP];
    V vppm[NP][NP], vppr[NP][NP], vinv[NP], vinr[NP];
    V vw1, vw20, vw21, vw30, vw31, vw32, vM, LIM, HM;
    V vW[NP][NC];
    V vWm[NP][NP], vJ[NP], vp3[NP]; /* the dot-product Garner (§32) */
    int nosc, direct;
    V direct_w[NP][QD], direct_p[PD];
    __m512d direct_recip[NP];
    int T, WD, HB, gen;             /* gen: CR_EMIT_GEN=1 forces the general compose at NP = 4 (A/B, gate) */
    int xk[2 * WDMAX + 1], xs[2 * WDMAX + 1];   /* runtime-T fallback: word u = digits xk.. >> xs */
    void init(const Garner &G, const Primes &PS, int T_){
        T = T_; WD = (T + 63) / 64; HB = T - 64 * (WD - 1);
        for(int q = 0; q < NP; ++q){
            vp[q] = PS.V[q].p; vpn[q] = PS.V[q].pn; vp2[q] = PS.V[q].p2;
            vsc[q] = vset(G.sc[q].c); vsr[q] = vset(G.sc[q].rec);
            vinv[q] = vset(G.inv[q].c); vinr[q] = vset(G.inv[q].rec);
            for(int k = 0; k < NP; ++k){ vppm[q][k] = vset(G.ppm[q][k].c); vppr[q][k] = vset(G.ppm[q][k].rec); vWm[q][k] = vset(G.Wm[q][k]); }
            for(int k = 0; k < NC; ++k) vW[q][k] = vset(G.W[q][k]);
            vJ[q] = PS.V[q].J; vp3[q] = _mm512_add_epi64(PS.V[q].p2, PS.V[q].p);
        }
        nosc = G.nosc;direct=G.direct;
        if(direct){
            // Products in radix 2^52, no GMP or allocation in production setup.
            for(int q=-1;q<NP;++q){
                uint64_t a[PD+1]{};a[0]=1;
                for(int j=0;j<NP;++j)if(j!=q){
                    u128 carry=0;
                    for(int k=0;k<PD;++k){const u128 v=(u128)a[k]*PR[j]+carry;a[k]=(uint64_t)v&M52;carry=v>>52;}
                    ::sbn::v3::require(!carry,SBN3_FATAL_MATH,"direct CRT table width");
                }
                if(q<0)for(int k=0;k<PD;++k)direct_p[k]=vset(a[k]);
                else {for(int k=0;k<QD;++k)direct_w[q][k]=vset(a[k]);direct_recip[q]=_mm512_set1_pd(1.0/double(PR[q]));}
            }
        }
        vw1 = vset(G.w1[0]); vw20 = vset(G.w2[0]); vw21 = vset(G.w2[1]);
        vw30 = vset(G.w3[0]); vw31 = vset(G.w3[1]); vw32 = vset(G.w3[2]);
        vM = vset(M52);
        LIM = vset(HB < 64 ? 1ull << HB : 0);
        HM = vset(HB < 64 ? (1ull << HB) - 1 : ~0ull);
        for(int u = 0; u < 2 * WD + 1; ++u){
            const int o = u < 2 * WD ? (u / WD) * T + 64 * (u % WD) : 2 * T;
            xk[u] = o / 52; xs[u] = o % 52;
        }
        gen = 0;
    }
};

/* the NP = 4 compose (hand-scheduled MACs), any T in [68, 128): gathered lazy residues x[q] → digits */
template<int Dummy = 0>
__attribute__((always_inline)) inline
Dig emit_digits(const EmitK &K, V x[NP]){
    static_assert(NP == 4 || Dummy != 0, "emit_digits is the NP = 4 compose");
    V vd[NP];
    for(int q = 0; q < NP; ++q)
        vd[q] = shp(b52(x[q], K.vsc[q], K.vsr[q], K.vpn[q]), K.vp[q]);   /* b52 of a < 2^52 is < 2p: one fold */
    for(int i = 1; i < NP; ++i){
        V acc = vd[0];
        for(int k = 1; k < i; ++k){
            V u = b52(vd[k], K.vppm[i][k], K.vppr[i][k], K.vpn[i]);
            acc = shp(_mm512_add_epi64(acc, u), K.vp[i]);
        }
        V d = _mm512_sub_epi64(_mm512_add_epi64(vd[i], K.vp2[i]), acc);   /* (0, 3p) */
        vd[i] = shp(b52(d, K.vinv[i], K.vinr[i], K.vpn[i]), K.vp[i]);
    }
    V c0 = vd[0], c1, c2, c3;
    const V z = _mm512_setzero_si512();
    c0 = _mm512_madd52lo_epu64(c0, vd[1], K.vw1);
    c1 = _mm512_madd52hi_epu64(z, vd[1], K.vw1);
    c0 = _mm512_madd52lo_epu64(c0, vd[2], K.vw20);
    c1 = _mm512_madd52hi_epu64(c1, vd[2], K.vw20);
    c1 = _mm512_madd52lo_epu64(c1, vd[2], K.vw21);
    c2 = _mm512_madd52hi_epu64(z, vd[2], K.vw21);
    c0 = _mm512_madd52lo_epu64(c0, vd[3], K.vw30);
    c1 = _mm512_madd52hi_epu64(c1, vd[3], K.vw30);
    c1 = _mm512_madd52lo_epu64(c1, vd[3], K.vw31);
    c2 = _mm512_madd52hi_epu64(c2, vd[3], K.vw31);
    c2 = _mm512_madd52lo_epu64(c2, vd[3], K.vw32);
    c3 = _mm512_madd52hi_epu64(z, vd[3], K.vw32);
    c1 = _mm512_add_epi64(c1, _mm512_srli_epi64(c0, 52)); c0 = _mm512_and_si512(c0, K.vM);
    c2 = _mm512_add_epi64(c2, _mm512_srli_epi64(c1, 52)); c1 = _mm512_and_si512(c1, K.vM);
    c3 = _mm512_add_epi64(c3, _mm512_srli_epi64(c2, 52)); c2 = _mm512_and_si512(c2, K.vM);
    const int T = K.T;
    Dig d;
    d.b0[0] = _mm512_or_si512(c0, _mm512_slli_epi64(c1, 52));
    d.b0[1] = _mm512_and_si512(_mm512_srli_epi64(c1, 12), K.HM);
    d.b1[0] = _mm512_or_si512(_mm512_srli_epi64(c1, (unsigned)(T - 52)), _mm512_slli_epi64(c2, (unsigned)(104 - T)));
    d.b1[1] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(c2, (unsigned)(T - 40)),
                                               _mm512_slli_epi64(c3, (unsigned)(92 - T))), K.HM);
    d.b2 = _mm512_srli_epi64(c3, (unsigned)(2 * T - 156));
    return d;
}

/* general compose: scale + Garner (NP(NP−1)/2 b52) + the radix-2^52 MAC X = Σ v_i·W_i, then the
 * T-digit words; T compile-time (immediate shifts, the digit registers stay in registers) */
template<int O>
__attribute__((always_inline)) inline
V xword(const V c[NC]){
    constexpr int k = O / 52, s = O % 52;
    V w = s ? _mm512_srli_epi64(c[k], s) : c[k];
    if constexpr(k + 1 < NC) w = _mm512_or_si512(w, _mm512_slli_epi64(c[k + 1], 52 - s));
    if constexpr(k + 2 < NC && s > 40) w = _mm512_or_si512(w, _mm512_slli_epi64(c[k + 2], 104 - s));
    return w;
}
/* Garner in dot-product form (§32): v_i = (x_i − v_0 − Σ_{k=1}^{i−1} v_k·W_k)·W_i^{-1} mod p_i with the
 * sum as one IFMA lo/hi accumulation (2 IFMA per term, all terms independent) and one REDC, instead
 * of a b52 + fold per term: 147 vs 203 ops at NP = 8. Bounds: lo < 7·2^52, hi < 7·2^44 → REDC
 * < p + 2^47 + 8; acc < 2p + 2^47; d = x_i + 3p − acc ∈ (0, 7p) < 2^52 → b52 → [0, 2p) → shp. */
__attribute__((always_inline)) inline
void garner_digits(const EmitK &K, V x[NP], V vd[NP]){
    if(K.nosc){
        /* pre-scaled; the fused pass hands over the inverse rows' LAZY outputs (< 4p): v_0 must be
         * canonical (it enters the compose as is), the others are folded by their own Garner step
         * (d < 4p + 3p = 7p < 2^52) */
        vd[0] = shp(sh2(x[0], K.vp2[0]), K.vp[0]);
        for(int q = 1; q < NP; ++q) vd[q] = x[q];
    }else for(int q = 0; q < NP; ++q) vd[q] = shp(b52(x[q], K.vsc[q], K.vsr[q], K.vpn[q]), K.vp[q]);
    const V z = _mm512_setzero_si512();
    for(int i = 1; i < NP; ++i){
        V acc = vd[0];
#if CR_GARNER_DOT
        if(i >= 2){
            V lo = _mm512_madd52lo_epu64(z, vd[1], K.vWm[i][1]), hi = _mm512_madd52hi_epu64(z, vd[1], K.vWm[i][1]);
            for(int k = 2; k < i; ++k){ lo = _mm512_madd52lo_epu64(lo, vd[k], K.vWm[i][k]); hi = _mm512_madd52hi_epu64(hi, vd[k], K.vWm[i][k]); }
            acc = _mm512_add_epi64(acc, redc(lo, hi, K.vJ[i], K.vp[i], K.vM));
        }
        V d = _mm512_sub_epi64(_mm512_add_epi64(vd[i], K.vp3[i]), acc);   /* (0, 7p) */
#else
        /* CR_GARNER_DOT=0: the b52-per-term form (acc = fold(v_0 + Σ b52(v_k, W_k)): shorter chain, more ops) */
        for(int k = 1; k < i; ++k){
            V u = b52(vd[k], K.vppm[i][k], K.vppr[i][k], K.vpn[i]);
            acc = shp(_mm512_add_epi64(acc, u), K.vp[i]);
        }
        V d = _mm512_sub_epi64(_mm512_add_epi64(vd[i], K.vp2[i]), acc);   /* (0, 3p) + the lazy x_i: < 6p */
#endif
        vd[i] = shp(b52(d, K.vinv[i], K.vinr[i], K.vpn[i]), K.vp[i]);
    }
}
/* Exact direct CRT after cofactor-inverse preconditioning. x[q] < 4p[q].
 * sum(x[q]/p[q]) < 4NP <= 40; nearest-integer quotient is floor or ceil
 * (absolute double error < 2^-40). Signed radix-52 normalization leaves a
 * representative in (-P,P); add P on negative lanes. Accumulators < 2^59.
 * See docs/direct-crt-proof-2026-09-06.md for the bounds and endpoint gates. */
__attribute__((always_inline)) inline
void direct_compose(const EmitK &K, V x[NP], V c[NC]){
    const V z=_mm512_setzero_si512();
    for(int k=0;k<NC;++k)c[k]=z;
    __m512d approx=_mm512_setzero_pd();
    for(int q=0;q<NP;++q){
        const V a=K.nosc?x[q]:shp(b52(x[q],K.vsc[q],K.vsr[q],K.vpn[q]),K.vp[q]);
        approx=_mm512_fmadd_pd(_mm512_cvtepu64_pd(a),K.direct_recip[q],approx);
        for(int k=0;k<QD;++k){
            c[k]=_mm512_madd52lo_epu64(c[k],a,K.direct_w[q][k]);
            c[k+1]=_mm512_madd52hi_epu64(c[k+1],a,K.direct_w[q][k]);
        }
    }
    const V quotient=_mm512_cvt_roundpd_epu64(approx,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    for(int k=0;k<PD;++k){
        c[k]=_mm512_sub_epi64(c[k],_mm512_madd52lo_epu64(z,quotient,K.direct_p[k]));
        c[k+1]=_mm512_sub_epi64(c[k+1],_mm512_madd52hi_epu64(z,quotient,K.direct_p[k]));
    }
    for(int k=0;k+1<NC;++k){c[k+1]=_mm512_add_epi64(c[k+1],_mm512_srai_epi64(c[k],52));c[k]=_mm512_and_si512(c[k],K.vM);}
    const __mmask8 neg=_mm512_movepi64_mask(c[NC-1]);
    if(neg){
        for(int k=0;k<PD;++k)c[k]=_mm512_mask_add_epi64(c[k],neg,c[k],K.direct_p[k]);
        for(int k=0;k+1<NC;++k){c[k+1]=_mm512_add_epi64(c[k+1],_mm512_srai_epi64(c[k],52));c[k]=_mm512_and_si512(c[k],K.vM);}
    }
}
/* the radix-2^52 MAC X = Σ v_i·W_i into the digit array (a rolling three-digit window with the words cut
 * as their digits complete was tried: no faster at NP = 6, 9 % slower at NP = 8 — the spills are not the cost) */
__attribute__((always_inline)) inline
void garner_compose(const EmitK &K, V x[NP], V c[NC]){
    V vd[NP];
    garner_digits(K, x, vd);
    const V z = _mm512_setzero_si512();
    c[0] = vd[0];
    for(int k = 1; k < NC; ++k) c[k] = z;
    for(int i = 1; i < NP; ++i)
        for(int k = 0; k < cr_ndig(i); ++k){
            c[k] = _mm512_madd52lo_epu64(c[k], vd[i], K.vW[i][k]);
            c[k + 1] = _mm512_madd52hi_epu64(c[k + 1], vd[i], K.vW[i][k]);
        }
    for(int k = 0; k + 1 < NC; ++k){ c[k + 1] = _mm512_add_epi64(c[k + 1], _mm512_srli_epi64(c[k], 52)); c[k] = _mm512_and_si512(c[k], K.vM); }
}
template<int WD, int T>
__attribute__((always_inline)) inline
DigW<WD> emit_digits_gen(const EmitK &K, V x[NP]){
    static_assert((T + 63) / 64 == WD, "words per T-digit");
    static_assert(48 * NP - 2 * T <= 64 && 2 * T < 48 * NP, "b2 must fit one word");
    V c[NC];
    if(K.direct)direct_compose(K,x,c);else garner_compose(K, x, c);
    DigW<WD> d;
    d.b0[0] = xword<0>(c); d.b1[0] = xword<T>(c);
    if constexpr(WD > 1){ d.b0[1] = xword<64>(c); d.b1[1] = xword<T + 64>(c); }
    if constexpr(WD > 2){ d.b0[2] = xword<128>(c); d.b1[2] = xword<T + 128>(c); }
    if constexpr(WD>3){d.b0[3]=xword<192>(c);d.b1[3]=xword<T+192>(c);}
    d.b2 = xword<2 * T>(c);
    constexpr int HB = T - 64 * (WD - 1);
    if constexpr(HB < 64){ d.b0[WD - 1] = _mm512_and_si512(d.b0[WD - 1], K.HM); d.b1[WD - 1] = _mm512_and_si512(d.b1[WD - 1], K.HM); }
    return d;
}
/* runtime-T fallback (any T the planner admits; the digit array lives on the stack) */
template<int WD>
__attribute__((noinline))
DigW<WD> emit_digits_rt(const EmitK &K, V x[NP]){
    V c[NC];
    if(K.direct)direct_compose(K,x,c);else garner_compose(K, x, c);
    DigW<WD> d;
    auto word = [&](int u) -> V {
        const int k = K.xk[u], s = K.xs[u];
        V w = _mm512_srli_epi64(c[k], (unsigned)s);
        if(k + 1 < NC) w = _mm512_or_si512(w, _mm512_slli_epi64(c[k + 1], (unsigned)(52 - s)));
        if(k + 2 < NC && s > 40) w = _mm512_or_si512(w, _mm512_slli_epi64(c[k + 2], (unsigned)(104 - s)));
        return w;
    };
    for(int j = 0; j < WD; ++j){ d.b0[j] = word(j); d.b1[j] = word(WD + j); }
    d.b2 = word(2 * WD);
    d.b0[WD - 1] = _mm512_and_si512(d.b0[WD - 1], K.HM);
    d.b1[WD - 1] = _mm512_and_si512(d.b1[WD - 1], K.HM);
    return d;
}
/* dispatcher: the NP = 4 hand path, else the compile-time-T general path over plan_T_auto's
 * candidate list (cr_geom.hpp: NP = 4 {88, 84, 80}; NP ≥ 5 {24·NP − 8 … 24·NP − 32} step 8) */
template<int WD, int T>
__attribute__((always_inline)) inline
DigW<WD> digits_T(const EmitK &K, V x[NP]){
    if constexpr((T + 63) / 64 == WD && 48 * NP - 2 * T <= 64 && 2 * T < 48 * NP) return emit_digits_gen<WD, T>(K, x);
    else return emit_digits_rt<WD>(K, x);
}
template<int WD>
__attribute__((always_inline)) inline
DigW<WD> digits_of(const EmitK &K, V x[NP]){
    if constexpr(NP == 4 && WD == 2){ if(!K.gen && !K.direct) return emit_digits(K, x); }
    constexpr int TA = NP == 4 ? 88 : 24 * NP - 8, TS = NP == 4 ? 4 : 8;
    switch(K.T){
    case TA:          return digits_T<WD, TA>(K, x);
    case TA - TS:     return digits_T<WD, TA - TS>(K, x);
    case TA - 2 * TS: return digits_T<WD, TA - 2 * TS>(K, x);
    case TA - 3 * TS: return digits_T<WD, TA - 3 * TS>(K, x);
    default:          return emit_digits_rt<WD>(K, x);
    }
}

/* radix-2^T lag-chain step (NOTES §20 E3, WD words §32): w = this vector's b0 words; add a1 (the
 * previous vector's b1, already one lane up), a2 (its b2, two lanes up) and the carry-in cin
 * (lane 0); every lane's carry-out (0..2) is added one lane up (a second round only when a lane
 * within 2 of 2^T received a carry: rare). Returns lane 7's carry-out (0..2). The words stay
 * canonical (< 2^64, the top word < 2^HB). */
template<int WD>
__attribute__((always_inline)) inline
unsigned lag_add(V w[WD], const V a1[WD], V a2, unsigned cin, V LIM, V HM, int HB){
    const V one = vset(1), z = _mm512_setzero_si512();
    V cv = z;
    for(int j = 0; j < WD; ++j){
        V s = _mm512_add_epi64(w[j], a1[j]);
        const __mmask8 m1 = _mm512_cmplt_epu64_mask(s, a1[j]);
        const V t = j == 0 ? a2 : cv;
        s = _mm512_add_epi64(s, t);
        const __mmask8 m2 = _mm512_cmplt_epu64_mask(s, t);
        w[j] = s;
        cv = _mm512_add_epi64(_mm512_maskz_set1_epi64(m1, 1), _mm512_maskz_set1_epi64(m2, 1));
    }
    V co;
    if(HB < 64){ co = _mm512_srli_epi64(w[WD - 1], (unsigned)HB); w[WD - 1] = _mm512_and_si512(w[WD - 1], HM); }
    else co = cv;
    unsigned cout = 0;
    for(;;){
        const V ci = _mm512_alignr_epi64(co, vset(cin), 7);
        cout += (unsigned)_mm_extract_epi64(_mm512_extracti64x2_epi64(co, 3), 1);
        V s = _mm512_add_epi64(w[0], ci);
        __mmask8 m = _mm512_cmplt_epu64_mask(s, ci);
        w[0] = s;
        for(int j = 1; j < WD; ++j){ w[j] = _mm512_mask_add_epi64(w[j], m, w[j], one); m &= _mm512_cmpeq_epi64_mask(w[j], z); }
        if(HB < 64){
            const __mmask8 ov = _mm512_cmpge_epu64_mask(w[WD - 1], LIM);
            if(__builtin_expect(!ov, 1)) break;
            co = _mm512_srli_epi64(w[WD - 1], (unsigned)HB); w[WD - 1] = _mm512_and_si512(w[WD - 1], HM);
        }else{
            if(__builtin_expect(!m, 1)) break;
            co = _mm512_maskz_set1_epi64(m, 1);
        }
        cin = 0;
    }
    return cout;
}

template<bool> struct RegroupFourthWord {};
template<> struct RegroupFourthWord<true> { __mmask8 m3[48]; };
struct Regroup {                    /* T-bit digit stream (WD words) → 64-bit limbs   */
    int onv, OT, OL, T, WD;
    int obase[48];
    V iu[48], iu1[48], sh[48], lsh[48];
    [[no_unique_address]] RegroupFourthWord<(WDMAX>3)> fourth;
    __mmask8 m1[48], m2[48];        /* lane's source word k >= 1 / k >= 2 */
    void init(int T_, int OT_, int OL_, int WD_){
        T = T_; OT = OT_; OL = OL_; WD = WD_; onv = OL / 8;
        for(int ov = 0; ov < onv; ++ov){
            obase[ov] = (int)(((long long)64 * 8 * ov) / T);
            alignas(64) uint64_t a[8], b[8], c[8], e[8]; __mmask8 k1 = 0, k2 = 0;[[maybe_unused]] __mmask8 k3=0;
            for(int l = 0; l < 8; ++l){
                const long long L = 8 * ov + l, u = (64 * L) / T, s = 64 * L - (long long)T * u;
                a[l] = (uint64_t)(u - obase[ov]); b[l] = a[l] + 1;
                c[l] = (uint64_t)(s & 63); e[l] = (uint64_t)(T - s < 64 ? T - s : 64);
                if(s >= 64) k1 |= (__mmask8)(1 << l);
                if(s >= 128) k2 |= (__mmask8)(1 << l);
                if(s>=192)k3|=(__mmask8)(1<<l);
            }
            iu[ov] = _mm512_load_si512(a); iu1[ov] = _mm512_load_si512(b);
            sh[ov] = _mm512_load_si512(c); lsh[ov] = _mm512_load_si512(e);
            m1[ov] = k1; m2[ov] = k2;
#if CR_NP>8
            fourth.m3[ov]=k3;
#endif
        }
    }
    /* superblock: digit words ds[k] at trunk base tb (readable 16 past), 8·onv limbs */
    __attribute__((always_inline)) inline
    void run(uint64_t *out, const uint64_t *const ds[WDMAX], int nt) const {
        const V z = _mm512_setzero_si512();
        for(int ov = 0; ov < onv; ++ov){
            const uint64_t *l0 = ds[0] + obase[ov];
            const V Wl0 = _mm512_loadu_si512(l0), Wl1 = _mm512_loadu_si512(l0 + 8);
            const V la = _mm512_permutex2var_epi64(Wl0, iu[ov], Wl1);
            const V lb = _mm512_permutex2var_epi64(Wl0, iu1[ov], Wl1);
            V fa = la, fb = z;
            if(WD >= 2){
                const uint64_t *h0 = ds[1] + obase[ov];
                const V ma = _mm512_permutex2var_epi64(_mm512_loadu_si512(h0), iu[ov], _mm512_loadu_si512(h0 + 8));
                V ha = z;
                if(WD >= 3){ const uint64_t *t0 = ds[2] + obase[ov]; ha = _mm512_permutex2var_epi64(_mm512_loadu_si512(t0), iu[ov], _mm512_loadu_si512(t0 + 8)); }
                fa = _mm512_mask_blend_epi64(m1[ov], la, ma);
                fb = _mm512_mask_blend_epi64(m1[ov], ma, ha);
                if(WD >= 3){
                    [[maybe_unused]] V xa=z;
                    if constexpr(WDMAX>3)if(WD>=4){const uint64_t *u=ds[3]+obase[ov];xa=_mm512_permutex2var_epi64(_mm512_loadu_si512(u),iu[ov],_mm512_loadu_si512(u+8));}
                    fa=_mm512_mask_blend_epi64(m2[ov],fa,ha);
#if CR_NP>8
                    fb=_mm512_mask_blend_epi64(m2[ov],fb,xa);
                    if(WD>=4){fa=_mm512_mask_blend_epi64(fourth.m3[ov],fa,xa);fb=_mm512_maskz_mov_epi64((__mmask8)~fourth.m3[ov],fb);}
#else
                    fb=_mm512_maskz_mov_epi64((__mmask8)~m2[ov],fb);
#endif
                }
            }
            const V limb = _mm512_or_si512(_mm512_shrdv_epi64(fa, fb, sh[ov]), _mm512_sllv_epi64(lb, lsh[ov]));
            if(nt) _mm512_stream_si512((__m512i *)(out + 8 * ov), limb);
            else _mm512_storeu_si512(out + 8 * ov, limb);
        }
    }
};

/* per-call constant tables (hoisted out of the chunk tasks) */
struct EmitTabs { EmitK EK; Regroup RG; };
inline void emit_tabs_init(EmitTabs &t, EmitCtx &c, const Primes &PS, const Plan &pl){
    t.EK.init(*c.G, PS, pl.T); t.RG.init(pl.T, pl.OT, pl.OL, pl.WD);
    c.EK = &t.EK; c.RG = &t.RG;
}

/* chunk task: view trunks [j0, j1) → limbs [j0·T/64, j1·T/64) + carry-out */
template<int L, int WD>
void emit_chunk_w(const EmitCtx &c, int task, scratch *ws, int wid, int touch, int nxt){
    FrameMark frame_mark(*ws); // the stage is dead on return, including multi-chunk tasks
    (void)wid;
    const Plan &pl = *c.pl;
    EP_T(t_all0);
    const size_t T = (size_t)pl.T;
    const size_t j0 = (size_t)task * OCH;
    const size_t j1 = j0 + OCH < pl.ntp ? j0 + OCH : pl.ntp;
    const size_t base = j0 * T / 64;
    /* digit streams on the STACK (hot, reused by every task of this
     * worker): a per-task SALLOC lands on cold arena memory when the
     * worker scratch is not restored between tasks — 66 KB of RFO per
     * chunk, which was the whole emit gap vs the frozen fout */
    alignas(64) uint64_t ds[WDMAX][OCH + 32];
    const Pk52 K = pk52_mk();
    Unpick<L> up;
    const EmitK &EK = *(const EmitK *)c.EK;          /* built once per emit call */
    const Regroup &RG = *(const Regroup *)c.RG;
    V gvc[NP][L == 8 ? 1 : L]; size_t lastg[NP]; for(int q = 0; q < NP; ++q) lastg[q] = ~(size_t)0;
    (void)gvc;
    /* phase 1: streaming corner-turn gather per prime into an L2 stage:
     * view vectors [s0, s1) with s0 = j0/8 − 1 (seed) … (j1+16)/8 */
    const size_t s0 = j0 >= 8 ? j0 / 8 - 1 : j0 / 8;
    // T128 has OT=4: the final chunk may end halfway through a vector.
    // The lag loop consumes ceil((j1+16)/8) vectors, including its tail.
    const size_t s1 = (j1 + 16 + 7) / 8;
    const size_t nsv = s1 - s0;
    V *stage = SALLOC(ws, V, (size_t)NP * nsv);
    EP_T(g0);
    /* per-prime view-order gather into the L2 stage (NO software prefetch:
     * a 16-ahead T0 prefetch of the strided 52-B slots cost 1.4-1.5x on the
     * emit at W16 and 1.3x at W1 — NOTES §1.14; dead view vectors → zero,
     * partial vectors masked) */
    const V RT = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    static const int epf = (TB >= 8 ? 16 : 0);
    for(int q = 0; q < NP; ++q){
        V *sq = stage + (size_t)q * nsv;
        for(size_t s = s0; s < s1; ++s){
            V x = _mm512_setzero_si512();
            size_t v; int live = 1;
            if(c.reversed){ if(s >= c.vlim) live = 0; else v = c.vlim - 1 - s; }
            else v = pl.voff / 8 + s;
            if(live){
                if constexpr(L == 8){
                    int ok; const uint8_t *e = plane_at(pl, c.plane[q], v, &ok);
                    if(epf){                                   /* sparse pieces (TB >= 8): prefetch epf view vectors ahead */
                        int ok2; const uint8_t *e2 = plane_at(pl, c.plane[q], v + (size_t)epf, &ok2);
                        if(ok2){ _mm_prefetch((const char *)e2, _MM_HINT_T0); if((((uintptr_t)e2) & 63) + SLOT > 64) _mm_prefetch((const char *)(e2 + 64), _MM_HINT_T0); }
                    }
                    if(ok){
                        /* whole-line pieces: the block's first slot pulls the
                         * whole piece (13 lines, sequential) so the other
                         * slots' sparse gathers hit L2/L3 */
                        if(touch) for(size_t o = 64; o < (size_t)TB * SLOT; o += 64) _mm_prefetch((const char *)(e + o), _MM_HINT_T1);
                        x = ld52(e, K);
                    }
                }else{
                    const size_t g = v / L, j = v % L;
                    const size_t r = g & (pl.C - 1), cc = g >> pl.lgC;
                    if(cc < pl.lbv){
                        if(g != lastg[q]){
                            for(int sidx = 0; sidx < L; ++sidx)
                                gvc[q][sidx] = ld52(c.plane[q] + plane_off(pl, r * L + sidx, cc), K);
                            lastg[q] = g;
                        }
                        x = up.get(gvc[q], (int)j);
                    }
                }
                if(c.reversed) x = _mm512_permutexvar_epi64(RT, x);
                const size_t j = s * 8;
                if(j + 8 > pl.vtrunks){
                    const size_t lv = pl.vtrunks > j ? pl.vtrunks - j : 0;
                    x = _mm512_maskz_mov_epi64((__mmask8)((1u << lv) - 1), x);
                }
            }
            sq[s - s0] = x;
        }
    }
    EP_T(g1);
    EP_ADD(gather, g0, g1);
#ifdef CR_EMIT_PROF
    if(g_ep_skip){ g_ep[wid].vec += (j1 - j0) / 8; g_ep[wid].total += __rdtsc() - t_all0; for(int i2 = 0; i2 < 8; ++i2) c.spill[task][i2] = 0; return; }
#endif
    /* phase 2: digits + lag chain */
    V pb1[WD], pb2 = _mm512_setzero_si512();
    for(int k = 0; k < WD; ++k) pb1[k] = pb2;
    if(j0 >= 8){
        V x[NP];
        for(int q = 0; q < NP; ++q) x[q] = stage[(size_t)q * nsv];
        DigW<WD> d = digits_of<WD>(EK, x);
        for(int k = 0; k < WD; ++k) pb1[k] = d.b1[k];
        pb2 = d.b2;
    }
    unsigned cA = 0, cout = 0;
    for(size_t j = j0; j < j1 + 16; j += 8){
        const size_t s = j / 8;
        V x[NP];
        EP_T(e1);
        for(int q = 0; q < NP; ++q) x[q] = stage[(size_t)q * nsv + (s - s0)];
        if(nxt){                                       /* whole-line pieces: the next slot's chunk (same rows) — its lines are pulled here, under this chunk's digit work */
            const size_t vn = pl.voff / 8 + s + pl.C;
            for(int q = 0; q < NP; ++q){
                int ok; const uint8_t *en = plane_at(pl, c.plane[q], vn, &ok);
                if(ok){ _mm_prefetch((const char *)en, _MM_HINT_T0); if((((uintptr_t)en) & 63) + SLOT > 64) _mm_prefetch((const char *)(en + 64), _MM_HINT_T0); }
            }
        }
        DigW<WD> d = digits_of<WD>(EK, x);
        EP_T(e2);
        if(j == j1){ cout = cA; cA = 0; }              /* tail: zero carry-in */
        V w[WD], a1[WD];
        for(int k = 0; k < WD; ++k){ w[k] = d.b0[k]; a1[k] = _mm512_alignr_epi64(d.b1[k], pb1[k], 7); }
        const V a2 = _mm512_alignr_epi64(d.b2, pb2, 6);
        cA = lag_add<WD>(w, a1, a2, cA, EK.LIM, EK.HM, EK.HB);
        for(int k = 0; k < WD; ++k){ pb1[k] = d.b1[k]; _mm512_storeu_si512(ds[k] + (j - j0), w[k]); }
        pb2 = d.b2;
        EP_T(e3);
        EP_ADD(dig, e1, e2); EP_ADD(chain, e2, e3);
#ifdef CR_EMIT_PROF
        ++g_ep[wid].vec;
#endif
    }
    /* regroup superblocks of OT trunks → OL limbs, straight into the sinks */
    const size_t nsb = (j1 - j0) / (size_t)pl.OT;
    alignas(64) uint64_t tmp[48 * 8];
    EP_T(r0);
    for(size_t sb = 0; sb < nsb; ++sb){
        const size_t tb = sb * (size_t)pl.OT;
        const size_t lb = base + sb * (size_t)pl.OL;
        const uint64_t *dsp[WDMAX];for(int k=0;k<WDMAX;++k)dsp[k]=ds[k]+tb;
        if(lb + (size_t)pl.OL <= pl.outcap) RG.run(c.rp + lb, dsp, c.nt_out);
        else{
            RG.run(tmp, dsp, 0);
            for(int i = 0; i < pl.OL; ++i) *emit_sink(c, lb + i) = tmp[i];
        }
    }
    EP_T(r1);
    EP_ADD(regroup, r0, r1);
    if(c.nt_out) _mm_sfence();
    for(int i2 = 0; i2 < 8; ++i2) c.spill[task][i2] = 0;
    c.spill[task][0] = cout;
#ifdef CR_EMIT_PROF
    g_ep[wid].total += __rdtsc() - t_all0;
#endif
}
template<int L>
void emit_chunk(const EmitCtx &c, int task, scratch *ws, int wid, int touch = 0, int nxt = 0){
    if constexpr(WDMAX>3)if(c.pl->WD==4){emit_chunk_w<L,4>(c,task,ws,wid,touch,nxt);return;}
    if(c.pl->WD == 3) emit_chunk_w<L, 3>(c, task, ws, wid, touch, nxt);
    else emit_chunk_w<L, 2>(c, task, ws, wid, touch, nxt);
}

/* serial seam join: add chunk k's spill at the next chunk's base */
inline void emit_join(const EmitCtx &c){
    const Plan &pl = *c.pl;
    for(int k = 0; k < c.nch; ++k){
        const size_t j1 = ((size_t)k + 1) * OCH < pl.ntp ? ((size_t)k + 1) * OCH : pl.ntp;
        const size_t at = j1 < pl.ntp ? j1 * (size_t)pl.T / 64 : pl.nl;
        if(at >= pl.nl) continue;
        unsigned char cy = 0;
        for(int i2 = 0; i2 < 8; ++i2){
            if(at + (size_t)i2 >= pl.nl) break;
            uint64_t *lp = emit_sink(c, at + (size_t)i2);
            u128 s = (u128)*lp + c.spill[k][i2] + cy;
            *lp = (uint64_t)s;
            cy = (unsigned char)(s >> 64);
        }
        for(size_t i2 = at + 8; cy && i2 < pl.nl; ++i2){
            uint64_t *lp = emit_sink(c, i2);
            cy = (unsigned char)(++*lp == 0);
        }
    }
}

/* cyclic ring fold: rp[0..rn) += top (limbs [rn, nl)), 2^(64rn) == 1 */
inline void fold_cyc(uint64_t *rp, size_t rn, const uint64_t *top, size_t ex){
    unsigned char cy = 0;
    for(size_t i = 0; i < ex && i < rn; ++i){
        u128 s = (u128)rp[i] + top[i] + cy;
        rp[i] = (uint64_t)s;
        cy = (unsigned char)(s >> 64);
    }
    for(size_t i = ex; cy && i < rn; ++i) cy = (unsigned char)(++rp[i] == 0);
    while(cy){
        cy = 0;
        size_t i = 0;
        for(; i < rn; ++i) if(++rp[i] != 0) break;
        if(i == rn) cy = 1;
    }
    // The all-one representative is zero modulo 2^(64*rn)-1. Keep the
    // public value canonical; this reads only the ring, never a full LIN tail.
    uint64_t all=UINT64_MAX;for(size_t i=0;i<rn;++i)all&=rp[i];
    if(all==UINT64_MAX)memset(rp,0,rn*8);
}

} // namespace sbn::v3::SBN3_P48_NS
