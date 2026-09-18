/* Imported from labs/cr/cr_rows.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_rows.hpp — row-axis transform kernels over W-batched rows of
 * 8-lane vectors. Element i of a W-batch occupies vectors
 * x[i*W .. i*W+W). Every kernel is a template on
 *   IS : butterfly STRUCTURE  (false = BFR  lo=a+wb, hi=a-wb, levels top-down;
 *                              true  = IBFR lo=a+b,  hi=(a-b)w, levels bottom-up)
 *   WI : twiddle TABLE        (false = w, true = winv)
 * which yields the four transforms of one code body:
 *   <false,false> forward F              <true,true>  unnormalized inverse n·F^-1
 *   <true,false>  F^T                    <false,true> (n·F^-1)^T
 * (transposition = reverse the level order + transpose each 2x2, which
 * is exactly IS-flip; the table stays.)
 *
 * Truncated kernels (vdH), radix-2 recursion with full-node fast paths:
 *   tft<TR>   truncated forward (TR=0) / its transpose (TR=1)
 *   itft<TR>  normalized truncated inverse (TR=0) / its transpose (TR=1)
 * Full interior nodes of itft use either the recursive per-level
 * normalized form (FAST=0, the vdH reference shape) or the radix-8
 * unnormalized walk + one scale pass (FAST=1; transposed: scale first).
 *
 * Lazy ranges: inputs/outputs in [0, 4p); b52 accepts any a < 2^52. */
#include "codec.hpp"
#include "tables.hpp"

namespace sbn::v3::SBN3_P48_NS {

/* ---- software prefetch cursor -------------------------------------------
 * Streams the NEXT work item's input into L2 (→ L3 on eviction) from inside
 * the transform loops, so the memory phase of item k+1 overlaps the ALU
 * phase of item k: n pieces of `lines` cache lines at base + i·stride,
 * `k` lines per step; chained cursors (next) follow. */
struct PfCur {
    const uint8_t *base; size_t stride; size_t n; uint32_t lines, k;
    size_t i; uint32_t l;
    PfCur *next;
    /* side job: an independent memory-bound task (e.g. the previous slot's
     * products streaming A from DRAM) run one step per `side_every` hooks,
     * so its DRAM stream hides under this kernel's ALU work */
    void (*side)(void *); void *sctx; uint32_t side_every, side_ctr;
    uint32_t hint;                /* 0 T0, 1 T1 (default), 2 NTA */
};
inline void pf_init(PfCur &c, const uint8_t *base, size_t stride, uint32_t lines, size_t n, int k){
    c.base = base; c.stride = stride; c.lines = lines; c.n = n; c.k = (uint32_t)k; c.i = 0; c.l = 0; c.next = NULL;
    c.side = NULL; c.sctx = NULL; c.side_every = 1; c.side_ctr = 0; c.hint = 1;
}
__attribute__((always_inline)) inline
void pf_step(PfCur *c, int mul){
    if(c->side && ++c->side_ctr >= c->side_every){ c->side_ctr = 0; c->side(c->sctx); }
    while(c && c->i >= c->n) c = c->next;
    if(!c) return;
    for(uint32_t t = 0; t < c->k * (uint32_t)mul; ++t){
        const char *pa = (const char *)(c->base + c->i * c->stride + (size_t)c->l * 64);
        if(c->hint == 1) _mm_prefetch(pa, _MM_HINT_T1); else if(c->hint == 2) _mm_prefetch(pa, _MM_HINT_NTA); else _mm_prefetch(pa, _MM_HINT_T0);
        if(++c->l == c->lines){ c->l = 0; if(++c->i >= c->n) break; }
    }
}
#define CR_PF(pf, m) do{ if(pf) pf_step((pf), (m)); }while(0)

/* Twiddles of tree node j and of its level-k children come from the block cursor
 * twcur<WI,H>(P, j) / twk<WI,H>(cursor, k, c) of tables.hpp: the compressed tower (H = false,
 * either direction) or the per-column materialized table of cr_col.hpp at the heap index
 * (T[nb + tn], H = true; the kernels are then entered at the heap root j = 1). */

template<bool IS>
__attribute__((always_inline)) inline
void bf(V &lo, V &hi, V c, V rec, const PrimeV &pv){
    if constexpr(!IS){
        V a = sh2(lo, pv.p2);
        V wb = b52(hi, c, rec, pv.pn);
        lo = _mm512_add_epi64(a, wb);
        hi = _mm512_add_epi64(a, _mm512_sub_epi64(pv.p2, wb));
    }else{
        V a = sh2(lo, pv.p2), b = sh2(hi, pv.p2);
        lo = _mm512_add_epi64(a, b);
        hi = b52(_mm512_add_epi64(_mm512_sub_epi64(a, b), pv.p2), c, rec, pv.pn);
    }
}
#define CR_BF(a_, b_, T_) bf<IS>(x[a_], x[b_], (T_).c, (T_).rec, pv)
#define CR_BFX(a_, b_, T_) bf<IS>(x[a_], x[b_], (T_).c, (T_).rec, pv)

/* ---- p48 fold-free butterflies (NOTES §24.2) -----------------------------
 * With p < 2^48 a 52-bit IFMA lane holds 16p. DIT (IS=false): lo' = a + wb,
 * hi' = a + 2p − wb, both < bound(a) + 2p; only the b52 input hi must be
 * < 16p. DIF (IS=true): lo' = a + b, hi' = (a − b + bias)·w with b < bias,
 * a + bias <= 16p. The moths keep the [0,4p) contract at their boundaries:
 * the inverse by a fixed per-lane fold schedule, the forward by folding
 * (foldz, 3 ops per lane) only where the DFS chain would exceed 16p
 * (fold_sched: every ~6 levels instead of 2 ops per pair per level). */
#ifdef CR_P48_CHECK
static inline void p48_chk(V x, V lim, const char *what){
    if(_mm512_cmpge_epu64_mask(x, lim)) ::sbn::v3::fatal(SBN3_FATAL_MATH, what);
}
#define P48_CHK(x, lim, what) p48_chk((x), (lim), (what))
#else
#define P48_CHK(x, lim, what) do{}while(0)
#endif
__attribute__((always_inline)) inline
void bfn_f(V &lo, V &hi, V c, V rec, const PrimeV &pv){
    P48_CHK(hi, vset(1ull << 52), "bfn_f hi");
    const V a = lo, wb = b52(hi, c, rec, pv.pn);
    lo = _mm512_add_epi64(a, wb);
    hi = _mm512_add_epi64(a, _mm512_sub_epi64(pv.p2, wb));
}
__attribute__((always_inline)) inline
void bfn_i(V &lo, V &hi, V c, V rec, V bias, const PrimeV &pv){
    P48_CHK(hi, bias, "bfn_i b < bias");
    P48_CHK(_mm512_add_epi64(lo, bias), vset(1ull << 52), "bfn_i a + bias");
    const V a = lo, b = hi;
    lo = _mm512_add_epi64(a, b);
    hi = b52(_mm512_add_epi64(_mm512_sub_epi64(a, b), bias), c, rec, pv.pn);
}
__attribute__((always_inline)) inline
V foldz_c(V x, const PrimeV &pv){ P48_CHK(x, vset(1ull << 52), "foldz"); return foldz(x, pv.r1, pv.pn); }
/* radix-8 levels on 8 registers. Forward: inputs < B·p (B <= 10), outputs < (B+6)p, FOLD → [0,2p).
 * Inverse: inputs < 4p → outputs < 4p (10 fold ops per 24 vector-levels). */
/* SC (inverse only): t1 arrives pre-multiplied by the scale s and the four lo outputs of the last
 * level are b52'd by s — the normalized inverse (cfinv) without its scale pass (NOTES §28) */
template<bool IS, bool FOLD, bool SC = false>
__attribute__((always_inline)) inline
void m8_lv(V x[8], const vcc &t1, const vcc &t2a, const vcc &t2b, const vcc &t30, const vcc &t31, const vcc &t32, const vcc &t33, const PrimeV &pv, const vcc *sc = NULL){
#define CR_BFF(a_, b_, T_) bfn_f(x[a_], x[b_], (T_).c, (T_).rec, pv)
#define CR_BFI(a_, b_, T_, B_) bfn_i(x[a_], x[b_], (T_).c, (T_).rec, (B_), pv)
    if constexpr(!IS){
        CR_BFF(0,4,t1); CR_BFF(1,5,t1); CR_BFF(2,6,t1); CR_BFF(3,7,t1);
        CR_BFF(0,2,t2a); CR_BFF(1,3,t2a); CR_BFF(4,6,t2b); CR_BFF(5,7,t2b);
        CR_BFF(0,1,t30); CR_BFF(2,3,t31); CR_BFF(4,5,t32); CR_BFF(6,7,t33);
        if constexpr(FOLD) for(int c = 0; c < 8; ++c) x[c] = foldz_c(x[c], pv);
    }else{
        CR_BFI(0,1,t30,pv.p4); CR_BFI(2,3,t31,pv.p4); CR_BFI(4,5,t32,pv.p4); CR_BFI(6,7,t33,pv.p4);   /* sums < 8p, diffs < 2p */
        CR_BFI(0,2,t2a,pv.p8); CR_BFI(1,3,t2a,pv.p2); CR_BFI(4,6,t2b,pv.p8); CR_BFI(5,7,t2b,pv.p2);   /* x0,x4 < 16p; x1,x5 < 4p; rest < 2p */
        x[0] = sh8(x[0], pv.p8); x[4] = sh8(x[4], pv.p8);
        if constexpr(SC){
            CR_BFI(0,4,t1,pv.p8); x[0] = b52(x[0], sc->c, sc->rec, pv.pn);   /* a + b < 16p */
            CR_BFI(1,5,t1,pv.p4); x[1] = b52(x[1], sc->c, sc->rec, pv.pn);
            CR_BFI(2,6,t1,pv.p2); CR_BFI(3,7,t1,pv.p2);
            x[2] = b52(x[2], sc->c, sc->rec, pv.pn); x[3] = b52(x[3], sc->c, sc->rec, pv.pn);
        }else{
            CR_BFI(0,4,t1,pv.p8); x[0] = sh4(sh8(x[0], pv.p8), pv.p4);
            CR_BFI(1,5,t1,pv.p4); x[1] = sh4(x[1], pv.p4);
            CR_BFI(2,6,t1,pv.p2); CR_BFI(3,7,t1,pv.p2);
        }
    }
#undef CR_BFF
#undef CR_BFI
}
/* the forward DFS chain: stage growths (moth8 +6p, b16 +8p, b32 +10p) from the [0,4p) entry;
 * fold after stage k when the next stage would take a b52/foldz input past 16p, and after the
 * last stage (the [0,4p) contract). lg 13 (moth, moth, moth, b16): folds after stages 1 and 3. */
struct FoldSched { unsigned char fold[24]; int n; };
static inline FoldSched fold_sched(size_t m, size_t LS){
    FoldSched S; S.n = 0;
    int g[24];
    for(size_t s = m; s > LS; s >>= 3) g[S.n++] = 6;
    g[S.n++] = LS == 8 ? 6 : LS == 16 ? 8 : 10;
    int B = 4;
    for(int k = 0; k < S.n; ++k){
        const int out = B + g[k];
        const int f = (k == S.n - 1) || (out + g[k + 1] > 16);
        S.fold[k] = (unsigned char)f; B = f ? 2 : out;
    }
    return S;
}

/* radix-8 moth: 3 levels of the (m, j) subtree at element stride e=m/8,
 * for element-rows v in [v0, v1) and every w < W. */
template<bool IS, size_t W, bool FOLD, bool SC = false>
void moth8_loop(uint64_t *B, size_t e, size_t v0, size_t v1, const vcc &t1, const vcc &t2a, const vcc &t2b,
                const vcc &t30, const vcc &t31, const vcc &t32, const vcc &t33, const PrimeV &pv, PfCur *pf, const vcc *sc = NULL){
#if CR_M8U2
    /* two element rows per iteration: the forward moth is a 3-level dependent b52 chain (~30 cycles
     * per 24 vector-levels); a second independent row in the same window lets the OoO core overlap
     * them (NOTES §25: p48 forward 0.87 → issue bound ~0.75 cycles per vector-level) */
    if constexpr(W == 1 && !IS){
        size_t v = v0;
        for(; v + 2 <= v1; v += 2){
            V x[8], y[8];
            CR_PF(pf, 2);
            for(int c = 0; c < 8; ++c){ x[c] = _mm512_load_si512(B + (v + (size_t)c * e) * 8); y[c] = _mm512_load_si512(B + (v + 1 + (size_t)c * e) * 8); }
            m8_lv<IS, FOLD>(x, t1, t2a, t2b, t30, t31, t32, t33, pv);
            m8_lv<IS, FOLD>(y, t1, t2a, t2b, t30, t31, t32, t33, pv);
            for(int c = 0; c < 8; ++c){ _mm512_store_si512(B + (v + (size_t)c * e) * 8, x[c]); _mm512_store_si512(B + (v + 1 + (size_t)c * e) * 8, y[c]); }
        }
        for(; v < v1; ++v){
            V x[8];
            CR_PF(pf, 1);
            for(int c = 0; c < 8; ++c) x[c] = _mm512_load_si512(B + (v + (size_t)c * e) * 8);
            m8_lv<IS, FOLD>(x, t1, t2a, t2b, t30, t31, t32, t33, pv);
            for(int c = 0; c < 8; ++c) _mm512_store_si512(B + (v + (size_t)c * e) * 8, x[c]);
        }
        return;
    }
#endif
    for(size_t v = v0; v < v1; ++v)
        for(size_t w = 0; w < W; ++w){
            V x[8];
            CR_PF(pf, 1);
            for(int c = 0; c < 8; ++c)
                x[c] = _mm512_load_si512(B + ((v + (size_t)c * e) * W + w) * 8);
            m8_lv<IS, FOLD, SC>(x, t1, t2a, t2b, t30, t31, t32, t33, pv, sc);
            for(int c = 0; c < 8; ++c)
                _mm512_store_si512(B + ((v + (size_t)c * e) * W + w) * 8, x[c]);
        }
}
/* fold: p48 forward only — fold the outputs to [0,2p) (fold_sched); ignored otherwise */
/* sc (inverse only): fused normalization — the last level's twiddle is w·s, the lo outputs b52'd by s */
template<bool IS, bool WI, size_t W, bool H = false>
void moth8(uint64_t *B, size_t m, size_t j, size_t v0, size_t v1,
           const Prime &P, const PrimeV &pv_, PfCur *pf = NULL, int fold = 1, const cc *sc = NULL){
    const PrimeV pv = pv_regs(pv_);
    const size_t e = m >> 3;
    const TwCur tc = twcur<WI, H>(P, j);
    const vcc t1  = (IS && sc) ? vcc_of(cc_of(mulm(cc_ld(twk<WI, H>(tc, 1, 0), P.p).c, sc->c, P.p), P.p)) : twv<WI, H>(tc, 1, 0, pv);
    const vcc t2a = twv<WI, H>(tc, 2, 0, pv), t2b = twv<WI, H>(tc, 2, 1, pv);
    const vcc t30 = twv<WI, H>(tc, 4, 0, pv), t31 = twv<WI, H>(tc, 4, 1, pv);
    const vcc t32 = twv<WI, H>(tc, 4, 2, pv), t33 = twv<WI, H>(tc, 4, 3, pv);
    if constexpr(IS){
        if(sc){ const vcc sv = vcc_of(*sc); moth8_loop<IS, W, true, true>(B, e, v0, v1, t1, t2a, t2b, t30, t31, t32, t33, pv, pf, &sv); return; }
    }
    if(!IS && !fold) moth8_loop<IS, W, false>(B, e, v0, v1, t1, t2a, t2b, t30, t31, t32, t33, pv, pf);
    else             moth8_loop<IS, W, true >(B, e, v0, v1, t1, t2a, t2b, t30, t31, t32, t33, pv, pf);
}

/* CR_TOPHOOK=0: the kernel hooks (side jobs / prefetch cursors) fire only in the levels whose
 * window fits L1 (span ≤ 512 vectors = 32 KB) and in the leaves — the top/L2-window levels
 * run undisturbed (NOTES §18) */
static inline PfCur *hook_for(size_t span, PfCur *pf){ (void)span; return pf; }

/* in-register 16-element bottom (one w-slice); the four levels are
 * macro blocks (lambdas were NOT inlined by clang → x[16] spilled
 * around every call: a 30% kernel regression caught by objdump) */
#define CR_B16_L1 do{ vcc t = twv<WI, H>(tc, 1, 0, pv); \
        CR_BF(0,8,t); CR_BF(1,9,t); CR_BF(2,10,t); CR_BF(3,11,t); \
        CR_BF(4,12,t); CR_BF(5,13,t); CR_BF(6,14,t); CR_BF(7,15,t); }while(0)
#define CR_B16_L2 do{ vcc a = twv<WI, H>(tc, 2, 0, pv), b = twv<WI, H>(tc, 2, 1, pv); \
        CR_BF(0,4,a); CR_BF(1,5,a); CR_BF(2,6,a); CR_BF(3,7,a); \
        CR_BF(8,12,b); CR_BF(9,13,b); CR_BF(10,14,b); CR_BF(11,15,b); }while(0)
#define CR_B16_L3 do{ for(int c = 0; c < 4; ++c){ vcc t = twv<WI, H>(tc, 4, (size_t)c, pv); \
        CR_BF(4*c, 4*c+2, t); CR_BF(4*c+1, 4*c+3, t); } }while(0)
#define CR_B16_L4 do{ for(int c = 0; c < 8; ++c){ vcc t = twv<WI, H>(tc, 8, (size_t)c, pv); \
        CR_BF(2*c, 2*c+1, t); } }while(0)
template<bool IS, bool WI, bool H = false, bool FOLD = true>
__attribute__((always_inline)) inline
void b16x(V x[16], size_t j, const Prime &P, const PrimeV &pv){
    const TwCur tc = twcur<WI, H>(P, j);
    /* forward: inputs < B·p (B <= 8), fold-free, FOLD → [0,2p); inverse: [0,4p) → [0,4p) with
     * the per-lane schedule of NOTES §24.2 (24 fold ops per 64 vector-levels) */
#define CR_F16_L1 do{ vcc t = twv<WI, H>(tc, 1, 0, pv); for(int c = 0; c < 8; ++c) bfn_f(x[c], x[c + 8], t.c, t.rec, pv); }while(0)
#define CR_F16_L2 do{ vcc a = twv<WI, H>(tc, 2, 0, pv), b = twv<WI, H>(tc, 2, 1, pv); \
        for(int c = 0; c < 4; ++c){ bfn_f(x[c], x[c + 4], a.c, a.rec, pv); bfn_f(x[8 + c], x[12 + c], b.c, b.rec, pv); } }while(0)
#define CR_F16_L3 do{ for(int c = 0; c < 4; ++c){ vcc t = twv<WI, H>(tc, 4, (size_t)c, pv); \
        bfn_f(x[4*c], x[4*c+2], t.c, t.rec, pv); bfn_f(x[4*c+1], x[4*c+3], t.c, t.rec, pv); } }while(0)
#define CR_F16_L4 do{ for(int c = 0; c < 8; ++c){ vcc t = twv<WI, H>(tc, 8, (size_t)c, pv); \
        bfn_f(x[2*c], x[2*c+1], t.c, t.rec, pv); } }while(0)
    if constexpr(!IS){
        CR_F16_L1; CR_F16_L2; CR_F16_L3; CR_F16_L4;
        if constexpr(FOLD) for(int c = 0; c < 16; ++c) x[c] = foldz_c(x[c], pv);
    }else{
        /* L4: pairs (2c, 2c+1), inputs < 4p → sums < 8p, diffs < 2p */
        for(int c = 0; c < 8; ++c){ vcc t = twv<WI, H>(tc, 8, (size_t)c, pv); bfn_i(x[2*c], x[2*c+1], t.c, t.rec, pv.p4, pv); }
        /* L3: (4c, 4c+2) sums of sums < 16p; (4c+1, 4c+3) < 4p */
        for(int c = 0; c < 4; ++c){ vcc t = twv<WI, H>(tc, 4, (size_t)c, pv);
            bfn_i(x[4*c], x[4*c+2], t.c, t.rec, pv.p8, pv); bfn_i(x[4*c+1], x[4*c+3], t.c, t.rec, pv.p2, pv); }
        /* L2: (c, c+4) per half: lanes 0/4 pre-folded to < 8p → x0 < 16p; x1 < 8p; x2, x3 < 4p; 4..7 < 2p */
        { vcc a = twv<WI, H>(tc, 2, 0, pv), b = twv<WI, H>(tc, 2, 1, pv);
          x[0] = sh8(x[0], pv.p8); x[4] = sh8(x[4], pv.p8);
          bfn_i(x[0], x[4], a.c, a.rec, pv.p8, pv); bfn_i(x[1], x[5], a.c, a.rec, pv.p4, pv);
          bfn_i(x[2], x[6], a.c, a.rec, pv.p2, pv); bfn_i(x[3], x[7], a.c, a.rec, pv.p2, pv);
          x[8] = sh8(x[8], pv.p8); x[12] = sh8(x[12], pv.p8);
          bfn_i(x[8], x[12], b.c, b.rec, pv.p8, pv); bfn_i(x[9], x[13], b.c, b.rec, pv.p4, pv);
          bfn_i(x[10], x[14], b.c, b.rec, pv.p2, pv); bfn_i(x[11], x[15], b.c, b.rec, pv.p2, pv); }
        /* L1: (c, c+8): x0,x8 < 16p pre-folded; x1,x9 < 8p; x2,x3,x10,x11 < 4p; rest < 2p → all outputs < 4p */
        { vcc t = twv<WI, H>(tc, 1, 0, pv);
          x[0] = sh8(x[0], pv.p8); x[8] = sh8(x[8], pv.p8);
          bfn_i(x[0], x[8], t.c, t.rec, pv.p8, pv); x[0] = sh4(sh8(x[0], pv.p8), pv.p4);
          bfn_i(x[1], x[9], t.c, t.rec, pv.p8, pv); x[1] = sh4(sh8(x[1], pv.p8), pv.p4);
          bfn_i(x[2], x[10], t.c, t.rec, pv.p4, pv); x[2] = sh4(x[2], pv.p4);
          bfn_i(x[3], x[11], t.c, t.rec, pv.p4, pv); x[3] = sh4(x[3], pv.p4);
          bfn_i(x[4], x[12], t.c, t.rec, pv.p2, pv); bfn_i(x[5], x[13], t.c, t.rec, pv.p2, pv);
          bfn_i(x[6], x[14], t.c, t.rec, pv.p2, pv); bfn_i(x[7], x[15], t.c, t.rec, pv.p2, pv); }
    }
#undef CR_F16_L1
#undef CR_F16_L2
#undef CR_F16_L3
#undef CR_F16_L4
}
template<bool IS, bool WI, size_t W, bool H = false>
void b16(uint64_t *B, size_t j, const Prime &P, const PrimeV &pv_, PfCur *pf = NULL, int fold = 1){
    const PrimeV pv = pv_regs(pv_);
    for(size_t w = 0; w < W; ++w){
        V x[16];
        CR_PF(pf, 2);
        for(int c = 0; c < 16; ++c) x[c] = _mm512_load_si512(B + ((size_t)c * W + w) * 8);
        if(!IS && !fold) b16x<IS, WI, H, false>(x, j, P, pv);
        else             b16x<IS, WI, H, true >(x, j, P, pv);
        for(int c = 0; c < 16; ++c) _mm512_store_si512(B + ((size_t)c * W + w) * 8, x[c]);
    }
}
/* dedicated LS = 8 leaf (NOTES §28): the moth8(…, v0 = 0, v1 = 1) call path cost 2.1 TSC per
 * vector-level for the p48 forward with fold (two calls, pv and the 7 twiddle pairs through the
 * stack, one exposed 3-level chain per 8 vectors) and 1.3× per level for every lg ≡ 0 (mod 3)
 * size in both builds. Here: twiddles in registers, TWO blocks per call so the OoO core overlaps
 * the two chains (leafprobe2: fwd fold 2.12 → 1.09, inv 0.82 → 0.89 TSC per vector-level). */
template<bool IS, bool FOLD>
__attribute__((always_inline)) inline
void m8_core(V x[8], const vcc &t1, const vcc &t2a, const vcc &t2b, const vcc &t30, const vcc &t31, const vcc &t32, const vcc &t33, const PrimeV &pv){
    m8_lv<IS, FOLD>(x, t1, t2a, t2b, t30, t31, t32, t33, pv);
}
#define CR_TW7(tc_) const vcc t1 = twv<WI, H>(tc_, 1, 0, pv), t2a = twv<WI, H>(tc_, 2, 0, pv), t2b = twv<WI, H>(tc_, 2, 1, pv), \
    t30 = twv<WI, H>(tc_, 4, 0, pv), t31 = twv<WI, H>(tc_, 4, 1, pv), t32 = twv<WI, H>(tc_, 4, 2, pv), t33 = twv<WI, H>(tc_, 4, 3, pv)
template<bool IS, bool WI, bool H, bool FOLD>
void leaf8x2(uint64_t *B, size_t j, const Prime &P, const PrimeV &pv_, PfCur *pf){
    const PrimeV pv = pv_regs(pv_);
    const TwCur tc0 = twcur<WI, H>(P, j), tc1 = twcur<WI, H>(P, j + 1);
    V x[8], y[8];
    CR_PF(pf, 2);
    for(int c = 0; c < 8; ++c){ x[c] = _mm512_load_si512(B + (size_t)c * 8); y[c] = _mm512_load_si512(B + 64 + (size_t)c * 8); }
    { CR_TW7(tc0); m8_core<IS, FOLD>(x, t1, t2a, t2b, t30, t31, t32, t33, pv); }
    { CR_TW7(tc1); m8_core<IS, FOLD>(y, t1, t2a, t2b, t30, t31, t32, t33, pv); }
    for(int c = 0; c < 8; ++c){ _mm512_store_si512(B + (size_t)c * 8, x[c]); _mm512_store_si512(B + 64 + (size_t)c * 8, y[c]); }
}
#undef CR_TW7
template<bool IS, bool WI, bool H>
inline void leaf8_pair(uint64_t *B, size_t j, const Prime &P, const PrimeV &pv, PfCur *pf, int fold){
    if(!IS && !fold) leaf8x2<IS, WI, H, false>(B, j, P, pv, pf);
    else             leaf8x2<IS, WI, H, true >(B, j, P, pv, pf);
}
template<bool IS, bool WI, size_t W, bool H = false>
void b32(uint64_t *B, size_t j, const Prime &P, const PrimeV &pv_, PfCur *pf = NULL, int fold = 1){
    const PrimeV pv = pv_regs(pv_);
    const TwCur tc = twcur<WI, H>(P, j);
    const vcc t = twv<WI, H>(tc, 1, 0, pv);
    /* forward: one fold-free level (+2p) then the b16s (which fold if asked); inverse: the b16s
     * ([0,4p) out) then the level with bias 4p, sums folded back to < 4p */
#define CR_B32_LVL do{ for(size_t v = 0; v < 16 * W; ++v){ \
            V lo = _mm512_load_si512(B + v * 8), hi = _mm512_load_si512(B + (v + 16 * W) * 8); \
            if constexpr(!IS) bfn_f(lo, hi, t.c, t.rec, pv); \
            else { bfn_i(lo, hi, t.c, t.rec, pv.p4, pv); lo = sh4(lo, pv.p4); } \
            _mm512_store_si512(B + v * 8, lo); \
            _mm512_store_si512(B + (v + 16 * W) * 8, hi); } }while(0)
    if constexpr(!IS){ CR_B32_LVL; b16<IS,WI,W,H>(B, 2*j, P, pv, pf, fold); b16<IS,WI,W,H>(B + 16*W*8, 2*j+1, P, pv, pf, fold); }
    else { b16<IS,WI,W,H>(B, 2*j, P, pv, pf); b16<IS,WI,W,H>(B + 16*W*8, 2*j+1, P, pv, pf); CR_B32_LVL; }
#undef CR_B32_LVL
}

inline size_t leaf_ls(size_t m){
    int lgm = 0; while(((size_t)1 << lgm) < m) ++lgm;
    const int r = lgm % 3;
    return r == 0 ? 8 : r == 1 ? 16 : 32;
}

/* the full (m, j) transform: DFS leaf-block walk of radix-8 moths with
 * in-register bottoms; forward structure = moths top-down then bottom,
 * inverse structure = bottom then moths bottom-up. */
inline void scale_vec(V *x, size_t cnt, cc s, const PrimeV &pv_);
/* sc (inverse only): normalized inverse with the scale fused into the top moth (NOTES §28) */
template<bool IS, bool WI, size_t W, bool H = false>
void full(uint64_t *B, size_t m, size_t j, const Prime &P, const PrimeV &pv_, PfCur *pf = NULL, const cc *sc = NULL){
    const PrimeV pv = pv_regs(pv_);
    const TwCur tc = twcur<WI, H>(P, j);
    if(m <= 32 && sc){                                /* no moth top: kernel + scale pass (tiny) */
        full<IS, WI, W, H>(B, m, j, P, pv, pf, NULL);
        scale_vec((V *)B, m * W, *sc, pv);
        return;
    }
    if(m <= 4){
        if(m == 1) return;
        for(size_t w = 0; w < W; ++w){
            V x[4];
            for(size_t c = 0; c < m; ++c) x[c] = _mm512_load_si512(B + (c * W + w) * 8);
            if(m == 2){ vcc t = twv<WI, H>(tc, 1, 0, pv); CR_BF(0,1,t); }
            else{
                vcc t = twv<WI, H>(tc, 1, 0, pv);
                vcc a = twv<WI, H>(tc, 2, 0, pv), b = twv<WI, H>(tc, 2, 1, pv);
                if constexpr(!IS){ CR_BF(0,2,t); CR_BF(1,3,t); CR_BF(0,1,a); CR_BF(2,3,b); }
                else             { CR_BF(0,1,a); CR_BF(2,3,b); CR_BF(0,2,t); CR_BF(1,3,t); }
            }
            for(size_t c = 0; c < m; ++c) _mm512_store_si512(B + (c * W + w) * 8, x[c]);
        }
        return;
    }
    if(m == 8){ moth8<IS,WI,W,H>(B, 8, j, 0, 1, P, pv, pf); return; }
    if(m == 16){ b16<IS,WI,W,H>(B, j, P, pv, pf); return; }
    if(m == 32){ b32<IS,WI,W,H>(B, j, P, pv, pf); return; }
    const size_t LS = leaf_ls(m), nb = m / LS;
    const size_t stop = m;                          /* the top radix-8 span */
    const FoldSched FS = fold_sched(m, LS);         /* p48 forward chain (unused otherwise) */
    const int pair = CR_LEAF8P && W == 1 && LS == 8 && nb >= 2;   /* LS = 8 leaves two blocks per call (leaf8x2) */
    for(size_t b = 0; b < nb; b += pair ? 2 : 1){
        const size_t o = b * LS;
        if constexpr(!IS){
            int k = 0;
            for(size_t s = stop; s > LS; s >>= 3, ++k){
                if(b & (s / LS - 1)) continue;
                moth8<IS,WI,W,H>(B + o * W * 8, s, j * (m / s) + o / s, 0, s >> 3, P, pv, hook_for(s, pf), FS.fold[k]);
            }
        }
        uint64_t *Bb = B + o * W * 8;
        const size_t jb = j * nb + b;
        const int lf = FS.fold[FS.n - 1];
        if(pair)          leaf8_pair<IS,WI,H>(Bb, jb, P, pv, pf, lf);
        else if(LS == 8)  moth8<IS,WI,W,H>(Bb, 8, jb, 0, 1, P, pv, pf, lf);
        else if(LS == 16) b16<IS,WI,W,H>(Bb, jb, P, pv, pf, lf);
        else              b32<IS,WI,W,H>(Bb, jb, P, pv, pf, lf);
        if constexpr(IS){
            const size_t bl = pair ? b + 1 : b, ol = bl * LS;      /* last block finished this iteration */
            for(size_t s = 8 * LS; s <= stop; s <<= 3){
                if((bl + 1) & (s / LS - 1)) continue;
                const size_t os = ol + LS - s;
                moth8<IS,WI,W,H>(B + os * W * 8, s, j * (m / s) + os / s, 0, s >> 3, P, pv, hook_for(s, pf), 1, s == stop ? sc : NULL);
            }
        }
    }
}

/* ---- codec-fused variants (W = 1, heap tables): the forward's top moth8
 * reads its 8 inputs PACKED from the plane pieces (ld52 at the piece
 * stride), the inverse's top moth8 (its last level) writes them packed
 * (st52), and the inverse multiplies every leaf block by A (the products)
 * before its leaf transform. The columns then never need a sweep-in, a
 * product pass or a sweep-out of their own (tile v5, NOTES §15). */
template<bool IS, int PIN, int POUT, bool FOLD>
void moth8_x_loop(uint64_t *B, size_t e, size_t v0, size_t v1, const vcc &t1, const vcc &t2a, const vcc &t2b,
                  const vcc &t30, const vcc &t31, const vcc &t32, const vcc &t33, const PrimeV &pv, PfCur *pf,
                  const uint8_t *pin, uint8_t *pout, size_t pstr, const Pk52 &K){
    /* PIN straight from a DRAM region: 8 sequential streams (one per c), each
     * prefetched CR_PINPF iterations ahead (2 lines per piece) */
    static const int pinpf = 16;
    for(size_t v = v0; v < v1; ++v){
        V x[8];
        CR_PF(pf, 1);
        if constexpr(PIN){
            if(pinpf && v + (size_t)pinpf < v1)
                for(int c = 0; c < 8; ++c){
                    const uint8_t *p = pin + (v + (size_t)pinpf + (size_t)c * e) * pstr;
                    _mm_prefetch((const char *)p, _MM_HINT_T0);
                    if((((uintptr_t)p) & 63) + SLOT > 64) _mm_prefetch((const char *)(p + 64), _MM_HINT_T0);
                }
        }
        for(int c = 0; c < 8; ++c){
            if constexpr(PIN) x[c] = ld52(pin + (v + (size_t)c * e) * pstr, K);
            else x[c] = _mm512_load_si512(B + (v + (size_t)c * e) * 8);
        }
        m8_lv<IS, FOLD>(x, t1, t2a, t2b, t30, t31, t32, t33, pv);
        for(int c = 0; c < 8; ++c){
            if constexpr(POUT) stslot<SLOTO>(pout + (v + (size_t)c * e) * pstr, x[c], K, pv);
            else _mm512_store_si512(B + (v + (size_t)c * e) * 8, x[c]);
        }
    }
}
template<bool IS, bool WI, bool H, int PIN, int POUT>
void moth8_x(uint64_t *B, size_t m, size_t j, size_t v0, size_t v1,
             const Prime &P, const PrimeV &pv_, PfCur *pf, const uint8_t *pin, uint8_t *pout, size_t pstr, const Pk52 &K, int fold = 1){
    const PrimeV pv = pv_regs(pv_);
    const size_t e = m >> 3;
    const TwCur tc = twcur<WI, H>(P, j);
    const vcc t1  = twv<WI, H>(tc, 1, 0, pv);
    const vcc t2a = twv<WI, H>(tc, 2, 0, pv), t2b = twv<WI, H>(tc, 2, 1, pv);
    const vcc t30 = twv<WI, H>(tc, 4, 0, pv), t31 = twv<WI, H>(tc, 4, 1, pv);
    const vcc t32 = twv<WI, H>(tc, 4, 2, pv), t33 = twv<WI, H>(tc, 4, 3, pv);
    if(!IS && !fold) moth8_x_loop<IS, PIN, POUT, false>(B, e, v0, v1, t1, t2a, t2b, t30, t31, t32, t33, pv, pf, pin, pout, pstr, K);
    else             moth8_x_loop<IS, PIN, POUT, true >(B, e, v0, v1, t1, t2a, t2b, t30, t31, t32, t33, pv, pf, pin, pout, pstr, K);
}
struct NoLeaf { __attribute__((always_inline)) inline void operator()(size_t, size_t) const {} };
template<bool IS, bool WI, int PIN, int POUT, class Leaf>
void full_x(uint64_t *B, size_t m, size_t j, const Prime &P, const PrimeV &pv_, PfCur *pf,
            const uint8_t *pin, uint8_t *pout, size_t pstr, const Pk52 &K, const Leaf &leaf){
    constexpr bool H = true; constexpr size_t W = 1;
    const PrimeV pv = pv_regs(pv_);
    if(m <= 32){                                      /* tiny columns: unpack, plain kernel, pack */
        if constexpr(PIN) for(size_t i = 0; i < m; ++i) _mm512_store_si512(B + i * 8, ld52(pin + i * pstr, K));
        if constexpr(IS) leaf(0, m);
        full<IS, WI, W, H>(B, m, j, P, pv, pf);
        if constexpr(POUT) for(size_t i = 0; i < m; ++i) stslot<SLOTO>(pout + i * pstr, _mm512_load_si512(B + i * 8), K, pv);
        return;
    }
    const size_t LS = leaf_ls(m), nb = m / LS;
    const size_t stop = m;
    const FoldSched FS = fold_sched(m, LS);         /* p48 forward chain (unused otherwise) */
    const int pair = CR_LEAF8P && LS == 8 && nb >= 2;
    for(size_t b = 0; b < nb; b += pair ? 2 : 1){
        const size_t o = b * LS;
        if constexpr(!IS){
            int k = 0;
            for(size_t s = stop; s > LS; s >>= 3, ++k){
                if(b & (s / LS - 1)) continue;
                if(s == m) moth8_x<IS,WI,H,PIN,0>(B + o * W * 8, s, j * (m / s) + o / s, 0, s >> 3, P, pv, hook_for(s, pf), pin, NULL, pstr, K, FS.fold[k]);
                else moth8<IS,WI,W,H>(B + o * W * 8, s, j * (m / s) + o / s, 0, s >> 3, P, pv, hook_for(s, pf), FS.fold[k]);
            }
        }
        uint64_t *Bb = B + o * W * 8;
        const size_t jb = j * nb + b;
        const int lf = FS.fold[FS.n - 1];
        if constexpr(IS){ leaf(o, LS); if(pair) leaf(o + LS, LS); }
        if(pair)          leaf8_pair<IS,WI,H>(Bb, jb, P, pv, pf, lf);
        else if(LS == 8)  moth8<IS,WI,W,H>(Bb, 8, jb, 0, 1, P, pv, pf, lf);
        else if(LS == 16) b16<IS,WI,W,H>(Bb, jb, P, pv, pf, lf);
        else              b32<IS,WI,W,H>(Bb, jb, P, pv, pf, lf);
        if constexpr(IS){
            const size_t bl = pair ? b + 1 : b, ol = bl * LS;
            for(size_t s = 8 * LS; s <= stop; s <<= 3){
                if((bl + 1) & (s / LS - 1)) continue;
                const size_t os = ol + LS - s;
                if(s == m) moth8_x<IS,WI,H,0,POUT>(B + os * W * 8, s, j * (m / s) + os / s, 0, s >> 3, P, pv, hook_for(s, pf), NULL, pout, pstr, K);
                else moth8<IS,WI,W,H>(B + os * W * 8, s, j * (m / s) + os / s, 0, s >> 3, P, pv, hook_for(s, pf));
            }
        }
    }
}
#undef CR_BFX
#undef CR_BF

/* x[i] = b52(x[i], s) over cnt vectors → [0, 2p) */
inline void scale_vec(V *x, size_t cnt, cc s, const PrimeV &pv_){
    const PrimeV pv = pv_regs(pv_);
    const vcc sv = vcc_of(s);
    for(size_t i = 0; i < cnt; ++i) x[i] = b52(x[i], sv.c, sv.rec, pv.pn);
}

/* ---- truncated forward (TR=0) and its transpose (TR=1) -------------- */
template<int TR, size_t W>
void tft_r(V *x, size_t n, size_t j, size_t l, const Prime &P, V p, V p2, V pn, PfCur *pf = NULL){
    if(l == 0 || n == 1) return;
    PrimeV pv; pv.from3(p, p2, pn, P);
    if(l == n){ full<TR != 0, false, W>((uint64_t *)x, n, j, P, pv, pf); return; }
    const size_t m = n >> 1, mW = m * W;
    const vcc w = vcc_ofr(P.e[j], pv.p);
    if constexpr(TR == 0){
        if(l <= m){
            for(size_t i = 0; i < mW; ++i){
                if(!(i & 7)) CR_PF(pf, 1);
                V wb = b52(x[i + mW], w.c, w.rec, pv.pn);
                x[i] = _mm512_add_epi64(sh2(x[i], pv.p2), wb);
            }
            tft_r<TR, W>(x, m, 2 * j, l, P, p, p2, pn, pf);
        }else{
            for(size_t i = 0; i < mW; ++i){
                if(!(i & 7)) CR_PF(pf, 1);
                V a = sh2(x[i], pv.p2);
                V wb = b52(x[i + mW], w.c, w.rec, pv.pn);
                x[i] = _mm512_add_epi64(a, wb);
                x[i + mW] = _mm512_add_epi64(a, _mm512_sub_epi64(pv.p2, wb));
            }
            tft_r<TR, W>(x, m, 2 * j, m, P, p, p2, pn, pf);
            tft_r<TR, W>(x + mW, m, 2 * j + 1, l - m, P, p, p2, pn, pf);
        }
    }else{
        if(l <= m){
            tft_r<TR, W>(x, m, 2 * j, l, P, p, p2, pn, pf);
            for(size_t i = 0; i < mW; ++i){                 /* hi += w·lo */
                V wb = b52(x[i], w.c, w.rec, pv.pn);
                x[i + mW] = _mm512_add_epi64(sh2(x[i + mW], pv.p2), wb);
            }
        }else{
            tft_r<TR, W>(x, m, 2 * j, m, P, p, p2, pn, pf);
            tft_r<TR, W>(x + mW, m, 2 * j + 1, l - m, P, p, p2, pn, pf);
            for(size_t i = 0; i < mW; ++i){                 /* [[1,1],[w,-w]] */
                V a = sh2(x[i], pv.p2), b = sh2(x[i + mW], pv.p2);
                x[i] = _mm512_add_epi64(a, b);
                V d = _mm512_add_epi64(a, _mm512_sub_epi64(pv.p2, b));
                x[i + mW] = b52(d, w.c, w.rec, pv.pn);
            }
        }
    }
}

template<int TR, size_t W>
inline void tft(V *x, size_t n, size_t j, size_t l, const Prime &P, const PrimeV &pv, PfCur *pf = NULL){
    tft_r<TR, W>(x, n, j, l, P, pv.p, pv.p2, pv.pn, pf);
}

/* ---- normalized full inverse (interior node of itft) ------------------ */
/* recursive per-level normalized form (vdH reference shape) */
template<int TR, size_t W>
void cfinv_rec(V *x, size_t n, size_t j, const Prime &P, V p, V p2, V pn, PfCur *pf = NULL){
    if(n == 1) return;
    PrimeV pv; pv.from3(p, p2, pn, P);
    const size_t m = n >> 1, mW = m * W;
    const vcc i2 = vcc_of(P.inv2);
    const vcc wi2 = wi2_of(P, j, pv.p);
    if constexpr(TR == 0){
        cfinv_rec<TR, W>(x, m, 2 * j, P, p, p2, pn, pf);
        cfinv_rec<TR, W>(x + mW, m, 2 * j + 1, P, p, p2, pn, pf);
        for(size_t i = 0; i < mW; ++i){
            if(!(i & 7)) CR_PF(pf, 1);
            V a = sh2(x[i], pv.p2), b = sh2(x[i + mW], pv.p2);
            x[i] = b52(_mm512_add_epi64(a, b), i2.c, i2.rec, pv.pn);
            x[i + mW] = b52(_mm512_add_epi64(a, _mm512_sub_epi64(pv.p2, b)), wi2.c, wi2.rec, pv.pn);
        }
    }else{
        for(size_t i = 0; i < mW; ++i){
            if(!(i & 7)) CR_PF(pf, 1);
            V t1 = b52(x[i], i2.c, i2.rec, pv.pn);
            V t2 = b52(x[i + mW], wi2.c, wi2.rec, pv.pn);
            x[i] = _mm512_add_epi64(t1, t2);
            x[i + mW] = _mm512_add_epi64(t1, _mm512_sub_epi64(pv.p2, t2));
        }
        cfinv_rec<TR, W>(x, m, 2 * j, P, p, p2, pn, pf);
        cfinv_rec<TR, W>(x + mW, m, 2 * j + 1, P, p, p2, pn, pf);
    }
}
/* FAST: radix-8 unnormalized walk + one scale (TR=1: scale first) */
template<int TR, size_t W>
void cfinv_fast(V *x, size_t n, size_t j, const Prime &P, const PrimeV &pv_, PfCur *pf = NULL){
    const PrimeV pv = pv_regs(pv_);
    if(n == 1) return;
    ::sbn::v3::require(n && !(n&(n-1)) && P.invpow2 && n<=(size_t(1)<<32),SBN3_FATAL_MATH,"prepared inverse scale");
    const cc s = P.invpow2[__builtin_ctzll(n)];
    if constexpr(TR == 0){
        full<true, true, W>((uint64_t *)x, n, j, P, pv, pf, &s);   /* scale fused into the top moth */
    }else{
        scale_vec(x, n * W, s, pv);
        full<false, true, W>((uint64_t *)x, n, j, P, pv, pf);
    }
}
template<int TR, int FAST, size_t W>
inline void cfinv(V *x, size_t n, size_t j, const Prime &P, V p, V p2, V pn, PfCur *pf = NULL){
    if constexpr(FAST){ PrimeV pv; pv.from3(p, p2, pn, P); cfinv_fast<TR, W>(x, n, j, P, pv, pf); }
    else cfinv_rec<TR, W>(x, n, j, P, p, p2, pn, pf);
}

/* ---- normalized truncated inverse (TR=0) and its transpose (TR=1) ----- */
/* ZT (TR = 0): the known tail [l, n) is zero (the root of a LIN product row) — its passes vanish */
template<int TR, int FAST, size_t W, bool ZT = false>
void itft_r(V *x, size_t n, size_t j, size_t l, const Prime &P, V p, V p2, V pn, PfCur *pf = NULL){
    if(l == 0 || n == 1) return;
    if(l == n){ cfinv<TR, FAST, W>(x, n, j, P, p, p2, pn, pf); return; }
    PrimeV pv; pv.from3(p, p2, pn, P);
    const size_t m = n >> 1, mW = m * W;
    const vcc w = vcc_ofr(P.e[j], pv.p);
    if constexpr(TR == 0){
        if(l <= m){
            if constexpr(!ZT) for(size_t i = l * W; i < mW; ++i){
                if(!(i & 7)) CR_PF(pf, 1);
                V wb = b52(x[i + mW], w.c, w.rec, pv.pn);
                x[i] = sh2(_mm512_add_epi64(sh2(x[i], pv.p2), wb), pv.p2);
            }
            itft_r<TR, FAST, W, ZT>(x, m, 2 * j, l, P, p, p2, pn, pf);
            if constexpr(!ZT) for(size_t i = 0; i < l * W; ++i){
                V wb = b52(x[i + mW], w.c, w.rec, pv.pn);
                x[i] = sh2(_mm512_add_epi64(sh2(x[i], pv.p2), _mm512_sub_epi64(pv.p2, wb)), pv.p2);
            }
        }else{
            cfinv<TR, FAST, W>(x, m, 2 * j, P, p, p2, pn, pf);
            if constexpr(ZT) for(size_t i = (l - m) * W; i < mW; ++i){     /* hi_i = u_i = lo_i */
                if(!(i & 7)) CR_PF(pf, 1);
                x[i + mW] = x[i];
            }
            else for(size_t i = (l - m) * W; i < mW; ++i){
                if(!(i & 7)) CR_PF(pf, 1);
                V wb = b52(x[i + mW], w.c, w.rec, pv.pn);
                V nb = _mm512_sub_epi64(pv.p2, wb);
                V ai = sh2(_mm512_add_epi64(sh2(x[i], pv.p2), nb), pv.p2);
                x[i] = ai;
                x[i + mW] = sh2(_mm512_add_epi64(ai, nb), pv.p2);
            }
            itft_r<TR, FAST, W, false>(x + mW, m, 2 * j + 1, l - m, P, p, p2, pn, pf);
            const vcc i2 = vcc_of(P.inv2);
            const vcc wi2 = wi2_of(P, j, pv.p);
            for(size_t i = 0; i < (l - m) * W; ++i){
                V b = sh2(x[i], pv.p2), c = sh2(x[i + mW], pv.p2);
                x[i] = b52(_mm512_add_epi64(b, c), i2.c, i2.rec, pv.pn);
                x[i + mW] = b52(_mm512_add_epi64(b, _mm512_sub_epi64(pv.p2, c)), wi2.c, wi2.rec, pv.pn);
            }
        }
    }else{
        if(l <= m){
            for(size_t i = 0; i < l * W; ++i){              /* hi -= w·lo */
                V wb = b52(x[i], w.c, w.rec, pv.pn);
                x[i + mW] = sh2(_mm512_add_epi64(sh2(x[i + mW], pv.p2), _mm512_sub_epi64(pv.p2, wb)), pv.p2);
            }
            itft_r<TR, FAST, W>(x, m, 2 * j, l, P, p, p2, pn, pf);
            for(size_t i = l * W; i < mW; ++i){             /* hi += w·lo */
                V wb = b52(x[i], w.c, w.rec, pv.pn);
                x[i + mW] = sh2(_mm512_add_epi64(sh2(x[i + mW], pv.p2), wb), pv.p2);
            }
        }else{
            const vcc i2 = vcc_of(P.inv2);
            const vcc wi2 = wi2_of(P, j, pv.p);
            for(size_t i = 0; i < (l - m) * W; ++i){
                V t1 = b52(x[i], i2.c, i2.rec, pv.pn);
                V t2 = b52(x[i + mW], wi2.c, wi2.rec, pv.pn);
                x[i] = sh2(_mm512_add_epi64(t1, t2), pv.p2);
                x[i + mW] = sh2(_mm512_add_epi64(t1, _mm512_sub_epi64(pv.p2, t2)), pv.p2);
            }
            itft_r<TR, FAST, W>(x + mW, m, 2 * j + 1, l - m, P, p, p2, pn, pf);
            for(size_t i = (l - m) * W; i < mW; ++i){       /* [[1,1],[-w,-2w]] */
                V a = sh2(x[i], pv.p2), b = sh2(x[i + mW], pv.p2);
                x[i] = sh2(_mm512_add_epi64(a, b), pv.p2);
                V s = sh2(_mm512_add_epi64(a, sh2(_mm512_add_epi64(b, b), pv.p2)), pv.p2);
                V t = b52(s, w.c, w.rec, pv.pn);
                x[i + mW] = _mm512_sub_epi64(pv.p2, t);     /* (0, 2p] */
            }
            cfinv<TR, FAST, W>(x, m, 2 * j, P, p, p2, pn, pf);
        }
    }
}

template<int TR, int FAST, size_t W, bool ZT = false>
inline void itft(V *x, size_t n, size_t j, size_t l, const Prime &P, const PrimeV &pv, PfCur *pf = NULL){
    itft_r<TR, FAST, W, ZT>(x, n, j, l, P, pv.p, pv.p2, pv.pn, pf);
}


/* ---- blocked (radix-8) truncated transforms, W = 1, tower tables (NOTES §28) ---------------
 * The vdH recursion above does one radix-2 memory pass per truncated level; at (8192, 6554)
 * that costs MORE than a full DFS (tftprobe, one core: tft 1.08×, itft 1.22× of full). Here
 * the top three levels of every truncated node are ONE register pass of 8-point truncated
 * kernels over the e = n/8 "columns" (Harvey's cache-friendly TFT/ITFT in radix-8 form): the
 * sub-blocks entirely needed/known run the full DFS, the single partial sub-block recurses.
 *
 * Column i of node (n, j) holds u[i + k·e], k < 8 = tree positions k of the node's 8-point
 * transform (nodes j → 2j, 2j+1 → 4j..4j+3, twiddles w[2·node]); sub-block k is node (e, 8j+k).
 *
 * Forward tft_blk(lout, lin): outputs [0, lout) needed, inputs [lin, n) known zero.
 *   kf = ceil(lout/e) output blocks, kin = ceil(lin/e) nonzero input blocks; the column kernel
 *   is the vdH tft on 8 registers with zero partners elided (a level-1 pair with a zero hi is a
 *   copy); then full DFS on blocks < kf−1, recursion on block kf−1 (all inputs nonzero).
 * Inverse itft_blk<ZT>(l): frequencies [0, l) known, time values [l, n) known — zero at the
 *   root (ZT), the partial sub-block's tail is not. kf = l/e full blocks, r = l − kf·e:
 *     1. normalized full inverse of blocks k < kf (cfinv);
 *     2. columns i >= r: vdH 8-point itft with L = kf → u[i + k·e] (k < kf); then the partial
 *        block's tail y^kf_i = forward output kf of the column's 8 time values;
 *     3. the partial block (e, r) recursively (ZT = false: its tail is step 2's output);
 *     4. columns i < r: the 8-point itft with L = kf + 1.
 * Lazy discipline: sh2 after every add (the plain [0,4p) contract; no p48 fold schedule here). */
/* CR_BLK: 0 vdH everywhere, 1 blocked forward + vdH inverse (default: the blocked inverse's 8-point
 * column kernel is ~2 moths and loses to vdH above fill 0.55 — tftprobe), 2 blocked both */

/* The 8-point column kernels take the seven node constants of their three levels from Col8Tw, built
 * once per column pass (heap index h: root 1, children 2h and 2h+1 = tree nodes 2^d·j + i). Inside
 * the per-column loop the stores may alias the tables, so nothing table-derived is left there. */
struct Col8Tw { vcc w[8], wi2[8], i2; };
inline void col8_build(Col8Tw &t, size_t j, const Prime &P, const PrimeV &pv){
    t.i2 = vcc_of(P.inv2);
    for(size_t d = 0; d < 3; ++d)
        for(size_t i = 0; i < ((size_t)1 << d); ++i){
            const size_t h = ((size_t)1 << d) + i, node = (j << d) + i;
            t.w[h] = vcc_ofr(P.e[node], pv.p);
            t.wi2[h] = wi2_of(P, node, pv.p);
        }
}
/* normalized full inverse of N registers at heap node Hh (cfinv_rec shape) */
template<int N, int Hh>
__attribute__((always_inline)) inline
void cfinv8_r(V *x, const Col8Tw &tw, const PrimeV &pv){
    if constexpr(N == 1) return;
    else{
        constexpr int m = N / 2;
        cfinv8_r<m, 2 * Hh>(x, tw, pv);
        cfinv8_r<m, 2 * Hh + 1>(x + m, tw, pv);
        const vcc i2 = tw.i2, wi2 = tw.wi2[Hh];
        for(int i = 0; i < m; ++i){
            V a = sh2(x[i], pv.p2), b = sh2(x[i + m], pv.p2);
            x[i] = b52(_mm512_add_epi64(a, b), i2.c, i2.rec, pv.pn);
            x[i + m] = b52(_mm512_add_epi64(a, _mm512_sub_epi64(pv.p2, b)), wi2.c, wi2.rec, pv.pn);
        }
    }
}
/* vdH truncated forward on N registers: outputs [0, L), inputs [KIN, N) zero */
template<int N, int L, int KIN, int Hh>
__attribute__((always_inline)) inline
void tft8_r(V *x, const Col8Tw &tw, const PrimeV &pv){
    if constexpr(L == 0 || N == 1) return;
    else{
        constexpr int m = N / 2;
        const vcc w = tw.w[Hh];
        if constexpr(L <= m){
            for(int i = 0; i < m; ++i)
                if(i + m < KIN){ V wb = b52(x[i + m], w.c, w.rec, pv.pn); x[i] = _mm512_add_epi64(sh2(x[i], pv.p2), wb); }
            tft8_r<m, L, (KIN < m ? KIN : m), 2 * Hh>(x, tw, pv);
        }else{
            for(int i = 0; i < m; ++i){
                if(i + m < KIN){
                    V a = sh2(x[i], pv.p2), wb = b52(x[i + m], w.c, w.rec, pv.pn);
                    x[i] = _mm512_add_epi64(a, wb);
                    x[i + m] = _mm512_add_epi64(a, _mm512_sub_epi64(pv.p2, wb));
                }else x[i + m] = x[i];                        /* zero partner: lo = hi = a */
            }
            tft8_r<m, m, m, 2 * Hh>(x, tw, pv);
            tft8_r<m, L - m, m, 2 * Hh + 1>(x + m, tw, pv);
        }
    }
}
/* output K (tree order) of the N-point forward from N time registers (inputs [KIN, N) zero) */
template<int N, int K, int KIN, int Hh>
__attribute__((always_inline)) inline
V fwd8_out(const V *x, const Col8Tw &tw, const PrimeV &pv){
    if constexpr(N == 1) return x[0];
    else{
        constexpr int m = N / 2;
        const vcc w = tw.w[Hh];
        V y[m];
        for(int i = 0; i < m; ++i){
            if(i + m < KIN){
                V a = sh2(x[i], pv.p2), wb = b52(x[i + m], w.c, w.rec, pv.pn);
                y[i] = K < m ? _mm512_add_epi64(a, wb) : _mm512_add_epi64(a, _mm512_sub_epi64(pv.p2, wb));
            }else y[i] = x[i];
        }
        if constexpr(K < m) return fwd8_out<m, K, (KIN < m ? KIN : m), 2 * Hh>(y, tw, pv);
        else return fwd8_out<m, K - m, m, 2 * Hh + 1>(y, tw, pv);
    }
}
/* vdH normalized truncated inverse on N registers, mixed format (itft_r TR = 0); ZT: the tail
 * (positions >= L) is known zero and never read */
template<int N, int L, bool ZT, int Hh>
__attribute__((always_inline)) inline
void itft8_r(V *x, const Col8Tw &tw, const PrimeV &pv){
    if constexpr(L == 0 || N == 1) return;
    else if constexpr(L == N) cfinv8_r<N, Hh>(x, tw, pv);
    else{
        constexpr int m = N / 2;
        const vcc w = tw.w[Hh];
        if constexpr(L <= m){
            if constexpr(!ZT) for(int i = L; i < m; ++i){
                V wb = b52(x[i + m], w.c, w.rec, pv.pn);
                x[i] = sh2(_mm512_add_epi64(sh2(x[i], pv.p2), wb), pv.p2);
            }
            itft8_r<m, L, ZT, 2 * Hh>(x, tw, pv);
            if constexpr(!ZT) for(int i = 0; i < L; ++i){
                V wb = b52(x[i + m], w.c, w.rec, pv.pn);
                x[i] = sh2(_mm512_add_epi64(sh2(x[i], pv.p2), _mm512_sub_epi64(pv.p2, wb)), pv.p2);
            }
        }else{
            cfinv8_r<m, 2 * Hh>(x, tw, pv);
            for(int i = L - m; i < m; ++i){
                if constexpr(ZT) x[i + m] = x[i];               /* zero tail: hi_i = u_i = lo_i */
                else{
                    V wb = b52(x[i + m], w.c, w.rec, pv.pn);
                    V nb = _mm512_sub_epi64(pv.p2, wb);
                    V ai = sh2(_mm512_add_epi64(sh2(x[i], pv.p2), nb), pv.p2);
                    x[i] = ai;
                    x[i + m] = sh2(_mm512_add_epi64(ai, nb), pv.p2);
                }
            }
            itft8_r<m, L - m, false, 2 * Hh + 1>(x + m, tw, pv);
            const vcc i2 = tw.i2, wi2 = tw.wi2[Hh];
            for(int i = 0; i < L - m; ++i){
                V b = sh2(x[i], pv.p2), c = sh2(x[i + m], pv.p2);
                x[i] = b52(_mm512_add_epi64(b, c), i2.c, i2.rec, pv.pn);
                x[i + m] = b52(_mm512_add_epi64(b, _mm512_sub_epi64(pv.p2, c)), wi2.c, wi2.rec, pv.pn);
            }
        }
    }
}

/* column passes over columns [i0, i1) of node (n = 8e, j) */
template<int KF, int KIN>
void colpass_fwd_t(V *x, size_t e, size_t j, const Prime &P, const PrimeV &pv_, PfCur *pf){
    const PrimeV pv = pv_regs(pv_);
    Col8Tw tw; col8_build(tw, j, P, pv);
    for(size_t i = 0; i < e; ++i){
        V r[8];
        CR_PF(pf, 1);
        for(int k = 0; k < 8; ++k) r[k] = k < KIN ? x[i + (size_t)k * e] : _mm512_setzero_si512();
        tft8_r<8, KF, KIN, 1>(r, tw, pv);
        for(int k = 0; k < KF; ++k) x[i + (size_t)k * e] = r[k];
    }
}
/* inverse: columns [i0, i1) with L known positions; TAIL: emit the forward output L (the partial
 * block's tail) — requires L < 8 */
template<int L, bool ZT, bool TAIL>
void colpass_inv_t(V *x, size_t e, size_t i0, size_t i1, size_t j, const Prime &P, const PrimeV &pv_, PfCur *pf){
    const PrimeV pv = pv_regs(pv_);
    Col8Tw tw; col8_build(tw, j, P, pv);
    for(size_t i = i0; i < i1; ++i){
        V r[8], t[8];
        CR_PF(pf, 1);
        for(int k = 0; k < 8; ++k) r[k] = (k < L || !ZT) ? x[i + (size_t)k * e] : _mm512_setzero_si512();
        if constexpr(TAIL) for(int k = 0; k < 8; ++k) t[k] = r[k];
        itft8_r<8, L, ZT, 1>(r, tw, pv);
        for(int k = 0; k < L; ++k) x[i + (size_t)k * e] = r[k];
        if constexpr(TAIL){
            for(int k = 0; k < L; ++k) t[k] = r[k];            /* time values: recovered + known tail */
            x[i + (size_t)(L < 8 ? L : 0) * e] = fwd8_out<8, (L < 8 ? L : 0), (ZT ? (L < 8 ? L : 8) : 8), 1>(t, tw, pv);
        }
    }
}
template<int KF>
static void colpass_fwd_k(V *x, size_t e, size_t j, int kin, const Prime &P, const PrimeV &pv, PfCur *pf){
    switch(kin){
    case 1: colpass_fwd_t<KF, 1>(x, e, j, P, pv, pf); break;
    case 2: colpass_fwd_t<KF, 2>(x, e, j, P, pv, pf); break;
    case 3: colpass_fwd_t<KF, 3>(x, e, j, P, pv, pf); break;
    case 4: colpass_fwd_t<KF, 4>(x, e, j, P, pv, pf); break;
    case 5: colpass_fwd_t<KF, 5>(x, e, j, P, pv, pf); break;
    case 6: colpass_fwd_t<KF, 6>(x, e, j, P, pv, pf); break;
    case 7: colpass_fwd_t<KF, 7>(x, e, j, P, pv, pf); break;
    default: colpass_fwd_t<KF, 8>(x, e, j, P, pv, pf); break;
    }
}
static inline void colpass_fwd(V *x, size_t e, size_t j, int kf, int kin, const Prime &P, const PrimeV &pv, PfCur *pf){
    switch(kf){
    case 1: colpass_fwd_k<1>(x, e, j, kin, P, pv, pf); break;
    case 2: colpass_fwd_k<2>(x, e, j, kin, P, pv, pf); break;
    case 3: colpass_fwd_k<3>(x, e, j, kin, P, pv, pf); break;
    case 4: colpass_fwd_k<4>(x, e, j, kin, P, pv, pf); break;
    case 5: colpass_fwd_k<5>(x, e, j, kin, P, pv, pf); break;
    case 6: colpass_fwd_k<6>(x, e, j, kin, P, pv, pf); break;
    case 7: colpass_fwd_k<7>(x, e, j, kin, P, pv, pf); break;
    default: colpass_fwd_k<8>(x, e, j, kin, P, pv, pf); break;
    }
}
template<bool ZT, bool TAIL>
static inline void colpass_inv(V *x, size_t e, size_t i0, size_t i1, size_t j, int L, const Prime &P, const PrimeV &pv, PfCur *pf){
    if(i0 >= i1) return;
    switch(L){
    case 0: if constexpr(TAIL) colpass_inv_t<0, ZT, true>(x, e, i0, i1, j, P, pv, pf); break;
    case 1: colpass_inv_t<1, ZT, TAIL>(x, e, i0, i1, j, P, pv, pf); break;
    case 2: colpass_inv_t<2, ZT, TAIL>(x, e, i0, i1, j, P, pv, pf); break;
    case 3: colpass_inv_t<3, ZT, TAIL>(x, e, i0, i1, j, P, pv, pf); break;
    case 4: colpass_inv_t<4, ZT, TAIL>(x, e, i0, i1, j, P, pv, pf); break;
    case 5: colpass_inv_t<5, ZT, TAIL>(x, e, i0, i1, j, P, pv, pf); break;
    case 6: colpass_inv_t<6, ZT, TAIL>(x, e, i0, i1, j, P, pv, pf); break;
    case 7: colpass_inv_t<7, ZT, TAIL>(x, e, i0, i1, j, P, pv, pf); break;
    default: colpass_inv_t<8, ZT, false>(x, e, i0, i1, j, P, pv, pf); break;
    }
}

/* Forward: outputs [0,lout), mathematical inputs [lin,n) zero. For n>CR_BLK_MIN
 * only [lin,ceil(lin/(n/8))*(n/8)) needs physical zeroes: colpass_fwd<KIN>
 * never loads the later complete blocks. The small fallback reads the full tail. */
inline void tft_blk(V *x, size_t n, size_t j, size_t lout, size_t lin, const Prime &P, const PrimeV &pv, PfCur *pf = NULL){
    if(lout == 0) return;
    if(lout > n) lout = n;
    if(lin > n) lin = n;
    if(lin == 0){ for(size_t i = 0; i < lout; ++i) x[i] = _mm512_setzero_si512(); return; }
    if(n <= CR_BLK_MIN){ tft<0, 1>(x, n, j, lout, P, pv, pf); return; }
    if(lout == n && lin == n){ full<false, false, 1>((uint64_t *)x, n, j, P, pv, pf); return; }
    const size_t e = n >> 3;
    const int kf = (int)((lout + e - 1) / e), kin = (int)((lin + e - 1) / e);
    colpass_fwd(x, e, j, kf, kin, P, pv, pf);
    for(int k = 0; k < kf - 1; ++k) full<false, false, 1>((uint64_t *)(x + (size_t)k * e), e, 8 * j + (size_t)k, P, pv, pf);
    tft_blk(x + (size_t)(kf - 1) * e, e, 8 * j + (size_t)(kf - 1), lout - (size_t)(kf - 1) * e, e, P, pv, pf);
}
/* inverse: frequencies [0, l) of node (n, j) known; the tail [l, n) known in memory (zero when ZT) */
template<bool ZT>
void itft_blk(V *x, size_t n, size_t j, size_t l, const Prime &P, const PrimeV &pv, PfCur *pf = NULL){
    if(l == 0) return;
    if(l >= n){ cfinv<0, 1, 1>(x, n, j, P, pv.p, pv.p2, pv.pn, pf); return; }
    if(n <= CR_BLK_MIN){ itft<0, 1, 1, ZT>(x, n, j, l, P, pv, pf); return; }
    const size_t e = n >> 3;
    const int kf = (int)(l / e); const size_t r = l - (size_t)kf * e;
    for(int k = 0; k < kf; ++k) cfinv<0, 1, 1>(x + (size_t)k * e, e, 8 * j + (size_t)k, P, pv.p, pv.p2, pv.pn, pf);
    colpass_inv<ZT, true>(x, e, r, e, j, kf, P, pv, pf);              /* u for k < kf; partial tail y^kf */
    if(r){
        itft_blk<false>(x + (size_t)kf * e, e, 8 * j + (size_t)kf, r, P, pv, pf);
        colpass_inv<ZT, false>(x, e, 0, r, j, kf + 1, P, pv, pf);
    }
}
} // namespace sbn::v3::SBN3_P48_NS
