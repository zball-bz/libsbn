/* Imported from labs/cr/cr_engine.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_engine.hpp — the passes and drivers of the three arms.
 *
 *   handle build : rows(A) [→ block(spec) when hspec = 1]                → handle planes
 *   LIN (default): [xpose(Y) at T = 80] → rows(Y) → tile v4 (cr_tile.hpp: A col + Y col + ring
 *                  products + inverse col, row-major product planes) → fused inverse rows + Garner
 *                  + emit (cr_fuse.hpp);  M2 > 8192: block tile in place → irow → emit
 *   CYC          : rows(Y) → block(conv) → irow → emit [→ fold]
 *   TMP          : mrow(Y) → block(mid)  → orow → emit(reversed)
 *   The optimizations in force and every knob: docs_p48_engine.md.
 *
 * Planes are BLOCKED (cr_geom.hpp): block j = slots [j·TB, (j+1)·TB) of
 * every virtual row, contiguous. The block pass streams one block region
 * in (sequential read), transforms its TB columns one at a time in L2,
 * streams A's block region through the products (read exactly once) and
 * streams the block back out in whole-line NT units — the corner turn
 * costs no partial lines and no L3 re-reads (NOTES §12).
 *
 * Every pass is a team task body (sbn_for_fn shape); the drivers run
 * them through sbn_parallel_for / sbn_run_tasks (width wio; wio <= 1 or
 * a NULL team runs inline). Phase times are recorded in Prof. */
#include "backend/ntt_p48/emit.hpp"
#include "backend/ntt_p48/col.hpp"
#include "backend/ntt_p48/scratch_adapter.hpp"
#include <time.h>
#include <sched.h>
#include <atomic>


namespace sbn::v3::SBN3_P48_NS {

inline double now_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
struct Prof { double rows, tile, irow, emit; };
#if defined(CR_TILE_PROF) || defined(CR_ROWS_PROF)
#include <x86intrin.h>
/* fenced timestamp: an unfenced __rdtsc is scheduled freely by the compiler
 * and the core, which mis-attributed whole phases (NOTES §12.8) */
__attribute__((always_inline)) inline uint64_t cr_tsc(void){ unsigned lo, hi, a, b, c, d; __asm__ __volatile__("cpuid\n\trdtscp\n\tmov %%eax, %0\n\tmov %%edx, %1\n\tcpuid" : "=r"(lo), "=r"(hi), "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0) : "memory"); (void)a; (void)b; (void)c; (void)d; return ((uint64_t)hi << 32) | lo; }
#endif
#ifdef CR_TILE_PROF
struct TpCnt { uint64_t gather, colf, lanes, leaf, coli, scatter, slots, vecs;  uint64_t pfl; };
static TpCnt g_tp[128];             /* [0,64) product tile, [64,128) spec pass */
#define TP_T(v) uint64_t v = cr_tsc()
#define TP_ADD(acc, a, b) g_tp[w + (SPEC ? 64 : 0)].acc += (b) - (a)
#else
#define TP_T(v)
#define TP_ADD(acc, a, b)
#endif

#ifdef CR_ROWS_PROF
struct RpCnt { uint64_t dec, tft, pack, rows, gat; };
static RpCnt g_rp[64];
#define RP_T(v) uint64_t v = cr_tsc()
#define RP_ADD(acc, a, b) g_rp[w < 0 || w >= 64 ? 0 : w].acc += (b) - (a)
#else
#define RP_T(v)
#define RP_ADD(acc, a, b)
#endif
/* itft interior full nodes: 0 recursive normalized radix-2 (cfinv_rec), 1 radix-8 DFS with the scale
 * fused into its top moth (§28). The old default 0 below M2 = 4096 came from a kbench before the
 * scale fusion and the leaf fix; the fused pass at (2048, 1024) / (1024, 2048) now reads itft per
 * row 94k → 57k / 203k → 95k cycles with 1 (§28.7) */
enum { FAST_INV = CR_FAST_INV };

struct CcxGate;
struct alignas(64) PassCounts {
    uint64_t row_forward=0,row_mid=0,column_forward=0,column_inverse=0,row_inverse=0,row_transpose_forward=0,leaf_products=0;
};
struct Ctx {
    PassCounts *counts=nullptr;
    unsigned frontier[2]{};
    const uint8_t *hp1[NP]{}; uint8_t *fp1[NP]{};
    const uint64_t *leaf_k[2]{}, *leaf_rec[2]{};
    CcxGate *gates;
    const Plan *pl;
    const Primes *PS;
    /* operand (row passes) */
    const uint64_t *a; size_t an; size_t natv;
    const uint64_t *dsc;                  /* build side, wide codec: the plane scale folded into A's decode (§32) */
    unsigned codec_mode;
    int reversed; size_t Zv;
    /* planes */
    const uint8_t *hp[NP]; size_t hbs;    /* handle planes + block stride  */
    int smalltw;                          /* block: per-column materialized twiddles (tower prefix only) */
    int rpf;                              /* rows: decode prefetch distance in slots */
    int ipf, tpf;                         /* software prefetch cursors: irow / tile, lines per step (0 = off); CR_IPF, CR_TPF */
    uint8_t *fp[NP];                      /* fresh / result planes      */
    int nts;
    unsigned fused_start_skew_us=40; // occupies the original padding before gen
    uint64_t gen;                         /* helper-tile pair handshake generation */
    const uint8_t *xp; size_t xrs, xs; size_t xp_row0 = 0;    /* transposed operand (E1, NOTES §20): row r's pieces contiguous, xs-byte slots; NULL = gather */
    uint8_t *op[NP]; size_t obs;          /* product planes at SLOTO (fused path, §21 hybrid); NULL = in place at SLOT */
    int prm; unsigned writeback_a=0; size_t rms;                  /* CR_PRM (NOTES §26): the product planes are ROW-MAJOR (row vr at vr·rms, slot cv at
                                           * cv·SLOTO): the tile scatters its block image row by row through a line assembler
                                           * (write side), the fused pass reads each row as one sequential stream (no gather) */
};

/* ---- E1: transposed operand (NOTES §20) ----------------------------------
 * The rows pass gathers row r's pieces (natural vector v = cv·C + r, T bytes
 * each) from 688 KB apart: latency-bound (~0.5M TSC per row at 2^28). With
 * CR_XPOSE=1 a blocked transpose first lays the operand out as XP[r][cv]:
 * 96-B slots holding the piece's limb-aligned window (its 8-B phase v & 1
 * when T ≡ 4 mod 8), row stride xrs (a 64-B multiple). Tiles of XTR rows ×
 * XTC pieces: XTC sequential input runs of XTR·T bytes, XTR output runs of
 * XTC·96 = 1536 B (24 whole NT lines). The rows pass then decodes straight
 * from its row's slots (a 300 KB sequential stream). LIN, natural order. */
enum { XS_DEF = 96, XTC_DEF = 16 };
struct XposeCtx { const uint64_t *a; size_t an, natv, T, C, ncv, xrs, tr, xs, xtc; uint8_t *xp; size_t row0=0; bool cached_store=false; };
static inline int xp_phase(size_t v, size_t T){ return (int)((v & 1) & ((T >> 2) & 1)); }   /* piece start mod 8 = 4·phase */
static void xpose_fn(void *c_, uint64_t lo, uint64_t hi, int w, scratch *ws){
    (void)w; (void)ws;
    const XposeCtx &x = *(const XposeCtx *)c_;
    const size_t PB = x.T, tr = x.tr;
    const uint8_t *in = (const uint8_t *)x.a; const size_t inb = x.an * 8;
    const size_t runb = tr * PB + 16, XS = x.xs, XTC = x.xtc;
    uint8_t *run = SALLOC(ws, uint8_t, ((runb + 63) & ~(size_t)63) * XTC);
    uint8_t *obuf = SALLOC(ws, uint8_t, ((XTC * XS + 63) & ~(size_t)63) + 64);
    for(uint64_t it = lo; it < hi; ++it){
        const size_t r0 = (size_t)it * tr;
        for(size_t cv0 = 0; cv0 < x.ncv; cv0 += XTC){
            const size_t tc = cv0 + XTC <= x.ncv ? XTC : x.ncv - cv0;
            for(size_t c = 0; c < tc; ++c){
                uint8_t *rc = run + c * ((runb + 63) & ~(size_t)63);
                const size_t v0 = (cv0 + c) * x.C + r0;
                const size_t b0 = v0 * PB - 4 * (size_t)xp_phase(v0, PB);
                if(b0 + runb <= inb) memcpy(rc, in + b0, runb);
                else { const size_t have = b0 < inb ? inb - b0 : 0; if(have) memcpy(rc, in + b0, have); memset(rc + have, 0, runb - have); }
            }
            for(size_t r = 0; r < tr; ++r){
                for(size_t c = 0; c < tc; ++c){
                    const uint8_t *rc = run + c * ((runb + 63) & ~(size_t)63);
                    const size_t v0 = (cv0 + c) * x.C + r0, v = v0 + r;
                    const size_t o = r * PB + 4 * (size_t)xp_phase(v0, PB) - 4 * (size_t)xp_phase(v, PB);
                    memcpy(obuf + c * XS, rc + o, XS);
                }
                uint8_t *dst = x.xp + (r0 + r - x.row0) * x.xrs + cv0 * XS;
                const size_t nb = tc * XS, nl = nb / 64;
                for(size_t l = 0; l < nl; ++l) {
                    if(x.cached_store) _mm512_store_si512((__m512i *)(dst+64*l),_mm512_load_si512(obuf+64*l));
                    else _mm512_stream_si512((__m512i *)(dst+64*l),_mm512_load_si512(obuf+64*l));
                }
                if(nb & 63) _mm512_mask_storeu_epi8(dst + 64 * nl, (__mmask64)((1ull << (nb & 63)) - 1), _mm512_load_si512(obuf + 64 * nl));
            }
        }
    }
    _mm_sfence();

}
template<int L, int MID>
void rows_fn(void *c_, uint64_t lo, uint64_t hi, int w, scratch *ws){
    (void)w;
#ifdef CR_ROWS_PROF
    if(w < 0 || w >= 64) w = 0;
#endif
    const Ctx &c = *(const Ctx *)c_;
    if(c.counts){ auto &n=c.counts[w]; (MID?n.row_mid:n.row_forward)+=(hi-lo)*CR_ROWW*NP; }
    const Plan &pl = *c.pl;
    const Primes &PS = *c.PS;
    constexpr size_t Lg = L == 8 ? 1 : (size_t)L;
    constexpr size_t RW = CR_ROWW;                                  /* rows per task */
    const size_t M2 = pl.M2, lbv = pl.lbv, lbw = pl.lbw, C = pl.C;
    const size_t TB8 = 8 * (size_t)pl.T;
    const size_t xsl = pl.T > 192 ? 16 : pl.T > 88 ? 8 : 0;                           /* the wide codec reads 3 loads (24 limbs) per vector */
    const size_t BLc = ((32 + TB8 + 63) >> 6) + 2 + xsl;           /* limbs copied per vector (reversed) */
    const size_t GBc = c.reversed ? RW * Lg * BLc : ((32 + TB8 * RW * Lg + 63) >> 6) + 2 + xsl;   /* per group (RW rows) */
    const size_t pieces_words=((size_t(pl.T)+47)/48)*8;
    const size_t ib_words=c.codec_mode==3 && pl.T>88 && pieces_words>GBc?pieces_words:GBc;
    uint64_t *ib = SALLOC(ws, uint64_t, M2 * ib_words + 16);
    /* One independently decoded prime at a time; live row buffer only. */
    V *row = SALLOC(ws, V, RW * Lg * M2);        /* element cv of sub-row s: row[(s·M2 + cv)·RW + w] */
    LineAsm la[NP];
    for(int q = 0; q < NP; ++q) lasm_init(la[q], ws, pl.nblk);
    const Pk52 K = pk52_mk();
    Dec d;
    dec_init(d, ib, pl, PS, c.dsc);
    Pick<L == 8 ? 1 : L> pick;
    const V RT = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    /* live natural vectors: reversed → Zv, else natv */
    const size_t live = c.reversed ? c.Zv : c.natv;
    const size_t an = c.an;
    const int xp_on = c.xp != NULL && Lg == 1 && RW == 1 && !c.reversed;      /* E1: transposed operand */
    for(uint64_t it = lo; it < hi; ++it){
        const size_t r0 = (size_t)it * RW;
        RP_T(t0);
        /* 1. gather the rows' pieces: group cv covers natural vectors [(cv·C + r0)·Lg, (cv·C + r0 + RW)·Lg) */
        size_t cv = 0;
        for(; cv < M2; ++cv){
            const size_t g0 = cv * C + r0;
            if(g0 * Lg >= live) break;
            if(xp_on) continue;                                            /* the row's slots are contiguous in c.xp */
            {   /* prefetch the piece CR_RPF (default 4) slots ahead */
                const size_t gp = (cv + (size_t)c.rpf) * C + r0;
                if(gp * Lg < live){
                    const size_t u = c.reversed ? c.Zv - 1 - gp * Lg : gp * Lg;
                    const uint64_t *pa = c.a + ((TB8 * u) >> 6);

                    _mm_prefetch((const char *)pa, _MM_HINT_T0);
                    _mm_prefetch((const char *)(pa + 8), _MM_HINT_T0);
                    if(RW > 1 || TB8 > 1024) _mm_prefetch((const char *)(pa + 16), _MM_HINT_T0);
                    
                }
            }
            uint64_t *dst = ib + cv * GBc;
            if(!c.reversed){
                const size_t l0 = (TB8 * g0 * Lg) >> 6;
                const size_t avail = l0 < an ? an - l0 : 0, n = avail < GBc ? avail : GBc;
                if(n == GBc) memcpy(dst, c.a + l0, GBc * 8);
                else { if(n) memcpy(dst, c.a + l0, n * 8); memset(dst + n, 0, (GBc - n) * 8); }
            }else{
                for(size_t j = 0; j < RW * Lg; ++j){
                    const size_t u = g0 * Lg + j;
                    if(u >= c.Zv){ memset(dst + j * BLc, 0, BLc * 8); continue; }
                    const size_t l0 = (TB8 * (c.Zv - 1 - u)) >> 6;
                    const size_t avail = l0 < an ? an - l0 : 0, n = avail < BLc ? avail : BLc;
                    if(n) memcpy(dst + j * BLc, c.a + l0, n * 8);
                    if(n < BLc) memset(dst + j * BLc + n, 0, (BLc - n) * 8);
                }
            }
        }
        const size_t ncv = cv;
        const bool prepared=c.codec_mode==3 && xp_on && d.t8;
        V *pieces=reinterpret_cast<V *>(ib);
        if(prepared)for(size_t v=0;v<ncv;++v){
            const auto *src=reinterpret_cast<const uint64_t *>(c.xp+(r0-c.xp_row0)*c.xrs+v*c.xs);
            const V L0=_mm512_loadu_si512(src),L1=_mm512_loadu_si512(src+8),L2=d.ld3?_mm512_loadu_si512(src+16):L0;
            V L3=L0;
#if CR_NP>8
            if(d.fourth.load)L3=_mm512_loadu_si512(src+24);
#endif
            for(int k=0;k<d.npc;++k)pieces[v*d.npc+k]=dec8_piece(d,k,L0,L1,L2,L3);
        }
        RP_T(t1);
        
        /* 2. per prime */
        for(int q = 0; q < NP; ++q){
            const Prime &P = PS.P[q]; const PrimeV &pv = PS.V[q];
            RP_T(t2);
            for(size_t v = 0; v < ncv; ++v){
                const size_t g0 = v * C + r0;
                const uint64_t *pb = ib + v * GBc;
                for(size_t rw = 0; rw < RW; ++rw){
                    const size_t g = g0 + rw;                              /* this row's group */
                    const int dead = g * Lg >= live;                       /* (the pair's second row past the operand) */
                    if constexpr(Lg == 1){
                        V o;
                        if(dead) o = _mm512_setzero_si512();
                        else if(prepared) o = dec8_prepared(d,pieces+v*d.npc,q);
                        else if(xp_on) o = dec1q(d, (const uint64_t *)(c.xp + (r0-c.xp_row0) * c.xrs + v * c.xs), q, xp_phase(g0, (size_t)pl.T));
                        else if(!c.reversed){ const size_t rel = ((TB8 * g0) & 63) + TB8 * rw; o = dec1q(d, pb + (rel >> 6), q, (int)((rel & 63) >> 5)); }
                        else{ const size_t u = g; o = dec1q(d, pb + rw * BLc, q, u < c.Zv ? dec_var(d, c.Zv - 1 - u) : 0); o = _mm512_permutexvar_epi64(RT, o); }
                        row[v * RW + rw] = o;
                    }else{
                        V nv[Lg];
                        for(size_t j = 0; j < Lg; ++j){
                            V o;
                            if(dead) o = _mm512_setzero_si512();
                            else if(!c.reversed){
                                const size_t rel = ((TB8 * g0 * Lg) & 63) + TB8 * (rw * Lg + j);
                                o = dec1q(d, pb + (rel >> 6), q, (int)((rel & 63) >> 5));
                            }else{
                                const size_t u = g * Lg + j;
                                o = dec1q(d, pb + (rw * Lg + j) * BLc, q, u < c.Zv ? dec_var(d, c.Zv - 1 - u) : 0);
                                o = _mm512_permutexvar_epi64(RT, o);
                            }
                            nv[j] = o;
                        }
                        for(size_t s = 0; s < Lg; ++s) row[(s * M2 + v) * RW + rw] = pick.get(nv, (int)s);
                    }
                }
            }
            for(size_t s = 0; s < Lg; ++s)
                for(size_t z = ncv * RW; z < M2 * RW; ++z) row[s * M2 * RW + z] = _mm512_setzero_si512();
            RP_T(t3);
            for(size_t s = 0; s < Lg; ++s){
                V *rw = row + s * M2 * RW;
                if constexpr(!MID){
                    if(pl.full) full<false,false,RW>((uint64_t *)rw, M2, 0, P, pv);
                    else if(RW == 1 && M2 > CR_BLK_MIN) tft_blk(rw, M2, 0, lbv, ncv, P, pv);   /* §28 blocked forward, zero input beyond ncv */
                    else tft<0,RW>(rw, M2, 0, lbv, P, pv);
                }else{
                    if(pl.mrow_unnorm) full<false,true,RW>((uint64_t *)rw, M2, 0, P, pv);
                    else itft<1, FAST_INV, RW>(rw, M2, 0, lbv, P, pv);
                }
                for(size_t b = lbv * RW; b < lbw * RW; ++b) rw[b] = _mm512_setzero_si512();
            }
            RP_T(t4);
            for(size_t rw = 0; rw < RW; ++rw)
                for(size_t s = 0; s < Lg; ++s)
                    st52blk(la[q], c.fp[q], pl, (r0 + rw) * Lg + s, row + s * M2 * RW + rw, RW, c.nts, K, pv);
            RP_T(t5);
            RP_ADD(dec, t2, t3); RP_ADD(tft, t3, t4); RP_ADD(pack, t4, t5);
        }
        RP_ADD(gat, t0, t1);
#ifdef CR_ROWS_PROF
        g_rp[w < 0 || w >= 64 ? 0 : w].rows += RW;
#endif
    }
    for(int q = 0; q < NP; ++q) lasm_flush(la[q]);
    if(c.nts) _mm_sfence();
}

/* ---- column/lane transforms of one gathered column (Lg·C vectors) ---- */
/* forward column+lane transform (build-side spec pass and A-side of a
 * non-hspec tile): colpass CF per s, then lane levels fwd per 8-column
 * block (leaf-L only). T (2C entries scratch) → small-table columns. */
template<int L>
__attribute__((always_inline)) inline
void col_fwd_all(V *xc, size_t C, size_t Cs, size_t b, const Prime &P, const PrimeV &pv, uint64_t *T = NULL, size_t lgC = 0, PfCur *pf = NULL){
    constexpr size_t Lg = L == 8 ? 1 : (size_t)L;
    if(T){ col_tw_build<false>(T, T + C, P, lgC, C, b, pv); colpass_t<false>((uint64_t *)xc, C, T, pv, &P, pf); }
    else for(size_t s = 0; s < Lg; ++s) colpass<false,false>((uint64_t *)(xc + s * Cs), C, b, P, pv);
    if constexpr(Lg > 1){
        /* leaf-L spectrum is kept TRANSPOSED per 8-column block */
        for(size_t a0 = 0; a0 < C; a0 += 8){
            LaneT lt; lane8_build<false>(lt, C * b + a0, P, pv);
            for(size_t s = 0; s < Lg; ++s) lane8_in<false>(xc + s * Cs + a0, lt, pv);
        }
    }
}

/* ---- product side job (native lanes): 8 columns per step, A streamed from
 * the spectrum region; driven from the kernel hooks of the next slot's
 * forward and the first inverse so that the DRAM stream overlaps ALU
 * (NOTES §12.11) ---- */
struct ProdJob {
    V *y; const uint8_t *ab; const V *ac; size_t sl; const uint64_t *r8, *ir8;
    size_t a, C; const Prime *P; const PrimeV *pv; int mid; Pk52 K;
};
static void prod_step(void *ctx){
    ProdJob &J = *(ProdJob *)ctx;
    if(J.a >= J.C) return;
    const size_t a0 = J.a; J.a += 8;
    const V *ap;
    V av[8];
    if(J.ac) ap = J.ac + a0;                              /* A's column in cache (fresh path) */
    else{
        { static const int d = 4;   /* steps ahead: the stream is too sparse for the hardware prefetcher */
          if(d && a0 + 8 * (size_t)d < J.C){ const uint8_t *p = J.ab + (a0 + 8 * (size_t)d) * (TB * SLOT); const size_t o0 = (J.sl * SLOT) & ~(size_t)63, o1 = (J.sl * SLOT + SLOT - 1) & ~(size_t)63;
              for(size_t k = 0; k < 8; ++k){ _mm_prefetch((const char *)(p + k * (TB * SLOT) + o0), _MM_HINT_T0); if(o1 != o0) _mm_prefetch((const char *)(p + k * (TB * SLOT) + o1), _MM_HINT_T0); } } }
        for(size_t k = 0; k < 8; ++k) av[k] = ld52(J.ab + (a0 + k) * (TB * SLOT) + J.sl * SLOT, J.K);
        ap = av;
    }
    if(!J.mid) conv8x8(J.y + a0, ap, J.r8 + a0, *J.pv);
    else conv8x8T(J.y + a0, ap, J.r8 + a0, J.ir8 + a0, *J.pv);
}
static inline void prod_finish(ProdJob &J){ while(J.a < J.C) prod_step(&J); }
/* sweep side job: column sl of a block region → dst[r], 8 rows per step (Lg = 1) */
/* Product side work overlaps the next column transform. */
struct SideJobs { ProdJob *pr; };

/* ---- per-CCX admission gate (NOTES §17 O1) --------------------------------
 * Lock-stepped cores share the CCX's read link (7 GB/s each with 8 sweepers
 * vs 45 GB/s alone). A FIFO ticket gate lets at most `cap` cores per CCX run
 * a memory phase at once: the sweeps de-phase and, the cycles being equal,
 * the stagger is stable (a start skew drifts back into lock-step). CCX of
 * rank w = (w & 15) >> 3 (CONTIG placement: ranks 0-7 = one CCX). */
struct CcxGate { alignas(64) std::atomic<uint32_t> next; alignas(64) std::atomic<uint32_t> done; };

static inline int cr_ccx(int w){ return (w & 15) >> 3; }
static inline void gate_acquire(CcxGate *gates, int ccx, uint32_t cap){
    CcxGate &g = gates[ccx & 3];
    const uint32_t t = g.next.fetch_add(1, std::memory_order_relaxed);
    while((int32_t)(g.done.load(std::memory_order_acquire) + cap - t) <= 0) _mm_pause();
}
static inline void gate_release(CcxGate *gates, int ccx){ gates[ccx & 3].done.fetch_add(1, std::memory_order_release); }
static void side_step2(void *ctx){
    SideJobs &J = *(SideJobs *)ctx;
    if(J.pr && J.pr->a < J.pr->C) prod_step(J.pr);
}
static inline void side_init(PfCur &c, SideJobs &j, int every){
    pf_init(c, NULL, 64, 1, 0, 0); c.side = side_step2; c.sctx = &j; c.side_every = (uint32_t)every;
}

/* ---- block pass: item = (q, blk) --------------------------------------
 * SPEC = 1: handle build, forward columns (+ lane levels) in place.
 * SPEC = 0: the product tile, per slot of the block:
 *      Y column in (L2) → CF(Y) → A column in (L2) [→ CF(A) when the handle
 *      holds rows, hspec = 0] → products in cache → CI(Y)
 *   then the block's units out. A's column is swept into a 512 KB buffer
 *   right before the products (the OOC pass structure: A col, B col, conv,
 *   icol in one cache residency) instead of being streamed from DRAM
 *   inside the product loop (NOTES §12.10).
 * Column buffers yc[(sl·Lg + s)·Cs + r] (slot sl of the block, virtual
 * sub-row s, row r); Cs = C + 8 keeps the planes off one L1 set. */
template<int L, int MID, int SPEC>
void block_fn(void *c_, uint64_t lo, uint64_t hi, int w, scratch *ws){
    (void)w;
    const Ctx &c = *(const Ctx *)c_;
    if constexpr(SPEC) if(c.counts)c.counts[w].column_forward+=(hi-lo)*TB;
#ifdef CR_TILE_PROF
    if(w < 0 || w >= 64) w = 0;
#endif
    const Plan &pl = *c.pl;
    const Primes &PS = *c.PS;
    constexpr size_t Lg = L == 8 ? 1 : (size_t)L;
    const size_t C = pl.C, nrows = pl.nrows, nblk = pl.nblk;
    const size_t Cs = C + 8;
    const int smt = (L == 8 && c.smalltw);
    V *yc = SALLOC(ws, V, (size_t)TB * Lg * Cs);
    V *ac = SPEC ? NULL : SALLOC(ws, V, (size_t)TB * Lg * Cs);       /* A's column per slot (fresh path) */
    uint64_t *TLs = smt ? SALLOC(ws, uint64_t, (size_t)TB * 2 * C) : NULL;   /* TL0, TL1 per slot (the products run late) */
    /* pipelined products (Lg = 1, A streamed): slot sl's products run as a
     * side job of slot sl+1's forward (the last slot's during the first
     * inverse); three product hooks are served per transform step */
    static const int side_k = 3;
    /* sweep side jobs (next slot's Y column / this slot's A column swept during
     * the forward): their 512 KB of stores evict the column being transformed
     * from L2 — tile 398 → 440 ms at 2^28; this schedule was retired. */
    uint64_t *Tsc = smt ? SALLOC(ws, uint64_t, 4 * C) : NULL;          /* T0, T1; A's forward tables */
    const Pk52 K = pk52_mk();
    /* A's column materialized in cache before the products (acol = 1: required
     * when the handle holds rows, hspec = 0) or streamed from the spectrum
     * region inside the product loop (acol = 0, cached handles: the DRAM stream
     * overlaps the conv8 ALU; labs/cr/NOTES.md §12.10). */
    const int acol = SPEC ? 0 : !pl.hspec;
    if(!SPEC && !pl.hspec && !acol) ::sbn::v3::fatal(SBN3_FATAL_MATH, "p48 column frontier");
    for(; lo < hi; ++lo){
        const int q = (int)(lo / nblk);
        const size_t blk = (size_t)(lo % nblk);
        const Prime &P = PS.P[q]; const PrimeV &pv = PS.V[q];
        uint8_t *yb = c.fp[q] + blk * pl.bstride;
        const uint8_t *ab = SPEC ? NULL : c.hp[q] + blk * c.hbs;      /* A: spectrum (hspec) or rows */
        /* prefetch cursors for the next item: its Y region, then (optionally) its A region */
        PfCur pfy; pfy.n = 0; pfy.next = NULL;
        if(c.tpf && lo + 1 < hi){
            const int q1 = (int)((lo + 1) / nblk); const size_t blk1 = (size_t)((lo + 1) % nblk);
            const size_t rl = (nrows * (TB * SLOT) + 63) / 64;
            pf_init(pfy, c.fp[q1] + blk1 * pl.bstride, 64, 1, rl, c.tpf);
            
        }
        PfCur *pp = pfy.n ? &pfy : NULL;
        /* The cursor drains in the LAST slot's product + inverse. Earlier
         * prefetches were evicted before the next block's sweep-in; the
         * historical experiment doubled Y-in time (7 → 14). */
        /* Cached A is streamed by the product loop. The historical early-A
         * prefetch variant has been retired; this cursor serves the next
         * block's Y region only. */
        
        auto sweep_in = [&](V *dst, const uint8_t *rb, size_t sl){   /* column sl of a block region → dst[s·Cs + r] */
            for(size_t vr = 0; vr < nrows; ++vr){
                const size_t r = vr / Lg, s = vr - r * Lg;
                dst[s * Cs + r] = ld52(rb + vr * (TB * SLOT) + sl * SLOT, K);
            }
        };
        /* Pipelined schedule (Lg = 1, small tables): column sweeps are serial;
         * this slot's A column is transformed when fresh. Slot sl's
         * products ride slot sl+1's forward, the last slot's the first
         * inverse; the last inverse carries the next block's prefetch cursor */
    #ifdef CR_TILE_PROF
    { static thread_local int said = -1; if(said != (int)SPEC){ said = (int)SPEC; fprintf(stderr, "%s worker %d on cpu %d\n", SPEC ? "spec" : "tile", w, sched_getcpu()); } }
#endif
    const int pipe = (!SPEC && Lg == 1 && smt && ab);
        ProdJob jobs[TB]; SideJobs sj[TB + 1]; PfCur sidecur[TB + 1];
        for(size_t sl = 0; sl < TB; ++sl){
            const size_t b = blk * TB + sl;
            V *y = yc + sl * Lg * Cs;
            TP_T(s0);
            sweep_in(y, yb, sl);
            TP_T(s1); TP_ADD(gather, s0, s1);
            if constexpr(SPEC){
                col_fwd_all<L>(y, C, Cs, b, P, pv, smt ? Tsc : NULL, pl.lgC, pp);
                TP_T(s2); TP_ADD(colf, s1, s2);
                continue;
            }
            /* Y forward (CF, or CI^T for the mid) */
            uint64_t *T0 = NULL, *T1 = NULL, *TL0 = NULL, *TL1 = NULL;
            if(smt){
                T0 = Tsc; T1 = Tsc + C; TL0 = TLs + sl * 2 * C; TL1 = TL0 + C;
                col_tw_build<false>(T0, TL0, P, pl.lgC, C, b, pv);
                col_tw_build<true>(T1, TL1, P, pl.lgC, C, b, pv);
                PfCur *pfw = NULL;
                if(pipe){
                    sj[sl].pr = NULL;
                    
                    if(sl > 0) sj[sl].pr = &jobs[sl - 1];                                          /* previous slot's products */
                    if(sj[sl].pr){ side_init(sidecur[sl], sj[sl], side_k); pfw = &sidecur[sl]; }
                }
                if constexpr(!MID) colpass_t<false>((uint64_t *)y, C, T0, pv, &P, pfw);
                else               colpass_t<false>((uint64_t *)y, C, T1, pv, &P, pfw);
                if(pipe){
                    TP_T(f0);
                    if(acol){                              /* fresh: serial A column sweep, then its forward */
                        sweep_in(ac + sl * Cs, ab, sl);
                        col_fwd_all<L>(ac + sl * Cs, C, Cs, b, P, pv, smt ? Tsc + 2 * C : NULL, pl.lgC, NULL);
                    }
                    if(sl > 0) prod_finish(jobs[sl - 1]);
                    TP_T(f1); TP_ADD(leaf, f0, f1);
                }
            }else
            for(size_t s = 0; s < Lg; ++s){
                if constexpr(!MID) colpass<false,false>((uint64_t *)(y + s * Cs), C, b, P, pv);
                else               colpass<false,true>((uint64_t *)(y + s * Cs), C, b, P, pv);
            }
            TP_T(s2); TP_ADD(colf, s1, s2);
            if(pipe){                                      /* queue this slot's products; run the inverses at the end */
                ProdJob &J = jobs[sl];
                J.y = y; J.ab = ab; J.sl = sl; J.r8 = TL0; J.ir8 = TL1; J.a = 0; J.C = C; J.P = &P; J.pv = &pv; J.mid = MID; J.K = K;
                J.ac = acol ? ac + sl * Cs : NULL;
                continue;
            }
            /* A's column in (the region is read once per slot; the second time from L3) */
            if(acol){
                sweep_in(ac + sl * Lg * Cs, ab, sl);
                if(!pl.hspec) col_fwd_all<L>(ac + sl * Lg * Cs, C, Cs, b, P, pv, smt ? Tsc + 2 * C : NULL, pl.lgC, NULL);
            }
            TP_T(s3); TP_ADD(lanes, s2, s3);
            /* products: A from the cached column, or streamed from the spectrum region */
            PfCur *ppl = (sl == TB - 1) ? pp : NULL;
            if constexpr(Lg == 1){
                /* 8 columns at a time in the transposed form (conv8x8: no
                 * in-lane shuffles; the old conv8 comparison is archived in labs/cr) */
                for(size_t a0 = 0; a0 < C; a0 += 8){
                    if(ppl) pf_step(ppl, 8);
                    V av[8]; const V *ap = ac + sl * Cs + a0;
                    if(!acol){ for(size_t k = 0; k < 8; ++k) av[k] = ld52(ab + (a0 + k) * (TB * SLOT) + sl * SLOT, K); ap = av; }
                    /* leaf roots w[C·b + a0 + k]: the slot's materialized TL tables, or the tower */
                    const V r8 = smt ? _mm512_loadu_si512(TL0 + a0) : tower8<false>(P, C * b + a0);
                    if constexpr(!MID) conv8x8(y + a0, ap, r8, pv);
                    else conv8x8T(y + a0, ap, r8, smt ? _mm512_loadu_si512(TL1 + a0) : tower8<true>(P, C * b + a0), pv);
                    
                }
            }else{
                /* per 8-column block, in the TRANSPOSED lane layout (A's
                 * spectrum is stored transposed; a fresh A was transposed by
                 * col_fwd_all above); streamed A: the block's 8·Lg vrows of
                 * this slot are gathered into a small local array */
                for(size_t a0 = 0; a0 < C; a0 += 8){
                    if(ppl) pf_step(ppl, 2);
                    V acb[Lg * 8]; const V *A8 = ac + sl * Lg * Cs + a0; size_t astr = Cs;
                    if(!acol){
                        for(size_t k = 0; k < 8; ++k)
                            for(size_t s = 0; s < Lg; ++s) acb[s * 8 + k] = ld52(ab + ((a0 + k) * Lg + s) * (TB * SLOT) + sl * SLOT, K);
                        A8 = acb; astr = 8;
                    }
                    LaneT lin, lout;
                    if constexpr(!MID){ lane8_build<false>(lin, C * b + a0, P, pv); lane8_build<true>(lout, C * b + a0, P, pv); }
                    else              { lane8_build<true>(lin, C * b + a0, P, pv); lane8_build<false>(lout, C * b + a0, P, pv); }
                    for(size_t s = 0; s < Lg; ++s) lane8_in<false>(y + s * Cs + a0, lin, pv);
                    V wr[8], wc[8];
                    roots8<false>(wr, wc, C * b + a0, P, pv);
                    if constexpr(!MID){
                        for(int t = 0; t < 8; ++t)
                            convL<L>(y + a0 + t, Cs, A8 + t, astr, wc[t], wr[t], pv);
                    }else{
                        V wir[8], wic[8];
                        roots8<true>(wir, wic, C * b + a0, P, pv);
                        for(int t = 0; t < 8; ++t)
                            convLT<L>(y + a0 + t, Cs, A8 + t, astr, wc[t], wr[t], wic[t], wir[t], pv);
                    }
                    for(size_t s = 0; s < Lg; ++s) lane8_out<true>(y + s * Cs + a0, lout, pv);
                }
            }
            TP_T(s4); TP_ADD(leaf, s3, s4);
            /* Y inverse */
            if(smt){
                if constexpr(!MID){ col_tw_build<true>(Tsc, Tsc + 2 * C, P, pl.lgC, C, b, pv); }
                else              { col_tw_build<false>(Tsc, Tsc + 2 * C, P, pl.lgC, C, b, pv); }
                colpass_t<true>((uint64_t *)y, C, Tsc, pv, &P, ppl);
            }else
            for(size_t s = 0; s < Lg; ++s){
                if constexpr(!MID) colpass<true,true>((uint64_t *)(y + s * Cs), C, b, P, pv);
                else               colpass<true,false>((uint64_t *)(y + s * Cs), C, b, P, pv);
            }
            TP_T(s5); TP_ADD(coli, s4, s5);
        }
        if(pipe){
            /* inverses: slot 0's carries the last slot's products; the last
             * slot's carries the next block's prefetch cursor */
            for(size_t sl = 0; sl < TB; ++sl){
                const size_t b = blk * TB + sl;
                V *y = yc + sl * Lg * Cs;
                TP_T(v0);
                if constexpr(!MID){ col_tw_build<true>(Tsc, Tsc + 2 * C, P, pl.lgC, C, b, pv); }
                else              { col_tw_build<false>(Tsc, Tsc + 2 * C, P, pl.lgC, C, b, pv); }
                PfCur *pfi = NULL;
                if(sl == 0){
                    sj[TB].pr = &jobs[TB - 1];
                    side_init(sidecur[TB], sj[TB], side_k);
                    pfi = &sidecur[TB];
                }else if(sl == TB - 1) pfi = pp;
                if(sl == 0 && TB == 1){ prod_finish(jobs[0]); pfi = pp; }
                colpass_t<true>((uint64_t *)y, C, Tsc, pv, &P, pfi);
                if(sl == 0) prod_finish(jobs[TB - 1]);
                TP_T(v1); TP_ADD(coli, v0, v1);
            }
        }
#ifdef CR_TILE_PROF
        g_tp[w + (SPEC ? 64 : 0)].slots += TB; g_tp[w + (SPEC ? 64 : 0)].vecs += TB * Lg * C;
#endif
        TP_T(t5);
        /* sweep-out: whole-line units of UR virtual rows × TB slots */
        for(size_t vr0 = 0; vr0 < nrows; vr0 += UR){
            alignas(64) uint8_t unit[16 * SLOT];
            for(size_t u = 0; u < UR; ++u){
                const size_t vr = vr0 + u, r = vr / Lg, s = vr - r * Lg;
                for(size_t sl = 0; sl < TB; ++sl) st52(unit + (u * TB + sl) * SLOT, yc[(sl * Lg + s) * Cs + r], K, pv);
            }
            st_lines(yb + vr0 * (TB * SLOT), unit, 16 * SLOT, c.nts);
        }
        TP_T(t6);
        TP_ADD(scatter, t5, t6);
    }
    if(c.nts) _mm_sfence();
}

} // namespace sbn::v3::SBN3_P48_NS
#include "backend/ntt_p48/tile.hpp"
#include "backend/ntt_p48/fuse.hpp"
namespace sbn::v3::SBN3_P48_NS {

/* ---- inverse / transposed-forward row pass: item = (q, IROW_G-row group) ---
 * The group's rows form whole-line units in every block (IROW_G·TB·SLOT is
 * a multiple of 64); each batch of IROW_W rows reads its slice of the units
 * (a batch boundary inside a line: the line is re-read from L2) and writes
 * through the line assembler (all NT). Reading the whole unit at once with
 * a packed staging for the other batch was slower (382 vs 317 ms, 2^28).
 * pf: the next item's slices are prefetched from inside the transform. */
template<int ARM>
void irow_fn(void *c_, uint64_t lo, uint64_t hi, int w, scratch *ws){
    (void)w;
    const Ctx &c = *(const Ctx *)c_;
    if(c.counts){auto &n=c.counts[w]; (ARM==ARM_TMP?n.row_transpose_forward:n.row_inverse)+=(hi-lo)*IROW_G;}
    const Plan &pl = *c.pl;
    const Primes &PS = *c.PS;
    const size_t M2 = pl.M2, lbv = pl.lbv, nblk = pl.nblk;
    const size_t ng = pl.nrows / IROW_G;
    constexpr size_t W = IROW_W;
    constexpr size_t BB = W * TB * SLOT;                    /* bytes per batch per block */
    V *rb = SALLOC(ws, V, M2 * W);
    LineAsm la; lasm_init(la, ws, nblk);
    const Pk52 K = pk52_mk();
    for(; lo < hi; ++lo){
        const size_t g = (size_t)(lo % ng);
        const int q = (int)(lo / ng);
        const Prime &P = PS.P[q]; const PrimeV &pv = PS.V[q];
        for(size_t bt = 0; bt < IROW_G; bt += W){
            const size_t r0 = g * IROW_G + bt;
            for(size_t blk = 0; blk < nblk; ++blk){
                const uint8_t *u = c.fp[q] + blk * pl.bstride + r0 * (TB * SLOT);
                for(size_t sl = 0; sl < TB; ++sl){
                    const size_t b = blk * TB + sl;
                    if(b >= lbv) break;
                    for(size_t i = 0; i < W; ++i) rb[b * W + i] = ld52(u + (i * TB + sl) * SLOT, K);
                }
            }
            for(size_t b = lbv * W; b < M2 * W; ++b) rb[b] = _mm512_setzero_si512();
            /* prefetch cursor: the next batch's slices (same group, or the next item's first batch) */
            PfCur pf; pf.n = 0;
            if(c.ipf){
                size_t r1 = r0 + W; int q1 = q; int ok = 1;
                if(bt + W >= IROW_G){
                    if(lo + 1 < hi){ const size_t g1 = (size_t)((lo + 1) % ng); q1 = (int)((lo + 1) / ng); r1 = g1 * IROW_G; }
                    else ok = 0;
                }
                static const int ipfh = 2;   /* 1/ipfh of the next batch's blocks: all = neutral (evicted from L3 before use), half = irow 286 → 241 ms at 2^28 */
                if(ok) pf_init(pf, c.fp[q1] + r1 * (TB * SLOT), pl.bstride, (BB + 63 + (r1 * (TB * SLOT)) % 64) / 64 + 0, nblk / (size_t)ipfh, c.ipf);
            }
            PfCur *pp = pf.n ? &pf : NULL;
            /* interior full nodes: radix-8 DFS + scale once rb spills L2 (M2 >= 4096: 289 vs 306 ms at 2^28), else the recursive normalized form */
            if constexpr(ARM == ARM_LIN){ if(pl.full) full<true,true,W>((uint64_t *)rb, M2, 0, P, pv, pp); else if(M2 >= 4096) itft<0, 1, W>(rb, M2, 0, lbv, P, pv, pp); else itft<0, FAST_INV, W>(rb, M2, 0, lbv, P, pv, pp); }
            else if constexpr(ARM == ARM_CYC) full<true,true,W>((uint64_t *)rb, M2, 0, P, pv, pp);
            else tft<1,W>(rb, M2, 0, lbv, P, pv, pp);
            for(size_t blk = 0; blk < nblk; ++blk){
                uint8_t *u = c.fp[q] + blk * pl.bstride + r0 * (TB * SLOT);
                alignas(64) uint8_t buf[BB + 64];
                for(size_t i = 0; i < W; ++i)
                    for(size_t sl = 0; sl < TB; ++sl) st52(buf + (i * TB + sl) * SLOT, rb[(blk * TB + sl) * W + i], K, pv);
                lasm_put(la, blk, u, buf, BB, c.nts);
            }
        }
        lasm_flush(la);
    }
    if(c.nts) _mm_sfence();
}

/* ---- emit task adapter ------------------------------------------------- */
/* Task order: when a slot holds a whole number of chunks (8C % OCH == 0)
 * and the view is natural, task t = (block, row range) runs the TB
 * chunks of that block's rows back to back, so the block's lines are read
 * once (the second slot's gather hits L2). Any order is correct. */
struct EmitMap { int nch, nrr, TBm; };
template<int L>
void emit_fn(void *c_, int task, int w, scratch *ws){
    const EmitCtx &e = *(const EmitCtx *)c_;
    const EmitMap &m = *(const EmitMap *)e.map;
    if(m.nrr){
        const int blk = task / m.nrr, rr = task % m.nrr;
        for(int sl = 0; sl < m.TBm; ++sl){
            const int k = (blk * m.TBm + sl) * m.nrr + rr;
            if(k < m.nch) emit_chunk<L>(e, k, ws, w, false, TB >= 8 && sl + 1 < m.TBm && k + m.nrr < m.nch);
        }
    }else emit_chunk<L>(e, task, ws, w);
}
inline int emit_tasks(EmitMap &m, const Plan &pl, int nch, int reversed){
    m.nch = nch; m.TBm = TB; m.nrr = 0;
    if(reversed || pl.Lg != 1 || (8 * pl.C) % OCH || pl.voff) return nch;
    m.nrr = (int)(8 * pl.C / OCH);
    const int nslots = (nch + m.nrr - 1) / m.nrr;
    return ((nslots + TB - 1) / TB) * m.nrr;
}


} // namespace sbn::v3::SBN3_P48_NS
