/* Imported from labs/cr/cr_fuse.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_fuse.hpp — fused inverse-row + emit pass (LIN arm, L = 8, M2 ≤ 8192; the default LIN path).
 * Bandwidth: the inverse rows never go back to DRAM (irow write 0.76 N + emit read 0.76 N saved:
 * 21 GB of 93 at 2^28, NOTES §14.3). Its input is the tile's ROW-MAJOR product planes (§26: one
 * sequential run per prime per row, c.op[q] + r·rms; the rolling one-prime-ahead prefetch CR_IPFR
 * streams prime q+1's run under prime q's inverse); the blocked-plane gather (row_in_g) remains for
 * CR_PRM=0.
 *
 * Item = a contiguous row range [r0, r1) (one per worker).  Per row r:
 *   for each prime q: read the row's run → rb[q] (M2 vectors, zero beyond lbw)
 *   → normalized truncated inverse (as irow_fn) — the NP inverse rows
 *   (NP × M2 × 64 B, L3-resident: M2 ≤ 8192) — then one sweep over the
 *   columns cv < lbv: Garner digits (digits_of) → the radix-2^T lag
 *   chain (lag_add, WD words per T-digit, §32) → the 8 digits packed into the piece's T bytes
 *   (Pack8) at byte offset T·(cv·C + r) of the output → the per-column NT line assembler (the
 *   pieces of consecutive rows of one column are contiguous).
 * The output stream is column-major over the plane, so the lag digits
 * (b1 lane 7, b2 lanes 6–7) and the carry persist PER COLUMN across
 * rows.  Seeds: item i first computes the digits of row
 * r0−1 (item 0: row C−1, shifted one column right: (cv, C−1) precedes
 * (cv+1, 0)); the carries into its first row are unknown and start at 0 —
 * the carry-out of every column's last row is journaled and added at the
 * next item's first row (the last item's at (cv+1, 0)) by fuse_join.
 * NOTES §14. */
#include <string.h>
namespace sbn::v3::SBN3_P48_NS {

/* T-byte piece packer: 8 T-bit digits (WD words each) → the T bytes Σ D_u·2^{T·u}.
 * T ≡ 0 mod 8 (byte form, §32): output byte j = byte (j mod T/8) of digit u = j / (T/8), i.e. byte
 * (j mod T/8) mod 8 of its word (j mod T/8) / 8 — one 2-source byte permute per 64-B output vector
 * over the word vectors (w0, w1), plus a zeroing permute of w2 or'ed in where WD = 3 (CR_PACKB=0:
 * the dword form below, WD = 2 only).
 * T ≡ 4 mod 8 (dword form): dword m takes bits [32m, 32m+32) of the stream: u = ⌊32m/T⌋,
 * s = 32m − T·u; (D_u >> s) | (D_{u+1} << (T − s)); the last dword ends exactly at 8T, so D_8 is
 * never needed. */
struct Pack8 {
    int WD, T, byte;
    int nd, nv;                                     /* dword form */
    V IU[3], IU1[3], S1[3], S2[3], S3[3], S4[3];
    __mmask8 m[3];
    int nvo;                                        /* byte form */
    V BI[WDMAX], BI2[WDMAX]; __mmask64 BM1[WDMAX], BM2[WDMAX], BS[WDMAX];   /* bytes from (w0,w1) / from w2 / stored */
    void init(int T_, int WD_){
        T = T_; WD = WD_;
        byte = (T % 8 == 0);
        if(WD >= 3 && !byte) ::sbn::v3::fatal(SBN3_FATAL_MATH, "p48 Pack8 digit width", WD, 2);
        if(byte){
            nvo = (T + 63) / 64; const int tb = T / 8;
            for(int v = 0; v < nvo; ++v){
                alignas(64) uint8_t i1[64], i2[64]; uint64_t m2 = 0, ms = 0;
                for(int b = 0; b < 64; ++b){
                    const int j = 64 * v + b;
                    i1[b] = 0; i2[b] = 0;
                    if(j >= T) continue;
                    ms |= 1ull << b;
                    const int u = j / tb, ob = j % tb, k = ob / 8, bb = ob % 8, src = 8 * u + bb;
                    if(k == 0) i1[b] = (uint8_t)src;
                    else if(k == 1) i1[b] = (uint8_t)(64 + src);
                    else { i2[b] = (uint8_t)(src+64*(k-2)); m2 |= 1ull << b; }
                }
                BI[v] = _mm512_load_si512(i1); BI2[v] = _mm512_load_si512(i2); BM1[v] = ms & ~m2; BM2[v] = m2; BS[v] = ms;
            }
            return;
        }
        nd = T / 4; nv = (nd + 7) / 8;
        for(int v = 0; v < nv; ++v){
            alignas(64) uint64_t iu[8], iu1[8], s1[8], s2[8], s3[8], s4[8]; __mmask8 mm = 0;
            for(int l = 0; l < 8; ++l){
                const int mi = v * 8 + l;
                if(mi >= nd){ iu[l] = iu1[l] = 0; s1[l] = s2[l] = s3[l] = s4[l] = 64; continue; }
                const int u = (32 * mi) / T, s = 32 * mi - T * u;
                iu[l] = (uint64_t)u; iu1[l] = (uint64_t)(u + 1 < 8 ? u + 1 : 7);
                s1[l] = s < 64 ? (uint64_t)s : 64;                 /* lo_u >> s           */
                s2[l] = s < 64 ? (uint64_t)(64 - s) : 64;          /* hi_u << (64 − s)    */
                s3[l] = s >= 64 ? (uint64_t)(s - 64) : 64;         /* hi_u >> (s − 64)    */
                s4[l] = (T - s) < 32 ? (uint64_t)(T - s) : 64;     /* lo_{u+1} << (T − s) */
                mm |= (__mmask8)(1u << l);
            }
            IU[v] = _mm512_load_si512(iu); IU1[v] = _mm512_load_si512(iu1);
            S1[v] = _mm512_load_si512(s1); S2[v] = _mm512_load_si512(s2);
            S3[v] = _mm512_load_si512(s3); S4[v] = _mm512_load_si512(s4);
            m[v] = mm;
        }
    }
    /* dst: writable to WDMAX*64 bytes; LineAsm callers additionally provide 64 bytes padding. */
    __attribute__((always_inline)) inline
    void run(uint8_t *dst, const V w[WDMAX]) const {
        if(byte){
            for(int v = 0; v < nvo; ++v){
                V o = _mm512_maskz_permutex2var_epi8(BM1[v], w[0], BI[v], w[1]);
                if constexpr(WDMAX>3){
                    if(WD==4 && BM2[v])o=_mm512_or_si512(o,_mm512_maskz_permutex2var_epi8(BM2[v],w[2],BI2[v],w[3]));
                    else if(WD==3 && BM2[v])o=_mm512_or_si512(o,_mm512_maskz_permutexvar_epi8(BM2[v],BI2[v],w[2]));
                }else if(WD==3 && BM2[v])o=_mm512_or_si512(o,_mm512_maskz_permutexvar_epi8(BM2[v],BI2[v],w[2]));
                _mm512_mask_storeu_epi8(dst + 64 * v, BS[v], o);
            }
            return;
        }
        const V lo = w[0], hi = w[1];
        for(int v = 0; v < nv; ++v){
            const V lu = _mm512_permutexvar_epi64(IU[v], lo), hu = _mm512_permutexvar_epi64(IU[v], hi);
            const V lu1 = _mm512_permutexvar_epi64(IU1[v], lo);
            V x = _mm512_or_si512(_mm512_srlv_epi64(lu, S1[v]), _mm512_sllv_epi64(hu, S2[v]));
            x = _mm512_or_si512(x, _mm512_or_si512(_mm512_srlv_epi64(hu, S3[v]), _mm512_sllv_epi64(lu1, S4[v])));
            _mm256_mask_storeu_epi32(dst + 32 * v, m[v], _mm512_cvtepi64_epi32(x));
        }
    }
};

#ifdef CR_TILE_PROF
struct FpCnt { uint64_t gather, itft, sweep, rows; };
static FpCnt g_fp[64];
#define FP_T(v) uint64_t v = cr_tsc()
#else
#define FP_T(v)
#endif
struct FuseCtx {
    const Ctx *c; const EmitK *EK;
    uint64_t *rp, *tail; int nts;
    size_t prefix=0;              /* exact unwrapped low window, or zero */
    int W;                        /* items: rows [i·C/W, (i+1)·C/W) */
    uint8_t *jrn;                 /* [W][lbv]: carry-out of the item's last row, per column */
};
/* Byte codecs use eight T-byte pieces; T84 needs sixteen for whole lines.
 * This also keeps odd-width teams from sharing a line across item seams. */
static inline size_t fuse_row_grain(int T){ return T%8 ? 16 : 8; }
static inline size_t fuse_row_start(size_t item,size_t C,size_t workers,int T){
    const size_t grain=fuse_row_grain(T);
    return (item*(C/grain)/workers)*grain;
}

/* byte sink for the few pieces past outcap: bytes of limbs [outcap, nl) → tail */
static inline void fuse_put_bytes(const FuseCtx &f, const Plan &pl, size_t ob, const uint8_t *src, size_t n){
    uint8_t *rp8 = (uint8_t *)f.rp, *tl8 = (uint8_t *)f.tail;
    const size_t oc = (f.prefix?f.prefix:pl.outcap) * 8, nl8 = f.prefix?oc:pl.nl * 8;
    for(size_t i = 0; i < n; ++i){
        const size_t b = ob + i;
        if(b < oc) rp8[b] = src[i];
        else if(b < nl8) tl8[b - oc] = src[i];
    }
}

template<int L, int WD>
void irowemit_w(void *f_, uint64_t lo, uint64_t hi, int w, scratch *ws){
    static_assert(L == 8, "fused irow+emit: Lg = 1 only");
    (void)w;
    const FuseCtx &f = *(const FuseCtx *)f_;
    const Ctx &c = *f.c;
    const Plan &pl = *c.pl;
    const Primes &PS = *c.PS;
    const EmitK &EK = *f.EK;
    const size_t M2 = pl.M2, lbv = pl.lbv, nblk = pl.nblk, C = pl.C;
    constexpr size_t PSB = (size_t)TB * SLOT, PSBO = (size_t)TB * SLOTO;
    const int oo = c.op[0] != NULL;                              /* product planes at SLOTO (§21 hybrid) */
    const size_t psb = oo ? PSBO : PSB;
    const size_t T = (size_t)pl.T;
    const size_t cols=f.prefix?std::min(lbv,(8*f.prefix+T*C-1)/(T*C)):lbv;
    const size_t out_bytes=8*(f.prefix?f.prefix:pl.outcap);
    const Pk52 K = pk52_mk();
    Pack8 PK; PK.init((int)T, WD);
    /* Capacity experiment: one working inverse row, previous primes packed only to lbv.
     * Intermediate ITFT tails still need M2, so only COMPLETED rows are compacted. */
    const bool packed_rows = CR_FPACK && c.prm;
    V *rb[NP];
    for(int q = 0; q < NP; ++q) rb[q] = packed_rows && q ? rb[0] : SALLOC(ws, V, M2 + 16);
    uint8_t *packed[NP]{};
    if(packed_rows) for(int q = 0; q + 1 < NP; ++q)
        packed[q] = SALLOC(ws, uint8_t, padded_row_stride(lbv * 48, 1));
    auto residue_at = [&](int q, size_t cv) -> V {
        return packed_rows && q + 1 < NP ? ldslot<48>(packed[q] + cv * 48, K) : rb[q][cv];
    };
    /* per-column lag state: lane 7 of the previous row's b1 words, lanes 6–7 of its b2, the carry */
    uint64_t *sb1[WD]; for(int k = 0; k < WD; ++k) sb1[k] = SALLOC(ws, uint64_t, lbv + 8);
    uint64_t *sb2a = SALLOC(ws, uint64_t, lbv + 8), *sb2b = SALLOC(ws, uint64_t, lbv + 8);
    uint8_t *scA = SALLOC(ws, uint8_t, lbv + 64);
    LineAsm la; lasm_init(la, ws, lbv);
    // Pack8 writes WD full vectors; LineAsm additionally requires readable padding.
    constexpr size_t packed_scratch_bytes=64*(WD+1);
    alignas(64) uint8_t pbuf[packed_scratch_bytes];
    alignas(64) uint64_t tl[8];
    const V ZERO = _mm512_setzero_si512();
    /* de-phase the workers: in lock-step every core gathers at the same time
     * (DRAM-bound) and computes at the same time (DRAM idle); a one-time
     * start offset of item·CR_FSKEW µs spreads the gathers over the row */
    const unsigned fskew = c.fused_start_skew_us;
    /* CR_FGATE=k: at most k cores per CCX gathering at once (NOTES §17 O1 / §18: the one
     * overlap idea that measured — fused pass −5..7% at cap 3 or 4; 0 = off) */
    static const int fgate = 4;
    const int ccx = cr_ccx(w);
    if(fskew > 0 && lo > 0){ const double t0 = now_ms(), dl = (double)lo * fskew / 1000.0; while(now_ms() - t0 < dl) _mm_pause(); }
    for(uint64_t it = lo; it < hi; ++it){
        const size_t r0 = fuse_row_start(it,C,f.W,pl.T), r1 = fuse_row_start(it+1,C,f.W,pl.T);
        if(r0 >= r1) continue;
        for(int k = 0; k < WD; ++k) memset(sb1[k], 0, 8 * lbv);
        memset(sb2a, 0, 8 * lbv); memset(sb2b, 0, 8 * lbv);
        memset(scA, 0, lbv);
        /* one row: gather + inverse for the 4 primes (next (q, row) pieces prefetched from the transform) */
        auto row_in_g = [&](size_t r, size_t rnext, int has_next){
            (void)rnext; (void)has_next;
            /* gather the four primes' pieces block-interleaved (four independent
             * streams: twice the memory-level parallelism of one plane at a
             * time, gprobe 22.7 vs 12.6 GB/s single-thread), CR_GPF pieces ahead
             * prefetched */
            static const int gpf = 24;
            auto gpf_line = [&](const uint8_t *p){ _mm_prefetch((const char *)p, _MM_HINT_T1); };
            const size_t po = (r * psb) & 63, pl1 = (po + psb + 63) / 64;
            FP_T(g0);
            if(fgate) gate_acquire(c.gates, ccx, (uint32_t)fgate);
            if(c.prm){                                               /* row-major product planes: one sequential stream per prime (NOTES §26) */
                for(int q = 0; q < NP; ++q){
                    V *x = rb[q];
                    const uint8_t *base = c.op[q] + r * c.rms;
                    for(size_t blk = 0; blk < nblk; ++blk){
                        const uint8_t *u = base + blk * psb;
                        if(gpf && blk + (size_t)gpf < nblk) gpf_line(u + (size_t)gpf * psb);
                        for(size_t sl = 0; sl < TB; ++sl) x[blk * TB + sl] = ldslot<SLOTO>(u + sl * SLOTO, K);
                    }
                    for(size_t b = nblk * TB; b < M2; ++b) x[b] = ZERO;
                }
            }else{
                for(int q = 0; q < NP; ++q){
                    V *x = rb[q];
                    for(size_t blk = 0; blk < nblk; ++blk){
                        const uint8_t *u = (oo ? c.op[q] + blk * c.obs : c.fp[q] + blk * pl.bstride) + r * psb;
                        if(gpf && blk + (size_t)gpf < nblk){ const uint8_t *p = u + (size_t)gpf * (oo ? c.obs : pl.bstride) - po; for(size_t l = 0; l < pl1; ++l) gpf_line(p + 64 * l); }
                        for(size_t sl = 0; sl < TB; ++sl) x[blk * TB + sl] = oo ? ldslot<SLOTO>(u + sl * SLOTO, K) : ld52(u + sl * SLOT, K);
                    }
                    for(size_t b = nblk * TB; b < M2; ++b) x[b] = ZERO;
                }
            }
            if(fgate) gate_release(c.gates, ccx);
            FP_T(g1);
#ifdef CR_TILE_PROF
            g_fp[w < 0 || w >= 64 ? 0 : w].gather += g1 - g0;
#endif
            for(int q = 0; q < NP; ++q){
                const Prime &P = PS.P[q]; const PrimeV &pv = PS.V[q];
                V *x = rb[q];
                FP_T(i0);
                if(c.counts)++c.counts[w].row_inverse;
                if(pl.full) full<true, true, 1>((uint64_t *)x, M2, 0, P, pv, NULL);
                else if(M2 >= 4096) itft<0, 1, 1, true>(x, M2, 0, lbv, P, pv, NULL);   /* ZT: zero tail (§28) */
                else itft<0, FAST_INV, 1, true>(x, M2, 0, lbv, P, pv, NULL);
                FP_T(i1);
#ifdef CR_TILE_PROF
                g_fp[w < 0 || w >= 64 ? 0 : w].itft += i1 - i0;
#endif
            }
        };
        /* CR_IPFR=k (§26, row-major product only): rolling one-prime-ahead. Prime q's run (340 KB at
         * 2^28) is read right before its inverse transform, having been prefetched (k lines per kernel
         * hook, ~6k hooks) under prime q−1's transform; the last prime's transform pulls the next row's
         * first run. The prefetched run sits beside the 512 KB row in L2 — the whole-row variant
         * (1.36 MB) evicted the row and the sweep's buffers and lost. */
        constexpr int rolling_prefetch_lines = 1;   /* §26: rolling prefetch = default */
        auto read_run = [&](int q, size_t rr){
            V *x = rb[q];
            const uint8_t *base = c.op[q] + rr * c.rms;
            for(size_t blk = 0; blk < nblk; ++blk){
                const uint8_t *u = base + blk * psb;
                for(size_t sl = 0; sl < TB; ++sl) x[blk * TB + sl] = ldslot<SLOTO>(u + sl * SLOTO, K);
            }
            for(size_t b = nblk * TB; b < M2; ++b) x[b] = ZERO;
        };
        auto row_in_roll = [&](size_t r, size_t rnext, int has_next){
            const uint32_t rl = (uint32_t)((nblk * psb + 63) / 64);
            FP_T(g0);
            read_run(0, r);
            FP_T(g1);
#ifdef CR_TILE_PROF
            g_fp[w < 0 || w >= 64 ? 0 : w].gather += g1 - g0;
#endif
            for(int q = 0; q < NP; ++q){
                const Prime &P = PS.P[q]; const PrimeV &pv = PS.V[q];
                V *x = rb[q];
                PfCur cur; cur.n = 0; PfCur *pf = NULL;
                if(q + 1 < NP){ pf_init(cur, c.op[q + 1] + r * c.rms, 64, 1, rl, rolling_prefetch_lines); cur.hint = 1; pf = &cur; }
                else if(has_next && rnext < C){ pf_init(cur, c.op[0] + rnext * c.rms, 64, 1, rl, rolling_prefetch_lines); cur.hint = 1; pf = &cur; }
                FP_T(i0);
                if(c.counts)++c.counts[w].row_inverse;
                if(pl.full) full<true, true, 1>((uint64_t *)x, M2, 0, P, pv, pf);
                else if(M2 >= 4096) itft<0, 1, 1, true>(x, M2, 0, lbv, P, pv, pf);     /* ZT: zero tail (§28) */
                else itft<0, FAST_INV, 1, true>(x, M2, 0, lbv, P, pv, pf);
                if(packed_rows && q + 1 < NP)
                    for(size_t cv = 0; cv < lbv; ++cv) stslot<48>(packed[q] + cv * 48, x[cv], K, pv);
                FP_T(i1);
#ifdef CR_TILE_PROF
                g_fp[w < 0 || w >= 64 ? 0 : w].itft += i1 - i0;
#endif
                if(q + 1 < NP){
                    FP_T(h0);
                    read_run(q + 1, r);
                    FP_T(h1);
#ifdef CR_TILE_PROF
                    g_fp[w < 0 || w >= 64 ? 0 : w].gather += h1 - h0;
#endif
                }
            }
        };
        auto row_in = [&](size_t r, size_t rnext, int has_next){
            if(c.prm) row_in_roll(r, rnext, has_next); else row_in_g(r, rnext, has_next);
        };
        /* seed: the lag digits of the stream predecessor of every (cv, r0) */
        {
            const size_t rs = it == 0 ? C - 1 : r0 - 1;
            const int shift = it == 0;
            row_in(rs, r0, 1);
            for(size_t cv = 0; cv + (size_t)shift < cols; ++cv){
                const size_t v = cv * C + rs, j = 8 * v;
                V x[NP];
                for(int q = 0; q < NP; ++q) x[q] = residue_at(q, cv);
                if(j + 8 > pl.vtrunks){ const size_t lv = pl.vtrunks > j ? pl.vtrunks - j : 0; for(int q = 0; q < NP; ++q) x[q] = _mm512_maskz_mov_epi64((__mmask8)((1u << lv) - 1), x[q]); }
                DigW<WD> d = digits_of<WD>(EK, x);
                const size_t k = cv + (size_t)shift;
                for(int wd = 0; wd < WD; ++wd){ _mm512_store_si512(tl, d.b1[wd]); sb1[wd][k] = tl[7]; }
                _mm512_store_si512(tl, d.b2);  sb2a[k] = tl[6]; sb2b[k] = tl[7];
            }
        }
        for(size_t r = r0; r < r1; ++r){
            row_in(r, r + 1, r + 1 < r1);
            FP_T(w0);
            
            /* two independent columns per step: two Garner chains in flight */
            auto digits_at = [&](size_t cv, size_t r_, DigW<WD> &d){
                const size_t v = cv * C + r_, j = 8 * v;
                V x[NP];
                for(int q = 0; q < NP; ++q) x[q] = residue_at(q, cv);
                if(j + 8 > pl.vtrunks){ const size_t lv = pl.vtrunks > j ? pl.vtrunks - j : 0; for(int q = 0; q < NP; ++q) x[q] = _mm512_maskz_mov_epi64((__mmask8)((1u << lv) - 1), x[q]); }
                d = digits_of<WD>(EK, x);
            };
            /* E3 (NOTES §20): deferred lane carries (lag_add, WD words §32). The digit streams are
             * summed lane-wise; the carry out of lane j (top word >> HB) is added to lane j+1 in one
             * shifted add; a second round only when a lane sits within 2 of 2^T (rare). No mask→GPR
             * round trips. */
            auto finish = [&](size_t cv, size_t r_, const DigW<WD> &d, uint8_t *pb){
                V w[WDMAX], a1[WD];
                for(int k = 0; k < WD; ++k){
                    const V pb1 = _mm512_set1_epi64((long long)sb1[k][cv]);
                    a1[k] = _mm512_alignr_epi64(d.b1[k], pb1, 7);
                    w[k] = d.b0[k];
                }
                const V pb2 = _mm512_mask_set1_epi64(_mm512_set1_epi64((long long)sb2b[cv]), (__mmask8)0x40, (long long)sb2a[cv]);
                const V a2 = _mm512_alignr_epi64(d.b2, pb2, 6);
                const unsigned cout = lag_add<WD>(w, a1, a2, scA[cv], EK.LIM, EK.HM, EK.HB);
                for(int k = 0; k < WD; ++k) sb1[k][cv] = (uint64_t)_mm256_extract_epi64(_mm512_extracti64x4_epi64(d.b1[k], 1), 3);
                const __m256i h2 = _mm512_extracti64x4_epi64(d.b2, 1);
                sb2a[cv] = (uint64_t)_mm256_extract_epi64(h2, 2); sb2b[cv] = (uint64_t)_mm256_extract_epi64(h2, 3);
                scA[cv] = (uint8_t)cout;
                PK.run(pb, w);
                const size_t ob = T * (cv * C + r_);
                if(ob + T <= out_bytes) lasm_put(la, cv, (uint8_t *)f.rp + ob, pb, T, f.nts);
                else fuse_put_bytes(f, pl, ob, pb, T);
            };
            alignas(64) uint8_t pb2buf[packed_scratch_bytes];
            size_t cv = 0;
            /* CR_FWAY=4: four columns per step (four Garner + carry chains in flight; K4, NOTES §17) */
            static const int fway = (NP >= 5 ? 4 : 2);   /* §32: four Garner chains in flight hide the wider CRT's latency (sweep −2..−5 % at NP ≥ 5) */
            alignas(64) uint8_t pb4[4][packed_scratch_bytes];
            if(fway >= 4)
                for(; cv + 4 <= cols; cv += 4){
                    DigW<WD> d[4];
                    for(int i = 0; i < 4; ++i) digits_at(cv + (size_t)i, r, d[i]);
                    for(int i = 0; i < 4; ++i) finish(cv + (size_t)i, r, d[i], pb4[i]);
                }
            for(; cv + 2 <= cols; cv += 2){
                DigW<WD> d0, d1;
                digits_at(cv, r, d0); digits_at(cv + 1, r, d1);
                finish(cv, r, d0, pbuf); finish(cv + 1, r, d1, pb2buf);
            }
            for(; cv < cols; ++cv){ DigW<WD> d0; digits_at(cv, r, d0); finish(cv, r, d0, pbuf); }
            FP_T(w1);
#ifdef CR_TILE_PROF
            g_fp[w < 0 || w >= 64 ? 0 : w].sweep += w1 - w0; g_fp[w < 0 || w >= 64 ? 0 : w].rows += 1;
#endif
        }
        // The packed rows stop after K=8*C*lbv trunks. The final b1/b2
        // streams can extend past K (full MAC, or CYC's carry fringe).
        // Materialize those two zero-coefficient digits; fuse_join subsequently
        // adds all journaled carries, including this item's final carry-out.
        const size_t end_bytes=T*C*lbv;
        // CYC exposes exactly K bits, but its fringe must still reach the
        // separate tail sink before modular carry closure.
        if(!f.prefix && r1==C && (pl.outcap*8>end_bytes || pl.ring_rn)){
            uint64_t extra[WD];uint64_t carry=0;
            for(int k=0;k<WD;++k){const u128 sum=(u128)sb1[k][lbv-1]+(k?carry:sb2a[lbv-1]);extra[k]=(uint64_t)sum;carry=(uint64_t)(sum>>64);}
            const unsigned hb=unsigned(pl.T)-(WD-1)*64;
            if(hb<64){carry=extra[WD-1]>>hb;extra[WD-1]&=(uint64_t(1)<<hb)-1;}
            V tail_words[WD];for(int k=0;k<WD;++k)tail_words[k]=_mm512_setr_epi64(extra[k],k?0:sb2b[lbv-1]+carry,0,0,0,0,0,0);
            PK.run(pbuf,tail_words);fuse_put_bytes(f,pl,end_bytes,pbuf,T);
        }
        lasm_flush(la);
        for(size_t cv = 0; cv < lbv; ++cv) f.jrn[it * lbv + cv] = cv<cols?scA[cv]:0;
    }
    if(f.nts) _mm_sfence();
}
template<int L>
void irowemit_fn(void *f_, uint64_t lo, uint64_t hi, int w, scratch *ws){
    const FuseCtx &f = *(const FuseCtx *)f_;
    if constexpr(WDMAX>3)if(f.c->pl->WD==4){irowemit_w<L,4>(f_,lo,hi,w,ws);return;}
    if(f.c->pl->WD == 3) irowemit_w<L, 3>(f_, lo, hi, w, ws);
    else irowemit_w<L, 2>(f_, lo, hi, w, ws);
}

/* add the journaled carries: item i's carry-out of column cv enters
 * (cv, r0 of item i+1), the last item's (cv+1, 0); at bit T·8·v */
static inline void fuse_join(const FuseCtx &f, const Plan &pl){
    const size_t C = pl.C, lbv = pl.lbv, T = (size_t)pl.T;
    const size_t limit=f.prefix?f.prefix:pl.nl;
    const size_t cols=f.prefix?std::min(lbv,(8*f.prefix+T*C-1)/(T*C)):lbv;
    auto sink = [&](size_t i) -> uint64_t * { return i < pl.outcap ? f.rp + i : f.tail + (i - pl.outcap); };
    for(int it = 0; it < f.W; ++it){
        const int last = it + 1 == f.W;
        const size_t rn = last ? 0 : fuse_row_start(it+1,C,f.W,pl.T);
        for(size_t cv = 0; cv < cols; ++cv){
            const unsigned cy0 = f.jrn[(size_t)it * lbv + cv];
            if(!cy0) continue;
            const size_t cvn = last ? cv + 1 : cv;
            if(cvn >= lbv && last){ /* past the last column: into the tail region if any */ }
            const size_t bit = T * 8 * (cvn * C + rn);
            size_t li = bit / 64; const unsigned sh = (unsigned)(bit % 64);
            if(li >= limit) continue;
            u128 s = (u128)*sink(li) + ((u128)cy0 << sh);
            *sink(li) = (uint64_t)s;
            unsigned char cy = (unsigned char)(s >> 64);
            for(++li; cy && li < limit; ++li){ uint64_t *lp = sink(li); cy = (unsigned char)(++*lp == 0); }
        }
    }
}

} // namespace sbn::v3::SBN3_P48_NS
