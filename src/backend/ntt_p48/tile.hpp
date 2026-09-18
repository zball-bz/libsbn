/* Imported from labs/cr/cr_tile.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_tile.hpp — the product tile: block tile v4 (TB = 2, L = 8; NOTES §15, §21, §26, cleaned in §30).
 *
 * Item = (prime q, block blk of TB = 2 slots), processed serially on one core:
 *   sweep Y(blk)      the block's nrows packed pieces → two unpacked columns (L3-resident ring of 2)
 *   per slot s:
 *     fwd(Y_s)        column DFS (cr_rows full<> on the heap twiddles of column b = blk·TB + s)
 *     [s == 0] sweep A(blk)   the handle: A's row planes (fresh: unpacked rows; hspec = 1: A's spectrum)
 *     [fresh]  fwd(A_s)
 *     inv(Y_s)        with the products fused into its leaf blocks (colpass_x + the conv8x8 8-lane ring)
 *     [s == 0] flush of the PREVIOUS block's image (serial, before the image is overwritten)
 *     pack slot s into the block image stg (48-bit slots: stslot<SLOTO>)
 *   the image leaves during the next item (or at once at the chunk's last item):
 *     prm (row-major product planes = the fused pass's input, §26): DRFlush — every row's piece goes to
 *       c.op[q] + r·rms + blk·PSBO through the per-row NT line assembler; with the 8-block dynamic
 *       chunks (tile_launch) a row's 768-B run leaves as 12 whole lines
 *     else: DFlush — whole-line NT copy into the block's region (in place over Y, or separate planes)
 * The L2 holds one 64 KB DFS sub-block, the current tables and the stream lines; the two columns
 * (2 × C vectors) and the image sit in the L3 share. What lost against this serial schedule (side
 * jobs on the kernel hooks, the CCX gate, the W = 2 Y/A pair, PIN-from-DRAM, next-block cursors,
 * the inverse-writes-image POUT, and the tiles v2 streaming / v3 helper thread / v5 packed images /
 * v6 per-slot / v7 serial-generalized) is recorded in NOTES §13, §15, §17, §18, §21.
 * SPEC = 1 (handle build with hspec = 1) and CR_TILE4=0 use block_fn (cr_engine.hpp). */
namespace sbn::v3::SBN3_P48_NS {

/* line run of slots [s0, s0 + ns) inside a piece: first line, line count */
static inline void slot_run(size_t s0, size_t ns, size_t &l0, size_t &nl){
    const size_t b0 = s0 * SLOT, b1 = (s0 + ns) * SLOT;
    l0 = b0 / 64; nl = (b1 + 63) / 64 - l0;
}

/* sparse column sweep (serial form): piece vr's slot-s bytes, CR_SWPF pieces ahead prefetched */
static inline void sweep_col(V *dst, const uint8_t *rb, size_t s, size_t n, const Pk52 &K){
    static const int d = 16;
    constexpr size_t PSB = (size_t)TB * SLOT;
    const size_t o0 = (s * SLOT) & ~(size_t)63, o1 = (s * SLOT + SLOT - 1) & ~(size_t)63;
    for(size_t vr = 0; vr < n; ++vr){
        if(d && vr + (size_t)d < n){
            const uint8_t *p = rb + (vr + (size_t)d) * PSB;
            _mm_prefetch((const char *)(p + o0), _MM_HINT_T0);
            if(o1 != o0) _mm_prefetch((const char *)(p + o1), _MM_HINT_T0);
        }
        dst[vr] = ld52(rb + vr * PSB + s * SLOT, K);
    }
}

struct DSweep { V *d0, *d1; const uint8_t *rb; size_t vr, n; Pk52 K; int nsl; size_t str; };   /* joint dense sweep of TB slots; destinations at stride str */
static void dsweep_step(void *ctx){
    DSweep &S = *(DSweep *)ctx;
    if(S.vr >= S.n) return;
    const size_t v0 = S.vr; S.vr += 8;
    constexpr size_t PSB = (size_t)TB * SLOT;
    /* CR_DSPF=d: software prefetch d pieces ahead (the dense sweep is OoO-window
     * bound at ~13 GB/s per core: ~6 ops per line keep only ~85 lines in flight) */
    static const int d = 48;    /* §21: NTA-hinted, 48 pieces ahead: the streamed region no longer evicts the rings/image */
    static const int dh = 2;
    if(d && v0 + (size_t)d + 8 <= S.n){
        const uint8_t *p = S.rb + (v0 + (size_t)d) * PSB;
        constexpr size_t NL = (8 * PSB + 63) / 64 + 1;
        if(dh == 0) for(size_t l = 0; l < NL; ++l) _mm_prefetch((const char *)(p + 64 * l), _MM_HINT_T0);
        else if(dh == 2) for(size_t l = 0; l < NL; ++l) _mm_prefetch((const char *)(p + 64 * l), _MM_HINT_NTA);
        else for(size_t l = 0; l < NL; ++l) _mm_prefetch((const char *)(p + 64 * l), _MM_HINT_T1);
    }
    for(size_t k = 0; k < 8; ++k){
        const uint8_t *p = S.rb + (v0 + k) * PSB;
        S.d0[(v0 + k) * S.str] = ld52(p, S.K);
        if(S.nsl > 1) S.d1[(v0 + k) * S.str] = ld52(p + SLOT, S.K);
    }
}
static inline void dsweep_finish(DSweep &S){ while(S.vr < S.n) dsweep_step(&S); }
struct DFlush { uint8_t *dst; const uint8_t *src; size_t i, n; };              /* NT copy of lines [i, n) */
static void dflush_step(void *ctx){
    DFlush &F = *(DFlush *)ctx;
    if(F.i >= F.n) return;
    size_t e = F.i + 8; if(e > F.n) e = F.n;
    for(; F.i < e; ++F.i) _mm512_stream_si512((__m512i *)(F.dst + F.i * 64), _mm512_load_si512(F.src + F.i * 64));
}
static inline void dflush_finish(DFlush &F){ while(F.i < F.n) dflush_step(&F); }
/* row-major flush (CR_PRM, NOTES §26): the block image's row pieces (psb bytes at src + vr·psb) go to
 * dst0 + vr·rms through the per-row line assembler; consecutive blocks of the chunk continue each
 * row's pending line, so with 8-block chunks every row's 768-B run leaves as 12 whole NT lines */
struct DRFlush { LineAsm *la; uint8_t *dst0; const uint8_t *src; size_t rms, psb, i, n; };
/* CR_PRMF=1: the fixed-geometry put. All rows of a block share the same line phase (the run offset
 * blk·psb mod 64 is per block, the row runs are line-aligned), so the pending-line bookkeeping of
 * lasm_put reduces to arithmetic: fill the pending line (if any) from the piece's head, stream the
 * whole lines, park the tail in the row's image. Falls back to lasm_put when the run does not start
 * at a line boundary (chunk head inside a line: prime tails only). */
static void drflush_step(void *ctx){
    DRFlush &F = *(DRFlush *)ctx;
    if(F.i >= F.n) return;
    size_t e = F.i + 64; if(e > F.n) e = F.n;
    if(CR_PRMF){
        const size_t po = (uintptr_t)F.dst0 & 63;                    /* phase of this block's pieces within the line */
        const size_t n = F.psb;
        {
            for(; F.i < e; ++F.i){
                uint8_t *dst = F.dst0 + F.i * F.rms;
                const uint8_t *src = F.src + F.i * F.psb;
                uint8_t *img = F.la->img + 64 * F.i;
                size_t pf = F.la->fill[F.i], o = 0;
                uint8_t *pa = F.la->addr[F.i];
                if(pa && dst == pa + pf){                              /* continue the pending line */
                    const size_t room = 64 - pf;
                    _mm512_mask_storeu_epi8(img + pf, (__mmask64)((1ull << room) - 1), _mm512_loadu_si512(src));
                    _mm512_stream_si512((__m512i *)pa, _mm512_load_si512(img));
                    o = room;
                }else if(pa || po){ lasm_put(*F.la, F.i, dst, src, n, 1); continue; }   /* generic path */
                for(; o + 64 <= n; o += 64) _mm512_stream_si512((__m512i *)(dst + o), _mm512_loadu_si512(src + o));
                if(o < n){ pa = dst + o; pf = n - o; _mm512_mask_storeu_epi8(img, (__mmask64)((1ull << pf) - 1), _mm512_loadu_si512(src + o)); }
                else { pa = NULL; pf = 0; }
                F.la->addr[F.i] = pa; F.la->fill[F.i] = (uint8_t)pf;
            }
            return;
        }
    }
    for(; F.i < e; ++F.i) lasm_put(*F.la, F.i, F.dst0 + F.i * F.rms, F.src + F.i * F.psb, F.psb, 1);
}
static inline void drflush_finish(DRFlush &F){ while(F.i < F.n) drflush_step(&F); }

/* the non-v4 tile (handle build with hspec = 1; CR_TILE4=0): the block tile of cr_engine.hpp */
template<int L, int MID, int SPEC>
static inline void (*tile_fn())(void *, uint64_t, uint64_t, int, scratch *){ return &block_fn<L, MID, SPEC>; }

// The inverse leaf depends on the product recipe, not on cache writeback.
template<int MID, int Recipe, bool Scaled>
struct TileProductLeaf { V *y,*a,*ye,*ae; const uint64_t *r8,*ir8; const PrimeV *pv; V kc,kr,kc1,kr1;
    __attribute__((always_inline)) inline void operator()(size_t o,size_t n) const {
        for(size_t k=o;k<o+n;k+=8){
            if constexpr(!MID)conv8x8<Scaled>(y+k,a+k,r8+k,*pv,kc,kr);
            else conv8x8T<Scaled>(y+k,a+k,r8+k,ir8+k,*pv,kc,kr);
            if constexpr(Recipe==2){
                conv8x8<Scaled>(ye+k,ae+k,r8+k,*pv,kc1,kr1);
                // Each REDC result is in [0,2p); their sum is in [0,4p).
                for(unsigned j=0;j<8;++j)y[k+j]=_mm512_add_epi64(y[k+j],ye[k+j]);
            }
        }
    }
};

template<int L, int MID, int SPEC, int Recipe=0, bool Scaled=false, bool WritebackA=false>
void block_fn4(void *c_, uint64_t lo, uint64_t hi, int w, scratch *ws){
    static_assert(L == 8 && !SPEC, "tile v4: product tile, Lg = 1");
    static_assert(TB == 2, "tile v4: two-slot pieces");
    (void)w;
    const Ctx &c = *(const Ctx *)c_;
#ifdef CR_TILE_PROF
    if(w < 0 || w >= 64) w = 0;
#endif
    const Plan &pl = *c.pl;
    const Primes &PS = *c.PS;
    const size_t C = pl.C, nrows = pl.nrows, nblk = pl.nblk, Cs = C + 8;
    constexpr size_t PSB = (size_t)TB * SLOT, PSBO = (size_t)TB * SLOTO;
    const int prm = c.prm;                                            /* row-major product planes (§26) */
    const size_t nbi = prm ? ((nblk + 7) & ~(size_t)7) : nblk;        /* item space per prime, padded to whole 8-block chunks */
    LineAsm rla; if(prm) lasm_init(rla, ws, nrows);
    DRFlush rf; rf.la = &rla; rf.i = rf.n = 0; rf.rms = c.rms; rf.psb = 0;
    static_assert(!WritebackA || (Recipe==0 && !MID));
    if constexpr(WritebackA) ::sbn::v3::require(!c.frontier[0], SBN3_FATAL_MATH, "paired A writeback frontier");
    const int afwd = !c.frontier[0];          /* fresh handle: A's forward in the tile */
    V *ycol = SALLOC(ws, V, 2 * Cs), *acol = Recipe==1?ycol:SALLOC(ws, V, 2 * Cs);
    V *yextra=nullptr,*aextra=nullptr;
    if constexpr(Recipe==2){yextra=SALLOC(ws,V,2*Cs);aextra=SALLOC(ws,V,2*Cs);}   /* rings of 2: 2 MB + image + tables = 3.1 MB per core (§21) */
    uint8_t *stg = SALLOC(ws, uint8_t, nrows * (PSBO > PSB ? PSBO : PSB) + 128); stg = (uint8_t *)(((uintptr_t)stg + 63) & ~(uintptr_t)63);
    uint64_t *Tb = SALLOC(ws, uint64_t, 4 * C);
    uint64_t *T0 = Tb, *TL0 = Tb + C, *T1 = Tb + 2 * C, *TL1 = Tb + 3 * C;
    const Pk52 K = pk52_mk();
    DSweep swy, swa; swy.vr = swy.n = 0; swa.vr = swa.n = 0; swy.K = swa.K = K; swy.nsl = swa.nsl = TB; swy.str = swa.str = 1;
    DFlush fl; fl.i = fl.n = 0;
    uint8_t *fl_dst_pending = NULL;                                    /* image of the previous block awaiting its flush */
    for(; lo < hi; ++lo){
        const int q = (int)(lo / nbi);
        const size_t blk = (size_t)(lo % nbi);
        if(blk >= nblk) continue;                                       /* padding item (prm) */
        const Prime &P = PS.P[q]; const PrimeV &pv = PS.V[q];
        uint8_t *yb = c.fp[q] + blk * pl.bstride;
        const uint8_t *ab = c.hp[q] + blk * c.hbs;
        uint8_t *fdst = c.op[q] ? (prm ? c.op[q] + blk * PSBO : c.op[q] + blk * c.obs) : yb;   /* product region: row-major run / separate planes / in place */
        const int oo = c.op[q] != NULL; const size_t psbo = oo ? PSBO : PSB;
        const size_t nfl = nrows * psbo / 64;
        rf.psb = psbo;
        auto flush_cfg = [&](uint8_t *dst){ if(prm){ rf.dst0 = dst; rf.src = stg; rf.i = 0; rf.n = nrows; } else { fl.dst = dst; fl.src = stg; fl.i = 0; fl.n = nfl; } };
        auto flush_fin = [&](){ if(prm) drflush_finish(rf); else dflush_finish(fl); };
        const int last = !(lo + 1 < hi && (size_t)((lo + 1) % nbi) < nblk);   /* no further real item in this chunk */
        V *y0 = ycol, *y1 = ycol + Cs, *a0 = acol, *a1 = acol + Cs;
        TP_T(t0);
        swy.d0 = y0; swy.d1 = y1; swy.rb = Recipe==1?ab:yb; swy.vr = 0; swy.n = nrows; dsweep_finish(swy);
        TP_T(t1); TP_ADD(gather, t0, t1);
        if constexpr(Recipe==2){
            DSweep sw{yextra,yextra+Cs,c.fp1[q]+blk*pl.bstride,0,nrows,K,TB,1};dsweep_finish(sw);
            sw.d0=aextra;sw.d1=aextra+Cs;sw.rb=c.hp1[q]+blk*c.hbs;sw.vr=0;dsweep_finish(sw);
        }
        for(size_t s = 0; s < TB; ++s){
            const size_t b = blk * TB + s;
            V *y = s == 0 ? y0 : y1, *a = s == 0 ? a0 : a1;
            col_tw_build<false>(T0, TL0, P, pl.lgC, C, b, pv);
            col_tw_build<true>(T1, TL1, P, pl.lgC, C, b, pv);
            TP_T(f0);
            if(Recipe!=1 || afwd){
                if constexpr(!MID) colpass_t<false>((uint64_t *)y, C, T0, pv, &P, NULL); else colpass_t<false>((uint64_t *)y, C, T1, pv, &P, NULL);
                if(c.counts)++c.counts[w].column_forward;
            }
            TP_T(f1); TP_ADD(colf, f0, f1);
            if(Recipe!=1 && s == 0){ TP_T(g0); swa.d0 = a0; swa.d1 = a1; swa.rb = ab; swa.vr = 0; swa.n = nrows; dsweep_finish(swa); TP_T(g1); TP_ADD(gather, g0, g1); }
            TP_T(l0);
            if(Recipe!=1 && afwd){colpass_t<false>((uint64_t *)a, C, T0, pv, &P, NULL);if(c.counts)++c.counts[w].column_forward;}
            V *ye=nullptr,*ae=nullptr;
            if constexpr(Recipe==2){
                ye=yextra+s*Cs;ae=aextra+s*Cs;
                colpass_t<false>((uint64_t *)ye,C,T0,pv,&P,NULL);
                if(!c.frontier[1])colpass_t<false>((uint64_t *)ae,C,T0,pv,&P,NULL);
                if(c.counts)c.counts[w].column_forward+=1+!c.frontier[1];
            }
            TP_T(l1); TP_ADD(lanes, l0, l1);
            /* inverse with the products in its leaf blocks (the y column is not re-read through L3 for them) */

            using Leaf=TileProductLeaf<MID,Recipe,Scaled>;
            Leaf lf{};lf.y=y;lf.a=a;lf.ye=ye;lf.ae=ae;lf.r8=TL0;lf.ir8=TL1;lf.pv=&pv;
            if constexpr(Scaled){lf.kc=vset(c.leaf_k[0][q]);lf.kr=vset(c.leaf_rec[0][q]);
                if constexpr(Recipe==2){lf.kc1=vset(c.leaf_k[1][q]);lf.kr1=vset(c.leaf_rec[1][q]);}}
            if(c.counts){++c.counts[w].column_inverse;c.counts[w].leaf_products+=(C/8)*(Recipe==2?2:1);}
            TP_T(i0);
            if constexpr(!MID) colpass_x<true, 0, 0, Leaf>((uint64_t *)y, C, T1, P, pv, NULL, NULL, NULL, PSB, K, lf);
            else               colpass_x<true, 0, 0, Leaf>((uint64_t *)y, C, T0, P, pv, NULL, NULL, NULL, PSB, K, lf);
            if(s == 0 && fl_dst_pending){ flush_cfg(fl_dst_pending); flush_fin(); fl_dst_pending = NULL; }   /* the image must be free before slot 0 packs */
            TP_T(i1); TP_ADD(coli, i0, i1);
            TP_T(k0);
            for(size_t vr = 0; vr < nrows; ++vr) (oo ? stslot<SLOTO>(stg + vr * PSBO + s * SLOTO, y[vr], K, pv) : st52(stg + vr * PSB + s * SLOT, y[vr], K, pv));
            TP_T(k1); TP_ADD(scatter, k0, k1);
        }
        if constexpr(WritebackA) {
            // Both slots have been gathered, transformed and consumed. A's
            // column buffers are read-only to conv8x8, so their spectrum can
            // now replace the owned row plane for the second product.
            static_assert(UR*TB==16 && (16*SLOT)%64==0);
            auto *saved=const_cast<uint8_t *>(ab);
            for(size_t vr0=0;vr0<nrows;vr0+=UR) {
                alignas(64) uint8_t unit[16*SLOT];
                for(size_t u=0;u<UR;++u) {
                    st52(unit+(u*TB)*SLOT,a0[vr0+u],K,pv);
                    st52(unit+(u*TB+1)*SLOT,a1[vr0+u],K,pv);
                }
                st_lines(saved+vr0*PSB,unit,sizeof unit,c.nts);
            }
        }
        fl_dst_pending = fdst;                                           /* flushed during the next item's slot 0 */
        if(last){ flush_cfg(fdst); TP_T(z0); flush_fin(); TP_T(z1); TP_ADD(scatter, z0, z1); fl_dst_pending = NULL; }
#ifdef CR_TILE_PROF
        g_tp[w].slots += TB; g_tp[w].vecs += TB * C;
#endif
    }
    if(prm) lasm_flush(rla);                                          /* partial lines at a prime's tail only */
    _mm_sfence();
}

} // namespace sbn::v3::SBN3_P48_NS
