/* Imported from labs/cr/cr_geom.hpp; source hash in planning/imports-2026-09-06.json. */
#pragma once
/* cr_geom.hpp — plan geometry for the three arms (DESIGN.md §0).
 *
 * Tower: N = 8·Lg·C·M2 trunks, Lg = L (leaf-L, L in {5,6,7}) or 1 (native
 * L = 8: the 8 lanes ARE the leaf ring). C = 2^lgC columns, M2 slots per
 * (virtual) row; nrows = C·Lg virtual rows of lbp packed slots.
 *
 * Trunk layout (native):  trunk 8v + t = vector v = cv·C + r, lane t
 *        (leaf-L):        trunk (leaf·L + s), leaf = 8(cv·C + r) + t,
 *                         stored at virtual row r·L + s, slot cv, lane t
 *
 * Window contract (both MP arms, bit-identical outputs):
 *   t0 = 8·floor((nat−1)/8), window trunks [t0, 8Zv), nwtp = 8Zv − t0,
 *   emitted as ntp = roundup(nwtp + 2, OT) trunks = nl limbs;
 *   cert obits = T·t0, hbits = T·nwt (nwt = 8Zv − nat + 1),
 *   ebits = T + 1 + clog2(nat). */
#include "tables.hpp"
#include "common/coefficient_bound.hpp"
#include "backend/ntt_p48/geometry_policy.hpp"
enum { SLOT = CR_SLOT };   /* plane slot bytes: 48 canonical 6-byte residues (the engine) or 64 unpacked (cr_codec.hpp) */
/* product-plane slot bytes of the fused LIN path (NOTES §21 hybrid: packed input planes for the
 * tile's sweeps, 64-bit product planes for the fused gather); default = SLOT (in place) */
enum { SLOTO = CR_SLOTO };
/* Blocked planes (NOTES §12): block j = slots [j·TB, (j+1)·TB) of every
 * virtual row, contiguous (nrows × TB slots), block stride bstride:
 *     addr(vr, slot) = (slot / TB)·bstride + (vr·TB + slot % TB)·SLOT
 * bstride = nrows·TB·SLOT + 64 is an ODD number of lines (the walks at
 * block stride — rows, irow — must not alias one L1/L2 set). The
 * whole-line unit is UR = 16/TB virtual rows × TB slots = 16 slots = 12
 * lines (SLOT 48) / 16 lines (SLOT 64); irow groups IROW_G rows so that
 * a group's bytes per block are whole lines. */
static_assert(CR_BPAD % 2 == 1, "block padding must be an odd number of lines");
enum { TB = CR_TB, UR = 16 / TB, IROW_G = TB >= 4 ? 4 : 8, IROW_W = CR_IROW_W };   /* IROW_W rows interleaved per irow batch (1, 2, 4) */
static_assert(IROW_G % IROW_W == 0, "irow batch");
static_assert(TB == 1 || TB == 2 || TB == 4 || TB == 8 || TB == 16, "block width");
#include <stdlib.h>
#include <math.h>

namespace sbn::v3::SBN3_P48_NS {

enum Arm { ARM_LIN = 0, ARM_CYC = 1, ARM_TMP = 2 };

struct Plan {
    Arm arm;
    int T, LW, BL, BL16;        /* trunk bits, lo width, limbs per 8 trunks (0 if not whole), per 16 */
    int WD;                     /* 64-bit words per T-bit digit = (T + 63) / 64 (cr_emit.hpp, §32) */
    int OT, OL;                 /* emit quantum: OT trunks = OL limbs     */
    int L;                      /* 5,6,7 leaf-L; 8 native                 */
    size_t Lg;                  /* L or 1                                  */
    size_t lgC, C, M2, M, N;
    size_t nrows, lbv, lbw, plane_bytes;        /* lbw = written slots (multiple of TB) */
    size_t nblk, bstride;       /* blocks, block stride in bytes (cr_geom.hpp header) */
    size_t bstride_o, plane_bytes_o;   /* product planes at SLOTO (fused path, when SLOTO != SLOT) */
    size_t lbp;                 /* FROZEN-layout row stride (passes-mode A/B only)    */
    int full;                   /* lbv == M2                               */
    int mrow_unnorm;            /* TMP full fill: swapped-table fwd mrow   */
    int hspec;                  /* handle holds the FULL A spectrum (columns
                                 * + lanes done at build; tile skips them)  */
    /* problem */
    size_t an, yn, nat, nyt;    /* cached / fresh limbs & trunks           */
    size_t natv_a, natv_y;      /* decode vectors (natural 8-trunk)        */
    size_t Zv, Zw, nwt, nwtp, t0, dlt;   /* mid window                     */
    size_t pc;                  /* LIN product trunks                      */
    size_t need;                /* trunks the tower must cover             */
    size_t ntp, nl, outcap;     /* emit trunks (OT-padded), limbs          */
    size_t voff;                /* emit view offset (trunks, LIN/CYC)      */
    size_t vtrunks;             /* valid view trunks (beyond → zero)       */
    size_t ring_rn;             /* CYC ring mode: rn (0 = MP window)       */
    /* Garner scale per prime: inv(C·rowscale·lanescale)·2^52 */
    uint64_t rowscale;          /* M2 (unnormalized row inverse) or 1      */
};

inline int clog2(size_t n){ int k = 0; while(((size_t)1 << k) < n) ++k; return k; }

/* A row has at least one line of readable padding. Adding a single line to an arbitrary
 * rounded payload does not guarantee an odd stride (lbw=3924 gave 0x2e000). Bit mask:
 * CR_ODDSTRIDE=1 product rows, =2 transposed inputs, =3 both; =0 is the historical layout.
 * CR_STRIDE_OVERRIDE is a benchmark-only expression for interleaving layouts at joined
 * operation boundaries; production builds use the fixed native layout. */
inline size_t padded_row_stride(size_t bytes, int kind){
#ifdef CR_STRIDE_OVERRIDE
    const int mask = CR_STRIDE_OVERRIDE;
#else
    static const int mask = 3;
#endif
    size_t lines = (bytes + 63) / 64 + 1;
    if(mask & kind) lines |= 1;
    return lines * 64;
}

/* CRT range: an output coefficient is a sum of <= min(nat, nyt) products of two T-bit trunks and
 * must stay below prod(PR[0..NP)) (2^191.996 at NP = 4: T = 88 to 2^15.99 trunks, 84 to 2^23.99,
 * 80 to 2^31.99). */
/* Exact integer capacity, independent of FP rounding: m*(2^T-1)^2 < P.
 * The old logarithmic predicate required a rounding margin; the wide integer
 * comparison removes that approximation, not the strict CRT inequality. */
inline constexpr auto crt_product = coefficient_bound::prime_product(PR, NP);
inline int plan_T_ok(int T, size_t nat, size_t nyt){
    const size_t m = nat < nyt ? nat : nyt;
    return T >= 0 && coefficient_bound::fits(crt_product, unsigned(T), m);
}
/* codec constants for T. NP = 4: T ∈ {80, 84, 88} (16 trunks = whole limbs → T ≡ 0 mod 4; a vector
 * starts at an intra-limb offset of 0 or 32: two decode variants; T = 80 has its byte-permute
 * funnel). NP ≥ 5 (§32): T ≡ 0 mod 8 in [96, 184] (byte-aligned trunks, the wide codec of
 * cr_codec.hpp, ≤ 3 loads per vector), with 48·NP − 2T ≤ 64 so that the top digit b2 of a trunk
 * value is one word. WD = words per T-digit. Emit quantum: OL = 8k limbs with 512k ≡ 0 mod T
 * (k = T/gcd(512,T): 80→40, 84→168, 88→88, 104→104, 128→8, 152→152, 176→88), OT = OL·64/T
 * trunks; Regroup holds 48 output vectors (OL ≤ 384). */
inline int plan_codec(Plan &pl, int T){
    if(T % 4 || T < 68 || T > 232) return 0;
    if(T > 88 && T % 8) return 0;
    if(48 * NP - 2 * T > 64 || 2 * T >= 48 * NP) return 0;
    pl.T = T; pl.LW = T - 52; pl.BL = (8 * T) % 64 == 0 ? 8 * T / 64 : 0;
    pl.BL16 = T / 4;
    pl.WD = (T + 63) / 64;
    int g = 512, b = T; while(b){ int r = g % b; g = b; b = r; }     /* gcd(512, T) */
    const int k = T / g;
    pl.OL = 8 * k; pl.OT = pl.OL * 64 / T;
    if(pl.OL > 384) return 0;                                          /* Regroup tables (48 output vectors) */
    return 1;
}
/* T candidates: NP = 4 {88, 84, 80}; NP ≥ 5 {24·NP − 8 … 24·NP − 32} by 8 (§32). The widest T the
 * exact CRT bound admits is chosen (plan_T_auto); the bench's --T forces one. */
inline int plan_T_max(void){ return NP == 4 ? 88 : 24 * NP - 8; }
inline int plan_T_min(void){ return NP == 4 ? 80 : 24 * NP - 32; }
inline int plan_T_step(void){ return NP == 4 ? 4 : 8; }
inline int plan_T_auto(size_t an, size_t yn){
    for(int T = plan_T_max(); T >= plan_T_min(); T -= plan_T_step()){
        const size_t nat = (an * 64 + (size_t)T - 1) / (size_t)T, nyt = (yn * 64 + (size_t)T - 1) / (size_t)T;
        if(plan_T_ok(T, nat, nyt)) return T;
    }
    return plan_T_min();
}

/* tower (L, lgM) with the m2t knee → C, M2; lgC in [3, 13], M2 >= 8 */
inline int plan_tower(Plan &pl, int L, int lgM, int m2t, int nocap = 0){
    pl.L = L; pl.Lg = L == 8 ? 1 : (size_t)L;
    /* same knee for leaf-L: measured at 1M/W16 the tile costs 99.9 TSC
     * cycles per column-vector at C=256 (1536 virtual rows) vs 135.7 at
     * C=64 — long strided row runs hurt more than the L2 spill */
    int lgC = lgM - m2t;
    const int min_lgC = nocap ? 3 : 6;
    if(lgC < min_lgC) lgC = min_lgC; /* automatic knee keeps C>=64; exact C>=8 is legal */
    /* tile working set: (C·Lg virtual rows) × 8-slot runs must stay
     * L2/L3-friendly — cap C·Lg at ~2048 rows (po2 C <= 2048, leaf-L
     * C <= 256; the E1 sweep showed leaf-L towers losing 10-25% at
     * lgM 16-18 with C = 512..2048 × L rows) */
    /* po2 column cap: 11 in the sweep band; in the deep band the balanced
     * split lgC = lgM/2 wins (2^28×2^28 W16: M2 32768 → 2.28 s, 8192 →
     * 1.64 s, 1024 → 2.42 s): rows of M2 vectors must stay L2-sized and
     * the tile's column runs must not explode either (NOTES §6) */
    int lgCcap = pl.Lg > 1 ? 8 : (lgM / 2 > 11 ? lgM / 2 : 11);
    if(nocap) lgCcap = 13;                 /* the §28 LIN rule sets lgC itself */
    if(lgC > lgCcap) lgC = lgCcap;
    if(lgC > 17) lgC = 17;
    if(lgC > lgM - 4) lgC = lgM - 4;      /* M2 >= 16: lbp <= M2 always */
    if(lgC < 3) return 0;
    pl.lgC = (size_t)lgC; pl.C = (size_t)1 << lgC;
    pl.M = (size_t)1 << lgM; pl.M2 = pl.M >> lgC;
    if(pl.M2 < 16) return 0;
    pl.N = 8 * pl.Lg * pl.M;
    pl.nrows = pl.C * pl.Lg;
    return 1;
}

inline void plan_finish_rows(Plan &pl, size_t keep_trunks){
    const size_t per_slot = 8 * pl.Lg * pl.C;       /* trunks per slot */
    size_t lbv = (keep_trunks + per_slot - 1) / per_slot;
    if(lbv > pl.M2) lbv = pl.M2;
    if(lbv < 1) lbv = 1;
    pl.lbv = lbv;
    pl.lbw = (lbv + TB - 1) & ~(size_t)(TB - 1);
    pl.nblk = pl.lbw / TB;
    /* CR_BPAD lines of padding per block (odd → the block walks never share an L1/L2 set; a bit-rich
     * count such as 255 also spreads consecutive blocks over more DRAM channel/bank hash classes than
     * the sparse 2^k·SLOT strides do — NOTES §25) */
    pl.bstride = pl.nrows * (size_t)TB * (size_t)SLOT + 64 * (size_t)CR_BPAD;
    pl.full = lbv == pl.M2;
    pl.plane_bytes = pl.nblk * pl.bstride + 128;
    pl.bstride_o = pl.nrows * (size_t)TB * (size_t)SLOTO + 64 * (size_t)CR_BPAD;
    pl.plane_bytes_o = pl.nblk * pl.bstride_o + 128;
    /* frozen row-major stride (passes mode): 16-slot granules, odd count */
    { const size_t lbw16 = (lbv + 15) & ~(size_t)15;
      if(SLOT == 64) pl.lbp = lbw16 + 1;
      else pl.lbp = lbw16 + ((((lbw16 / 16) & 1) == 0) ? 16 : 0); }
}

/* tower chooser over {5,6,7,8} x lgM: the cheapest by the measured cost
 * model (fill sweep, W=16):  cost = N · kappa_L · g(f),  f = need/N,
 *   g(f) = 1 (full or f >= 0.95: the near-full rule lands on full fill)
 *        = 0.27 + 0.73·f (truncated arms: LIN, TMP)
 *   kappa_8 = 1.00, kappa_{5,6,7} = 1.05 (leaf-L per-point premium)
 * CYC (always full fill) uses g = 1. Ties → smaller N. */
inline double tower_cost(size_t need, int L, size_t N, int truncated){
    const double f = (double)need / (double)N;
    const double kappa = L == 8 ? 1.0 : 1.12;   /* leaf-L per-point premium (E1: 1.05-1.17) */
    /* truncation efficiency g(f) = a + (1-a)·f with the fixed share a
     * SHRINKING with size (fill sweeps, W=16: a ≈ 0.27 at N = 2^19
     * trunks, ≈ 0.05 at 2^21: the full-fill pipeline is DRAM-bound at
     * scale while the truncated one moves ~f of the bytes) */
    double lgN = 0; while(((size_t)1 << (size_t)lgN) < N) lgN += 1;
    double a = 0.27 - 0.11 * (lgN - 19.0);
    if(a < 0.03) a = 0.03;
    if(a > 0.35) a = 0.35;
    const double g = (!truncated || f >= 0.95) ? 1.0 : (a + (1.0 - a) * f);
    return (double)N * kappa * g;
}
inline int tower_choose(size_t need, unsigned Lset, int *Lout, int *lgMout, int lgMmin, int truncated){
    double best = 0; size_t bestN = 0; int bl = 0, blg = 0;
    static const int Ls[4] = { 8, 5, 6, 7 };
    for(int i = 0; i < 4; ++i){
        const int L = Ls[i];
        if(!(Lset & (1u << (L - 5)))) continue;
        const size_t Lg = L == 8 ? 1 : (size_t)L;
        for(int lg = lgMmin; lg <= 27; ++lg){
            const size_t N = 8 * Lg * ((size_t)1 << lg);
            if(N < need) continue;
            const double cst = tower_cost(need, L, N, truncated);
            if(!bestN || cst < best - 1e-9 || (cst < best + 1e-9 && N < bestN)){ best = cst; bestN = N; bl = L; blg = lg; }
            break;                                  /* larger lg only costs more */
        }
    }
    if(!bestN) return 0;
    *Lout = bl; *lgMout = blg;
    return 1;
}

/* MP window quantities from (nat, nyt) */
inline void plan_window(Plan &pl){
    pl.Zv = (pl.nyt + 7) / 8;
    pl.nwt = 8 * pl.Zv + 1 - pl.nat;             /* requires 8Zv >= nat */
    pl.t0 = 8 * ((pl.nat - 1) / 8);
    pl.dlt = (pl.nat - 1) - pl.t0;
    pl.nwtp = 8 * pl.Zv - pl.t0;
    pl.Zw = pl.nwtp / 8;
}

/* build a plan. arm LIN: product an×yn. CYC/TMP: MP(an cached, yn fresh).
 * L = 0: choose over Lset; else forced L. lgM = 0: smallest; else forced.
 * m2t: the knee (lgM − lgC). ring_rn != 0 (CYC): ring mode a·b mod 2^(64rn)−1
 * with N = rn·64/T exactly. */
inline int plan_make(Plan &pl, Arm arm, size_t an, size_t yn, int T, int L, int lgM,
                     int m2t, unsigned Lset, size_t ring_rn, int force_full = 0, int no_nearfull = 0, int exact_geometry = 0, size_t coefficient_terms = 0){
    memset(&pl, 0, sizeof pl);
    pl.arm = arm;
    if(!plan_codec(pl, T)) return 0;
    pl.an = an; pl.yn = yn;
    pl.nat = (an * 64 + T - 1) / T;
    pl.nyt = (yn * 64 + T - 1) / T;
    pl.natv_a = (pl.nat + 7) / 8;
    pl.natv_y = (pl.nyt + 7) / 8;
    if(!plan_T_ok(T, coefficient_terms?coefficient_terms:pl.nat, coefficient_terms?coefficient_terms:pl.nyt)) return 0;     /* CRT range (above) */
    pl.ring_rn = ring_rn;
    size_t need;
    if(arm == ARM_LIN){
        pl.pc = pl.nat + pl.nyt - 1;
        need = pl.pc;
    }else if(arm == ARM_CYC && ring_rn){
        if(an>ring_rn || yn>ring_rn || (ring_rn*64)%T) return 0;
        need=ring_rn*64/T;
    }else{
        if(pl.nat < 9) return 0;
        plan_window(pl);
        if(!ring_rn && 8 * pl.Zv < pl.nat) return 0;   /* nwt >= 1 */
        if(arm == ARM_CYC){
            if(ring_rn){
                if((ring_rn * 64) % T) return 0;
                need = ring_rn * 64 / T;
            }else
                need = 8 * pl.Zv + pl.dlt;        /* wrap-exact window: N >= nyt + dlt */
        }else
            need = 8 * pl.Zv + pl.dlt;            /* nat + nwtp − 1 = 8Zv + dlt */
    }
    pl.need = need;
    int Lc = L, lg = lgM;
    if(!Lc || !lg){
        int L2, lg2;
        if(!tower_choose(need, L ? (1u << (L - 5)) : Lset, &L2, &lg2, 6, arm != ARM_CYC)) return 0;
        if(!Lc) Lc = L2;
        if(!lg) lg = lg2;
    }
    const auto geometry = p48_geometry::choose_rows(arm == ARM_LIN, arm == ARM_CYC,
        Lc, lg, m2t, need, exact_geometry, force_full);
    lg = geometry.log_m; m2t = geometry.row_log; force_full = geometry.force_full;
    const int nocap = geometry.uncapped;
    if(!plan_tower(pl, Lc, lg, m2t, nocap)) return 0;
    if(pl.N < need) return 0;
    if(arm == ARM_CYC && ring_rn && pl.N != need) return 0;
    /* kept rows */
    if(arm == ARM_CYC) plan_finish_rows(pl, pl.N);            /* full fill */
    else plan_finish_rows(pl, need);
    /* TMP: a near-full truncated inverse^T pays ~2 b52/pair on its chain
     * of near-full nodes; above 95% fill the swapped-table full forward
     * (1 b52/pair, deferred scale) is cheaper than the truncation saves */
    if(!no_nearfull && (arm == ARM_TMP || (arm == ARM_LIN && (Lc != 8))) && pl.lbv * p48_geometry::near_full_denominator > pl.M2 * p48_geometry::near_full_numerator) plan_finish_rows(pl, pl.N);   /* §28: LIN po2 never rounds up to full */
    if(force_full) plan_finish_rows(pl, pl.N);
    if(8 * pl.Lg * pl.C * pl.lbv < need) return 0;
    if(arm == ARM_LIN && 8 * pl.natv_a > pl.N) return 0;
    /* emit view */
    const size_t OT = (size_t)pl.OT;
    if(arm == ARM_LIN){
        pl.voff = 0;
        pl.ntp = ((pl.pc + 2 + OT - 1) / OT) * OT;
        pl.vtrunks = pl.pc;
        pl.nl = pl.ntp * (size_t)T / 64;
        pl.outcap = an + yn;
        if(pl.outcap > pl.nl) return 0;
        pl.rowscale = 1;
    }else if(arm == ARM_CYC && ring_rn){
        pl.voff = 0;
        pl.ntp = ((pl.N + 16 + OT - 1) / OT) * OT;
        pl.vtrunks = pl.N;
        pl.nl = pl.ntp * (size_t)T / 64;
        pl.outcap = ring_rn;
        pl.rowscale = pl.M2;
    }else{
        pl.voff = pl.t0;
        pl.ntp = ((pl.nwtp + 2 + OT - 1) / OT) * OT;
        pl.vtrunks = pl.nwtp;
        pl.nl = pl.ntp * (size_t)T / 64;
        pl.outcap = pl.nl;
        pl.rowscale = arm == ARM_CYC ? pl.M2 : 1;
    }
    pl.mrow_unnorm = 0;
    if(arm == ARM_TMP && pl.full){ pl.mrow_unnorm = 1; pl.rowscale = pl.M2; }
    if(arm == ARM_LIN && pl.full) pl.rowscale = pl.M2;      /* unnormalized full irow */
    pl.hspec = 0;   /* 0: A's rows in the handle, columns inside the tile */
    return 1;
}

/* One-dimensional transform of the x^8 coefficient vectors. It has
 * a distinct geometry and dense completed-coefficient view, not a Bailey
 * shape with relaxed validation. Root k's leaf ring is x^8 - w[k]. */
inline int plan_flat(Plan &pl, Arm arm, size_t an,size_t yn,int T,size_t coefficient_terms=0,size_t fixed_M2=0,size_t ring_rn=0){
    memset(&pl,0,sizeof pl);if(!plan_codec(pl,T) || !an || !yn)return 0;
    pl.arm=arm;pl.an=an;pl.yn=yn;pl.nat=(an*64+T-1)/T;pl.nyt=(yn*64+T-1)/T;
    if(!plan_T_ok(T,coefficient_terms?coefficient_terms:pl.nat,coefficient_terms?coefficient_terms:pl.nyt))return 0;
    pl.natv_a=(pl.nat+7)/8;pl.natv_y=(pl.nyt+7)/8;
    if(arm==ARM_LIN)pl.pc=pl.need=pl.nat+pl.nyt-1;
    else if(arm==ARM_CYC && ring_rn){
        if(an>ring_rn || yn>ring_rn || (ring_rn*64)%T)return 0;
        pl.ring_rn=ring_rn;pl.need=ring_rn*64/T;
    }
    else {if(arm!=ARM_TMP || pl.nat<9)return 0;plan_window(pl);if(8*pl.Zv<pl.nat)return 0;pl.need=8*pl.Zv+pl.dlt;}
    pl.C=1;pl.lgC=0;pl.nrows=1;pl.L=8;pl.Lg=1;pl.lbv=(pl.need+7)/8;pl.M2=16;
    while(pl.M2<pl.lbv)pl.M2*=2;
    if(fixed_M2){if(fixed_M2<pl.M2 || (fixed_M2&(fixed_M2-1)))return 0;pl.M2=fixed_M2;}
    if(pl.M2>(size_t(1)<<20))return 0;
    pl.M=pl.M2;pl.N=8*pl.M;pl.lbw=(pl.lbv+7)&~size_t(7);pl.nblk=pl.lbw/TB;pl.full=pl.lbv==pl.M2;
    if(arm==ARM_CYC && (!ring_rn || pl.N!=pl.need))return 0;
    pl.bstride=pl.bstride_o=TB*SLOT;pl.plane_bytes=pl.plane_bytes_o=pl.lbw*SLOT+128;pl.lbp=pl.lbw;
    pl.rowscale=pl.full?pl.M2:1;pl.mrow_unnorm=arm==ARM_TMP && pl.full;
    pl.voff=arm==ARM_TMP?pl.t0:0;pl.vtrunks=arm==ARM_TMP?pl.nwtp:pl.pc;
    pl.ntp=((pl.vtrunks+2+pl.OT-1)/pl.OT)*pl.OT;pl.nl=pl.ntp*T/64;pl.outcap=arm==ARM_TMP?pl.nl:an+yn;
    if(arm==ARM_CYC){pl.vtrunks=pl.N;pl.ntp=((pl.N+16+pl.OT-1)/pl.OT)*pl.OT;pl.nl=pl.ntp*T/64;pl.outcap=ring_rn;}
    return pl.outcap<=pl.nl;
}

/* byte offset of (virtual row vr, slot) in a plane of this plan */
__attribute__((always_inline)) inline
size_t plane_off(const Plan &pl, size_t vr, size_t slot){
    return (slot / TB) * pl.bstride + (vr * TB + (slot & (TB - 1))) * (size_t)SLOT;
}

inline size_t plan_tower_entries(const Plan &pl){
    /* po2: rows read w[2j] (j < M2), columns read the shared prefix
     * w[0..2C) plus the factor tables (Primes::factors_for(M2)) — the full
     * tower is never touched (NOTES §10); leaf-L lane roots still read
     * w[8·(C·b+a) + t] (< 8M) */
    return pl.Lg == 1 ? 2 * (pl.M2 > pl.C ? pl.M2 : pl.C) : 8 * pl.M;
}

/* the smallest ring rn >= target on the tower grid for T */
inline size_t ring_next(size_t target, int T, unsigned Lset, int *Lout, int *lgMout){
    size_t best = 0; int bl = 0, blg = 0;
    static const int Ls[4] = { 8, 5, 6, 7 };
    for(int i = 0; i < 4; ++i){
        const int L = Ls[i];
        if(!(Lset & (1u << (L - 5)))) continue;
        const size_t Lg = L == 8 ? 1 : (size_t)L;
        for(int lg = 6; lg <= 27; ++lg){
            const size_t N = 8 * Lg * ((size_t)1 << lg);
            const size_t rn = N * (size_t)T / 64;
            if(rn < target) continue;
            if(!best || rn < best){ best = rn; bl = L; blg = lg; }
            break;
        }
    }
    if(Lout) *Lout = bl;
    if(lgMout) *lgMout = blg;
    return best;
}

} // namespace sbn::v3::SBN3_P48_NS
