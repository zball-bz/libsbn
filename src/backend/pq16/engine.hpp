/* Imported from libsbn/include/sbn/detail/engine/pq16.h; retain v2 numerical guards. Tables/Frame/team are explicit. */
#pragma once
#include "backend/pq16/roots.hpp"
namespace sbn::v3::pq16 {
#ifndef PQ16_CENTER_DIRECT
#define PQ16_CENTER_DIRECT 3
#endif
#ifndef PQ16_PFA_PARALLEL
#define PQ16_PFA_PARALLEL 0
#endif
#ifndef PQ16_PFA_FINE
#define PQ16_PFA_FINE 0
#endif
// 0: staged; 1: direct input; 2: direct input/output; 3: direct at W <= 2.
// The wider teams retain staging: it improves their PFA gather locality.
static_assert(PQ16_CENTER_DIRECT >= 0 && PQ16_CENTER_DIRECT <= 3);
static constexpr bool pq16_direct_input(unsigned workers){
    return PQ16_CENTER_DIRECT == 3 ? workers <= 2 : PQ16_CENTER_DIRECT != 0;
}
static constexpr bool pq16_direct_output(unsigned workers){
    return PQ16_CENTER_DIRECT >= 2 && pq16_direct_input(workers);
}

// pq16.h - AVX-512 PQ FFT bigint multiply (clean rewrite of fft16.h).
// u64-limb I/O, u16-digit internals. Public entry:
//   pq16_mul_r(rp, ap, an, bp, bn, scratch*)  /  pq16_mul(...) [thread scratch]
// returns 1 on success, 0 if an+bn is outside the supported band (caller
// falls back to the toom route). Persistent twiddle tables live in a
// thread-local plan; ALL per-call workspace comes from scratch.h.
//
// ---- algorithm + optimization inventory (every choice was pinned by
//      microbench + full-mul A/B on the fft16.h ancestor; see bench/) ----
//
// codec    operands are consumed as u16 digits (4 per limb). N complex =
//          2*(an+bn) via even/odd packing and PQ real-convolution recovery:
//          digits (2k, 2k+1) form
//          one complex point, the pointwise step evaluates the product at
//          partner pairs (k, N/2-k) sharing cross-terms (pq_eval_pair).
//          Scale 1/N is folded into the pointwise constants.
// layout   AoSoV tiles of 8 complex per zmm pair [re0..7 | im0..7], 128 B.
//          Forward is DIF, root e^{+2*pi*i/N}, natural in / bit-reversed out.
// stages   fwd: fused u16-decode + radix-2^2 unroll-2 at len N (operands are
//          read straight from the caller's buffers with fault-suppressed
//          masked tails -- no zero-padded copy) -> radix-2^3 passes -> leaf.
//          leaf 32x2/64/128 by lg(n) parity: across-stage + 8x8 transpose +
//          lane-parallel DFT8. Leaves skip the transpose-back ("pi layout"):
//          stored tile q lane t = canonical-BR position t*8+q; the pointwise
//          table is built in pi order and the inverse leaves consume pi
//          directly, so both transposes cancel out of the transform.
//          pointwise is fused with the inverse leaves (mirror group pairs
//          multiplied then immediately inverse-leafed cache-hot); the final
//          inverse radix-2^2 at len N is fused with the u16 emit and a
//          base-2^64 SWAR carry chain per quarter-front + junction fixups,
//          so no canonical coefficient array is ever materialized.
// tables   tw22[lg]/tw8[lg] keyed by stage length, shared across transform
//          sizes; PQ g-table is prefix-stable in canonical-BR index, so the
//          largest build serves every smaller n as a prefix. Stage lengths
//          >= 2^PQ16_TWC_MIN_LG store COMPACT tables (2x smaller; squares
//          derived in-kernel by short-depth reconstruction) -- a few cmuls per group against
//          megabytes of table stream at the DRAM-bound sizes. (A 4x scheme
//          deriving W4 = (W1^2)^2 at ~3 ulp fails the adversarial probe at
//          the centered cap by one rounding flip -- do not revisit blindly.)
// blocking r8 stages run through a recursive 3-tier ladder (~L3/L2/L1):
//          stages longer than the tier run as full-span sweeps, everything
//          below recurses per chunk, so each chunk finishes its remaining
//          stages + leaves (and on the inverse side starts them) while
//          cache-resident.
// PFA      Good-Thomas M in {3,5,7} x pow2 branch fills the octave between
//          powers of two (worst-case grid padding 14% instead of 100%).
//          M=3 input decodes 3 stripes with constant lane-residue masked
//          merges; M=5/7 use the twiddle-fixup form (y_b(l) = sum_s w^{b r_s}
//          w^{b l} t_s(l)) with stripe pointers held in residue order so the
//          per-iter relabel is pure GPR renaming and all vector slots are
//          compile-time constants. Pointwise pairs branch b with branch M-b
//          (antipodal mirror) rotating the shared g-table by omega_M^b.
//          M=5/7 emit pre-rotates by conj(w^{b l}) and stages both sub-tiles
//          in an L1 buffer (the register-resident variant spills at M=7).
//          Losing variants removed: radix-3x2 fused input/emit (-6..9%),
//          register-rotating emit (spills), see fft16.h history.
// centered transforms above 2^17 complex run on centered digits d - 2^15
//          (4x precision headroom, probed to 2^19 complex = 131072x131072
//          limbs with all-max and sparse adversarial operands). Centering is
//          a 1-op XOR with 0x8000 folded into a staging copy; the decode
//          sign-extends via a byte-shuffle + arithmetic shift. The signed
//          coefficient c^ is corrected as c = c^ + (S<<15) - (C<<30) with S
//          a running digit-window sum (segmented: rising/middle/falling
//          windows need half the loads and a closed-form C) and C the window
//          overlap length. The centered band emits FLAT: one natural-order
//          pass (mem_ir22 / pfa_natural) then a single sequential emit
//          stream + one carry chain -- the fused 4-front emit's ~25 streams
//          go latency-bound right where the band begins. Operand + result
//          staging double as layout normalizers: reading caller buffers
//          in-place in this BW-bound band cost up to 2x depending on
//          allocation layout. PFA-7's direct-form butterfly fails first
//          UNcentered (7*2^14 all-max); centered headroom re-admits it.
// gates    pow2/PFA-3/5 hold to 2^17 uncentered; PFA-7 capped at 2^16
//          uncentered; everything to 2^19 centered. All empirical.






// double-vector companions to the types.h macro set (aligned forms; every
// tile is 64-byte aligned by construction)
#define load_dvec(p)     sb__fn(load_pd)((const void *)(p))
#define store_dvec(p, v) sb__fn(store_pd)((void *)(p), (v))
#define fmadd(a, b, c)   sb__fn(fmadd_pd)((a), (b), (c))
#define fmsub(a, b, c)   sb__fn(fmsub_pd)((a), (b), (c))
#define permd(idx, a)    sb__fn(permutexvar_pd)((idx), (a))

// ------------------------------------------------------------------
// complex tile: 8 complex per zmm pair
// ------------------------------------------------------------------
typedef struct { sb_dvec re, im; } qcv;

static inline qcv q_ld(const double *p){
    qcv r = { load_dvec(p), load_dvec(p + 8) };
    return r;
}
static inline void q_st(double *p, qcv x){
    store_dvec(p, x.re); store_dvec(p + 8, x.im);
}
static inline qcv q_add(qcv a, qcv b){
    qcv r = { sb_add(a.re, b.re), sb_add(a.im, b.im) };
    return r;
}
static inline qcv q_sub(qcv a, qcv b){
    qcv r = { sb_sub(a.re, b.re), sb_sub(a.im, b.im) };
    return r;
}
static inline qcv q_mul(qcv a, qcv w){            // a * w
    qcv r = { fmsub(a.re, w.re, sb_mul(a.im, w.im)),
              fmadd(a.re, w.im, sb_mul(a.im, w.re)) };
    return r;
}
static inline qcv q_mulc(qcv a, qcv w){           // a * conj(w)
    qcv r = { fmadd(a.im, w.im, sb_mul(a.re, w.re)),
              fmsub(a.im, w.re, sb_mul(a.re, w.im)) };
    return r;
}
static inline qcv q_j(qcv a){                     // +i * a
    qcv r = { sb_sub(sb_dzero(), a.im), a.re };
    return r;
}
static inline qcv q_mj(qcv a){                    // -i * a
    qcv r = { a.im, sb_sub(sb_dzero(), a.re) };
    return r;
}
static inline qcv q_rev(qcv a){                   // lane reverse
    const sb_vec ix = sb_setr_64(7, 6, 5, 4, 3, 2, 1, 0);
    qcv r = { permd(ix, a.re), permd(ix, a.im) };
    return r;
}

#define PQ16_C8 0x1.6a09e667f3bcdp-1  /* correctly rounded cos(pi/4) */

// rot8[t](a) = a * e^{+2*pi*i*t/8}, t = 1, 3 (t = 0 id, t = 2 = q_j)
static inline qcv q_rot81(qcv a){
    const sb_dvec c = sb_set1_d(PQ16_C8);
    qcv r = { sb_mul(c, sb_sub(a.re, a.im)), sb_mul(c, sb_add(a.re, a.im)) };
    return r;
}
static inline qcv q_rot83(qcv a){
    const sb_dvec c = sb_set1_d(PQ16_C8);
    qcv r = { sb_sub(sb_dzero(), sb_mul(c, sb_add(a.re, a.im))), sb_mul(c, sb_sub(a.re, a.im)) };
    return r;
}
// conjugate rotations for the inverse
static inline qcv q_irot81(qcv a){
    const sb_dvec c = sb_set1_d(PQ16_C8);
    qcv r = { sb_mul(c, sb_add(a.re, a.im)), sb_mul(c, sb_sub(a.im, a.re)) };
    return r;
}
static inline qcv q_irot83(qcv a){
    const sb_dvec c = sb_set1_d(PQ16_C8);
    qcv r = { sb_mul(c, sb_sub(a.im, a.re)), sb_sub(sb_dzero(), sb_mul(c, sb_add(a.re, a.im))) };
    return r;
}

// ------------------------------------------------------------------
// radix-2^2 quad butterflies -- THE shared primitive: the fused input,
// in-memory r22 passes, r8 stages, all leaves and the fused emit are
// instances of these two on different twiddle sets.
// DIF forward: v = { x0+x2+(x1+x3),  ((x0+x2)-(x1+x3))w2,
//                    (x0-x2)w1+(x1-x3)jw1,  ((x0-x2)w1-(x1-x3)jw1)w2 }
// ------------------------------------------------------------------
static inline void q_bf4(qcv v[4], qcv w1, qcv w2){
    qcv b0 = q_add(v[0], v[2]), b2 = q_mul(q_sub(v[0], v[2]), w1);
    qcv b1 = q_add(v[1], v[3]), b3 = q_mul(q_sub(v[1], v[3]), q_j(w1));
    v[0] = q_add(b0, b1); v[1] = q_mul(q_sub(b0, b1), w2);
    v[2] = q_add(b2, b3); v[3] = q_mul(q_sub(b2, b3), w2);
}
static inline void q_ibf4(qcv v[4], qcv w1, qcv w2){
    qcv s  = q_mulc(v[1], w2);
    qcv b0 = q_add(v[0], s), b1 = q_sub(v[0], s);
    s      = q_mulc(v[3], w2);
    qcv b2 = q_add(v[2], s), b3 = q_sub(v[2], s);
    s = q_mulc(b2, w1);
    v[0] = q_add(b0, s); v[2] = q_sub(b0, s);
    s = q_mulc(b3, q_j(w1));
    v[1] = q_add(b1, s); v[3] = q_sub(b1, s);
}

// radix-2^3 on 8 tiles: level-1 split (W1/W1o rotations) + two quads.
// Outputs: v[0..3] = even half, v[4..7] = odd half (DIF slot order).
static inline void q_bf8(qcv v[8], qcv W1, qcv W1o, qcv W2, qcv W4){
    qcv e[4], o[4];
    e[0] = q_add(v[0], v[4]); e[1] = q_add(v[1], v[5]);
    e[2] = q_add(v[2], v[6]); e[3] = q_add(v[3], v[7]);
    o[0] = q_mul(q_sub(v[0], v[4]), W1);
    o[1] = q_mul(q_sub(v[1], v[5]), W1o);
    o[2] = q_j(q_mul(q_sub(v[2], v[6]), W1));
    o[3] = q_j(q_mul(q_sub(v[3], v[7]), W1o));
    q_bf4(e, W2, W4);
    q_bf4(o, W2, W4);
    for(int i = 0; i < 4; ++i){ v[i] = e[i]; v[4 + i] = o[i]; }
}
static inline void q_ibf8(qcv v[8], qcv W1, qcv W1o, qcv W2, qcv W4){
    q_ibf4(v, W2, W4);
    q_ibf4(v + 4, W2, W4);
    qcv o0 = q_mulc(v[4], W1);
    qcv o1 = q_mulc(v[5], W1o);
    qcv o2 = q_mulc(q_mj(v[6]), W1);
    qcv o3 = q_mulc(q_mj(v[7]), W1o);
    qcv e0 = v[0], e1 = v[1], e2 = v[2], e3 = v[3];
    v[0] = q_add(e0, o0); v[4] = q_sub(e0, o0);
    v[1] = q_add(e1, o1); v[5] = q_sub(e1, o1);
    v[2] = q_add(e2, o2); v[6] = q_sub(e2, o2);
    v[3] = q_add(e3, o3); v[7] = q_sub(e3, o3);
}

// ------------------------------------------------------------------
// 8x8 double transpose (24 shuffles)
// ------------------------------------------------------------------
static inline void q_tr8(sb_dvec r[8]){
    sb_dvec t0 = sb__fn(unpacklo_pd)(r[0], r[1]);
    sb_dvec t1 = sb__fn(unpackhi_pd)(r[0], r[1]);
    sb_dvec t2 = sb__fn(unpacklo_pd)(r[2], r[3]);
    sb_dvec t3 = sb__fn(unpackhi_pd)(r[2], r[3]);
    sb_dvec t4 = sb__fn(unpacklo_pd)(r[4], r[5]);
    sb_dvec t5 = sb__fn(unpackhi_pd)(r[4], r[5]);
    sb_dvec t6 = sb__fn(unpacklo_pd)(r[6], r[7]);
    sb_dvec t7 = sb__fn(unpackhi_pd)(r[6], r[7]);

    sb_dvec v0 = sb__fn(shuffle_f64x2)(t0, t2, 0x44);  // cols 0,2 lo
    sb_dvec v1 = sb__fn(shuffle_f64x2)(t4, t6, 0x44);
    sb_dvec v2 = sb__fn(shuffle_f64x2)(t0, t2, 0xEE);  // cols 4,6 hi
    sb_dvec v3 = sb__fn(shuffle_f64x2)(t4, t6, 0xEE);
    sb_dvec v4 = sb__fn(shuffle_f64x2)(t1, t3, 0x44);  // cols 1,3
    sb_dvec v5 = sb__fn(shuffle_f64x2)(t5, t7, 0x44);
    sb_dvec v6 = sb__fn(shuffle_f64x2)(t1, t3, 0xEE);  // cols 5,7
    sb_dvec v7 = sb__fn(shuffle_f64x2)(t5, t7, 0xEE);

    r[0] = sb__fn(shuffle_f64x2)(v0, v1, 0x88);
    r[2] = sb__fn(shuffle_f64x2)(v0, v1, 0xDD);
    r[4] = sb__fn(shuffle_f64x2)(v2, v3, 0x88);
    r[6] = sb__fn(shuffle_f64x2)(v2, v3, 0xDD);
    r[1] = sb__fn(shuffle_f64x2)(v4, v5, 0x88);
    r[3] = sb__fn(shuffle_f64x2)(v4, v5, 0xDD);
    r[5] = sb__fn(shuffle_f64x2)(v6, v7, 0x88);
    r[7] = sb__fn(shuffle_f64x2)(v6, v7, 0xDD);
}
static inline void q_tr8cv(qcv v[8]){
    sb_dvec re[8], im[8];
    for(int i = 0; i < 8; ++i){ re[i] = v[i].re; im[i] = v[i].im; }
    q_tr8(re); q_tr8(im);
    for(int i = 0; i < 8; ++i){ v[i].re = re[i]; v[i].im = im[i]; }
}

// ------------------------------------------------------------------
// lane-parallel 8-point DFT across 8 vectors (DIF, e^{+} root, outputs
// in sequential = bit-reversed slot order). Unscaled inverse.
// ------------------------------------------------------------------
static inline void q_dft8(qcv v[8]){
    qcv e0 = q_add(v[0], v[4]), o0 = q_sub(v[0], v[4]);
    qcv e1 = q_add(v[1], v[5]), o1 = q_rot81(q_sub(v[1], v[5]));
    qcv e2 = q_add(v[2], v[6]), o2 = q_j(q_sub(v[2], v[6]));
    qcv e3 = q_add(v[3], v[7]), o3 = q_rot83(q_sub(v[3], v[7]));

    qcv ee0 = q_add(e0, e2), eo0 = q_sub(e0, e2);
    qcv ee1 = q_add(e1, e3), eo1 = q_j(q_sub(e1, e3));
    qcv oe0 = q_add(o0, o2), oo0 = q_sub(o0, o2);
    qcv oe1 = q_add(o1, o3), oo1 = q_j(q_sub(o1, o3));

    v[0] = q_add(ee0, ee1); v[1] = q_sub(ee0, ee1);
    v[2] = q_add(eo0, eo1); v[3] = q_sub(eo0, eo1);
    v[4] = q_add(oe0, oe1); v[5] = q_sub(oe0, oe1);
    v[6] = q_add(oo0, oo1); v[7] = q_sub(oo0, oo1);
}
static inline void q_idft8(qcv v[8]){
    qcv ee0 = q_add(v[0], v[1]), ee1 = q_sub(v[0], v[1]);
    qcv eo0 = q_add(v[2], v[3]), eo1 = q_sub(v[2], v[3]);
    qcv oe0 = q_add(v[4], v[5]), oe1 = q_sub(v[4], v[5]);
    qcv oo0 = q_add(v[6], v[7]), oo1 = q_sub(v[6], v[7]);

    qcv e0 = q_add(ee0, eo0), e2 = q_sub(ee0, eo0);
    qcv t  = q_mj(eo1);
    qcv e1 = q_add(ee1, t),   e3 = q_sub(ee1, t);
    qcv o0 = q_add(oe0, oo0), o2 = q_sub(oe0, oo0);
    t      = q_mj(oo1);
    qcv o1 = q_add(oe1, t),   o3 = q_sub(oe1, t);

    o1 = q_irot81(o1); o2 = q_mj(o2); o3 = q_irot83(o3);
    v[0] = q_add(e0, o0); v[4] = q_sub(e0, o0);
    v[1] = q_add(e1, o1); v[5] = q_sub(e1, o1);
    v[2] = q_add(e2, o2); v[6] = q_sub(e2, o2);
    v[3] = q_add(e3, o3); v[7] = q_sub(e3, o3);
}

// ------------------------------------------------------------------
// plan: persistent twiddle tables + decode shuffles (thread-local,
// grown once, shared across transform sizes). NO per-call workspace
// lives here -- that comes from scratch.
// ------------------------------------------------------------------
#define PQ16_MAX_R8 4
#define PQ16_MAX_LG 21
#ifndef PQ16_FORCE_CENTERED
#define PQ16_FORCE_CENTERED 0   // test hook: centered codec at every size
#endif

// per-transform stage shape, derived from lg(n) alone
typedef struct {
    uint32_t leaf;               // 32, 64 or 128
    uint32_t cnt;                // number of r8 ancestor stages
    uint32_t len[PQ16_MAX_R8];   // their lengths, descending
} pq16_shape;

static inline pq16_shape pq16_shape_of(uint32_t n){
    unsigned lgn = (unsigned)__builtin_ctz(n);
    pq16_shape s;
    if(lgn == 6){ s.leaf = 64; s.cnt = 0; return s; }   // branch 64 (wide-codec odd radix, 2026-09-09): the leaf alone, no radix-4 stem
    uint32_t leaf_lg = 5u + ((lgn - 7u) % 3u);    // branch >= 128; the radix-4 stem feeds 32/64/128 leaves
    s.leaf = 1u << leaf_lg;
    s.cnt  = (lgn - 2u - leaf_lg) / 3u;
    uint32_t len = n / 4u;
    for(uint32_t i = 0; i < s.cnt; ++i, len /= 8u) s.len[i] = len;
    return s;
}

// Twiddle tables are shared across transform sizes and grown once:
//  - tw22[lg]: len-2^lg r22 unroll-2 table (input + final-emit stage of a
//    size-2^lg transform). Depends on len only.
//  - tw8[lg]:  len-2^lg r8 stage table. Depends on len only.
//  - pq: pi-ordered pointwise g-table. Prefix-stable in the canonical-BR
//    index (bitrev(k,b)/2^b is width-independent), so the largest build
//    serves every smaller n as a prefix.
typedef struct {
    Frame *builder;
    uint32_t pq_n;               // pow2 branch size the PQ table covers
    double  *tw22[PQ16_MAX_LG];
    double  *tw8[PQ16_MAX_LG];
    double  *pq;                 // compact PQ g-table: per 8-tile group,
                                 // 32 doubles [re x16 | im x16]
    // leaf constants (transform-size independent)
    alignas(64) double l32w[2 * 16];   // w1, w2 at len 32
    alignas(64) double l64w[4 * 16];   // W1, W1*w8, W2, W4 at len 64
    alignas(64) double l128w[8 * 16];  // {W1, W2} at len 128, t = 0..3
    alignas(64) double gh0[16];        // head g-values: g(p>>2), p = 0..7
    alignas(64) double wt5[5 * 16];    // PFA lane fixup w_5^{b*(l%5)}
    alignas(64) double wt7[7 * 16];    // PFA lane fixup w_7^{b*(l%7)}
    sb_vec dec_idx[4];             // u16 digit decode shuffles
    sb_vec dec_idxc[4];            // centered: digit bytes at dword 2-3
    int  leaf_init;
    uint32_t ready_mask;           // bit lg set == ensure(2^lg) fully done
                                   // (shared-plan fast path, acquire/release)
} pq16_plan;

static inline void *pq16_alloc(pq16_plan *pl,size_t bytes){
    require(pl->builder,SBN3_FATAL_LIFETIME,"pq16 table construction only");
    return pl->builder->allocate(bytes,128);
}

/* pre-ensure cap for the pi-ordered g-table (NORMATIVE, 03-amendments
 * A4 note): built ONCE at the band cap on first use, so pq/gh0 never
 * regrow -- the free()+rebuild use-after-free class is gone by
 * construction. Must equal PQ16_MAX_N_C (guarded where it's defined). */
#ifndef PQ16_PLAN_CAP_BRANCH
#define PQ16_PLAN_CAP_BRANCH (1u << 19)
#endif

static inline uint32_t pq16_bitrev(uint32_t x, unsigned bits){
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    x = (x << 16) | (x >> 16);
    return bits ? (x >> (32u - bits)) : 0u;
}

// root e^{+2*pi*i*k/n}
static inline void pq16_root(double *wr, double *wi, uint64_t k, uint64_t n){
    // Preparation only; callers use power-of-two 8 <= n <= 2^19. Reduce in
    // integers, then compose two extended-precision seeds on the first
    // octant. No transcendental calls and no growing recurrence chain.
    static_assert(__LDBL_MANT_DIG__>=64,"pow2 seed composition needs extended precision");
    static_assert(PQ16_PLAN_CAP_BRANCH<=(1u<<PQ16_ROOT_GRID_LOG2),"extend the root seed grid with the numerical domain");
    k %= n;
    const uint64_t quarter=n/4,quadrant=k/quarter;
    k %= quarter;
    const bool reflect=k>n/8;
    if(reflect)k=quarter-k;
    const uint64_t index=k<<(PQ16_ROOT_GRID_LOG2-(unsigned)__builtin_ctzll(n));
    const auto *a=PQ16_ROOT_COARSE[index>>8],*b=PQ16_ROOT_FINE[index&255];
    double c=(double)(a[0]*b[0]-a[1]*b[1]);
    double s=(double)(a[0]*b[1]+a[1]*b[0]);
    if(reflect){const double t=c;c=s;s=t;}
    switch(quadrant){
    case 0:*wr=c;*wi=s;break;
    case 1:*wr=-s;*wi=c;break;
    case 2:*wr=-c;*wi=-s;break;
    default:*wr=s;*wi=-c;break;
    }
}
// one twiddle cv: lanes l = root(scale*(j+l), len) at dst[0..7 | 8..15]
static inline void pq16_twcv(double *dst, uint64_t j, uint64_t scale, uint64_t len){
    for(int l = 0; l < 8; ++l)
        pq16_root(dst + l, dst + 8 + l, (j + (uint64_t)l) * scale, len);
}
// same multiplied by e^{+2*pi*i/8}
static inline void pq16_twcv8(double *dst, uint64_t j, uint64_t scale, uint64_t len){
    for(int l = 0; l < 8; ++l){
        pq16_root(dst+l,dst+8+l,(j+(uint64_t)l)*scale+len/8,len);
    }
}

static inline void pq16_leaf_init(pq16_plan *pl){
    if(pl->leaf_init) return;
    pq16_twcv (pl->l32w + 0,  0, 1, 32);
    pq16_twcv (pl->l32w + 16, 0, 2, 32);
    pq16_twcv (pl->l64w + 0,  0, 1, 64);
    pq16_twcv8(pl->l64w + 16, 0, 1, 64);
    pq16_twcv (pl->l64w + 32, 0, 2, 64);
    pq16_twcv (pl->l64w + 48, 0, 4, 64);
    for(int t = 0; t < 4; ++t){
        pq16_twcv(pl->l128w + 32 * t,      (uint64_t)(8 * t), 1, 128);
        pq16_twcv(pl->l128w + 32 * t + 16, (uint64_t)(8 * t), 2, 128);
    }
    // PFA lane-twiddle fixups: wtM[b] lane l = e^{-2*pi*i*b*(l%M)/M}
    for(int b = 0; b < 5; ++b)
        for(int l = 0; l < 8; ++l){
            const unsigned k=(b*l)%5;
            pl->wt5[16*b+l]=PQ16_W5_RE[k];pl->wt5[16*b+8+l]=PQ16_W5_IM[k];
        }
    for(int b = 0; b < 7; ++b)
        for(int l = 0; l < 8; ++l){
            const unsigned k=(b*l)%7;
            pl->wt7[16*b+l]=PQ16_W7_RE[k];pl->wt7[16*b+8+l]=PQ16_W7_IM[k];
        }
    // digit decode: parity x tile -> 8 dword slots of (2-byte digit, 2 zero)
    for(int v = 0; v < 4; ++v){
        int parity = v & 1, tile = v >> 1;
        char idx[64];
        for(int l = 0; l < 8; ++l){
            int digit = tile * 16 + 2 * l + parity;
            idx[l * 4 + 0] = (char)(digit * 2);
            idx[l * 4 + 1] = (char)(digit * 2 + 1);
            idx[l * 4 + 2] = (char)0x40;          // zero source
            idx[l * 4 + 3] = (char)0x40;
        }
        for(int i = 32; i < 64; ++i) idx[i] = (char)0x40;
        memcpy(&pl->dec_idx[v], idx, 64);
        // centered: digit bytes at dword bytes 2-3 (sign-extend via srai)
        char idc[64];
        for(int i = 0; i < 64; ++i) idc[i] = (char)0x40;
        for(int l = 0; l < 8; ++l){
            int digit = (v >> 1) * 16 + 2 * l + (v & 1);
            idc[l * 4 + 2] = (char)(digit * 2);
            idc[l * 4 + 3] = (char)(digit * 2 + 1);
        }
        memcpy(&pl->dec_idxc[v], idc, 64);
    }
    pl->leaf_init = 1;
}

// Tables for stage lengths >= 2^PQ16_TWC_MIN_LG are stored COMPACT (2x
// smaller); the missing entries use one square / one double-constant rot8.
// This is NOT a <=1-component-ulp bound: seed error is propagated and
// arithmetic adds rounding. See docs/twiddle-accuracy-probe-2026-09-07.md.
// The affected lengths sit inside the
// centered band's 4x precision headroom (adversarially re-probed).
#ifndef PQ16_TWC_MIN_LG
#define PQ16_TWC_MIN_LG 16
#endif
// len-2^lg r22 unroll-2 table: per iter {w1a, w2a, w1b, w2b}, or compact
// {w1a, w1b} (squares derived)
static inline void pq16_build_tw22(pq16_plan *pl, unsigned lg){
    if(pl->tw22[lg]) return;
    uint32_t len = 1u << lg;
    if(lg >= PQ16_TWC_MIN_LG){
        double *tab = (double *)pq16_alloc(pl,(size_t)len * 4);
        for(uint32_t i = 0; i < len / 64u; ++i){
            pq16_twcv(tab + 32u * i,       16u * i,      1, len);
            pq16_twcv(tab + 32u * i + 16u, 16u * i + 8u, 1, len);
        }
        pl->tw22[lg] = tab;
        return;
    }
    double *tab = (double *)pq16_alloc(pl,(size_t)len * 8);
    for(uint32_t i = 0; i < len / 64u; ++i){
        double  *d = tab + 64u * i;
        uint64_t j = 16u * i;
        pq16_twcv(d + 0,  j,     1, len);
        pq16_twcv(d + 16, j,     2, len);
        pq16_twcv(d + 32, j + 8, 1, len);
        pq16_twcv(d + 48, j + 8, 2, len);
    }
    pl->tw22[lg] = tab;
}
// len-2^lg r8 stage table: per iter {W1, W1*w8, W2, W4}, or compact
// {W1, W2} (W1o = rot8(W1), W4 = W2^2 derived)
static inline void pq16_build_tw8(pq16_plan *pl, unsigned lg){
    if(pl->tw8[lg]) return;
    uint32_t len = 1u << lg;
    if(lg >= PQ16_TWC_MIN_LG){
        double *tab = (double *)pq16_alloc(pl,(size_t)len * 4);
        for(uint32_t i = 0; i < len / 64u; ++i){
            pq16_twcv(tab + 32u * i,       8u * i, 1, len);
            pq16_twcv(tab + 32u * i + 16u, 8u * i, 2, len);
        }
        pl->tw8[lg] = tab;
        return;
    }
    double *tab = (double *)pq16_alloc(pl,(size_t)len * 8);
    for(uint32_t i = 0; i < len / 64u; ++i){
        double  *d = tab + 64u * i;
        uint64_t j = 8u * i;
        pq16_twcv (d + 0,  j, 1, len);
        pq16_twcv8(d + 16, j, 1, len);
        pq16_twcv (d + 32, j, 2, len);
        pq16_twcv (d + 48, j, 4, len);
    }
    pl->tw8[lg] = tab;
}
// stage-length predicate + per-iter table stride
static inline int    pq16_twc(uint32_t len){ return (unsigned)__builtin_ctz(len) >= PQ16_TWC_MIN_LG; }
static inline size_t pq16_tw_step(int twc){ return twc ? 32 : 64; }

// the 4 r22 twiddles for one unroll-2 iter
static inline void pq16_tw22_get(const double *twp, int twc,
                                 qcv *w1a, qcv *w2a, qcv *w1b, qcv *w2b){
    if(!twc){
        *w1a = q_ld(twp);      *w2a = q_ld(twp + 16);
        *w1b = q_ld(twp + 32); *w2b = q_ld(twp + 48);
        return;
    }
    *w1a = q_ld(twp);
    *w1b = q_ld(twp + 16);
    *w2a = q_mul(*w1a, *w1a);
    *w2b = q_mul(*w1b, *w1b);
}
// the 4 r8-stage twiddles
static inline void pq16_tw8_get(const double *twp, int twc,
                                qcv *W1, qcv *W1o, qcv *W2, qcv *W4){
    if(!twc){
        *W1 = q_ld(twp);      *W1o = q_ld(twp + 16);
        *W2 = q_ld(twp + 32); *W4  = q_ld(twp + 48);
        return;
    }
    *W1 = q_ld(twp);
    *W2 = q_ld(twp + 16);
    *W1o = q_rot81(*W1);
    *W4  = q_mul(*W2, *W2);
}

// grow every table the pow2 branch size needs (PFA shares them: the
// branch transform is plain pow2; the PQ table is keyed by branch too).
//
// SHARED-PLAN growth discipline (05-parallel par.2.2): the unlocked
// fast path is an acquire read of ready_mask (bit lg == ensure(2^lg)
// fully published); all building happens under the process plan mutex.
// The PQ g-table is built ONCE at PQ16_PLAN_CAP_BRANCH -- entries are
// keyed by the width-independent fraction bitrev(k,b)/2^b, so the cap
// build serves every smaller branch as a prefix (gh0 identically), and
// the table never regrows: no reader can ever see a moved/freed table.
#define PQ16_ODD_PLAN 1   /* odd-radix branches run pq16_odd_fwd / pq16_odd_inv (probe bridges key on this) */
typedef struct {
    uint32_t leaf, cnt;
    uint32_t len[PQ16_MAX_R8 + 1];
    uint32_t radix[PQ16_MAX_R8 + 1];
} pq16_odd_plan;
static inline pq16_odd_plan pq16_odd_plan_of(uint32_t n);
static inline void pq16_plan_ensure(pq16_plan *pl,uint32_t branch,int odd_plan){ // odd_plan: also the odd-radix branch plan tables
    require(pl->builder && !pl->ready_mask,SBN3_FATAL_LIFETIME,"pq16 immutable table build");
    const unsigned lgc=(unsigned)__builtin_ctz(branch);pq16_leaf_init(pl);
    pl->pq=(double *)pq16_alloc(pl,(size_t)branch*4);
    for(uint32_t g=0;g<branch/64;++g)for(uint32_t k=0;k<16;++k)
        pq16_root(pl->pq+32*g+k,pl->pq+32*g+16+k,pq16_bitrev(g*16+k,lgc-2),branch);
    for(uint32_t p=0;p<8;++p)pq16_root(pl->gh0+p,pl->gh0+8+p,pq16_bitrev(p>>2,lgc-2),branch);
    pl->pq_n=branch;pq16_build_tw22(pl,lgc);pq16_shape shape=pq16_shape_of(branch);
    for(uint32_t i=0;i<shape.cnt;++i)pq16_build_tw8(pl,(unsigned)__builtin_ctz(shape.len[i]));
    if(odd_plan){const pq16_odd_plan odd=pq16_odd_plan_of(branch); // odd-radix branch plan tables (pow2 shapes skip them)
     for(uint32_t i=0;i<odd.cnt;++i){const unsigned lg=(unsigned)__builtin_ctz(odd.len[i]);if(odd.radix[i]==8)pq16_build_tw8(pl,lg);else pq16_build_tw22(pl,lg);}}
    pl->ready_mask=1u<<lgc;pl->builder=nullptr;
}

// ------------------------------------------------------------------
// operand loaders + u16 digit decode. Operands are read straight from
// their buffers with fault-suppressed masked tails (no padded copy);
// rem = valid limbs remaining at p.
// ------------------------------------------------------------------
// Tail windows (rem < 8) never address past the operand when that would cross
// a page: a fault-suppressed masked load that touches an unmapped page costs a
// microcode assist (~70-140 ns per operand stream measured 2026-09-09 with
// guard-page buffers, 10-25 % of a 128-limb product). The window is then
// rebuilt from the one or two 64-byte-aligned blocks that hold its limbs (both
// mapped, since each contains a valid limb).
static inline sb_vec q_raw8(const uint64_t *p, int64_t rem){
    if(rem <= 0) return sb_zero(); // do not send an all-zero masked load to an unreadable page
    if(rem >= 8) return sb_load(p);
    const uintptr_t addr = (uintptr_t)p;
    if(((addr & 4095u) + 64u) <= 4096u) return sb_load(p, 0xFFu >> (8 - rem));   // stays inside its page
    const uint64_t *block = (const uint64_t *)(addr & ~(uintptr_t)63);
    const unsigned off = (unsigned)((addr & 63u) >> 3);
    const sb_vec lo = sb__fn(load_si512)((const void *)block);
    const sb_vec hi = off + (unsigned)rem > 8u ? sb__fn(load_si512)((const void *)(block + 8)) : lo;
    const sb_vec idx = sb_add(sb_setr_64(0, 1, 2, 3, 4, 5, 6, 7), sb_set1_64(off));
    return sb__fn(maskz_permutex2var_epi64)((__mmask8)(0xFFu >> (8 - rem)), lo, idx, hi);
}
// tail store of `rem` (< 8) limbs without addressing past them across a page boundary
static inline void q_st_tail(uint64_t *p, sb_vec v, int64_t rem){
    if(rem <= 0) return;
    if(rem >= 8){ sb_store(p, v); return; }
    const uintptr_t addr = (uintptr_t)p;
    if(((addr & 4095u) + 64u) <= 4096u){ sb_store(p, v, 0xFFu >> (8 - rem)); return; }
    alignas(64) uint64_t tmp[8]; sb_store(tmp, v);
    for(int64_t i = 0; i < rem; ++i) p[i] = tmp[i];
}
static inline sb_vec q_raw8_center(const uint64_t *p,int64_t rem,int cen){
    if(rem<=0)return sb_zero();
    sb_vec raw=q_raw8(p,rem);
    if constexpr(PQ16_CENTER_DIRECT)if(cen==2){const __mmask8 valid=rem>=8?0xff:__mmask8(0xffu>>(8-rem));
        raw=_mm512_mask_xor_epi64(raw,valid,raw,_mm512_set1_epi64(0x8000800080008000ull));}
    return raw;
}
static inline __mmask8 q_st_mask(int64_t rem){
    return sb_k8(rem >= 8 ? 0xFF : (rem <= 0 ? 0 : (0xFFu >> (8 - rem))));
}

// 8 digits of one zmm -> 8 doubles. Uncentered: digit at dword bytes 0-1,
// zero-extended. Centered: PRE-XORED digit at dword bytes 2-3; the
// arithmetic shift sign-extends, giving d - 2^15 in [-2^15, 2^15).
static inline sb_dvec q_dec1(sb_vec raw, sb_vec idx){
    sb_vec d = sb__fn(permutex2var_epi8)(raw, idx, sb_zero());
    return sb__fn(cvtepu32_pd)(sb__fn(castsi512_si256)(d));
}
static inline sb_dvec q_dec1c(sb_vec raw, sb_vec idx){
    sb_vec d = sb__fn(permutex2var_epi8)(raw, idx, sb_zero());
    return sb__fn(cvtepi32_pd)(sb__fn(castsi512_si256)(sb__fn(srai_epi32)(d, 16)));
}
// decode one zmm of limbs (16 complex) into 2 tiles
static inline void q_dec2(qcv *t0, qcv *t1, sb_vec raw, const sb_vec *IDX, int cen){
    if(cen){
        t0->re = q_dec1c(raw, IDX[0]); t0->im = q_dec1c(raw, IDX[1]);
        t1->re = q_dec1c(raw, IDX[2]); t1->im = q_dec1c(raw, IDX[3]);
    }else{
        t0->re = q_dec1(raw, IDX[0]); t0->im = q_dec1(raw, IDX[1]);
        t1->re = q_dec1(raw, IDX[2]); t1->im = q_dec1(raw, IDX[3]);
    }
}

// ------------------------------------------------------------------
// forward input stage: fused u16-decode + r22 unroll-2 at len n
// ------------------------------------------------------------------
static void pq16_input_rng(double *data, const uint64_t *src, int64_t cnt,
                           uint32_t n, const pq16_plan *pl, int cen,
                           uint32_t i0, uint32_t i1);
/* serial = the range form over [0, l/16) (audit stage 7: the two bodies
 * were verbatim clones; the range form with i0 = 0 folds the offsets) */
static inline void pq16_input_stage(double *data, const uint64_t *src,
                                    int64_t cnt, uint32_t n,
                                    const pq16_plan *pl, int cen){
    pq16_input_rng(data, src, cnt, n, pl, cen, 0, (n / 4u) / 16u);
}

/* G5-P4: input over iteration range [i0, i1) -- all four quarter
 * streams advance linearly with i, so a range invocation just offsets
 * the pointers (identical per-i work, disjoint stores). */
typedef struct {
    double *data;
    const uint64_t *src;
    int64_t cnt;
    uint32_t n;
    const pq16_plan *pl;
    int cen;
} pq16_in_fctx;
static void pq16_input_rng(double *data, const uint64_t *src, int64_t cnt,
                           uint32_t n, const pq16_plan *pl, int cen,
                           uint32_t i0, uint32_t i1){
    const uint32_t l = n / 4u;
    double *p[4];
    const uint64_t *s[4];
    int64_t r[4];
    for(uint32_t q = 0; q < 4; ++q){
        p[q] = data + 2u * (size_t)l * q + 32u * (size_t)i0;
        s[q] = src + (size_t)q * (l / 2u) + 8u * (size_t)i0;
        r[q] = cnt - (int64_t)q * (int64_t)(l / 2u) - (int64_t)(8u * i0);
    }
    const int twc = pq16_twc(n);
    const double *twp = pl->tw22[__builtin_ctz(n)]
                        + (size_t)i0 * pq16_tw_step(twc);
    const sb_vec *IDX = cen ? pl->dec_idxc : pl->dec_idx;
    for(uint32_t i = i0; i < i1; ++i){
        qcv a[4], b[4];
        for(int q = 0; q < 4; ++q){
            q_dec2(&a[q], &b[q], q_raw8_center(s[q], r[q],cen), IDX, cen);
            s[q] += 8; r[q] -= 8;
        }
        qcv w1a, w2a, w1b, w2b;
        pq16_tw22_get(twp, twc, &w1a, &w2a, &w1b, &w2b);
        twp += pq16_tw_step(twc);
        q_bf4(a, w1a, w2a);
        q_bf4(b, w1b, w2b);
        for(int q = 0; q < 4; ++q){
            q_st(p[q], a[q]);
            q_st(p[q] + 16, b[q]);
            p[q] += 32;
        }
    }
}
static void pq16_input_forfn(void *c_, uint64_t lo, uint64_t hi, int w,
                             scratch *ws){
    pq16_in_fctx *c = (pq16_in_fctx *)c_;
    (void)w; (void)ws;
    pq16_input_rng(c->data, c->src, c->cnt, c->n, c->pl, c->cen,
                   (uint32_t)lo, (uint32_t)hi);
}
static inline void pq16_input_stage_w(double *data, const uint64_t *src,
                                      int64_t cnt, uint32_t n,
                                      const pq16_plan *pl, int cen,
                                      sbn_team *t, int nthr){
    const uint32_t iters = (n / 4u) / 16u;
    if(t && nthr > 1 && iters >= 128){
        pq16_in_fctx c = { data, src, cnt, n, pl, cen };
        sbn_parallel_for(t, nthr, 0, iters, 16, SBN_FOR_STATIC,
                         pq16_input_forfn, &c);
    }else{
        pq16_input_rng(data, src, cnt, n, pl, cen, 0, iters);
    }
}

// ------------------------------------------------------------------
// in-memory r22 unroll-2 stage at len n (PFA branch first stage / the
// centered band's separated final inverse)
// ------------------------------------------------------------------
static inline void pq16_mem_r22(double *data, uint32_t n, const double *tw){
    const uint32_t l = n / 4u;
    double *p[4];
    for(int s = 0; s < 4; ++s) p[s] = data + 2u * (size_t)l * (uint32_t)s;
    const int twc = pq16_twc(n);
    for(uint32_t i = 0; i < l / 16u; ++i){
        qcv w1a, w2a, w1b, w2b;
        pq16_tw22_get(tw, twc, &w1a, &w2a, &w1b, &w2b);
        tw += pq16_tw_step(twc);
        for(int u = 0; u < 2; ++u){
            qcv v[4];
            for(int s = 0; s < 4; ++s) v[s] = q_ld(p[s] + 16 * u);
            q_bf4(v, u ? w1b : w1a, u ? w2b : w2a);
            for(int s = 0; s < 4; ++s) q_st(p[s] + 16 * u, v[s]);
        }
        for(int s = 0; s < 4; ++s) p[s] += 32;
    }
}
static void pq16_mem_ir22_rng(double *data, uint32_t n, const double *tw,
                              uint32_t i0, uint32_t i1);
static inline void pq16_mem_ir22(double *data, uint32_t n, const double *tw){
    pq16_mem_ir22_rng(data, n, tw, 0, (n / 4u) / 16u);
}

/* G5-P4: mem_ir22 over iteration range [i0, i1) -- the four quarter
 * streams and the twiddle pointer advance linearly with i */
typedef struct {
    double *data;
    const double *tw;
    uint32_t n;
} pq16_ir22_fctx;
static void pq16_mem_ir22_rng(double *data, uint32_t n, const double *tw,
                              uint32_t i0, uint32_t i1){
    const uint32_t l = n / 4u;
    double *p[4];
    for(int s = 0; s < 4; ++s)
        p[s] = data + 2u * (size_t)l * (uint32_t)s + 32u * (size_t)i0;
    const int twc = pq16_twc(n);
    tw += (size_t)i0 * pq16_tw_step(twc);
    for(uint32_t i = i0; i < i1; ++i){
        qcv w1a, w2a, w1b, w2b;
        pq16_tw22_get(tw, twc, &w1a, &w2a, &w1b, &w2b);
        tw += pq16_tw_step(twc);
        for(int u = 0; u < 2; ++u){
            qcv v[4];
            for(int s = 0; s < 4; ++s) v[s] = q_ld(p[s] + 16 * u);
            q_ibf4(v, u ? w1b : w1a, u ? w2b : w2a);
            for(int s = 0; s < 4; ++s) q_st(p[s] + 16 * u, v[s]);
        }
        for(int s = 0; s < 4; ++s) p[s] += 32;
    }
}
static void pq16_ir22_forfn(void *c_, uint64_t lo, uint64_t hi, int w,
                            scratch *ws){
    pq16_ir22_fctx *c = (pq16_ir22_fctx *)c_;
    (void)w; (void)ws;
    pq16_mem_ir22_rng(c->data, c->n, c->tw, (uint32_t)lo, (uint32_t)hi);
}
static inline void pq16_mem_ir22_w(double *data, uint32_t n,
                                   const double *tw, sbn_team *t, int nthr){
    const uint32_t iters = (n / 4u) / 16u;
    if(t && nthr > 1 && iters >= 128){
        pq16_ir22_fctx c = { data, tw, n };
        sbn_parallel_for(t, nthr, 0, iters, 16, SBN_FOR_STATIC,
                         pq16_ir22_forfn, &c);
    }else{
        pq16_mem_ir22_rng(data, n, tw, 0, iters);
    }
}

// ------------------------------------------------------------------
// PFA (Good-Thomas) radix-M butterflies, M in {3, 5, 7}.
// omega_M^k = exp(-2*pi*i*k/M) tables (reference int_fft values).
// ------------------------------------------------------------------
// radix-3 Winograd (e^- DFT convention)
static inline void q_pfa3(const qcv *x, qcv *y, int inv){
    const sb_dvec ch = sb_set1_d(-0.5);
    const sb_dvec sh = sb_set1_d(-PQ16_W3_IM[1]);
    qcv s = q_add(x[1], x[2]), d = q_sub(x[1], x[2]);
    y[0] = q_add(x[0], s);
    qcv u = { fmadd(ch, s.re, x[0].re), fmadd(ch, s.im, x[0].im) };
    qcv v = { sb_mul(sh, d.re), sb_mul(sh, d.im) };
    if(!inv){
        y[1].re = sb_add(u.re, v.im); y[1].im = sb_sub(u.im, v.re);
        y[2].re = sb_sub(u.re, v.im); y[2].im = sb_add(u.im, v.re);
    }else{
        y[1].re = sb_sub(u.re, v.im); y[1].im = sb_add(u.im, v.re);
        y[2].re = sb_add(u.re, v.im); y[2].im = sb_sub(u.im, v.re);
    }
}

// radix-5 Winograd (5 real mults, 17 adds)
static inline void q_pfa5(const qcv *x, qcv *y, int inv){
    const sb_dvec CP  = sb_set1_d(-0.25);
    const sb_dvec CM  = sb_set1_d(PQ16_W5_CM);
    const sb_dvec S1  = sb_set1_d(-PQ16_W5_IM[1]);
    const sb_dvec S1p = sb_set1_d(PQ16_W5_S1P);
    const sb_dvec S2m = sb_set1_d(PQ16_W5_S2M);
    qcv u14 = q_add(x[1], x[4]), v14 = q_sub(x[1], x[4]);
    qcv u25 = q_add(x[2], x[3]), v25 = q_sub(x[2], x[3]);
    qcv us = q_add(u14, u25), um = q_sub(u14, u25), vs = q_sub(v14, v25);
    qcv m1 = { sb_mul(us.re, CP),   sb_mul(us.im, CP) };
    qcv m2 = { sb_mul(um.re, CM),   sb_mul(um.im, CM) };
    qcv m3 = { sb_mul(vs.re, S1),   sb_mul(vs.im, S1) };
    qcv m4 = { sb_mul(v25.re, S1p), sb_mul(v25.im, S1p) };
    qcv m5 = { sb_mul(v14.re, S2m), sb_mul(v14.im, S2m) };
    qcv I14 = q_add(m3, m4), I23 = q_add(m3, m5);
    qcv mid = q_add(x[0], m1);
    qcv R14 = q_add(mid, m2), R23 = q_sub(mid, m2);
    y[0] = q_add(x[0], us);
    if(!inv){
        y[1].re = sb_add(R14.re, I14.im); y[1].im = sb_sub(R14.im, I14.re);
        y[4].re = sb_sub(R14.re, I14.im); y[4].im = sb_add(R14.im, I14.re);
        y[2].re = sb_add(R23.re, I23.im); y[2].im = sb_sub(R23.im, I23.re);
        y[3].re = sb_sub(R23.re, I23.im); y[3].im = sb_add(R23.im, I23.re);
    }else{
        y[1].re = sb_sub(R14.re, I14.im); y[1].im = sb_add(R14.im, I14.re);
        y[4].re = sb_add(R14.re, I14.im); y[4].im = sb_sub(R14.im, I14.re);
        y[2].re = sb_sub(R23.re, I23.im); y[2].im = sb_add(R23.im, I23.re);
        y[3].re = sb_add(R23.re, I23.im); y[3].im = sb_sub(R23.im, I23.re);
    }
}

// radix-7 direct form (chained FMAs; this is the precision-critical one:
// it fails first uncentered, see the PQ16_PFA7_MAX_N gate)
static inline void q_pfa7(const qcv *x, qcv *y, int inv){
    const sb_dvec C1 = sb_set1_d(PQ16_W7_RE[1]);
    const sb_dvec C2 = sb_set1_d(PQ16_W7_RE[2]);
    const sb_dvec C3 = sb_set1_d(PQ16_W7_RE[3]);
    const sb_dvec S1 = sb_set1_d(-PQ16_W7_IM[1]);
    const sb_dvec S2 = sb_set1_d(-PQ16_W7_IM[2]);
    const sb_dvec S3 = sb_set1_d(-PQ16_W7_IM[3]);
    qcv u16 = q_add(x[1], x[6]), v16 = q_sub(x[1], x[6]);
    qcv u25 = q_add(x[2], x[5]), v25 = q_sub(x[2], x[5]);
    qcv u34 = q_add(x[3], x[4]), v34 = q_sub(x[3], x[4]);
    qcv R1, R2, R3, I1, I2, I3;
#define PQ16_R(dst, cA, cB, cC) \
    dst.re = fmadd(cC, u34.re, fmadd(cB, u25.re, fmadd(cA, u16.re, x[0].re))); \
    dst.im = fmadd(cC, u34.im, fmadd(cB, u25.im, fmadd(cA, u16.im, x[0].im)))
    PQ16_R(R1, C1, C2, C3);
    PQ16_R(R2, C2, C3, C1);
    PQ16_R(R3, C3, C1, C2);
#undef PQ16_R
    I1.re = fmadd(S3, v34.re, fmadd(S2, v25.re, sb_mul(S1, v16.re)));
    I1.im = fmadd(S3, v34.im, fmadd(S2, v25.im, sb_mul(S1, v16.im)));
    I2.re = fmsub(S2, v16.re, fmadd(S1, v34.re, sb_mul(S3, v25.re)));
    I2.im = fmsub(S2, v16.im, fmadd(S1, v34.im, sb_mul(S3, v25.im)));
    I3.re = fmadd(S3, v16.re, fmsub(S2, v34.re, sb_mul(S1, v25.re)));
    I3.im = fmadd(S3, v16.im, fmsub(S2, v34.im, sb_mul(S1, v25.im)));
    y[0].re = sb_add(sb_add(x[0].re, u16.re), sb_add(u25.re, u34.re));
    y[0].im = sb_add(sb_add(x[0].im, u16.im), sb_add(u25.im, u34.im));
#define PQ16_P(ya, yb, R, I) \
    if(!inv){ ya.re = sb_add(R.re, I.im); ya.im = sb_sub(R.im, I.re); \
              yb.re = sb_sub(R.re, I.im); yb.im = sb_add(R.im, I.re); } \
    else    { ya.re = sb_sub(R.re, I.im); ya.im = sb_add(R.im, I.re); \
              yb.re = sb_add(R.re, I.im); yb.im = sb_sub(R.im, I.re); }
    PQ16_P(y[1], y[6], R1, I1)
    PQ16_P(y[2], y[5], R2, I2)
    PQ16_P(y[3], y[4], R3, I3)
#undef PQ16_P
}

static inline void q_pfa_bfly(const qcv *x, qcv *y, uint32_t M, int inv){
    if(M == 3)      q_pfa3(x, y, inv);
    else if(M == 5) q_pfa5(x, y, inv);
    else            q_pfa7(x, y, inv);
}

// lane-residue merge masks: km[d] selects lanes l with l % M == d
static inline void q_kmasks(uint8_t km[8], uint32_t M){
    for(uint32_t d = 0; d < M; ++d){
        uint8_t m = 0;
        for(uint32_t l = 0; l < 8; ++l)
            if(l % M == d) m |= (uint8_t)(1u << l);
        km[d] = m;
    }
}
// assemble the residue-(shifted) view: lane l takes y[(ph + l) mod M]
static inline qcv q_lane_merge(const qcv *y, uint32_t M, uint32_t ph,
                               const uint8_t *km){
    qcv o = y[0];
    for(uint32_t j = 0; j < M; ++j){
        uint32_t d = (j + M - ph) % M;
        o.re = sb__fn(mask_mov_pd)(o.re, sb_k8(km[d]), y[j].re);
        o.im = sb__fn(mask_mov_pd)(o.im, sb_k8(km[d]), y[j].im);
    }
    return o;
}

// PFA input, merge form (wins at M = 3): decode M natural-order stripes
// (stripe s = complex [s*n, (s+1)*n)) and apply the radix-M DFT across the
// a = k mod M dimension. Lane l of the stripe with lane-0 residue r has
// residue (r+l) mod M, so the per-residue inputs x[a] assemble from the
// stripes by constant lane-residue masks.
__attribute__((always_inline))
static inline void q_pfa_input_merge(double *data, const uint64_t *src,
                                     int64_t cnt, uint32_t n, uint32_t M,
                                     const pq16_plan *pl, int cen){
    uint8_t km[8]; q_kmasks(km, M);
    const uint64_t *s[7]; double *p[7]; uint32_t r[7]; int64_t rm[7];
    for(uint32_t b = 0; b < M; ++b){
        s[b]  = src + (size_t)b * (n / 2u);
        rm[b] = cnt - (int64_t)b * (int64_t)(n / 2u);
        p[b]  = data + 2u * (size_t)n * b;
        r[b]  = (uint32_t)(((uint64_t)b * n) % M);
    }
    const sb_vec *IDX = cen ? pl->dec_idxc : pl->dec_idx;
    for(uint32_t i = 0; i < n / 16u; ++i){
        qcv t0[7], t1[7];
        for(uint32_t b = 0; b < M; ++b){
            q_dec2(&t0[b], &t1[b], q_raw8_center(s[b], rm[b],cen), IDX, cen);
            s[b] += 8;
            rm[b] -= 8;
        }
        for(int u = 0; u < 2; ++u){
            const qcv *t = u ? t1 : t0;
            qcv x[7], y[7];
            for(uint32_t a = 0; a < M; ++a){
                x[a] = t[0];
                for(uint32_t b = 0; b < M; ++b){
                    uint32_t ru = (r[b] + (uint32_t)(8 * u)) % M;
                    uint32_t d  = (a + M - ru) % M;
                    x[a].re = sb__fn(mask_mov_pd)(x[a].re, sb_k8(km[d]), t[b].re);
                    x[a].im = sb__fn(mask_mov_pd)(x[a].im, sb_k8(km[d]), t[b].im);
                }
            }
            q_pfa_bfly(x, y, M, 0);
            for(uint32_t b = 0; b < M; ++b)
                q_st(p[b] + 16 * u, y[b]);
        }
        for(uint32_t b = 0; b < M; ++b){
            p[b] += 32;
            r[b] = (r[b] + 16u) % M;
        }
    }
}
// twiddle-fixup form (wins at M = 5, 7): the M-DFT runs directly on
// residue-relabeled stripe vectors, y[b](l) = sum_s w^{b r_s} w^{b l}
// t_s(l), so the M^2 lane merges collapse to M constant per-lane cmuls.
// Stripe POINTERS are held in residue order (sp[rho] = stripe whose lane-0
// residue is currently rho) and rotated by the constant 16 mod M each
// iteration: every vector-slot index is compile-time constant, the runtime
// relabel is pure GPR pointer renaming. Sub-tile 1 relabels by 8 mod M.
__attribute__((always_inline))
static inline void q_pfa_input_tw(double *data, const uint64_t *src,
                                  int64_t cnt, uint32_t n, uint32_t M,
                                  const double *wt, const pq16_plan *pl,
                                  int cen){
    const uint64_t *sp[7]; double *p[7]; int64_t rm[7];
    for(uint32_t b = 0; b < M; ++b){
        uint32_t rho = (uint32_t)(((uint64_t)b * n) % M);
        sp[rho] = src + (size_t)b * (n / 2u);
        rm[rho] = cnt - (int64_t)b * (int64_t)(n / 2u);
        p[b] = data + 2u * (size_t)n * b;
    }
    const sb_vec *IDX = cen ? pl->dec_idxc : pl->dec_idx;
    const uint32_t sh8 = 8u % M, sh16 = 16u % M;
    for(uint32_t i = 0; i < n / 16u; ++i){
        qcv t0[7], t1[7], x[7], y[7];
        for(uint32_t rho = 0; rho < M; ++rho){
            q_dec2(&t0[rho], &t1[rho], q_raw8_center(sp[rho], rm[rho],cen), IDX, cen);
            sp[rho] += 8;
            rm[rho] -= 8;
        }
        q_pfa_bfly(t0, y, M, 0);
        for(uint32_t b = 0; b < M; ++b){
            qcv w = q_ld(wt + 16u * b);
            q_st(p[b], q_mul(y[b], w));
        }
        for(uint32_t rho = 0; rho < M; ++rho)
            x[(rho + sh8) % M] = t1[rho];
        q_pfa_bfly(x, y, M, 0);
        for(uint32_t b = 0; b < M; ++b){
            qcv w = q_ld(wt + 16u * b);
            q_st(p[b] + 16, q_mul(y[b], w));
            p[b] += 32;
        }
        const uint64_t *tp[7]; int64_t tr[7];
        for(uint32_t rho = 0; rho < M; ++rho){
            tp[(rho + sh16) % M] = sp[rho];
            tr[(rho + sh16) % M] = rm[rho];
        }
        for(uint32_t rho = 0; rho < M; ++rho){ sp[rho] = tp[rho]; rm[rho] = tr[rho]; }
    }
}
// literal-M dispatch so the impls specialize (x[]/y[] enregister, the
// residue arithmetic strength-reduces). Merge vs tw split is end-to-end
// measured (bench/pfa_io_bench + full-mul A/B).
static inline void q_pfa_input(double *data, const uint64_t *src, int64_t cnt,
                               uint32_t n, uint32_t M, const pq16_plan *pl,
                               int cen){
    switch(M){
    case 3:  q_pfa_input_merge(data, src, cnt, n, 3u, pl, cen); break;
    case 5:  q_pfa_input_tw(data, src, cnt, n, 5u, pl->wt5, pl, cen); break;
    default: q_pfa_input_tw(data, src, cnt, n, 7u, pl->wt7, pl, cen); break;
    }
}

// ------------------------------------------------------------------
// radix-2^3 ancestor passes at length len
// ------------------------------------------------------------------
/* G5-P4: one r8 stage over the FLATTENED (block x group) item range
 * [k0, k1) -- every item touches its own disjoint 8x16-double slots,
 * the parallel decomposition unit for the above-tier full-span sweeps.
 * inv selects the inverse butterfly. Bit-identical to the nested loops
 * (same ops, same per-item twiddles, disjoint writes). */
static void pq16_r8_stage_rng(double *data, uint32_t len, const double *tw,
                              int inv, size_t k0, size_t k1){
    const size_t st = 2u * (size_t)(len / 8u);
    const int twc = pq16_twc(len);
    const size_t gpb = (size_t)(len / 64u);
    const size_t tws = pq16_tw_step(twc);
    for(size_t k = k0; k < k1; ++k){
        const size_t b = k / gpb;
        const uint32_t t = (uint32_t)(k % gpb);
        double *p[8];
        p[0] = data + 2u * (size_t)len * b + 16u * (size_t)t;
        for(int i = 1; i < 8; ++i) p[i] = p[i - 1] + st;
        const double *twp = tw + (size_t)t * tws;
        qcv v[8];
        for(int i = 0; i < 8; ++i) v[i] = q_ld(p[i]);
        qcv W1, W1o, W2, W4;
        pq16_tw8_get(twp, twc, &W1, &W1o, &W2, &W4);
        if(inv) q_ibf8(v, W1, W1o, W2, W4);
        else    q_bf8(v, W1, W1o, W2, W4);
        for(int i = 0; i < 8; ++i) q_st(p[i], v[i]);
    }
}

__attribute__((always_inline))
static inline void pq16_r8_stage(double *data, uint32_t n, uint32_t len,
                                 const double *tw){
    const size_t st = 2u * (size_t)(len / 8u);       // doubles per eighth
    const int twc = pq16_twc(len);
    for(uint32_t base = 0; base < n; base += len){
        double *p[8];
        p[0] = data + 2u * (size_t)base;
        for(int i = 1; i < 8; ++i) p[i] = p[i - 1] + st;
        const double *twp = tw;
        for(uint32_t t = 0; t < len / 64u; ++t){
            qcv v[8];
            for(int i = 0; i < 8; ++i) v[i] = q_ld(p[i]);
            qcv W1, W1o, W2, W4;
            pq16_tw8_get(twp, twc, &W1, &W1o, &W2, &W4);
            twp += pq16_tw_step(twc);
            q_bf8(v, W1, W1o, W2, W4);
            for(int i = 0; i < 8; ++i){ q_st(p[i], v[i]); p[i] += 16; }
        }
    }
}
__attribute__((always_inline))
static inline void pq16_ir8_stage(double *data, uint32_t n, uint32_t len,
                                  const double *tw){
    const size_t st = 2u * (size_t)(len / 8u);
    const int twc = pq16_twc(len);
    for(uint32_t base = 0; base < n; base += len){
        double *p[8];
        p[0] = data + 2u * (size_t)base;
        for(int i = 1; i < 8; ++i) p[i] = p[i - 1] + st;
        const double *twp = tw;
        for(uint32_t t = 0; t < len / 64u; ++t){
            qcv v[8];
            for(int i = 0; i < 8; ++i) v[i] = q_ld(p[i]);
            qcv W1, W1o, W2, W4;
            pq16_tw8_get(twp, twc, &W1, &W1o, &W2, &W4);
            twp += pq16_tw_step(twc);
            q_ibf8(v, W1, W1o, W2, W4);
            for(int i = 0; i < 8; ++i){ q_st(p[i], v[i]); p[i] += 16; }
        }
    }
}

// ------------------------------------------------------------------
// leaves. Output: stored tile slot q lane t = canonical-BR t*8+q (pi
// layout, no transpose-back -- the pointwise + inverse leaves consume it)
// ------------------------------------------------------------------
static inline void q_l64const(qcv lw[4], const pq16_plan *pl){
    lw[0] = q_ld(pl->l64w);      lw[1] = q_ld(pl->l64w + 16);
    lw[2] = q_ld(pl->l64w + 32); lw[3] = q_ld(pl->l64w + 48);
}

// 64-point body: 8 tiles, across-r8 + transpose + DFT8
static inline void q_body64(qcv v[8], const qcv lw[4]){
    q_bf8(v, lw[0], lw[1], lw[2], lw[3]);
    q_tr8cv(v);
    q_dft8(v);
}
static inline void q_ibody64(qcv v[8], const qcv lw[4]){
    q_idft8(v);
    q_tr8cv(v);
    q_ibf8(v, lw[0], lw[1], lw[2], lw[3]);
}

static inline void pq16_leaf64_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv lw[4]; q_l64const(lw, pl);
    for(uint32_t g = 0; g < span / 64u; ++g){
        double *p = d + 128u * (size_t)g;
        qcv v[8];
        for(int i = 0; i < 8; ++i) v[i] = q_ld(p + 16 * i);
        q_body64(v, lw);
        for(int i = 0; i < 8; ++i) q_st(p + 16 * i, v[i]);
    }
}
static inline void pq16_ileaf64_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv lw[4]; q_l64const(lw, pl);
    for(uint32_t g = 0; g < span / 64u; ++g){
        double *p = d + 128u * (size_t)g;
        qcv v[8];
        for(int i = 0; i < 8; ++i) v[i] = q_ld(p + 16 * i);
        q_ibody64(v, lw);
        for(int i = 0; i < 8; ++i) q_st(p + 16 * i, v[i]);
    }
}

// two 32-point blocks batched (8 tiles): per-block across-r22, combined
// transpose + DFT8. Lanes 0..3 = block A sub-blocks, 4..7 = block B.
static inline void pq16_leaf32x2_one(double *p, qcv W1, qcv W2){
    qcv v[8];
    for(int i = 0; i < 8; ++i) v[i] = q_ld(p + 16 * i);
    q_bf4(v, W1, W2);
    q_bf4(v + 4, W1, W2);
    q_tr8cv(v);
    q_dft8(v);
    for(int i = 0; i < 8; ++i) q_st(p + 16 * i, v[i]);
}
static inline void pq16_ileaf32x2_one(double *p, qcv W1, qcv W2){
    qcv v[8];
    for(int i = 0; i < 8; ++i) v[i] = q_ld(p + 16 * i);
    q_idft8(v);
    q_tr8cv(v);
    q_ibf4(v, W1, W2);
    q_ibf4(v + 4, W1, W2);
    for(int i = 0; i < 8; ++i) q_st(p + 16 * i, v[i]);
}
static inline void pq16_leaf32x2_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv W1 = q_ld(pl->l32w), W2 = q_ld(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 64u; ++g)
        pq16_leaf32x2_one(d + 128u * (size_t)g, W1, W2);
}
static inline void pq16_ileaf32x2_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv W1 = q_ld(pl->l32w), W2 = q_ld(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 64u; ++g)
        pq16_ileaf32x2_one(d + 128u * (size_t)g, W1, W2);
}

// 128-point leaf: across-r22 at len 128 (2 levels, streaming over 4-tile
// groups) + two 32x2 batches (5 levels each). Two register-clean L1
// passes; the r2 + two-64-body form needed three and spilled.
static inline void pq16_leaf128_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv W1_32 = q_ld(pl->l32w), W2_32 = q_ld(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 128u; ++g){
        double *p = d + 256u * (size_t)g;
        for(int t = 0; t < 4; ++t){
            qcv v[4] = { q_ld(p + 16 * t),       q_ld(p + 16 * (t + 4)),
                         q_ld(p + 16 * (t + 8)), q_ld(p + 16 * (t + 12)) };
            qcv W1 = q_ld(pl->l128w + 32 * t);
            qcv W2 = q_ld(pl->l128w + 32 * t + 16);
            q_bf4(v, W1, W2);
            q_st(p + 16 * t,        v[0]);
            q_st(p + 16 * (t + 4),  v[1]);
            q_st(p + 16 * (t + 8),  v[2]);
            q_st(p + 16 * (t + 12), v[3]);
        }
        pq16_leaf32x2_one(p,       W1_32, W2_32);
        pq16_leaf32x2_one(p + 128, W1_32, W2_32);
    }
}
static inline void pq16_ileaf128_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv W1_32 = q_ld(pl->l32w), W2_32 = q_ld(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 128u; ++g){
        double *p = d + 256u * (size_t)g;
        pq16_ileaf32x2_one(p,       W1_32, W2_32);
        pq16_ileaf32x2_one(p + 128, W1_32, W2_32);
        for(int t = 0; t < 4; ++t){
            qcv v[4] = { q_ld(p + 16 * t),       q_ld(p + 16 * (t + 4)),
                         q_ld(p + 16 * (t + 8)), q_ld(p + 16 * (t + 12)) };
            qcv W1 = q_ld(pl->l128w + 32 * t);
            qcv W2 = q_ld(pl->l128w + 32 * t + 16);
            q_ibf4(v, W1, W2);
            q_st(p + 16 * t,        v[0]);
            q_st(p + 16 * (t + 4),  v[1]);
            q_st(p + 16 * (t + 8),  v[2]);
            q_st(p + 16 * (t + 12), v[3]);
        }
    }
}

static inline void pq16_leaf_run(double *d, uint32_t span, uint32_t leaf,
                                 const pq16_plan *pl){
    if(leaf == 64u)      pq16_leaf64_run(d, span, pl);
    else if(leaf == 32u) pq16_leaf32x2_run(d, span, pl);
    else                 pq16_leaf128_run(d, span, pl);
}
static inline void pq16_ileaf_run(double *d, uint32_t span, uint32_t leaf,
                                  const pq16_plan *pl){
    if(leaf == 64u)      pq16_ileaf64_run(d, span, pl);
    else if(leaf == 32u) pq16_ileaf32x2_run(d, span, pl);
    else                 pq16_ileaf128_run(d, span, pl);
}

// ------------------------------------------------------------------
// drivers. Recursive blocking ladder (~L3-slice / L2 / L1 of tile data):
// stages too long for the current tier run as full-span sweeps, everything
// below recurses into tier-sized chunks so a chunk's remaining stages +
// leaves happen while cache-resident.
// ------------------------------------------------------------------
#ifndef PQ16_BLOCK
#define PQ16_BLOCK 2048u    /* L1 tier: 32 KB of tile data */
#endif
#ifndef PQ16_BLOCK2
#define PQ16_BLOCK2 32768u  /* L2 tier: 512 KB */
#endif
#ifndef PQ16_BLOCK3
#define PQ16_BLOCK3 131072u /* L3 tier: 2 MB */
#endif
static const uint32_t PQ16_TIERS[3] = { PQ16_BLOCK3, PQ16_BLOCK2, PQ16_BLOCK };

static void pq16_fwd_rec(double *d, uint32_t span, const pq16_shape *sh,
                         uint32_t s, unsigned tier, const pq16_plan *pl){
    while(tier < 3 && PQ16_TIERS[tier] >= span) tier++;
    if(tier == 3){
        for(uint32_t ss = s; ss < sh->cnt; ++ss)
            pq16_r8_stage(d, span, sh->len[ss], pl->tw8[__builtin_ctz(sh->len[ss])]);
        pq16_leaf_run(d, span, sh->leaf, pl);
        return;
    }
    uint32_t chunk = PQ16_TIERS[tier];
    while(s < sh->cnt && sh->len[s] > chunk){
        pq16_r8_stage(d, span, sh->len[s], pl->tw8[__builtin_ctz(sh->len[s])]);
        s++;
    }
    for(uint32_t base = 0; base < span; base += chunk)
        pq16_fwd_rec(d + 2u * (size_t)base, chunk, sh, s, tier + 1u, pl);
}
static void pq16_inv_rec(double *d, uint32_t span, const pq16_shape *sh,
                         uint32_t s, unsigned tier, const pq16_plan *pl){
    while(tier < 3 && PQ16_TIERS[tier] >= span) tier++;
    if(tier == 3){
        for(uint32_t ss = sh->cnt; ss-- > s; )
            pq16_ir8_stage(d, span, sh->len[ss], pl->tw8[__builtin_ctz(sh->len[ss])]);
        return;
    }
    uint32_t chunk = PQ16_TIERS[tier];
    uint32_t s1 = s;
    while(s1 < sh->cnt && sh->len[s1] > chunk) s1++;
    for(uint32_t base = 0; base < span; base += chunk)
        pq16_inv_rec(d + 2u * (size_t)base, chunk, sh, s1, tier + 1u, pl);
    for(uint32_t ss = s1; ss-- > s; )
        pq16_ir8_stage(d, span, sh->len[ss], pl->tw8[__builtin_ctz(sh->len[ss])]);
}

static inline void pq16_fwd_core(double *data, uint32_t n, const pq16_plan *pl){
    pq16_shape sh = pq16_shape_of(n);
    pq16_fwd_rec(data, n, &sh, 0, 0, pl);
}
// inverse r8 stages only (leaves already applied by the fused pointwise)
static inline void pq16_inv_r8_only(double *data, uint32_t n, const pq16_plan *pl){
    pq16_shape sh = pq16_shape_of(n);
    pq16_inv_rec(data, n, &sh, 0, 0, pl);
}

// ------------------------------------------------------------------
// odd-radix branch plan (2026-09-08). The pow2 branch of a CT / PFA /
// right-angle product runs its stages from this list instead of the
// separate r22 pass + shape_of: radix-8 full-span stages, in-cache radix-4
// stages (the r22 kernel per block) and the 32-point leaf wherever the level
// count allows -- the fused 32x2 pointwise leaf is the cheapest one, and a
// radix-4 pass over the whole branch costs more per level than a radix-8
// stage. lg n <= 10 and >= 14 keep the classic split (r22 pass, then
// shape_of) bit-identically. Measured on the odd-radix public points
// (bench/probes/odd_radix_2026-09-08, 5 rounds): lg 11 -1.8..-4.4 %,
// lg 12 -1.3..-3.4 %, lg 13 -2.7 %. The wide (W > 1) PFA drivers keep the
// classic split; both table sets are built at prepare.
// ------------------------------------------------------------------
static inline pq16_odd_plan pq16_odd_plan_of(uint32_t n){
    const unsigned lgn = (unsigned)__builtin_ctz(n);
    pq16_odd_plan p; p.cnt = 0;
    if(lgn == 6){ p.leaf = 64; return p; }   // branch 64: the 64-point leaf is the whole branch transform
    if(lgn == 11){ p.radix[0] = 8; p.radix[1] = 8; p.cnt = 2; p.leaf = 32; }
    else if(lgn == 12){ p.radix[0] = 8; p.radix[1] = 4; p.radix[2] = 4; p.cnt = 3; p.leaf = 32; }
    else if(lgn == 13){ p.radix[0] = 8; p.radix[1] = 8; p.radix[2] = 4; p.cnt = 3; p.leaf = 32; }
    else{
        const pq16_shape sh = pq16_shape_of(n);
        p.radix[p.cnt++] = 4;
        for(uint32_t i = 0; i < sh.cnt; ++i) p.radix[p.cnt++] = 8;
        p.leaf = sh.leaf;
    }
    uint32_t len = n;
    for(uint32_t i = 0; i < p.cnt; ++i){ p.len[i] = len; len /= p.radix[i]; }
    return p;
}
static inline uint32_t pq16_odd_leaf(uint32_t n){ return pq16_odd_plan_of(n).leaf; }
static inline void pq16_odd_stage(double *d, uint32_t span, const pq16_odd_plan *p,
                                  uint32_t s, const pq16_plan *pl, int inv){
    const uint32_t len = p->len[s];
    const unsigned lg = (unsigned)__builtin_ctz(len);
    if(p->radix[s] == 8){
        if(inv) pq16_ir8_stage(d, span, len, pl->tw8[lg]);
        else    pq16_r8_stage(d, span, len, pl->tw8[lg]);
    }else{
        for(uint32_t b = 0; b < span; b += len){
            if(inv) pq16_mem_ir22(d + 2u * (size_t)b, len, pl->tw22[lg]);
            else    pq16_mem_r22(d + 2u * (size_t)b, len, pl->tw22[lg]);
        }
    }
}
static void pq16_odd_fwd_rec(double *d, uint32_t span, const pq16_odd_plan *p,
                             uint32_t s, unsigned tier, const pq16_plan *pl){
    while(tier < 3 && PQ16_TIERS[tier] >= span) tier++;
    if(tier == 3){
        for(uint32_t ss = s; ss < p->cnt; ++ss) pq16_odd_stage(d, span, p, ss, pl, 0);
        pq16_leaf_run(d, span, p->leaf, pl);
        return;
    }
    const uint32_t chunk = PQ16_TIERS[tier];
    while(s < p->cnt && p->len[s] > chunk){ pq16_odd_stage(d, span, p, s, pl, 0); s++; }
    for(uint32_t base = 0; base < span; base += chunk)
        pq16_odd_fwd_rec(d + 2u * (size_t)base, chunk, p, s, tier + 1u, pl);
}
static void pq16_odd_inv_rec(double *d, uint32_t span, const pq16_odd_plan *p,
                             uint32_t s, unsigned tier, const pq16_plan *pl){
    while(tier < 3 && PQ16_TIERS[tier] >= span) tier++;
    if(tier == 3){
        for(uint32_t ss = p->cnt; ss-- > s; ) pq16_odd_stage(d, span, p, ss, pl, 1);
        return;
    }
    const uint32_t chunk = PQ16_TIERS[tier];
    uint32_t s1 = s;
    while(s1 < p->cnt && p->len[s1] > chunk) s1++;
    for(uint32_t base = 0; base < span; base += chunk)
        pq16_odd_inv_rec(d + 2u * (size_t)base, chunk, p, s1, tier + 1u, pl);
    for(uint32_t ss = s1; ss-- > s; ) pq16_odd_stage(d, span, p, ss, pl, 1);
}
// whole forward of one odd-radix branch after its M-stage (all stages + leaf)
static inline void pq16_odd_fwd(double *data, uint32_t n, const pq16_plan *pl){
    const pq16_odd_plan p = pq16_odd_plan_of(n);
    pq16_odd_fwd_rec(data, n, &p, 0, 0, pl);
}
// whole inverse of one odd-radix branch after the pointwise-fused leaves (natural order out)
static inline void pq16_odd_inv(double *data, uint32_t n, const pq16_plan *pl){
    const pq16_odd_plan p = pq16_odd_plan_of(n);
    pq16_odd_inv_rec(data, n, &p, 0, 0, pl);
}

/* ---- G5-P4: wide drivers -------------------------------------------
 * The tier-0 recursion decomposes exactly: above-L3-tier full-span r8
 * sweeps (flattened item ranges, disjoint slots) then span/BLOCK3
 * INDEPENDENT chunk descents (each stays serial = cache-resident,
 * identical to the serial recursion on its subrange). Serial fallback
 * (t == NULL / nthr <= 1 / span <= one chunk) is the stock recursion. */
typedef struct {
    double *d;
    const double *tw;
    uint32_t len;
    int inv;
} pq16_sweep_fctx;
static void pq16_sweep_forfn(void *c_, uint64_t lo, uint64_t hi, int w,
                             scratch *ws){
    pq16_sweep_fctx *c = (pq16_sweep_fctx *)c_;
    (void)w; (void)ws;
    pq16_r8_stage_rng(c->d, c->len, c->tw, c->inv, (size_t)lo, (size_t)hi);
}
static void pq16_sweep_w(double *d, uint32_t span, uint32_t len,
                         const double *tw, int inv, sbn_team *t, int nthr){
    const size_t items = ((size_t)span / len) * (len / 64u);
    if(t && nthr > 1 && items >= 64){
        pq16_sweep_fctx c = { d, tw, len, inv };
        sbn_parallel_for(t, nthr, 0, items, 16, SBN_FOR_STATIC,
                         pq16_sweep_forfn, &c);
    }else{
        pq16_r8_stage_rng(d, len, tw, inv, 0, items);
    }
}
typedef struct {
    double *d;
    const pq16_shape *sh;
    const pq16_plan *pl;
    uint32_t s, chunk;
    unsigned tier;
    int inv;
} pq16_chunk_fctx;
static void pq16_chunk_taskfn(void *c_, int i, int w, scratch *ws){
    pq16_chunk_fctx *c = (pq16_chunk_fctx *)c_;
    (void)w; (void)ws;
    double *d = c->d + 2u * (size_t)i * c->chunk;
    if(c->inv) pq16_inv_rec(d, c->chunk, c->sh, c->s, c->tier + 1u, c->pl);
    else       pq16_fwd_rec(d, c->chunk, c->sh, c->s, c->tier + 1u, c->pl);
}
/* pick the chunk tier for a wide descent: L3 chunks when they already
 * feed nthr tasks, else drop to the L2 tier (more tasks; each worker's
 * chunk goes L2-resident in its own core -- the full-span sweeps do
 * the same total traffic either way, order is irrelevant on disjoint
 * slots). Returns the tier index, or 3 = too small, stay serial. */
static inline unsigned pq16_chunk_tier(uint32_t n, int nthr){
    if(n > PQ16_TIERS[0] && n / PQ16_TIERS[0] >= (uint32_t)nthr) return 0;
    if(n > PQ16_TIERS[1]) return 1;
    return 3;
}
static inline unsigned pq16_pfa_tier(uint32_t n,uint32_t M,int nthr){
    const unsigned tier=pq16_chunk_tier(n,nthr);
    if constexpr(PQ16_PFA_FINE)if(tier==3 && nthr>1 && n>=8192 && M%unsigned(nthr))return 2;
    return tier;
}
static void pq16_fwd_core_w(double *data, uint32_t n, const pq16_plan *pl,
                            sbn_team *t, int nthr){
    pq16_shape sh = pq16_shape_of(n);
    const unsigned ct = t && nthr > 1 ? pq16_chunk_tier(n, nthr) : 3;
    if(ct == 3){
        pq16_fwd_rec(data, n, &sh, 0, 0, pl);
        return;
    }
    const uint32_t chunk = PQ16_TIERS[ct];
    uint32_t s = 0;
    while(s < sh.cnt && sh.len[s] > chunk){
        pq16_sweep_w(data, n, sh.len[s],
                     pl->tw8[__builtin_ctz(sh.len[s])], 0, t, nthr);
        s++;
    }
    pq16_chunk_fctx c = { data, &sh, pl, s, chunk, ct, 0 };
    const int nt = (int)(n / chunk);
    sbn_run_tasks(t, nthr, nt, pq16_chunk_taskfn, &c);
}
static void pq16_inv_r8_only_w(double *data, uint32_t n, const pq16_plan *pl,
                               sbn_team *t, int nthr){
    pq16_shape sh = pq16_shape_of(n);
    const unsigned ct = t && nthr > 1 ? pq16_chunk_tier(n, nthr) : 3;
    if(ct == 3){
        pq16_inv_rec(data, n, &sh, 0, 0, pl);
        return;
    }
    const uint32_t chunk = PQ16_TIERS[ct];
    uint32_t s1 = 0;
    while(s1 < sh.cnt && sh.len[s1] > chunk) s1++;
    pq16_chunk_fctx c = { data, &sh, pl, s1, chunk, ct, 1 };
    const int nt = (int)(n / chunk);
    sbn_run_tasks(t, nthr, nt, pq16_chunk_taskfn, &c);
    for(uint32_t ss = s1; ss-- > 0; )
        pq16_sweep_w(data, n, sh.len[ss],
                     pl->tw8[__builtin_ctz(sh.len[ss])], 1, t, nthr);
}
static inline void pq16_fwd(double *data, const uint64_t *src, int64_t cnt,
                            uint32_t n, const pq16_plan *pl, int cen){
    pq16_input_stage(data, src, cnt, n, pl, cen);
    pq16_fwd_core(data, n, pl);
}
// PFA forward: radix-M input, then mem_r22 + core per branch
static inline void pq16_pfa_fwd(double *data, const uint64_t *src, int64_t cnt,
                                uint32_t n, uint32_t M, const pq16_plan *pl,
                                int cen){
    q_pfa_input(data, src, cnt, n, M, pl, cen);
    for(uint32_t b = 0; b < M; ++b)
        pq16_odd_fwd(data + 2u * (size_t)n * b, n, pl);
}

/* G5-P4: PFA branch transforms as independent tasks (each branch stays
 * a serial cache-resident descent; branches never touch each other) */
typedef struct {
    double *data;
    const double *tw;
    uint32_t n;
    const pq16_plan *pl;
    int inv;                        /* 0: mem_r22+fwd; 1: inv+mem_ir22 */
} pq16_branch_fctx;
static void pq16_branch_taskfn(void *c_, int b, int w, scratch *ws){
    pq16_branch_fctx *c = (pq16_branch_fctx *)c_;
    (void)w; (void)ws;
    double *d = c->data + 2u * (size_t)c->n * (uint32_t)b;
    if(c->inv){
        pq16_inv_r8_only(d, c->n, c->pl);
        pq16_mem_ir22(d, c->n, c->tw);
    }else{
        pq16_mem_r22(d, c->n, c->tw);
        pq16_fwd_core(d, c->n, c->pl);
    }
}
/* flattened (branch x chunk) grid: phase 1 = per-branch mem_r22 (+ any
 * above-chunk sweeps, which are empty for PFA branch sizes at the L2
 * tier: len[0] = n/4 <= 32768 whenever n <= 2^17); phase 2 = all
 * branches' chunk descents pulled off one task counter */
typedef struct {
    double *data;
    const pq16_shape *sh;
    const pq16_plan *pl;
    const double *tw;
    uint32_t n, s, chunk;
    unsigned tier;
    int inv, nchunk;
} pq16_pfa_fctx;
static void pq16_pfa_r22_taskfn(void *c_, int b, int w, scratch *ws){
    pq16_pfa_fctx *c = (pq16_pfa_fctx *)c_;
    (void)w; (void)ws;
    double *d = c->data + 2u * (size_t)c->n * (uint32_t)b;
    if(c->inv){
        for(uint32_t ss = c->s; ss-- > 0; )
            pq16_r8_stage_rng(d, c->sh->len[ss],
                              c->pl->tw8[__builtin_ctz(c->sh->len[ss])], 1,
                              0, ((size_t)c->n / c->sh->len[ss])
                                 * (c->sh->len[ss] / 64u));
        pq16_mem_ir22(d, c->n, c->tw);
    }else{
        pq16_mem_r22(d, c->n, c->tw);
        for(uint32_t ss = 0; ss < c->s; ++ss)
            pq16_r8_stage_rng(d, c->sh->len[ss],
                              c->pl->tw8[__builtin_ctz(c->sh->len[ss])], 0,
                              0, ((size_t)c->n / c->sh->len[ss])
                                 * (c->sh->len[ss] / 64u));
    }
}
static void pq16_pfa_chunk_taskfn(void *c_, int i, int w, scratch *ws){
    pq16_pfa_fctx *c = (pq16_pfa_fctx *)c_;
    (void)w; (void)ws;
    const uint32_t b = (uint32_t)(i / c->nchunk), k = (uint32_t)(i % c->nchunk);
    double *d = c->data + 2u * (size_t)c->n * b + 2u * (size_t)k * c->chunk;
    if(c->inv) pq16_inv_rec(d, c->chunk, c->sh, c->s, c->tier + 1u, c->pl);
    else       pq16_fwd_rec(d, c->chunk, c->sh, c->s, c->tier + 1u, c->pl);
}
static inline void pq16_pfa_fwd_w(double *data, const uint64_t *src,
                                  int64_t cnt, uint32_t n, uint32_t M,
                                  const pq16_plan *pl, int cen,
                                  sbn_team *t, int nthr){
    q_pfa_input(data, src, cnt, n, M, pl, cen);
    const unsigned ct = t && nthr > 1 ? pq16_pfa_tier(n,M,nthr) : 3;
    if(ct == 3){
        if(t && nthr > 1){       /* branch tasks only (small branches) */
            pq16_branch_fctx c = { data, pl->tw22[__builtin_ctz(n)], n,
                                   pl, 0 };
            sbn_run_tasks(t, nthr, (int)M,
                          pq16_branch_taskfn, &c);
        }else{
            const double *tw = pl->tw22[__builtin_ctz(n)];
            for(uint32_t b = 0; b < M; ++b){
                double *d = data + 2u * (size_t)n * b;
                pq16_mem_r22(d, n, tw);
                pq16_fwd_core(d, n, pl);
            }
        }
        return;
    }
    pq16_shape sh = pq16_shape_of(n);
    const uint32_t chunk = PQ16_TIERS[ct];
    uint32_t s = 0;
    while(s < sh.cnt && sh.len[s] > chunk) s++;
    pq16_pfa_fctx c = { data, &sh, pl, pl->tw22[__builtin_ctz(n)],
                        n, s, chunk, ct, 0, (int)(n / chunk) };
    sbn_run_tasks(t, nthr, (int)M,
                  pq16_pfa_r22_taskfn, &c);
    const int nt = (int)M * c.nchunk;
    sbn_run_tasks(t, nthr, nt, pq16_pfa_chunk_taskfn, &c);
}

// ------------------------------------------------------------------
// PQ pointwise multiply (in pi layout, A *= B elementwise-PQ).
// Per-group twiddle reconstruction from the compact g-table: lane t of
// stored vector (g, q) needs g[g*16 + 2t + (q>>2)] (one vpermt2pd per
// component, hoisted per group) and the 1 +- g pattern by q & 3.
// ------------------------------------------------------------------
// paired 8-lane PQ eval (partner positions k, N/2-k share cross-terms)
static inline void q_pq_pair(qcv *outl, qcv *outr, qcv x, qcv xn, qcv y, qcv yn,
                             qcv w, sb_dvec sc, sb_dvec qsc){
    qcv pql = q_mul(x, y);
    qcv pqr = q_mul(xn, yn);
    qcv dp = { sb_sub(x.re, xn.re), sb_add(x.im, xn.im) };
    qcv dq = { sb_sub(y.re, yn.re), sb_add(y.im, yn.im) };
    qcv t = q_mul(dp, dq);
    qcv c = q_mul(t, w);
    sb_dvec qcr = sb_mul(qsc, c.re);
    sb_dvec qci = sb_mul(qsc, c.im);
    outl->re = fmsub(pql.re, sc, qcr);
    outl->im = fmsub(pql.im, sc, qci);
    outr->re = fmsub(pqr.re, sc, qcr);
    outr->im = fmadd(pqr.im, sc, qci);
}
// single-sided 8-lane PQ eval: each lane uses its own twiddle
static inline qcv q_pq_eval(qcv x, qcv xn, qcv y, qcv yn, qcv w,
                            sb_dvec sc, sb_dvec qsc){
    qcv pq = q_mul(x, y);
    qcv dp = { sb_sub(x.re, xn.re), sb_add(x.im, xn.im) };
    qcv dq = { sb_sub(y.re, yn.re), sb_add(y.im, yn.im) };
    qcv c = q_mul(q_mul(dp, dq), w);
    qcv z = { fmsub(pq.re, sc, sb_mul(qsc, c.re)),
              fmsub(pq.im, sc, sb_mul(qsc, c.im)) };
    return z;
}

typedef struct { sb_dvec gr[2], gi[2]; } q_gsel;
static inline q_gsel q_pq_gsel(const double *gblk){
    const sb_vec IE = sb_setr_64(0, 2, 4, 6, 8, 10, 12, 14);
    const sb_vec IO = sb_setr_64(1, 3, 5, 7, 9, 11, 13, 15);
    sb_dvec r0 = load_dvec(gblk),      r1 = load_dvec(gblk + 8);
    sb_dvec i0 = load_dvec(gblk + 16), i1 = load_dvec(gblk + 24);
    q_gsel s;
    s.gr[0] = sb__fn(permutex2var_pd)(r0, IE, r1);
    s.gr[1] = sb__fn(permutex2var_pd)(r0, IO, r1);
    s.gi[0] = sb__fn(permutex2var_pd)(i0, IE, i1);
    s.gi[1] = sb__fn(permutex2var_pd)(i0, IO, i1);
    return s;
}
static inline qcv q_pq_w(const q_gsel *s, uint32_t q){
    sb_dvec gr = s->gr[q >> 2], gi = s->gi[q >> 2];
    const sb_dvec one = sb_set1_d(1.0);
    const sb_dvec nz  = sb_set1_d(-0.0);
    qcv w;
    switch(q & 3u){
    case 0:  w.re = sb_add(one, gr); w.im = gi; break;
    case 1:  w.re = sb_sub(one, gr); w.im = sb__fn(xor_pd)(gi, nz); break;
    case 2:  w.re = sb_sub(one, gi); w.im = gr; break;
    default: w.re = sb_add(one, gi); w.im = sb__fn(xor_pd)(gr, nz); break;
    }
    return w;
}
// gsel with the PFA branch rotation g' = g * w3 (w3 = omega_M^b)
static inline q_gsel q_pq_gsel_rot(const double *gblk, double w3r, double w3i){
    q_gsel s = q_pq_gsel(gblk);
    const sb_dvec vr = sb_set1_d(w3r), vi = sb_set1_d(w3i);
    for(int b = 0; b < 2; ++b){
        sb_dvec nr = fmsub(s.gr[b], vr, sb_mul(s.gi[b], vi));
        sb_dvec ni = fmadd(s.gr[b], vi, sb_mul(s.gi[b], vr));
        s.gr[b] = nr; s.gi[b] = ni;
    }
    return s;
}
// head twiddle for canonical positions 0..7 (lane p), from the gh0
// g-values rotated by w3; the per-lane 1 +- g pattern (case = p & 3) is
// applied with constant blend/sign masks
static inline qcv q_pq_head_w(const pq16_plan *pl, double w3r, double w3i){
    sb_dvec gr = load_dvec(pl->gh0), gi = load_dvec(pl->gh0 + 8);
    const sb_dvec vr = sb_set1_d(w3r), vi = sb_set1_d(w3i);
    sb_dvec gr2 = fmsub(gr, vr, sb_mul(gi, vi));
    sb_dvec gi2 = fmadd(gr, vi, sb_mul(gi, vr));
    const sb_dvec one = sb_set1_d(1.0);
    const sb_dvec sre = sb__fn(setr_pd)(0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0);
    const sb_dvec sim = sb__fn(setr_pd)(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    qcv w;
    w.re = sb_add(one, sb__fn(xor_pd)(sb__fn(mask_blend_pd)(sb_k8(0xCC), gr2, gi2), sre));
    w.im = sb__fn(xor_pd)(sb__fn(mask_blend_pd)(sb_k8(0xCC), gi2, gr2), sim);
    return w;
}

// the part8 lane map (pairing pattern of canonical positions 0..7) and the
// column-gather index for stored group 0 lane 0
#define PQ16_P8()  sb_setr_64(0, 1, 3, 2, 7, 6, 5, 4)
#define PQ16_COL() _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112)
static inline qcv q_perm8(qcv a, sb_vec P8){
    qcv r = { permd(P8, a.re), permd(P8, a.im) };
    return r;
}

// pointwise over stored group 0 (canonical positions [0, 64)). Lane 0
// holds the first 8 positions: their pairing is part8 across slots,
// evaluated single-sided per lane (gather column, permute, scatter).
// Lanes 1..7 are the conjugate-mirror pairing of bases 8/16/32, which in
// stored coords is slot-reverse (q <-> 7-q) with the SAME part8 pattern as
// a lane map -- four masked pq_pair calls.
static inline void q_pw_head(double *A, double *B, const pq16_plan *pl,
                             sb_dvec sc, sb_dvec qsc){
    const sb_vec P8 = PQ16_P8();
    const __m256i COL = PQ16_COL();

    qcv x, y;
    x.re = sb__fn(i32gather_pd)(COL, A, 8);
    x.im = sb__fn(i32gather_pd)(COL, A + 8, 8);
    y.re = sb__fn(i32gather_pd)(COL, B, 8);
    y.im = sb__fn(i32gather_pd)(COL, B + 8, 8);
    qcv w  = q_pq_head_w(pl, 1.0, 0.0);
    qcv xn = q_perm8(x, P8);
    qcv yn = q_perm8(y, P8);
    qcv z = q_pq_eval(x, xn, y, yn, w, sc, qsc);
    sb__fn(i32scatter_pd)(A, COL, z.re, 8);
    sb__fn(i32scatter_pd)(A + 8, COL, z.im, 8);

    q_gsel gs = q_pq_gsel(pl->pq);
    for(uint32_t q = 0; q < 4; ++q){
        size_t vl = q, vr = 7u - q;
        qcv xq  = q_ld(A + 16u * vl);
        qcv xnq = q_perm8(q_ld(A + 16u * vr), P8);
        qcv yq  = q_ld(B + 16u * vl);
        qcv ynq = q_perm8(q_ld(B + 16u * vr), P8);
        qcv wq  = q_pq_w(&gs, q);
        qcv outl, outr;
        q_pq_pair(&outl, &outr, xq, xnq, yq, ynq, wq, sc, qsc);
        sb__fn(mask_store_pd)(A + 16u * vl,     sb_k8(0xFE), outl.re);
        sb__fn(mask_store_pd)(A + 16u * vl + 8, sb_k8(0xFE), outl.im);
        outr = q_perm8(outr, P8);
        sb__fn(mask_store_pd)(A + 16u * vr,     sb_k8(0xFE), outr.re);
        sb__fn(mask_store_pd)(A + 16u * vr + 8, sb_k8(0xFE), outr.im);
    }
}

// pointwise one group pair: stored vec (gl, q) <-> rev(vec (gr, 7-q)).
// gl == gr handles the in-group mirror (base 64): q = 0..3 only.
static inline void q_pw_groups(double *A, double *B, uint32_t gl, uint32_t gr,
                               const pq16_plan *pl, sb_dvec sc, sb_dvec qsc){
    uint32_t qe = gl == gr ? 4u : 8u;
    q_gsel gs = q_pq_gsel(pl->pq + 32u * (size_t)gl);
    for(uint32_t q = 0; q < qe; ++q){
        size_t vl = (size_t)gl * 8u + q, vr = (size_t)gr * 8u + 7u - q;
        qcv x  = q_ld(A + 16u * vl);
        qcv xn = q_rev(q_ld(A + 16u * vr));
        qcv y  = q_ld(B + 16u * vl);
        qcv yn = q_rev(q_ld(B + 16u * vr));
        qcv w  = q_pq_w(&gs, q);
        qcv outl, outr;
        q_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        q_st(A + 16u * vl, outl);
        q_st(A + 16u * vr, q_rev(outr));
    }
}

// inverse leaf over one 64-cx stored group (or its 128 block for leaf128)
static inline void q_ileaf_group(double *A, uint32_t g, uint32_t leaf,
                                 const pq16_plan *pl){
    if(leaf == 128u) pq16_ileaf128_run(A + 256u * (size_t)(g >> 1), 128u, pl);
    else             pq16_ileaf_run(A + 128u * (size_t)g, 64u, leaf, pl);
}

// fused pointwise + inverse leaves: each mirror pair of groups is
// multiplied and immediately inverse-leafed while cache-hot. For leaf128
// the pair structure is block-aligned (gl even, partner descending), so
// leaves run once per touched block.
/* the octave gl-range body (audit stage 7: was pasted in the serial
 * octave walk AND pq16_pw_forfn; one range core serves both). gl, gr
 * mirror-pair inside octave `base`; leaf128's (even, odd) coupling is
 * the caller's grain contract. */
static inline void pq16_pw_rng(double *A, double *B, const pq16_plan *pl,
                               sb_dvec sc, sb_dvec qsc, uint32_t leaf,
                               uint32_t base, uint32_t gl0, uint32_t gl1){
    const uint32_t g0 = base / 64u;
    for(uint32_t gl = gl0; gl < gl1; ++gl){
        uint32_t gr = 3u * g0 - 1u - gl;
        q_pw_groups(A, B, gl, gr, pl, sc, qsc);
        if(leaf != 128u){
            q_ileaf_group(A, gl, leaf, pl);
            q_ileaf_group(A, gr, leaf, pl);
        }else if(base == 128u){
            // gl, gr are the two halves of one 128-block (block-scale
            // analog of the base-64 in-group mirror)
            pq16_ileaf128_run(A + 256u * (size_t)(gl >> 1), 128u, pl);
        }else if(gl & 1u){   // both blocks of the pair now complete
            pq16_ileaf128_run(A + 256u * (size_t)((gl - 1u) >> 1), 128u, pl);
            pq16_ileaf128_run(A + 256u * (size_t)(gr >> 1), 128u, pl);
        }
    }
}
static inline void pq16_pointwise_ileaves(double *A, double *B, uint32_t n,
                                          double sc_d, uint32_t leaf,
                                          const pq16_plan *pl){
    const sb_dvec sc = sb_set1_d(sc_d), qsc = sb_set1_d(0.25 * sc_d);

    q_pw_head(A, B, pl, sc, qsc);            // group 0
    q_pw_groups(A, B, 1, 1, pl, sc, qsc);    // base 64: in-group mirror
    if(leaf == 128u) pq16_ileaf128_run(A, 128u, pl);
    else { q_ileaf_group(A, 0, leaf, pl); q_ileaf_group(A, 1, leaf, pl); }

    for(uint32_t base = 128; base < n; base <<= 1){
        const uint32_t g0 = base / 64u;
        pq16_pw_rng(A, B, pl, sc, qsc, leaf, base, g0, g0 + base / 128u);
    }
}

/* G5-P4: the octave inner loop as a gl-range (mirror pairs colocate by
 * construction: worker owns [gl_lo, gl_hi) and derives gr; each pair
 * touches only groups gl/gr -- disjoint across gl. leaf128's (even,
 * odd) gl coupling is respected by STATIC grain 2 from an even g0. */
typedef struct {
    double *A, *B;
    const pq16_plan *pl;
    double sc_d;
    uint32_t leaf, base, n;
} pq16_pw_fctx;
static void pq16_pw_forfn(void *c_, uint64_t lo, uint64_t hi, int w,
                          scratch *ws){
    pq16_pw_fctx *c = (pq16_pw_fctx *)c_;
    (void)w; (void)ws;
    const sb_dvec sc = sb_set1_d(c->sc_d), qsc = sb_set1_d(0.25 * c->sc_d);
    pq16_pw_rng(c->A, c->B, c->pl, sc, qsc, c->leaf, c->base,
                (uint32_t)lo, (uint32_t)hi);
}
static inline void pq16_pointwise_ileaves_w(double *A, double *B, uint32_t n,
                                            double sc_d, uint32_t leaf,
                                            const pq16_plan *pl,
                                            sbn_team *t, int nthr){
    if(!t || nthr <= 1){
        pq16_pointwise_ileaves(A, B, n, sc_d, leaf, pl);
        return;
    }
    const sb_dvec sc = sb_set1_d(sc_d), qsc = sb_set1_d(0.25 * sc_d);
    q_pw_head(A, B, pl, sc, qsc);
    q_pw_groups(A, B, 1, 1, pl, sc, qsc);
    if(leaf == 128u) pq16_ileaf128_run(A, 128u, pl);
    else { q_ileaf_group(A, 0, leaf, pl); q_ileaf_group(A, 1, leaf, pl); }
    pq16_pw_fctx c = { A, B, pl, sc_d, leaf, 0, n };
    for(uint32_t base = 128; base < n; base <<= 1){
        const uint32_t g0 = base / 64u, ngl = base / 128u;
        c.base = base;
        if(ngl >= 32)
            sbn_parallel_for(t, nthr, g0, g0 + ngl, 2, SBN_FOR_STATIC,
                             pq16_pw_forfn, &c);
        else
            pq16_pw_forfn(&c, g0, g0 + ngl, 0, NULL);
    }
}

// ------------------------------------------------------------------
// PFA cross-pair pointwise: primary positions in branch L pair with the
// mirrored positions of branch R (the (b, M-b) antipodal pair), twiddle g
// rotated by the constant w3 = omega_M^b. Unlike self-pairs every (L, R)
// pairing is distinct: the full group/slot range is iterated and both
// heads are evaluated single-sidedly.
// ------------------------------------------------------------------
static inline void q_pw_head_cross(double *L, double *R, double *oL, double *oR,
                                   const pq16_plan *pl, double w3r, double w3i,
                                   sb_dvec sc, sb_dvec qsc){
    const sb_vec P8 = PQ16_P8();
    const __m256i COL = PQ16_COL();

    // lane 0: positions 0..7 of both branches, partner = part8 of the other
    qcv xl, yl, xr, yr;
    xl.re = sb__fn(i32gather_pd)(COL, L, 8);  xl.im = sb__fn(i32gather_pd)(COL, L + 8, 8);
    yl.re = sb__fn(i32gather_pd)(COL, oL, 8); yl.im = sb__fn(i32gather_pd)(COL, oL + 8, 8);
    xr.re = sb__fn(i32gather_pd)(COL, R, 8);  xr.im = sb__fn(i32gather_pd)(COL, R + 8, 8);
    yr.re = sb__fn(i32gather_pd)(COL, oR, 8); yr.im = sb__fn(i32gather_pd)(COL, oR + 8, 8);
    qcv xrp = q_perm8(xr, P8), yrp = q_perm8(yr, P8);
    qcv xlp = q_perm8(xl, P8), ylp = q_perm8(yl, P8);
    qcv wl = q_pq_head_w(pl, w3r, w3i);
    qcv wr = q_pq_head_w(pl, w3r, -w3i);
    qcv zl = q_pq_eval(xl, xrp, yl, yrp, wl, sc, qsc);
    qcv zr = q_pq_eval(xr, xlp, yr, ylp, wr, sc, qsc);
    sb__fn(i32scatter_pd)(L, COL, zl.re, 8);
    sb__fn(i32scatter_pd)(L + 8, COL, zl.im, 8);
    sb__fn(i32scatter_pd)(R, COL, zr.re, 8);
    sb__fn(i32scatter_pd)(R + 8, COL, zr.im, 8);

    // lanes 1..7, full slot range: L (q, t) <-> R (7-q, part8-lane)
    q_gsel gs = q_pq_gsel_rot(pl->pq, w3r, w3i);
    for(uint32_t q = 0; q < 8; ++q){
        size_t vl = q, vr = 7u - q;
        qcv x  = q_ld(L + 16u * vl);
        qcv xn = q_perm8(q_ld(R + 16u * vr), P8);
        qcv y  = q_ld(oL + 16u * vl);
        qcv yn = q_perm8(q_ld(oR + 16u * vr), P8);
        qcv w = q_pq_w(&gs, q);
        qcv outl, outr;
        q_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        sb__fn(mask_store_pd)(L + 16u * vl,     sb_k8(0xFE), outl.re);
        sb__fn(mask_store_pd)(L + 16u * vl + 8, sb_k8(0xFE), outl.im);
        outr = q_perm8(outr, P8);
        sb__fn(mask_store_pd)(R + 16u * vr,     sb_k8(0xFE), outr.re);
        sb__fn(mask_store_pd)(R + 16u * vr + 8, sb_k8(0xFE), outr.im);
    }
}

static inline void q_pw_groups_cross(double *L, double *R, double *oL, double *oR,
                                     uint32_t gl, uint32_t gr, const pq16_plan *pl,
                                     double w3r, double w3i, sb_dvec sc, sb_dvec qsc){
    q_gsel gs = q_pq_gsel_rot(pl->pq + 32u * (size_t)gl, w3r, w3i);
    for(uint32_t q = 0; q < 8; ++q){
        size_t vl = (size_t)gl * 8u + q, vr = (size_t)gr * 8u + 7u - q;
        qcv x  = q_ld(L + 16u * vl);
        qcv xn = q_rev(q_ld(R + 16u * vr));
        qcv y  = q_ld(oL + 16u * vl);
        qcv yn = q_rev(q_ld(oR + 16u * vr));
        qcv w  = q_pq_w(&gs, q);
        qcv outl, outr;
        q_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        q_st(L + 16u * vl, outl);
        q_st(R + 16u * vr, q_rev(outr));
    }
}

// fused cross pointwise + inverse leaves on both branches
static inline void pq16_pointwise_cross_ileaves(double *L, double *R,
                                                double *oL, double *oR,
                                                uint32_t n, double sc_d,
                                                double w3r, double w3i,
                                                uint32_t leaf,
                                                const pq16_plan *pl){
    const sb_dvec sc = sb_set1_d(sc_d), qsc = sb_set1_d(0.25 * sc_d);

    q_pw_head_cross(L, R, oL, oR, pl, w3r, w3i, sc, qsc);          // group 0
    q_pw_groups_cross(L, R, oL, oR, 1, 1, pl, w3r, w3i, sc, qsc);  // base 64
    if(leaf == 128u){
        pq16_ileaf128_run(L, 128u, pl);
        pq16_ileaf128_run(R, 128u, pl);
    }else{
        q_ileaf_group(L, 0, leaf, pl); q_ileaf_group(L, 1, leaf, pl);
        q_ileaf_group(R, 0, leaf, pl); q_ileaf_group(R, 1, leaf, pl);
    }
    for(uint32_t base = 128; base < n; base <<= 1){
        uint32_t g0 = base / 64u;
        for(uint32_t gl = g0; gl < 2u * g0; ++gl){       // full range
            uint32_t gr = 3u * g0 - 1u - gl;
            q_pw_groups_cross(L, R, oL, oR, gl, gr, pl, w3r, w3i, sc, qsc);
            if(leaf != 128u){
                q_ileaf_group(L, gl, leaf, pl);
                q_ileaf_group(R, gr, leaf, pl);
            }else if(gl & 1u){
                pq16_ileaf128_run(L + 256u * (size_t)((gl - 1u) >> 1), 128u, pl);
                pq16_ileaf128_run(R + 256u * (size_t)(gr >> 1), 128u, pl);
            }
        }
    }
}

// ------------------------------------------------------------------
// emit layer: round, pack 2 tiles -> 8 limbs (lo/hi), SWAR carry chain.
// ------------------------------------------------------------------
// unsigned magic-bias round (clips negatives to 0): x in [0, 2^51)
static inline sb_vec q_mround(sb_dvec x){
    const sb_vec bias = sb_set1_64(0x4330000000000000LL);
    x = sb__fn(max_pd)(x, sb_dzero());
    return sb_sub(sb_as_ivec(sb_add(x, sb_as_dvec(bias))), bias);
}
// signed magic-bias round (no clip): |x| < 2^51
static inline sb_vec q_mround_s(sb_dvec x){
    const sb_vec bias = sb_set1_64(0x4338000000000000LL);
    return sb_sub(sb_as_ivec(sb_add(x, sb_as_dvec(bias))), bias);
}
typedef struct { sb_vec lo, hi; } q_lohi;
// 2 tiles of (re, im) integer coefficient pairs -> 8 packed u64 limbs.
// Wide split: each digit may carry up to 64 bits (lo + hi parts), so the
// codec band is precision-, not container-, limited.
template<bool PairFits64=false>
static inline q_lohi q_pack2i(sb_vec re0, sb_vec im0, sb_vec re1, sb_vec im1){
    sb_vec ul0 = sb_add(re0, sb_slli(im0, 16));
    sb_vec ul1 = sb_add(re1, sb_slli(im1, 16));
    const sb_vec IE = sb_setr_64(0, 2, 4, 6, 8, 10, 12, 14);
    const sb_vec IO = sb_setr_64(1, 3, 5, 7, 9, 11, 13, 15);
    sb_vec elo = sb__fn(permutex2var_epi64)(ul0, IE, ul1);
    sb_vec olo = sb__fn(permutex2var_epi64)(ul0, IO, ul1);
    q_lohi r;
    r.lo = sb_add(elo, sb_slli(olo, 32));
    const sb_vec carry=sb_maskz(sb_ltu(r.lo,elo),sb_set1_64(1));
    if constexpr(PairFits64){
        // C=ell*(2^16-1)^2, ell<=2^16 => C*(1+2^16)<2^64.
        // The first coefficient pair has no high half. The second
        // combine and the eventual inter-limb carry remain exact.
        r.hi=sb_add(sb_srli(olo,32),carry);
    }else{
        const sb_vec uh0=sb_add(sb_srli(im0,48),sb_maskz(sb_ltu(ul0,re0),sb_set1_64(1)));
        const sb_vec uh1=sb_add(sb_srli(im1,48),sb_maskz(sb_ltu(ul1,re1),sb_set1_64(1)));
        const sb_vec ehi=sb__fn(permutex2var_epi64)(uh0,IE,uh1);
        const sb_vec ohi=sb__fn(permutex2var_epi64)(uh0,IO,uh1);
        r.hi=sb_add(sb_add(ehi,sb_srli(olo,32)),sb_add(sb_slli(ohi,32),carry));
    }
    return r;
}
template<bool PairFits64=false>
static inline q_lohi q_pack2(qcv t0, qcv t1,size_t output_limb=0){
#ifdef PQ16_COEFFICIENT_OBSERVER
    alignas(64) double o0[16],o1[16];q_st(o0,t0);q_st(o1,t1);
    PQ16_COEFFICIENT_OBSERVER(output_limb,o0,o1);
#endif
    if constexpr(PairFits64){
        // An interior-binade bias rounds tiny negative FFT errors to zero
        // without max(x,0). Its integer bit pattern vanishes after <<16,
        // so subtract the bias only once per coefficient pair.
        constexpr uint64_t bits=0x4338000000000000ull;
        static_assert((bits<<16)==0);
        const sb_vec bias=sb_set1_64(bits);const sb_dvec db=sb_as_dvec(bias);
        const sb_vec ul0=sb_sub(sb_add(sb_as_ivec(sb_add(t0.re,db)),sb_slli(sb_as_ivec(sb_add(t0.im,db)),16)),bias);
        const sb_vec ul1=sb_sub(sb_add(sb_as_ivec(sb_add(t1.re,db)),sb_slli(sb_as_ivec(sb_add(t1.im,db)),16)),bias);
        const sb_vec IE=sb_setr_64(0,2,4,6,8,10,12,14),IO=sb_setr_64(1,3,5,7,9,11,13,15);
        const sb_vec elo=sb__fn(permutex2var_epi64)(ul0,IE,ul1),olo=sb__fn(permutex2var_epi64)(ul0,IO,ul1);
        q_lohi q;q.lo=sb_add(elo,sb_slli(olo,32));q.hi=sb_add(sb_srli(olo,32),sb_maskz(sb_ltu(q.lo,elo),sb_set1_64(1)));
        return q;
    }
    return q_pack2i<PairFits64>(q_mround(t0.re), q_mround(t0.im),
                    q_mround(t1.re), q_mround(t1.im));
}

// SWAR carry chain: each step adds the previous step's hi (lane-shifted)
// into this step's lo and resolves the 8-lane carry ripple with a
// generate/propagate mask trick (one 8-bit add does the ripple).
typedef struct { sb_vec prev_hi; unsigned cin; } q_chain;
static inline sb_vec q_chain_step(q_chain *c, q_lohi p){
    sb_vec his = sb_alignr64(p.hi, c->prev_hi, 7);
    sb_vec sum = sb_add(p.lo, his);
    unsigned g  = (unsigned)sb_ltu(sum, his);
    unsigned pr = (unsigned)sb_eq(sum, sb_ones());
    unsigned cn = pr + ((g << 1) | c->cin);
    unsigned cy = cn ^ pr;
    sum = sb_add(sum, sb_set1_64(1), sb_k8(cy), sum);
    c->cin = (cn >> 8) & 1u;
    c->prev_hi = p.hi;
    return sum;
}
// close nf parallel front chains over rp[0, rl): front f's trailing carry
// (hi lane 7 + carry-in) ripples into front f+1; the carry past rl must
// vanish (returns 0 otherwise -- defensive, indicates a precision fault)
static inline int q_chains_close(uint64_t *rp, int64_t rl, const q_chain *ch,
                                 uint32_t nf, size_t limbs_front){
    for(uint32_t f = 0; f < nf; ++f){
        uint64_t hi7;
        memcpy(&hi7, (const char *)&ch[f].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[f].cin;
        if((size_t)(f + 1) * limbs_front >= (size_t)rl){
            if(c) return 0;
            continue;
        }
        uint64_t *q = rp + (size_t)(f + 1) * limbs_front;
        uint64_t *qe = rp + rl;
        while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
        if(c) return 0;
    }
    return 1;
}
// Exact low prefix of a non-wrapping integer product: only carries into
// the requested prefix matter. Carries beyond it are deliberately discarded.
static inline int q_chains_close_prefix(uint64_t *r,int64_t size,const q_chain *ch,uint32_t fronts,size_t stride){
    for(uint32_t f=0;f<fronts;++f){size_t at=size_t(f+1)*stride;if(at>=size_t(size))break;
        uint64_t hi;memcpy(&hi,(const char *)&ch[f].prev_hi+56,8);unsigned __int128 carry=(unsigned __int128)hi+ch[f].cin;
        while(carry&&at<size_t(size)){carry+=r[at];r[at++]=uint64_t(carry);carry>>=64;}
    }return 1;
}
// cyclic close (wrap / middle-product use, rl == nf*limbs_front): any
// carry surviving past rl re-enters at limb 0 -- the emitted array is
// then the product mod 2^(64*rl) - 1. Terminates in <= 2 sweeps (a full
// first sweep only survives an all-0xFF array, which it zeroes).
static inline int q_chains_close_cyc(uint64_t *rp, int64_t rl,
                                     const q_chain *ch, uint32_t nf,
                                     size_t limbs_front){
    uint64_t *qe = rp + rl;
    for(uint32_t f = 0; f < nf; ++f){
        uint64_t hi7;
        memcpy(&hi7, (const char *)&ch[f].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[f].cin;
        uint64_t *q = (size_t)(f + 1) * limbs_front >= (size_t)rl
                    ? rp : rp + (size_t)(f + 1) * limbs_front;
        while(c){
            while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
            q = rp;
        }
    }
    return 1;
}

// final inverse r22 at len n FUSED with the u16 emit: 4 quarter-fronts,
// each with its own carry chain (uncentered band). cyc: close the carry
// chains cyclically (mod 2^(64 rl) - 1) instead of demanding vanish.
template<bool PairFits64=false>
static inline int pq16_inv_final_emit(uint64_t *rp, int64_t rl, double *data,
                                      uint32_t n, const pq16_plan *pl,
                                      int cyc){
    const size_t limbs_q = n / 8u;                   // limbs per quarter
    q_chain ch[4];
    int64_t rem[4];
    for(int f = 0; f < 4; ++f){
        ch[f].prev_hi = sb_zero(); ch[f].cin = 0;
        rem[f] = rl - (int64_t)((size_t)f * limbs_q);
    }
    // named per-front state: a front-indexed o[4][2] defeats SROA here
    // (clang keeps it in memory, ~20 spills in the hot loop)
    double *p0 = data;
    double *p1 = data + 2u * (size_t)(n / 4u);
    double *p2 = data + 4u * (size_t)(n / 4u);
    double *p3 = data + 6u * (size_t)(n / 4u);
    size_t off0 = 0;
    size_t off1 = 1u * limbs_q;
    size_t off2 = 2u * limbs_q;
    size_t off3 = 3u * limbs_q;
    const double *twp = pl->tw22[__builtin_ctz(n)];
    const int twc = pq16_twc(n);
    for(uint32_t t = 0; t < n / 64u; ++t){
        qcv o0[2], o1[2], o2[2], o3[2];
        qcv w1a, w2a, w1b, w2b;
        pq16_tw22_get(twp, twc, &w1a, &w2a, &w1b, &w2b);
        twp += pq16_tw_step(twc);
        for(int u = 0; u < 2; ++u){
            qcv v[4] = { q_ld(p0 + 16 * u), q_ld(p1 + 16 * u),
                         q_ld(p2 + 16 * u), q_ld(p3 + 16 * u) };
            q_ibf4(v, u ? w1b : w1a, u ? w2b : w2a);
            o0[u] = v[0]; o1[u] = v[1]; o2[u] = v[2]; o3[u] = v[3];
        }
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
        q_lohi q;
        if(cyc!=2||rem[0]>0){q = q_pack2<PairFits64>(o0[0], o0[1],off0);
            const auto out=q_chain_step(&ch[0],q);if(rem[0]>0)q_st_tail(rp+off0,out,rem[0]);}
        if(cyc!=2||rem[1]>0){q = q_pack2<PairFits64>(o1[0], o1[1],off1);
            const auto out=q_chain_step(&ch[1],q);if(rem[1]>0)q_st_tail(rp+off1,out,rem[1]);}
        if(cyc!=2||rem[2]>0){q = q_pack2<PairFits64>(o2[0], o2[1],off2);
            const auto out=q_chain_step(&ch[2],q);if(rem[2]>0)q_st_tail(rp+off2,out,rem[2]);}
        if(cyc!=2||rem[3]>0){q = q_pack2<PairFits64>(o3[0], o3[1],off3);
            const auto out=q_chain_step(&ch[3],q);if(rem[3]>0)q_st_tail(rp+off3,out,rem[3]);}
        off0 += 8; off1 += 8; off2 += 8; off3 += 8;
        rem[0] -= 8; rem[1] -= 8; rem[2] -= 8; rem[3] -= 8;
    }
    return cyc==2 ? q_chains_close_prefix(rp,rl,ch,4,limbs_q) : cyc ? q_chains_close_cyc(rp, rl, ch, 4, limbs_q)
               : q_chains_close(rp, rl, ch, 4, limbs_q);
}

// ------------------------------------------------------------------
// centered codec corrections. Digits are centered in a staging copy or
// during bounded input loading: (i16)(d ^ 0x8000) = d - 2^15. Recover the
// signed coefficients via  c_k = c^_k + (S_k << 15) - (C_k << 30)  with S
// the running window digit sum and C the closed-form window overlap.
// ------------------------------------------------------------------
typedef struct {
    const uint16_t *a; const uint16_t *b;
    int64_t na, nb;              // digit counts (4 * limbs)
    int xored;                   // streams are xor-centered: un-xor on load
} q_cctx;

// 16 digits at offset k (may be negative / past the end): fault-suppressed
// masked load, invalid lanes zero
static inline sb_vec q_dig16(const uint16_t *d, int64_t ndig, int64_t k, int xored){
    uint32_t lo = k < 0 ? (uint32_t)(-k > 16 ? 16 : -k) : 0;
    int64_t rem = ndig - k;
    uint32_t hi = rem >= 16 ? 0u : (rem <= 0 ? 16u : (uint32_t)(16 - rem));
    __mmask16 m = sb_k16((0xFFFFu << lo) & (0xFFFFu >> hi));
    if(!m)return sb_zero();
    __m256i v = _mm256_maskz_loadu_epi16(m, d + k);
    if(xored)
        v = _mm256_maskz_mov_epi16(m,
            _mm256_xor_si256(v, _mm256_set1_epi16((short)0x8000)));
    return sb__fn(cvtepu16_epi32)(v);
}
static inline sb_vec q_prefix16_i32(sb_vec x){
    const sb_vec z = sb_zero();
    x = sb__fn(add_epi32)(x, sb__fn(alignr_epi32)(x, z, 15));
    x = sb__fn(add_epi32)(x, sb__fn(alignr_epi32)(x, z, 14));
    x = sb__fn(add_epi32)(x, sb__fn(alignr_epi32)(x, z, 12));
    x = sb__fn(add_epi32)(x, sb__fn(alignr_epi32)(x, z, 8));
    return x;
}
// shared tail: dS prefix-sum -> even/odd S lanes -> corrections with the
// given even/odd C vectors. Updates *S past the 16-digit window.
static inline void q_corr_tail(sb_vec *ce, sb_vec *co, sb_vec dS, int64_t *S,
                               sb_vec Ce, sb_vec Co){
    sb_vec pS = q_prefix16_i32(dS);
    const sb_vec IE = sb_setr_32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0);
    const sb_vec IO = sb_setr_32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0);
    sb_vec base = sb_set1_64(*S);
    sb_vec Se = sb_add(base, sb__fn(cvtepi32_epi64)(
        sb__fn(castsi512_si256)(sb_perm32(IE, pS))));
    sb_vec So = sb_add(base, sb__fn(cvtepi32_epi64)(
        sb__fn(castsi512_si256)(sb_perm32(IO, pS))));
    __m128i top = sb__fn(extracti32x4_epi32)(pS, 3);
    *S += (int32_t)_mm_extract_epi32(top, 3);
    *ce = sb_sub(sb_slli(Se, 15), sb_slli(Ce, 30));
    *co = sb_sub(sb_slli(So, 15), sb_slli(Co, 30));
}
// generic window (handles segment-straddling and out-of-range windows):
// dS = a_fwd + b_fwd - a_lag - b_lag, C(j) = min(j+1, na) - clamp(j+1-nb)
static inline void q_corr_generic(sb_vec *ce, sb_vec *co, const q_cctx *cx,
                                  int64_t k, int64_t *S){
    sb_vec af = q_dig16(cx->a, cx->na, k, cx->xored);
    sb_vec bf = q_dig16(cx->b, cx->nb, k, cx->xored);
    sb_vec al = q_dig16(cx->a, cx->na, k - cx->nb, cx->xored);
    sb_vec bl = q_dig16(cx->b, cx->nb, k - cx->na, cx->xored);
    sb_vec dS = sb__fn(sub_epi32)(sb__fn(add_epi32)(af, bf),
                                sb__fn(add_epi32)(al, bl));
    const sb_vec lane2 = sb_setr_64(1, 3, 5, 7, 9, 11, 13, 15);
    sb_vec j1e = sb_add(sb_set1_64(k), lane2);               // j+1, even digits
    sb_vec j1o = sb_add(j1e, sb_set1_64(1));
    sb_vec nav = sb_set1_64(cx->na), nbv = sb_set1_64(cx->nb);
    sb_vec z = sb_zero();
    sb_vec Ce = sb_sub(sb__fn(min_epi64)(j1e, nav),
        sb__fn(min_epi64)(sb__fn(max_epi64)(sb_sub(j1e, nbv), z), nav));
    sb_vec Co = sb_sub(sb__fn(min_epi64)(j1o, nav),
        sb__fn(min_epi64)(sb__fn(max_epi64)(sb_sub(j1o, nbv), z), nav));
    q_corr_tail(ce, co, dS, S, Ce, Co);
}
// segmented windows: a 16-digit window entirely inside one validity
// segment needs half the digit loads and a linear C.
//   S1 [0, lo):        dS = a_f + b_f          C = k+1        (rising)
//   S2 [lo, hi):       dS = f_fwd - f_lag      C = lo         (constant;
//                      f = the LONGER operand -- note an < bn is legal)
//   S3 [hi, na+nb):    dS = -(a_lag + b_lag)   C = na+nb-1-k  (falling)
// with lo = min(na, nb), hi = max(na, nb).
static inline void q_corr_seg(sb_vec *ce, sb_vec *co, const q_cctx *cx,
                              int64_t k, int64_t *S, int seg){
    const __m256i X = _mm256_set1_epi16((short)0x8000);
    sb_vec dS;
    if(seg == 1){
        __m256i a = _mm256_loadu_si256((const __m256i *)(cx->a + k));
        __m256i b = _mm256_loadu_si256((const __m256i *)(cx->b + k));
        if(cx->xored){ a = _mm256_xor_si256(a, X); b = _mm256_xor_si256(b, X); }
        dS = sb__fn(add_epi32)(sb__fn(cvtepu16_epi32)(a), sb__fn(cvtepu16_epi32)(b));
    }else if(seg == 2){
        const uint16_t *f; const uint16_t *l2;
        if(cx->na >= cx->nb){ f = cx->a + k; l2 = cx->a + k - cx->nb; }
        else                { f = cx->b + k; l2 = cx->b + k - cx->na; }
        __m256i a = _mm256_loadu_si256((const __m256i *)f);
        __m256i l = _mm256_loadu_si256((const __m256i *)l2);
        if(cx->xored){ a = _mm256_xor_si256(a, X); l = _mm256_xor_si256(l, X); }
        dS = sb__fn(sub_epi32)(sb__fn(cvtepu16_epi32)(a), sb__fn(cvtepu16_epi32)(l));
    }else{
        __m256i l = _mm256_loadu_si256((const __m256i *)(cx->a + k - cx->nb));
        __m256i m = _mm256_loadu_si256((const __m256i *)(cx->b + k - cx->na));
        if(cx->xored){ l = _mm256_xor_si256(l, X); m = _mm256_xor_si256(m, X); }
        dS = sb__fn(sub_epi32)(sb_zero(),
            sb__fn(add_epi32)(sb__fn(cvtepu16_epi32)(l), sb__fn(cvtepu16_epi32)(m)));
    }
    sb_vec Ce, Co;
    const sb_vec lane2 = sb_setr_64(0, 2, 4, 6, 8, 10, 12, 14);
    if(seg == 1){
        sb_vec je = sb_add(sb_set1_64(k + 1), lane2);
        Ce = je; Co = sb_add(je, sb_set1_64(1));
    }else if(seg == 2){
        Ce = Co = sb_set1_64(cx->na <= cx->nb ? cx->na : cx->nb);
    }else{
        sb_vec je = sb_sub(sb_set1_64(cx->na + cx->nb - 1 - k), lane2);
        Ce = je; Co = sb_sub(je, sb_set1_64(1));
    }
    q_corr_tail(ce, co, dS, S, Ce, Co);
}
static inline void q_corr_auto(sb_vec *ce, sb_vec *co, const q_cctx *cx,
                               int64_t k, int64_t *S){
    int64_t k2 = k + 16;
    int64_t lo = cx->nb <= cx->na ? cx->nb : cx->na;
    int64_t hi = cx->nb <= cx->na ? cx->na : cx->nb;
    if(k2 <= lo)                              q_corr_seg(ce, co, cx, k, S, 1);
    else if(k >= lo && k2 <= hi)              q_corr_seg(ce, co, cx, k, S, 2);
    else if(k >= hi && k2 <= cx->na + cx->nb) q_corr_seg(ce, co, cx, k, S, 3);
    else                                      q_corr_generic(ce, co, cx, k, S);
}
// sum of digits j with 0 <= j < min(t, ndig) (front checkpoint helper)
static inline int64_t q_digsum(const uint16_t *d, int64_t ndig, int64_t t,
                               int xored){
    if(t > ndig) t = ndig;
    if(t <= 0) return 0;
    const __m256i X = _mm256_set1_epi16((short)0x8000);
    int64_t s = 0, i = 0;
    sb_vec acc = sb_zero();
    for(; i + 32 <= t; i += 32){
        __m256i a = _mm256_loadu_si256((const __m256i *)(d + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(d + i + 16));
        if(xored){ a = _mm256_xor_si256(a, X); b = _mm256_xor_si256(b, X); }
        acc = sb__fn(add_epi32)(acc, sb__fn(add_epi32)(
            sb__fn(cvtepu16_epi32)(a), sb__fn(cvtepu16_epi32)(b)));
        if((i & 8191) == 8160){ s += sb__fn(reduce_add_epi32)(acc); acc = sb_zero(); }
    }
    s += sb__fn(reduce_add_epi32)(acc);
    for(; i < t; ++i) s += (uint16_t)(xored ? d[i] ^ 0x8000 : d[i]);
    return s;
}
static inline int64_t q_sbase(const q_cctx *cx, int64_t k0){
    int x = cx->xored;
    return q_digsum(cx->a, cx->na, k0, x) - q_digsum(cx->a, cx->na, k0 - cx->nb, x)
         + q_digsum(cx->b, cx->nb, k0, x) - q_digsum(cx->b, cx->nb, k0 - cx->na, x);
}

// flat centered emit: data is already fully inverse-transformed into
// natural order (the final r22 / PFA butterfly ran as a separate mem
// pass). One sequential spectrum stream + the correction digit streams +
// one write stream + a single carry chain: prefetch-friendly where the
// fused emit's ~25 concurrent streams go latency-bound (this whole band
// is DRAM-bound).
// cycd = 0: linear (early stop at na+nb digits, trailing carry must
// vanish). cycd = D > 0: CYCLIC over D digits (wrap/middle-product use,
// value mod 2^(16 D) - 1): wrapped coefficients c_e + c_{e+D} need the
// SUM of both diagonals' centering corrections, so a SECOND correction
// stream runs at kd + D over the wrapped prefix (primed via q_digsum
// checkpoints: S_{D-1} = the a/b window sums at digit D-1); the trailing
// carry re-enters at limb 0.
static inline int pq16_emit_c_flat(uint64_t *rp, int64_t rl, const double *data,
                                   uint32_t n, const q_cctx *cx,
                                   int64_t cycd){
    q_chain ch; ch.prev_hi = sb_zero(); ch.cin = 0;
    int64_t S = 0, kd = 0, rem = rl;
    // coefficients beyond na+nb digits are exactly zero: stop there
    // (cyclic: every position of the D-digit array is live)
    const uint32_t tend = (uint32_t)(((uint64_t)(cx->na + cx->nb) + 31) / 32);
    const uint32_t tlim = cycd ? n / 16u
                        : (tend < n / 16u ? tend : n / 16u);
    int64_t S2 = 0;
    if(cycd){
        int64_t ta = cycd - cx->nb, tb = cycd - cx->na;
        S2 = q_digsum(cx->a, cx->na, cycd, cx->xored)
           - q_digsum(cx->a, cx->na, ta, cx->xored)
           + q_digsum(cx->b, cx->nb, cycd, cx->xored)
           - q_digsum(cx->b, cx->nb, tb, cx->xored);
    }
    uint64_t *rp0 = rp;
    for(uint32_t t = 0; t < tlim; ++t){
        qcv o0 = q_ld(data + 32u * (size_t)t);
        qcv o1 = q_ld(data + 32u * (size_t)t + 16u);
        sb_vec e0, c0, e1, c1;
        q_corr_auto(&e0, &c0, cx, kd, &S);
        q_corr_auto(&e1, &c1, cx, kd + 16, &S);
        if(cycd && kd + cycd < cx->na + cx->nb){
            sb_vec e2, c2, e3, c3;
            q_corr_auto(&e2, &c2, cx, kd + cycd, &S2);
            q_corr_auto(&e3, &c3, cx, kd + 16 + cycd, &S2);
            e0 = sb_add(e0, e2); c0 = sb_add(c0, c2);
            e1 = sb_add(e1, e3); c1 = sb_add(c1, c3);
        }
        kd += 32;
        q_lohi q = q_pack2i(sb_add(q_mround_s(o0.re), e0),
                            sb_add(q_mround_s(o0.im), c0),
                            sb_add(q_mround_s(o1.re), e1),
                            sb_add(q_mround_s(o1.im), c1));
        q_st_tail(rp, q_chain_step(&ch, q), rem);
        rp += 8; rem -= 8;
    }
    uint64_t hi7;
    memcpy(&hi7, (const char *)&ch.prev_hi + 56, 8);
    if(!cycd)
        return hi7 + ch.cin == 0;
    {   /* trailing carry re-enters at limb 0 (mod 2^(64 rl) - 1) */
        unsigned __int128 c = (unsigned __int128)hi7 + ch.cin;
        uint64_t *q = rp0, *qe = rp0 + rl;
        while(c){
            while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
            q = rp0;
        }
    }
    return 1;
}

// ------------------------------------------------------------------
// PFA inverse output. Uncentered: fused with the emit -- M branch
// streams, inverse radix-M per tile, lane-shuffle into M natural-order
// region fronts (region b = natural complex [b*n, (b+1)*n)), one carry
// chain each. Region b's tile at step t takes butterfly output
// y[(b*n + 8t) mod M] lane-shifted by the residue masks.
// ------------------------------------------------------------------
// merge form (M = 3). Both sub-tiles of all fronts cannot stay in
// registers; the shuffled outputs stage through an L1 buffer.
template<bool PairFits64=false>
__attribute__((always_inline))
static inline int q_pfa_emit_merge(uint64_t *rp, int64_t rl, double *data,
                                   uint32_t n, uint32_t M, int cyc){
    uint8_t km[8]; q_kmasks(km, M);
    const double *br[7]; size_t offsets[7]; uint32_t phi[7];
    int64_t rme[7];
    q_chain ch[7];
    const size_t limbs_b = (size_t)n / 2u;
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b;
        offsets[b] = (size_t)b * limbs_b;
        rme[b] = rl - (int64_t)((size_t)b * limbs_b);
        phi[b] = (uint32_t)(((uint64_t)b * n) % M);
        ch[b].prev_hi = sb_zero(); ch[b].cin = 0;
    }
    alignas(64) double stage[7 * 2 * 16];
    for(uint32_t t = 0; t < n / 8u; t += 2){
        for(int u = 0; u < 2; ++u){
            qcv x[7], y[7];
            for(uint32_t b = 0; b < M; ++b){
                x[b] = q_ld(br[b]);
                br[b] += 16;
            }
            q_pfa_bfly(x, y, M, 1);
            for(uint32_t b = 0; b < M; ++b){
                uint32_t ph = (phi[b] + (uint32_t)(8 * u)) % M;
                q_st(stage + 32u * b + 16u * (uint32_t)u, q_lane_merge(y, M, ph, km));
            }
        }
        for(uint32_t b = 0; b < M; ++b){
            if(cyc!=2||rme[b]>0){q_lohi q=q_pack2<PairFits64>(q_ld(stage+32u*b),q_ld(stage+32u*b+16u),offsets[b]);
                const auto out=q_chain_step(&ch[b],q);if(rme[b]>0)q_st_tail(rp+offsets[b],out,rme[b]);}
            offsets[b] += 8; rme[b] -= 8;
            phi[b] = (phi[b] + 16u) % M;
        }
    }
    return cyc==2 ? q_chains_close_prefix(rp,rl,ch,M,limbs_b) : cyc ? q_chains_close_cyc(rp, rl, ch, M, limbs_b)
               : q_chains_close(rp, rl, ch, M, limbs_b);
}
// twiddle-fixup form (M = 5, 7): pre-rotate by conj(w^{b l}), inverse-DFT
// over relabeled branches; the region output is then a pure index relabel
// (rg[phi] rotates by 16 mod M, all vector slots stay constant).
template<bool PairFits64=false>
__attribute__((always_inline))
static inline int q_pfa_emit_tw(uint64_t *rp, int64_t rl, double *data,
                                uint32_t n, uint32_t M, const double *wt,
                                int cyc){
    const double *br[7]; size_t offsets[7];
    int64_t rme[7];
    q_chain ch[7];
    const size_t limbs_b = (size_t)n / 2u;
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b;
        offsets[b] = (size_t)b * limbs_b;
        rme[b] = rl - (int64_t)((size_t)b * limbs_b);
        ch[b].prev_hi = sb_zero(); ch[b].cin = 0;
    }
    uint32_t rg[7];
    for(uint32_t b = 0; b < M; ++b)
        rg[(uint32_t)(((uint64_t)b * n) % M)] = b;
    const uint32_t sh8 = 8u % M, sh16 = 16u % M;
    alignas(64) double stage[7 * 2 * 16];
    for(uint32_t t = 0; t < n / 8u; t += 2){
        for(int u = 0; u < 2; ++u){
            qcv x[7], y[7];
            for(uint32_t b = 0; b < M; ++b){
                qcv w = q_ld(wt + 16u * b);
                x[b] = q_mulc(q_ld(br[b]), w);
                br[b] += 16;
            }
            q_pfa_bfly(x, y, M, 1);
            for(uint32_t j = 0; j < M; ++j){
                uint32_t slot = u ? rg[(j + M - sh8) % M] : rg[j];
                q_st(stage + 32u * slot + 16u * (uint32_t)u, y[j]);
            }
        }
        for(uint32_t b = 0; b < M; ++b){
            if(cyc!=2||rme[b]>0){q_lohi q=q_pack2<PairFits64>(q_ld(stage+32u*b),q_ld(stage+32u*b+16u),offsets[b]);
                const auto out=q_chain_step(&ch[b],q);if(rme[b]>0)q_st_tail(rp+offsets[b],out,rme[b]);}
            offsets[b] += 8; rme[b] -= 8;
        }
        uint32_t tmp[7];
        for(uint32_t j = 0; j < M; ++j) tmp[(j + sh16) % M] = rg[j];
        for(uint32_t j = 0; j < M; ++j) rg[j] = tmp[j];
    }
    return cyc==2 ? q_chains_close_prefix(rp,rl,ch,M,limbs_b) : cyc ? q_chains_close_cyc(rp, rl, ch, M, limbs_b)
               : q_chains_close(rp, rl, ch, M, limbs_b);
}
template<bool PairFits64=false>
static inline int pq16_pfa_emit(uint64_t *rp, int64_t rl, double *data,
                                uint32_t n, uint32_t M, const pq16_plan *pl,
                                int cyc){
    switch(M){
    case 3:  return q_pfa_emit_merge<PairFits64>(rp, rl, data, n, 3u, cyc);
    case 5:  return q_pfa_emit_tw<PairFits64>(rp, rl, data, n, 5u, pl->wt5, cyc);
    default: return q_pfa_emit_tw<PairFits64>(rp, rl, data, n, 7u, pl->wt7, cyc);
    }
}

// PFA inverse butterfly to natural order, no pack (centered band: the
// corrections then run in the prefetch-friendly flat emit). dst must not
// alias data -- the spare spectrum buffer serves.
static inline void q_pfa_natural_rng(double *dst, const double *data,
                                     uint32_t n, uint32_t M,
                                     uint32_t t0, uint32_t t1);
__attribute__((always_inline))
static inline void q_pfa_natural_impl(double *dst, const double *data,
                                      uint32_t n, uint32_t M){
    q_pfa_natural_rng(dst, data, n, M, 0, n / 8u);
}
static inline void pq16_pfa_natural(double *dst, const double *data,
                                    uint32_t n, uint32_t M){
    switch(M){
    case 3:  q_pfa_natural_impl(dst, data, n, 3u); break;
    case 5:  q_pfa_natural_impl(dst, data, n, 5u); break;
    default: q_pfa_natural_impl(dst, data, n, 7u); break;
    }
}

/* G5-P4: pfa_natural over t-range [t0, t1): stream pointers advance 16
 * doubles per t and phi rotates (+8) mod M -- both closed-form at t0 */
__attribute__((always_inline))
static inline void q_pfa_natural_rng(double *dst, const double *data,
                                     uint32_t n, uint32_t M,
                                     uint32_t t0, uint32_t t1){
    uint8_t km[8]; q_kmasks(km, M);
    const double *br[7]; double *w[7]; uint32_t phi[7];
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b + 16u * (size_t)t0;
        w[b]   = dst + 2u * (size_t)n * b + 16u * (size_t)t0;
        phi[b] = (uint32_t)(((uint64_t)b * n + 8u * (uint64_t)t0) % M);
    }
    for(uint32_t t = t0; t < t1; ++t){
        qcv x[7], y[7];
        for(uint32_t b = 0; b < M; ++b){
            x[b] = q_ld(br[b]);
            br[b] += 16;
        }
        q_pfa_bfly(x, y, M, 1);
        for(uint32_t b = 0; b < M; ++b){
            q_st(w[b], q_lane_merge(y, M, phi[b], km));
            w[b] += 16;
            phi[b] = (phi[b] + 8u) % M;
        }
    }
}
typedef struct {
    double *dst;
    const double *data;
    uint32_t n, M;
} pq16_nat_fctx;
static void pq16_nat_forfn(void *c_, uint64_t lo, uint64_t hi, int w,
                           scratch *ws){
    pq16_nat_fctx *c = (pq16_nat_fctx *)c_;
    (void)w; (void)ws;
    q_pfa_natural_rng(c->dst, c->data, c->n, c->M, (uint32_t)lo,
                      (uint32_t)hi);
}
static inline void pq16_pfa_natural_w(double *dst, const double *data,
                                      uint32_t n, uint32_t M,
                                      sbn_team *t, int nthr){
    const uint32_t iters = n / 8u;
    if(t && nthr > 1 && iters >= 128){
        pq16_nat_fctx c = { dst, data, n, M };
        sbn_parallel_for(t, nthr, 0, iters, 16, SBN_FOR_STATIC,
                         pq16_nat_forfn, &c);
    }else{
        q_pfa_natural_rng(dst, data, n, M, 0, iters);
    }
}

// ------------------------------------------------------------------
// size chooser + public entry
// ------------------------------------------------------------------
/* the historical TLS singleton is now an alias of the SHARED plan
 * (G5-P0b; storage in src/plans.cpp -- link libsbn.a). Kept as a macro
 * so every `&pq16_tls_plan` call site stays verbatim. */


// smallest supported transform >= need: pow2 N or M*2^L (M = 3: branch >=
// 256; M = 5, 7: branch >= 128). Ties prefer the smaller M / pow2.
#ifndef PQ16_PFA7_MIN_BRANCH
#define PQ16_PFA7_MIN_BRANCH 128u
#endif
// largest supported transform (complex points); probed empirically with
// adversarial operands: pow2 and PFA-3/5 hold to 2^17 uncentered, the
// direct-form radix-7 butterfly's longer FMA chains break first at
// N = 7*2^14, so PFA-7 is precision-capped at 2^16 uncentered. The
// centered band's headroom admits everything to 2^19.
#ifndef PQ16_MAX_N
#define PQ16_MAX_N (1u << 17)
#endif
#ifndef PQ16_PFA7_MAX_N
#define PQ16_PFA7_MAX_N (1u << 16)
#endif
#ifndef PQ16_MAX_N_C
#define PQ16_MAX_N_C (1u << 19)
#endif
#if PQ16_PLAN_CAP_BRANCH != PQ16_MAX_N_C
#error "PQ16_PLAN_CAP_BRANCH must track PQ16_MAX_N_C (shared-plan pre-ensure)"
#endif
typedef struct { uint32_t nfull, branch, M; } pq16_size;

static inline pq16_size pq16_choose(uint32_t need, uint32_t maxn, int cen, uint32_t minimum_pow2=512){
    uint32_t p2 = minimum_pow2;
    while(p2 < need) p2 <<= 1;
    pq16_size best = { p2, p2, 1 };
    static const uint32_t Ms[3] = { 3, 5, 7 };
    for(int i = 0; i < 3; ++i){
        uint32_t M = Ms[i];
        uint32_t br = M == 3 ? 256u : 128u;
        while(M * br < need) br <<= 1;
        if(M == 7u && br < PQ16_PFA7_MIN_BRANCH) continue;
        if(M == 7u && !cen && M * br > PQ16_PFA7_MAX_N) continue;
        if(M * br < best.nfull && M * br <= maxn){
            best.nfull = M * br; best.branch = br; best.M = M;
        }
    }
    return best;
}

// rp[0 .. an+bn) = {ap, an} * {bp, bn}. Returns 1 on success, 0 if the
// size is outside the supported band. rp must not overlap the inputs.
// Workspace comes from sc; the twiddle tables persist thread-locally.
// Linear-product geometry choice INCLUDING the band/precision guards;
// nfull == 0 means out of band (caller falls back). Single source for
// pq16_mul_r, the band tests, and the interface negotiate.
//
// Top-grid peak guard (2026-07-08 probe): at the ONE pure-radix-2 top
// centered point nfull = 2^19 (M = 1, deepest chain), saturated products
// with peak coefficient fill 4*min(an,bn) above ~0.88*nfull round a
// single coefficient wrong (1-ulp digit-slot errors, bit-identical in
// fft16 -- a legacy envelope gap; the historical adversarial suite hit
// only the exact cap point 131072^2, whose particular rounding happens
// to land: probes show BOTH balanced and unbalanced shapes fail in the
// (~0.88, 1] fill range, e.g. 124416^2, 130560^2, 131072x131071).
// Every PFA point is clean at full saturated fill. Guard at 3/4 fill
// (2x margin below observed onset). Proper fix = emit-correction /
// error-analysis work item (ROADMAP R0.1 note).
static inline pq16_size pq16_lin_choose(uint64_t an, uint64_t bn, int *cen_out, uint32_t minimum_pow2=512){
    pq16_size z = { 0, 0, 0 };
    if(!an || !bn) return z;
    const uint64_t need = 2u * (an + bn);                 // complex points
    const int cen = PQ16_FORCE_CENTERED || need > PQ16_MAX_N;
    const uint32_t maxn = cen ? PQ16_MAX_N_C : PQ16_MAX_N;
    if(cen_out) *cen_out = cen;
    if(need > maxn) return z;                             // precision cap
    pq16_size fs = pq16_choose((uint32_t)need, maxn, cen, minimum_pow2);
    if(cen && fs.M == 1u && fs.nfull == PQ16_MAX_N_C &&
       4u * (an < bn ? an : bn) > 3u * (uint64_t)(fs.nfull / 4u))
        return z;                                         // top-grid guard
    return fs;
}

// The ONE pointwise -> inverse -> emit tail shared by every pipeline user
// (pq16_mul_r, pq16_mulmid_r, the ring product, the interface apply --
// historically FOUR hand-expanded copies; A12b collapse). A is CONSUMED
// (the product lands in it); B is preserved. cyc = emit the full ring
// (cyclic digit length 2*nfull), else the linear value. nat = spare
// spectrum for the centered PFA natural-order pass (callers whose B is
// disposable pass B itself; callers preserving B pass fresh scratch;
// unused when !cen || M == 1). Emits rl limbs DIRECTLY into rp (the
// centered emit may spill a few limbs of pad past rl -- callers without
// pad stage through scratch). Returns 1, or 0 on the emit's
// carry-past-rl safety net.
static inline int pq16_conv_emit(uint64_t *rp, int64_t rl, double *A,
                                 const double *B, uint32_t n, uint32_t M,
                                 uint32_t nfull, int cen, int cyc,
                                 q_cctx *cx, double *nat, pq16_plan *pl){
    const int64_t cycd = cyc==1 ? 2 * (int64_t)nfull : 0;
    if(M == 1u){
        pq16_pointwise_ileaves(A, (double *)B, n, 1.0 / (double)n,
                               pq16_shape_of(n).leaf, pl);
        pq16_inv_r8_only(A, n, pl);
        if(cen){
            pq16_mem_ir22(A, n, pl->tw22[__builtin_ctz(n)]);
            return pq16_emit_c_flat(rp, rl, A, n, cx, cycd);
        }
        if(!cyc && nfull<=(1u<<16))
            return pq16_inv_final_emit<true>(rp,rl,A,n,pl,cyc);
        return pq16_inv_final_emit(rp, rl, A, n, pl, cyc);
    }
    const double sc_d = 1.0 / (double)nfull;
    const double *wrt = M == 3 ? PQ16_W3_RE : M == 5 ? PQ16_W5_RE : PQ16_W7_RE;
    const double *wit = M == 3 ? PQ16_W3_IM : M == 5 ? PQ16_W5_IM : PQ16_W7_IM;
    const uint32_t pwleaf = pq16_odd_leaf(n);
    // branch 0 self-pairs; branches (b, M-b) cross-pair with omega_M^b
    pq16_pointwise_ileaves(A, (double *)B, n, sc_d, pwleaf, pl);
    for(uint32_t b = 1; b <= M / 2u; ++b){
        const uint32_t b2 = M - b;
        pq16_pointwise_cross_ileaves(
            A + 2u * (size_t)n * b,           A + 2u * (size_t)n * b2,
            (double *)B + 2u * (size_t)n * b, (double *)B + 2u * (size_t)n * b2,
            n, sc_d, wrt[b], wit[b], pwleaf, pl);
    }
    for(uint32_t b = 0; b < M; ++b)
        pq16_odd_inv(A + 2u * (size_t)n * b, n, pl);
    if(cen){
        pq16_pfa_natural(nat, A, n, M);
        return pq16_emit_c_flat(rp, rl, nat, (uint32_t)(M * n), cx, cycd);
    }
    if(!cyc && nfull<=(1u<<16))
        return pq16_pfa_emit<true>(rp,rl,A,n,M,pl,cyc);
    return pq16_pfa_emit(rp, rl, A, n, M, pl, cyc);
}

/* G5-P4: conv_emit with wide inverse legs -- pointwise and the emits
 * stay serial (mirror-pair bands / carry-chain fronts are the recorded
 * P4 remainder); M == 1 fans the inverse r8 chunks, M > 1 runs the
 * per-branch inverse+ir22 as tasks. Bit-identical to pq16_conv_emit. */
struct pq16_pair_tasks {double *A,*B;uint32_t n,M,leaf;pq16_plan *plan;double scale;};
static void pq16_pair_task(void *arg,int pair,int,scratch *){
    const auto &c=*static_cast<pq16_pair_tasks *>(arg);
    if(!pair){pq16_pointwise_ileaves(c.A,c.B,c.n,c.scale,c.leaf,c.plan);return;}
    const unsigned b=unsigned(pair),other=c.M-b;
    const double *wr=c.M==3?PQ16_W3_RE:c.M==5?PQ16_W5_RE:PQ16_W7_RE;
    const double *wi=c.M==3?PQ16_W3_IM:c.M==5?PQ16_W5_IM:PQ16_W7_IM;
    pq16_pointwise_cross_ileaves(c.A+2ul*c.n*b,c.A+2ul*c.n*other,c.B+2ul*c.n*b,c.B+2ul*c.n*other,c.n,c.scale,wr[b],wi[b],c.leaf,c.plan);
}
static inline int pq16_conv_emit_w(uint64_t *rp, int64_t rl, double *A,
                                   const double *B, uint32_t n, uint32_t M,
                                   uint32_t nfull, int cen, int cyc,
                                   q_cctx *cx, double *nat, pq16_plan *pl,
                                   sbn_team *t, int nthr){
    if(!t || nthr <= 1)
        return pq16_conv_emit(rp, rl, A, B, n, M, nfull, cen, cyc, cx,
                              nat, pl);
    const int64_t cycd = cyc==1 ? 2 * (int64_t)nfull : 0;
    if(M == 1u){
        pq16_pointwise_ileaves_w(A, (double *)B, n, 1.0 / (double)n,
                                 pq16_shape_of(n).leaf, pl, t, nthr);
        pq16_inv_r8_only_w(A, n, pl, t, nthr);
        if(cen){
            pq16_mem_ir22_w(A, n, pl->tw22[__builtin_ctz(n)], t, nthr);
            return pq16_emit_c_flat(rp, rl, A, n, cx, cycd);
        }
        if(!cyc && nfull<=(1u<<16))
            return pq16_inv_final_emit<true>(rp,rl,A,n,pl,cyc);
        return pq16_inv_final_emit(rp, rl, A, n, pl, cyc);
    }
    const double sc_d = 1.0 / (double)nfull;
    const double *wrt = M == 3 ? PQ16_W3_RE : M == 5 ? PQ16_W5_RE : PQ16_W7_RE;
    const double *wit = M == 3 ? PQ16_W3_IM : M == 5 ? PQ16_W5_IM : PQ16_W7_IM;
    const uint32_t pwleaf = pq16_shape_of(n).leaf;
    if constexpr(PQ16_PFA_PARALLEL){
        pq16_pair_tasks jobs{A,(double *)B,n,M,pwleaf,pl,sc_d};sbn_run_tasks(t,nthr,int((M+1)/2),pq16_pair_task,&jobs);
    }else{
        pq16_pointwise_ileaves(A, (double *)B, n, sc_d, pwleaf, pl);
        for(uint32_t b = 1; b <= M / 2u; ++b){
            const uint32_t b2 = M - b;
            pq16_pointwise_cross_ileaves(A+2ul*n*b,A+2ul*n*b2,(double *)B+2ul*n*b,(double *)B+2ul*n*b2,n,sc_d,wrt[b],wit[b],pwleaf,pl);
        }
    }
    {
        const unsigned ct = pq16_pfa_tier(n,M,nthr);
        if(ct == 3){
            pq16_branch_fctx c = { A, pl->tw22[__builtin_ctz(n)], n, pl, 1 };
            sbn_run_tasks(t, nthr, (int)M,
                          pq16_branch_taskfn, &c);
        }else{
            /* chunk grid first (inverse = post-order), then per-branch
             * trailing sweeps + mem_ir22 */
            pq16_shape sh = pq16_shape_of(n);
            const uint32_t chunk = PQ16_TIERS[ct];
            uint32_t s1 = 0;
            while(s1 < sh.cnt && sh.len[s1] > chunk) s1++;
            pq16_pfa_fctx c = { A, &sh, pl, pl->tw22[__builtin_ctz(n)],
                                n, s1, chunk, ct, 1, (int)(n / chunk) };
            const int nt = (int)M * c.nchunk;
            sbn_run_tasks(t, nthr, nt,
                          pq16_pfa_chunk_taskfn, &c);
            sbn_run_tasks(t, nthr, (int)M,
                          pq16_pfa_r22_taskfn, &c);
        }
    }
    if(cen){
        pq16_pfa_natural_w(nat, A, n, M, t, nthr);
        return pq16_emit_c_flat(rp, rl, nat, (uint32_t)(M * n), cx, cycd);
    }
    if(!cyc && nfull<=(1u<<16))
        return pq16_pfa_emit<true>(rp,rl,A,n,M,pl,cyc);
    return pq16_pfa_emit(rp, rl, A, n, M, pl, cyc);
}

/* G5-P4 core: t/nthr fan the input + forward + inverse legs out; the
 * staging copies, pointwise, and emits stay serial. */
static inline int pq16_mul_core(uint64_t *rp,
                                const uint64_t *ap, ptrdiff_t an,
                                const uint64_t *bp, ptrdiff_t bn,
                                scratch *sc, sbn_team *t, int nthr, pq16_plan *pl, pq16_size fs, int cen){
    if(an <= 0 || bn <= 0) return 0;
    if(!fs.nfull) return 0;
    require(2u*(uint64_t(an)+uint64_t(bn))<=fs.nfull,SBN3_FATAL_ARGUMENT,"pq16 planned length");
    require(!cen || fs.M!=1 || fs.nfull!=PQ16_MAX_N_C || 4u*uint64_t(an<bn?an:bn)<=3u*(fs.nfull/4u),SBN3_FATAL_MATH,"pq16 peak coefficient guard");
    const uint32_t n = fs.branch, M = fs.M, nfull = fs.nfull;

    require(pl->ready_mask==(1u<<__builtin_ctz(n)) && pl->pq_n>=n,SBN3_FATAL_ARGUMENT,"pq16 prepared basis");

    SCRATCH(sc);
    // both spectra in one allocation, manually 128B-aligned (one full
    // tile; the stack tier guarantees only 64); the +32 doubles keep db
    // 128B-aligned while staggering it off an exact power-of-two stride
    // from da
    const bool square=ap==bp && an==bn;
    // Centered PFA still needs a disjoint natural-order inverse destination.
    const bool one_spectrum=square && (!cen || M==1);
    double *daw = SALLOC(sc, double, (one_spectrum?2:4) * (size_t)nfull + 48);
    double *da = (double *)(((uintptr_t)daw + 127) & ~(uintptr_t)127);
    double *spare=one_spectrum?nullptr:da+2*(size_t)nfull+32;
    double *db=square?da:spare;

    const int64_t rl = (int64_t)(an + bn);
    q_cctx cx = { (const uint16_t *)ap, (const uint16_t *)bp,
                  4 * (int64_t)an, 4 * (int64_t)bn, 0 };
    const uint64_t *sa = ap, *sb = bp;
    int64_t ca_cnt = an, cb_cnt = bn;
    uint64_t *cr = NULL;
    int decode_cen=cen;
    if(cen){
        if(pq16_direct_input(nthr)){
            // XOR at the bounded load produces exactly the same signed
            // digits as the old staging copy. Invalid lanes remain zero.
            decode_cen=2;
            if(pq16_direct_output(nthr))cr=rp;
            else cr=SALLOC(sc,uint64_t,(size_t)rl+8);
        }else{
        // staging: centering XOR rides the copy; the copies also normalize
        // the memory layout (reading caller buffers in-place in this
        // BW-bound band costs up to 2x depending on allocation layout)
        const uint64_t XC = 0x8000800080008000ull;
        uint64_t *ca = SALLOC(sc, uint64_t, (size_t)nfull / 2u + 8);
        uint64_t *cb = square?ca:SALLOC(sc, uint64_t, (size_t)nfull / 2u + 8);
        cr = SALLOC(sc, uint64_t, (size_t)rl + 8);
        for(int64_t i = 0; i < an; ++i) ca[i] = ap[i] ^ XC;
        memset(ca + an, 0, ((size_t)(nfull / 2u) - (size_t)an) * 8);
        if(!square){for(int64_t i = 0; i < bn; ++i) cb[i] = bp[i] ^ XC;
            memset(cb + bn, 0, ((size_t)(nfull / 2u) - (size_t)bn) * 8);}
        cx.a = (const uint16_t *)ca;
        cx.b = (const uint16_t *)cb;
        cx.xored = 1;
        sa = ca; sb = cb;
        ca_cnt = cb_cnt = (int64_t)(nfull / 2u);
        }
    }

    // B first so A's spectrum is the cache-hot one when the pointwise
    // consumes and overwrites it; inverse leaves are fused into the
    // pointwise pass. Pointwise -> inverse -> emit = pq16_conv_emit (the
    // shared pipeline tail); db doubles as the centered PFA natural
    // buffer (disposable after the pointwise).
    if(M == 1){
        if(!square){pq16_input_stage_w(db, sb, cb_cnt, n, pl, decode_cen, t, nthr);
            pq16_fwd_core_w(db, n, pl, t, nthr);}
        pq16_input_stage_w(da, sa, ca_cnt, n, pl, decode_cen, t, nthr);
        pq16_fwd_core_w(da, n, pl, t, nthr);
    }else{
        if(!square)pq16_pfa_fwd_w(db, sb, cb_cnt, n, M, pl, decode_cen, t, nthr);
        pq16_pfa_fwd_w(da, sa, ca_cnt, n, M, pl, decode_cen, t, nthr);
    }
    if(cen){
        if(!pq16_conv_emit_w(cr, rl, da, db, n, M, nfull, 1, 0, &cx, spare, pl,
                             t, nthr))
            return 0;
        if(!pq16_direct_output(nthr))memcpy(rp,cr,(size_t)rl*8);
    }else{
        if(!pq16_conv_emit_w(rp, rl, da, db, n, M, nfull, 0, 0, &cx, db, pl,
                             t, nthr))
            return 0;
    }
    return 1;
}


}
