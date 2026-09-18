/* Imported from libsbn/include/sbn/detail/mulmid/mulmid_kara.h; kernel arithmetic retained. */
#pragma once
namespace sbn::v3::u52 {

/**
 * Karatsuba middle product (Harvey's integer algorithm, GMP mpn_toom42_mulmid,
 * translated to 52-bit redundant limbs). Reference: D. Harvey, "The Karatsuba
 * middle product for integers"; GMP mpn/generic/toom42_mulmid.c.
 *
 * Conventions (ours; GMP's a/b are swapped): a is the SHORT operand (an = n),
 * b the LONG (bn = 2n-1, balanced), output = n+1 RAW limbs, value
 * sum r[i]*2^(52i) = the exact algebraic MP (same contract as mulmid_basecase,
 * except limbs are SIGNED: interior |r[i]| < ~2^55, top limb < 2^63).
 *
 * Structure per level (m = n/2):
 *   s   = {b,3m-1} + {b+m,3m-1}   canonical 52-bit add, carries exposed
 *   t   = |a_hi - a_lo|           canonical sub, borrows exposed, sign flag
 *   p0  = MP(a_hi, s[0..2m-2])    p1 = MP(t, b+m)     p2 = MP(a_lo, s+m)
 *   out = (P~0 + tau*P~1) + B^m (P~2 - tau*P~1)
 * where P~i = pi plus error terms: dot products of operand limbs with the
 * carry/borrow bits in reversed order (Harvey's telescoping lemma: only the
 * O(n) boundary carries survive). In raw land each dot fits ONE u64
 * (< m*2^52), is applied by two wrapping adds at limbs (j, j+1), and the
 * recombination is plain 64-bit adds -- no carry propagation anywhere.
 *
 * Bounds audit (why this never wraps past 2^63): basecase sub-results have
 * limbs <= 2m*2^52 with THIN edges (limb 0 is lo-only, limb m hi-only,
 * <= m*2^52); composite sub-results are rebalanced to |limb| <= 2^52+2^11
 * with the overflow pushed into an extension limb p[m+1] (buffers are m+2).
 * Leaf-level combine: 4*(m+1)*2^52 < 2^63 for m <= 255 (threshold cap);
 * composite levels: 4*(2^52+2^11) + dots-split ~ 2^55. The final top limb
 * absorbs the extension fold: |r[2m]| <= 2^62.5.
 *
 * Input contract: a, b canonical (< 2^52 per limb). b must be readable up to
 * bn + 32 limbs (basecase over-read, same as elsewhere); a-side uses
 * fault-suppressed masked loads below index 0.
 */

#ifndef MULMID_KARA_THRESHOLD
#define MULMID_KARA_THRESHOLD 128
#endif

#define M52_ ((1ull << 52) - 1)

// s-pass: sb[0..3m-2] = {b,3m-1} + {b+m,3m-1} canonical, ONE sweep, plus the
// four error dots of Harvey's segmentation (e0: carries in [0,m-1) against
// a_hi reversed; e1/e2: [m-1,2m-1) against a_hi/a_lo; e3: [2m-1,3m-1) against
// a_lo). Fused because the old three segment calls put segments 2 and 3 on
// odd limb offsets (b+m-1), splitting a cache line on EVERY load and store --
// in situ that ran at ~2x the aligned c/l. One i = 0 mod 8 grid keeps all
// u/v/sb traffic b-aligned; only the 1-per-chunk dot y-loads stay odd.
// The dots share two reversed y streams: lane at global i reads
//   ya[i] = a[2m-2-i] (e0 in seg1, e2 in seg2; live i < 2m-1)
//   yb[i] = a[3m-2-i] (e1 in seg2, e3 in seg3; live i >= m-1)
// so per-chunk dot cost matches the old kernels; only the two boundary
// chunks (containing m-1 and 2m-1) split their carry mask between
// accumulators. Carry resolution per chunk: g/p masks + the 9-bit GPR
// lookahead c = ((g<<1|ci)+p)^p. The final carry-out is discarded (it
// telescopes out of the MP window). Requires m >= 16 (boundary chunks
// distinct); a readable at masked lanes only, as elsewhere.
static inline void u52_s_err4(uint64_t *sb, const uint64_t *b, const uint64_t *a, int64_t m,
                              uint64_t *e0, uint64_t *e1, uint64_t *e2, uint64_t *e3){
    if(m < 16){ // scalar fallback: only reachable at test-sized thresholds
        uint64_t x0 = 0, x1 = 0, x2 = 0, x3 = 0, cy = 0;
        for(int64_t i = 0; i < 3*m - 1; ++i){
            const uint64_t s = b[i] + b[m + i] + cy;
            sb[i] = s & M52_;
            cy = s >> 52;
            if(!cy) continue;
            if(i < m - 1)        x0 += a[2*m - 2 - i];
            else if(i < 2*m - 1) x1 += a[3*m - 2 - i], x2 += a[2*m - 2 - i];
            else                 x3 += a[3*m - 2 - i];
        }
        *e0 = x0; *e1 = x1; *e2 = x2; *e3 = x3;
        return;
    }
    const sb_vec m52v = sb_set1_64(M52_), onev = sb_set1_64(1);
    const sb_vec rev = sb_setr_64(7,6,5,4,3,2,1,0);
    sb_vec ev0 = sb_zero(), ev1 = sb_zero(), ev2 = sb_zero(), ev3 = sb_zero();
    const int64_t n3 = 3*m - 1;
    uint64_t cy = 0;
    int64_t i = 0;
    // one full-width chunk: sum, canonical store, carry-out mask in cm
    #define SE4_CHUNK(cm) \
        const sb_vec t = sb_add(sb_load(b + i), sb_load(b + m + i)); \
        const uint64_t g = _mm512_cmpgt_epu64_mask(t, m52v); \
        const uint64_t p = _mm512_cmpeq_epi64_mask(t, m52v); \
        const uint64_t c = (((g << 1) | cy) + p) ^ p; \
        cy = (c >> 8) != 0; \
        sb_store(sb + i, sb_and(sb_add(t, onev, (__mmask8)c, t), m52v)); \
        const __mmask8 cm = (__mmask8)(((c >> 1) & 0x7f) | (cy << 7));
    for(; i + 8 <= m - 1; i += 8){                    // seg1: e0 against ya
        SE4_CHUNK(cm)
        ev0 = sb_add(ev0, sb_perm64(rev, sb_load(a + (2*m - 9 - i))), cm, ev0);
    }
    {   // boundary chunk containing m-1: lanes < lo are seg1, >= lo seg2
        const int64_t lo = m - 1 - i;
        SE4_CHUNK(cm)
        const __mmask8 ml = (__mmask8)((1u << lo) - 1);
        const sb_vec ya = sb_perm64(rev, sb_load(a + (2*m - 9 - i)));
        const sb_vec yb = sb_perm64(rev, sb_load(a + (3*m - 9 - i), (__mmask8)((1u << (8 - lo)) - 1)));
        ev0 = sb_add(ev0, ya, (__mmask8)(cm &  ml), ev0);
        ev2 = sb_add(ev2, ya, (__mmask8)(cm & ~ml), ev2);
        ev1 = sb_add(ev1, yb, (__mmask8)(cm & ~ml), ev1);
        i += 8;
    }
    for(; i + 8 <= 2*m - 1; i += 8){                  // seg2: e2/ya + e1/yb
        SE4_CHUNK(cm)
        ev2 = sb_add(ev2, sb_perm64(rev, sb_load(a + (2*m - 9 - i))), cm, ev2);
        ev1 = sb_add(ev1, sb_perm64(rev, sb_load(a + (3*m - 9 - i))), cm, ev1);
    }
    {   // boundary chunk containing 2m-1: lanes < lo are seg2, >= lo seg3
        const int64_t lo = 2*m - 1 - i;
        SE4_CHUNK(cm)
        const __mmask8 ml = (__mmask8)((1u << lo) - 1);
        const sb_vec ya = sb_perm64(rev, sb_load(a + (2*m - 9 - i), (__mmask8)(0xFFu << (8 - lo))));
        const sb_vec yb = sb_perm64(rev, sb_load(a + (3*m - 9 - i)));
        ev2 = sb_add(ev2, ya, (__mmask8)(cm &  ml), ev2);
        ev1 = sb_add(ev1, yb, (__mmask8)(cm &  ml), ev1);
        ev3 = sb_add(ev3, yb, (__mmask8)(cm & ~ml), ev3);
        i += 8;
    }
    for(; i + 8 <= n3; i += 8){                       // seg3: e3 against yb
        SE4_CHUNK(cm)
        ev3 = sb_add(ev3, sb_perm64(rev, sb_load(a + (3*m - 9 - i))), cm, ev3);
    }
    #undef SE4_CHUNK
    if(i < n3){ // masked tail, q < 8 lanes; invalid lanes give g = p = 0
        const int64_t q = n3 - i;
        const __mmask8 vm = (__mmask8)((1u << q) - 1);
        const sb_vec t = sb_add(sb_load(b + i, vm), sb_load(b + m + i, vm));
        const uint64_t g = _mm512_cmpgt_epu64_mask(t, m52v);
        const uint64_t p = _mm512_cmpeq_epi64_mask(t, m52v);
        const uint64_t c = (((g << 1) | cy) + p) ^ p;
        cy = ((c >> q) | (c >> (q + 1))) & 1;
        sb_store(sb + i, sb_and(sb_add(t, onev, (__mmask8)c, t), m52v), vm);
        const __mmask8 cm = (__mmask8)(((c >> 1) & ~(1u << (q - 1))) | (cy << (q - 1)));
        ev3 = sb_add(ev3, sb_perm64(rev, sb_load(a + (n3 - i - 8), 0xFFu << (8 - q))), cm, ev3);
    }
    *e0 = (uint64_t)_mm512_reduce_add_epi64(ev0);
    *e1 = (uint64_t)_mm512_reduce_add_epi64(ev1);
    *e2 = (uint64_t)_mm512_reduce_add_epi64(ev2);
    *e3 = (uint64_t)_mm512_reduce_add_epi64(ev3);
}

// r[0..n) = u - v canonical (caller guarantees u >= v as integers, so the
// final borrow is 0); two error dots against the borrow bits (sub_err2_n).
static inline void u52_sub_err2_n(uint64_t *r, const uint64_t *u, const uint64_t *v,
                                  const uint64_t *y1, const uint64_t *y2,
                                  uint64_t *e1, uint64_t *e2, int64_t n){
    const sb_vec m52v = sb_set1_64(M52_), onev = sb_set1_64(1);
    const sb_vec rev = sb_setr_64(7,6,5,4,3,2,1,0);
    sb_vec ev1 = sb_zero(), ev2 = sb_zero();
    uint64_t by = 0;
    int64_t i = 0;
    for(; i + 8 <= n; i += 8){
        const sb_vec uv = sb_load(u + i), vv = sb_load(v + i);
        const sb_vec t = sb_sub(uv, vv);
        const uint64_t g = _mm512_cmplt_epu64_mask(uv, vv);
        const uint64_t p = _mm512_cmpeq_epi64_mask(uv, vv);
        const uint64_t c = (((g << 1) | by) + p) ^ p;
        by = (c >> 8) != 0;
        sb_store(r + i, sb_and(sb_sub(t, onev, (__mmask8)c, t), m52v));
        const __mmask8 cm = (__mmask8)(((c >> 1) & 0x7f) | (by << 7));
        ev1 = sb_add(ev1, sb_perm64(rev, sb_load(y1 + (n - 8 - i))), cm, ev1);
        ev2 = sb_add(ev2, sb_perm64(rev, sb_load(y2 + (n - 8 - i))), cm, ev2);
    }
    if(i < n){
        const int64_t q = n - i;
        const __mmask8 vm = (__mmask8)((1u << q) - 1);
        const sb_vec uv = sb_load(u + i, vm), vv = sb_load(v + i, vm);
        const sb_vec t = sb_sub(uv, vv);
        const uint64_t g = _mm512_cmplt_epu64_mask(uv, vv);
        const uint64_t p = _mm512_mask_cmpeq_epi64_mask(vm, uv, vv); // invalid lanes: 0==0 must NOT propagate
        const uint64_t c = (((g << 1) | by) + p) ^ p;
        by = ((c >> q) | (c >> (q + 1))) & 1;
        sb_store(r + i, sb_and(sb_sub(t, onev, (__mmask8)c, t), m52v), vm);
        const __mmask8 cm = (__mmask8)(((c >> 1) & ~(1u << (q - 1))) | (by << (q - 1)));
        const __mmask8 ym = (__mmask8)(0xFFu << (8 - q));
        ev1 = sb_add(ev1, sb_perm64(rev, sb_load(y1 + (n - i - 8), ym)), cm, ev1);
        ev2 = sb_add(ev2, sb_perm64(rev, sb_load(y2 + (n - i - 8), ym)), cm, ev2);
    }
    *e1 = (uint64_t)_mm512_reduce_add_epi64(ev1);
    *e2 = (uint64_t)_mm512_reduce_add_epi64(ev2);
    // final borrow == 0 guaranteed by the caller's compare
}

static inline int u52_cmp_n(const uint64_t *x, const uint64_t *y, int64_t n){
    for(int64_t i = n - 1; i >= 0; --i)
        if(x[i] != y[i]) return x[i] > y[i] ? 1 : -1;
    return 0;
}

// In-place signed rebalance of p[0..len-1] raw limbs; writes the overflow of
// the top into a NEW limb p[len]. Result: |p[i]| <= 2^52 + 2^11 for i < len,
// |p[len]| <= 2^11. Descending order so each step reads the original p[i-1].
static inline void u52_rebal_ext(uint64_t *p, int64_t len){
    p[len] = (uint64_t)((int64_t)p[len - 1] >> 52);
    for(int64_t i = len - 1; i > 0; --i)
        p[i] = (p[i] & M52_) + (uint64_t)((int64_t)p[i - 1] >> 52);
    p[0] &= M52_;
}

// Horizontal row strip, t in [1,7] rows in one output sweep:
//   r[s] += sum_j lo52(arows[j] * b[s + t-1-j]),  r[s+1] += hi52(same)
// for s in [0, len); touches r[0..len] (accumulating). arows = the stripped
// a-rows; the row splats are loop-invariant, so the sweep is madd-bound:
// 2t madds + t alignr per 8 outputs. b must be readable to len + t + 15.
static inline void u52_mulmid_rowt(sb_plimb r, const sb_limb *b, const sb_limb *arows, int64_t t, int64_t len){
    sb_vec av[7];
    for(int64_t k = 0; k < t; ++k) av[k] = sb_splat_load(arows, t - 1 - k); // av[k] pairs with b-shift k
    sb_vec hprev = sb_zero(), b0 = sb_load(b), b1;
    int64_t s = 0;
    for(; s + 8 <= len; s += 8){
        b1 = sb_load(b + s + 8);
        sb_vec rv = sb_load(r + s), hv = sb_zero();
        #define mm_rt(k) case k + 1: { \
            const sb_vec _bv = sb_alignr64(b1, b0, k); \
            rv = sb_madd52lo(rv, av[k], _bv); \
            hv = sb_madd52hi(hv, av[k], _bv); \
        }
        switch((int)t){ mm_rt(6) mm_rt(5) mm_rt(4) mm_rt(3) mm_rt(2) mm_rt(1) mm_rt(0) }
        #undef mm_rt
        sb_store(r + s, sb_add(rv, sb_alignr64(hv, hprev, 7)));
        hprev = hv;
        b0 = b1;
    }
    // unified epilogue, q in [0,8): lo madds masked to real outputs; lane q of
    // the hi alignr is hi52 of the last output's products -> r[len], so the
    // store widens by one lane and no scalar fixup is needed.
    const int64_t q = len - s;
    const __mmask8 lm = (__mmask8)((1u << q) - 1), sm = (__mmask8)((2u << q) - 1);
    b1 = sb_load(b + s + 8);
    sb_vec rv = sb_load(r + s, sm), hv = sb_zero();
    #define mm_rt(k) case k + 1: { \
        const sb_vec _bv = sb_alignr64(b1, b0, k); \
        rv = sb_madd52lo(rv, av[k], _bv, lm); \
        hv = sb_madd52hi(hv, av[k], _bv); \
    }
    switch((int)t){ mm_rt(6) mm_rt(5) mm_rt(4) mm_rt(3) mm_rt(2) mm_rt(1) mm_rt(0) }
    #undef mm_rt
    sb_store(r + s, sb_add(rv, sb_alignr64(hv, hprev, 7)), sm);
}

// Thin ACCUMULATING middle product, t in [1,7] diagonals in ONE a-sweep:
//   r[j] += lo-sums(diag an-1+j), r[j+1] += hi-sums(diag an-1+j), j < t
// i.e. r[0..t] += raw MP of (a, an) x (b, an+t-1). wtail's batch-vertical
// shape: reversed-a chunks x alignr'd b windows, 2 madds per diagonal per
// chunk, one unpack/sb_shufi64x2 reduction tree at the end (lane k = hsum of
// acc[k]). Requires an >= 8; b readable to an+16. This is the tail engine for
// near-balanced driver shapes (Newton's n x (2n+1)) and the odd-n peel.
static inline void u52_mulmid_vdiags(sb_plimb r, const sb_limb *a, const sb_limb *b, int64_t an, int64_t t){
    sb_vec acc[8];
    for(int k = 0; k < 8; ++k) acc[k] = sb_zero();
    const sb_vec rev = sb_setr_64(7,6,5,4,3,2,1,0);
    const sb_limb *bptr = b;
    sb_vec b0 = sb_load(bptr), b1;
    int64_t j = an - 8;
    // Large an: two a-chunks per iteration into SEPARATE acc sets. A single
    // set is latency-capped (each interior acc eats 2 chained madds per chunk
    // -> ~8c/chunk however small t is); alternation halves the chain density.
    // Measured: -15..-33% for t <= 5 at an >= 256, wash at t = 7, loses below
    // ~192 (zeroing/merge fixed cost) -- hence the gate. Both chunks live
    // under ONE switch so loop-unswitching on t survives.
    if(an >= 192){
        sb_vec acc2[8];
        for(int k = 0; k < 8; ++k) acc2[k] = sb_zero();
        sb_vec b2;
        #define mm_vd2(ind) case ind: { \
            const sb_vec _v = sb_alignr64(b1, b0, ind - 1); \
            acc[ind-1] = sb_madd52lo(acc[ind-1], _v, ax); \
            acc[ind]   = sb_madd52hi(acc[ind],   _v, ax); \
            const sb_vec _v2 = sb_alignr64(b2, b1, ind - 1); \
            acc2[ind-1] = sb_madd52lo(acc2[ind-1], _v2, ax2); \
            acc2[ind]   = sb_madd52hi(acc2[ind],   _v2, ax2); \
        }
        for(; j >= 8; j -= 16){
            b1 = sb_load(bptr + 8); b2 = sb_load(bptr + 16); bptr += 16;
            const sb_vec ax  = sb_perm64(rev, sb_load(a + j));
            const sb_vec ax2 = sb_perm64(rev, sb_load(a + j - 8));
            switch((int)t){
                mm_vd2(7)
                mm_vd2(6)
                mm_vd2(5)
                mm_vd2(4)
                mm_vd2(3)
                mm_vd2(2)
                mm_vd2(1)
            }
            b0 = b2;
        }
        #undef mm_vd2
        for(int k = 0; k <= (int)t; ++k) acc[k] = sb_add(acc[k], acc2[k]);
    }
    #define mm_vd(ind) case ind: { \
        const sb_vec _v = sb_alignr64(b1, b0, ind - 1); \
        acc[ind-1] = sb_madd52lo(acc[ind-1], _v, ax); \
        acc[ind]   = sb_madd52hi(acc[ind],   _v, ax); \
    }
    for(; j > -8; j -= 8){
        b1 = sb_load(bptr += 8);
        const sb_vec ax = sb_perm64(rev, j >= 0 ? sb_load(a + j)
                                           : sb_load(a + j, 0xFFu << (0 - j)));
        switch((int)t){
            mm_vd(7)
            mm_vd(6)
            mm_vd(5)
            mm_vd(4)
            mm_vd(3)
            mm_vd(2)
            mm_vd(1)
        }
        b0 = b1;
    }
    #undef mm_vd
    acc[0] = sb_add(sb_unpacklo64(acc[0], acc[1]), sb_unpackhi64(acc[0], acc[1]));
    acc[2] = sb_add(sb_unpacklo64(acc[2], acc[3]), sb_unpackhi64(acc[2], acc[3]));
    acc[4] = sb_add(sb_unpacklo64(acc[4], acc[5]), sb_unpackhi64(acc[4], acc[5]));
    acc[6] = sb_add(sb_unpacklo64(acc[6], acc[7]), sb_unpackhi64(acc[6], acc[7]));
    acc[0] = sb_add(sb_shufi64x2(acc[0], acc[2], 0x88), sb_shufi64x2(acc[0], acc[2], 0xDD));
    acc[4] = sb_add(sb_shufi64x2(acc[4], acc[6], 0x88), sb_shufi64x2(acc[4], acc[6], 0xDD));
    acc[0] = sb_add(sb_shufi64x2(acc[0], acc[4], 0x88), sb_shufi64x2(acc[0], acc[4], 0xDD));
    const __mmask8 wm = (__mmask8)((2u << t) - 1);
    sb_store(r, sb_add(sb_load(r, wm), acc[0]), wm);
}

static void mulmid_kara_n(sb_plimb r, const sb_limb *a, const sb_limb *b, int64_t n, scratch *sc);

// Sub-MP into a (m+2)-limb buffer: basecase leaves keep their raw form
// (thin-edge bound) with a zero extension limb; composite results arrive
// semi-canonical from the fused combine emit, so only the fat top limb needs
// splitting into the extension slot -- the old full rebalance pass is gone.
static inline void mulmid_kara_rec(sb_plimb p, const sb_limb *a, const sb_limb *b, int64_t m, scratch *sc){
    if(m < MULMID_KARA_THRESHOLD){
        mulmid_basecase(p, a, b, m, 2*m - 1);
        p[m + 1] = 0;
    }else{
        mulmid_kara_n(p, a, b, m, sc);
        p[m + 1] = (uint64_t)((int64_t)p[m] >> 52);
        p[m] &= (1ull << 52) - 1;
    }
}

// Balanced Karatsuba MP: an = n, bn = 2n-1, output n+1 raw signed limbs.
// Dispatches to mulmid_basecase below MULMID_KARA_THRESHOLD; odd n peels the
// top a-row + top diagonal (GMP's "odd row and diagonal") and recurses even.
static void mulmid_kara_n(sb_plimb r, const sb_limb *a, const sb_limb *b, int64_t n, scratch *sc){
    // threshold cap: leaf-level combine bound 4*(m+1)*2^52 < 2^63 needs leaf
    // m <= 500; n cap: the top raw limb carries the value overflow, bounded by
    // n*2^52 + interior slack (value argument: interior limbs are semi-small,
    // so the top must equal value-minus-small) -- needs n <= 1024
    assert(MULMID_KARA_THRESHOLD >= 8 && MULMID_KARA_THRESHOLD <= 500 && n <= 1024);
    if(n < MULMID_KARA_THRESHOLD){
        mulmid_basecase(r, a, b, n, 2*n - 1);
        return;
    }
    // Residue strip: t rows go to one full-throughput horizontal band plus t
    // extra diagonals over the core, whose rows [0, nc) x all diags form a
    // balanced kara(nc) at b+t (+ vdiags for the top t diagonals). t = 1 is
    // the classic odd-n row+diagonal peel; rounding n down to a multiple of 8
    // additionally keeps every level below even, avoiding the 3x-per-level
    // peel cascade. Interleaved A/B says the mod-8 strip only pays in the
    // deep band at high residues (n >= 4*thresh, n%8 >= 5: -2..-3.5%); it is
    // a wash-to-slight-loss elsewhere, so it gates exactly there. (Newton
    // precisions are not limb-aligned, so callers cannot choose nice n.)
    {
        const int64_t t = (n >= 4*MULMID_KARA_THRESHOLD && (n & 7) >= 5) ? n & 7
                                                                         : n & 1;
        if(t){
            const int64_t nc = n - t;
            mulmid_kara_n(r, a, b + t, nc, sc);
            for(int64_t i = 1; i <= t; ++i) r[nc + i] = 0;
            u52_mulmid_vdiags(r + nc, a, b + t + nc, nc, t);
            u52_mulmid_rowt(r, b, a + nc, t, n);
            return;
        }
    }
    const int64_t m = n >> 1;
    SCRATCH(sc);
    // one block: sb, t, p0, p1, p2 -- every region on an 8-limb grid so the
    // sub stores, p1's leaf a-loads (= t) and the combine's p-loads don't
    // split cache lines (for m = 0 mod 8; odd shapes degrade gracefully).
    // sb region = 3m-1 data + 32 basecase b-over-read pad.
    const int64_t st_sb = (3*m + 38) & ~7, st_t = (m + 15) & ~7, st_p = (m + 9) & ~7;
    uint64_t *sb = SALLOC(sc, uint64_t, st_sb + st_t + 3*st_p);
    uint64_t *t  = sb + st_sb;
    uint64_t *p0 = t + st_t;
    uint64_t *p1 = p0 + st_p;
    uint64_t *p2 = p1 + st_p;
    uint64_t e0, e1, e2, e3, e4, e5;

    // s = {b,3m-1} + {b+m,3m-1}; one fused aligned sweep, four error dots
    u52_s_err4(sb, b, a, m, &e0, &e1, &e2, &e3);

    // t = |a_hi - a_lo|; borrow dots against b windows. sgn=0: t = a_hi - a_lo.
    int sgn = 0;
    const int c0 = u52_cmp_n(a + m, a, m);
    if(c0 == 0){ // equal halves: t = 0, P1 = 0, no borrows -- skip the sub-MP
        for(int64_t i = 0; i <= m + 1; ++i) p1[i] = 0;
        e4 = e5 = 0;
    }else if(c0 > 0)
        u52_sub_err2_n(t, a + m, a, b + m - 1, b + 2*m - 1, &e4, &e5, m);
    else
        u52_sub_err2_n(t, a, a + m, b + m - 1, b + 2*m - 1, &e4, &e5, m), sgn = 1;

    mulmid_kara_rec(p0, a + m, sb,     m, sc);
    if(c0) mulmid_kara_rec(p1, t, b + m, m, sc);
    mulmid_kara_rec(p2, a,     sb + m, m, sc);

    // out = (P~0 + tau P~1) + B^m (P~2 - tau P~1); tau = -1 when sgn == 0.
    // FUSED combine + semi-canonical emit: one left-to-right masked-vector
    // pass computes raw[i] = (p0 -+ p1 for i <= m+1) + (p2 +- p1 for i >= m)
    // (<= 2 boundary vectors carry both regions under masks) and emits
    // r[i] = (raw & M52) + (raw[i-1] >>a 52) with the lag riding in registers
    // via one alignr -- this replaces the old two combine loops PLUS the
    // caller's whole rebalance pass. Extension limbs and the final lag fold
    // into the fat top limb r[2m].
    //
    // The error terms collapse onto the OUTPUT (so no p-array RMW ahead of
    // the vector loads -- that was a store-forwarding hazard):
    //   corr = (-e0 + tau e4) + B^m (e1 - tau e5 - e2 - tau e4)
    //        + B^2m (e3 + tau e5)
    // applied as signed split adds after the emit; limbs {0,1,m,m+1} may
    // exceed 2^52 by the dot bounds, all audited.
    {
        const sb_vec m52v = sb_set1_64(M52_);
        sb_vec sprev = sb_zero();
        int64_t i = 0;
        #define KEMIT(raw_expr) { \
            const sb_vec _raw = (raw_expr); \
            const sb_vec _sra = sb_srai(_raw, 52); \
            sb_store(r + i, sb_add(sb_and(_raw, m52v), sb_alignr64(_sra, sprev, 7))); \
            sprev = _sra; \
        }
        #define KCOMBINE(OPA, OPB, OPBS) \
            for(; i + 8 <= m; i += 8) /* pure low half */ \
                KEMIT(OPA(sb_load(p0 + i), sb_load(p1 + i))); \
            for(; i + 8 <= 2*m && i <= m + 1; i += 8){ /* <= 2 boundary vectors */ \
                const int64_t na = m + 2 - i; \
                const __mmask8 ma = (__mmask8)(na >= 8 ? 0xFFu : (1u << na) - 1); \
                const __mmask8 mb = (__mmask8)(i >= m ? 0xFFu : 0xFFu << (m - i)); \
                KEMIT(sb_add(OPA(sb_load(p0 + i, ma), sb_load(p1 + i, ma)), \
                          OPB(sb_load(p2 + (i - m), mb), sb_load(p1 + (i - m), mb)))); \
            } \
            for(; i + 8 <= 2*m; i += 8) /* pure high half */ \
                KEMIT(OPB(sb_load(p2 + (i - m)), sb_load(p1 + (i - m)))); \
            { /* masked tail lanes [i, 2m) + the fat top limb 2m */ \
                const int64_t q = 2*m - i; \
                uint64_t toplag; \
                if(q){ \
                    const __mmask8 lm = (__mmask8)((1u << q) - 1); \
                    const sb_vec _raw = OPB(sb_load(p2 + (i - m), lm), sb_load(p1 + (i - m), lm)); \
                    const sb_vec _sra = sb_srai(_raw, 52); \
                    sb_store(r + i, sb_add(sb_and(_raw, m52v), sb_alignr64(_sra, sprev, 7)), lm); \
                    toplag = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(sb_perm64(sb_set1_64(q - 1), _sra))); \
                }else \
                    toplag = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(sb_perm64(sb_set1_64(7), sprev))); \
                r[2*m] = OPBS(p2[m], p1[m]) + (OPBS(p2[m+1], p1[m+1]) << 52) + toplag; \
            }
        #define KADD(x, y) sb_add(x, y)
        #define KSUB(x, y) sb_sub(x, y)
        #define KADDS(x, y) ((x) + (y))
        #define KSUBS(x, y) ((x) - (y))
        uint64_t x0, xm, x2;
        if(sgn == 0){
            KCOMBINE(KSUB, KADD, KADDS)
            x0 = 0 - e0 - e4; xm = e1 + e5 - e2 + e4; x2 = e3 - e5;
        }else{
            KCOMBINE(KADD, KSUB, KSUBS)
            x0 = 0 - e0 + e4; xm = e1 - e5 - e2 - e4; x2 = e3 + e5;
        }
        #undef KADD
        #undef KSUB
        #undef KADDS
        #undef KSUBS
        #undef KCOMBINE
        #undef KEMIT
        r[0]   += x0 & M52_;  r[1]   += (uint64_t)((int64_t)x0 >> 52);
        r[m]   += xm & M52_;  r[m+1] += (uint64_t)((int64_t)xm >> 52);
        r[2*m] += x2;
    }
}

// General middle product driver (mirrors GMP mpn_mulmid): balanced Karatsuba
// blocks of an diagonals marching along b, joined at 1-limb raw seams; the
// <= 7-diagonal remainder goes to the batch vertical kernel (this is the
// whole tail for Newton's an x (2an+1) shapes), wider remainders to basecase.
// Thin regions (rn < an) and small an fall through to basecase whole.
// Output: rn+1 raw signed limbs, same value contract as mulmid_kara_n.
static inline void mulmid_dc(sb_plimb r, const sb_limb *a, const sb_limb *b, int64_t an, int64_t bn, scratch *sc){
    const int64_t rn = bn - an + 1;
    if(an < MULMID_KARA_THRESHOLD || rn < an){
        mulmid_basecase(r, a, b, an, bn);
        return;
    }
    mulmid_kara_n(r, a, b, an, sc);
    int64_t done = an;
    while(rn - done >= an){
        const uint64_t seam = r[done];
        mulmid_kara_n(r + done, a, b + done, an, sc);
        r[done] += seam;
        done += an;
    }
    const int64_t d = rn - done;
    if(d > 0){
        if(d <= 7){
            for(int64_t i = 1; i <= d; ++i) r[done + i] = 0;
            u52_mulmid_vdiags(r + done, a, b + done, an, d); // accumulates over the seam
        }else{
            const uint64_t seam = r[done];
            mulmid_basecase(r + done, a, b + done, an, an + d - 1);
            r[done] += seam;
        }
    }
}

#undef M52_

}
