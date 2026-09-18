/* Imported from libsbn/include/sbn/detail/u52/mul.h; arithmetic preserved, private namespace and prepared Frame adapter. */
#pragma once
namespace sbn::v3::u52 {







inline uint64_t u52_vec_count(uint64_t n) {
    return (n + 7) >> 3;
}

inline sb_pvec u52_pvec_add(sb_pvec p, uint64_t limbs) {
    return (sb_pvec)((uint64_t *)(void *)p + limbs);
}

inline sb_cpvec u52_cpvec_add(sb_cpvec p, uint64_t limbs) {
    return (sb_cpvec)((const uint64_t *)(const void *)p + limbs);
}

// Dispatch thresholds. A tune build makes them mutable globals so the tuner
// can re-measure code that recurses through them (GMP TUNE_PROGRAM_BUILD).


// n0 : output accumulator (persistent across the chunk)
// n1 : single rotating diagonal reg; incoming offset ind+1, then reborn as offset ind
#define mul1_substep(a, ind) {                              \
    n1 = sb_madd52hi(n1, (a), bx[ind]);                        \
    n0 = sb_add(n0, sb_alignr64(n1, m[ind], 7-(ind)));            \
    m[ind] = n1;                                            \
    n1 = sb_madd52lo(sb_zero(), (a), bx[ind]);                    \
}
#define mul1_substep0(a) {                                  \
    n1  = sb_madd52hi(n1, (a), bx[0]);                         \
    n0  = sb_add(n0, sb_alignr64(n1, m[0], 7));                   \
    m[0] = n1;                                              \
    lo0 = sb_madd52lo(sb_zero(), (a), bx[0]);                     \
}

inline static void mul_u52_bn1(sb_pvec r, sb_cpvec a, uint64_t an, sb_vec b0){
    if(!an) return;
    sb_vec carry = sb_zero();
    uint64_t c = an >> 3;
    const uint64_t tail = an & 7;
#pragma GCC unroll 4
    for(; c; --c){
        const sb_vec ax = sb_load(a++);
        sb_vec hi = sb_madd52hi(sb_zero(), ax, b0);
        sb_store(r++, sb_add(sb_madd52lo(sb_zero(), ax, b0), sb_alignr64(hi, carry, 7)));
        carry = hi;
    }
    if(tail){
        const sb_vec ax = sb_load(a, u52_limb_mask(tail));
        const sb_vec hi = sb_madd52hi(sb_zero(), ax, b0);
        const sb_vec lo = sb_madd52lo(sb_zero(), ax, b0);
        sb_store(r, sb_add(lo, sb_alignr64(hi, carry, 7)), u52_limb_mask(tail + 1));
    }else{
        sb_store(r, sb_alignr64(sb_zero(), carry, 7), u52_limb_mask(1));
    }
}

inline static void mul_u52_bn2(sb_pvec r, sb_cpvec a, uint64_t an, sb_vec b0, sb_vec b1){
    if(!an) return;
    sb_vec pMid = sb_zero(), pH1 = sb_zero();
    uint64_t c = an >> 3;
    const uint64_t tail = an & 7;

    #define MUL2_CHUNK(ax, dst) do {                       \
        sb_vec L0 = sb_madd52lo(sb_zero(), (ax), b0);              \
        sb_vec H0 = sb_madd52hi(sb_zero(), (ax), b0);              \
        sb_vec L1 = sb_madd52lo(sb_zero(), (ax), b1);              \
        sb_vec H1 = sb_madd52hi(sb_zero(), (ax), b1);              \
        sb_vec mid = sb_add(H0, L1);                            \
        (dst) = sb_add(L0, sb_add(sb_alignr64(mid, pMid, 7),        \
                            sb_alignr64(H1,  pH1,  6)));      \
        pMid = mid; pH1 = H1;                              \
    } while(0)

#pragma GCC unroll 2
    for(; c; --c){
        const sb_vec ax = sb_load(a++);
        sb_vec o;
        MUL2_CHUNK(ax, o);
        sb_store(r++, o);
    }
    if(tail){
        const sb_vec ax = sb_load(a, u52_limb_mask(tail));
        sb_vec o;
        const uint64_t tail_limbs = tail + 2;
        MUL2_CHUNK(ax, o);
        if(tail_limbs <= 8){
            sb_store(r, o, u52_limb_mask(tail_limbs));
        }else{
            sb_store(r++, o);
            sb_store(r, sb_add(sb_alignr64(sb_zero(), pMid, 7),
                             sb_alignr64(sb_zero(), pH1,  6)), u52_limb_mask(tail_limbs - 8));
        }
    }else{
        sb_store(r, sb_add(sb_alignr64(sb_zero(), pMid, 7),
                         sb_alignr64(sb_zero(), pH1,  6)), u52_limb_mask(2));
    }
    #undef MUL2_CHUNK
}

inline static void mul_u52_1(sb_pvec r, sb_cpvec a, sb_cpvec b, uint64_t an, const uint64_t bn){
    if(!an) return;
    if(bn == 1) return mul_u52_bn1(r, a, an, sb_splat_load(b, 0));
    else if(bn == 2) return mul_u52_bn2(r, a, an, sb_splat_load(b, 0), sb_splat_load(b, 1));
    sb_vec n0, n1, lo0, ax, m[8], bx[8];
    m[0]=m[1]=m[2]=m[3]=m[4]=m[5]=m[6]=m[7]=sb_zero();
    if(bn > 4){
        bx[7] = sb_splat_load(b, 7);
        bx[6] = sb_splat_load(b, 6);
        bx[5] = sb_splat_load(b, 5);
        bx[4] = sb_splat_load(b, 4);
    }
    bx[3] = sb_splat_load(b, 3);
    bx[2] = sb_splat_load(b, 2);
    bx[1] = sb_splat_load(b, 1);
    bx[0] = sb_splat_load(b, 0);
    const uint64_t tail = an & 7;
    for(an = an >> 3; an; --an){
        ax = sb_load(a++);
        n0 = sb_zero();
        n1 = sb_zero();
        switch(bn){
            case 8: mul1_substep (ax, 7);
            case 7: mul1_substep (ax, 6);
            case 6: mul1_substep (ax, 5);
            case 5: mul1_substep (ax, 4);
            case 4: mul1_substep (ax, 3);
            case 3: mul1_substep (ax, 2);
            case 2: mul1_substep (ax, 1);
            case 1: mul1_substep0(ax);
        }
        n0 = sb_add(n0, lo0);
        sb_store(r++, n0);
    }
    if(tail){
        ax = sb_load(a, u52_limb_mask(tail));
        n0 = sb_zero();
        n1 = sb_zero();
        switch(bn){
            case 8: mul1_substep (ax, 7);
            case 7: mul1_substep (ax, 6);
            case 6: mul1_substep (ax, 5);
            case 5: mul1_substep (ax, 4);
            case 4: mul1_substep (ax, 3);
            case 3: mul1_substep (ax, 2);
            case 2: mul1_substep (ax, 1);
            case 1: mul1_substep0(ax);
        }
        n0 = sb_add(n0, lo0);
        const uint64_t tail_limbs = tail + bn;
        if(tail_limbs <= 8) sb_store(r, n0, u52_limb_mask(tail_limbs));
        else sb_store(r++, n0);
    }
    n0 = sb_zero(); n1 = sb_zero();
    switch(bn){
        case 8: n0 = sb_add(n0, sb_alignr64(sb_zero(), m[7], 0));
        case 7: n1 = sb_add(n1, sb_alignr64(sb_zero(), m[6], 1));
        case 6: n0 = sb_add(n0, sb_alignr64(sb_zero(), m[5], 2));
        case 5: n1 = sb_add(n1, sb_alignr64(sb_zero(), m[4], 3));
        case 4: n0 = sb_add(n0, sb_alignr64(sb_zero(), m[3], 4));
        case 3: n1 = sb_add(n1, sb_alignr64(sb_zero(), m[2], 5));
        case 2: n0 = sb_add(n0, sb_alignr64(sb_zero(), m[1], 6));
        case 1: n1 = sb_add(n1, sb_alignr64(sb_zero(), m[0], 7));
    }
    n0 = sb_add(n0, n1);
    if(tail){
        const uint64_t tail_limbs = tail + bn;
        if(tail_limbs > 8) sb_store(r, n0, u52_limb_mask(tail_limbs - 8));
    }else{
        sb_store(r, n0, u52_limb_mask(bn));
    }
}

#define mul_accum(a, b, ind) {\
    bx = sb_splat_load(b, ind); \
    n[ind] = sb_madd52lo(n[ind], a, bx); \
    n[ind+1] = sb_madd52hi(n[ind+1], a, bx); \
}

#define apply_shift_n(ind) { \
    n[0]   = sb_add(n[0], sb_alignr64(n[ind], m[ind], 8 - (ind))); \
    m[ind] = n[ind]; \
    n[ind] = sb_zero(); \
}

#define apply_shift_z(ind) { \
    n[0]   = sb_add(n[0], sb_alignr64(n[ind], sb_zero(), 8 - (ind))); \
    m[ind] = n[ind]; \
    n[ind] = sb_zero(); \
}

#define harvest { \
    n[0] = sb_add(n[0], m[8]); m[8] = n[8]; n[8] = sb_zero(); \
    apply_shift_n(7); apply_shift_n(6); apply_shift_n(5); apply_shift_n(4); \
    apply_shift_n(3); apply_shift_n(2); apply_shift_n(1); \
}
#define harvest_z { \
    m[8]=n[8]; n[8] = sb_zero(); \
    apply_shift_z(7); apply_shift_z(6); apply_shift_z(5); apply_shift_z(4); \
    apply_shift_z(3); apply_shift_z(2); apply_shift_z(1); \
}

#define add_flush_tail { \
    n[0] = sb_add(n[0], m[8]); \
    n[0] = sb_add(n[0], sb_alignr64(sb_zero(), m[7], 1)); \
    n[0] = sb_add(n[0], sb_alignr64(sb_zero(), m[6], 2)); \
    n[0] = sb_add(n[0], sb_alignr64(sb_zero(), m[5], 3)); \
    n[0] = sb_add(n[0], sb_alignr64(sb_zero(), m[4], 4)); \
    n[0] = sb_add(n[0], sb_alignr64(sb_zero(), m[3], 5)); \
    n[0] = sb_add(n[0], sb_alignr64(sb_zero(), m[2], 6)); \
    n[0] = sb_add(n[0], sb_alignr64(sb_zero(), m[1], 7)); \
}

#define apply_shift_n_p(ind) { \
    n[ind] = sb_lane_rotl(n[ind], ind);\
    const __mmask8 imask = (1 << ind) - 1;\
    n[0] = sb_add(n[0], n[ind], ~imask, n[0]); \
    n[8] = sb_add(n[8], n[ind], imask, n[8]); \
}

inline static void mul_u52_basecase(sb_pvec r, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn){
    assert(an >= bn);
    if(bn <= 8) return mul_u52_1(r, a, b, an, bn);
    sb_vec n[9], m[9], bx, ax;
    sb_cpvec bp, ap;
    n[0]=sb_zero(); n[1]=sb_zero(); n[2]=sb_zero(); n[3]=sb_zero(); n[4]=sb_zero();
    n[5]=sb_zero(); n[6]=sb_zero(); n[7]=sb_zero(); n[8]=sb_zero();
    --an, --bn;
    const uint64_t au = an >> 3, bu = bn >> 3, su = au + bu;
    an &= 7, bn &= 7;
    uint64_t i, j;
    for(i = 0; i < su; ++i){
        if(i >= bu){
            bp = b + bu; ap = a + (i - bu);
            j = i >= au ? su - i : bu + 1;
            ax = sb_load(ap);
            switch(bn){
                non_sat:
                for(; j; ++ap, --bp, --j){
                    ax = sb_load(ap);
                    case 7: mul_accum(ax, bp, 7);
                    case 6: mul_accum(ax, bp, 6);
                    case 5: mul_accum(ax, bp, 5);
                    case 4: mul_accum(ax, bp, 4);
                    case 3: mul_accum(ax, bp, 3);
                    case 2: mul_accum(ax, bp, 2);
                    case 1: mul_accum(ax, bp, 1);
                    case 0: mul_accum(ax, bp, 0);
                }
            }
            if(i >= au){
                ax = sb_load(bp);
                switch(an){
                    case 7: mul_accum(ax, ap, 7);
                    case 6: mul_accum(ax, ap, 6);
                    case 5: mul_accum(ax, ap, 5);
                    case 4: mul_accum(ax, ap, 4);
                    case 3: mul_accum(ax, ap, 3);
                    case 2: mul_accum(ax, ap, 2);
                    case 1: mul_accum(ax, ap, 1);
                    case 0: mul_accum(ax, ap, 0);
                }
            }
        }else{
            j = i + 1; bp = b + i; ap = a;
            ax = sb_load(a);
            goto non_sat;
        }
        if(i) harvest else harvest_z;
        sb_store(r, n[0]); ++r;
        n[0] = sb_zero();
    }
    const uint64_t an_tail = an, bn_tail = bn;
    const uint64_t tail_limbs = an_tail + bn_tail + 2;
    if(an_tail < bn_tail){
        bn = an_tail;
        bp = a + au;
        ax = sb_load(b + bu, u52_limb_mask(bn_tail + 1));
    }else{
        bp = b + bu;
        ax = sb_load(a + au, u52_limb_mask(an_tail + 1));
    }
    switch(bn){
        case 7: mul_accum(ax, bp, 7);
        case 6: mul_accum(ax, bp, 6); apply_shift_n_p(7);
        case 5: mul_accum(ax, bp, 5); apply_shift_n_p(6);
        case 4: mul_accum(ax, bp, 4); apply_shift_n_p(5);
        case 3: mul_accum(ax, bp, 3); apply_shift_n_p(4);
        case 2: mul_accum(ax, bp, 2); apply_shift_n_p(3);
        case 1: mul_accum(ax, bp, 1); apply_shift_n_p(2);
        case 0: mul_accum(ax, bp, 0); apply_shift_n_p(1);
    }
    add_flush_tail;
    if(tail_limbs <= 8){
        sb_store(r, n[0], u52_limb_mask(tail_limbs));
    }else{
        sb_store(r, n[0]);
        sb_store(r+1, n[8], u52_limb_mask(tail_limbs - 8));
    }
}

// Returns the output class: 0 = canonical or non-negative redundant within
// the basecase lane bound L0 (safe toom-point / fold-free input); 1 = nc
// composite, lanes signed-redundant up to 3*L0 (a consumer must fold).
static int mul_u52_dispatch(sb_pvec r, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn, scratch *s);
// dispatch + force class 0: canonicalizes a class-1 result in place
static inline int mul_u52_dispatch_canon(sb_pvec r, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn, scratch *s){
    if(mul_u52_dispatch(r, a, b, an, bn, s))
        u52_canon_neg_tail(r, (sb_cpvec)r, an + bn);
    return 0;
}
// contract: caller guarantees a is larger

static inline int mul_u52_karatsuba(sb_pvec r, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn, scratch *s) {
    assert(an >= bn); // contract: caller guarantees a is larger
    const uint64_t split_limbs = an >> 1;
    if(!split_limbs || bn <= split_limbs){
        mul_u52_basecase(r, a, b, an, bn);
        return 0;
    }
    const uint64_t a1_len = an - split_limbs;
    const uint64_t b1_len = bn - split_limbs;
    const uint64_t eval_limbs = a1_len > split_limbs ? a1_len : split_limbs;
    sb_cpvec a1 = u52_cpvec_add(a, split_limbs);
    sb_cpvec b1 = u52_cpvec_add(b, split_limbs);
    SCRATCH(s);
    sb_pvec point_n1  = SALLOC(s, sb_vec, u52_vec_count(eval_limbs * 2));
    sb_pvec point_0   = r;
    sb_pvec point_mid = u52_pvec_add(r, split_limbs);
    sb_pvec point_inf = u52_pvec_add(r, split_limbs * 2);

    sb_pvec a_n1 = point_0, b_n1 = u52_pvec_add(point_0, eval_limbs);
    // reuse point_0 space for n=-1 operands; this is before eval of point_0, so no overwrite risk
    uint8_t flag = 0;
    int64_t sa_len = u52_cmp_and_sub(a_n1, a, a1, split_limbs, a1_len);
    if(sa_len < 0){
        sa_len = -sa_len;
        flag = 1;
    }
    int64_t sb_len = u52_cmp_and_sub(b_n1, b, b1, split_limbs, b1_len);
    if(sb_len < 0){
        sb_len = -sb_len;
        flag ^= 1;
    }
    const uint64_t n1_len = (sa_len && sb_len) ? (uint64_t)sa_len + (uint64_t)sb_len : 0;
    int cls = 0;
    if(n1_len){
        cls |= mul_u52_dispatch(point_n1, a_n1, b_n1, (uint64_t)sa_len, (uint64_t)sb_len, s);
    }
    const int cls_p0 = mul_u52_dispatch(point_0, a, b, split_limbs, split_limbs, s);
    cls |= cls_p0;
    cls |= mul_u52_dispatch(point_inf, a1, b1, a1_len, b1_len, s);
    // point_n1 = point_0 + point_inf - mid
    // which means that mid = point_0 + point_inf - point_n1
    // note that mid is guaranteed >= 0, so the subtraction doesn't need compare
    add_nc_52(point_inf, point_mid, point_inf, split_limbs);
    add_nc_52(point_mid, point_inf, point_0, split_limbs);
    const uint64_t vinf_len = a1_len + b1_len;
    if(vinf_len > split_limbs){
        add_nc_52(point_inf, point_inf, u52_pvec_add(point_inf, split_limbs), vinf_len - split_limbs);
    }
    // Alternating contract: fold only when a child is class-1 (its lanes can
    // reach 3*L0, so one more nc level would breach the budget); class-0
    // children allow pure nc recomposition, returning class 1. Either way the
    // fold, when needed, is the final operand application over [split, top)
    // -- never a standalone pass. Budget: fold internals <= 9*L0, which the
    // tuner's T22 <= 112 clamp keeps under 2^63.
    if(cls){
        if(n1_len && !flag)
            u52_fold_sub_canon(point_mid, (sb_cpvec)point_n1, n1_len,
                                   an + bn - split_limbs);
        else
            u52_fold_add_canon(point_mid, (sb_cpvec)point_n1, n1_len,
                                   an + bn - split_limbs);
        // The fold canonicalizes [split, top) only; [0, split) keeps point_0's
        // lanes, so the OUTPUT class is point_0's class. (At >= 3 composite
        // levels point_0 is class-1: returning 0 here once sent signed lanes
        // into the positive-chain boundary encode -- caught by the GMP
        // differential at 532x426 u64 limbs.)
        return cls_p0;
    }
    if(n1_len){
        if(flag) add_nc_52(point_mid, point_mid, point_n1, n1_len);
        else     sub_nc_52(point_mid, point_mid, point_n1, n1_len);
    }
    return 1;
}

// mul_u52_dispatch: shape-aware band/ratio dispatcher, defined at end of file.

// Toom-32 for bn < an < 3bn (an ~ 1.5*bn), GMP points {0, +-1, inf}. Output
// contract matches karatsuba: redundant signed lanes, caller canonicalizes
// (u52_canon_neg). The vm1 sign rides a flag; magnitudes from u52_cmp_and_sub.
// GMP's per-limb carry threading (cy/hi/INCR/DECR) disappears: redundant lanes
// absorb carries and the final canonicalization resolves them.
//
// Memory follows GMP's order: ap1/bp1/am1 live in pp and are consumed by the
// v1/vm1 products; vm1's product then lands at pp[0,2n+2) and is consumed by
// the halving and y steps before v0's product overwrites it. Only bm1 and v1
// need scratch (~3n limbs vs ~8n for the naive layout). Requires s + t > n
// (B^4 region for y2's top limb) and s + t >= 3 (am1 fits behind vm1).
static inline int mul_u52_toom32(sb_pvec pp, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn, scratch *sc){
    const uint64_t n = (2*an >= 3*bn) ? (an + 2) / 3 : ((bn + 1) >> 1);
    const uint64_t s = an - 2*n, t = bn - n;
    assert(0 < s && s <= n && 0 < t && t <= n && s + t > n && s + t >= 3);
    sb_cpvec a0 = a, a1 = u52_cpvec_add(a, n), a2 = u52_cpvec_add(a, 2*n);
    sb_cpvec b0 = b, b1 = u52_cpvec_add(b, n);
    SCRATCH(sc);
    const uint64_t pvcs = u52_vec_count(2*n + 2);
    sb_pvec ap1 = pp;                                  // n+1 limbs, dead after v1
    sb_pvec bp1 = u52_pvec_add(pp, n + 1);             // n+1 limbs, dead after v1
    sb_pvec am1 = u52_pvec_add(pp, 2*n + 2);           // n+1 limbs, dead after vm1
    sb_pvec bm1 = SALLOC(sc, sb_vec, u52_vec_count(n));
    sb_pvec v1  = SALLOC(sc, sb_vec, pvcs);
    sb_pvec vm1 = pp;                                  // 2n+2 limbs once ap1/bp1 die

    // evaluations (canonical operands for the recursive muls)
    u52_add_carry(am1, a0, a2, n, s, true);                       // t02 = a0+a2, n+1 limbs
    u52_add_carry(ap1, (sb_cpvec)am1, a1, n + 1, n, false);          // as1 = t02+a1 < 3B^n, n+1 limbs
    int64_t sa = u52_cmp_and_sub(am1, (sb_cpvec)am1, a1, n + 1, n);  // asm1 = |t02-a1| in place
    u52_add_carry(bp1, b0, b1, n, t, true);                       // bs1, n+1 limbs
    int64_t sb = u52_cmp_and_sub(bm1, b0, b1, n, t);              // bsm1 = |b0-b1|
    const uint8_t vm1_neg = (sa < 0) ^ (sb < 0);
    const uint64_t lam = (uint64_t)(sa < 0 ? -sa : sa);
    const uint64_t lbm = (uint64_t)(sb < 0 ? -sb : sb);

    // v1 = as1*bs1 consumes ap1/bp1; vm1 = asm1*bsm1 then reuses their space
    // (its 2n+2-limb output cannot reach am1 at pp+2n+2)
    sb_store(v1 + pvcs - 1, sb_zero());
    // toom points must be class-0: the interpolation stacks several point
    // magnitudes, which only fits the lane budget for class-0 inputs
    mul_u52_dispatch_canon(v1, (sb_cpvec)ap1, (sb_cpvec)bp1, n + 1, n + 1, sc);  // 2n+2 limbs
    if(lam && lbm) mul_u52_dispatch_canon(vm1, (sb_cpvec)am1, (sb_cpvec)bm1, lam, lbm, sc);
    // zero vm1's tail through the vector boundary for the halving pass (also
    // the whole region when vm1 == 0); am1 is dead, stomping it is fine
    const uint64_t lvm = (lam && lbm) ? lam + lbm : 0;
    memset((uint64_t *)(void *)vm1 + lvm, 0, (pvcs * 8 - lvm) * 8);

    // v1 <- (v1 + vm1_signed)/2 = x0 + x2 =: W, redundant halving (no carry
    // resolution needed); the stored vm1 magnitude is subtracted when negative
    u52_addsub_rsh1_nc(v1, (sb_cpvec)v1, (sb_cpvec)vm1, 2*n + 2, vm1_neg);

    // y = W*(B+1) - vm1(signed) = y0 + y1*B + y2*B^2 (+ w2n*B^3 in y2's top):
    // y0 = W_lo (in v1), y1 = W_lo + W_hi (built in pp[2n,3n)), y2 = W_hi + w2n.
    // vm1[2n] aliases y1's first lane, so the one-limb ops go first.
    sb_pvec p1 = u52_pvec_add(pp, n), p2 = u52_pvec_add(pp, 2*n);
    sb_pvec p3 = u52_pvec_add(pp, 3*n), p4 = u52_pvec_add(pp, 4*n);
    sb_pvec v1n = u52_pvec_add(v1, n), v1_2n = u52_pvec_add(v1, 2*n);
    sb_cpvec vm1n = u52_cpvec_add((sb_cpvec)vm1, n);
    // y1 must read W_hi pristine, the y2-lane0 fix needs vm1[2n], and the y1
    // build overwrites pp[2n] == vm1[2n]: snapshot the limb, build y1, then
    // apply the lane fix as scalar arithmetic on the redundant lane
    const uint64_t vm1top = ((const uint64_t *)(const void *)vm1)[2*n];
    add_nc_52(p2, (sb_cpvec)v1, (sb_cpvec)v1n, n);                 // y1 = W_lo + W_hi
    uint64_t *v1l = (uint64_t *)(void *)v1;
    v1l[n] += v1l[2*n] + (vm1_neg ? vm1top : 0ull - vm1top); // y2 lane0 += w2n -+ vm1 top
    v1l[n + 1] += v1l[2*n + 1];   // W is redundant now: C2's second lane is live too
    if(vm1_neg){
        add_nc_52(v1, (sb_cpvec)v1, (sb_cpvec)vm1, n);             // y0 -+ vm1 lo
        add_nc_52(p2, (sb_cpvec)p2, vm1n, n);                   // y1 -+ vm1 mid
    }else{
        sub_nc_52(v1, (sb_cpvec)v1, (sb_cpvec)vm1, n);
        sub_nc_52(p2, (sb_cpvec)p2, vm1n, n);
    }

    // vm1 fully consumed: v0 may now overwrite it; x3 overwrites dead am1
    mul_u52_dispatch_canon(pp, a0, b0, n, n, sc);                      // x0
    mul_u52_dispatch_canon(p3, a2, b1, s, t, sc);                      // x3

    // result = Lx0 + (y0 + Hx0 - Lx3)B + (y1 - Lx0 - Hx3)B^2
    //        + (y2 - (Hx0 - Lx3))B^3 + (Hx3 + w2n)B^4
    sub_nc_52(p1, (sb_cpvec)p1, (sb_cpvec)p3, n);              // Hx0 - Lx3
    sub_nc_52(p2, (sb_cpvec)p2, (sb_cpvec)pp, n);              // y1 - Lx0
    sub_nc_52(p2, (sb_cpvec)p2, (sb_cpvec)p4, s + t - n);      //    - Hx3
    sub_nc_52(p3, (sb_cpvec)v1n, (sb_cpvec)p1, n);             // y2 - (Hx0 - Lx3)
    add_nc_52(p4, (sb_cpvec)p4, (sb_cpvec)v1_2n, 2);           // + C2 (redundant, 2 lanes)
    // final op doubles as the canonneg fold over the whole mixed span
    // [n, an+bn): applies y0 and resolves every wrapped lane; [0,n) keeps
    // v0's class-0 child lanes
    u52_fold_add_canon(p1, (sb_cpvec)v1, n, an + bn - n);
    return 0;
}


// Shared 5-point interpolation (toom33/toom42), GMP toom_interpolate_5pts
// order over redundant points. Layout contract: v0 = pp[0,2n); v1 occupies
// pp[2n,4n+2) with its top two lanes parked where vinf's lanes 0..1 belong
// (the real lanes ride in vinf0/vinf1); vinf = pp[4n,4n+twor); vm1 (magnitude,
// sign sa) and v2 sit in scratch, 2n+2 limbs each, zero-padded to their vector
// boundary. Output pp redundant, like every mul in this file.
//
// Both halvings use the redundant signed halving identity (no canonize, no
// mask<->GPR traffic); only divexact3's input is canonicalized. t1 = v1 - v0
// is never materialized: t2 folds it into one fused pass and step (5) takes
// the 3-operand form. c1's placement is fused with its computation. When
// twor < n + 2 a redundant t2's high lanes would not fit the output at B^3n,
// so that case falls back to a canonical t2 (zero high lanes, truncable add).
static inline void u52_toom_interp_5pts(sb_pvec pp, sb_pvec v2, sb_pvec vm1, uint64_t n,
                                        uint64_t twor, uint8_t sa,
                                        uint64_t vinf0, uint64_t vinf1){
    assert(n >= 2 && twor >= 2);
    sb_pvec v1 = u52_pvec_add(pp, 2*n);
    uint64_t *ppl = (uint64_t *)(void *)pp;
    uint64_t *v1l = ppl + 2*n;
    uint64_t *v2l = (uint64_t *)(void *)v2;
    const uint64_t *vm1l = (const uint64_t *)(const void *)vm1;
    const uint64_t kk2 = 2*n + 2;
    const int t2_nc = twor >= n + 2;

    // (1) v2 <- (v2 - vm1_signed)/3                         (5 3 1 1 0)
    // (two passes: a fused canon+div3 loop measured 3-5% slower overall --
    // it forfeits divexact3's 4x unroll and in-block carry chaining)
    if(sa) u52_add_canon(v2, (sb_cpvec)v2, (sb_cpvec)vm1, kk2, sb_zero());
    else   u52_sub_canon(v2, (sb_cpvec)v2, (sb_cpvec)vm1, kk2, sb_zero());
    u52_divexact3(v2, (sb_cpvec)v2, kk2);
    // (2) vm1 <- tm1 = (v1 - vm1_signed)/2 = c1 + c3        (0 1 0 1 0)
    u52_addsub_rsh1_nc(vm1, (sb_cpvec)v1, (sb_cpvec)vm1, kk2, !sa);
    // (3)+(4) v2 <- t2 = (v2 - (v1 - v0))/2 = 2*c4 + c3     (2 1 0 0 0)
    if(t2_nc){
        u52_subadd_rsh1_nc(v2, (sb_cpvec)v2, (sb_cpvec)v1, (sb_cpvec)pp, kk2, 2*n);
    }else{
        sub_nc_52(v1, (sb_cpvec)v1, (sb_cpvec)pp, 2*n);            // t1 = v1 - v0
        u52_sub_rsh1_canon(v2, (sb_cpvec)v2, (sb_cpvec)v1, kk2, sb_zero());
    }
    // (5) v1 <- t1 - tm1 = c4 + c2                          (1 0 1 0 0)
    if(t2_nc){
        u52_sub3_nc(v1, (sb_cpvec)v1, (sb_cpvec)pp, (sb_cpvec)vm1, 2*n);
    }else{
        sub_nc_52(v1, (sb_cpvec)v1, (sb_cpvec)vm1, 2*n);           // t1 already in v1
    }
    v1l[2*n]     -= vm1l[2*n];
    v1l[2*n + 1] -= vm1l[2*n + 1];
    // (6) v2 <- t2 - 2*vinf = c3 (vinf lanes 0..1 are the scalars)
    v2l[0] -= 2 * vinf0;
    v2l[1] -= 2 * vinf1;
    if(twor > 2)
        sublsh1_nc_52(u52_pvec_add(v2, 2), (sb_cpvec)u52_pvec_add(v2, 2),
                      u52_cpvec_add((sb_cpvec)pp, 4*n + 2), twor - 2);
    // (7) v1 <- v1 - vinf = c2
    v1l[0] -= vinf0;
    v1l[1] -= vinf1;
    if(twor > 2)
        sub_nc_52(u52_pvec_add(v1, 2), (sb_cpvec)u52_pvec_add(v1, 2),
                  u52_cpvec_add((sb_cpvec)pp, 4*n + 2), twor - 2);

    // assembly: restore the real vinf lanes (c2's parked tops move home), then
    // (8) fused with c1's placement: pp+n += tm1 - c3; finally c3 at B^3n.
    const uint64_t w0 = ppl[4*n], w1 = ppl[4*n + 1];
    ppl[4*n]     = vinf0 + w0;
    ppl[4*n + 1] = vinf1 + w1;
    // c3's placement stays a cheap nc add; the LAST operation is then a single
    // canonneg fold over the entire mixed span [n, total): it applies c1
    // (= tm1 - c3) at B^n and resolves every wrapped lane in one dual-chain
    // pass that ends at the top of the number (where the whole-value
    // non-negativity makes the final borrow genuinely zero). [0,n) keeps c0's
    // non-negative child lanes, so the output is non-negative redundant with
    // the children's lane bound -- no standalone canonicalization pass.
    const uint64_t L3 = t2_nc ? kk2 : (kk2 < n + twor ? kk2 : n + twor);
    add_nc_52(u52_pvec_add(pp, 3*n), u52_cpvec_add((sb_cpvec)pp, 3*n), (sb_cpvec)v2, L3);
    u52_fold_addsub_canon(u52_pvec_add(pp, n), (sb_cpvec)vm1, kk2, (sb_cpvec)v2, kk2,
                              3*n + twor);
}

// Toom-33 for near-balanced operands, GMP points {0, +-1, 2, inf} and GMP's
// memory plan: bs1/as2/bs2 live in pp and die into the v1/vinf products; gp
// and then vm1 share the bottom of scratch; v2 overlays dead asm1/bsm1.
// Output redundant; caller canonicalizes. Scratch high-water ~5n+5 limbs
// plus recursion, matching GMP.
static inline int mul_u52_toom33_n(sb_pvec pp, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn,
                                    uint64_t n, scratch *sc){
    const uint64_t s = an - 2*n, t = bn - 2*n;
    assert(an >= bn && 0 < s && s <= n && 0 < t && t <= n && n >= 2);
    sb_cpvec a0 = a, a1 = u52_cpvec_add(a, n), a2 = u52_cpvec_add(a, 2*n);
    sb_cpvec b0 = b, b1 = u52_cpvec_add(b, n), b2 = u52_cpvec_add(b, 2*n);
    SCRATCH(sc);
    const uint64_t evec = u52_vec_count(n + 1);
    const uint64_t pvcs = u52_vec_count(2*n + 2);
    sb_pvec vm1  = SALLOC(sc, sb_vec, pvcs);            // gp, then vm1
    sb_pvec asm1 = SALLOC(sc, sb_vec, evec);            // then v2 (spans asm1+bsm1)
    sb_pvec bsm1 = SALLOC(sc, sb_vec, evec);
    sb_pvec as1  = SALLOC(sc, sb_vec, evec);
    sb_pvec gp = vm1, v2 = asm1;
    sb_pvec bs1 = pp;                                 // n+1, dies into v0
    sb_pvec as2 = u52_pvec_add(pp, n + 1);            // n+1, dies into v1
    sb_pvec bs2 = u52_pvec_add(pp, 2*n + 2);          // n+1, dies into v1

    // evaluations
    u52_add_carry(gp, a0, a2, n, s, true);                          // a0+a2, n+1
    u52_add_carry(as1, (sb_cpvec)gp, a1, n + 1, n, false);             // A(1) < 3B^n
    int64_t sa = u52_cmp_and_sub(asm1, (sb_cpvec)gp, a1, n + 1, n);    // |a0+a2-a1|
    u52_add_carry(gp, b0, b2, n, t, true);                          // b0+b2, n+1
    u52_add_carry(bs1, (sb_cpvec)gp, b1, n + 1, n, false);             // B(1) < 3B^n
    int64_t sb = u52_cmp_and_sub(bsm1, (sb_cpvec)gp, b1, n + 1, n);    // |b0+b2-b1|
    u52_eval2_canon(as2, a0, a1, a2, NULL, n, s);                   // A(2), top <= 6
    u52_eval2_canon(bs2, b0, b1, b2, NULL, n, t);                   // B(2), top <= 6
    const uint8_t vm1_neg = (sa < 0) ^ (sb < 0);
    const uint64_t lam = (uint64_t)(sa < 0 ? -sa : sa);
    const uint64_t lbm = (uint64_t)(sb < 0 ? -sb : sb);

    // products, GMP order: vm1 (kills gp), v2 (kills asm1/bsm1), vinf, v1
    // (kills as2/bs2, top two lanes park over vinf), v0 (kills bs1)
    if(lam && lbm) mul_u52_dispatch_canon(vm1, (sb_cpvec)asm1, (sb_cpvec)bsm1, lam, lbm, sc);
    const uint64_t lvm = (lam && lbm) ? lam + lbm : 0;
    memset((uint64_t *)(void *)vm1 + lvm, 0, (pvcs * 8 - lvm) * 8);
    sb_store(v2 + pvcs - 1, sb_zero());
    mul_u52_dispatch_canon(v2, (sb_cpvec)as2, (sb_cpvec)bs2, n + 1, n + 1, sc);
    mul_u52_dispatch_canon(u52_pvec_add(pp, 4*n), a2, b2, s, t, sc);   // vinf
    uint64_t *ppl = (uint64_t *)(void *)pp;
    const uint64_t vinf0 = ppl[4*n], vinf1 = ppl[4*n + 1];
    mul_u52_dispatch_canon(u52_pvec_add(pp, 2*n), (sb_cpvec)as1, (sb_cpvec)bs1, n + 1, n + 1, sc);
    mul_u52_dispatch_canon(pp, a0, b0, n, n, sc);                      // v0

    u52_toom_interp_5pts(pp, v2, vm1, n, s + t, vm1_neg, vinf0, vinf1);
    return 0;
}

static inline int mul_u52_toom33(sb_pvec pp, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn, scratch *sc){
    return mul_u52_toom33_n(pp, a, b, an, bn, (an + 2) / 3, sc);
}

// Toom-42 for an ~ 2*bn, GMP points {0, +-1, 2, inf}. Evals follow GMP's
// toom_eval_dgr3_pm1 (p13 borrows pp as its temporary) and bs2 = bs1 + b1.
// Same shared interpolation; output redundant.
static inline int mul_u52_toom42(sb_pvec pp, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn, scratch *sc){
    const uint64_t n = (an >= 2*bn) ? (an + 3) >> 2 : ((bn + 1) >> 1);
    const uint64_t s = an - 3*n, t = bn - n;
    assert(0 < s && s <= n && 0 < t && t <= n && n >= 2);
    sb_cpvec a0 = a, a1 = u52_cpvec_add(a, n), a2 = u52_cpvec_add(a, 2*n), a3 = u52_cpvec_add(a, 3*n);
    sb_cpvec b0 = b, b1 = u52_cpvec_add(b, n);
    SCRATCH(sc);
    const uint64_t evec = u52_vec_count(n + 1);
    const uint64_t pvcs = u52_vec_count(2*n + 2);
    sb_pvec vm1  = SALLOC(sc, sb_vec, pvcs);
    sb_pvec asm1 = SALLOC(sc, sb_vec, evec);            // p02, then asm1, then v2
    sb_pvec bsm1 = SALLOC(sc, sb_vec, evec);
    sb_pvec as1  = SALLOC(sc, sb_vec, evec);
    sb_pvec as2  = SALLOC(sc, sb_vec, evec);
    sb_pvec bs1  = SALLOC(sc, sb_vec, evec);
    sb_pvec bs2  = SALLOC(sc, sb_vec, evec);
    sb_pvec p13 = pp, v2 = asm1;                      // p13 borrows pp (GMP: tp)

    // evaluations
    u52_add_carry(p13, a1, a3, n, s, true);                          // a1+a3, n+1
    u52_add_carry(asm1, a0, a2, n, n, true);                         // p02 = a0+a2
    u52_add_carry(as1, (sb_cpvec)asm1, (sb_cpvec)p13, n + 1, n + 1, false);// A(1) < 4B^n
    int64_t sa = u52_cmp_and_sub(asm1, (sb_cpvec)asm1, (sb_cpvec)p13, n + 1, n + 1);
    u52_eval2_canon(as2, a0, a1, a2, a3, n, s);                      // A(2), top <= 14
    u52_add_carry(bs1, b0, b1, n, t, true);                          // B(1)
    int64_t sb = u52_cmp_and_sub(bsm1, b0, b1, n, t);                // |b0-b1|
    u52_add_carry(bs2, (sb_cpvec)bs1, b1, n + 1, t, false);             // B(2) = bs1+b1 < 3B^n
    const uint8_t vm1_neg = (sa < 0) ^ (sb < 0);
    const uint64_t lam = (uint64_t)(sa < 0 ? -sa : sa);
    const uint64_t lbm = (uint64_t)(sb < 0 ? -sb : sb);

    // products, GMP order
    if(lam && lbm) mul_u52_dispatch_canon(vm1, (sb_cpvec)asm1, (sb_cpvec)bsm1, lam, lbm, sc);
    const uint64_t lvm = (lam && lbm) ? lam + lbm : 0;
    memset((uint64_t *)(void *)vm1 + lvm, 0, (pvcs * 8 - lvm) * 8);
    sb_store(v2 + pvcs - 1, sb_zero());
    mul_u52_dispatch_canon(v2, (sb_cpvec)as2, (sb_cpvec)bs2, n + 1, n + 1, sc);
    mul_u52_dispatch_canon(u52_pvec_add(pp, 4*n), a3, b1, s, t, sc);   // vinf
    uint64_t *ppl = (uint64_t *)(void *)pp;
    const uint64_t vinf0 = ppl[4*n], vinf1 = ppl[4*n + 1];
    mul_u52_dispatch_canon(u52_pvec_add(pp, 2*n), (sb_cpvec)as1, (sb_cpvec)bs1, n + 1, n + 1, sc);
    mul_u52_dispatch_canon(pp, a0, b0, n, n, sc);                      // v0

    u52_toom_interp_5pts(pp, v2, vm1, n, s + t, vm1_neg, vinf0, vinf1);
    return 0;
}

// ---------------------------------------------------------------------------
// Shape-aware dispatcher, modeled on GMP mpn_mul (mpn/generic/mul.c): the
// smaller operand bn picks the size band, the ratio an/bn picks the family
// member, with cutpoints at the midpoints between the algorithms' ideal
// ratios (5/4 between 1:1 and 3:2; 7/4 between 3:2 and 2:1). Extreme
// imbalance is strip-mined: toom42 over (2bn x bn) blocks marching along a,
// overlap-adding a bn-limb seam. In redundant arithmetic the seam is a plain
// add_nc + copy -- GMP's saved-triangle/INCR carry threading vanishes, and
// each output lane is touched by at most two strips (+1 bit of headroom).

static inline int u52_toom32_shape_ok(uint64_t an, uint64_t bn){
    const uint64_t n = (2*an >= 3*bn) ? (an + 2) / 3 : ((bn + 1) >> 1);
    if(n < 2 || an <= 2*n || bn <= n) return 0;
    const uint64_t s = an - 2*n, t = bn - n;
    return s <= n && t <= n && s + t > n && s + t >= 3;
}
static inline int u52_toom42_shape_ok(uint64_t an, uint64_t bn){
    const uint64_t n = (an >= 2*bn) ? (an + 3) >> 2 : ((bn + 1) >> 1);
    if(n < 2 || an <= 3*n || bn <= n) return 0;
    return an - 3*n <= n && bn - n <= n;
}
static inline int u52_toom33_shape_ok(uint64_t an, uint64_t bn){
    const uint64_t n = (an + 2) / 3;
    return n >= 2 && an > 2*n && bn > 2*n && an - 2*n <= n && bn - 2*n <= n;
}

static int mul_u52_stripmine(sb_pvec pp, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn, scratch *sc){
    SCRATCH(sc);
    sb_pvec ws = SALLOC(sc, sb_vec, u52_vec_count(4*bn));
    // Sliver guard: when an mod 2bn <= 8, the natural 2bn slicing leaves an
    // awkward near-empty tail; derive equalized widths from an instead
    // (every block 2bn + <=4 limbs, still canonical toom42 territory, and no
    // dimension ever degenerates to <= 8).
    if(an % (2*bn) <= 8 && (an + bn) / (2*bn) >= 2){
        const uint64_t nblk = (an + bn) / (2*bn);
        uint64_t done = 0;
        for(uint64_t i = 0; i < nblk; ++i){
            const uint64_t w = an / nblk + (i < an % nblk ? 1 : 0);
            if(i == 0){
                mul_u52_dispatch_canon(pp, a, b, w, bn, sc);
            }else{
                mul_u52_dispatch_canon(ws, u52_cpvec_add(a, done), b, w, bn, sc);
                sb_pvec seam = u52_pvec_add(pp, done);
                add_nc_52(seam, (sb_cpvec)seam, (sb_cpvec)ws, bn);
                memcpy((uint64_t *)(void *)pp + done + bn,
                       (const uint64_t *)(const void *)ws + bn, w * 8);
            }
            done += w;
        }
        return 0;
    }
    mul_u52_toom42(pp, a, b, 2*bn, bn, sc);
    a = u52_cpvec_add(a, 2*bn); an -= 2*bn;
    pp = u52_pvec_add(pp, 2*bn);
    while(an >= 3*bn){
        mul_u52_toom42(ws, a, b, 2*bn, bn, sc);
        add_nc_52(pp, (sb_cpvec)pp, (sb_cpvec)ws, bn);                  // seam
        memcpy((uint64_t *)(void *)pp + bn,
               (const uint64_t *)(const void *)ws + bn, 2*bn * 8);
        a = u52_cpvec_add(a, 2*bn); an -= 2*bn;
        pp = u52_pvec_add(pp, 2*bn);
    }
    // remainder: 0 < an < 3bn (an may be < bn; dispatch swaps). Blocks and the
    // remainder are class-0, so seam lanes stay non-negative and small.
    mul_u52_dispatch_canon(ws, a, b, an, bn, sc);
    add_nc_52(pp, (sb_cpvec)pp, (sb_cpvec)ws, bn < an + bn ? bn : an + bn);
    memcpy((uint64_t *)(void *)pp + bn,
           (const uint64_t *)(const void *)ws + bn, an * 8);
    return 0;
}



// Measurement-based dispatch: per ratio band, evaluate 2-3 candidate
// solutions with the fitted cost curves and recurse into the cheapest.
// Direct algorithms are costed at their canonical-shape curve; compositions
// (strip-mining, halve-and-seam) sum their parts. Continuous costs make the
// choice stable through u52-size quantization wobble (no cutpoint flapping),
// and shapes between canonical ratios pick whichever neighbor is truly
// cheaper at that size. bn <= 32 exits straight to basecase so the selection
// cost is paid only where it is negligible.
static int mul_u52_dispatch(sb_pvec r, sb_cpvec a, sb_cpvec b, uint64_t an, uint64_t bn, scratch *s) {
    if(!an || !bn) return 0;
    if(an < bn){
        uint64_t nt = an; an = bn; bn = nt;
        sb_cpvec t = a; a = b; b = t;
    }
    if(bn < MUL_U52_T22_THRESHOLD){
        mul_u52_basecase(r, a, b, an, bn);
        return 0;
    }
    if(bn < MUL_U52_T33_THRESHOLD){            // ToomX2 band
        if(an >= 3*bn)
            return mul_u52_stripmine(r, a, b, an, bn, s);
        if(4*an >= 5*bn && bn >= MUL_U52_T32_OK_THRESHOLD){
            if(4*an < 7*bn){
                if(u52_toom32_shape_ok(an, bn))
                    return mul_u52_toom32(r, a, b, an, bn, s);
            }else if(bn >= MUL_U52_T42_OK_THRESHOLD && u52_toom42_shape_ok(an, bn))
                return mul_u52_toom42(r, a, b, an, bn, s);
        }
        return mul_u52_karatsuba(r, a, b, an, bn, s);
    }
    // ToomX3 band (no toom43/53/63: collapse GMP's ladder onto 33/32/42)
    if(2*an >= 5*bn)
        return mul_u52_stripmine(r, a, b, an, bn, s);
    if(6*an < 7*bn && u52_toom33_shape_ok(an, bn))
        return mul_u52_toom33(r, a, b, an, bn, s);
    if(4*an < 7*bn && u52_toom32_shape_ok(an, bn))
        return mul_u52_toom32(r, a, b, an, bn, s);
    if(u52_toom42_shape_ok(an, bn))
        return mul_u52_toom42(r, a, b, an, bn, s);
    return mul_u52_karatsuba(r, a, b, an, bn, s);
}

} // namespace
