// Imported from libsbn v2 detail/div/dc_div.h; caller-owned Frame replaces TLS scratch.
#pragma once
namespace sbn::v3::u52 {
// Divide-and-conquer block division (u52, beta = 2^416 = 8 digits/block),
// modelled on GMP's mpn_dcpi1_div_qr / _qr_n.  The bulk work is full
// multiplication (mul_u52_dispatch_canon) instead of the rank-1 submul, and the
// recursion is sub-quadratic.  Non-power-of-2 sizes are handled the GMP way:
// the recursion splits n into lo=n>>1 / hi=n-lo (odd ok), and the dispatcher
// peels the quotient in dn-block chunks with a properly-handled irregular top
// chunk (never a naive split).

#define INLINE static inline __attribute__((always_inline))

// Flat leaf below 12 blocks; inherited v2 crossover, measured again through
// the complete native service before promoting an automatic policy.
inline constexpr uint64_t dc_leaf_blocks = 12;

// ---- block multi-precision helpers (carry/borrow threaded across blocks) ----

// qp[0..len) -= 1 ; returns borrow out.
INLINE unsigned block_sub_1(sb_limb* qp, uint64_t len){
    unsigned b = 1;
    for(uint64_t i = 0; i < len && b; i++){
        sb_vec x = sb_load((sb_cpvec)(qp + 8*i));
        x = block_subb(x, sb_zero(), &b);
        sb_store((sb_pvec)(qp + 8*i), x);
    }
    return b;
}
// tp[0..an+bn) = a[0..an) * b[0..bn)  (blocks), canonical.  Uses the arena.
INLINE void blk_mul(sb_limb* tp, const sb_limb* a, uint64_t an, const sb_limb* b, uint64_t bn,Frame &space){
    scratch* sc = &space;
    SCRATCH(sc);
    if(an >= bn) mul_u52_dispatch_canon((sb_pvec)tp, (sb_cpvec)a, (sb_cpvec)b, an*8, bn*8, sc);
    else         mul_u52_dispatch_canon((sb_pvec)tp, (sb_cpvec)b, (sb_cpvec)a, bn*8, an*8, sc);
    // dispatch_canon only folds class-1; a class-0 result may still be
    // non-negative-redundant (lanes >= 2^52).  The block sub/add chains need
    // canonical digits, so force a positive canonicalize.
    u52_canon_pos((sb_pvec)tp, (sb_cpvec)tp, (an+bn)*8);
}

// ---- 2n/n recursive division (mpn_dcpi1_div_qr_n analogue), returns qh ----
// np: 2n blocks -> quotient qp[0..n), remainder in np[0..n). dp: n blocks. tp: n blocks scratch.
static int blk_dcpi1_div_qr_n(sb_limb* qp, sb_limb* np, const sb_limb* dp, uint64_t n,
                              const sb_limb* V18, sb_limb* tp,Frame &space){
    const uint64_t lo = n >> 1, hi = n - lo;
    int qh, ql; unsigned cy;
    // high half quotient: divide top 2hi blocks by top hi blocks of d
    if(hi < dc_leaf_blocks) qh = div2b_core(qp + 8*lo, np + 8*(2*lo), 2*hi, dp + 8*lo, hi, V18, 3);
    else                  qh = blk_dcpi1_div_qr_n(qp + 8*lo, np + 8*(2*lo), dp + 8*lo, hi, V18, tp,space);
    blk_mul(tp, qp + 8*lo, hi, dp, lo,space);             // Qhi * Dlo  (n blocks)
    cy = block_sub_n(np + 8*lo, tp, n);
    if(qh) cy += block_sub_n(np + 8*n, dp, lo);
    while(cy){ qh -= block_sub_1(qp + 8*lo, hi); cy -= block_add_n(np + 8*lo, dp, n); }
    // low half quotient: divide np[hi..hi+2lo) by top lo blocks of d
    if(lo < dc_leaf_blocks) ql = div2b_core(qp, np + 8*hi, 2*lo, dp + 8*hi, lo, V18, 3);
    else                  ql = blk_dcpi1_div_qr_n(qp, np + 8*hi, dp + 8*hi, lo, V18, tp,space);
    blk_mul(tp, dp, hi, qp, lo,space);                    // Dhi(low hi) * Qlo  (n blocks)
    cy = block_sub_n(np, tp, n);
    if(ql) cy += block_sub_n(np + 8*lo, dp, hi);
    while(cy){ block_sub_1(qp, lo); cy -= block_add_n(np, dp, n); }
    return qh;
}

// ---- one quotient chunk of c blocks from window np[0..c+dn) / d ; returns qh ----
static int blk_chunk_divide(sb_limb* qp, sb_limb* np, uint64_t c, const sb_limb* dp, uint64_t dn,
                            const sb_limb* V18, sb_limb* tp,Frame &space){
    if(c < dc_leaf_blocks)
        return div2b_core(qp, np, c + dn, dp, dn, V18, 3);    // full-divisor schoolbook
    // c >= THRESHOLD (>=2): 2c/c on the top, then cross-correct with low (dn-c) of d
    const uint64_t k = dn - c;
    int qh = blk_dcpi1_div_qr_n(qp, np + 8*k, dp + 8*k, c, V18, tp,space);   // top 2c / top c of d
    if(c != dn){
        blk_mul(tp, qp, c, dp, k,space);                       // Q * Dlow  (dn blocks)
        unsigned cy = block_sub_n(np, tp, dn);
        if(qh) cy += block_sub_n(np + 8*c, dp, k);
        while(cy){ qh -= block_sub_1(qp, c); cy -= block_add_n(np, dp, dn); }
    }
    return qh;
}

// ---- general dispatcher (mpn_dcpi1_div_qr analogue), returns qh ----
// np: nn blocks -> remainder in low dn ; quotient qp[0..nn-dn). dp: dn>=2 normalized.
static int blk_dcpi1_div_qr(sb_limb* qp, sb_limb* np, uint64_t nn, const sb_limb* dp, uint64_t dn,
                            const sb_limb* V18, sb_limb* tp,Frame &space){
    const uint64_t qn = nn - dn;
    uint64_t c = qn % dn; if(c == 0) c = dn;            // top chunk size in [1,dn]
    uint64_t p = qn - c;
    int qh = blk_chunk_divide(qp + 8*p, np + 8*p, c, dp, dn, V18, tp,space);
    while(p > 0){
        p -= dn;
        blk_dcpi1_div_qr_n(qp + 8*p, np + 8*p, dp, dn, V18, tp,space);
    }
    return qh;
}


} // namespace sbn::v3::u52
