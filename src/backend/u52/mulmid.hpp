/* Imported from libsbn/include/sbn/detail/mulmid/mulmid.h; kernel arithmetic retained. */
#pragma once
namespace sbn::v3::u52 {

/**
 * Middle product formulation:
 * convention: B = |b| > A = |a|
 * MP(a,b) = sum_{t in [A-1, B-1]} sum_{i in [0, A-1]} a[i]*b[t-i]
 * E.G. the middle part of a polynomial multiplication that is rectangular.
 * Example:
 * a\b | 0 1 2 3 4 5 6 7
 *  0  | O O O * * * * *
 *  1  | O O * * * * * .
 *  2  | O * * * * * . .
 *  3  | * * * * * . . .
 * DIAG| 3 4 5 6 7 8 9 a
 *      [SUMMED UP] <- those diagonals
 */

static inline void mulmid_basecase(sb_plimb r, const sb_limb *a, const sb_limb *b, int64_t an, int64_t bn){
    /**
     * Design of basecase kernel:
     * a. We have to accumulate lo and hi sums separately, since hi sum leak from 
     * below, and lo sum leak from above the boundary.
     * This affect our semantical correctness, also invalidates divide and conquer.
     * b. We first iterate via b indices. This is because B is larger and the result
     * length is under B+1. (Actually B-A+1+1; the final +1 come from the hi offset)
     * c. Thus, we indeed iterate via result index. Every time we want to accumulate
     * products for r[i..i+8], which has a lower bound on b that involved in the sum
     * of b[i..i+8]
     * d. We then decrease a pointer. Process by block of 8 in a value, however we
     * accumulate like dividing a by 4. The intuition is the following:
     * [case a] accumulate by 8: we cannot avoid that, for the first block, we have
     * to waste half of the multiplication for useless garbage. The shape is:
     * | notice the    * | * * * * * * * * |
     * | upper tri-  * * | * * * * * * * * |
     * | angle     * * * | * * * * * * *   |
     * | here    * * * * | * * * * * *     |
     * |       * * * * * | * * * * *       |
     * |     * * * * * * | * * * *         |
     * |   * * * * * * * | * * *           |
     * | * * * * * * * * | * *             |
     * [bptr .. bptr+8  ][bptr+8 .. bptr+16]
     * The upper triangle is indeed not reducible under this formulation, because we
     * need to do this diagonal triangle anyways. We also have to take care of the
     * lo/hi separation. A naive idea would be doing 16 accumulator vectors; a more
     * advanced idea is to load bptr and bptr+1 vector, then do even/odd split.
     *
     * This is why we want to do step 4 on a.
     *                   | * * * * * * * * | * ...
     *                   [bptr+8 .. bptr+16] <- next block, at here swaps for the 2nd
     *                                          vector; and load [bptr+16 .. bptr+24]
     * |       * * * * * | * * * * *       |   /\  # Figure of this 4 step is shifted
     * |     * * * * * * | * * * *         |  /||\ # 4 limbs front
     * |   * * * * * * * | * * *           |   || Direction of computation
     * | * * * * * * * * | * *             |   ||
     * [bptr+ 4..bptr+12] <- We don't load this vector; instead, since we need to do
     *                       next block as [bptr+8 .. bptr+16], we can pre-load this
     *                       vector, then compute [bptr+ 4 .. bptr+12] as a simple
     *                       vpalignq(hi, lo, 4). This reduce the load occupation due
     *                       to potential unaligned load jankiness.
     * |       * * * * * | * * * * *       |   /\
     * |     * * * * * * | * * * *         |  /||\
     * |   * * * * * * * | * * *           |   || Direction of computation
     * | * * * * * * * * | * *             |   ||
     * [bptr .. bptr+8  ][bptr+8 .. bptr+16] <- load two vectors right now
     * E. Tail computation: we do it vertically
     *                   | * * * * * * * * |   * ...
     * ------------------------------------|a  ||                   
     * |       * * * * * | * * * * *       |0 \||/ in this direction a increases
     * |     * * * * * * | * * * * |       |1  \/  We load a in vector A
     * |   * * * * * * * | * * * | |       |2 
     * | * * * * * * * * | * * | | |       |3  If we want to do vertical point
     *              bptr + 4 5 6 7 8           multiplication, then we have to load B
     *                                         in the underlying form: [44440000]
     *                                         [55551111][66662222][77773333], etc.
     * |       * * * * * | * * * * *       |4 
     * |     * * * * * * | * * * *         |5  Thing is essentially [44440000] *
     * |   * * * * * * * | * * *           |6  [01234567] -> [45674567]
     * | * * * * * * * * | * *             |7  Note that:
     *              bptr + 0 1 2 3 4           [456789ab] -> acc[3]
     *                                         [56789abc] -> acc[2]
     *                                         [6789abcd] -> acc[1]
     *                                         [789abcde] -> acc[0]
     */
    /**
     * Bookkeeping (verified): acc_lo[q] lane l accumulates diagonal s = i-q+l,
     * acc_hi[q] lane l accumulates s = i-q+1+l. So relative to window
     * W_i = r[i..i+8): lo0/hi1 aligned, lo1/hi2 down 1, lo2/hi3 down 2,
     * lo3 down 3, hi0 UP 1. Down-spilled top lanes of W_i come from the NEXT
     * block, so W_i is stored at the end of block i+8 from {last, curr} pairs.
     * hi0 is the exception (it would need block i-8 at that point): it is
     * retired one block early -- into hi1, not lo0, so the final window keeps
     * lo/hi separated for the masked top-limb add.
     *
     * Tail: quarter (q, s) = the q-group's products on diagonal s. Block i
     * computes exactly the quarters with 0 <= s-i+q <= 7, so after the last
     * horizontal block i_L everything with q+s >= V := i_L+8 is missing.
     * The vertical patterns [c+4 c+4 c+4 c+4 c c c c] (44440000, 55551111, ..)
     * anchored at b+V sweep exactly the anti-diagonal q+s = V+c, two products
     * per quarter per 8 rows of a; masking to valid diagonals s in [0, rn-1]
     * both tapers the tail triangle and keeps non-MP terms (s = rn lo, i.e.
     * diagonal bn) out of r[rn]. Horizontal blocks run while rn-i >= 5, hence
     * P = rn+3-V in [0,7]; rn ≡ 5 (mod 8) needs no tail at all. P <= 4 uses
     * the vertical patterns, P >= 5 one more full-width block (see below).
     * Fold lanes l/l+4 (two trapezoids, same diagonal), align pattern c up by
     * c lanes: composite clo lane t <-> output V-3+t, chi lane t <-> V-2+t.
     */
    const int64_t rn = bn - an + 1; // output length = rn+1
    // Raw {last, curr} acc sets: each store term is ONE alignr(curr, last, k).
    // The bh-group and b0-group accumulate into SEPARATE sets (16 chains of
    // 1 madd each instead of 8 chains of 2): 4c madd latency against the 8c
    // tile gives scheduling slack, which is what actually reaches the 2-pipe
    // madd52 bound on Zen5 (8x2 serial chains are zero-slack and lose ~15%).
    // ~28 zmm live; compiles spill-free, rotation moves rename-eliminated.
    sb_vec acc_lo[4], acc_hi[4], acc2_lo[4], acc2_hi[4], last_lo[4], last_hi[4], b0, b1, bh;
    __mmask8 stmask = 0xFE; // hi-side mask for the next window store; 0xFE until W_0 is stored
    for(int k = 0; k < 4; ++k){ last_lo[k] = sb_zero(); last_hi[k] = sb_zero(); }
    #define mm_acc(CLO, CHI, ind, aind, bv) { \
        const sb_vec _a = sb_splat_load(aptr, aind); \
        CLO[ind] = sb_madd52lo(CLO[ind], _a, bv); \
        CHI[ind] = sb_madd52hi(CHI[ind], _a, bv); \
    }
    int64_t i = 0;
    for(; rn - i >= 5; i += 8){
        for(int k = 0; k < 4; ++k){ acc_lo[k] = sb_zero(); acc_hi[k] = sb_zero(); acc2_lo[k] = sb_zero(); acc2_hi[k] = sb_zero(); }
        const sb_limb *bptr = b + i;
        b0 = sb_load(bptr);
        for(int64_t j = an; j > 0; j -= 8){
            const sb_limb *aptr = a + (j-8);
            b1 = sb_load(bptr += 8);
            bh = sb_alignr64(b1, b0, 4);
            if(j <= 8) switch(j){
                full:
                case 8: mm_acc(acc2_lo, acc2_hi, 3, 0, bh);
                case 7: mm_acc(acc2_lo, acc2_hi, 2, 1, bh);
                case 6: mm_acc(acc2_lo, acc2_hi, 1, 2, bh);
                case 5: mm_acc(acc2_lo, acc2_hi, 0, 3, bh);
                case 4: mm_acc(acc_lo, acc_hi, 3, 4, b0);
                case 3: mm_acc(acc_lo, acc_hi, 2, 5, b0);
                case 2: mm_acc(acc_lo, acc_hi, 1, 6, b0);
                case 1: mm_acc(acc_lo, acc_hi, 0, 7, b0);
            }
            else goto full;
            b0 = b1;
        }
        for(int k = 0; k < 4; ++k){ acc_lo[k] = sb_add(acc_lo[k], acc2_lo[k]); acc_hi[k] = sb_add(acc_hi[k], acc2_hi[k]); }
        // retire the up-spilling hi0 while the previous block's hi0 is alive
        acc_hi[1] = sb_add(acc_hi[1], sb_alignr64(acc_hi[0], last_hi[0], 7));
        if(i){ // store W_{i-8}: binary-tree harvest, lo/hi kept separate.
               // stmask = 0xFE on the W_0 store only: block 0's sub-band
               // hi(diag -1) garbage all lands in lane 0 of the hi aggregate,
               // and r[0] has no legitimate hi part -- one masked add drops it.
            const sb_vec _l = sb_add(sb_add(last_lo[0], sb_alignr64(acc_lo[1], last_lo[1], 1)),
                                sb_add(sb_alignr64(acc_lo[2], last_lo[2], 2), sb_alignr64(acc_lo[3], last_lo[3], 3)));
            const sb_vec _h = sb_add(sb_add(last_hi[1], sb_alignr64(acc_hi[2], last_hi[2], 1)),
                                sb_alignr64(acc_hi[3], last_hi[3], 2));
            sb_store(r + i - 8, sb_add(_l, _h, stmask, _l));
            stmask = 0xFF;
        }
        for(int k = 0; k < 4; ++k){ last_lo[k] = acc_lo[k]; last_hi[k] = acc_hi[k]; }
    }
    const int64_t V = i, P = rn + 3 - V; // pattern count in [0,7]
    sb_vec F_lo[4], F_hi[4];
    if(P >= 5){
        // Generic-block tail: one more full-width block at V covers all
        // remaining quarters (needed q+s <= V+n2+2 <= V+7); garbage columns
        // clip at the store masks. Measured crossover vs patterns is P ~ 4.5
        // on Zen5 at every an: the vertical sweep costs ~3.6 + 1.3P c per
        // a-chunk (its base overhead never amortizes) while a full block runs
        // at the steady-state 8 c/chunk with everything hidden under the madds.
        for(int k = 0; k < 4; ++k){ acc_lo[k] = sb_zero(); acc_hi[k] = sb_zero(); acc2_lo[k] = sb_zero(); acc2_hi[k] = sb_zero(); }
        const sb_limb *bptr = b + V;
        b0 = sb_load(bptr);
        for(int64_t j = an; j > 0; j -= 8){
            const sb_limb *aptr = a + (j-8);
            b1 = sb_load(bptr += 8);
            bh = sb_alignr64(b1, b0, 4);
            if(j <= 8) switch(j){
                full2:
                case 8: mm_acc(acc2_lo, acc2_hi, 3, 0, bh);
                case 7: mm_acc(acc2_lo, acc2_hi, 2, 1, bh);
                case 6: mm_acc(acc2_lo, acc2_hi, 1, 2, bh);
                case 5: mm_acc(acc2_lo, acc2_hi, 0, 3, bh);
                case 4: mm_acc(acc_lo, acc_hi, 3, 4, b0);
                case 3: mm_acc(acc_lo, acc_hi, 2, 5, b0);
                case 2: mm_acc(acc_lo, acc_hi, 1, 6, b0);
                case 1: mm_acc(acc_lo, acc_hi, 0, 7, b0);
            }
            else goto full2;
            b0 = b1;
        }
        for(int k = 0; k < 4; ++k){ F_lo[k] = sb_add(acc_lo[k], acc2_lo[k]); F_hi[k] = sb_add(acc_hi[k], acc2_hi[k]); }
    }else{
        // Vertical pattern engine, broadcast form: B_c = [b[c+4] x4 | b[c] x4] is
        // just two scalar broadcasts (plain vpbroadcastq + one 0x0F-merged one) --
        // no permute, no index vectors, no b0/b1, and NO validity masks: patterns
        // run unmasked into 16 (reused) acc slots. Pair-folding (c, c+4) with two
        // sb_shufi64x2 rebuilds the accumulators of the virtual block at V in the
        // standard convention (F lane l <-> diag V+c-3+l, slot q = 3-c), so all
        // boundary clipping is done by the standard store masks: top garbage
        // (diag >= rn) falls outside lomask/wmask, sub-band garbage (V = 0) is
        // discarded by the harvest shifts, and hi(diag -1) is the stmask case.
        sb_vec flo[8], fhi[8], av;
        for(int k = 0; k < 8; ++k){ flo[k] = sb_zero(); fhi[k] = sb_zero(); }
        if(P > 0){
            const sb_limb *bptr = b + V;
            for(int64_t j = an; j > 0; j -= 8, bptr += 8){
                av = j >= 8 ? sb_load(a + (j-8))
                            : sb_load(a + (j-8), 0xFFu << (8-j));
                #define mm_vpat(c) { \
                    const sb_vec _bv = sb_splat_load_m(sb_splat_load(bptr, c), 0x0F, bptr, c + 4); \
                    flo[c] = sb_madd52lo(flo[c], av, _bv); \
                    fhi[c] = sb_madd52hi(fhi[c], av, _bv); \
                }
                switch((int)P){
                    case 4: mm_vpat(3);
                    case 3: mm_vpat(2);
                    case 2: mm_vpat(1);
                    case 1: mm_vpat(0);
                }
                #undef mm_vpat
            }
        }
        for(int c = 0; c < 4; ++c){
            F_lo[3-c] = sb_add(sb_shufi64x2(flo[c], flo[c+4], 0x44), sb_shufi64x2(flo[c], flo[c+4], 0xEE));
            F_hi[3-c] = sb_add(sb_shufi64x2(fhi[c], fhi[c+4], 0x44), sb_shufi64x2(fhi[c], fhi[c+4], 0xEE));
        }
    }
    #undef mm_acc
    F_hi[1] = sb_add(F_hi[1], sb_alignr64(F_hi[0], last_hi[0], 7)); // standard retire
    if(V){ // W_{V-8}: standard {last, F} combine; masked iff it is the final window
        const int64_t n1 = rn - (V - 8) < 8 ? rn - (V - 8) : 8; // in [5, 8]
        sb_vec L1 = sb_add(sb_add(last_lo[0], sb_alignr64(F_lo[1], last_lo[1], 1)),
                      sb_add(sb_alignr64(F_lo[2], last_lo[2], 2), sb_alignr64(F_lo[3], last_lo[3], 3)));
        sb_vec H1 = sb_maskz(stmask, sb_add(sb_add(last_hi[1], sb_alignr64(F_hi[2], last_hi[2], 1)),
                                    sb_alignr64(F_hi[3], last_hi[3], 2)));
        stmask = 0xFF;
        if(n1 == 8) sb_store(r + V - 8, sb_add(L1, H1));
        else{
            sb_vec res = sb_add(H1, L1, (uint8_t)((1u << n1) - 1), H1);
            sb_store(r + V - 8, res, (uint8_t)((2u << n1) - 1));
        }
    }
    if(rn >= V){ // partial window at V: n2 in [0,4] outputs, top limb at lane n2
        const int64_t n2 = rn - V;
        sb_vec L2 = sb_add(sb_add(F_lo[0], sb_lane_rsh(F_lo[1], 1)),
                      sb_add(sb_lane_rsh(F_lo[2], 2), sb_lane_rsh(F_lo[3], 3)));
        sb_vec H2 = sb_maskz(stmask, sb_add(sb_add(F_hi[1], sb_lane_rsh(F_hi[2], 1)), sb_lane_rsh(F_hi[3], 2)));
        sb_vec res = sb_add(H2, L2, (uint8_t)((1u << n2) - 1), H2);
        sb_store(r + V, res, (uint8_t)((2u << n2) - 1));
    }
}


}
