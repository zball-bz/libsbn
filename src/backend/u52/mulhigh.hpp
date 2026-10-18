/* High half of an n x n u64-limb product through u52 digits (2026-10-09).
 *
 * Contract: n output limbs r with 0 <= floor(A*B / 2^q) - r <= 1, q = 64n.
 *
 * Truncation column. With digit base beta = 2^52 write q = 52t + s (0 <= s < 52).
 * The kernel keeps every digit product a_i b_j with i+j >= K and the high half
 * of each product with i+j = K-1 (it lands in column K); the kept sum S is
 * exact. As for the scalar kernels (tools/generate_high_half.py), the exact
 * geometric sum with n-1 replaced by K bounds the omission by
 *   E <= (2K-1) beta^K - 2K beta^(K-1) + 1 < (2K-1) 2^(52K).
 * E < 2^q holds for K = t when 2t-1 <= 2^s, and always for K = t-1
 * ((2t-3) 2^(52t-52) < 2^q). Then A*B - 2^q < S <= A*B and r = floor(S/2^q).
 * K = t-1 is needed at n = 9, 13, 22, 35, 48, 213, 226, 230, ... (q mod 52 small);
 * n = 231..233 take K = t, 234 (q = 52*288, the validated range end) K = t-1.
 *
 * u52_blocks_high is mul_u52_blocks (mul.hpp, imported from
 * libsbn/include/sbn/detail/u52/mul.h) for a square shape, starting at output
 * vector kv-1 = floor(K/8)-1 with zero spill buckets: vectors below kv are
 * neither computed nor written, lanes < K%8 of vector kv are cleared. Its first
 * diagonal i0 = kv-1 holds only full input blocks (kv <= (d-1)/8) and only feeds
 * vector kv: a product a_{8p+l} b_{8q+r} lands its low half in column
 * 8*i0 + l + r and its high half one higher, so columns >= K = 8kv + k receive
 * no low half from rows r <= k and only the high half of row k (l = 7). That
 * diagonal runs rows k+1..7 in full plus row k's high half: 15-2k IFMAs per
 * block pair instead of 16. k is fixed per call (one fallthrough entry).
 * fixed::sqr_blocks_high restricts sqr_blocks (mul_small.hpp) the same way, and
 * fixed::mul_rows_window_high the window rows for 17..24 digits (kv = 2); their
 * column sums >= K, hence their words, equal u52_blocks_high's.
 *
 * u52_emit_high: one rolling pass from vector kv, canonize -> pack52 -> 64-byte
 * output windows. The packed stream starts at bit 416kv, so bit q is the whole
 * byte o = 8n - 52kv. A window starting x bytes into a 52-byte packed block
 * spans two blocks for x <= 40 and three for x = 44, 48 (e.g. n = 32: 4+52+8
 * bytes); the last three packed blocks stay in registers. The last digit vector
 * is read with the mask of the actual digit count, later vectors are zero.
 */
#pragma once
namespace sbn::v3::u52 {

// K for q = 64n (see above), by the integer inequality
static inline uint64_t u52_high_column(uint64_t n){
    const uint64_t q = 64 * n, t = q / 52, s = q % 52;
    return 2 * t - 1 <= (1ull << s) ? t : t - 1;
}

#define high_accum(a, b, ind) {\
    bx = sb_splat_load(b, ind); \
    n[ind] = sb_madd52lo(n[ind], a, bx); \
    if constexpr (SPLIT) h[ind+1] = sb_madd52hi(h[ind+1], a, bx); \
    else n[ind+1] = sb_madd52hi(n[ind+1], a, bx); \
}
#define high_only(a, b, ind) {\
    bx = sb_splat_load(b, ind); \
    if constexpr (SPLIT) h[ind+1] = sb_madd52hi(h[ind+1], a, bx); \
    else n[ind+1] = sb_madd52hi(n[ind+1], a, bx); \
}
#define high_merge { \
    if constexpr (SPLIT) for(int k_ = 1; k_ < 9; ++k_){ n[k_] = sb_add(n[k_], h[k_]); h[k_] = sb_zero(); } \
}

// r[kv..) = digit columns >= K of the d x d digit product {a, b} (plus the high
// halves of column K-1), redundant non-negative lanes; d >= 9, 1 <= kv <= (d-1)/8,
// k = K % 8. Writes vectors kv..ceil(2d/8)-1, the last one masked to 2d digits.
template<bool SPLIT>
__attribute__((noinline)) static void u52_blocks_high(sb_pvec r, sb_cpvec a, sb_cpvec b, uint64_t d, uint64_t kv, int k){
    sb_vec n[9], m[9], h[9], bx, ax;
    sb_cpvec bp, ap;
    for(int t = 0; t < 9; ++t){ n[t] = sb_zero(); h[t] = sb_zero(); m[t] = sb_zero(); }
    uint64_t an = d - 1, bn = d - 1;
    const uint64_t au = an >> 3, bu = bn >> 3, su = au + bu;
    an &= 7, bn &= 7;
    uint64_t i = kv - 1, j;
    // first diagonal i0 = kv-1: only its spill into vector kv is kept
    for(bp = b + i, ap = a, j = i + 1; j; ++ap, --bp, --j){
        ax = sb_load(ap);
        switch(k){                                // rows k+1..7: both halves
        case 0: high_accum(ax, bp, 1); [[fallthrough]];
        case 1: high_accum(ax, bp, 2); [[fallthrough]];
        case 2: high_accum(ax, bp, 3); [[fallthrough]];
        case 3: high_accum(ax, bp, 4); [[fallthrough]];
        case 4: high_accum(ax, bp, 5); [[fallthrough]];
        case 5: high_accum(ax, bp, 6); [[fallthrough]];
        case 6: high_accum(ax, bp, 7); [[fallthrough]];
        default: break;
        }
        switch(k){                                // row k: high half only
        case 0: high_only(ax, bp, 0); break;
        case 1: high_only(ax, bp, 1); break;
        case 2: high_only(ax, bp, 2); break;
        case 3: high_only(ax, bp, 3); break;
        case 4: high_only(ax, bp, 4); break;
        case 5: high_only(ax, bp, 5); break;
        case 6: high_only(ax, bp, 6); break;
        default: high_only(ax, bp, 7); break;
        }
    }
    high_merge;
    harvest_z;
    n[0] = sb_zero();
    for(++i; i < su; ++i){
        if(i >= bu){
            bp = b + bu; ap = a + (i - bu);
            j = i >= au ? su - i : bu + 1;
            ax = sb_load(ap);
            switch(bn){
                non_sat:
                for(; j; ++ap, --bp, --j){
                    ax = sb_load(ap);
                    case 7: high_accum(ax, bp, 7);
                    case 6: high_accum(ax, bp, 6);
                    case 5: high_accum(ax, bp, 5);
                    case 4: high_accum(ax, bp, 4);
                    case 3: high_accum(ax, bp, 3);
                    case 2: high_accum(ax, bp, 2);
                    case 1: high_accum(ax, bp, 1);
                    case 0: high_accum(ax, bp, 0);
                }
            }
            if(i >= au){
                ax = sb_load(bp);
                switch(an){
                    case 7: high_accum(ax, ap, 7);
                    case 6: high_accum(ax, ap, 6);
                    case 5: high_accum(ax, ap, 5);
                    case 4: high_accum(ax, ap, 4);
                    case 3: high_accum(ax, ap, 3);
                    case 2: high_accum(ax, ap, 2);
                    case 1: high_accum(ax, ap, 1);
                    case 0: high_accum(ax, ap, 0);
                }
            }
        }else{
            j = i + 1; bp = b + i; ap = a;
            ax = sb_load(a);
            goto non_sat;
        }
        high_merge;
        harvest;
        if(i == kv) n[0] = sb_maskz((__mmask8)(0xFFu << k), n[0]);
        sb_store(r + i, n[0]);
        n[0] = sb_zero();
    }
    high_merge;
    sb_pvec rt = r + su;
    const uint64_t an_tail = an, bn_tail = bn;
    const uint64_t tail_limbs = an_tail + bn_tail + 2;
    if(an_tail < bn_tail){ bn = an_tail; bp = a + au; ax = sb_load(b + bu, u52_limb_mask(bn_tail + 1)); }
    else{ bp = b + bu; ax = sb_load(a + au, u52_limb_mask(an_tail + 1)); }
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
    if(su == kv) n[0] = sb_maskz((__mmask8)(0xFFu << k), n[0]);
    if(tail_limbs <= 8) sb_store(rt, n[0], u52_limb_mask(tail_limbs));
    else{ sb_store(rt, n[0]); sb_store(rt + 1, n[8], u52_limb_mask(tail_limbs - 8)); }
}
#undef high_accum
#undef high_only
#undef high_merge

namespace fixed {
// mul_rows_window<3,3,C> (mul_small.hpp) for 17 <= d <= 24 with kv = 2: output
// stationary, every product's halves land in their own output vector, so dropping
// the pairs q+m < 2 leaves outputs 2..5 with the full column sums; lanes below k of
// output 2 are cleared. Columns >= K equal u52_blocks_high's. A quarter fewer IFMAs
// and 16 accumulators (C = 2) next to a's three vectors, no spill.
template<int C>
__attribute__((noinline)) static void mul_rows_window_high(uint64_t *r, const uint64_t *a, const uint64_t *b, int64_t d, int k) {
    constexpr int VX = 3, MY = 3, KN = VX + MY, KF = 2;  // outputs KF..KN-1 only
    const int64_t an = d, bn = d, tx = an - 8 * (VX - 1), ty = bn - 8 * (MY - 1), total = an + bn;
    const uint64_t *bl = b + 8 * (MY - 1);
    sb_vec X[VX];
    unroll<VX - 1>([&](auto q) __attribute__((always_inline)) { X[q] = sb_load(a + 8 * q); });
    X[VX - 1] = tail_vec(a, an, tx);
    sb_vec L[C][KN], H[C][KN];
    unroll<C>([&](auto c) __attribute__((always_inline)) {
        unroll<KN>([&](auto j) __attribute__((always_inline)) { L[c][j] = sb_zero(); H[c][j] = sb_zero(); });
    });
    unroll<8>([&](auto S) __attribute__((always_inline)) {
        constexpr int s = S, c = s % C;
        auto row = [&](auto m, sb_vec W, auto j) __attribute__((always_inline)) {
            if constexpr (decltype(j)::value >= KF) {
                L[c][j] = sb_madd52lo(L[c][j], W, sb_set1_64(b[8 * m + s]));
                if constexpr (s > 0 || m > 0) H[c][j] = sb_madd52hi(H[c][j], W, sb_set1_64(b[8 * m + s - 1]));
            }
        };
        auto lastrow = [&](sb_vec W, auto j) __attribute__((always_inline)) {   // s <= ty
            if constexpr (decltype(j)::value >= KF) {
                if(s < ty) L[c][j] = sb_madd52lo(L[c][j], W, sb_set1_64(bl[s]));
                H[c][j] = sb_madd52hi(H[c][j], W, sb_set1_64(bl[s - 1]));
            }
        };
        const bool top = tx >= 9 - s, last = ty >= s;
        unroll<VX>([&](auto q) __attribute__((always_inline)) {
            const sb_vec W = q == 0 ? window<8 - s>(X[0], sb_zero()) : window<8 - s>(X[q], X[q > 0 ? q - 1 : 0]);
            unroll<MY - 1>([&](auto m) __attribute__((always_inline)) { row(m, W, std::integral_constant<int, q + m>{}); });
            if(last) lastrow(W, std::integral_constant<int, q + MY - 1>{});
        });
        if constexpr (s > 0) if(top) {
            const sb_vec W = window<8 - s>(sb_zero(), X[VX - 1]);
            unroll<MY - 1>([&](auto m) __attribute__((always_inline)) { row(m, W, std::integral_constant<int, VX + m>{}); });
            if(last) lastrow(W, std::integral_constant<int, VX + MY - 1>{});
        }
    });
    if(ty == 8){        // hi of row bn-1 lands in row block MY at class 0
        const sb_vec y = sb_set1_64(b[bn - 1]);
        unroll<VX>([&](auto q) __attribute__((always_inline)) { H[0][q + MY] = sb_madd52hi(H[0][q + MY], X[q], y); });
    }
    unroll<KN - KF>([&](auto i) __attribute__((always_inline)) {
        constexpr int j = i + KF;
        sb_vec o = sb_add(L[0][j], H[0][j]);
        unroll<C - 1>([&](auto c) __attribute__((always_inline)) { o = sb_add(o, sb_add(L[c + 1][j], H[c + 1][j])); });
        if constexpr (j == KF) o = sb_maskz((__mmask8)(0xFFu << k), o);
        const int64_t left = total - 8 * j;
        if(left >= 8) sb_store(r + 8 * j, o); else if(left > 0) store_end(r, total, left, o);
    });
}

// The square of d digits restricted the same way (d >= 9, 1 <= kv <= (d-1)/8, k = K % 8):
// sqr_blocks (mul_small.hpp) from output vector kv-1 with zero spills. Diagonal kv-1 holds
// whole blocks only; its pairs and its diagonal block (kv-1 even) run rows k+1..7 in full and
// row k's high half before and after the doubling, then spill into vector kv, whose lanes
// below k are cleared. Column sums equal those of the d x d product, so the kept sum and the
// result are those of u52_blocks_high(a, a).
__attribute__((noinline)) static void sqr_blocks_high(uint64_t *r, const uint64_t *a, int64_t D, int64_t kv, int k) {
    const int64_t V = (D + 7) / 8, t = D - 8 * (V - 1), total = 2 * D;
    sb_vec n[9], m[9];
    for(int i = 0; i < 9; ++i){ n[i] = sb_zero(); m[i] = sb_zero(); }
    auto row = [&](auto T, sb_vec ax, const uint64_t *bq) __attribute__((always_inline)) {
        const sb_vec bx = sb_set1_64(bq[T]);
        n[T] = sb_madd52lo(n[T], ax, bx);
        n[T + 1] = sb_madd52hi(n[T + 1], ax, bx);
    };
    auto high_row = [&](auto T, sb_vec ax, const uint64_t *bq) __attribute__((always_inline)) {
        n[T + 1] = sb_madd52hi(n[T + 1], ax, sb_set1_64(bq[T]));
    };
    auto rows8 = [&](sb_vec ax, const uint64_t *bq) __attribute__((always_inline)) {
        unroll<8>([&](auto T) __attribute__((always_inline)) { row(T, ax, bq); });
    };
    using std::integral_constant;
    auto rows_top = [&](sb_vec ax, const uint64_t *bq) __attribute__((always_inline)) {
        switch(t){                                    // the top block's t digits as rows
        case 8: row(integral_constant<int, 7>{}, ax, bq); [[fallthrough]];
        case 7: row(integral_constant<int, 6>{}, ax, bq); [[fallthrough]];
        case 6: row(integral_constant<int, 5>{}, ax, bq); [[fallthrough]];
        case 5: row(integral_constant<int, 4>{}, ax, bq); [[fallthrough]];
        case 4: row(integral_constant<int, 3>{}, ax, bq); [[fallthrough]];
        case 3: row(integral_constant<int, 2>{}, ax, bq); [[fallthrough]];
        case 2: row(integral_constant<int, 1>{}, ax, bq); [[fallthrough]];
        default: row(integral_constant<int, 0>{}, ax, bq);
        }
    };
    auto rows_trim = [&](sb_vec ax, const uint64_t *bq) __attribute__((always_inline)) {
        switch(k){                                    // rows k+1..7: both halves
        case 0: row(integral_constant<int, 1>{}, ax, bq); [[fallthrough]];
        case 1: row(integral_constant<int, 2>{}, ax, bq); [[fallthrough]];
        case 2: row(integral_constant<int, 3>{}, ax, bq); [[fallthrough]];
        case 3: row(integral_constant<int, 4>{}, ax, bq); [[fallthrough]];
        case 4: row(integral_constant<int, 5>{}, ax, bq); [[fallthrough]];
        case 5: row(integral_constant<int, 6>{}, ax, bq); [[fallthrough]];
        case 6: row(integral_constant<int, 7>{}, ax, bq); [[fallthrough]];
        default: break;
        }
        switch(k){                                    // row k: high half only
        case 0: high_row(integral_constant<int, 0>{}, ax, bq); break;
        case 1: high_row(integral_constant<int, 1>{}, ax, bq); break;
        case 2: high_row(integral_constant<int, 2>{}, ax, bq); break;
        case 3: high_row(integral_constant<int, 3>{}, ax, bq); break;
        case 4: high_row(integral_constant<int, 4>{}, ax, bq); break;
        case 5: high_row(integral_constant<int, 5>{}, ax, bq); break;
        case 6: high_row(integral_constant<int, 6>{}, ax, bq); break;
        default: high_row(integral_constant<int, 7>{}, ax, bq); break;
        }
    };
    {
        const int64_t i = kv - 1;
        for(int64_t p = i; 2 * p > i; --p) rows_trim(sb_load(a + 8 * p), a + 8 * (i - p));
        for(int j = 0; j < 9; ++j) n[j] = sb_add(n[j], n[j]);
        if(!(i & 1)){ const uint64_t *bq = a + 8 * (i / 2); rows_trim(sb_load(bq), bq); }
        for(int j = 1; j < 9; ++j) m[j] = n[j];
        for(int j = 0; j < 9; ++j) n[j] = sb_zero();
    }
    for(int64_t i = kv; i < 2 * V - 1; ++i){
        for(int64_t p = i < V - 2 ? i : V - 2; 2 * p > i; --p) rows8(sb_load(a + 8 * p), a + 8 * (i - p));
        if(i >= V - 1 && 2 * (V - 1) > i) rows_top(sb_load(a + 8 * (i - V + 1)), a + 8 * (V - 1));
        for(int j = 0; j < 9; ++j) n[j] = sb_add(n[j], n[j]);
        if(!(i & 1)){
            const int64_t d = i / 2;
            const uint64_t *bq = a + 8 * d;
            if(d < V - 1) rows8(sb_load(bq), bq);
            else rows_top(tail_vec(a, D, t), bq);
        }
        sb_vec o = sb_add(n[0], m[8]);
        unroll<7>([&](auto I) __attribute__((always_inline)) {
            constexpr int j = I + 1;
            o = sb_add(o, sb_alignr64(n[j], m[j], 8 - j));
        });
        for(int j = 1; j < 9; ++j) m[j] = n[j];
        for(int j = 0; j < 9; ++j) n[j] = sb_zero();
        if(i == kv) o = sb_maskz((__mmask8)(0xFFu << k), o);
        const int64_t left = total - 8 * i;
        if(left >= 8) sb_store(r + 8 * i, o); else store_end(r, total, left, o);
    }
    sb_vec o = m[8];
    unroll<7>([&](auto I) __attribute__((always_inline)) {
        constexpr int j = I + 1;
        o = sb_add(o, sb_alignr64(sb_zero(), m[j], 8 - j));
    });
    const int64_t left = total - 8 * (2 * V - 1);
    if(left > 0) store_end(r, total, left, o);
}
} // namespace fixed

static inline void u52_emit_high(uint64_t *out, sb_cpvec R, uint64_t kv, uint64_t digits, uint64_t n){
    u52_emit_window(out,R,kv,digits,n,n);
}


} // namespace sbn::v3::u52
