/* Fixed-shape u52 product kernels (2026-10-08).
 *
 * Every kernel writes the same redundant digits as the scanning basecase it
 * replaces: digit p = sum lo(a_i*b_j) over i+j=p plus sum hi(a_i*b_j) over
 * i+j+1=p. Only the schedule differs, so the lane bound and every caller's
 * class accounting are unchanged.
 *
 * Register shapes are template parameters (rows of mul1, vector counts of the
 * row kernels); digit counts stay runtime. A runtime row count switched inside
 * the chunk loop made clang peel the first chunk and spill the carried
 * diagonals to the stack (36-48 stack accesses in the shipped object).
 *
 * Operands are never read before their first digit. The top partial vector of an
 * operand or of the product is read and written masked at its aligned position:
 * the operands were usually just written by conversion or Toom evaluation with
 * whole-vector stores, and the product is read right back by encoding. Full
 * vectors ending at the last digit (one vpermq) avoid the 300-650 cycle assist a
 * masked access takes next to an inaccessible page, but cannot be forwarded from
 * those stores: +14 cycles end to end at 8x8 u64 limbs. The kernels only see u52
 * conversion buffers, which keep a slack vector. */
#pragma once
#include <utility>
namespace sbn::v3::u52 {
namespace fixed {

// clamped to [0, 8] (bzhi reads only the low byte of its count)
static inline __mmask8 lanes_below(int64_t left) {
    return (__mmask8)_bzhi_u32(0xffu, (uint32_t)(left < 0 ? 0 : left > 8 ? 8 : left));
}

// lanes s..7 of lo followed by lanes 0..s-1 of hi: the window starting s digits into lo
template<int S> static inline sb_vec window(sb_vec hi, sb_vec lo) {
    if constexpr (S == 0) return lo;
    else if constexpr (S == 8) return hi;
    else return sb_alignr64(hi, lo, S);
}

// mask of the lanes l >= x
static constexpr int lanes_from(int x) { return x <= 0 ? 0xff : x >= 8 ? 0 : (0xff << x) & 0xff; }

// f(integral_constant<int, i>) for i = 0..N-1
template<int N, class F> __attribute__((always_inline)) static inline void unroll(F &&f) {
    [&]<int... I>(std::integer_sequence<int, I...>) __attribute__((always_inline)) {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

// lanes 0..t-1 = x[n-t..n-1], the rest zero (1 <= t <= 8): the top partial vector of x
static inline sb_vec tail_vec(const uint64_t *x, int64_t n, int64_t t) {
    return sb_load(x + n - t, lanes_below(t));
}

// o holds the last `left` digits (1..7) of an n-digit result at r+n-left
static inline void store_end(uint64_t *r, int64_t n, int64_t left, sb_vec o) {
    sb_store(r + n - left, o, lanes_below(left));
}

// Operand scanning, one valignq per row: row t's lo shares its digit offset
// with row t-1's hi, so both are rotated into place together; m[t] carries the
// lanes shifted out of each chunk into the next.
template<int BN>
__attribute__((noinline)) static void mul1_scan(uint64_t *r, const uint64_t *a, uint64_t an, const uint64_t *b) {
    sb_vec bx[BN], m[BN];
    for(int t = 0; t < BN; ++t){ bx[t] = sb_set1_64(b[t]); m[t] = sb_zero(); }
    auto chunk = [&](sb_vec ax) __attribute__((always_inline)) {
        sb_vec n0 = sb_zero(), n1 = sb_zero(), lo0 = sb_zero();
        [&]<int... T>(std::integer_sequence<int, T...>) __attribute__((always_inline)) {
            ([&] {
                constexpr int t = BN - 1 - T;           // rows from the top down
                n1 = sb_madd52hi(n1, ax, bx[t]);        // row t's hi joins row t+1's lo
                n0 = sb_add(n0, window<7 - t>(n1, m[t]));
                m[t] = n1;
                if constexpr (t) n1 = sb_madd52lo(sb_zero(), ax, bx[t]);
                else lo0 = sb_madd52lo(sb_zero(), ax, bx[0]);
            }(), ...);
        }(std::make_integer_sequence<int, BN>{});
        return sb_add(n0, lo0);
    };
    const uint64_t full = an >> 3, tail = an & 7;
    for(uint64_t c = 0; c < full; ++c, a += 8, r += 8) sb_store(r, chunk(sb_load(a)));
    if(tail){
        const sb_vec o = chunk(sb_load(a, lanes_below(tail)));
        if(tail + BN <= 8){ sb_store(r, o, lanes_below(tail + BN)); return; }
        sb_store(r, o); r += 8;
    }
    // flush: m[t] holds digits at offset t+1 whose top lanes were never stored
    sb_vec n0 = m[BN - 1];
    if constexpr (BN < 8) n0 = window<8 - BN>(sb_zero(), n0);
    [&]<int... T>(std::integer_sequence<int, T...>) __attribute__((always_inline)) {
        ([&] { n0 = sb_add(n0, window<7 - T>(sb_zero(), m[T])); }(), ...);
    }(std::make_integer_sequence<int, BN - 1>{});
    sb_store(r, n0, lanes_below(tail ? tail + BN - 8 : BN));
}

// Output stationary: chunk c of r accumulates lo(W(8c-t)*b_t) + hi(W(8c-t-1)*b_t)
// directly in the IFMA accumulators, W(o) = a digits o..o+7 built by one valignq
// from the two most recent chunk loads. No additions, copies or zeroing per row;
// cheaper than scanning only for long a and mid-size BN (measured 1-14%).
template<int BN>
__attribute__((noinline)) static void mul1_stationary(uint64_t *r, const uint64_t *a, uint64_t an, const uint64_t *b) {
    sb_vec bx[BN];
    for(int t = 0; t < BN; ++t) bx[t] = sb_set1_64(b[t]);
    auto chunk = [&](sb_vec prev, sb_vec cur) __attribute__((always_inline)) {
        sb_vec lo = sb_zero(), hi = sb_zero();
        [&]<int... T>(std::integer_sequence<int, T...>) __attribute__((always_inline)) {
            ([&] {
                lo = sb_madd52lo(lo, window<8 - T>(cur, prev), bx[T]);
                hi = sb_madd52hi(hi, window<7 - T>(cur, prev), bx[T]);
            }(), ...);
        }(std::make_integer_sequence<int, BN>{});
        return sb_add(lo, hi);
    };
    const int64_t on = (int64_t)an + BN, full = (int64_t)an / 8, tail = (int64_t)an % 8;
    sb_vec prev = sb_zero();
    int64_t c = 0;
    for(; c < full; ++c){
        const sb_vec cur = sb_load(a + 8 * c);
        sb_store(r + 8 * c, chunk(prev, cur));
        prev = cur;
    }
    // past the full chunks a is its top partial vector, then zeros (an >= 80)
    for(; 8 * c < on; ++c){
        const sb_vec cur = c == full && tail ? tail_vec(a, an, tail) : sb_zero();
        const sb_vec o = chunk(prev, cur);
        if(on - 8 * c >= 8) sb_store(r + 8 * c, o); else store_end(r, on, on - 8 * c, o);
        prev = cur;
    }
}

// 1 <= bn <= 8, an >= bn: output stationary wins only from about 80 digits of a, and only
// for 3..6 rows; scanning is faster for 7..8 rows and for every shorter a. The planner
// sends u52 no b shorter than 2 limbs (3 digits); 1..2 rows come from short Karatsuba
// differences and unnormalized inputs.
static inline void mul1(uint64_t *r, const uint64_t *a, uint64_t an, const uint64_t *b, uint64_t bn) {
    switch(bn){
    case 1: return mul1_scan<1>(r, a, an, b);
    case 2: return mul1_scan<2>(r, a, an, b);
    case 3: return an >= 80 ? mul1_stationary<3>(r, a, an, b) : mul1_scan<3>(r, a, an, b);
    case 4: return an >= 80 ? mul1_stationary<4>(r, a, an, b) : mul1_scan<4>(r, a, an, b);
    case 5: return an >= 80 ? mul1_stationary<5>(r, a, an, b) : mul1_scan<5>(r, a, an, b);
    case 6: return an >= 80 ? mul1_stationary<6>(r, a, an, b) : mul1_scan<6>(r, a, an, b);
    case 7: return mul1_scan<7>(r, a, an, b);
    default: return mul1_scan<8>(r, a, an, b);
    }
}

// Row kernels, 9 <= bn <= 24: output stationary with no harvest. Output vector k
// collects the windows W(8q-s) of a (s = 0..7 the alignment class, one valignq each)
// against row 8m+s of b (lo) and row 8m+s-1 (hi), k = q+m, so each window serves every
// row block m of b. Rows -1 and >= bn contribute nothing. Class s accumulates into set
// s % C, which halves the IFMA chains where the registers allow it.

// an <= 24: all of a in registers, classes unrolled, VX = ceil(an/8), MY = ceil(bn/8).
// Per class one branch on a's top window (tx >= 9-s) and one on b's last row block (s <= ty).
template<int VX, int MY, int C>
__attribute__((noinline)) static void mul_rows_window(uint64_t *r, const uint64_t *a, int64_t an, const uint64_t *b, int64_t bn) {
    constexpr int K = VX + MY;
    const int64_t tx = an - 8 * (VX - 1), ty = bn - 8 * (MY - 1), total = an + bn;
    const uint64_t *bl = b + 8 * (MY - 1);
    sb_vec X[VX];
    unroll<VX - 1>([&](auto q) __attribute__((always_inline)) { X[q] = sb_load(a + 8 * q); });
    X[VX - 1] = tail_vec(a, an, tx);
    sb_vec L[C][K], H[C][K];
    unroll<C>([&](auto c) __attribute__((always_inline)) {
        unroll<K>([&](auto k) __attribute__((always_inline)) { L[c][k] = sb_zero(); H[c][k] = sb_zero(); });
    });
    unroll<8>([&](auto S) __attribute__((always_inline)) {
        constexpr int s = S, c = s % C;
        auto row = [&](auto m, sb_vec W, auto k) __attribute__((always_inline)) {
            L[c][k] = sb_madd52lo(L[c][k], W, sb_set1_64(b[8 * m + s]));
            if constexpr (s > 0 || m > 0) H[c][k] = sb_madd52hi(H[c][k], W, sb_set1_64(b[8 * m + s - 1]));
        };
        auto lastrow = [&](sb_vec W, auto k) __attribute__((always_inline)) {   // s <= ty
            if(s < ty) L[c][k] = sb_madd52lo(L[c][k], W, sb_set1_64(bl[s]));
            H[c][k] = sb_madd52hi(H[c][k], W, sb_set1_64(bl[s - 1]));
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
    if(ty == 8){        // hi of row bn-1 lands in row block MY at class 0, where W(8q) = X[q]
        const sb_vec y = sb_set1_64(b[bn - 1]);
        unroll<VX>([&](auto q) __attribute__((always_inline)) { H[0][q + MY] = sb_madd52hi(H[0][q + MY], X[q], y); });
    }
    unroll<K>([&](auto k) __attribute__((always_inline)) {
        sb_vec o = sb_add(L[0][k], H[0][k]);
        unroll<C - 1>([&](auto c) __attribute__((always_inline)) { o = sb_add(o, sb_add(L[c + 1][k], H[c + 1][k])); });
        const int64_t left = total - 8 * k;
        if(left >= 8) sb_store(r + 8 * k, o); else if(left > 0) store_end(r, total, left, o);
    });
}

// acc + lo/hi(w * broadcast(*row)) with the row as an embedded broadcast. Through the
// intrinsics clang merges a row that two IFMAs share (lo of class s, hi of class s+1)
// into a separate broadcast instruction, and specialises accumulators it knows are zero.
static inline sb_vec madd52lo_row(sb_vec acc, sb_vec w, const uint64_t *row) {
    __asm__("vpmadd52luq %2%{1to8%}, %1, %0" : "+v"(acc) : "v"(w), "m"(*row));
    return acc;
}
static inline sb_vec madd52hi_row(sb_vec acc, sb_vec w, const uint64_t *row) {
    __asm__("vpmadd52huq %2%{1to8%}, %1, %0" : "+v"(acc) : "v"(w), "m"(*row));
    return acc;
}

// Any an: step q takes a's vector q, so outputs q..q+MY-1 are live and output q retires
// after the step; the windows that reach past a follow the last step. Per step, two
// entries with targets constant in a call: classes ty-1..0 take the full row blocks and
// b's last block, classes ty..7 only the full blocks, so every window is built once; class
// ty of the last block (the hi of row bn-1) is one runtime-shift window. b's rows are
// embedded broadcasts.
template<int MY, int C>
__attribute__((noinline)) static void mul_rows_stream(uint64_t *r, const uint64_t *a, int64_t an, const uint64_t *b, int64_t bn) {
    using std::integral_constant;
    const int64_t VX = (an + 7) / 8, tx = an - 8 * (VX - 1), ty = bn - 8 * (MY - 1), total = an + bn;
    sb_vec L[C][MY], H[C][MY];                  // outputs q .. q+MY-1
    unroll<C>([&](auto c) __attribute__((always_inline)) {
        unroll<MY>([&](auto i) __attribute__((always_inline)) { L[c][i] = sb_zero(); H[c][i] = sb_zero(); });
    });
    const sb_vec idxt = sb_add(sb_setr_64(0, 1, 2, 3, 4, 5, 6, 7), sb_set1_64(8 - ty)), ylast = sb_set1_64(b[bn - 1]);
    sb_vec prev = sb_zero();
    // class s of window W: full row blocks, and b's last block when `last`
    auto rows = [&](auto S, sb_vec W, const uint64_t *lo, const uint64_t *hi, auto last) __attribute__((always_inline)) {
        constexpr int s = S, c = s % C;
        unroll<MY - 1>([&](auto m) __attribute__((always_inline)) {
            L[c][m] = madd52lo_row(L[c][m], W, lo + 8 * m + s);
            if constexpr (s > 0 || m > 0) H[c][m] = madd52hi_row(H[c][m], W, hi + 8 * m + s - 1);
        });
        if constexpr (last) {
            L[c][MY - 1] = madd52lo_row(L[c][MY - 1], W, lo + 8 * (MY - 1) + s);
            H[c][MY - 1] = madd52hi_row(H[c][MY - 1], W, hi + 8 * (MY - 1) + s - 1);
        }
    };
    auto step = [&](sb_vec cur) __attribute__((always_inline)) {
        const uint64_t *lo = b, *hi = b;
        auto both = [&](auto S) __attribute__((always_inline)) { rows(S, window<8 - S>(cur, prev), lo, hi, std::true_type{}); };
        auto full = [&](auto S) __attribute__((always_inline)) { rows(S, window<8 - S>(cur, prev), lo, hi, std::false_type{}); };
        switch(ty){
        case 1: goto b0;
        case 2: goto b1;
        case 3: goto b2;
        case 4: goto b3;
        case 5: goto b4;
        case 6: goto b5;
        case 7: goto b6;
        default: goto b7;
        }
        b7: both(integral_constant<int, 7>{});
        b6: both(integral_constant<int, 6>{});
        b5: both(integral_constant<int, 5>{});
        b4: both(integral_constant<int, 4>{});
        b3: both(integral_constant<int, 3>{});
        b2: both(integral_constant<int, 2>{});
        b1: both(integral_constant<int, 1>{});
        b0: both(integral_constant<int, 0>{});
        switch(ty){
        case 1: goto f1;
        case 2: goto f2;
        case 3: goto f3;
        case 4: goto f4;
        case 5: goto f5;
        case 6: goto f6;
        case 7: goto f7;
        default: goto fe;
        }
        f1: full(integral_constant<int, 1>{});
        f2: full(integral_constant<int, 2>{});
        f3: full(integral_constant<int, 3>{});
        f4: full(integral_constant<int, 4>{});
        f5: full(integral_constant<int, 5>{});
        f6: full(integral_constant<int, 6>{});
        f7: full(integral_constant<int, 7>{});
        fe:
        H[0][MY - 1] = sb_madd52hi(H[0][MY - 1], sb_perm64x2(prev, idxt, cur), ylast);
    };
    // windows that reach past a: W(8VX-s) = window<8-s>(0, a's top vector), non-empty for
    // s >= 9-tx, so the classes 9-tx..7 are entered once at 9-tx
    auto flush = [&](auto S) __attribute__((always_inline)) {
        constexpr int s = S, c = s % C;
        const sb_vec W = window<8 - s>(sb_zero(), prev);
        unroll<MY - 1>([&](auto m) __attribute__((always_inline)) {
            L[c][m] = madd52lo_row(L[c][m], W, b + 8 * m + s);
            H[c][m] = madd52hi_row(H[c][m], W, b + 8 * m + s - 1);
        });
        if(s <= ty){
            const uint64_t *bl = b + 8 * (MY - 1);
            if(s < ty) L[c][MY - 1] = madd52lo_row(L[c][MY - 1], W, bl + s);
            H[c][MY - 1] = madd52hi_row(H[c][MY - 1], W, bl + s - 1);
        }
    };
    auto retire = [&]() __attribute__((always_inline)) {     // output q, then shift q+1.. down
        sb_vec o = sb_add(L[0][0], H[0][0]);
        unroll<C - 1>([&](auto c) __attribute__((always_inline)) { o = sb_add(o, sb_add(L[c + 1][0], H[c + 1][0])); });
        unroll<C>([&](auto c) __attribute__((always_inline)) {
            unroll<MY - 1>([&](auto i) __attribute__((always_inline)) { L[c][i] = L[c][i + 1]; H[c][i] = H[c][i + 1]; });
            L[c][MY - 1] = sb_zero(); H[c][MY - 1] = sb_zero();
        });
        return o;
    };
    auto emit = [&](int64_t k) __attribute__((always_inline)) {
        const sb_vec o = retire();
        const int64_t left = total - 8 * k;
        if(left >= 8) sb_store(r + 8 * k, o); else if(left > 0) store_end(r, total, left, o);
    };
    const sb_vec tail = tail_vec(a, an, tx);
    for(int64_t q = 0; q < VX; ++q){            // output q < VX is always a full vector
        const sb_vec cur = q + 1 < VX ? sb_load(a + 8 * q) : tail;
        step(cur);
        prev = cur;
        sb_store(r + 8 * q, retire());
    }
    switch(tx){
    case 8: flush(integral_constant<int, 1>{}); [[fallthrough]];
    case 7: flush(integral_constant<int, 2>{}); [[fallthrough]];
    case 6: flush(integral_constant<int, 3>{}); [[fallthrough]];
    case 5: flush(integral_constant<int, 4>{}); [[fallthrough]];
    case 4: flush(integral_constant<int, 5>{}); [[fallthrough]];
    case 3: flush(integral_constant<int, 6>{}); [[fallthrough]];
    case 2: flush(integral_constant<int, 7>{}); [[fallthrough]];
    default: break;
    }
    if(ty == 8) H[0][MY - 1] = sb_madd52hi(H[0][MY - 1], prev, ylast);   // class 8: W(8VX-8) = a's top vector
    emit(VX);
    unroll<MY - 1>([&](auto i) __attribute__((always_inline)) { emit(VX + 1 + i); });
}

// Shapes for the row kernels (an >= bn >= 9): window kernels while a fits in registers
// (3 vectors), streaming beyond up to 21 rows. From 22 rows the stream's window ramp (about
// 7 windows with empty lanes per call) costs as much as the block kernel's harvest
// (measured 0.97..1.06 of it, worst at an = 1..3 mod 8). bn > 16 rows keep one accumulator
// set in the window kernel: two would need 24 accumulators next to the windows and spill.
static inline bool rows_shape(uint64_t an, uint64_t bn) { return bn <= 24 && (an <= 24 || bn <= 21); }
static inline void mul_rows(uint64_t *r, const uint64_t *a, uint64_t an, const uint64_t *b, uint64_t bn) {
    const int64_t x = (int64_t)an, y = (int64_t)bn;
    if(x <= 16) return mul_rows_window<2, 2, 2>(r, a, x, b, y);
    if(x <= 24) return y <= 16 ? mul_rows_window<3, 2, 2>(r, a, x, b, y) : mul_rows_window<3, 3, 1>(r, a, x, b, y);
    return y <= 16 ? mul_rows_stream<2, 2>(r, a, x, b, y) : mul_rows_stream<3, 2>(r, a, x, b, y);
}

// Square of D digits, 8(V-1) < D <= 8V, parity split: output block k (digits 16k..16k+15)
// is E (digits 2m) and O (digits 2m+1), m = 8k+l, lane-wise products of windows W(o) =
// a[o..o+7] (zero outside a), so there are no broadcasts and no masks. Along e = e*..1:
//   P_e    a_{m-e} a_{m+e}:     lo -> E, hi -> O
//   Q_e-1  a_{m-e+1} a_{m+e}:   lo -> O
//   Q'_e-1 a_{m-e} a_{m+e-1}:   hi -> E
// every unordered pair once; then E = 2E + lo(a_m^2), O = 2O + hi(a_m^2). The steps are one
// straight-line sequence entered at the last step e*(k) with a pair in the block, so D stays
// runtime within V; each step builds two windows and hands two to the next. Step e
// accumulates into set e % C.
template<int V, int C>
__attribute__((noinline)) static void sqr_rows_eo(uint64_t *r, const uint64_t *a, int64_t D) {
    constexpr int EM = 4 * V, NA = V + 7;                 // A[q+3] holds a's vector q
    const int64_t t = D - 8 * (V - 1), total = 2 * D;
    sb_vec A[NA];
    unroll<NA>([&](auto q) __attribute__((always_inline)) { A[q] = sb_zero(); });
    unroll<V - 1>([&](auto q) __attribute__((always_inline)) { A[q + 3] = sb_load(a + 8 * q); });
    A[V + 2] = tail_vec(a, D, t);
    auto W = [&](auto O) __attribute__((always_inline)) {            // digits O..O+7, O >= -24
        constexpr int q = (O + 24) / 8 - 3, s = O - 8 * q;
        return window<s>(A[q + 4], A[q + 3]);
    };
    const sb_vec ilo = sb_setr_64(0, 8, 1, 9, 2, 10, 3, 11), ihi = sb_setr_64(4, 12, 5, 13, 6, 14, 7, 15);
    using std::integral_constant;
    unroll<V>([&](auto K) __attribute__((always_inline)) {
        constexpr int k = K, b = 8 * k;
        sb_vec E[C], O[C];
        unroll<C>([&](auto c) __attribute__((always_inline)) { E[c] = sb_zero(); O[c] = sb_zero(); });
        // step e takes l = W(b-e), h = W(b+e) from the step before (or its entry) and leaves
        // W(b-e+1), W(b+e-1) for the next: two windows per step
        sb_vec l, h;
        auto step = [&](auto e) __attribute__((always_inline)) {
            constexpr int c = e % C;
            const sb_vec l1 = W(integral_constant<int, b - e + 1>{}), h1 = W(integral_constant<int, b + e - 1>{});
            E[c] = sb_madd52lo(E[c], l, h);
            O[c] = sb_madd52hi(O[c], l, h);
            O[c] = sb_madd52lo(O[c], l1, h);
            E[c] = sb_madd52hi(E[c], l, h1);
            l = l1;
            h = h1;
        };
        // e*: last step with a pair in the block, the max over m in [8k, min(8k+7, D-1)] of
        // max(min(m+1, D-1-m), min(m, D-m)), which rises to D/2 and falls: hi+1, D-8k or D/2
        int64_t es = D - b < D / 2 ? D - b : D / 2;
        if(b + 8 < es) es = b + 8;
        static_assert(EM <= 20);
        switch(es){                                   // es in 1..EM: steps es, es-1, .., 1
        case 20: if constexpr (20 <= EM) { l = W(integral_constant<int, b - 20>{}); h = W(integral_constant<int, b + 20>{}); goto s20; } break;
        case 19: if constexpr (19 <= EM) { l = W(integral_constant<int, b - 19>{}); h = W(integral_constant<int, b + 19>{}); goto s19; } break;
        case 18: if constexpr (18 <= EM) { l = W(integral_constant<int, b - 18>{}); h = W(integral_constant<int, b + 18>{}); goto s18; } break;
        case 17: if constexpr (17 <= EM) { l = W(integral_constant<int, b - 17>{}); h = W(integral_constant<int, b + 17>{}); goto s17; } break;
        case 16: if constexpr (16 <= EM) { l = W(integral_constant<int, b - 16>{}); h = W(integral_constant<int, b + 16>{}); goto s16; } break;
        case 15: if constexpr (15 <= EM) { l = W(integral_constant<int, b - 15>{}); h = W(integral_constant<int, b + 15>{}); goto s15; } break;
        case 14: if constexpr (14 <= EM) { l = W(integral_constant<int, b - 14>{}); h = W(integral_constant<int, b + 14>{}); goto s14; } break;
        case 13: if constexpr (13 <= EM) { l = W(integral_constant<int, b - 13>{}); h = W(integral_constant<int, b + 13>{}); goto s13; } break;
        case 12: if constexpr (12 <= EM) { l = W(integral_constant<int, b - 12>{}); h = W(integral_constant<int, b + 12>{}); goto s12; } break;
        case 11: if constexpr (11 <= EM) { l = W(integral_constant<int, b - 11>{}); h = W(integral_constant<int, b + 11>{}); goto s11; } break;
        case 10: if constexpr (10 <= EM) { l = W(integral_constant<int, b - 10>{}); h = W(integral_constant<int, b + 10>{}); goto s10; } break;
        case 9: if constexpr (9 <= EM) { l = W(integral_constant<int, b - 9>{}); h = W(integral_constant<int, b + 9>{}); goto s9; } break;
        case 8: if constexpr (8 <= EM) { l = W(integral_constant<int, b - 8>{}); h = W(integral_constant<int, b + 8>{}); goto s8; } break;
        case 7: if constexpr (7 <= EM) { l = W(integral_constant<int, b - 7>{}); h = W(integral_constant<int, b + 7>{}); goto s7; } break;
        case 6: if constexpr (6 <= EM) { l = W(integral_constant<int, b - 6>{}); h = W(integral_constant<int, b + 6>{}); goto s6; } break;
        case 5: if constexpr (5 <= EM) { l = W(integral_constant<int, b - 5>{}); h = W(integral_constant<int, b + 5>{}); goto s5; } break;
        case 4: if constexpr (4 <= EM) { l = W(integral_constant<int, b - 4>{}); h = W(integral_constant<int, b + 4>{}); goto s4; } break;
        case 3: if constexpr (3 <= EM) { l = W(integral_constant<int, b - 3>{}); h = W(integral_constant<int, b + 3>{}); goto s3; } break;
        case 2: if constexpr (2 <= EM) { l = W(integral_constant<int, b - 2>{}); h = W(integral_constant<int, b + 2>{}); goto s2; } break;
        case 1: if constexpr (1 <= EM) { l = W(integral_constant<int, b - 1>{}); h = W(integral_constant<int, b + 1>{}); goto s1; } break;
        default: goto done;
        }
        s20: if constexpr (20 <= EM) step(integral_constant<int, 20>{});
        s19: if constexpr (19 <= EM) step(integral_constant<int, 19>{});
        s18: if constexpr (18 <= EM) step(integral_constant<int, 18>{});
        s17: if constexpr (17 <= EM) step(integral_constant<int, 17>{});
        s16: if constexpr (16 <= EM) step(integral_constant<int, 16>{});
        s15: if constexpr (15 <= EM) step(integral_constant<int, 15>{});
        s14: if constexpr (14 <= EM) step(integral_constant<int, 14>{});
        s13: if constexpr (13 <= EM) step(integral_constant<int, 13>{});
        s12: if constexpr (12 <= EM) step(integral_constant<int, 12>{});
        s11: if constexpr (11 <= EM) step(integral_constant<int, 11>{});
        s10: if constexpr (10 <= EM) step(integral_constant<int, 10>{});
        s9: if constexpr (9 <= EM) step(integral_constant<int, 9>{});
        s8: if constexpr (8 <= EM) step(integral_constant<int, 8>{});
        s7: if constexpr (7 <= EM) step(integral_constant<int, 7>{});
        s6: if constexpr (6 <= EM) step(integral_constant<int, 6>{});
        s5: if constexpr (5 <= EM) step(integral_constant<int, 5>{});
        s4: if constexpr (4 <= EM) step(integral_constant<int, 4>{});
        s3: if constexpr (3 <= EM) step(integral_constant<int, 3>{});
        s2: if constexpr (2 <= EM) step(integral_constant<int, 2>{});
        s1: if constexpr (1 <= EM) step(integral_constant<int, 1>{});
        done:
        const sb_vec c = A[k + 3];                       // the squares
        sb_vec e = E[0], o = O[0];
        unroll<C - 1>([&](auto i) __attribute__((always_inline)) { e = sb_add(e, E[i + 1]); o = sb_add(o, O[i + 1]); });
        e = sb_madd52lo(sb_add(e, e), c, c);
        o = sb_madd52hi(sb_add(o, o), c, c);
        const sb_vec x = sb_perm64x2(e, ilo, o), y = sb_perm64x2(e, ihi, o);
        const int64_t left0 = total - 16 * k, left1 = left0 - 8;
        if(left0 >= 8) sb_store(r + 16 * k, x); else store_end(r, total, left0, x);
        if(left1 >= 8) sb_store(r + 16 * k + 8, y); else if(left1 > 0) store_end(r, total, left1, y);
    });
}

// Square of D digits, D > 8, block diagonal: output vector k collects the block pairs
// p > q, p+q = k, once (a's vector p in lanes, the 8 digits of block q broadcast) into nine
// offset buckets, doubles them, then adds the diagonal block (k/2, k/2) as a full self
// product, which is 2*triangle + squares. The pairs with a's partial top block (t digits)
// broadcast its t digits against block q in lanes (2t IFMAs, entered at t); the top
// diagonal block runs its t rows once per call. Buckets are harvested as in mul_u52_blocks.
__attribute__((noinline)) static void sqr_blocks(uint64_t *r, const uint64_t *a, int64_t D) {
    const int64_t V = (D + 7) / 8, t = D - 8 * (V - 1), total = 2 * D;
    sb_vec n[9], m[9];
    for(int i = 0; i < 9; ++i){ n[i] = sb_zero(); m[i] = sb_zero(); }
    auto row = [&](auto T, sb_vec ax, const uint64_t *bq) __attribute__((always_inline)) {
        const sb_vec bx = sb_set1_64(bq[T]);
        n[T] = sb_madd52lo(n[T], ax, bx);
        n[T + 1] = sb_madd52hi(n[T + 1], ax, bx);
    };
    auto rows8 = [&](sb_vec ax, const uint64_t *bq) __attribute__((always_inline)) {
        unroll<8>([&](auto T) __attribute__((always_inline)) { row(T, ax, bq); });
    };
    using std::integral_constant;
    for(int64_t k = 0; k < 2 * V - 1; ++k){
        // off-diagonal pairs (p, q = k-p), p > q: whole blocks first, then the pair on the top
        // block, so its switch never sees buckets known to be zero (clang would rematerialize
        // zeros per entry and spill)
        for(int64_t p = k < V - 2 ? k : V - 2; 2 * p > k; --p) rows8(sb_load(a + 8 * p), a + 8 * (k - p));
        if(k >= V - 1 && 2 * (V - 1) > k){
            // the top block has t digits: its digits are the rows, block q the lanes (2t IFMAs)
            const sb_vec ax = sb_load(a + 8 * (k - V + 1));
            const uint64_t *bq = a + 8 * (V - 1);
            switch(t){
            case 8: row(integral_constant<int, 7>{}, ax, bq); [[fallthrough]];
            case 7: row(integral_constant<int, 6>{}, ax, bq); [[fallthrough]];
            case 6: row(integral_constant<int, 5>{}, ax, bq); [[fallthrough]];
            case 5: row(integral_constant<int, 4>{}, ax, bq); [[fallthrough]];
            case 4: row(integral_constant<int, 3>{}, ax, bq); [[fallthrough]];
            case 3: row(integral_constant<int, 2>{}, ax, bq); [[fallthrough]];
            case 2: row(integral_constant<int, 1>{}, ax, bq); [[fallthrough]];
            default: row(integral_constant<int, 0>{}, ax, bq);
            }
        }
        for(int i = 0; i < 9; ++i) n[i] = sb_add(n[i], n[i]);
        if(!(k & 1)){
            const int64_t d = k / 2;
            const uint64_t *bq = a + 8 * d;
            if(d < V - 1){
                const sb_vec ax = sb_load(bq);
                unroll<8>([&](auto T) __attribute__((always_inline)) { row(T, ax, bq); });
            } else {
                const sb_vec top = tail_vec(a, D, t);           // a's partial top vector
                switch(t){
                case 8: row(integral_constant<int, 7>{}, top, bq); [[fallthrough]];
                case 7: row(integral_constant<int, 6>{}, top, bq); [[fallthrough]];
                case 6: row(integral_constant<int, 5>{}, top, bq); [[fallthrough]];
                case 5: row(integral_constant<int, 4>{}, top, bq); [[fallthrough]];
                case 4: row(integral_constant<int, 3>{}, top, bq); [[fallthrough]];
                case 3: row(integral_constant<int, 2>{}, top, bq); [[fallthrough]];
                case 2: row(integral_constant<int, 1>{}, top, bq); [[fallthrough]];
                default: row(integral_constant<int, 0>{}, top, bq);
                }
            }
        }
        // harvest output vector k: bucket i holds digit offset i from 8k
        sb_vec o = sb_add(n[0], m[8]);
        unroll<7>([&](auto I) __attribute__((always_inline)) {
            constexpr int i = I + 1;
            o = sb_add(o, sb_alignr64(n[i], m[i], 8 - i));
        });
        for(int i = 1; i < 9; ++i) m[i] = n[i];
        for(int i = 0; i < 9; ++i) n[i] = sb_zero();
        const int64_t left = total - 8 * k;                       // short only for k = 2V-2, t < 4
        if(left >= 8) sb_store(r + 8 * k, o); else store_end(r, total, left, o);
    }
    // output vector 2V-1: the spills of the top diagonal
    sb_vec o = m[8];
    unroll<7>([&](auto I) __attribute__((always_inline)) {
        constexpr int i = I + 1;
        o = sb_add(o, sb_alignr64(sb_zero(), m[i], 8 - i));
    });
    const int64_t left = total - 8 * (2 * V - 1);
    if(left > 0) store_end(r, total, left, o);
}

} // namespace fixed
} // namespace sbn::v3::u52
