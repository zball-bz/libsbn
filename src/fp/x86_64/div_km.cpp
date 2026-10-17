// fp kernel division, IFMA side: sbn3f_div_km, k = 22..512 quotient limbs (called from sbn3f_div, src/fp/x86_64/div.S,
// with the operands' full limb arrays; contract: src/fp/kernel.hpp). Karp-Markstein on the reciprocal's Newton levels:
// the reciprocal X_h of the divisor to h = floor(n/2) + 1 digits (the seed and the level engine of sbn3f_inv_qv,
// src/fp/x86_64/inv_qv.cpp, run for n = h), the quotient's top Q_h = floor(A X_h) - 1 (a high product), then one
// quotient level h -> n: the residual's window A - D Q_h (a middle product; by one Karatsuba step from 18 window vectors),
// G its top w + 1 digits, Q_n = Q_h beta^w + X_h G / beta^h (a high product), z = floor(Q_n / 2^sh). Cost (Zen 5, hot):
// about 1.4 x sbn3f_inv at the same k above 64 limbs (the quotient's products are n^2 / 16 vector rows for any h; the
// reciprocal's last level n^2 * 3/64). Stack: 86.5 KB at most (k <= 512, measured), the engine's Karatsuba scratch included.
// Research record: experiments/results/fp_divsqrt512_2026-10-18/div (DIV512, bh session fdiv-k512).
//
// Digits (beta = 2^52). y = |b| 2^ld / B^eb in [1/2, 1) (ld = lzcnt of b's top limb), alpha = |a| / B^ea in [B^-1, 1).
// e_top = ea - eb + 1, k = e_top - L(2), T = 64k - 64 + ld: |a / b| B^-L(2) = alpha 2^T / y =: v, and |z - v| < 4 is the
// contract. n = ceil((T + 1) / 52), sh = 52n - T in [1, 52]. D = floor(y beta^(n+1)) (n + 1 digits, from b's whole
// array), Y = D / beta^(n+1): y - beta^(-n-1) < Y <= y. A = the digits of alpha down to beta^-NA, NA = n + 9 (from a's
// whole array): A[m] has weight beta^(m - NA). Every approximation below refers to Y; alpha beta^n (1/Y - 1/y) < 4.02 /
// beta, in z units below 2^-53.
//
// X_h. sbn3f_inv_qv's sequence for n = h on D_h = D's top h + 1 digits (Y_h = D_h / beta^(h+1), Y - beta^(-h-1) < Y_h <=
// Y), with a guarded last level: the table's chain with u = 0 for h >= 20, the seed's m0 = ceil((h + 1) / 2) -> h for h
// 10..17, 6 -> 10 -> h for h 18, 19 (every level guarded, its base the seed or a guarded level). Then beta^h / Y_h - X_h is
// in (-2(h + 1.01) / beta, 3.01) (inv_qv.cpp's header), so eps_x = beta^h / Y - X_h in (-(2h + 6.04) / beta, 3.01).
// Q_h. Ã = floor(alpha beta^d) (A's top d = h + 3 digits); S = the halves of Ã X_h landing in columns >= K = d - 2 (the
// others, < 2d beta^K, dropped); Q_h = floor(S / beta^d) - 1. With q_h* = alpha beta^h / Y: Q_h < q_h* (S / beta^d <=
// alpha (beta^h / Y + (2h + 6.04) / beta) and (2h + 6.04) / beta < 1), and q_h* - Q_h = delta < 3.01 + 2 + tiny < 5.03
// (beta^(h-d) X_h's share 2 beta^-3, the dropped halves 2d / beta^2).
// The quotient level (w = n - h <= h - 1; NV = ceil((w + 3) / 8) vectors of window, e = 8 NV - w - 1 in 2..9, columns
// c0 = h + 1 - e .. n + 1). R = alpha beta^(n+1+h) - D Q_h = Y delta beta^(n+1) is in [0, 5.03 beta^(n+1)). MP: the
// halves of D Q_h in columns c0 .. n + 1 (the dropped ones below c0 sum Def < 2(h + 1) beta^c0), or from NV = KARA_NV its
// full columns c0 .. n + 1 (kara_mp; the dropped products, those below c0, sum Def < (h + 1) beta^(c0 + 1)), canonical mod
// beta^(8 NV); G = (A's digits in those columns - MP) mod beta^(8 NV) = (R - A_low + Def) / beta^c0 exactly (A_low < beta^c0,
// alpha's part below c0; the value lies in [0, beta^(8 NV))). G_hi = G's digits e .. 8 NV - 1 (columns h + 1 .. n + 1):
// with g = R / beta^(h+1) = Y delta beta^w, G_hi in (g - 1 - beta^-e, g + (h + 1) beta^(1-e)].
// Correction: the halves of X_h G_hi in columns h - e .. n (dropped below: < 2(w + 1) beta^(h-e)), canonical; C = its
// digits at h .. n (no carry out: X_h G_hi < 2 beta^h 5.04 beta^w < beta^(n+1)): X_h G_hi / beta^h - C in [0, 1 + 2(w + 1)
// / beta^e). Q_n = Q_h beta^w + C.
// Error: alpha beta^n / Y - Q_n = (g - G_hi) / Y + eps_x G_hi beta^-h + (X_h G_hi beta^-h - C) lies in (-2.01 (h + 1) /
// beta - tiny, 2.01 (1 + beta^-2) + 15.2 / beta + 1 + tiny) (|eps_x G_hi| beta^-h < 3.01 * 5.04 beta^(w-h), w <= h - 1):
// |Q_n - alpha beta^n / Y| < 3.02, so z = floor(Q_n / 2^sh) lies in (v - 3.02 / 2^sh - 1 - 2^-53, v + 2^-53]: |z - v| < 2.52.
// z overflows k limbs only when Q_n >= 2^(64k + sh), i.e. ld = 63 and Q_n's top digit >= 2: then v > 2^(64k) - 3.6, and
// z = B^k - 1 is within the contract.
#include "fp/kernel.hpp"
#include "inv_sched.hpp"
#ifndef KM_FORCE_EXACT
#define KM_FORCE_EXACT 0                                        // test builds: the canon's serial pass every time
#endif
#ifndef KM_KARA_CHECK
#define KM_KARA_CHECK 0                                         // test builds: the Karatsuba window against the plain one
#endif
#ifndef KARA_NV
#define KARA_NV 18                                              // the middle product by Karatsuba from this window width
#endif
#ifndef KM_OUTL
#define KM_OUTL [[gnu::noinline]]                               // the helpers called more than once: one copy
#endif
#ifndef KM_ST
#define KM_ST(i)                                                // stage stamps (diagnostic builds only)
#endif
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

using sbn::v3::fp::fx;
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_seed(uint64_t *, const fx *, int64_t, int64_t) noexcept;
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_chain(uint64_t *X, const uint64_t *D, uint64_t *G, unsigned n,
                                                                   const unsigned *lv, unsigned L) noexcept;
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_fix(uint64_t *X, unsigned lo, unsigned n);
extern "C" __attribute__((visibility("hidden"))) void sbn3f_div_km(uint64_t *z, const uint64_t *a, uint64_t na, const uint64_t *b,
                                                                uint64_t nb, uint64_t k) noexcept;

namespace {
using V = __m512i;
struct ch32_t { uint32_t c[297]; };
constexpr ch32_t make_ch32() {
    ch32_t r{};
    for (unsigned t = 20; t <= 316; ++t) r.c[t - 20] = uint32_t(qv_chain[t][0]);
    return r;
}
constexpr bool ch32_fits() {
    for (unsigned t = 20; t <= 316; ++t) if (qv_chain[t][0] >> 32) return false;
    return true;
}
static_assert(ch32_fits(), "a schedule entry does not fit 32 bits");
alignas(64) constexpr ch32_t ch32 = make_ch32();               // the reciprocal's chains with a guarded last level (u = 0)
constexpr uint64_t M52 = (uint64_t(1) << 52) - 1;
constexpr unsigned NMAX = 632;                                  // digits: n <= 631 (k <= 512)
constexpr unsigned PAD = 64;                                    // zero digits on both sides of a vector operand (wprod)

// D[i] = bits [base + 52i, +52) of a[0 .. an) (zero outside), i < 8 nblk (inv_qv.cpp's decode_desc)
KM_OUTL void decode_desc(uint64_t *D, unsigned nblk, const uint64_t *a, int64_t an, int64_t base) {
    alignas(64) static constexpr uint8_t p0[64] = {0,  1,  2,  3,  4,  5,  6,  7,  6,  7,  8,  9,  10, 11, 12, 13,
                                                   13, 14, 15, 16, 17, 18, 19, 20, 19, 20, 21, 22, 23, 24, 25, 26,
                                                   26, 27, 28, 29, 30, 31, 32, 33, 32, 33, 34, 35, 36, 37, 38, 39,
                                                   39, 40, 41, 42, 43, 44, 45, 46, 45, 46, 47, 48, 49, 50, 51, 52};
    const unsigned sh = unsigned(base & 7);
    V perm = _mm512_load_si512(p0);
    if (sh >= 4) perm = _mm512_add_epi8(perm, _mm512_set_epi64(0x0101010101010101ll, 0, 0x0101010101010101ll, 0,
                                                               0x0101010101010101ll, 0, 0x0101010101010101ll, 0));
    const V shifts = _mm512_set_epi64((sh + 4) & 7, sh, (sh + 4) & 7, sh, (sh + 4) & 7, sh, (sh + 4) & 7, sh);
    const V M = _mm512_set1_epi64(int64_t(M52));
    const char *ab = reinterpret_cast<const char *>(a);
    const int64_t end = 8 * an;
    int64_t byte = (base >> 3) + 52 * int64_t(nblk - 1);
    int b = int(nblk) - 1;
    auto put = [&](V v) __attribute__((always_inline)) {
        _mm512_storeu_si512(D + 8 * b, _mm512_and_si512(_mm512_srlv_epi64(_mm512_permutexvar_epi8(perm, v), shifts), M));
    };
#pragma clang loop unroll(disable)
    for (; b >= 0 && byte + 64 > end; --b, byte -= 52) {
        uint64_t m = byte >= end ? 0 : _bzhi_u64(~uint64_t(0), unsigned(end - byte));
        if (byte < 0) m &= ~_bzhi_u64(~uint64_t(0), unsigned(-byte < 64 ? -byte : 64));
        put(_mm512_maskz_loadu_epi8(m, ab + byte));
    }
#pragma clang loop unroll(disable)
    for (; b >= 0 && byte >= 0; --b, byte -= 52) put(_mm512_loadu_si512(ab + byte));
#pragma clang loop unroll(disable)
    for (; b >= 0; --b, byte -= 52) put(_mm512_maskz_loadu_epi8(~_bzhi_u64(~uint64_t(0), unsigned(-byte < 64 ? -byte : 64)), ab + byte));
}
// z[0 .. k) = bits [sh + 64i, +64) of the digits X (zero padded above) (inv_qv.cpp's encode)
inline void encode(uint64_t *z, const uint64_t *X, uint64_t sh, uint64_t k) {
    const V iota = _mm512_set_epi64(448, 384, 320, 256, 192, 128, 64, 0), c1261 = _mm512_set1_epi64(1261);
    const V one = _mm512_set1_epi64(1), c52 = _mm512_set1_epi64(52), c104 = _mm512_set1_epi64(104);
    uint64_t q0 = sh / 52, r0 = sh % 52;
    auto block = [&]() __attribute__((always_inline)) {
        const V t = _mm512_add_epi64(_mm512_set1_epi64(int64_t(r0)), iota);
        const V q = _mm512_srli_epi64(_mm512_mul_epu32(t, c1261), 16);
        const V r = _mm512_sub_epi64(t, _mm512_add_epi64(_mm512_slli_epi64(q, 5), _mm512_add_epi64(_mm512_slli_epi64(q, 4), _mm512_slli_epi64(q, 2))));
        const V lo = _mm512_loadu_si512(X + q0), hi = _mm512_loadu_si512(X + q0 + 8);
        const V q1 = _mm512_add_epi64(q, one), q2 = _mm512_add_epi64(q1, one);
        V v = _mm512_srlv_epi64(_mm512_permutex2var_epi64(lo, q, hi), r);
        v = _mm512_or_si512(v, _mm512_sllv_epi64(_mm512_permutex2var_epi64(lo, q1, hi), _mm512_sub_epi64(c52, r)));
        return _mm512_or_si512(v, _mm512_sllv_epi64(_mm512_permutex2var_epi64(lo, q2, hi), _mm512_sub_epi64(c104, r)));
    };
    uint64_t i = 0;
    for (; i + 8 <= k; i += 8) {
        _mm512_storeu_si512(z + i, block());
        r0 += 44;
        q0 += 9;
        if (r0 >= 52) { r0 -= 52; ++q0; }
    }
    if (i < k) _mm512_mask_storeu_epi64(z + i, __mmask8((1u << (k - i)) - 1), block());
}

// Window products. The lanes of the 8 columns c' .. c' + 7 of a vector: the sum over rows j of lo(U[j] V[c' - j]) and
// hi(U[j] V[c' - 1 - j]) (the halves landing in column c'). Rows ascend; one load per row and vector: row j's hi takes
// row j + 1's load (V[c' - (j + 1)]), so a vector's sums lag one row in hi (the tail adds the last row's).
// wseg: rows a .. b - 1 on vectors 0 .. M - 1 of a group (Vc = V + the group's first column). Segments of at most NB
// vectors take two rows per iteration into a second set of accumulators (lo2 / hi2): each sum is then updated once per two
// rows, which hides the IFMA latency when few vectors are active.
#ifndef WG_NB
#define WG_NB 3
#endif
constexpr int NB = WG_NB;
template <int M, int GS>
[[gnu::always_inline]] inline void wseg(V (&lo)[GS], V (&hi)[GS], V (&lo2)[NB], V (&hi2)[NB], V &up, const uint64_t *U,
                                        const uint64_t *Vc, int64_t a, int64_t b) {
    int64_t j = a;
    if constexpr (M <= NB) {
#pragma clang loop unroll(disable)
        for (; j + 1 < b; j += 2) {
            const V u0 = _mm512_set1_epi64(int64_t(U[j])), u1 = _mm512_set1_epi64(int64_t(U[j + 1]));
            for (int g = 0; g < M; ++g) {
                const V x0 = _mm512_loadu_si512(Vc + 8 * g - j), x1 = _mm512_loadu_si512(Vc + 8 * g - j - 1);
                lo[g] = _mm512_madd52lo_epu64(lo[g], u0, x0);
                hi[g] = _mm512_madd52hi_epu64(hi[g], up, x0);
                lo2[g] = _mm512_madd52lo_epu64(lo2[g], u1, x1);
                hi2[g] = _mm512_madd52hi_epu64(hi2[g], u0, x1);
            }
            up = u1;
        }
    }
#pragma clang loop unroll(disable)
    for (; j < b; ++j) {
        const V u = _mm512_set1_epi64(int64_t(U[j]));
        for (int g = 0; g < M; ++g) {
            const V x = _mm512_loadu_si512(Vc + 8 * g - j);
            lo[g] = _mm512_madd52lo_epu64(lo[g], u, x);
            hi[g] = _mm512_madd52hi_epu64(hi[g], up, x);
        }
        up = u;
    }
}
template <int GS, int M = 1>
[[gnu::always_inline]] inline void wsegs(V (&lo)[GS], V (&hi)[GS], V (&lo2)[NB], V (&hi2)[NB], V &up, const uint64_t *U,
                                         const uint64_t *Vc, const int64_t *s) {
    lo[M - 1] = _mm512_setzero_si512();                         // a vector's sums start with its first segment (fewer live
    hi[M - 1] = _mm512_setzero_si512();                         // registers in the narrow segments)
    wseg<M, GS>(lo, hi, lo2, hi2, up, U, Vc, s[M - 1], s[M]);
    if constexpr (M < GS) wsegs<GS, M + 1>(lo, hi, lo2, hi2, up, U, Vc, s);
}
// A group of 8 vectors (columns c .. c + 63; vectors from gs on are not computed nor stored), rows j0 .. j1: vector g takes
// the rows from s_g = max(j0, c + 8g - vn) (below, its lanes read only V's zero padding above vn: nothing is skipped);
// s_g grows with g, so the rows run as a staircase of segments [s_{M-1}, s_M) on vectors 0 .. M - 1. up starts at zero:
// a vector's first hi product (the row before s_g) multiplies V[vn] = 0, or there is no such row (s_g = j0 = 0).
// m0: the full-column window (lane 0 without the hi halves of column c - 1).
constexpr int GS = 8;
[[gnu::always_inline]] inline void wgroup(uint64_t *L, const uint64_t *U, const uint64_t *Vp, int64_t c, int64_t vn, int64_t j0, int64_t j1,
                                          int64_t gs, bool m0) {
    V lo[GS], hi[GS], lo2[NB], hi2[NB], up = _mm512_setzero_si512();
    for (int g = 0; g < NB; ++g) { lo2[g] = _mm512_setzero_si512(); hi2[g] = _mm512_setzero_si512(); }
    int64_t s[GS + 1];
    for (int g = 0; g < GS; ++g) {
        const int64_t t = c + 8 * g - vn;
        s[g] = g >= gs ? j1 + 1 : t < j0 ? j0 : t > j1 + 1 ? j1 + 1 : t;
    }
    s[GS] = j1 + 1;
    wsegs<GS>(lo, hi, lo2, hi2, up, U, Vp + c, s);
    for (int g = 0; g < GS; ++g)                                // the last row's hi (vectors that took rows: s_g <= j1)
        if (s[g] <= j1) hi[g] = _mm512_madd52hi_epu64(hi[g], up, _mm512_loadu_si512(Vp + c + 8 * g - 1 - j1));
    if (m0) {
        hi[0] = _mm512_maskz_mov_epi64(0xfe, hi[0]);
        hi2[0] = _mm512_maskz_mov_epi64(0xfe, hi2[0]);
    }
    for (int g = 0; g < GS; ++g) {
        if (g >= gs) break;
        V t = _mm512_add_epi64(lo[g], hi[g]);
        if (g < NB) t = _mm512_add_epi64(t, _mm512_add_epi64(lo2[g < NB ? g : 0], hi2[g < NB ? g : 0]));
        _mm512_storeu_si512(L + 8 * g, t);
    }
}
// L[0 .. 8 nv) = the lanes of columns cl .. cl + 8 nv - 1 of the product U[0 .. un) x V[0 .. vn) (every half landing there;
// a group of 8 vectors takes rows max(0, c - vn) .. min(un - 1, c + 63), each vector from its own first row). V is read on
// [cl - min(un, cl + 8 nv) - 1, vn + 8) at most and must be zero there outside [0, vn). Lane sums: at most 2 un halves
// below 2^52 (un < 2^10: below 2^62). m0: the full-column window.
[[gnu::noinline]] void wprod(uint64_t *L, const uint64_t *U, int64_t un, const uint64_t *Vp, int64_t vn, int64_t cl, int64_t nv, bool m0 = false) {
#pragma clang loop unroll(disable)
    for (int64_t v = 0; v < nv; v += GS) {
        const int64_t gs = nv - v < GS ? nv - v : GS, c = cl + 8 * v;
        const int64_t j0 = c - vn > 0 ? c - vn : 0, j1 = c + 8 * gs - 1 < un - 1 ? c + 8 * gs - 1 : un - 1;
        wgroup(L + 8 * v, U, Vp, c, vn, j0, j1, gs, m0 && v == 0);
    }
}
// Signed lanes L[0 .. m) (m a multiple of 8, |lane| < 2^62) to the canonical digits of sum L[i] beta^i mod beta^m; returns
// the (signed) carry out of the top. Two vector rounds (lane & M52 plus the arithmetic >> 52 of the lane below: after the
// first every lane is in [-2^10, M52 + 2^10], after the second in [-1, M52 + 1]), then the serial pass only when a lane is
// left outside [0, M52] (cold: a run of M52 digits meets a carry, or of zero digits a borrow).
[[gnu::noinline, gnu::cold]] int64_t canon_exact(uint64_t *L, int64_t m, int64_t c) {
    for (int64_t i = 0; i < m; ++i) {
        const int64_t t = int64_t(L[i]) + c;
        L[i] = uint64_t(t) & M52;
        c = t >> 52;
    }
    return c;
}
// The first round adds t to lane p (|t| < 2^52; in registers: a scalar store into the lanes just written by the window
// product would stall the vector loads here).
KM_OUTL int64_t canon_s(uint64_t *L, int64_t m, int64_t p = 0, int64_t t = 0) {
    const V M = _mm512_set1_epi64(int64_t(M52)), idx = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 15);
    int64_t cout = 0;
    V bad = _mm512_setzero_si512();
    for (int r = 0; r < 2; ++r) {
        V ph = _mm512_setzero_si512();                       // the previous vector's carries (lane 7 moves to lane 0)
#pragma clang loop unroll(disable)
        for (int64_t i = 0; i < m; i += 8) {
            V x = _mm512_loadu_si512(L + i);
            if (r == 0 && uint64_t(p - i) < 8) x = _mm512_mask_add_epi64(x, __mmask8(1u << (p - i)), x, _mm512_set1_epi64(t));
            const V hcar = _mm512_srai_epi64(x, 52);
            const V y = _mm512_add_epi64(_mm512_and_si512(x, M), _mm512_permutex2var_epi64(hcar, idx, ph));
            _mm512_storeu_si512(L + i, y);
            if (r) bad = _mm512_or_si512(bad, _mm512_srai_epi64(y, 52));
            ph = hcar;
        }
        cout += _mm256_extract_epi64(_mm512_extracti64x4_epi64(ph, 1), 3);
    }
    if (KM_FORCE_EXACT || __builtin_expect(_mm512_test_epi64_mask(bad, bad) != 0, 0)) cout += canon_exact(L, m, 0);
    return cout;
}
// Digits with carries: x[0 .. m) signed lanes (|x| < 2^62, m a multiple of 8) to y[i] in [0, beta) and the carries
// kap[i] with x[i] = y[i] + beta kap[i] - kap[i-1] (kap[-1] = 0): two vector rounds (kap = the sum of their carries),
// then a serial pass over the lanes left outside [0, M52] (cold).
[[gnu::noinline, gnu::cold]] void digits_exact(uint64_t *y, int64_t *kap, int64_t m) {
    int64_t c = 0;
    for (int64_t i = 0; i < m; ++i) {
        const int64_t t = int64_t(y[i]) + c;
        y[i] = uint64_t(t) & M52;
        c = t >> 52;
        kap[i] += c;
    }
}
KM_OUTL void digits_k(uint64_t *y, int64_t *kap, const int64_t *x, int64_t m) {
    const V M = _mm512_set1_epi64(int64_t(M52)), idx = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 15);
    V p1 = _mm512_setzero_si512(), p2 = _mm512_setzero_si512(), bad = _mm512_setzero_si512();
#pragma clang loop unroll(disable)
    for (int64_t i = 0; i < m; i += 8) {
        const V v = _mm512_loadu_si512(x + i), c1 = _mm512_srai_epi64(v, 52);
        const V y1 = _mm512_add_epi64(_mm512_and_si512(v, M), _mm512_permutex2var_epi64(c1, idx, p1));
        const V c2 = _mm512_srai_epi64(y1, 52);
        _mm512_storeu_si512(kap + i, _mm512_add_epi64(c1, c2));
        p1 = c1;
        // the second round needs the previous vector's c2: its lane 7 enters lane 0
        const V y2 = _mm512_add_epi64(_mm512_and_si512(y1, M), _mm512_permutex2var_epi64(c2, idx, p2));
        p2 = c2;
        bad = _mm512_or_si512(bad, _mm512_srai_epi64(y2, 52));
        _mm512_storeu_si512(y + i, y2);
    }
    if (__builtin_expect(KM_FORCE_EXACT || _mm512_test_epi64_mask(bad, bad) != 0, 0)) digits_exact(y, kap, m);
}
// sum over j < m of lo / hi(u[j] v[top - j]) (IFMA halves), and of u[j] k[top - j] (u < 2^52, |k| <= 3: exact 64-bit)
inline V rev8(V x) { return _mm512_permutexvar_epi64(_mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7), x); }
inline uint64_t dot_hi(const uint64_t *u, const uint64_t *v, int64_t top, int64_t m) {
    V a = _mm512_setzero_si512();
#pragma clang loop unroll(disable)
    for (int64_t j = 0; j < m; j += 8)
        a = _mm512_madd52hi_epu64(a, _mm512_loadu_si512(u + j), rev8(_mm512_loadu_si512(v + top - j - 7)));
    return uint64_t(_mm512_reduce_add_epi64(a));
}
inline int64_t dot_k(const uint64_t *u, const int64_t *k, int64_t top, int64_t m) {
    V a = _mm512_setzero_si512();
#pragma clang loop unroll(disable)
    for (int64_t j = 0; j < m; j += 8)
        a = _mm512_add_epi64(a, _mm512_mullo_epi64(_mm512_loadu_si512(u + j), rev8(_mm512_loadu_si512(k + top - j - 7))));
    return _mm512_reduce_add_epi64(a);
}
inline int64_t dot_kr(const int64_t *k, const uint64_t *v, int64_t top, int64_t m) {
    V a = _mm512_setzero_si512();
#pragma clang loop unroll(disable)
    for (int64_t j = 0; j < m; j += 8)
        a = _mm512_add_epi64(a, _mm512_mullo_epi64(_mm512_loadu_si512(k + j), rev8(_mm512_loadu_si512(v + top - j - 7))));
    return _mm512_reduce_add_epi64(a);
}
// lane p += t, split into digits (t's top part at lane p + 1; |t| < 2^62)
inline void add_lanes(uint64_t *L, int64_t p, int64_t t) {
    L[p] += uint64_t(t & int64_t(M52));
    L[p + 1] += uint64_t(t >> 52);
}

// The quotient level's middle product by one Karatsuba step (a transposed Karatsuba): W[x] (x < 8 NV) = the full columns
// c0 + x of D Q_h (every full product of those columns, none below: the deficit is the products below c0, < (h + 1)
// beta^(c0 + 1)), mod beta^(8 NV), as signed lanes. Halves of C = 8V columns (V = ceil(NV / 2); the top half V' = NV - V
// vectors); the rows as one pair of C-row chunks q0 = Q[r ..], q1 = Q[r + C ..], r = h + 1 - 2C <= 0 (zero rows below
// Q_h: h + 1 <= 8 NV <= 2C), E[m] = D[m - r]:
//   P1 = MP(t, E), t = q0 - q1;   P0 = MP(q1, S), P2 = MP(q0, S shifted C), S[m] = E[m] + E[m - C];
//   W = (P0 + P1) + beta^C (P2 - P1)  (polynomial identities of the full products).
// S and t enter the IFMA rows as digits: S = S' + beta kap - kap(shifted), t = t' + beta kt - kt(shifted); the window
// sums telescope, so the difference is exact boundary terms (K: dots of the rows with the carries at the window edges,
// kt_top times the E row R), and the lanes of the C-column windows miss their top column's hi halves (added at lane C).
// The window identities (with K1 = q1 . kap, K0 = q0 . kap, Kt = kt . E at column c):
//   win(P0) = lanes0 + beta^C (H0top + K1[c0 + C - 1]) - K1[c0 - 1]
//   win(P1) = lanes1 + beta^C (H1top + Kt[c0 + C - 1]) - Kt[c0 - 1] + kt_top R,  R = sum_x beta^x E[c0 - C + x]
//   win(P2) = lanes2 - K0[c0 + C - 1]  (mod beta^C; its top terms land at beta^2C and vanish mod beta^(8 NV))
constexpr unsigned KC_MAX = 8 * 20;
struct kara_buf {
    alignas(64) uint64_t q0[KC_MAX + 8], q1[KC_MAX + 8], tp[KC_MAX + 8], Sb[3 * KC_MAX + 16];
    alignas(64) int64_t x[3 * KC_MAX + 16], kap[3 * KC_MAX + 16], kt[KC_MAX + 8];
    alignas(64) uint64_t L0[KC_MAX + 8], L1[KC_MAX + 8], L2[KC_MAX + 8];
};
inline void kara_mp(uint64_t *W, const uint64_t *Q, int64_t h, const uint64_t *D, int64_t c0, unsigned NV, kara_buf &kb) {
    const unsigned Vv = (NV + 1) / 2, Vp = NV - Vv;
    const int64_t C = 8 * int64_t(Vv), r = h + 1 - 2 * C;
    // rows (zero below Q_h[0]) and t = q0 - q1 as digits
    for (int64_t j = 0; j < C; j += 8) {
        const __m512i io = _mm512_add_epi64(_mm512_set1_epi64(r + j), _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0));
        const __mmask8 m0 = _mm512_cmpge_epi64_mask(io, _mm512_setzero_si512());
        const __mmask8 m1 = _mm512_cmpge_epi64_mask(_mm512_add_epi64(io, _mm512_set1_epi64(C)), _mm512_setzero_si512());
        const V a = _mm512_maskz_loadu_epi64(m0, Q + r + j), b = _mm512_maskz_loadu_epi64(m1, Q + r + C + j);
        _mm512_storeu_si512(kb.q0 + j, a);
        _mm512_storeu_si512(kb.q1 + j, b);
        _mm512_storeu_si512(kb.x + j, _mm512_sub_epi64(a, b));
    }
    digits_k(kb.tp, kb.kt, kb.x, C);
    const int64_t kt_top = kb.kt[C - 1];
    // S[m] = E[m] + E[m - C] for m = c0 - C .. c0 + 2C - 1 (Sb[i] = S'[c0 - C + i]); E[m] = D[m - r]
    const uint64_t *E = D - r;
    for (int64_t i = 0; i < 3 * C; i += 8)
        _mm512_storeu_si512(kb.x + i, _mm512_add_epi64(_mm512_loadu_si512(E + c0 - C + i), _mm512_loadu_si512(E + c0 - 2 * C + i)));
    digits_k(kb.Sb, kb.kap, kb.x, 3 * C);
    // the three middle products (C rows; full columns)
    constexpr int64_t BIG = int64_t(1) << 40;                    // no staircase: every row reaches the window
    wprod(kb.L0, kb.q1, C, kb.Sb, BIG, C, Vv, true);              // lanes x: columns C + x of the V array (index C + x - j)
    wprod(kb.L1, kb.tp, C, E + c0 - C, BIG, C, Vv, true);
    wprod(kb.L2, kb.q0, C, kb.Sb + C, BIG, C, Vp, true);
    for (int64_t i = 0; i < C; i += 8)
        _mm512_storeu_si512(W + i, _mm512_add_epi64(_mm512_loadu_si512(kb.L0 + i), _mm512_loadu_si512(kb.L1 + i)));
    for (int64_t i = 0; i < 8 * int64_t(Vp); i += 8)
        _mm512_storeu_si512(W + C + i, _mm512_sub_epi64(_mm512_loadu_si512(kb.L2 + i), _mm512_loadu_si512(kb.L1 + i)));
    // boundary terms
    const int64_t K1lo = dot_k(kb.q1, kb.kap, C - 1, C), K1hi = dot_k(kb.q1, kb.kap, 2 * C - 1, C);
    const int64_t K0hi = dot_k(kb.q0, kb.kap, 2 * C - 1, C);
    const int64_t Ktlo = dot_kr(kb.kt, E, c0 - 1, C), Kthi = dot_kr(kb.kt, E, c0 + C - 1, C);
    const uint64_t H0 = dot_hi(kb.q1, kb.Sb, 2 * C - 1, C), H1 = dot_hi(kb.tp, E, c0 + C - 1, C);
    add_lanes(W, 0, -(K1lo + Ktlo));
    add_lanes(W, C, int64_t(H0 + H1) + K1hi + Kthi - K0hi + Ktlo);
    if (kt_top) {                                               // + kt_top R in the low half, - kt_top R in the high half
        const V kv = _mm512_set1_epi64(kt_top);
        for (int64_t i = 0; i < C; i += 8) {
            const V rr = _mm512_mullo_epi64(kv, _mm512_loadu_si512(E + c0 - C + i));
            _mm512_storeu_si512(W + i, _mm512_add_epi64(_mm512_loadu_si512(W + i), rr));
            if (i < 8 * int64_t(Vp)) _mm512_storeu_si512(W + C + i, _mm512_sub_epi64(_mm512_loadu_si512(W + C + i), rr));
        }
    }
}
inline void copy_digits(uint64_t *d, const uint64_t *s, int64_t m) {
#pragma clang loop unroll(disable)
    for (int64_t i = 0; i < m; i += 8) {
        const int64_t r = m - i;
        const __mmask8 mk = r >= 8 ? __mmask8(0xff) : __mmask8((1u << r) - 1);
        _mm512_mask_storeu_epi64(d + i, mk, _mm512_maskz_loadu_epi64(mk, s + i));
    }
}
inline void zero_digits(uint64_t *d, int64_t m) {
#pragma clang loop unroll(disable)
    for (int64_t i = 0; i < m; i += 8) _mm512_storeu_si512(d + i, _mm512_setzero_si512());
}
}

extern "C" void sbn3f_div_km(uint64_t *z, const uint64_t *a, uint64_t na, const uint64_t *b, uint64_t nb, uint64_t k) noexcept {
    const unsigned ld = unsigned(__builtin_clzll(b[nb - 1]));
    const int64_t T = 64 * int64_t(k) - 64 + ld;
    const unsigned n = unsigned((T + 1 + 51) / 52);
    const unsigned h = n / 2 + 1, w = n - h, NV = (w + 10) >> 3, e = 8 * NV - w - 1;
    const unsigned NA = n + 9, d = h + 3;
    // the reciprocal's buffers as sbn3f_inv_qv lays them out (D_h / X / G at fixed relative offsets: 4K aliasing)
    struct frame_t { alignas(64) uint64_t Dbuf[NMAX + 48]; alignas(64) uint64_t Xbuf[NMAX + NMAX / 2 + 80]; alignas(64) uint64_t Gbuf[NMAX / 2 + 48]; } fr;
    alignas(64) uint64_t Dp[PAD + NMAX + 8 + PAD];              // D, n + 1 digits, zero padded
    alignas(64) uint64_t Ab[NMAX + 16];                          // A, NA digits (8-digit blocks)
    alignas(64) uint64_t Lq[NMAX / 2 + 24];                      // Q_h's window lanes, then Q_h at Lq + 2
    alignas(64) uint64_t Lm[NMAX / 2 + 24];                      // the MP's lanes, then G
    alignas(64) uint64_t Lc[NMAX / 2 + 24];                      // the correction's lanes
    alignas(64) uint64_t Qn[NMAX + 32];
    kara_buf kb;
    uint64_t *D = Dp + PAD;
    KM_ST(0);
    // D: n + 1 digits of y (as sbn3f_inv_qv), zero padded
    const unsigned nd = (n + 1 + 7) / 8;
    zero_digits(Dp, PAD);
    zero_digits(D + 8 * nd, PAD + 8);
    decode_desc(D, nd, b, int64_t(nb), 64 * int64_t(nb) - int64_t(ld) - 52 * int64_t(n + 1));
    // A: NA digits of alpha (a's top limb at weight B^-1)
    decode_desc(Ab, (NA + 7) / 8, a, int64_t(na), 64 * int64_t(na) - 52 * int64_t(NA));
    KM_ST(1);
    // X_h: sbn3f_inv_qv's sequence for n = h, the guarded last level (u = 0)
    // h >= 20: the reciprocal's table, guarded last level (u = 0); h 10..17: one guarded level from the seed's m0 =
    // ceil((h + 1) / 2) in 6..9; h 18, 19: 6 -> 10 -> h (both guarded)
    unsigned lv[16], L = 0;
    if (h >= 20) {
        const uint64_t c = ch32.c[h - 20];
        unsigned t = h;
        lv[L++] = t;
#pragma clang loop unroll(disable)
        for (unsigned i = 0; i < (c & 15); ++i) {
            t = (t >> 1) + unsigned(c >> (4 + 4 * i) & 15);
            lv[L++] = t;
        }
    } else if (h >= 18) {
        lv[0] = h; lv[1] = 10; lv[2] = 6; L = 3;
    } else {
        lv[0] = h; lv[1] = (h + 2) / 2 < 6 ? 6 : (h + 2) / 2; L = 2;
    }
    const unsigned m0 = lv[L - 1];
    uint64_t *Dh = fr.Dbuf + 8, *X = fr.Xbuf + 16, *G = fr.Gbuf + 16;
    {
        const V zero = _mm512_setzero_si512();
        _mm512_store_si512(fr.Dbuf, zero);
        copy_digits(Dh, D + (n - h), h + 1);
        _mm512_storeu_si512(Dh + h + 1, zero);
        _mm512_storeu_si512(Dh + h + 9, zero);
        const int64_t ns = (52 * int64_t(m0) + 64) >> 6;
        const fx bf{b, 0, nb};
        sbn3f_inv_seed(X + (h - m0), &bf, ns, int64_t(m0));
        _mm512_storeu_si512(X + h + 1, zero);
        _mm512_storeu_si512(X + h + 9, zero);
        _mm512_storeu_si512(X + h + 17, zero);
        _mm512_storeu_si512(X + (int64_t(h) - int64_t(m0) - 8), zero);
        _mm512_storeu_si512(X + (int64_t(h) - int64_t(m0) - 16), zero);
        uint64_t *p = fr.Xbuf + 8 * ((h - m0 + 16) / 8);
#pragma clang loop unroll(disable)
        while (p != fr.Xbuf) {
            p -= 8;
            _mm512_store_si512(p, zero);
            asm("" : "+r"(p));
        }
        if (X[h - m0] > M52 || X[h] == 0) sbn3f_inv_fix(X, h - m0, h);
        sbn3f_inv_chain(X, Dh, G, h, lv, L);
    }
    KM_ST(2);
    // X_h padded for the window products
    // X_h is read in place by the two products on X: Q_h's reads X[K - d - 1 = -3 ..], the correction's X[h - e - w - 1 >=
    // -e - 1 >= -10 ..], both up to X[h + 8]: zero on [-16, 0) and (h, h + 25) (the zero stores above; the engine changes
    // only X[0 .. h])
    const uint64_t *Xh = X;
    // Q_h = floor(S / beta^d) - 1: rows Ã[0 .. d) = A[NA - d .. NA), columns K = d - 2 .. (lanes 2 .. h + 2 hold Q_h)
    const unsigned nvq = (h + 3 + 7) / 8;
    wprod(Lq, Ab + (NA - d), d, Xh, h + 1, d - 2, nvq);
    canon_s(Lq, 8 * nvq, 2, -1);
    const uint64_t *Qh = Lq + 2;
    KM_ST(3);
    // the quotient level h -> n: MP lanes of D Q_h in columns c0 .. n + 1 (from NV = KARA_NV by one Karatsuba step)
    const int64_t c0 = int64_t(h) + 1 - e;
    if (NV >= KARA_NV) {
        kara_mp(Lm, Qh, h, D, c0, NV, kb);
#if KM_KARA_CHECK
        alignas(64) uint64_t Lr[NMAX / 2 + 24];
        wprod(Lr, Qh, h + 1, D, n + 1, c0, NV, true);
        canon_s(Lr, 8 * NV);
        alignas(64) uint64_t Lk[NMAX / 2 + 24];
        for (unsigned i = 0; i < 8 * NV; ++i) Lk[i] = Lm[i];
        canon_s(Lk, 8 * NV);
        for (unsigned i = 0; i < 8 * NV; ++i) if (Lk[i] != Lr[i]) __builtin_trap();
#endif
    } else
        wprod(Lm, Qh, h + 1, D, n + 1, c0, NV);
    KM_ST(4);
    // G = (A's digits in columns c0 .. n + 1) - MP mod beta^(8 NV): A index c + NA - n - 1 - h, from NA - n - e
    {
        const uint64_t *Aw = Ab + (NA - n - e);
#pragma clang loop unroll(disable)
        for (unsigned i = 0; i < 8 * NV; i += 8)
            _mm512_storeu_si512(Lm + i, _mm512_sub_epi64(_mm512_loadu_si512(Aw + i), _mm512_loadu_si512(Lm + i)));
        canon_s(Lm, 8 * NV);
    }
    KM_ST(5);
    // correction: lanes of X_h G_hi (G_hi = G[e .. 8 NV), w + 1 digits) in columns h - e .. n, plus Q_h[0] at column n;
    // canonical: C + Q_h[0] beta^w at lanes e .., the carry out goes into Q_h[1 ..]
    wprod(Lc, Lm + e, w + 1, Xh, h + 1, int64_t(h) - e, NV);
    const int64_t ctop = canon_s(Lc, 8 * NV, 8 * NV - 1, int64_t(Qh[0]));
    KM_ST(6);
    // Q_n = Q_h beta^w + C
    copy_digits(Qn, Lc + e, w + 1);
    copy_digits(Qn + w + 1, Qh + 1, h);
    zero_digits(Qn + n + 1, 24);
    if (ctop) {
        uint64_t c = uint64_t(ctop);
#pragma clang loop unroll(disable)
        for (unsigned i = w + 1; i <= n && c; ++i) {
            const uint64_t t = Qn[i] + c;
            Qn[i] = t & M52;
            c = t >> 52;
        }
    }
    if (__builtin_expect(ld == 63 && Qn[n] >= 2, 0)) {
        for (uint64_t i = 0; i < k; ++i) z[i] = ~uint64_t(0);
        return;
    }
    encode(z, Qn, uint64_t(52 * int64_t(n) - T), k);
    KM_ST(7);
}
