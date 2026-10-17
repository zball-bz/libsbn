// fp kernel reciprocal, IFMA side: sbn3f_inv_qv, k = 18..256 limbs (the entry sbn3f_inv, src/fp/x86_64/inv.S, hands
// these over unchanged; contract: src/fp/kernel.hpp). Seeded by the scalar kernel's hidden entry sbn3f_inv_seed at
// ns = 5..8 limbs, then one-sided Newton levels in beta = 2^52 digits: one conversion in, the levels (the generated
// engine sbn3f_inv_chain, src/fp/x86_64/inv_engine.S), one conversion out. The schedule of levels is per-n data
// (inv_sched.hpp). Research record: experiments/results/fpk_2026-10-12/newton_inv (versions A..C, then
// bitprec/claude_impl; this is frozen/inv_s4x_v1).
//
// Digits. y = |b| normalized into [1/2, 1); T = 64k - 64 + ld (z = floor(2^T / y) within 4 units is the contract).
// n = ceil((T + 1) / 52) fraction digits; D = floor(y beta^(n+1)) (n + 1 digits, exact: the decode reads b's bits);
// Y = D / beta^(n+1), so y - beta^(-n-1) < Y <= y. Every level approximates exact_m = beta^m / Y (one reference for all
// levels: the windows read D at its full length, not D's top m + 1 digits). All levels share one digit array X with
// X_m's digit 0 at index n - m (its leading digit at n); the digits below n - m are zero until a later level.
// Invariant below the last level: beta^m <= X_m < exact_m, Delta_m = exact_m - X_m < 5.01 after a guarded level
// (nn <= 2h - 1) or the seed, < 30.12 after an unguarded one (nn = 2h, from a base with Delta < 5.01).
// Seed: Zs = sbn3f_inv_seed(ns) = X_ns + c, ns + 1 limbs, ns = floor(52 m0 / 64) + 1, |Zs - B^ns / y| < d (exact
// models: d < 4 for the guarded ns = 5..7, d < 2.5e3 for the coarse ns = 8). X_m0 = floor(Zs / 2^sh0) - 2 with
// sh0 = 64 ns - 52 m0 in [4, 64] (8, 20, 32, 44 for m0 = 6..9; 60 for m0 = 5): (d + 1) / 2^sh0 < 2, so Delta in
// (0, 5 + 4 beta^(m0-n-1)), clamped up to beta^m0. The seed is ld-independent (it runs the ld = 0 route on the
// normalized y).
// Level h -> nn (w = nn - h <= h; guarded: any h >= floor(nn/2) + 1, i.e. nn <= 2h - 1 (the schedule takes a larger h
// where the windows are then a vector narrower), unguarded: nn = 2h; both windows NV = ceil((w + 3) / 8) vectors,
// 8 NV columns aligned at their top, so e = 8 NV - w - 1 >= 2 columns below the kept digits; Delta_h < 30.12 keeps
// 0 < E' + deficit < beta^(8NV) (E' = beta^(8NV-1) Y Delta_h), G_hi < 31 beta^w and X_h G_hi < 62 beta^(h+w) <
// beta^(h+w+1)):
//   MP: columns c0 = nn + 2 - 8 NV .. nn + 1 of D X_h (D at weight beta^(nn-n): D'[i] = D[n - nn + i], every lower
//     digit included), every half below column c0 dropped (the deficit, < (h+1)(beta + 3) units of column c0 <= h - 1,
//     is below (h+1.01)/beta units of G). mod beta^(8NV) this is beta^(8NV) - E' (E' = E / beta^c0, E = beta^(nn+h+1)
//     - D X_h ... = beta^(nn+1) Y Delta_h > 0). G = the complement of the window, G_hi = its top w + 1 digits (columns
//     h+1 .. nn+1, lanes e ..): g - G_hi in (-(h+1.01)/beta, 1 + beta^-2), g = E / beta^(h+1).
//   Correction: columns nn + 1 - 8 NV .. h+w of X_h G_hi (halves below dropped), C = the digits at h .. h+w (lanes
//     e ..): X_h G_hi / beta^h - C in [0, 1 + (w+2)/beta).
//   X_nn = X_h beta^w + C: exact_nn - X_nn = Y Delta_h^2 beta^(nn-2h) + X_h (g - G_hi) / beta^h + (X_h G_hi / beta^h
//     - C) in (-2(h+1.01)/beta, 3 + A1 + tiny), A1 = Y Delta_h^2 beta^(nn-2h): < 30.12^2 / beta guarded, < 5.01^2
//     unguarded. Below the last level X_nn -= 2 (Delta in (1.99, 5.01) guarded, (1.99, 30.12) unguarded), clamped up
//     to beta^nn.
// Output: z = floor(X_n / 2^sh), sh = 52n - T >= 1; 2^T / y in (exact_n - 4/beta, exact_n] 2^-sh, so
// |z - 2^T / y| < 3.01 / 2 + 1 < 4 (guarded last level), < 28.12 / 16 + 1 < 4 (unguarded, only where sh >= 4).
// X_n = 2 beta^n (y = 1/2) overflows k limbs only at ld = 63: all ones.
// Accumulator lanes: rows <= 2 NMAX add two 52-bit halves each: below 2^63.
//
// The engine's preconditions at every level h -> nn: X[i] = 0 for i < n - h (a level reads up to 9 digits below X_h[0],
// whose products the proof takes as zero), X[n] = 1, X[n - h .. n) canonical, X = 0 on (n, n + 15]. Established
// here: the zero stores of X[-16 .. 8 ceil((n + 32) / 8) - 16) before the seed digits are written from X[n - m0] up
// (sbn3f_inv_fix after them works on X[n - m0 .. n]); a level h -> nn changes only X[n - nn .. n].
// Order: the seed call first, then D's decode and X's zeros (explicit stores), then the seed's digits (its limbs by
// scalar loads into registers), the engine, a direct encode (two loads, three two-source permutes per 8 limbs).
#include "backend/u52/island.hpp"
#include "fp/kernel.hpp"
#include "fp/x86_64/inv_sched.hpp"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

using sbn::v3::fp::fx;
using sbn::v3::fp::fxr;
using sbn::v3::fp::limb;
// the scalar kernel's hidden seed entry (inv.S): x[0 .. n] = X_n + c, n = 5..8
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_seed(limb *, const fx *, int64_t) noexcept;
// the generated level engine (inv_engine.S): the levels lv[L-1] = m0 -> lv[L-2] -> .. -> lv[0] = n on X, D, G
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_chain(uint64_t *X, const uint64_t *D, uint64_t *G, unsigned n,
                                                                   const unsigned *lv, unsigned L) noexcept;
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_fix(uint64_t *X, unsigned lo, unsigned n);
extern "C" __attribute__((visibility("hidden"))) fxr sbn3f_inv_qv(limb *z, const fx *bp, int64_t leps) noexcept;

namespace {
using V = __m512i;
constexpr uint64_t M52 = (uint64_t(1) << 52) - 1;
constexpr unsigned NMAX = 320;                                  // digits: n <= ceil((64 * 256 + 64) / 52) = 317

// D[i] = bits [base + 52i, +52) of a[0 .. an) (zero outside), i < 8 nblk: one byte load per 8 digits (a byte
// permutation), masked only where the 64 bytes leave a (masked-off bytes do not fault)
[[gnu::always_inline]] inline void decode(uint64_t *D, unsigned nblk, const uint64_t *a, int64_t an, int64_t base) {
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
    const int64_t end = 8 * an;
    int64_t byte = base >> 3;
    for (unsigned b = 0; b < nblk; ++b, byte += 52) {
        const int64_t hi = end - byte;                            // the valid bytes: [max(0, -byte), min(64, hi))
        V v;
        if (__builtin_expect(byte >= 0 && hi >= 64, 1)) v = _mm512_loadu_si512(reinterpret_cast<const char *>(a) + byte);
        else {
            uint64_t m = _bzhi_u64(~uint64_t(0), unsigned(hi < 0 ? 0 : hi > 64 ? 64 : hi));
            if (byte < 0) m &= ~_bzhi_u64(~uint64_t(0), unsigned(byte < -64 ? 64 : -byte));
            v = _mm512_maskz_loadu_epi8(m, reinterpret_cast<const char *>(a) + byte);
        }
        _mm512_storeu_si512(D + 8 * b, _mm512_and_si512(_mm512_srlv_epi64(_mm512_permutexvar_epi8(perm, v), shifts), M));
    }
}

// X[0 .. m0] = the digits of floor(Zs / 2^sh0) (bits sh0 + 52i of a[0 .. s)) minus 2 at digit 0, zeros above up to the
// block's end; true when that is canonical with a nonzero leading digit (else fix() resolves the borrow / clamp).
// The seed's s <= 9 limbs a[0 .. s): the first 8 into one vector by scalar loads, each through an empty asm so clang
// cannot merge them into vector loads (those would wait until the seed's scalar stores commit, 20..30 cycles), lanes
// from s up zero; limb 8 (s = 9 only) by one vmovq; then per 8 digits (bit t = sh0 + 52i: limb q = t / 64,
// r = t % 64) two two-source permutes and variable shifts (lanes reading limbs from 16 up take zero)
[[gnu::always_inline]] inline V limbs8(const uint64_t *a, int64_t s) {
    uint64_t l[8];
    for (int i = 0; i < 8; ++i) {
        uint64_t v = i < s ? a[i] : 0;
        asm("" : "+r"(v));
        l[i] = v;
    }
    return _mm512_set_epi64(int64_t(l[7]), int64_t(l[6]), int64_t(l[5]), int64_t(l[4]), int64_t(l[3]), int64_t(l[2]), int64_t(l[1]), int64_t(l[0]));
}
inline bool seed_digits_reg(uint64_t *X, unsigned m0, const uint64_t *a, int64_t s, int64_t sh0) {
    uint64_t a8 = s > 8 ? a[8] : 0;                             // s <= 9: lane 8 is the only limb above the first vector
    asm("" : "+r"(a8));
    const V L0 = limbs8(a, s), L1 = _mm512_castsi128_si512(_mm_cvtsi64_si128(int64_t(a8)));
    const V M = _mm512_set1_epi64(int64_t(M52)), c63 = _mm512_set1_epi64(63), c64 = _mm512_set1_epi64(64), one = _mm512_set1_epi64(1);
    const V c16 = _mm512_set1_epi64(16);
    V t = _mm512_add_epi64(_mm512_set1_epi64(sh0), _mm512_set_epi64(364, 312, 260, 208, 156, 104, 52, 0));
    bool wrap = false;
    for (unsigned b = 0; b < (m0 + 1 + 7) / 8; ++b, t = _mm512_add_epi64(t, _mm512_set1_epi64(416))) {
        const V q = _mm512_srli_epi64(t, 6), r = _mm512_and_si512(t, c63), q1 = _mm512_add_epi64(q, one);
        const V lo = _mm512_maskz_permutex2var_epi64(_mm512_cmplt_epu64_mask(q, c16), L0, q, L1);
        const V hi = _mm512_maskz_permutex2var_epi64(_mm512_cmplt_epu64_mask(q1, c16), L0, q1, L1);
        V d = _mm512_and_si512(_mm512_or_si512(_mm512_srlv_epi64(lo, r), _mm512_sllv_epi64(hi, _mm512_sub_epi64(c64, r))), M);
        if (b == 0) {
            d = _mm512_sub_epi64(d, _mm512_maskz_set1_epi64(1, 2));
            wrap = _mm512_cmpgt_epu64_mask(d, M) & 1;
        }
        _mm512_storeu_si512(X + 8 * b, d);
    }
    return !wrap && X[m0];
}

// z[0 .. k) = bits [sh + 64i, +64) of the digits X (zero padded above): per 8 limbs, the digits q0 .. q0 + 11 from two
// loads; limb i takes digits q, q + 1, q + 2 (bit offset t = r0 + 64i from q0: q - q0 = floor(t / 52) =
// (1261 t) >> 16 for t < 500, r = t - 52 q) by three two-source permutes and variable shifts (counts >= 64 give zero)
// (block bk starts at bit p0 = sh + 512 bk: q0 = p0 / 52 and r0 = p0 % 52 advance by 512 = 9 * 52 + 44; full blocks
// stored plainly, the last one masked)
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

// sbn3f_inv_seed serves m0 = 5..9 only (ns = 5..8: K = ns + 1 = 6..9 takes the ld == 0 trim and its copy exit)
constexpr bool seeds_in_range() {
    for (unsigned n = 20; n <= NMAX; ++n)
        for (unsigned u = 0; u < 2; ++u) {
            const uint64_t c = qv_chain[n][u];
            unsigned t = n;
            for (unsigned i = 0; i < (c & 15); ++i) t = (t >> 1) + unsigned(c >> (4 + 4 * i) & 15);
            if (t < 5 || t > 9) return false;
        }
    return true;
}
static_assert(seeds_in_range(), "the schedule asks for a seed sbn3f_inv_seed does not serve (m0 outside 5..9)");
// The engine dispatches a level h -> t by NV = (w + 10) >> 3 (w = t - h) through tables of NV 1..20 (e = 8 NV - w - 1
// is then 2..9); nothing in the engine checks it. Every level of every entry the driver can index (n <= 316: k <= 256)
// must have 1 <= w <= h and NV <= 20
constexpr bool levels_in_engine_range() {
    for (unsigned n = 20; n <= 316; ++n)
        for (unsigned u = 0; u < 2; ++u) {
            const uint64_t c = qv_chain[n][u];
            unsigned t = n;
            for (unsigned i = 0; i < (c & 15); ++i) {
                const unsigned h = (t >> 1) + unsigned(c >> (4 + 4 * i) & 15), w = t - h;
                if (w < 1 || w > h || ((w + 10) >> 3) > 20) return false;
                t = h;
            }
        }
    return true;
}
static_assert(levels_in_engine_range(), "a level's window is outside the engine's NV 1..20");
}

// digits X[lo .. n] as signed lanes (a carry above M52 or a borrow below 0) to canonical; then the clamp to beta^m
// (called by the engine when a digit leaves [0, 2^52), and here after the seed digits)
extern "C" [[gnu::noinline, gnu::cold]] void sbn3f_inv_fix(uint64_t *X, unsigned lo, unsigned n) {
    int64_t c = 0;
    for (unsigned i = lo; i < n; ++i) {
        const int64_t t = int64_t(X[i]) + c;
        X[i] = uint64_t(t) & M52;
        c = t >> 52;
    }
    const int64_t top = int64_t(X[n]) + c;
    if (top <= 0) {
        for (unsigned i = lo; i < n; ++i) X[i] = 0;
        X[n] = 1;
    } else X[n] = uint64_t(top);
}

extern "C" fxr sbn3f_inv_qv(limb *z, const fx *bp, int64_t leps) noexcept {
    const int64_t etop = 1 - bp->e, k = etop - ((leps - 2) >> 6);
    const uint64_t nb = bp->ns & ~sbn::v3::fp::sign_bit, sign = bp->ns & sbn::v3::fp::sign_bit;
    const uint64_t *b = bp->d;
    const unsigned ld = unsigned(__builtin_clzll(b[nb - 1]));
    const int64_t T = 64 * k - 64 + ld;
    const unsigned n = unsigned((T + 1 + 51) / 52);
    // levels: lv[0] = n down to the seed's m0 = 5..9 (seeds_in_range). Level h -> t, code c (4 bits per level):
    // c = 0 unguarded h = t/2 (t even) where allowed: below the last level when the level under it is guarded (its base
    // then has Delta < 5.01), the last level also needs sh >= 4; c >= 1 guarded h = floor(t/2) + c (any h >= floor(t/2)
    // + 1 is guarded: nn <= 2h - 1; c > 1 where the windows are then a vector narrower)
    unsigned lv[16], L = 0;
    {
        const uint64_t c = qv_chain[n][52 * int64_t(n) - T >= 4];
        unsigned t = n;
        lv[L++] = t;
#pragma clang loop unroll(disable)
        for (unsigned i = 0; i < (c & 15); ++i) {
            t = (t >> 1) + unsigned(c >> (4 + 4 * i) & 15);
            lv[L++] = t;
        }
    }
    const unsigned m0 = lv[L - 1];
    alignas(64) uint64_t Dbuf[NMAX + 48], Xbuf[NMAX + NMAX / 2 + 80], Gbuf[NMAX / 2 + 48];
    uint64_t *D = Dbuf + 8, *X = Xbuf + 16, *G = Gbuf + 16;
    // seed call first (ns + 1 limbs); D (n + 1 digits of y, zero padded: 8 below, to n + 8 above: the windows read D up
    // to index n + 8) and X's zeros (-16 .. n + 15: the correction reads X from 9 digits below X_h up to X[n + 7], the
    // encode up to X[n + 15]) while its last stores retire; then X_m0 = floor(Zs / 2^sh0) - 2. The zeros by explicit
    // stores (a loop clang turns into a memset call costs about 40 cycles here)
    uint64_t zs[16];
    const int64_t ns = (52 * int64_t(m0) + 64) >> 6;
    sbn3f_inv_seed(zs, bp, ns);
    {
        const V zero = _mm512_setzero_si512();
        _mm512_store_si512(Dbuf, zero);
        const unsigned nd = (n + 1 + 7) / 8;
        decode(D, nd, b, int64_t(nb), 64 * int64_t(nb) - int64_t(ld) - 52 * int64_t(n + 1));
        _mm512_storeu_si512(D + 8 * nd, zero);
        uint64_t *p = Xbuf;
        for (unsigned i = (n + 32 + 7) / 8; i > 0; --i, p += 8) {
            _mm512_store_si512(p, zero);
            asm("" : "+r"(p));
        }
    }
    if (!seed_digits_reg(X + (n - m0), m0, zs, ns + 1, 64 * ns - 52 * int64_t(m0))) sbn3f_inv_fix(X, n - m0, n);
    sbn3f_inv_chain(X, D, G, n, lv, L);                          // every level lv[L-1] = m0 -> .. -> lv[0] = n
    // z = floor(X_n / 2^sh), sh = 52n - T in [1, 52]
    if (X[n] >= 2 && ld == 63) {
        for (int64_t i = 0; i < k; ++i) z[i] = ~uint64_t(0);
        return {etop, uint64_t(k) | sign};
    }
    encode(z, X, uint64_t(52 * int64_t(n) - T), uint64_t(k));
    return {etop, uint64_t(k) | sign};
}
