// fp kernel reciprocal, IFMA side: sbn3f_inv_qv, k = 18..512 limbs (and k = 17 at ld >= 62; the entry sbn3f_inv,
// src/fp/x86_64/inv.S, hands these over unchanged; contract: src/fp/kernel.hpp). Seeded by the scalar kernel's hidden entry
// sbn3f_inv_seed at ns = 5..8 limbs, which writes X_m0's digits into X itself, then one-sided Newton levels in beta = 2^52
// digits: one conversion in (D), the levels (the generated engine sbn3f_inv_chain, src/fp/x86_64/inv_engine.S), one
// conversion out. The schedule of levels is per-n data (inv_sched.hpp).
// Research record: experiments/results/fpk_2026-10-12: newton_inv (versions A..C), bitprec/claude_impl (frozen/inv_s4x_v1,
// k <= 256), ext512 (k 257..512: qv_top), slim_inv (the 32-bit tables), fix_inv (fd), ovh2_inv (this driver: oz, accepted
// 2026-10-16; ovh2_inv/src/inv_qv.cpp with -DSLIM_T32 -DOVH_SDX -DOVH_Z2 -DOVH_DDESC -DOVH_DPRE -DOVH_FRAME, its other knobs
// and the stage stamps removed: the machine code is oz_inv_qv.o's).
//
// Digits. y = |b| normalized into [1/2, 1); T = 64k - 64 + ld (z = floor(2^T / y) within 4 units is the contract).
// n = ceil((T + 1) / 52) fraction digits; D = floor(y beta^(n+1)) (n + 1 digits, exact: the decode reads b's bits);
// Y = D / beta^(n+1), so y - beta^(-n-1) < Y <= y. Every level approximates exact_m = beta^m / Y (one reference for all
// levels: the windows read D at its full length, not D's top m + 1 digits). All levels share one digit array X with
// X_m's digit 0 at index n - m (its leading digit at n); the digits below n - m are zero until a later level.
// Invariant below the last level: beta^m <= X_m < exact_m, Delta_m = exact_m - X_m < 5.01 after a guarded level
// (nn <= 2h - 1) or the seed, < 30.12 after an unguarded one (nn = 2h, from a base with Delta < 5.01).
// Seed: Zs = X_ns + c, ns + 1 limbs, ns = floor(52 m0 / 64) + 1, |Zs - B^ns / y| < d (exact models: d < 4 for the guarded
// ns = 5 and 7, d < 180.05 for ns = 6 (one coarse cubic step 2 -> 6, fix_inv), d < 2.5e3 for the coarse ns = 8).
// X_m0 = floor(Zs / 2^sh0) - 2 with sh0 = 64 ns - 52 m0 (8, 20, 32, 44 for m0 = 6..9; the schedule never ends at m0 = 5):
// (d + 1) / 2^sh0 < 2, so Delta in (0, 5 + 4 beta^(m0-n-1)), clamped up to beta^m0. sbn3f_inv_seed(X + n - m0, b, ns, m0)
// writes X_m0's digits straight from its limb registers (minus 2 at digit 0, a borrow left wrapped: the test after the call
// sends that case, and a zero leading digit, to sbn3f_inv_fix). The seed is ld-independent (it runs the ld = 0 route on the
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
// Accumulator lanes: an MP lane adds at most h + 4 rows of two 52-bit halves (h + 4 <= 1024: below 2^63; checked for every
// reachable level below), a correction lane at most 8 NV <= 160.
// Above n = 316 (k > 256) the windows stay at NV <= 20 (the engine's range): the chain climbs from the table's
// first h <= 316 by guarded levels t <- h with t - h <= 157 (qv_top, inv_sched.hpp), as many as needed; every term of
// the derivation above holds for any h with h + 4 <= 1024 (the h-dependent ones are O(h / beta)).
//
// The engine's preconditions at every level h -> nn: X[i] = 0 for i < n - h (a level reads up to 9 digits below X_h[0],
// whose products the proof takes as zero), X[n] = 1, X[n - h .. n) canonical, X = 0 on (n, n + 15]. Established
// here: the seed's digits X[n - m0 .. n], then zero stores that never touch them (X[n + 1 .. n + 25), X[n - m0 - 16 .. n - m0),
// then the aligned blocks below n - m0 down to X[-16]; sbn3f_inv_fix after them works on X[n - m0 .. n]); a level h -> nn
// changes only X[n - nn .. n].
// Order: D's decode (top-down), the seed call (it writes X_m0's digits), X's zeros (explicit stores, the first level's
// neighbourhood first), the borrow / leading-digit test, the engine, a direct encode (two loads, three two-source permutes
// per 8 limbs).
// Frame: D, X, G in one struct (frame_t) with fixed relative offsets (Xbuf - Dbuf = 0x1540 B, Gbuf - Xbuf = 0x2040 B: mod
// 4 KB 0x540 and 0x40). This layout is load-bearing for speed: a layout with X 8 KB above D made the D windows and the X digits 4K-alias
// (store / load false dependencies; ovh2_inv/RESULTS.md section 2.1).
#include "backend/u52/island.hpp"
#include "fp/kernel.hpp"
#include "inv_sched.hpp"
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

using sbn::v3::fp::fx;
using sbn::v3::fp::fxr;
using sbn::v3::fp::limb;
// the scalar kernel's hidden seed entry (inv.S), ns = 5..8: x[0 .. m0] = the digits of X_m0 (floor(Zs / 2^sh0) - 2 at digit 0);
// m0 = 0: x[0 .. ns] = Zs (the research gate's seed checks)
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_seed(uint64_t *, const fx *, int64_t, int64_t) noexcept;
// the generated level engine (inv_engine.S): the levels lv[L-1] = m0 -> lv[L-2] -> .. -> lv[0] = n on X, D, G
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_chain(uint64_t *X, const uint64_t *D, uint64_t *G, unsigned n,
                                                                   const unsigned *lv, unsigned L) noexcept;
extern "C" __attribute__((visibility("hidden"))) void sbn3f_inv_fix(uint64_t *X, unsigned lo, unsigned n);
extern "C" __attribute__((visibility("hidden"))) fxr sbn3f_inv_qv(limb *z, const fx *bp, int64_t leps) noexcept;

namespace {
using V = __m512i;
constexpr uint64_t M52 = (uint64_t(1) << 52) - 1;
constexpr unsigned NMAX = 632;                                  // digits: n <= ceil((64 * 512 + 64) / 52) = 631

// D[i] = bits [base + 52i, +52) of a[0 .. an) (zero outside), i < 8 nblk: one byte load per 8 digits (a byte permutation),
// masked only where the 64 bytes leave a (masked-off bytes do not fault). Blocks from the top down (the first levels read
// D's top blocks: their stores are then the oldest and commit first), in three loops: top blocks whose 64 bytes leave a at
// the top (masked load; a block's first byte is below the end: its digit 8b <= n lies in a), the plain blocks, the blocks
// starting below a[0] (masked)
[[gnu::always_inline]] inline void decode_desc(uint64_t *D, unsigned nblk, const uint64_t *a, int64_t an, int64_t base) {
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
        uint64_t m = _bzhi_u64(~uint64_t(0), unsigned(end - byte));
        if (byte < 0) m &= ~_bzhi_u64(~uint64_t(0), unsigned(-byte < 64 ? -byte : 64));
        put(_mm512_maskz_loadu_epi8(m, ab + byte));
    }
#pragma clang loop unroll(disable)
    for (; b >= 0 && byte >= 0; --b, byte -= 52) put(_mm512_loadu_si512(ab + byte));
#pragma clang loop unroll(disable)
    for (; b >= 0; --b, byte -= 52) put(_mm512_maskz_loadu_epi8(~_bzhi_u64(~uint64_t(0), unsigned(-byte < 64 ? -byte : 64)), ab + byte));
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

// the chain of (n, u) as the driver decodes it (levels lv[0] = n .. lv[L-1] = m0; L = 0: no row)
struct chain_t { unsigned lv[16], L; };
constexpr unsigned NMIN = 20;
// the schedule as uint32 tables (inv_sched.hpp's uint64 tables are read only here, at compile time): qv_chain rows n = 20..316,
// qv_top rows n = 317..631 (at most 3 levels of 10 bits); every entry checked to fit
constexpr unsigned TMIN = NMIN;
struct c32_t { uint32_t c[317 - TMIN][2]; };
struct t32_t { uint32_t c[315]; };
constexpr c32_t make_c32() {
    c32_t r{};
    for (unsigned t = NMIN; t <= 316; ++t)
        for (unsigned u = 0; u < 2; ++u) r.c[t - TMIN][u] = uint32_t(qv_chain[t][u]);
    return r;
}
constexpr t32_t make_t32() { t32_t r{}; for (unsigned i = 0; i < 315; ++i) r.c[i] = uint32_t(qv_top[i]); return r; }
constexpr bool fits32() {
    for (unsigned t = NMIN; t <= 316; ++t)
        for (unsigned u = 0; u < 2; ++u) if (qv_chain[t][u] >> 32) return false;
    for (unsigned i = 0; i < 315; ++i) if (qv_top[i] >> 30) return false;       // <= 3 levels above 316 (decode stops at t <= 316)
    return true;
}
static_assert(fits32(), "a schedule entry does not fit 32 bits");
alignas(64) constexpr c32_t qv_c32 = make_c32();
alignas(64) constexpr t32_t qv_t32 = make_t32();
#define QV_C(t, u) uint64_t(qv_c32.c[(t) - TMIN][(u)])
#define QV_T(n) uint64_t(qv_t32.c[(n) - 317])
constexpr chain_t chain_of(unsigned n, unsigned u) {
    chain_t r{};
    unsigned t = n;
    if (n > 316) {
        uint64_t x = QV_T(n);
        unsigned g = 0;
        do { r.lv[r.L++] = t; t = unsigned(x & 1023); x >>= 10; } while (t > 316 && ++g < 6);
        if (t > 316 || t < 20) { r.L = 0; return r; }
        u = 1;
    }
    const uint64_t c = QV_C(t, u);
    r.lv[r.L++] = t;
    for (unsigned i = 0; i < (c & 15); ++i) {
        if (r.L >= 16) { r.L = 0; return r; }
        t = (t >> 1) + unsigned(c >> (4 + 4 * i) & 15);
        r.lv[r.L++] = t;
    }
    return r;
}
// sbn3f_inv_seed serves m0 = 6..9 only (ns = 5..8: K = ns + 1 = 6..9 takes the ld == 0 trim and its digit exit)
constexpr bool seeds_in_range() {
    for (unsigned n = NMIN; n < NMAX; ++n)
        for (unsigned u = 0; u < 2; ++u) {
            const chain_t c = chain_of(n, u);
            if (c.L < 2 || c.lv[c.L - 1] < 6 || c.lv[c.L - 1] > 9) return false;    // the seed's digit exit at ns = 5 serves m0 = 6 only
        }
    return true;
}
static_assert(seeds_in_range(), "the schedule asks for a seed sbn3f_inv_seed does not serve (m0 outside 6..9)");
// The engine dispatches a level h -> t by NV = (w + 10) >> 3 (w = t - h) through tables of NV 1..20 (e = 8 NV - w - 1
// is then 2..9); nothing in the engine checks it. Every level of every entry the driver can index (n <= 631: k <= 512)
// must have 1 <= w <= h and NV <= 20; also: h + 4 <= 1024 (the MP lanes), the levels above 316 guarded
// (t <= 2h - 1), and the error recurrence of the header: an unguarded level (t = 2h) only on a guarded level or the seed,
// and as the last level only where u = 1 (sh >= 4)
constexpr bool levels_in_engine_range() {
    for (unsigned n = NMIN; n < NMAX; ++n)
        for (unsigned u = 0; u < 2; ++u) {
            const chain_t c = chain_of(n, u);
            if (c.L < 2) return false;
            for (unsigned i = 0; i + 1 < c.L; ++i) {
                const unsigned t = c.lv[i], h = c.lv[i + 1], w = t - h;
                if (h >= t || w < 1 || w > h || ((w + 10) >> 3) > 20 || h + 4 > 1024) return false;
                const bool unguarded = t == 2 * h;
                if (t > 316 && unguarded) return false;
                if (unguarded && i + 2 < c.L && c.lv[i + 1] == 2 * c.lv[i + 2]) return false;   // its base unguarded
                if (unguarded && i == 0 && !(n <= 316 && u == 1)) return false;
            }
        }
    return true;
}
static_assert(levels_in_engine_range(), "a level's window is outside the engine's NV 1..20, or the error recurrence fails");
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
        unsigned t = n, u = 52 * int64_t(n) - T >= 4;
        if (__builtin_expect(n > 316, 0)) {                     // k 257..512: the guarded levels of qv_top first
            uint64_t x = QV_T(n);
#pragma clang loop unroll(disable)
            do {
                lv[L++] = t;
                t = unsigned(x & 1023);
                x >>= 10;
            } while (t > 316);
            u = 1;                                              // the table's chain below a guarded level (chain_of)
        }
        const uint64_t c = QV_C(t, u);
        lv[L++] = t;
#pragma clang loop unroll(disable)
        for (unsigned i = 0; i < (c & 15); ++i) {
            t = (t >> 1) + unsigned(c >> (4 + 4 * i) & 15);
            lv[L++] = t;
        }
    }
    const unsigned m0 = lv[L - 1];
    // D, X, G in one struct (each 64-byte aligned, contiguous): their offsets relative to each other are fixed whatever the rest
    // of the frame (Xbuf - Dbuf = 680 words, Gbuf - Xbuf = 1032 words; mod 4 KB 0x540 and 0x40 bytes; X - D = 688 words). Load-bearing for speed: when clang
    // ordered three separate arrays itself, X landed 8 KB above D, and the D windows and the X digits 4K-aliased (+1.3 % at
    // k 65..256; experiments/results/fpk_2026-10-12/ovh2_inv/RESULTS.md section 2.1).
    struct frame_t { alignas(64) uint64_t Dbuf[NMAX + 48]; alignas(64) uint64_t Xbuf[NMAX + NMAX / 2 + 80]; alignas(64) uint64_t Gbuf[NMAX / 2 + 48]; } fr;
    static_assert(offsetof(frame_t, Xbuf) - offsetof(frame_t, Dbuf) == 0x1540 && offsetof(frame_t, Gbuf) - offsetof(frame_t, Xbuf) == 0x2040, "D / X / G layout");
    uint64_t *Dbuf = fr.Dbuf, *Xbuf = fr.Xbuf, *Gbuf = fr.Gbuf;
    uint64_t *D = Dbuf + 8, *X = Xbuf + 16, *G = Gbuf + 16;
    // D first: n + 1 digits of y, decoded top-down, zero padded (8 below, to n + 8 above: the windows read D up to index
    // n + 8). Then the seed call: it writes X_m0's digits x[0 .. m0] at X + n - m0 itself, from its limb registers (minus 2
    // at digit 0, a borrow left wrapped); they are canonical with a nonzero leading digit exactly when x[0] <= M52 and
    // X[n] != 0, else fix(). Then X's zeros, after the seed call and never over the digits: X[n + 1 .. n + 25) and
    // X[n - m0 - 16 .. n - m0) (what the first level reads around X_m0) first, then the aligned blocks below n - m0, top-down
    // (the correction reads X from 9 digits below X_h up to X[n + 7], the encode up to X[n + 15]: the precondition X = 0
    // below n - h and on (n, n + 15] holds when the engine starts). The zeros by explicit stores (a loop clang turns into a
    // memset call costs about 40 cycles here).
    const int64_t ns = (52 * int64_t(m0) + 64) >> 6;
    const V zero = _mm512_setzero_si512();
    const unsigned nd = (n + 1 + 7) / 8;
    const int64_t dbase = 64 * int64_t(nb) - int64_t(ld) - 52 * int64_t(n + 1);
    _mm512_storeu_si512(D + 8 * nd, zero);
    decode_desc(D, nd, b, int64_t(nb), dbase);
    _mm512_store_si512(Dbuf, zero);
    sbn3f_inv_seed(X + (n - m0), bp, ns, int64_t(m0));
    _mm512_storeu_si512(X + n + 1, zero);
    _mm512_storeu_si512(X + n + 9, zero);
    _mm512_storeu_si512(X + n + 17, zero);
    // the aligned blocks below cover X[-16 .. 8 floor((n - m0 + 16) / 8) - 16), i.e. up to at least n - m0 - 7: the store at n - m0 - 8
    // completes X[.. n - m0) (required); the one at n - m0 - 16 only writes zeros the aligned loop writes again, but early (the first
    // level reads up to 9 digits below X_m0: those zeros then commit with the first stores after the seed)
    _mm512_storeu_si512(X + (int64_t(n) - int64_t(m0) - 8), zero);
    _mm512_storeu_si512(X + (int64_t(n) - int64_t(m0) - 16), zero);     // n - m0 >= 21 - 9: at or above X - 4
    {
        uint64_t *p = Xbuf + 8 * ((n - m0 + 16) / 8);
#pragma clang loop unroll(disable)
        while (p != Xbuf) {
            p -= 8;
            _mm512_store_si512(p, zero);
            asm("" : "+r"(p));
        }
    }
    if (X[n - m0] > M52 || X[n] == 0) sbn3f_inv_fix(X, n - m0, n);
    sbn3f_inv_chain(X, D, G, n, lv, L);                          // every level lv[L-1] = m0 -> .. -> lv[0] = n
    // z = floor(X_n / 2^sh), sh = 52n - T in [1, 52]
    if (X[n] >= 2 && ld == 63) {
        for (int64_t i = 0; i < k; ++i) z[i] = ~uint64_t(0);
        return {etop, uint64_t(k) | sign};
    }
    encode(z, X, uint64_t(52 * int64_t(n) - T), uint64_t(k));
    return {etop, uint64_t(k) | sign};
}
