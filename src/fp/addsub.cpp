// fp kernel add / sub, portable C++ (the same algorithm as src/fp/x86_64/addsub.S; also the
// codegen reference). Contract: kernel.hpp, draft.md section 4 (add/sub spec line).
//
// Algorithm. sigma = effective sign difference. Grid g = max(min(ga, gb), L) with L = floor((leps - 1
// + sigma) / 64): same signs drop two tails (< 2 units), opposite signs one difference of tails (< 1 unit).
// Windows [pa, ea), [pb, eb) with pa = max(g, ga), pb = max(g, gb); limbs below g are dropped (no sticky).
// After ordering so that pa <= pb (operand A starts lowest), the output starting at lo = pa is
//   R1 [pa, pb)        A only      (copy, or negate when A is the smaller magnitude)
//   R2 [pb, mid)       A op B      (add_n / sub_n chain)
//   R3 [mid, top)      H only      (carry / borrow ripple; H = the operand with the higher top)
// mid = min(ea, eb), top = max(ea, eb); disjoint windows (ea <= pb) concatenate instead. Same signs: a
// carry out of top adds the limb 1. Opposite signs: big - small, then leading zero limbs are stripped.
#include "fp/kernel.hpp"
#include <string.h>

namespace sbn::v3::fp {
namespace {

inline limb add_chain(limb *z, const limb *x, const limb *y, int64_t n, unsigned char c) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        unsigned long long s;
        c = __builtin_ia32_addcarryx_u64(c, x[i], y[i], &s);
        z[i] = s;
    }
    return c;
}
inline limb sub_chain(limb *z, const limb *x, const limb *y, int64_t n, unsigned char c) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        unsigned long long s;
        c = __builtin_ia32_subborrow_u64(c, x[i], y[i], &s);
        z[i] = s;
    }
    return c;
}
inline void copy(limb *z, const limb *x, int64_t n) noexcept {
    if (n > 16) memcpy(z, x, size_t(n) * 8);
    else for (int64_t i = 0; i < n; ++i) z[i] = x[i];
}
// z = x + c over n limbs (c in {0, 1}); returns the carry out
inline limb inc_chain(limb *z, const limb *x, int64_t n, limb c) noexcept {
    int64_t i = 0;
    for (; c && i < n; ++i) { z[i] = x[i] + 1; c = z[i] == 0; }
    copy(z + i, x + i, n - i);
    return c;
}
inline limb dec_chain(limb *z, const limb *x, int64_t n, limb b) noexcept {
    int64_t i = 0;
    for (; b && i < n; ++i) { b = x[i] == 0; z[i] = x[i] - 1; }
    copy(z + i, x + i, n - i);
    return b;
}
inline int64_t strip(const limb *z, int64_t n) noexcept {
    while (z[n - 1] == 0) --n;
    return n;
}
// z = -x over n limbs; returns the borrow out (x != 0)
inline limb neg_chain(limb *z, const limb *x, int64_t n) noexcept {
    int64_t i = 0;
    for (; i < n && x[i] == 0; ++i) z[i] = 0;
    if (i == n) return 0;
    z[i] = -x[i];
    for (++i; i < n; ++i) z[i] = ~x[i];
    return 1;
}

inline fxr result(int64_t e, int64_t n, uint64_t sign) noexcept { return {e, sign | uint64_t(n)}; }

fxr addsub(limb *z, const fx &a, const fx &b, int64_t leps, uint64_t op) noexcept {
    uint64_t sa = a.ns & sign_bit, sb = (b.ns ^ op) & sign_bit;
    int64_t na = len(a.ns), nb = len(b.ns), ea = a.e, eb = b.e, ga = ea - na, gb = eb - nb;
    int64_t sigma = int64_t((sa ^ sb) >> 63);
    int64_t g = ga < gb ? ga : gb, L = (leps - 1 + sigma) >> 6;
    if (ga == gb && L <= ga) {                         // fast path: equal grids, no truncation
        bool along = na >= nb;
        int64_t nmax = along ? na : nb, d2 = along ? nb : na;
        const limb *A = a.d, *B = b.d;
        if (!sigma) {
            limb c = add_chain(z, A, B, d2, 0);
            c = inc_chain(z + d2, (along ? A : B) + d2, nmax - d2, c);
            z[nmax] = c;
            return result(ga + nmax + int64_t(c), nmax + int64_t(c), sa);
        }
        bool abig = na != nb ? na > nb : false;
        if (na == nb) {
            int64_t k = na - 1;
            while (k >= 0 && A[k] == B[k]) --k;
            if (k < 0) return {0, 0};
            abig = A[k] > B[k];
        }
        const limb *big = abig ? A : B, *small = abig ? B : A;
        limb bo = sub_chain(z, big, small, d2, 0);
        dec_chain(z + d2, big + d2, nmax - d2, bo);
        int64_t n = strip(z, nmax);
        return result(ga + n, n, abig ? sa : sb);
    }
    if (L > g) g = L;
    int64_t pa = ga > g ? ga : g, pb = gb > g ? gb : g;
    const limb *A = a.d + (pa - ga), *B = b.d + (pb - gb);
    if (pa >= ea) {                                    // A negligible
        if (pb >= eb) return {0, 0};
        copy(z, B, eb - pb);
        return result(eb, eb - pb, sb);
    }
    if (pb >= eb) {                                    // B negligible
        copy(z, A, ea - pa);
        return result(ea, ea - pa, sa);
    }
    if (pa > pb) {                                     // A starts lowest
        const limb *t = A; A = B; B = t;
        int64_t u = pa; pa = pb; pb = u; u = ea; ea = eb; eb = u;
        uint64_t v = sa; sa = sb; sb = v;
    }
    int64_t lo = pa;
    if (ea <= pb) {                                    // disjoint: A entirely below B's window
        int64_t la = ea - pa, gap = pb - ea, lb = eb - pb;
        if (!sigma) {
            copy(z, A, la);
            memset(z + la, 0, size_t(gap) * 8);
            copy(z + la + gap, B, lb);
            return result(eb, eb - lo, sa);
        }
        neg_chain(z, A, la);                           // borrow 1: A's top limb is nonzero
        memset(z + la, 0xff, size_t(gap) * 8);
        dec_chain(z + la + gap, B, lb, 1);
        int64_t n = eb - lo;
        n = strip(z, n);
        return result(lo + n, n, sb);
    }
    int64_t d1 = pb - pa, mid = ea < eb ? ea : eb, top = ea < eb ? eb : ea;
    int64_t d2 = mid - pb, d3 = top - mid;
    const limb *Ah = A + d1;                           // A at pb
    const limb *H = ea >= eb ? Ah + d2 : B + d2;       // the higher operand at mid
    if (!sigma) {
        copy(z, A, d1);
        limb c = add_chain(z + d1, Ah, B, d2, 0);
        c = inc_chain(z + d1 + d2, H, d3, c);
        z[top - lo] = c;
        return result(top + int64_t(c), top - lo + int64_t(c), sa);
    }
    bool abig;                                         // |A window| > |B window|
    if (ea != eb) abig = ea > eb;
    else {
        int64_t k = d2 - 1;
        while (k >= 0 && Ah[k] == B[k]) --k;
        if (k >= 0) abig = Ah[k] > B[k];
        else {
            int64_t j = d1 - 1;
            while (j >= 0 && A[j] == 0) --j;
            if (j < 0) return {0, 0};
            abig = true;
        }
    }
    unsigned char bin = 0;
    if (abig) copy(z, A, d1);
    else bin = (unsigned char)neg_chain(z, A, d1);
    const limb *big = abig ? Ah : B, *small = abig ? B : Ah;
    limb bo = sub_chain(z + d1, big, small, d2, bin);
    dec_chain(z + d1 + d2, H, d3, bo);
    int64_t n = top - lo;
    n = strip(z, n);
    return result(lo + n, n, abig ? sa : sb);
}

} // namespace

namespace portable {
fxr add(limb *z, const fx &a, const fx &b, int64_t leps) noexcept { return addsub(z, a, b, leps, 0); }
fxr sub(limb *z, const fx &a, const fx &b, int64_t leps) noexcept { return addsub(z, a, b, leps, sign_bit); }
}

} // namespace sbn::v3::fp
