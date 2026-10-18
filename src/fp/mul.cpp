// fp kernel mul / sqr, portable C++ (the specification of kernel.hpp's mul_shape; also the codegen
// reference). Product scanning with a 192-bit column accumulator: the high product keeps columns >= k - 2
// of the k x k product (omitted < B^k, so 0 <= floor(A'B' / B^k) - r <= 1), the full product all columns.
#include "fp/kernel.hpp"

namespace sbn::v3::fp {
namespace {

typedef unsigned __int128 u128;

// columns c0 .. ca + cb - 2 of the ca x cb product a * b (c0 = 0: the full product); writes column c to
// z[c - zc] for c >= zc, then the final carry limb; returns the number of limbs written
inline int64_t product_columns(limb *z, const limb *a, int64_t ca, const limb *b, int64_t cb, int64_t c0, int64_t zc) noexcept {
    u128 acc = 0;
    limb top = 0;
    int64_t w = 0;
    for (int64_t c = c0; c <= ca + cb - 2; ++c) {
        int64_t i0 = c - (cb - 1) > 0 ? c - (cb - 1) : 0, i1 = c < ca - 1 ? c : ca - 1;
        for (int64_t i = i0; i <= i1; ++i) {
            u128 p = u128(a[i]) * b[c - i];
            acc += p;
            top += acc < p;
        }
        if (c >= zc) z[w++] = limb(acc);
        acc = (acc >> 64) | (u128(top) << 64);
        top = 0;
    }
    z[w++] = limb(acc);
    return w;
}
// the same for a square (a_i a_j, i < j, doubled)
inline int64_t square_columns(limb *z, const limb *a, int64_t ca, int64_t c0, int64_t zc) noexcept {
    u128 acc = 0;
    limb top = 0;
    int64_t w = 0;
    for (int64_t c = c0; c <= 2 * ca - 2; ++c) {
        int64_t i0 = c - (ca - 1) > 0 ? c - (ca - 1) : 0;
        u128 s = 0;
        limb st = 0;
        for (int64_t i = i0; 2 * i < c; ++i) {
            u128 p = u128(a[i]) * a[c - i];
            s += p;
            st += s < p;
        }
        st = (st << 1) | limb(s >> 127);                    // double the off-diagonal sum
        s <<= 1;
        if ((c & 1) == 0) {
            u128 p = u128(a[c / 2]) * a[c / 2];
            s += p;
            st += s < p;
        }
        acc += s;
        top += st + (acc < s);
        if (c >= zc) z[w++] = limb(acc);
        acc = (acc >> 64) | (u128(top) << 64);
        top = 0;
    }
    z[w++] = limb(acc);
    return w;
}
inline fxr finish(limb *z, int64_t n, int64_t e, uint64_t sign) noexcept {
    while (n && z[n - 1] == 0) --n, --e;
    if (!n) return {0, 0};
    return {e, sign | uint64_t(n)};
}

fxr mul_or_sqr(limb *z, const fx &a, const fx &b, int64_t leps, bool square) noexcept {
    mulshape s = mul_shape(a, b, leps);
    if (s.k <= 0) return {0, 0};
    uint64_t sign = (a.ns ^ b.ns) & sign_bit;
    const limb *A = a.d + (len(a.ns) - s.ka), *B = b.d + (len(b.ns) - s.kb);
    int64_t e = a.e + b.e, n;
    if (!s.full) {                                          // the high product: k limbs at grid L2
        int64_t k = s.k;
        if (square) square_columns(z, A, k, k > 2 ? k - 2 : 0, k);
        else product_columns(z, A, k, B, k, k > 2 ? k - 2 : 0, k);
        n = k;
    } else {                                                // the full product of the windows
        if (square) square_columns(z, A, s.ka, 0, 0);
        else product_columns(z, A, s.ka, B, s.kb, 0, 0);
        n = s.ka + s.kb;
    }
    return finish(z, n, e, sign);
}

} // namespace

namespace portable {
fxr mul(limb *z, const fx &a, const fx &b, int64_t leps) noexcept { return mul_or_sqr(z, a, b, leps, false); }
fxr sqr(limb *z, const fx &a, int64_t leps) noexcept { return mul_or_sqr(z, a, a, leps, true); }
}

} // namespace sbn::v3::fp
