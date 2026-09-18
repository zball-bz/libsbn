#pragma once
/* Private word-division primitives. Ported from GMP 6.3.0 (gmp-impl.h,
 * longlong.h) through libsbn/include/sbn/scalar.h; the retained notice is
 * src/core/x86_64/scalar-donor-notice.txt (LGPLv3+ / GPLv2+). */
#include "core/x86_64/word.hpp"
#include <stddef.h>
#include <stdint.h>
extern "C" uint64_t sbn3i_submul_1(uint64_t *, const uint64_t *, long, uint64_t);
namespace sbn::v3::divrem_words {
inline void umul_ppmm(uint64_t &h, uint64_t &l, uint64_t a, uint64_t b) noexcept {
    const __uint128_t p = __uint128_t(a) * b;
    h = uint64_t(p >> 64);
    l = uint64_t(p);
}
inline void add_ssaaaa(uint64_t &sh, uint64_t &sl, uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) noexcept {
    const uint64_t l = al + bl;
    sh = ah + bh + (l < al);
    sl = l;
}
inline void sub_ddmmss(uint64_t &sh, uint64_t &sl, uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) noexcept {
    const uint64_t l = al - bl;
    sh = ah - bh - (al < bl);
    sl = l;
}
/* floor((2^128-1)/d) - 2^64 for normalized d. */
inline uint64_t invert_limb(uint64_t d) noexcept {
    const __uint128_t num = (__uint128_t(~d) << 64) | ~uint64_t(0);
    return uint64_t(num / d);
}
/* 3/2 reciprocal of <d1,d0>, d1 normalized. */
inline uint64_t invert_pi1(uint64_t d1, uint64_t d0) noexcept {
    uint64_t v = invert_limb(d1), p = d1 * v, t1, t0, mask;
    p += d0;
    if (p < d0) {
        v--;
        mask = 0 - uint64_t(p >= d1);
        p -= d1;
        v += mask;
        p -= mask & d1;
    }
    umul_ppmm(t1, t0, d0, v);
    p += t1;
    if (p < t1) {
        v--;
        if (p >= d1 && (p > d1 || t0 >= d0))
            v--;
    }
    return v;
}
/* <nh,nl>/d, d normalized, di=invert_limb(d). */
#define udiv_qrnnd_preinv(q, r, nh, nl, d, di)                                                                 \
    do {                                                                                                       \
        uint64_t _qh, _ql, _r, _mask;                                                                          \
        ::sbn::v3::divrem_words::umul_ppmm(_qh, _ql, (nh), (di));                                               \
        ::sbn::v3::divrem_words::add_ssaaaa(_qh, _ql, _qh, _ql, (nh) + 1, (nl));                                \
        _r = (nl) - _qh * (d);                                                                                 \
        _mask = 0 - uint64_t(_r > _ql);                                                                        \
        _qh += _mask;                                                                                          \
        _r += _mask & (d);                                                                                     \
        if (_r >= (d)) {                                                                                       \
            _r -= (d);                                                                                         \
            _qh++;                                                                                             \
        }                                                                                                      \
        (r) = _r;                                                                                              \
        (q) = _qh;                                                                                             \
    } while (0)
/* <n2,n1,n0>/<d1,d0>, d1 normalized, dinv=invert_pi1(d1,d0). Canonical
 * call pattern (q,r1,r0,r1,r0,n0,...): outputs may repeat inputs as in GMP. */
#define udiv_qr_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv)                                                     \
    do {                                                                                                       \
        uint64_t _q0, _t1, _t0, _mask;                                                                         \
        ::sbn::v3::divrem_words::umul_ppmm((q), _q0, (n2), (dinv));                                             \
        ::sbn::v3::divrem_words::add_ssaaaa((q), _q0, (q), _q0, (n2), (n1));                                    \
        (r1) = (n1) - (d1) * (q);                                                                              \
        ::sbn::v3::divrem_words::sub_ddmmss((r1), (r0), (r1), (n0), (d1), (d0));                                \
        ::sbn::v3::divrem_words::umul_ppmm(_t1, _t0, (d0), (q));                                                \
        ::sbn::v3::divrem_words::sub_ddmmss((r1), (r0), (r1), (r0), _t1, _t0);                                  \
        (q)++;                                                                                                 \
        _mask = 0 - uint64_t((r1) >= _q0);                                                                     \
        (q) += _mask;                                                                                          \
        ::sbn::v3::divrem_words::add_ssaaaa((r1), (r0), (r1), (r0), _mask & (d1), _mask & (d0));                \
        if ((r1) >= (d1)) {                                                                                    \
            if ((r1) > (d1) || (r0) >= (d0)) {                                                                 \
                (q)++;                                                                                         \
                ::sbn::v3::divrem_words::sub_ddmmss((r1), (r0), (r1), (r0), (d1), (d0));                        \
            }                                                                                                  \
        }                                                                                                      \
    } while (0)
inline int compare(const uint64_t *a, const uint64_t *b, size_t n) noexcept {
    while (n-- > 0)
        if (a[n] != b[n])
            return a[n] > b[n] ? 1 : -1;
    return 0;
}
/* r = a<<bits (0<bits<64) over n limbs, returns the bits shifted out. r==a allowed. */
inline uint64_t shift_left(uint64_t *r, const uint64_t *a, size_t n, unsigned bits) noexcept {
    const uint64_t out = a[n - 1] >> (64 - bits);
    for (size_t k = n; k-- > 1;)
        r[k] = (a[k] << bits) | (a[k - 1] >> (64 - bits));
    r[0] = a[0] << bits;
    return out;
}
/* r = a>>bits (0<bits<64) over n limbs, high bits zero. r==a allowed. */
inline void shift_right(uint64_t *r, const uint64_t *a, size_t n, unsigned bits) noexcept {
    for (size_t k = 0; k + 1 < n; ++k)
        r[k] = (a[k] >> bits) | (a[k + 1] << (64 - bits));
    r[n - 1] = a[n - 1] >> bits;
}
/* {np,nn} / {dp,dn}: dp normalized, dn>=3, nn>=dn. Quotient nn-dn limbs to
 * qp, remainder left in np[0..dn); returns the high quotient limb (0/1). */
uint64_t sbpi1_div_qr(uint64_t *qp, uint64_t *np, size_t nn, const uint64_t *dp, size_t dn, uint64_t dinv) noexcept;
/* Quotient un limbs (d need not be normalized), returns the remainder. */
uint64_t divrem_1(uint64_t *qp, const uint64_t *up, size_t un, uint64_t d) noexcept;
/* dp[1] normalized, nn>=2: nn-2 quotient limbs to qp, remainder in np[0..2), returns the high limb. */
uint64_t divrem_2(uint64_t *qp, uint64_t *np, size_t nn, const uint64_t *dp) noexcept;
/* Complete unsigned schoolbook division with normalization; scratch has
 * nn+dn+1 limbs. Writes all nn-dn+1 quotient limbs when nn>=dn (high
 * zeros included, also for a shorter effective numerator) and dn remainder limbs. Returns the normalized quotient length. */
size_t schoolbook(uint64_t *q, uint64_t *r, const uint64_t *n, size_t nn, const uint64_t *d, size_t dn,
                  uint64_t *scratch) noexcept;
} // namespace sbn::v3::divrem_words
