/* Word and schoolbook division. The reciprocal primitives, divrem_1/divrem_2
 * and the 3/2 schoolbook loop are ports of the GNU MP Library 6.3.0 generic
 * code (gmp-impl.h invert_limb/invert_pi1/udiv_qrnnd_preinv/udiv_qr_3by2,
 * mpn/generic/divrem_1.c, divrem_2.c, sbpi1_div_qr.c), carried through
 * libsbn/include/sbn/scalar.h and sbn/divrem.h. See scalar-donor-notice.txt
 * in src/core/x86_64 for the retained copyright and dual LGPLv3+/GPLv2+
 * notice; those terms apply to this file.
 * Copyright 1991-2016 Free Software Foundation, Inc. (ported parts). */
#include "sbn3/divrem.h"
#include "value/divrem_words.hpp"
#include "common/checked.hpp"
#include "common/small_checks.h"
#include <string.h>
using namespace sbn::v3;
namespace sbn::v3::divrem_words {
uint64_t sbpi1_div_qr(uint64_t *qp, uint64_t *np, size_t nn, const uint64_t *dp, size_t dn,
                      uint64_t dinv) noexcept {
    uint64_t qh, n1, n0, d1, d0, cy, cy1, q;
    np += nn;
    qh = compare(np - dn, dp, dn) >= 0;
    if (qh)
        sbn3i_sub_n(np - dn, np - dn, dp, long(dn));
    qp += nn - dn;
    dn -= 2;
    d1 = dp[dn + 1];
    d0 = dp[dn + 0];
    np -= 2;
    n1 = np[1];
    for (size_t i = nn - (dn + 2); i > 0; i--) {
        np--;
        if (n1 == d1 && np[1] == d0) {
            q = ~uint64_t(0);
            sbn3i_submul_1(np - dn, dp, long(dn + 2), q);
            n1 = np[1];
        } else {
            udiv_qr_3by2(q, n1, n0, n1, np[1], np[0], d1, d0, dinv);
            cy = sbn3i_submul_1(np - dn, dp, long(dn), q);
            cy1 = n0 < cy;
            n0 = n0 - cy;
            cy = n1 < cy1;
            n1 = n1 - cy1;
            np[0] = n0;
            if (cy) {
                n1 += d1 + sbn3i_add_n(np - dn, np - dn, dp, long(dn + 1));
                q--;
            }
        }
        *--qp = q;
    }
    np[1] = n1;
    return qh;
}
uint64_t divrem_1(uint64_t *qp, const uint64_t *up, size_t un, uint64_t d) noexcept {
    uint64_t r = 0;
    if (!un)
        return 0;
    qp += un - 1;
    if (d >> 63) {
        r = up[un - 1];
        const uint64_t q = r >= d;
        *qp-- = q;
        r -= d & (0 - q);
        un--;
        const uint64_t dinv = invert_limb(d);
        for (size_t i = un; i-- > 0;) {
            udiv_qrnnd_preinv(*qp, r, r, up[i], d, dinv);
            qp--;
        }
        return r;
    }
    const unsigned cnt = unsigned(__builtin_clzll(d));
    uint64_t n1 = up[un - 1];
    if (n1 < d) {
        r = n1;
        *qp-- = 0;
        if (--un == 0)
            return r;
    }
    d <<= cnt;
    r <<= cnt;
    const uint64_t dinv = invert_limb(d);
    n1 = up[un - 1];
    r |= n1 >> (64 - cnt);
    for (size_t i = un - 1; i-- > 0;) {
        const uint64_t n0 = up[i], nshift = (n1 << cnt) | (n0 >> (64 - cnt));
        udiv_qrnnd_preinv(*qp, r, r, nshift, d, dinv);
        qp--;
        n1 = n0;
    }
    udiv_qrnnd_preinv(*qp, r, r, n1 << cnt, d, dinv);
    return r >> cnt;
}
uint64_t divrem_2(uint64_t *qp, uint64_t *np, size_t nn, const uint64_t *dp) noexcept {
    np += nn - 2;
    const uint64_t d1 = dp[1], d0 = dp[0];
    uint64_t r1 = np[1], r0 = np[0], most = 0;
    if (r1 >= d1 && (r1 > d1 || r0 >= d0)) {
        sub_ddmmss(r1, r0, r1, r0, d1, d0);
        most = 1;
    }
    const uint64_t dinv = invert_pi1(d1, d0);
    for (size_t i = nn - 2; i-- > 0;) {
        uint64_t q;
        const uint64_t n0 = np[-1];
        udiv_qr_3by2(q, r1, r0, r1, r0, n0, d1, d0, dinv);
        np--;
        qp[i] = q;
    }
    np[1] = r1;
    np[0] = r0;
    return most;
}
size_t schoolbook(uint64_t *q, uint64_t *r, const uint64_t *n, size_t nn, const uint64_t *d, size_t dn,
                  uint64_t *scratch) noexcept {
    const size_t qcount = nn >= dn ? nn - dn + 1 : 0;
    while (nn && !n[nn - 1])
        --nn;
    if (nn < dn) {
        memcpy(r, n, nn * 8);
        memset(r + nn, 0, (dn - nn) * 8);
        memset(q, 0, qcount * 8);
        return 0;
    }
    const size_t qn = nn - dn + 1;
    memset(q + qn, 0, (qcount - qn) * 8);
    if (dn == 1) {
        r[0] = divrem_1(q, n, nn, d[0]);
    } else {
        const unsigned cnt = unsigned(__builtin_clzll(d[dn - 1]));
        uint64_t *n2 = scratch;
        uint64_t small[2];
        const uint64_t *d2 = d;
        uint64_t *d2buf = dn == 2 ? small : scratch + nn + 1;
        if (cnt) {
            shift_left(d2buf, d, dn, cnt);
            d2 = d2buf;
            n2[nn] = shift_left(n2, n, nn, cnt);
        } else {
            memcpy(n2, n, nn * 8);
            n2[nn] = 0;
        }
        if (dn == 2)
            divrem_2(q, n2, nn + 1, d2);
        else
            sbpi1_div_qr(q, n2, nn + 1, d2, dn, invert_pi1(d2[dn - 1], d2[dn - 2]));
        if (cnt)
            shift_right(r, n2, dn, cnt);
        else
            memcpy(r, n2, dn * 8);
    }
    size_t count = qn;
    while (count && !q[count - 1])
        --count;
    return count;
}
} // namespace sbn::v3::divrem_words
extern "C" size_t sbn3_divrem_basecase(uint64_t *q, uint64_t *r, const uint64_t *n, size_t nn,
                                       const uint64_t *d, size_t dn, uint64_t *scratch) {
    require(dn && d && d[dn - 1] && r && (!nn || n) && nn <= size_t(LONG_MAX) - 2, SBN3_FATAL_ARGUMENT,
            "basecase division divisor/spans");
    if constexpr (SBN3_CHECK_SMALL) {
        const size_t nb = bytes_for(nn, 8), db = bytes_for(dn, 8), qb = nn >= dn ? (nn - dn + 1) * 8 : 0,
                     sb = (nn + dn + 1) * 8;
        valid_span(n, nb, "basecase numerator");
        valid_span(q, qb, "basecase quotient");
        valid_span(scratch, sb, "basecase scratch");
        require(nn < dn || (q && scratch), SBN3_FATAL_ARGUMENT, "basecase outputs");
        const void *spans[5] = {q, r, n, d, scratch};
        const size_t sizes[5] = {qb, db, nb, db, sb};
        for (unsigned a = 0; a < 5; ++a)
            for (unsigned b = a + 1; b < 5; ++b)
                require(!overlaps(spans[a], sizes[a], spans[b], sizes[b]), SBN3_FATAL_ARGUMENT, "basecase overlap");
    }
    require(nn < dn || (q && scratch), SBN3_FATAL_ARGUMENT, "basecase outputs");
    return divrem_words::schoolbook(q, r, n, nn, d, dn, scratch);
}
extern "C" void sbn3_int_divrem_basecase(sbn3_int *q, sbn3_int *r, sbn3_int_view n, sbn3_int_view d,
                                         sbn3_limbs scratch) {
    require(q && r && n.negative <= 1 && d.negative <= 1, SBN3_FATAL_ARGUMENT, "integer division arguments");
    while (d.size && !d.data[d.size - 1])
        --d.size;
    while (n.size && !n.data[n.size - 1])
        --n.size;
    require(d.size, SBN3_FATAL_ARGUMENT, "integer division by zero");
    const size_t qn = n.size >= d.size ? n.size - d.size + 1 : 0;
    require(q->capacity >= qn && r->capacity >= d.size && scratch.capacity >= n.size + d.size + 1,
            SBN3_FATAL_SIZE, "integer division capacity");
    const size_t count = sbn3_divrem_basecase(q->data, r->data, n.data, n.size, d.data, d.size, scratch.data);
    q->size = count;
    q->negative = count ? (n.negative ^ d.negative) : 0;
    size_t rn = d.size;
    while (rn && !r->data[rn - 1])
        --rn;
    r->size = rn;
    r->negative = rn ? n.negative : 0;
}
