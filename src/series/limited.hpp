#pragma once
#include "sbn3/series.h"
#include "sbn3/value.h"
#include "common/checked.hpp"
#include <algorithm>
#include <cstring>
namespace sbn::v3::series {
// Internal signed integer mantissa with a binary exponent. A zero limit means
// exact arithmetic. Limiting drops low whole words towards zero; no division.
// Update only the value metadata and return its discarded low-word count.
// Keeping this separate avoids copying descriptors at every small tree node.
inline size_t limit_low_words(sbn3_series_value &v, size_t words) {
    auto &m = v.mantissa;
    sbn3_int_normalize(&m);
    if (!m.size) {
        v.exponent2 = 0;
        return 0;
    }
    size_t drop = words && m.size > words ? m.size - words : 0;
    while (drop < m.size && !m.data[drop])
        ++drop;
    if (!drop)
        return 0;
    const __int128 e = __int128(v.exponent2) + __int128(drop) * 64;
    require(e <= INT64_MAX, SBN3_FATAL_SIZE, "series limited exponent");
    m.size -= drop;
    v.exponent2 = int64_t(e);
    return drop;
}
// Borrowed window: the backing allocation/lease does not move. Use only while
// that allocation remains live; fixed output slots still use limit_value/copy.
inline void limit_window(sbn3_series_value &v, size_t words) {
    if (const auto drop = limit_low_words(v, words)) {
        v.mantissa.data += drop;
        v.mantissa.capacity -= drop;
    }
}
inline void limit_value(sbn3_series_value &v, size_t words) {
    if (const auto drop = limit_low_words(v, words))
        std::memmove(v.mantissa.data, v.mantissa.data + drop, v.mantissa.size * 8);
}
inline void copy_limited(sbn3_series_value &out, const sbn3_series_value &in, size_t words) {
    const size_t drop = words && in.mantissa.size > words ? in.mantissa.size - words : 0;
    const size_t n = in.mantissa.size - drop;
    const __int128 e = __int128(in.exponent2) + __int128(drop) * 64;
    require(n <= out.mantissa.capacity && e <= INT64_MAX, SBN3_FATAL_WORKSPACE, "series limited copy");
    if (n)
        std::memmove(out.mantissa.data, in.mantissa.data + drop, n * 8);
    out.mantissa.size = n;
    out.mantissa.negative = n && in.mantissa.negative;
    out.exponent2 = n ? int64_t(e) : 0;
}
template<bool Borrow> inline void align_limited(sbn3_series_value &v, int64_t exponent) {
    auto &m = v.mantissa;
    if (!m.size) {
        v.exponent2 = exponent;
        return;
    }
    const __int128 delta = __int128(v.exponent2) - exponent;
    require(delta % 64 == 0, SBN3_FATAL_MATH, "series word-aligned limited values");
    if (delta < 0) {
        const size_t drop = size_t(std::min(-delta / 64, __int128(m.size)));
        if constexpr (Borrow) {
            m.data += drop;
            m.capacity -= drop;
        } else
            std::memmove(m.data, m.data + drop, (m.size - drop) * 8);
        m.size -= drop;
        if (!m.size)
            m.negative = 0;
    } else {
        require(delta / 64 <= m.capacity - m.size, SBN3_FATAL_WORKSPACE, "series limited alignment");
        const size_t shift = size_t(delta / 64);
        std::memmove(m.data + shift, m.data, m.size * 8);
        std::memset(m.data, 0, shift * 8);
        m.size += shift;
    }
    v.exponent2 = exponent;
}
// x/y are disposable product buffers with at least limit+3 words. Alignment
// keeps one additional low word; cancellation therefore has an absolute error
// bound based on the operands, not a false relative bound on the small sum.
// Fixed-slot nodes keep the thin in-place path. Borrow=true is for temporary
// windows whose backing allocation survives the whole merge, never a size knob.
template<bool Borrow = false>
inline void add_limited(sbn3_series_value &out, sbn3_series_value &x, sbn3_series_value &y, size_t words) {
    require(words, SBN3_FATAL_ARGUMENT, "series add precision");
    if (!x.mantissa.size) {
        copy_limited(out, y, words);
        return;
    }
    if (!y.mantissa.size) {
        copy_limited(out, x, words);
        return;
    }
    const __int128 high = std::max(__int128(x.exponent2) + __int128(x.mantissa.size) * 64,
                                   __int128(y.exponent2) + __int128(y.mantissa.size) * 64);
    const __int128 low =
        std::max(__int128(std::min(x.exponent2, y.exponent2)), high - __int128(words + 1) * 64);
    require(low >= INT64_MIN && low <= INT64_MAX, SBN3_FATAL_SIZE, "series sum exponent");
    if constexpr (!Borrow) {
        align_limited<false>(x, int64_t(low));
        align_limited<false>(y, int64_t(low));
        sbn3_int_add(&x.mantissa, {x.mantissa.data, x.mantissa.size, x.mantissa.negative},
                    {y.mantissa.data, y.mantissa.size, y.mantissa.negative});
        limit_value(x, words);
        copy_limited(out, x, words);
    } else {
        auto wx = x, wy = y;
        align_limited<true>(wx, int64_t(low));
        align_limited<true>(wy, int64_t(low));
        // A short low operand can lose all of its tail capacity when viewed
        // inside a tightly sized slot. Reclaim its prefix only if the sum needs
        // that space (including the integer adder's carry word).
        if (std::max(wx.mantissa.size, wy.mantissa.size) >= wx.mantissa.capacity &&
            wx.mantissa.data != x.mantissa.data) {
            if (wx.mantissa.size)
                std::memmove(x.mantissa.data, wx.mantissa.data, wx.mantissa.size * 8);
            wx.mantissa.data = x.mantissa.data;
            wx.mantissa.capacity = x.mantissa.capacity;
        }
        require(std::max(wx.mantissa.size, wy.mantissa.size) < wx.mantissa.capacity,
                SBN3_FATAL_WORKSPACE, "series sum window capacity");
        sbn3_int_add(&wx.mantissa, {wx.mantissa.data, wx.mantissa.size, wx.mantissa.negative},
                     {wy.mantissa.data, wy.mantissa.size, wy.mantissa.negative});
        limit_window(wx, words);
        copy_limited(out, wx, words);
    }
}
} // namespace sbn::v3::series
