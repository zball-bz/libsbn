#pragma once
#include "sbn3/base.h"
#include <limits.h>

namespace sbn::v3 {
[[noreturn]] void fatal(sbn3_fatal_kind kind, const char *where,
                        uint64_t need = 0, uint64_t have = 0) noexcept;
inline void require(bool ok, sbn3_fatal_kind kind, const char *where,
                    uint64_t need = 0, uint64_t have = 0) noexcept {
    if (!ok) fatal(kind, where, need, have);
}
inline bool add_size(size_t a, size_t b, size_t &r) noexcept {
    return !__builtin_add_overflow(a, b, &r);
}
inline bool mul_size(size_t a, size_t b, size_t &r) noexcept {
    return !__builtin_mul_overflow(a, b, &r);
}
inline bool align_size(size_t a, size_t alignment, size_t &r) noexcept {
    if (!alignment || (alignment & (alignment - 1))) return false;
    size_t sum;
    if (!add_size(a, alignment - 1, sum)) return false;
    r = sum & ~(alignment - 1);
    return true;
}
inline size_t bytes_for(size_t n, size_t width) noexcept {
    size_t out;
    require(mul_size(n, width, out), SBN3_FATAL_SIZE, "bytes_for", n, width);
    return out;
}
inline bool overlaps(const void *a, size_t an, const void *b, size_t bn) noexcept {
    if (!an || !bn) return false;
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
inline void valid_span(const void *p, size_t n, const char *where) noexcept {
    const uintptr_t x = reinterpret_cast<uintptr_t>(p);
    require(!n || (p && n <= UINTPTR_MAX - x), SBN3_FATAL_ARGUMENT, where, n, 0);
}
}
