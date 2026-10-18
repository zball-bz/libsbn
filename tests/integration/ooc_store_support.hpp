// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "product_support.hpp"
#include "external/store.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <memory>
using namespace sbn::v3;
using namespace sbn::v3::external;
struct Aligned {
    uint8_t *data;
    size_t bytes;
    explicit Aligned(size_t n): bytes(n) {
        void *p = nullptr; assert(!posix_memalign(&p, 4096, std::max(size_t(4096), up(n,4096))));
        data = static_cast<uint8_t *>(p);
    }
    ~Aligned() { free(data); }
    Aligned(const Aligned &) = delete;
    std::span<uint8_t> span() { return {data, bytes}; }
};
struct Device {
    std::vector<uint8_t> bytes;
    uint64_t reads = 0, writes = 0, fail_read = 0, fail_write = 0, short_write = 0, short_read = 0;
    uint64_t corrupt_at = UINT64_MAX, cancel_after_write = 0;
    uint64_t fail_write_at=UINT64_MAX,fail_read_at=UINT64_MAX;
    uint64_t cancel_write_at=UINT64_MAX,arm_read_failure_at=UINT64_MAX;
    bool read_failure_armed=false;
    int corrupt_reads = 0;
    bool cancelled = false;
    explicit Device(size_t n): bytes(n, 0xa5) {}
    static Completion read(void *ptr, uint64_t at, std::span<uint8_t> out) noexcept {
        auto &d = *static_cast<Device *>(ptr);const auto ticket=__atomic_add_fetch(&d.reads,uint64_t(1),__ATOMIC_RELAXED);
        if (ticket == d.fail_read||__atomic_load_n(&d.read_failure_armed,__ATOMIC_ACQUIRE)) return {IoCode::io, 0, EIO};
        if(d.fail_read_at>=at&&d.fail_read_at-at<out.size())return {IoCode::io,0,EIO};
        const size_t n = ticket == d.short_read ? out.size()/2 : out.size();
        assert(at <= d.bytes.size() && n <= d.bytes.size()-at);
        std::memcpy(out.data(), d.bytes.data()+at, n);
        if (d.corrupt_at >= at && d.corrupt_at-at < n && __atomic_load_n(&d.corrupt_reads,__ATOMIC_RELAXED)) {
            out[size_t(d.corrupt_at-at)] ^= 0x80;
            if (__atomic_load_n(&d.corrupt_reads,__ATOMIC_RELAXED)>0)__atomic_fetch_sub(&d.corrupt_reads,1,__ATOMIC_RELAXED);
        }
        return {IoCode::ok, n};
    }
    static Completion write(void *ptr, uint64_t at, std::span<const uint8_t> in) noexcept {
        auto &d = *static_cast<Device *>(ptr);const auto ticket=__atomic_add_fetch(&d.writes,uint64_t(1),__ATOMIC_RELAXED);
        if (ticket == d.fail_write) return {IoCode::io, 0, ENOSPC};
        if(d.fail_write_at>=at&&d.fail_write_at-at<in.size())return {IoCode::io,0,EIO};
        const size_t n = ticket == d.short_write ? in.size()/2 : in.size();
        assert(at <= d.bytes.size() && n <= d.bytes.size()-at);
        std::memcpy(d.bytes.data()+at, in.data(), n);
        if(d.arm_read_failure_at>=at&&d.arm_read_failure_at-at<n)__atomic_store_n(&d.read_failure_armed,true,__ATOMIC_RELEASE);
        if(d.cancel_write_at>=at&&d.cancel_write_at-at<n)__atomic_store_n(&d.cancelled,true,__ATOMIC_RELEASE);
        if (ticket == d.cancel_after_write) __atomic_store_n(&d.cancelled,true,__ATOMIC_RELEASE);
        return {IoCode::ok, n};
    }
    static bool cancel(void *ptr) noexcept { return __atomic_load_n(&static_cast<Device *>(ptr)->cancelled,__ATOMIC_ACQUIRE); }
    Transport transport() {
        Transport t{}; t.context = this; t.read = read; t.write = write;
        t.capacity = bytes.size(); t.alignment = 4096; t.max_transfer = 1u<<20;
        t.cancel = {this, cancel}; return t;
    }
};
[[maybe_unused]] static void fill(std::span<uint8_t> s) {
    for (size_t i = 0; i < s.size(); ++i) s[i] = uint8_t((i*37) ^ (i>>5) ^ (i>>17));
}
[[maybe_unused]] static Region fresh(size_t n, uint64_t at = 0, uint64_t identity = 1, RegionKind kind = RegionKind::value) {
    RegionLayout l{}; assert(!region_layout(n, at, l));
    Region r{}; assert(!begin_region(r, {identity, 0x53424e, 1}, identity+17, l, kind)); return r;
}
[[maybe_unused]] static void write_value(Transport &t, Region &r, StoreBuffers b, std::span<const uint8_t> input, size_t piece = 12347) {
    ValueWriter writer;
    allocation_watch_start();
    assert(!writer.open(t, r, b));
    while (!input.empty()) {
        const size_t n = std::min(piece, input.size()); assert(!writer.append(input.first(n))); input = input.subspan(n);
    }
    assert(!writer.seal()); assert(!allocation_watch_stop());
}
