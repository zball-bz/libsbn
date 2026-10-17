#pragma once
#include "sbn3/arena.h"
#include "common/checked.hpp"
#include <pthread.h>

namespace sbn::v3 {
class Arena {
public:
    static constexpr size_t max_regions = 128, max_leases = 512;
    struct Span { size_t begin, end; };
    struct Lease { size_t begin, end, references; uint64_t generation; bool guard; };
    pthread_mutex_t mutex{};
    uint8_t *base = nullptr;
    size_t virtual_bytes = 0, budget = 0, page_size = 0, control_bytes = 0;
    Span regions[max_regions]{};
    size_t region_count = 0, resident_bytes = 0, peak_bytes = 0;
    Lease leases[max_leases]{};
    // Live-slot index: bit i is set exactly when leases[i].references != 0
    // (0 -> 1 in acquire/reserve_guard, -> 0 in release), so the overlap,
    // guard and free-slot scans visit live leases only.
    uint64_t live[max_leases / 64]{};
    uint64_t prepares = 0, trims = 0;
    size_t computations = 0;

    sbn3_status prepare(size_t offset, size_t bytes, sbn3_error *) noexcept;
    sbn3_status trim(size_t offset, size_t bytes, sbn3_error *) noexcept;
    sbn3_lease acquire(size_t offset, size_t bytes) noexcept;
    sbn3_status reserve_guard(size_t offset, size_t bytes, sbn3_lease &) noexcept;
    bool unleased(size_t offset, size_t bytes) noexcept;
    void retain(const sbn3_lease &) noexcept;
    void claim_unshared(const sbn3_lease &) noexcept;
    void release(const sbn3_lease &) noexcept;
    void enter_compute() noexcept;
    void leave_compute() noexcept;
    void stats(sbn3_arena_stats &) noexcept;
    bool contains(size_t begin, size_t end) const noexcept;
private:
    Lease &find(const sbn3_lease &) noexcept;
    bool page_span(size_t offset, size_t bytes, Span &) const noexcept;
    // true if pred holds for a live lease
    template <class P> bool any_live(P &&pred) const noexcept {
        for (size_t w = 0; w < max_leases / 64; ++w)
            for (uint64_t m = live[w]; m; m &= m - 1)
                if (pred(leases[64 * w + size_t(__builtin_ctzll(m))])) return true;
        return false;
    }
    // the lowest slot without references, max_leases if none
    size_t free_slot() const noexcept {
        for (size_t w = 0; w < max_leases / 64; ++w)
            if (~live[w]) return 64 * w + size_t(__builtin_ctzll(~live[w]));
        return max_leases;
    }
    void set_live(size_t i, bool on) noexcept {
        const uint64_t bit = uint64_t(1) << (i % 64);
        live[i / 64] = on ? live[i / 64] | bit : live[i / 64] & ~bit;
    }
};

static_assert(Arena::max_leases % 64 == 0, "live-slot words cover every lease slot");

class ArenaLock {
    Arena &a_;
public:
    explicit ArenaLock(Arena &a) noexcept : a_(a) {
        require(pthread_mutex_lock(&a_.mutex)==0,SBN3_FATAL_LIFETIME,"arena mutex lock");
    }
    ~ArenaLock() { require(pthread_mutex_unlock(&a_.mutex)==0,SBN3_FATAL_LIFETIME,"arena mutex unlock"); }
};
}
struct sbn3_arena : sbn::v3::Arena {};
