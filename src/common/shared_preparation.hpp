#pragma once
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3 {
// Exact, versioned table identity; never a hash-only equality test. Keys name
// immutable data, not the input lengths or mutable workspace of a consumer.
struct SharedPreparation {
    uint64_t key[8]{};
    size_t bytes = 0, alignment = 128;
    bool operator==(const SharedPreparation &) const = default;
};
struct SharedPlacement {
    size_t offset = 0, bytes = 0;
    bool owner = false;
};
// A scheduled merge has at most two product families in the v1 executor.
// Identity/construction remain backend-owned; unused requests have zero bytes.
struct SharedPreparations { SharedPreparation requests[2]{}; };
// Bounded query-time interner. Exhaustion only stops interning new keys; it
// never rejects a valid plan or changes arithmetic. No heap or runtime lock.
template<size_t Capacity = 512> struct SharedPreparationIndex {
    struct Entry { SharedPreparation request; size_t offset; };
    Entry entries[Capacity]{};
    size_t count = 0;
    bool find(const SharedPreparation &r, size_t &offset) const {
        for (size_t j = 0; j < count; ++j)
            if (entries[j].request == r) { offset = entries[j].offset; return true; }
        return false;
    }
    void insert(const SharedPreparation &r, size_t offset) {
        if (count < Capacity) entries[count++] = {r, offset};
    }
};
} // namespace sbn::v3
