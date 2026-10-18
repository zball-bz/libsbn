// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/store.hpp"

namespace sbn::v3::external {
class ValueStore;
class StoredValue;
// Unique unpublished reservation. Destruction abandons it and returns backing;
// it never flushes or performs IO. The Store and its metadata outlive all handles.
class RegionLease {
    friend class ValueStore;
    ValueStore *owner_ = nullptr;
    size_t slot_ = 0;
public:
    RegionLease() = default;
    RegionLease(const RegionLease &) = delete;
    RegionLease &operator=(const RegionLease &) = delete;
    RegionLease(RegionLease &&) noexcept;
    RegionLease &operator=(RegionLease &&) noexcept;
    ~RegionLease();
    explicit operator bool() const noexcept { return owner_ != nullptr; }
    Region &region() const noexcept;
    void reset() noexcept;
    IoError publish(StoredValue &) noexcept;
};
// Immutable backing reference, also usable as an original-request IO pin.
// Copies are explicit; a moved/reset handle has no reference. The single Store
// controller owns refcount changes; workers and IO must join before releasing
// their pin on that controller. No atomics in resident arithmetic are required.
class StoredValue {
    friend class ValueStore;
    friend class RegionLease;
    ValueStore *owner_ = nullptr;
    size_t slot_ = 0;
public:
    StoredValue() = default;
    StoredValue(const StoredValue &) = delete;
    StoredValue &operator=(const StoredValue &) = delete;
    StoredValue(StoredValue &&) noexcept;
    StoredValue &operator=(StoredValue &&) noexcept;
    ~StoredValue();
    explicit operator bool() const noexcept { return owner_ != nullptr; }
    const Region &region() const noexcept;
    ValueStore *store() const noexcept { return owner_; }
    bool unique() const noexcept;
    void reset() noexcept;
    void share(StoredValue &) const noexcept;
};
struct ValueStoreStorage {
    size_t bytes = 0, alignment = 64, regions = 0, references = 0, extents = 0, slots = 0;
};
bool query_value_store(size_t slots, ValueStoreStorage &) noexcept;
struct SlotStorePlan {
    ValueStoreStorage metadata{};
    uint64_t live_byte_limit=0,slot_bytes=0,address_bytes=0;
    unsigned slot_shift=0;
};
// Virtual slots do not reserve file/RAM bytes. Each object gets its own
// address interval; allocation admission uses the sum of live object sizes.
bool query_slot_store(size_t slots,uint64_t live_byte_limit,SlotStorePlan &) noexcept;
struct RegionLifecycle {
    void *context=nullptr;
    IoError (*create)(void *,const Region &) noexcept=nullptr;
    // No IO/allocation in these notifications. Backends defer cleanup to a
    // fallible collect/create boundary; publication marks value data read-only.
    void (*retire)(void *,const Region &) noexcept=nullptr;
    void (*publish)(void *,const Region &) noexcept=nullptr;
    IoError (*attach)(void *,const Region &) noexcept=nullptr;
};
struct ValueStoreUsage {
    uint64_t live_bytes = 0, peak_bytes = 0, free_bytes = 0, largest_free = 0;
    size_t live_regions = 0, peak_regions = 0;
};
class ValueStore {
    friend class RegionLease;
    friend class StoredValue;
    struct Extent { uint64_t offset = 0, bytes = 0; };
    friend bool query_value_store(size_t, ValueStoreStorage &) noexcept;
    Region *regions_ = nullptr;
    uint64_t *references_ = nullptr;
    Extent *free_ = nullptr;
    size_t slots_ = 0, free_count_ = 0;
    uint64_t uuid_low_ = 0, uuid_high_ = 0, serial_ = 0, capacity_ = 0, backing_offset_ = 0;
    uint64_t slot_bytes_=0;
    RegionLifecycle lifecycle_{};
    ValueStoreUsage usage_{};
    void drop(size_t) noexcept;
    void retain(size_t) noexcept;
    Region &at(size_t) const noexcept;
public:
    ValueStore() = default;
    ValueStore(const ValueStore &) = delete;
    ValueStore &operator=(const ValueStore &) = delete;
    ~ValueStore();
    // Reserves no file/pages: bind a pre-admitted aligned backing interval and
    // caller-owned metadata. Metadata slots+1 guarantees infallible coalescing
    // on release, including after failed writers or while immutable pins live.
    IoError open(uint64_t uuid_low, uint64_t uuid_high, uint64_t backing_offset,
                 uint64_t backing_bytes, std::span<uint8_t> metadata, size_t slots) noexcept;
    IoError open_slots(uint64_t uuid_low,uint64_t uuid_high,const SlotStorePlan &,
                       std::span<uint8_t> metadata,const RegionLifecycle & = {}) noexcept;
    IoError reserve(uint64_t logical_bytes, RegionKind, RegionLease &) noexcept;
    // Import an immutable identity at a fresh virtual slot. The backend must
    // protect preexisting files until the caller validates/adopts all inputs.
    IoError import(const Region &,StoredValue &) noexcept;
    ValueStoreUsage usage() const noexcept;
    size_t available_regions() const noexcept {return slots_-usage_.live_regions;}
};
} // namespace sbn::v3::external
