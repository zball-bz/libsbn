// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/store.hpp"
#include "sbn3/team.h"

namespace sbn::v3::external {
class ScratchPlane;
class PlaneView {
    friend class ScratchPlane;
    const ScratchPlane *owner_ = nullptr;
    uint64_t epoch_ = 0;
};
class PlaneUpdate {
    friend class ScratchPlane;
    ScratchPlane *owner_ = nullptr;
    uint64_t epoch_ = 0;
public:
    PlaneUpdate() = default;
    PlaneUpdate(const PlaneUpdate &) = delete;
    PlaneUpdate &operator=(const PlaneUpdate &) = delete;
};
struct PlaneIndexStorage {
    size_t image_bytes = 0, dirty_bytes = 0;
    size_t alignment = index_page_bytes;
};
IoError plane_index_storage(const RegionLayout &, PlaneIndexStorage &) noexcept;
// Borrowed full-chunk reader for a joined stage. Descriptor values are copied;
// index/retained spans are borrowed. A concurrent writer may touch only disjoint
// chunks (including their index words). It never mutates ScratchPlane state.
// The caller owns the read Transport copy and merges its read counters on join.
class PlaneChunkReader {
    friend class ScratchPlane;
    RegionId id_{};RegionLayout layout_{};Transport*transport_=nullptr;
    std::span<const uint8_t>image_{},retained_{};
public:
    IoError read(uint64_t offset,std::span<uint8_t>) noexcept;
    IoError read_many(std::span<const ReadRange>) noexcept;
};
// Single controlling owner, synchronous original-IO completion. Ordinary reads
// return verified copies. The explicit resident callback borrows an exclusive
// alias only between joined integrity passes. No active kernel/IO may
// still need the old backing when its coordinator starts a consuming update.
class ScratchPlane {
    friend class ChunkWriter;
    Region *region_ = nullptr;
    Transport *transport_ = nullptr;
    StoreBuffers io_{};
    std::span<uint8_t> image_{}, dirty_{}, retained_{};
    uint64_t epoch_ = 1;
    bool editing_ = false, resident_busy_ = false;
    IoError error_{};
    IoError fail(IoError) noexcept;
    IoError read_current(uint64_t, std::span<uint8_t>) noexcept;
    IoError read_chunks(uint64_t first, std::span<uint8_t>) noexcept;
    IoError flush() noexcept;
    IoError materialize() noexcept;
    uint64_t expectation(uint64_t chunk) const noexcept;
    void record(uint64_t chunk, std::span<const uint8_t>) noexcept;
    IoError attach(Transport &, Region &, StoreBuffers, std::span<uint8_t>, std::span<uint8_t>, bool created,
                   std::span<uint8_t> retained={}) noexcept;
    struct ResidentCall;
public:
    ScratchPlane() = default;
    ScratchPlane(const ScratchPlane &) = delete;
    ScratchPlane &operator=(const ScratchPlane &) = delete;
    // Consumes an exclusively owned, initialized scratch region. Value-kind
    // regions are rejected. Index image is verified before ownership changes.
    IoError take(Transport &, Region &, StoreBuffers, std::span<uint8_t> index_image,
                 std::span<uint8_t> dirty_pages) noexcept;
    IoError view(PlaneView &) const noexcept;
    IoError read(PlaneView, uint64_t offset, std::span<uint8_t>) noexcept;
    IoError read_many(PlaneView,std::span<const ReadRange>) noexcept;
    IoError begin_update(PlaneUpdate &) noexcept;
    IoError read(const PlaneUpdate &, uint64_t offset, std::span<uint8_t>) noexcept;
    IoError borrow_chunks(PlaneView,Transport &read_lane,PlaneChunkReader &) const noexcept;
    IoError borrow_chunks(const PlaneUpdate &,Transport &read_lane,PlaneChunkReader &) const noexcept;
    Transport *transport() const noexcept {return transport_;}
    // Exact byte patches preserve verified unowned chunk guards; complete
    // replacements avoid the old-data read. Input is disjoint from IO/index.
    IoError patch(const PlaneUpdate &, uint64_t offset, std::span<const uint8_t>) noexcept;
    using ResidentAction = void (*)(void *,std::span<uint8_t>,sbn3_team_scope *);
    size_t resident_bytes() const noexcept {return retained_.size();}
    // No IO or allocation. Verify covered resident chunks, run one synchronous
    // native callback, and refresh expectations in parallel on the same team.
    // The alias cannot escape or re-enter this plane. finish() still closes the
    // caller's update token; integrity failure invalidates private scratch.
    IoError edit_resident(PlaneUpdate &,uint64_t offset,size_t bytes,sbn3_team &,ResidentAction,void *) noexcept;
    // Explicit persistence materializes the resident prefix before the index.
    IoError finish(PlaneUpdate &, bool persist = false) noexcept;
    // Release materializes payload and dirty index before generic Store reads.
    IoError release() noexcept;
    // Disposable state needs neither retained-payload nor index writeback.
    // No destructor IO.
    void discard() noexcept;
    uint64_t logical_bytes() const noexcept { return region_ ? region_->layout.logical_bytes : 0; }
    bool overlaps_backing(const ScratchPlane &) const noexcept;
};
} // namespace sbn::v3::external
