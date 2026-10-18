// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/io.hpp"
#include "external/checksum.hpp"

namespace sbn::v3::external {
inline constexpr size_t chunk_bytes = 16384, index_page_bytes = 4096, index_header_bytes = 32;
inline constexpr size_t records_per_page = (index_page_bytes-index_header_bytes)/8;
inline constexpr uint64_t extent_alignment = 65536;
inline constexpr uint64_t index_magic = 0x315844494e4253ULL;
struct RegionLayout {
    uint64_t logical_bytes = 0, chunks = 0;
    uint64_t data_offset = 0, data_bytes = 0, index_offset = 0, index_bytes = 0, allocation_bytes = 0;
    bool operator==(const RegionLayout &) const = default;
};
IoError region_layout(uint64_t logical_bytes, uint64_t offset, RegionLayout &) noexcept;
enum class RegionState { empty, building, writing, sealed, scratch_owned, invalid };
enum class RegionKind { value, scratch };
struct Region {
    RegionId id{};
    uint64_t index_instance = 0;
    RegionLayout layout{};
    RegionState state = RegionState::empty;
    RegionKind kind = RegionKind::value;
    uint64_t revision = 1; // runtime value-view epoch; not part of the stable chunk checksum domain
};
// Sealed means initialized and integrity-indexed, not durably committed or
// arithmetically checked. Immutable descriptors remain caller-owned and pinned.
IoError begin_region(Region &, RegionId, uint64_t index_instance, const RegionLayout &,
                     RegionKind = RegionKind::value) noexcept;
struct StoreBuffers {
    // Data is exclusive during each synchronous call. The index page remains
    // exclusive while a reader caches it; reopen after another phase reuses it.
    std::span<uint8_t> data;  // >=1 integrity chunk, multiple chunks; private ISA-free storage
    std::span<uint8_t> index; // exactly one page used; same transport alignment
};
// Position of the first/last nonzero whole limb, accumulated from finalized
// chunks in any order. Valid only after the enclosing writer seals. No extra
// normalization IO; padding outside logical_bytes is excluded.
struct WordExtent {
    uint64_t first=UINT64_MAX,end=0,top_word=0,residue_m61=0;
    void observe(uint64_t word_offset,std::span<const uint8_t>,bool checksum=true,bool complement=false) noexcept;
};
IoError validate_store_region(const Transport &, const Region &, StoreBuffers) noexcept;
void finalize_index_page(const Region &, uint64_t first_record, std::span<uint8_t>) noexcept;
bool valid_index_page(const Region &, uint64_t page_number, std::span<uint8_t>) noexcept;
// Sequential finalized-value writer. No destructor IO, allocation or implicit
// publication. A failure poisons only this unpublished region; reinitialize
// with a fresh incarnation to reuse its storage. Scratch-plane overwrite and
// durable commit use separate contracts, not this immutable-value writer.
class ValueWriter {
    Transport *transport_ = nullptr;
    Region *region_ = nullptr;
    StoreBuffers buffers_{};
    uint64_t accepted_ = 0, written_chunks_ = 0, first_record_ = 0;
    size_t buffered_ = 0, records_ = 0;
    IoError error_{};
    WordExtent *extent_=nullptr;
    IoError fail(IoError) noexcept;
    size_t batch_capacity() const noexcept;
    IoError flush_data(size_t) noexcept;
    IoError write_data(std::span<const uint8_t>) noexcept;
    IoError flush_index() noexcept;
public:
    ValueWriter() = default;
    ValueWriter(const ValueWriter &) = delete;
    ValueWriter &operator=(const ValueWriter &) = delete;
    IoError open(Transport &, Region &, StoreBuffers,WordExtent * = nullptr) noexcept;
    // Input must be disjoint from both retained IO buffers.
    IoError append(std::span<const uint8_t>) noexcept;
    IoError seal() noexcept;
    void abandon() noexcept;
};
// Range reads return only verified bytes on success. On failure the destination
// is discardable. One immediate reread of a bad page/chunk; no short-IO loop.
class ValueReader {
    Transport *transport_ = nullptr;
    const Region *region_ = nullptr;
    StoreBuffers buffers_{};
    uint64_t cached_page_ = UINT64_MAX;
    RegionId opened_id_{};
    uint64_t opened_index_ = 0, opened_revision_ = 0;
    std::span<uint8_t> pinned_index_{};
    std::span<const uint8_t> memory_{};
    IoError index_page(uint64_t, bool fresh = false) noexcept;
    IoError verify(uint64_t, std::span<uint8_t>) noexcept;
public:
    ValueReader() = default;
    ValueReader(const ValueReader &) = delete;
    ValueReader &operator=(const ValueReader &) = delete;
    IoError open(Transport &, const Region &, StoreBuffers) noexcept;
    // Borrow a stable, already checked resident value for this reader's
    // lifetime. No transport/IO or checksum-index allocation is involved.
    IoError open_memory(std::span<const uint8_t>) noexcept;
    // Caller admits the complete immutable index image. It is verified once;
    // subsequent strided reads/retries use it without metadata rereads.
    IoError open_pinned(Transport &, const Region &, StoreBuffers, std::span<uint8_t> index_image) noexcept;
    IoError read(uint64_t offset, std::span<uint8_t>) noexcept;
};
} // namespace sbn::v3::external
