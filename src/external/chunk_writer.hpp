// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/scratch_plane.hpp"
namespace sbn::v3::external {
class ChunkWriter;
struct ChunkWrite {ChunkWriter*writer=nullptr;uint64_t first_chunk=0;std::span<const uint8_t>data{};};
struct ChunkWriterStorage {size_t index_bytes=0,coverage_bytes=0;};
IoError chunk_writer_storage(const RegionLayout &,ChunkWriterStorage &) noexcept;
// Write-once finalized chunks in arbitrary order. Coverage is independent of
// checksum values. No old payload read or preliminary zero-plane write.
class ChunkWriter {
    Transport *transport_=nullptr;
    Region *region_=nullptr;
    StoreBuffers io_{};
    std::span<uint8_t> image_{},coverage_{},retained_{};
    uint64_t written_=0;
    uint64_t logical_bytes_=0,first_byte_=0;
    bool dispatched_=false;
    IoError error_{};
    WordExtent *extent_=nullptr;
    IoError fail(IoError) noexcept;
    IoError complete() noexcept;
public:
    ChunkWriter()=default;
    ChunkWriter(const ChunkWriter &)=delete;
    ChunkWriter &operator=(const ChunkWriter &)=delete;
    // Optional chunk-aligned scratch prefix stays resident through handoff.
    // Its bytes and index storage remain exclusive until plane release/discard.
    IoError open(Transport &,Region &,StoreBuffers,std::span<uint8_t> index_image,std::span<uint8_t> coverage,
                 std::span<uint8_t> retained_prefix={},WordExtent * = nullptr) noexcept;
    // Retain [first_byte, first_byte+region.logical_bytes) from a larger producer
    // stream. First byte is integrity-chunk aligned. Configure before dispatch,
    // for value-kind output only; upstream still resolves/checks the full stream.
    IoError project(uint64_t logical_bytes,uint64_t first_byte) noexcept;
    // Complete chunks only, except the region's final covered short chunk.
    // Aligned full chunks can transfer directly from the stable source span.
    IoError write(uint64_t first_chunk,std::span<const uint8_t>) noexcept;
    // Joined scatter batch over stable, complete, aligned chunks. Unsupported
    // partial/projected shapes retain scalar semantics. No source span escapes.
    static IoError write_many(std::span<const ChunkWrite>) noexcept;
    IoError seal() noexcept;
    // Ownership moves without index IO. Dirty bits may reuse coverage after
    // complete coverage has been checked; all index pages start dirty.
    IoError finish_plane(ScratchPlane &,std::span<uint8_t> dirty_pages) noexcept;
    void abandon() noexcept;
    const Region *region() const noexcept {return region_;}
    Transport *transport() noexcept {return transport_;}
    const Transport *transport() const noexcept {return transport_;}
    uint64_t written_chunks() const noexcept {return written_;}
    uint64_t logical_bytes() const noexcept {return logical_bytes_;}
};
} // namespace sbn::v3::external
