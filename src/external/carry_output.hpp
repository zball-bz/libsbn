// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/chunk_writer.hpp"
#include "product/crt.hpp"
namespace sbn::v3::external {
struct CarryFragment {uint64_t at=0;size_t bytes=0;};
struct CarrySegment {
    product::CrtCarry tail{};
    uint8_t guard[64]{};
    uint64_t ff_bytes=0;
    CarryFragment fragments[2]{};
    uint16_t endpoint=256; // 256: no endpoint, remaining segment is all 0xff
    bool present=false,resolved=false;
};
struct CarryOutputPlan {
    uint64_t segments=0,segment_bytes=0,output_bytes=0,coefficient_bytes=0,visible_segments=0;
    size_t descriptor_bytes=0,fragment_bytes=0,assembly_bytes=chunk_bytes;
    unsigned slots_per_segment=0;
    uint64_t stream_segments=0; // consecutive segments per ordered row; zero permits arbitrary arrival
    bool extra_last_slot=false,cyclic=false;
};
bool query_carry_output(uint64_t segments,uint64_t segment_bytes,uint64_t output_bytes,CarryOutputPlan &,bool cyclic=false,
                        uint64_t stream_segments=0) noexcept;
// Positive normalization: each segment is evaluated with incoming carry zero.
// Stable full chunks are written once at submit, arbitrary edge bytes retained,
// and long leading 0xff runs kept symbolically. No result reads or corrections.
class CarryOutput {
    CarryOutputPlan plan_{};
    ChunkWriter *writer_=nullptr;
    std::span<CarrySegment> segments_{};
    std::span<uint8_t> fragments_{},assembly_{};
    uint64_t submitted_=0,position_=0;
    size_t fill_=0;
    bool finished_=false;
    uint64_t m61_=0;
    bool middle_=false,gap_nonzero_=false;
    uint64_t gap_first_=0,gap_end_=0;
    WordExtent full_extent_{},complement_extent_{};
    IoError error_{};
    std::span<uint8_t> fragment(uint64_t sid,unsigned slot) noexcept;
    IoError fail(IoError) noexcept;
    IoError append(uint64_t at,std::span<const uint8_t>) noexcept;
    IoError fill(uint64_t at,uint8_t value,uint64_t count) noexcept;
    IoError publish() noexcept;
    void observe(uint64_t at,std::span<const uint8_t>) noexcept;
public:
    IoError open(const CarryOutputPlan &,ChunkWriter &,std::span<CarrySegment>,std::span<uint8_t> fragments,
                 std::span<uint8_t> assembly) noexcept;
    // Ordered rows consume the mutable segment once its incoming carry is
    // known. Rows may interleave, but segments within each row arrive in order.
    IoError submit(uint64_t sid,std::span<uint8_t> normalized,const product::CrtCarry &) noexcept;
    // Resolves in logical order irrespective of submit order. Leaves the output
    // writer unsealed so the caller can finish required arithmetic checks first.
    IoError finish() noexcept;
    uint64_t residue() const noexcept {return m61_;}
    // Change the arithmetic check to m64 and observe this exact gap while
    // finalizing existing chunks; no reread or additional payload buffer.
    void inspect_middle(uint64_t first_byte,uint64_t end_byte) noexcept;
    bool middle_exact() const noexcept {return gap_first_==gap_end_||gap_nonzero_;}
    WordExtent full_extent() const noexcept {return full_extent_;}
    WordExtent complement_extent() const noexcept {return complement_extent_;}
};
} // namespace sbn::v3::external
