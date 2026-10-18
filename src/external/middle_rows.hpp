// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/scratch_plane.hpp"
#include "external/chunk_writer.hpp"
#include "product/modular_row.hpp"

namespace sbn::v3::external {
struct MiddleRowsSpec {
    unsigned prime = 0, global_log = 0, workers = 1;
    size_t cells = 0;
    uint64_t first_row = 0, rows = 0, byte_offset = 0, row_stride = 0;
    uint64_t product_scale = 1;
    bool square = false;
    bool cached_square = false; // read immutable A, produce new rows once
    bool a_transformed=false; // A already completed the inner transform
};
struct MiddleRowsPlan {
    MiddleRowsSpec spec{};
    product::ModularRowInfo row{};
    size_t storage_bytes = 0, prepared = 0, scratch = 0, a = 0, b = 0;
    uint64_t byte_end = 0;
    // Optional two packed-slot pipeline. Arithmetic scratch/arrays stay shared;
    // one existing Team lane owns all IO, the other lanes compute.
    size_t serial_storage_bytes=0,pipeline_storage_bytes=0,slot_offset=0,slot_bytes=0;
    product::ModularRowInfo pipeline_row{};
    size_t compact_storage_bytes=0,second_a=0,second_b=0;
    size_t duplex_storage_bytes=0,third_a=0,third_b=0;
    product::ModularRowInfo duplex_row{};
    bool pipeline=false,recycle_a=false,compact=false,duplex=false;
    bool parallel_reads=false; // two stored inputs may load on separate Team lanes
};
struct MiddleRowsStats {
    uint64_t rows=0,read_ns=0,write_ns=0,compute_ns=0,io_wait_ns=0,compute_wait_ns=0,elapsed_ns=0;
    unsigned compute_workers=0;
    uint64_t read_wait_ns=0,write_wait_ns=0;
    bool duplex=false;
};
// First concrete wire adapter: contiguous canonical packed48 rows, optional
// stride/prefix padding. Inputs are OUTER Fourier rows before the cross twist;
// row index is bit-reversed outer frequency. Output B contains normalized inner
// coefficients with the cross twist removed. Every requested row is full.
bool query_middle_rows(const MiddleRowsSpec &, MiddleRowsPlan &) noexcept;
// Deterministic execution placement, no arithmetic/basis search. Never grows
// the supplied grant; callers may retain the synchronous plan if it cannot fit.
bool enable_middle_pipeline(MiddleRowsPlan &,size_t available) noexcept;
// Two expanded banks are also the packed IO slots. No separate packed copies.
// Currently applies to ordinary MUL/SQR with chunk-aligned plane rows.
bool enable_middle_compact_pipeline(MiddleRowsPlan &,size_t available) noexcept;
// Disposable packed A: finish its first two rows before borrowing those dead
// bytes as the two B staging slots. No additional storage or IO. This plan must
// be executed through run_middle_consuming_a, never the immutable adapters.
bool enable_middle_recycled_pipeline(MiddleRowsPlan &) noexcept;
// Three slots, separate read/write owners, remaining Team lanes compute. Raw A
// borrows three consumed rows; compact mode reserves a third expanded bank.
// Providers without duplex capability use the two-stage schedule in this grant.
bool enable_middle_duplex(MiddleRowsPlan &,size_t available) noexcept;
// Plane index/IO buffers and context tables/stacks are separate admitted grants.
// MUL preserves A's backing; SQR requires &a == &b and reserves one row array.
// B's index stays in RAM on success. Failure discards B; original value inputs
// are outside these explicitly disposable planes.
IoError run_middle_rows(const MiddleRowsPlan &, ScratchPlane &a, ScratchPlane &b,
                        std::span<uint8_t> workspace, sbn3_team &,MiddleRowsStats * = nullptr) noexcept;
IoError run_middle_cached_square(const MiddleRowsPlan &,ScratchPlane &,ChunkWriter &,
                                std::span<uint8_t>,sbn3_team &,MiddleRowsStats * = nullptr) noexcept;
// Same arithmetic with an immutable caller-owned canonical packed48 A prime,
// disjoint from workspace through this synchronous call.
IoError run_middle_memory_a(const MiddleRowsPlan &,std::span<const uint8_t> a,ScratchPlane &b,
                           std::span<uint8_t> workspace,sbn3_team &,MiddleRowsStats * = nullptr) noexcept;
// A is private disposable scratch, not a value/cache. The first two packed
// rows may be overwritten after their last compute read; other rows survive.
// No aliases or IO references to A may remain on entry/return.
IoError run_middle_consuming_a(const MiddleRowsPlan &,std::span<uint8_t> a,ScratchPlane &b,
                              std::span<uint8_t> workspace,sbn3_team &,MiddleRowsStats * = nullptr) noexcept;
} // namespace sbn::v3::external
