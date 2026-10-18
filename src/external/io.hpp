// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace sbn::v3::external {
enum class IoCode { ok, invalid, capacity, overflow, io, short_io, corrupt, cancelled, incomplete, stale, incompatible, commit_unknown };
struct IoError {
    IoCode code = IoCode::ok;
    const char *where = nullptr;
    uint64_t offset = 0, need = 0, have = 0;
    int system_error = 0;
    explicit operator bool() const noexcept { return code != IoCode::ok; }
};
struct Completion {
    IoCode code = IoCode::ok;
    size_t bytes = 0;
    int system_error = 0;
};
struct Cancel {
    void *context = nullptr;
    bool (*requested)(void *) noexcept = nullptr;
    bool test() const noexcept { return requested && requested(context); }
};
struct IoCounters {
    uint64_t read_submitted = 0, write_submitted = 0;
    uint64_t read_completed = 0, write_completed = 0;
    uint64_t reads = 0, writes = 0;
};
inline constexpr size_t io_batch_max=64;
struct ReadRange {uint64_t offset=0;std::span<uint8_t> data{};};
struct ReadRequest {uint64_t offset=0;std::span<uint8_t> data{};Completion result{};};
struct WriteRequest {uint64_t offset=0;std::span<const uint8_t> data{};Completion result{};};
struct OperationTrace {
    const char*kind=nullptr;
    uint64_t n=0,m=0,elapsed_ns=0,read_bytes=0,write_bytes=0;
    unsigned primes=0,trunk_bits=0;
    size_t outer=0,inner=0;
};
// Blocking completion contract: callbacks retain no buffer/reference on return,
// including errors. Async transports must satisfy this through an explicit join.
// Metadata and data use the same ledger. Failed submitted writes consume budget.
struct Transport {
    void *context = nullptr;
    Completion (*read)(void *, uint64_t, std::span<uint8_t>) noexcept = nullptr;
    Completion (*write)(void *, uint64_t, std::span<const uint8_t>) noexcept = nullptr;
    uint64_t capacity = 0;
    size_t alignment = 1, max_transfer = 1u << 20;
    uint64_t write_limit = std::numeric_limits<uint64_t>::max();
    Cancel cancel{};
    IoCounters counters{};
    // Provider capability: read and write callback groups may overlap
    // on disjoint mutable ranges, up to the per-direction counts below.
    // The caller keeps separate read/write ledgers and
    // joins before lifecycle or snapshot operations. Cancel must be thread-safe.
    bool duplex=false;
    unsigned parallel_writes=1; // concurrent callbacks to disjoint ranges
    uint32_t write_ledger_lock=0; // joins precede copying/snapshotting a Transport
    unsigned parallel_reads=1;
    uint32_t read_ledger_lock=0;
    // Optional joined batch: every original request has ceased accessing its
    // buffer on return, including errors. Read destinations / write extents are
    // pairwise disjoint. Wrappers changing context must forward or clear these.
    void (*read_batch)(void *,std::span<ReadRequest>) noexcept=nullptr;
    void (*write_batch)(void *,std::span<WriteRequest>) noexcept=nullptr;
    unsigned batch_limit=1;
    // Optional diagnostic observer at joined coarse operation boundaries.
    // Owner-thread callback only; it must not re-enter execution or mutate IO.
    void*trace_context=nullptr;
    void (*trace)(void*,const OperationTrace&) noexcept=nullptr;
};
IoError read_exact(Transport &, uint64_t, std::span<uint8_t>) noexcept;
IoError write_exact(Transport &, uint64_t, std::span<const uint8_t>) noexcept;
// Validate/reserve the whole bounded group before dispatch. All admitted write
// bytes consume the cap even if a provider failure leaves some requests unsent.
// Results and completed ledgers remain observable after a joined error.
IoError read_batch_exact(Transport &,std::span<ReadRequest>) noexcept;
IoError write_batch_exact(Transport &,std::span<WriteRequest>) noexcept;
} // namespace sbn::v3::external
