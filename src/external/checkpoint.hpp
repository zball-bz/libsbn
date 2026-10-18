// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/job.hpp"
#include "external/file_store.hpp"
#include <array>
namespace sbn::v3::external {
struct CheckpointKey {uint64_t word[4]{};bool operator==(const CheckpointKey&)const=default;};
enum class CheckpointEvent { values_sync,data_directory_sync,manifest_open,manifest_write,manifest_sync,rename_root,directory_sync };
struct CheckpointHook {void*context=nullptr;IoError(*before)(void*,CheckpointEvent)noexcept=nullptr;};
struct CheckpointUsage {uint64_t manifest_reads=0,manifest_writes=0,manifest_bytes_read=0,manifest_bytes_submitted=0,manifest_bytes_written=0,commits=0,resolves=0;};
// IO scratch for import verification and a bounded orphan-name snapshot.
size_t checkpoint_storage_bytes(size_t file_slots) noexcept;
// Strict same-build/job phase checkpoints in an exclusive object directory.
// The explicit compatibility key must identify the application's formula code
// and numeric library build; the in-process job cookie alone is insufficient.
class Checkpoint {
    struct Record {
        unsigned role=0;
        Region region{};
        int64_t exponent=0;
        bool negative=false;
        size_t first=0,words=0;
        WordExtent extent{};
    };
    struct Snapshot {
        uint64_t generation=0,job=0,charged=0,reserved=0,certified=0,integer=0,slots=0;
        JobPhase next=JobPhase::series;
        unsigned count=0;
        Record records[5]{};
    };
    FileStore*files_=nullptr;
    int directory_=-1,lock_=-1;
    CheckpointKey key_{};
    CheckpointHook hook_{};
    Snapshot current_{},pending_{};
    StoredValue pins_[5]{},pending_pins_[5]{};
    std::array<uint8_t,4096> current_image_{},pending_image_{};
    uint64_t nonce_=0,attempt_=0;
    uint64_t manifest_write_limit_=UINT64_MAX;
    bool loaded_=false,exists_=false,unknown_=false,needs_gc_=false;
    CheckpointUsage usage_{};
    IoError event(CheckpointEvent) noexcept;
    IoError capture(const JobPlan&,const JobState&,uint64_t,Snapshot&,StoredValue*) noexcept;
    void encode(const Snapshot&,std::span<uint8_t>) const noexcept;
    IoError decode(const JobPlan&,std::span<uint8_t>,Snapshot&) const noexcept;
    IoError read_head(std::span<uint8_t>,bool&found) noexcept;
    IoError publish(const JobPlan&,const JobState&,uint64_t reservation,bool ignore_cancel,Cancel) noexcept;
    void promote() noexcept;
    bool matches(const JobState&) const noexcept;
public:
    Checkpoint()=default;
    Checkpoint(const Checkpoint&)=delete;
    Checkpoint&operator=(const Checkpoint&)=delete;
    ~Checkpoint(); // explicit close_preserving; no destructor IO
    IoError open(FileStore&,CheckpointKey,CheckpointHook={}) noexcept;
    // No existing HEAD: start with an initial state. Existing HEAD must be
    // restored/verified before any replacement or orphan cleanup is permitted.
    IoError restore(const JobPlan&,JobState&,Transport&,ValueStore&,std::span<uint8_t>) noexcept;
    IoError commit(const JobPlan&,const JobState&,Cancel={}) noexcept;
    IoError resolve() noexcept;
    // Durably reserve this phase's worst writes BEFORE execution. After a
    // crash the reservation stays charged; normal return reconciles actual
    // submitted writes. Caller retries metadata publication before further math.
    IoError advance(JobPlan&,JobState&,Transport&,ValueStore&,sbn3_arena&,size_t,sbn3_team&) noexcept;
    IoError collect(std::span<uint8_t>scratch) noexcept;
    // Mark current/uncertain root files for preservation, release checkpoint
    // pins and close metadata descriptors. Then discard caller JobState and
    // explicitly close FileStore. No new objects may be created after this.
    IoError close_preserving() noexcept;
    bool unknown()const noexcept{return unknown_;}
    uint64_t generation()const noexcept{return current_.generation;}
    CheckpointUsage usage()const noexcept{return usage_;}
    // Metadata writes have a separate per-writer allowance/counter. The job
    // ledger and crash reservation cover operand/index writes, not manifests.
    void set_manifest_write_limit(uint64_t bytes) noexcept {manifest_write_limit_=bytes;}
};
} // namespace sbn::v3::external
