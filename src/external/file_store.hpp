// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/values.hpp"
#include "runtime/linux/uring.hpp"
namespace sbn::v3::external {
struct FileStoreStorage {size_t bytes=0,alignment=64;};
bool query_file_store(size_t slots,FileStoreStorage &) noexcept;
struct FileStoreUsage {
    size_t open_files=0,live_files=0,retired_files=0;
    uint64_t claimed_bytes=0,retired_bytes=0,peak_claimed_bytes=0;
    uint64_t created=0,removed=0,read_calls=0,write_calls=0,read_bytes=0,write_bytes=0,sync_calls=0,imported=0,detached_bytes=0;
};
// Linux object-file adapter. One small registry maps each virtual slot to a
// compact file; a slot's address span is never allocated or truncated on disk.
// All files are ephemeral here; durable publication/import is a separate layer.
class FileStore {
    friend bool query_file_store(size_t,FileStoreStorage &) noexcept;
    struct Entry {
        int fd=-1;
        RegionId id{};
        uint64_t bytes=0;
        bool live=false,retired=false,readonly=false,keep=false,synced=false;
        char name[64]{};
    };
    Entry *entries_=nullptr;
    SlotStorePlan plan_{};
    int directory_=-1;
    bool memory_=true,direct_=false,detaching_=false,suspended_=false;
    FileStoreUsage usage_{};
    os::Uring read_ring_,write_ring_;
    uint32_t read_ring_lock_=0,write_ring_lock_=0;
    IoError open(const SlotStorePlan &,std::span<uint8_t>,int directory,bool direct) noexcept;
    IoError create(const Region &) noexcept;
    IoError attach(const Region &) noexcept;
    void retire(const Region &) noexcept;
    void published(const Region &) noexcept;
    Entry *entry(const Region &) const noexcept;
    Completion transfer(uint64_t,void *,size_t,bool write) noexcept;
    static IoError create_thunk(void*,const Region&) noexcept;
    static void retire_thunk(void*,const Region&) noexcept;
    static void publish_thunk(void*,const Region&) noexcept;
    static IoError attach_thunk(void*,const Region&) noexcept;
    static Completion read_thunk(void*,uint64_t,std::span<uint8_t>) noexcept;
    static Completion write_thunk(void*,uint64_t,std::span<const uint8_t>) noexcept;
    static void read_batch_thunk(void*,std::span<ReadRequest>) noexcept;
    static void write_batch_thunk(void*,std::span<WriteRequest>) noexcept;
public:
    FileStore()=default;
    FileStore(const FileStore&)=delete;
    FileStore&operator=(const FileStore&)=delete;
    ~FileStore(); // requires explicit successful close; performs no IO
    IoError open_memory(const SlotStorePlan &,std::span<uint8_t>) noexcept;
    // Duplicates a caller-supplied exclusive directory descriptor. Direct IO
    // is explicit; creation failures never silently fall back to buffered IO.
    IoError open_directory(const SlotStorePlan &,std::span<uint8_t>,int directory,bool direct=true) noexcept;
    RegionLifecycle lifecycle() noexcept;
    // Data callbacks support the advertised reader/writer counts on disjoint
    // mutable ranges. All control/lifecycle/usage snapshots require prior join.
    Transport transport(Cancel={}) noexcept;
    // Explicit, quiescent context preparation. No arithmetic-time allocation,
    // implicit fallback or extra user threads. Scalar callbacks stay available.
    // The two bounded rings permit a reader batch and a writer batch to overlap.
    // Bounded fallback workers are an explicit context resource, separate from
    // queue depth. They may run when the filesystem cannot issue IO inline.
    IoError enable_uring(unsigned depth,unsigned workers=1) noexcept;
    static constexpr size_t uring_memory_bound(unsigned depth) noexcept {return 2*os::Uring::memory_bound(depth);}
    size_t uring_mapped_bytes() const noexcept {return read_ring_.mapped_bytes()+write_ring_.mapped_bytes();}
    os::UringUsage uring_read_usage() const noexcept {return read_ring_.usage();}
    os::UringUsage uring_write_usage() const noexcept {return write_ring_.usage();}
    // Drop/retire only marks metadata. Before ANY new object creation, all
    // retired files are collected, so deferred cleanup cannot inflate the
    // historical sum-of-live-bytes peak. Cleanup failures block new creation.
    IoError collect() noexcept;
    IoError sync(const Region &) noexcept;
    IoError close() noexcept;
    // Imported files are protected until complete verification succeeds.
    // Failed import cleanup closes them without deleting original root data.
    IoError adopt(const Region &) noexcept;
    // Preserve selected committed files on close; forbids further creation.
    IoError preserve(const Region &) noexcept;
    void suspend_creates(bool value) noexcept {suspended_=value;}
    int directory_descriptor() const noexcept {return directory_;} // borrowed
    size_t slot_capacity() const noexcept {return plan_.metadata.slots;}
    // Exclusive checkpoint directory, after restore/commit. Removes only
    // well-formed object names absent from live mappings. Scratch needs
    // 64*max_files bytes, plus a directory enumeration page.
    IoError collect_orphans(std::span<uint8_t> scratch,size_t max_files,bool checkpoint_drafts=false) noexcept;
    FileStoreUsage usage() const noexcept {return usage_;}
    // Borrowed while the owning Region is live. Caller must not close it.
    // Useful for durable import/export adapters and fault-injection fixtures.
    int descriptor(const Region &) const noexcept;
};
} // namespace sbn::v3::external
