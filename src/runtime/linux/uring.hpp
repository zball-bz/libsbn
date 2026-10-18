// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
namespace sbn::v3::os {
// Single owner, no allocation in transfer(). open/close are explicit resource
// boundaries. Buffers stay borrowed until every original CQE has been reaped.
struct UringRequest {
    int fd=-1;
    void *data=nullptr;
    uint64_t offset=0;
    unsigned bytes=0;
    bool write=false;
    int result=0; // bytes, or -errno
};
struct UringUsage {uint64_t batches=0,requests=0,enters=0;};
class Uring {
    int fd_=-1;
    unsigned entries_=0;
    unsigned *sq_head_=nullptr,*sq_tail_=nullptr,*sq_mask_=nullptr,*sq_array_=nullptr;
    unsigned *cq_head_=nullptr,*cq_tail_=nullptr,*cq_mask_=nullptr;
    void *sqes_=nullptr,*cqes_=nullptr,*sq_map_=nullptr,*cq_map_=nullptr;
    size_t sq_bytes_=0,cq_bytes_=0,sqe_bytes_=0;
    bool single_=false,busy_=false;
    UringUsage usage_{};
public:
    static constexpr unsigned max_entries=64;
    // Native Linux 4 KiB pages, no SQPOLL/IOPOLL or extra kernel flags.
    static constexpr size_t memory_bound(unsigned entries) noexcept {
        return entries&&entries<=max_entries&&!(entries&(entries-1))?3*4096+((entries*100+4095)/4096)*4096:0;
    }
    Uring()=default;Uring(const Uring&)=delete;Uring&operator=(const Uring&)=delete;
    ~Uring();
    int open(unsigned entries,unsigned workers=1) noexcept;
    void close() noexcept; // quiescent only; performs no cancellation
    int transfer(std::span<UringRequest>) noexcept;
    unsigned entries() const noexcept{return entries_;}
    size_t mapped_bytes() const noexcept{return sq_bytes_+(single_?0:cq_bytes_)+sqe_bytes_;}
    UringUsage usage() const noexcept{return usage_;}
};
}
