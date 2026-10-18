// SPDX-License-Identifier: LGPL-3.0-or-later
/* Adapted from libsbn v2 pal_uring.c / pal_uring.h, originally lifted from
 * labs/swap/iouring_bench.c. Raw setup + shared SQ/CQ/SQE mappings; no liburing.
 * The v3 owner is caller-provided, with explicit preparation and a joined
 * bounded batch. The old calloc/free handle and permissive teardown are gone.
 * Atomics pair with the KERNEL's shared rings, not other user submitters. */
#include "runtime/linux/uring.hpp"
#include "common/checked.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
namespace sbn::v3::os {
namespace {size_t pages(size_t n){return (n+4095)&~size_t(4095);}}
Uring::~Uring(){require(fd_<0&&!busy_,SBN3_FATAL_LIFETIME,"io_uring requires explicit quiescent close");}
int Uring::open(unsigned entries,unsigned worker_limit) noexcept {
    if(fd_>=0||!entries||entries>max_entries||(entries&(entries-1))||!worker_limit||worker_limit>max_entries)return -EINVAL;
    if(sysconf(_SC_PAGESIZE)!=4096)return -ENOTSUP;
    io_uring_params p{};p.flags=IORING_SETUP_CQSIZE;p.cq_entries=2*entries;
    fd_=int(syscall(SYS_io_uring_setup,entries,&p));if(fd_<0)return -errno;
    if(!(p.features&IORING_FEAT_NODROP)||p.sq_entries!=entries||p.cq_entries!=2*entries){close();return -ENOTSUP;}
    entries_=p.sq_entries;single_=p.features&IORING_FEAT_SINGLE_MMAP;
    sq_bytes_=pages(p.sq_off.array+p.sq_entries*sizeof(unsigned));cq_bytes_=pages(p.cq_off.cqes+p.cq_entries*sizeof(io_uring_cqe));
    if(single_)sq_bytes_=cq_bytes_=std::max(sq_bytes_,cq_bytes_);
    sqe_bytes_=pages(p.sq_entries*sizeof(io_uring_sqe));
    if(mapped_bytes()>memory_bound(entries)){close();return -EOVERFLOW;}
    auto map=[&](size_t bytes,uint64_t offset,void*&ptr){
        void*x=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,fd_,off_t(offset));
        if(x==MAP_FAILED)return errno;ptr=x;return 0;};
    int error=map(sq_bytes_,IORING_OFF_SQ_RING,sq_map_);
    if(!error){if(single_)cq_map_=sq_map_;else error=map(cq_bytes_,IORING_OFF_CQ_RING,cq_map_);}
    if(!error)error=map(sqe_bytes_,IORING_OFF_SQES,sqes_);
    if(error){close();return -error;}
    auto*sq=static_cast<uint8_t*>(sq_map_);auto*cq=static_cast<uint8_t*>(cq_map_);
    sq_head_=reinterpret_cast<unsigned*>(sq+p.sq_off.head);sq_tail_=reinterpret_cast<unsigned*>(sq+p.sq_off.tail);
    sq_mask_=reinterpret_cast<unsigned*>(sq+p.sq_off.ring_mask);sq_array_=reinterpret_cast<unsigned*>(sq+p.sq_off.array);
    cq_head_=reinterpret_cast<unsigned*>(cq+p.cq_off.head);cq_tail_=reinterpret_cast<unsigned*>(cq+p.cq_off.tail);
    cq_mask_=reinterpret_cast<unsigned*>(cq+p.cq_off.ring_mask);cqes_=cq+p.cq_off.cqes;
    // No forced io-wq execution. Bound fallback pools as part of preparation.
    unsigned workers[2]{worker_limit,1};
    if(syscall(SYS_io_uring_register,fd_,IORING_REGISTER_IOWQ_MAX_WORKERS,workers,2)<0){error=errno;close();return -error;}
    usage_={};return 0;
}
void Uring::close() noexcept {
    require(!busy_,SBN3_FATAL_LIFETIME,"io_uring close before original completions");
    if(sqes_)munmap(sqes_,sqe_bytes_);
    if(cq_map_&&!single_)munmap(cq_map_,cq_bytes_);
    if(sq_map_)munmap(sq_map_,sq_bytes_);
    if(fd_>=0)::close(fd_);
    fd_=-1;entries_=0;sqes_=cqes_=sq_map_=cq_map_=nullptr;sq_bytes_=cq_bytes_=sqe_bytes_=0;
    sq_head_=sq_tail_=sq_mask_=sq_array_=cq_head_=cq_tail_=cq_mask_=nullptr;
}
int Uring::transfer(std::span<UringRequest> requests) noexcept {
    for(auto&r:requests)r.result=-EINVAL;
    if(fd_<0||busy_||requests.size()>entries_)return -EINVAL;
    if(requests.empty())return 0;
    const unsigned first=__atomic_load_n(sq_head_,__ATOMIC_ACQUIRE),tail=*sq_tail_;
    require(first==tail&&*cq_head_==__atomic_load_n(cq_tail_,__ATOMIC_ACQUIRE),SBN3_FATAL_LIFETIME,"io_uring batch still live");
    for(auto&r:requests)if(!r.data||!r.bytes||r.bytes>INT32_MAX)return -EINVAL;
    busy_=true;++usage_.batches;usage_.requests+=requests.size();
    auto*sqes=static_cast<io_uring_sqe*>(sqes_);
    for(unsigned i=0;i<requests.size();++i){auto&r=requests[i];const unsigned slot=(tail+i)&*sq_mask_;auto&s=sqes[slot];
        s={};s.opcode=r.write?IORING_OP_WRITE:IORING_OP_READ;s.fd=r.fd;s.addr=uintptr_t(r.data);s.len=r.bytes;s.off=r.offset;s.user_data=i;
        sq_array_[slot]=slot;r.result=-ECANCELED;
    }
    const unsigned end=tail+unsigned(requests.size());__atomic_store_n(sq_tail_,end,__ATOMIC_RELEASE);
    int error=0;unsigned consumed=0;
    while(consumed<requests.size()){
        const unsigned previous=consumed;
        ++usage_.enters;const long rc=syscall(SYS_io_uring_enter,fd_,unsigned(requests.size())-consumed,0,0,nullptr,0);
        consumed=__atomic_load_n(sq_head_,__ATOMIC_ACQUIRE)-first;
        require(consumed<=requests.size(),SBN3_FATAL_LIFETIME,"io_uring SQ ownership");
        if(rc>=0&&consumed>previous)continue;
        if(rc<0&&errno==EINTR)continue;
        error=rc>=0?-EAGAIN:-errno;
        // No SQPOLL: after enter returns, unsent SQEs have no kernel owner.
        __atomic_store_n(sq_tail_,first+consumed,__ATOMIC_RELEASE);
        for(size_t i=consumed;i<requests.size();++i)requests[i].result=error;
        break;
    }
    uint64_t seen=0;unsigned done=0;auto*cqes=static_cast<io_uring_cqe*>(cqes_);
    while(done<consumed){
        unsigned head=*cq_head_;const unsigned available=__atomic_load_n(cq_tail_,__ATOMIC_ACQUIRE);
        while(head!=available){const auto&c=cqes[head&*cq_mask_];
            require(c.user_data<consumed&&!(seen&(uint64_t(1)<<c.user_data)),SBN3_FATAL_LIFETIME,"io_uring original CQ identity");
            seen|=uint64_t(1)<<c.user_data;requests[size_t(c.user_data)].result=c.res;++head;++done;
        }
        __atomic_store_n(cq_head_,head,__ATOMIC_RELEASE);
        if(done==consumed)break;
        ++usage_.enters;
        if(syscall(SYS_io_uring_enter,fd_,0,consumed-done,IORING_ENTER_GETEVENTS,nullptr,0)<0){
            const int why=errno;
            // An invalid exclusively owned ring is an internal lifetime bug.
            // Transient pressure/signals cannot release still-borrowed buffers.
            require(why!=EBADF&&why!=EINVAL&&why!=ENOSYS,SBN3_FATAL_LIFETIME,"io_uring invalid owned wait");
            if(why!=EINTR){timespec pause{0,1000000};nanosleep(&pause,nullptr);}
        }
    }
    busy_=false;return error;
}
}
