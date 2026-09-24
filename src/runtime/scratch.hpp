#pragma once
#include "runtime/arena.hpp"
#include "common/small_checks.h"
#include <string.h>
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define SBN3_FRAME_POISON(p,n) __asan_poison_memory_region((p),(n))
#define SBN3_FRAME_UNPOISON(p,n) __asan_unpoison_memory_region((p),(n))
#endif
#endif
#ifndef SBN3_FRAME_POISON
#define SBN3_FRAME_POISON(p,n) ((void)(p),(void)(n))
#define SBN3_FRAME_UNPOISON(p,n) ((void)(p),(void)(n))
#endif

namespace sbn::v3 {
class Frame {
    Arena *arena_;
    sbn3_lease lease_;
    Frame *parent_;
    uint8_t *base_;
    size_t bytes_, cursor_=0, peak_=0;
    uint64_t epoch_=1;
    unsigned children_=0;
    Frame(Arena *a, const sbn3_lease &l, uint8_t *p, size_t n, Frame *parent,bool poison=true) noexcept
        : arena_(a),lease_(l),parent_(parent),base_(p),bytes_(n) {
        // Child lifetime is already protected by the parent's child count.
        // Only roots retain the arena lease: never contend on the arena mutex
        // for every recursive SALLOC frame or kernel-worker subframe.
        if(!parent_ && a)a->retain(l);
        if(parent_) __atomic_add_fetch(&parent_->children_,1,__ATOMIC_RELAXED);
        if(parent_ && poison)SBN3_FRAME_POISON(base_,bytes_);
    }
public:
    struct Mark { const Frame *frame;size_t offset;uint64_t epoch; };
    Frame(Arena &a,const sbn3_lease &l) noexcept : Frame(&a,l,static_cast<uint8_t *>(l.data),l.bytes,nullptr) {}
    // A caller-owned scratch span. Its owner keeps it live and exclusive;
    // this view performs no arena operation and cannot grow the span.
    static Frame external(void *data,size_t bytes) noexcept {
        require(data || !bytes,SBN3_FATAL_ARGUMENT,"external scratch span");
        return Frame(nullptr,{},static_cast<uint8_t *>(data),bytes,nullptr);
    }
    // A preplanned subrange of one retained shared arena lease. The scheduler
    // proves exclusive concurrent use of the subrange; no new lease identity
    // or page operation is needed for each recursive arithmetic operation.
    static Frame borrow(Arena &a,const sbn3_lease &l,void *data,size_t bytes) noexcept {
        const uintptr_t base=reinterpret_cast<uintptr_t>(l.data),at=reinterpret_cast<uintptr_t>(data);
        require(at>=base && at-base<=l.bytes && bytes<=l.bytes-(at-base),
                SBN3_FATAL_WORKSPACE,"borrowed frame range",bytes,l.bytes);
        return Frame(&a,l,static_cast<uint8_t *>(data),bytes,nullptr);
    }
    Frame(const Frame &)=delete;
    Frame &operator=(const Frame &)=delete;
    ~Frame() {
        require(!__atomic_load_n(&children_,__ATOMIC_ACQUIRE),SBN3_FATAL_LIFETIME,"destroy parent frame");
        Arena *arena=arena_;const sbn3_lease lease=lease_;Frame *parent=parent_;
        SBN3_FRAME_UNPOISON(base_,bytes_);
        if(parent) __atomic_sub_fetch(&parent->children_,1,__ATOMIC_RELEASE);
        // The Frame itself may live in this lease. Do not read its members
        // after the final release allows the controller to reclaim the span.
        if(!parent && arena)arena->release(lease);
    }
    void *allocate(size_t n,size_t alignment=64) noexcept {
        const uintptr_t address=reinterpret_cast<uintptr_t>(base_);
        size_t pos=0,end=0;
        require(align_size(address+cursor_,alignment,pos) && pos>=address &&
                add_size(pos-address,n,end) && end<=bytes_,SBN3_FATAL_WORKSPACE,"frame allocation",n,bytes_-cursor_);
        cursor_=end;if(end>peak_)peak_=end;
        SBN3_FRAME_UNPOISON(reinterpret_cast<void *>(pos),n);
        const size_t room=bytes_-end;
        SBN3_FRAME_POISON(base_+end,room<64?room:64);
        return reinterpret_cast<void *>(pos);
    }
    template<class T> T *alloc(size_t n) noexcept {
        return static_cast<T *>(allocate(bytes_for(n,sizeof(T)),alignof(T)>64?alignof(T):64));
    }
    // Internal kernels with a proved scratch bound; caller owns this Frame
    // exclusively. Diagnostic builds retain normal bounds/poison checking.
    template<class T> T *alloc_assumed(size_t n) noexcept {
        if constexpr(SBN3_CHECK_SMALL)return alloc<T>(n);
        constexpr size_t alignment=alignof(T)>64?alignof(T):64;
        const auto base=reinterpret_cast<uintptr_t>(base_);
        const auto pos=(base+cursor_+alignment-1)&~uintptr_t(alignment-1);
        cursor_=pos-base+n*sizeof(T);if(cursor_>peak_)peak_=cursor_;
        return reinterpret_cast<T *>(pos);
    }
    template<class T> T *alloc_zero(size_t n) noexcept {
        T *p=alloc<T>(n);memset(p,0,bytes_for(n,sizeof(T)));return p;
    }
    Frame subframe(size_t n,size_t alignment=64) noexcept {
        auto *p=static_cast<uint8_t *>(allocate(n,alignment));
        return Frame(arena_,lease_,p,n,this);
    }
    // Revisit an already allocated region (e.g. a kernel control object and
    // its scratch). Keep the parent live and forbid rewind until this view
    // dies. Do not poison the live control object inside the region.
    Frame borrowed_view(void *data,size_t bytes) noexcept {
        const uintptr_t base=reinterpret_cast<uintptr_t>(base_),at=reinterpret_cast<uintptr_t>(data);
        require(at>=base && at-base<=cursor_ && bytes<=cursor_-(at-base),
                SBN3_FATAL_WORKSPACE,"borrowed allocated view",bytes,cursor_);
        return Frame(arena_,lease_,static_cast<uint8_t *>(data),bytes,this,false);
    }
    Mark mark() const noexcept {return {this,cursor_,epoch_};}
    void rewind(Mark m) noexcept {
        require(m.frame==this && m.epoch==epoch_ && m.offset<=cursor_ &&
                !__atomic_load_n(&children_,__ATOMIC_ACQUIRE),SBN3_FATAL_LIFETIME,"frame rewind");
        SBN3_FRAME_POISON(base_+m.offset,cursor_-m.offset);
        cursor_=m.offset;
    }
    void rewind_assumed(Mark m) noexcept {
        if constexpr(SBN3_CHECK_SMALL)rewind(m);
        else cursor_=m.offset;
    }
    void reset() noexcept {
        rewind({this,0,epoch_});
        require(epoch_!=UINT64_MAX,SBN3_FATAL_LIFETIME,"frame epoch overflow");++epoch_;
    }
    size_t used() const noexcept {return cursor_;}
    size_t peak() const noexcept {return peak_;}
    size_t capacity() const noexcept {return bytes_;}
    uint8_t *data() const noexcept {return base_;}
};
class FrameMark {
    Frame &f_;Frame::Mark mark_;
public:
    explicit FrameMark(Frame &f) noexcept : f_(f),mark_(f.mark()) {}
    ~FrameMark() {f_.rewind(mark_);}
};
class AssumedFrameMark {
    Frame &f_;Frame::Mark mark_;
public:
    explicit AssumedFrameMark(Frame &f) noexcept:f_(f),mark_(f.mark()){}
    ~AssumedFrameMark(){f_.rewind_assumed(mark_);}
};
class ComputeLease {
    Arena &a_;
public:
    explicit ComputeLease(Arena &a) noexcept : a_(a) {a_.enter_compute();}
    ~ComputeLease() {a_.leave_compute();}
};
}
