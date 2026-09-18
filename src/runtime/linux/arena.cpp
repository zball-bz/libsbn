/* OS boundary adapted from the v2 pal/scratch design. All mappings and
 * page locking occur in explicit create/prepare/trim operations. */
#include "runtime/arena.hpp"
#include <errno.h>
#include <new>
#include <sys/mman.h>
#include <unistd.h>
using namespace sbn::v3;

namespace {
sbn3_status error(sbn3_error *e, sbn3_status code, const char *where,
                  int sys = 0, size_t need = 0, size_t have = 0) noexcept {
    if (e) *e = {code, sys, need, have, where};
    return code;
}
void coalesce(Arena::Span *spans, size_t &n) noexcept {
    for (size_t i=1;i<n;++i) {
        auto s=spans[i]; size_t j=i;
        while(j && spans[j-1].begin>s.begin) { spans[j]=spans[j-1]; --j; }
        spans[j]=s;
    }
    size_t out=0;
    for(size_t i=0;i<n;++i) {
        if(out && spans[i].begin<=spans[out-1].end) {
            if(spans[i].end>spans[out-1].end) spans[out-1].end=spans[i].end;
        } else spans[out++]=spans[i];
    }
    n=out;
}
}

bool Arena::page_span(size_t offset, size_t bytes, Span &s) const noexcept {
    size_t end;
    if(!bytes || !add_size(offset,bytes,end) || end>virtual_bytes) return false;
    s.begin=offset & ~(page_size-1);
    return align_size(end,page_size,s.end);
}
bool Arena::contains(size_t begin, size_t end) const noexcept {
    for(size_t i=0;i<region_count;++i)
        if(regions[i].begin<=begin && end<=regions[i].end) return true;
    return false;
}
sbn3_status Arena::prepare(size_t offset, size_t bytes, sbn3_error *e) noexcept {
    ArenaLock lock(*this); ++prepares;
    if(computations) return error(e,SBN3_EBUSY,"prepare during compute");
    Span requested;
    if(!page_span(offset,bytes,requested)) return error(e,SBN3_ECAPACITY,"prepare range",0,bytes,virtual_bytes);
    for(const auto &l:leases) if(l.references && l.guard && l.begin<requested.end && requested.begin<l.end)
        return error(e,SBN3_EBUSY,"prepare protected guard");
    Span next[max_regions+1];
    for(size_t i=0;i<region_count;++i) next[i]=regions[i];
    size_t nn=region_count;next[nn++]=requested;coalesce(next,nn);
    if(nn>max_regions) return error(e,SBN3_ECAPACITY,"resident region descriptors",0,nn,max_regions);
    Span gaps[max_regions+1];size_t ng=0, pos=requested.begin, fresh=0;
    for(size_t i=0;i<region_count && pos<requested.end;++i) {
        if(regions[i].end<=pos) continue;
        if(regions[i].begin>=requested.end) break;
        if(pos<regions[i].begin) gaps[ng++]={pos,regions[i].begin};
        if(regions[i].end>pos) pos=regions[i].end;
    }
    if(pos<requested.end) gaps[ng++]={pos,requested.end};
    for(size_t i=0;i<ng;++i) fresh+=gaps[i].end-gaps[i].begin;
    size_t total;
    if(!add_size(resident_bytes,control_bytes,total) || !add_size(total,fresh,total) || total>budget)
        return error(e,SBN3_ECAPACITY,"resident budget",0,fresh,budget-control_bytes-resident_bytes);
    size_t done=0;
    for(;done<ng;++done) {
        const Span s=gaps[done]; void *p=base+s.begin;const size_t n=s.end-s.begin;
        if(mprotect(p,n,PROT_READ|PROT_WRITE)) {
            const int saved=errno;
            require(mprotect(p,n,PROT_NONE)==0 && madvise(p,n,MADV_DONTNEED)==0,
                    SBN3_FATAL_LIFETIME,"failed-protect rollback");
            for(size_t j=0;j<done;++j) {
                const auto g=gaps[j];const size_t z=g.end-g.begin;
                require(munlock(base+g.begin,z)==0 && mprotect(base+g.begin,z,PROT_NONE)==0 &&
                        madvise(base+g.begin,z,MADV_DONTNEED)==0,SBN3_FATAL_LIFETIME,"prepare rollback");
            }
            return error(e,SBN3_EOS,"mprotect prepare",saved,n,0);
        }
        // Match the donor's large-buffer THP preparation, before population.
        // This is only a placement hint; successful mlock remains mandatory.
        if(n>=(size_t(2)<<20))(void)madvise(p,n,MADV_HUGEPAGE);
        // mlock populates and locks the range; no first-touch remains in SALLOC.
        if(mlock(p,n)) {
            const int saved=errno;
            // Linux may lock a prefix before failing: the entire fresh gap is ours.
            require(munlock(p,n)==0 && mprotect(p,n,PROT_NONE)==0 && madvise(p,n,MADV_DONTNEED)==0,
                    SBN3_FATAL_LIFETIME,"failed-lock rollback");
            for(size_t j=0;j<done;++j) {
                const auto g=gaps[j];const size_t z=g.end-g.begin;
                require(munlock(base+g.begin,z)==0 && mprotect(base+g.begin,z,PROT_NONE)==0 &&
                        madvise(base+g.begin,z,MADV_DONTNEED)==0,SBN3_FATAL_LIFETIME,"prepare rollback");
            }
            return error(e,SBN3_ELOCK,"mlock prepare",saved,n,0);
        }
    }
    for(size_t i=0;i<nn;++i) regions[i]=next[i];region_count=nn;
    resident_bytes+=fresh;if(total>peak_bytes)peak_bytes=total;
    return error(e,SBN3_OK,"prepare");
}
sbn3_status Arena::trim(size_t offset, size_t bytes, sbn3_error *e) noexcept {
    ArenaLock lock(*this);++trims;
    if(computations) return error(e,SBN3_EBUSY,"trim during compute");
    Span s;
    if(!page_span(offset,bytes,s)) return error(e,SBN3_ECAPACITY,"trim range",0,bytes,virtual_bytes);
    for(const auto &l:leases) if(l.references && l.begin<s.end && s.begin<l.end)
        return error(e,SBN3_EBUSY,"trim live lease");
    Span next[max_regions+1], removed[max_regions];size_t nn=0,nr=0;
    for(size_t i=0;i<region_count;++i) {
        auto r=regions[i];
        if(r.end<=s.begin || r.begin>=s.end) { next[nn++]=r;continue; }
        const size_t begin=r.begin>s.begin?r.begin:s.begin, end=r.end<s.end?r.end:s.end;
        removed[nr++]={begin,end};
        if(r.begin<begin) next[nn++]={r.begin,begin};
        if(end<r.end) next[nn++]={end,r.end};
    }
    if(nn>max_regions) return error(e,SBN3_ECAPACITY,"trim descriptors",0,nn,max_regions);
    // Only dead pages are affected. Keep each successfully retired subrange in
    // the metadata even if a later OS step fails; live leases are untouched.
    for(size_t i=0;i<nr;++i) {
        const auto r=removed[i];const size_t n=r.end-r.begin;
        // These ranges are known mapped/locked, with no live references. An
        // unexpected retirement error must not leave a falsely pinned receipt.
        require(munlock(base+r.begin,n)==0 && mprotect(base+r.begin,n,PROT_NONE)==0 && madvise(base+r.begin,n,MADV_DONTNEED)==0,
                SBN3_FATAL_LIFETIME,"trim mapping");
        resident_bytes-=n;
        Span current[max_regions+1];size_t nc=0;
        for(size_t j=0;j<region_count;++j) {
            auto v=regions[j];
            if(v.end<=r.begin || v.begin>=r.end) current[nc++]=v;
            else { if(v.begin<r.begin) current[nc++]={v.begin,r.begin};if(r.end<v.end)current[nc++]={r.end,v.end}; }
        }
        require(nc<=max_regions,SBN3_FATAL_LIFETIME,"trim intermediate descriptors");
        for(size_t j=0;j<nc;++j) regions[j]=current[j];region_count=nc;
    }
    return error(e,SBN3_OK,"trim");
}
Arena::Lease &Arena::find(const sbn3_lease &l) noexcept {
    const size_t i=static_cast<size_t>(l.token & 0xffffu);
    require(i<max_leases,SBN3_FATAL_LIFETIME,"lease index");
    Lease &r=leases[i];
    require(r.references && r.generation==(l.token>>16) && l.data==base+r.begin && l.bytes==r.end-r.begin,
            SBN3_FATAL_LIFETIME,"stale or foreign lease");
    return r;
}
sbn3_lease Arena::acquire(size_t offset,size_t bytes) noexcept {
    ArenaLock lock(*this);size_t end;
    require(bytes && add_size(offset,bytes,end) && end<=virtual_bytes && contains(offset,end),
            SBN3_FATAL_WORKSPACE,"lease unprepared range",bytes,virtual_bytes);
    for(size_t i=0;i<max_leases;++i) if(!leases[i].references) {
        auto &l=leases[i];require(l.generation<UINT64_MAX>>16,SBN3_FATAL_LIFETIME,"lease generation overflow");
        ++l.generation;l.begin=offset;l.end=end;l.references=1;l.guard=false;
        return {base+offset,bytes,(l.generation<<16)|i};
    }
    fatal(SBN3_FATAL_WORKSPACE,"lease descriptors",max_leases+1,max_leases);
}
bool Arena::unleased(size_t offset,size_t bytes) noexcept {
    ArenaLock lock(*this);size_t end;
    if(computations || !add_size(offset,bytes,end) || end>virtual_bytes)return false;
    for(const auto &l:leases)if(l.references && l.begin<end && offset<l.end)return false;
    return true;
}
sbn3_status Arena::reserve_guard(size_t offset,size_t bytes,sbn3_lease &out) noexcept {
    ArenaLock lock(*this);Span s;
    if(computations)return SBN3_EBUSY;
    if(!page_span(offset,bytes,s) || s.begin!=offset || s.end-offset!=bytes)return SBN3_ECAPACITY;
    for(size_t i=0;i<region_count;++i)if(regions[i].begin<s.end && s.begin<regions[i].end)return SBN3_EBUSY;
    for(const auto &l:leases)if(l.references && l.begin<s.end && s.begin<l.end)return SBN3_EBUSY;
    for(size_t i=0;i<max_leases;++i)if(!leases[i].references) {
        auto &l=leases[i];require(l.generation<UINT64_MAX>>16,SBN3_FATAL_LIFETIME,"guard generation overflow");
        ++l.generation;l.begin=s.begin;l.end=s.end;l.references=1;l.guard=true;
        out={base+s.begin,bytes,(l.generation<<16)|i};return SBN3_OK;
    }
    return SBN3_ECAPACITY;
}
void Arena::claim_unshared(const sbn3_lease &l) noexcept {
    ArenaLock lock(*this);auto &r=find(l);
    require(r.references==1 && !r.guard,SBN3_FATAL_LIFETIME,"storage already in use");
    for(const auto &other:leases)if(&other!=&r && other.references)
        require(other.end<=r.begin || r.end<=other.begin,SBN3_FATAL_LIFETIME,"aliased storage lease");
    ++r.references;
}
void Arena::retain(const sbn3_lease &l) noexcept { ArenaLock lock(*this);auto &r=find(l);require(r.references<SIZE_MAX,SBN3_FATAL_LIFETIME,"lease ref overflow");++r.references; }
void Arena::release(const sbn3_lease &l) noexcept { ArenaLock lock(*this);--find(l).references; }
void Arena::enter_compute() noexcept { ArenaLock lock(*this);++computations; }
void Arena::leave_compute() noexcept { ArenaLock lock(*this);require(computations,SBN3_FATAL_LIFETIME,"compute ref underflow");--computations; }
void Arena::stats(sbn3_arena_stats &s) noexcept {
    ArenaLock lock(*this);s={virtual_bytes,budget,resident_bytes,control_bytes,peak_bytes,0,0,computations,prepares,trims};
    for(const auto &l:leases) if(l.references) {++s.active_leases;s.lease_references+=l.references;}
}

extern "C" sbn3_status sbn3_arena_create(const sbn3_arena_config *cfg,sbn3_arena **out,sbn3_error *e) {
    require(cfg && out,SBN3_FATAL_ARGUMENT,"arena_create arguments");*out=nullptr;
    const long page=sysconf(_SC_PAGESIZE);size_t control,virtual_bytes;
    if(page<=0 || !align_size(sizeof(sbn3_arena),static_cast<size_t>(page),control) ||
       !align_size(cfg->virtual_bytes,static_cast<size_t>(page),virtual_bytes) || !virtual_bytes || cfg->resident_budget<control)
        return error(e,SBN3_ECAPACITY,"arena configuration");
    void *raw=mmap(nullptr,control,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(raw==MAP_FAILED)return error(e,SBN3_ENOMEM,"arena control mmap",errno,control,0);
    if(mlock(raw,control)) {int saved=errno;munmap(raw,control);return error(e,SBN3_ELOCK,"arena control mlock",saved,control,0);}
    // Placement construction in explicitly mapped storage; no ordinary new.
    auto *a=::new(raw) sbn3_arena{};
    a->page_size=static_cast<size_t>(page);a->control_bytes=control;a->budget=cfg->resident_budget;
    a->virtual_bytes=virtual_bytes;a->peak_bytes=control;
    void *base=mmap(nullptr,virtual_bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
    if(base==MAP_FAILED) {int saved=errno;a->~sbn3_arena();munlock(raw,control);munmap(raw,control);return error(e,SBN3_ENOMEM,"arena VA mmap",saved,virtual_bytes,0);}
    a->base=static_cast<uint8_t *>(base);
    const int rc=pthread_mutex_init(&a->mutex,nullptr);
    if(rc) {munmap(base,virtual_bytes);a->~sbn3_arena();munlock(raw,control);munmap(raw,control);return error(e,SBN3_EOS,"arena mutex init",rc);}
    *out=a;return error(e,SBN3_OK,"arena create");
}
extern "C" sbn3_status sbn3_arena_prepare(sbn3_arena *a,size_t off,size_t n,sbn3_error *e) {require(a,SBN3_FATAL_ARGUMENT,"null arena");return a->prepare(off,n,e);}
extern "C" sbn3_status sbn3_arena_trim(sbn3_arena *a,size_t off,size_t n,sbn3_error *e) {require(a,SBN3_FATAL_ARGUMENT,"null arena");return a->trim(off,n,e);}
extern "C" void sbn3_arena_get_stats(sbn3_arena *a,sbn3_arena_stats *s) {require(a&&s,SBN3_FATAL_ARGUMENT,"arena stats");a->stats(*s);}
extern "C" void sbn3_arena_acquire(sbn3_arena *a,size_t off,size_t n,sbn3_lease *l) {require(a&&l,SBN3_FATAL_ARGUMENT,"arena acquire");*l=a->acquire(off,n);}
extern "C" void sbn3_arena_release(sbn3_arena *a,sbn3_lease *l) {require(a&&l,SBN3_FATAL_ARGUMENT,"arena release");a->release(*l);*l={};}
extern "C" void sbn3_arena_destroy(sbn3_arena *a) {
    if(!a)return;
    sbn3_arena_stats s{};a->stats(s);
    require(!s.lease_references && !s.active_computations,SBN3_FATAL_LIFETIME,"destroy live arena",s.lease_references,0);
    require(pthread_mutex_destroy(&a->mutex)==0,SBN3_FATAL_LIFETIME,"arena mutex destroy");
    void *base=a->base;const size_t n=a->virtual_bytes,c=a->control_bytes;
    a->~sbn3_arena();
    require(munmap(base,n)==0 && munlock(a,c)==0 && munmap(a,c)==0,SBN3_FATAL_LIFETIME,"arena unmap");
}
