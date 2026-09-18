#include "runtime/scratch.hpp"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace sbn::v3;

static sbn3_arena *create(size_t budget=4u<<20) {
    sbn3_arena *a=nullptr;sbn3_error e{};
    sbn3_arena_config c{size_t(1)<<40,budget};
    assert(sbn3_arena_create(&c,&a,&e)==SBN3_OK);
    return a;
}
static void fatal_case(int which) {
    sbn3_arena *a=create();sbn3_error e{};
    assert(sbn3_arena_prepare(a,0,16384,&e)==SBN3_OK);
    sbn3_lease l{};sbn3_arena_acquire(a,0,16384,&l);
    if(which==0) {Frame f(*a,l);auto m=f.mark();f.reset();f.rewind(m);}
    if(which==1) {Frame f(*a,l);auto child=f.subframe(64);f.reset();}
    if(which==2) {Frame f(*a,l);f.allocate(16385);}
    if(which==3) {auto stale=l;sbn3_arena_release(a,&l);a->retain(stale);}
    if(which==4) sbn3_arena_destroy(a);
    _exit(99);
}
int main() {
    const size_t pg=static_cast<size_t>(sysconf(_SC_PAGESIZE));
    sbn3_arena *a=create();sbn3_error e{};sbn3_arena_stats s{},before{};
    sbn3_arena_get_stats(a,&s);assert(s.virtual_bytes==size_t(1)<<40 && !s.payload_resident_bytes);
    assert(s.control_resident_bytes && s.peak_resident_bytes==s.control_resident_bytes);
    assert(sbn3_arena_prepare(a,17,2*pg,&e)==SBN3_OK); // rounds to three pages
    assert(sbn3_arena_prepare(a,2*pg,2*pg,&e)==SBN3_OK); // union four, not five
    sbn3_arena_get_stats(a,&s);assert(s.payload_resident_bytes==4*pg);
    sbn3_lease retained{},nearby{};
    sbn3_arena_acquire(a,0,8,&retained);sbn3_arena_acquire(a,pg+32,16,&nearby);
    *static_cast<uint64_t *>(retained.data)=0x91323f87400ULL;
    assert(sbn3_arena_trim(a,128,1,&e)==SBN3_EBUSY); // same page, disjoint payload
    assert(sbn3_arena_trim(a,pg+512,1,&e)==SBN3_EBUSY);
    assert(sbn3_arena_trim(a,2*pg,pg,&e)==SBN3_OK); // hole in the resident interval
    sbn3_arena_get_stats(a,&s);assert(s.payload_resident_bytes==3*pg);
    assert(sbn3_arena_prepare(a,2*pg+7,100,&e)==SBN3_OK);
    assert(*static_cast<uint64_t *>(retained.data)==0x91323f87400ULL);
    {
        ComputeLease computing(*a);
        assert(sbn3_arena_prepare(a,4*pg,pg,&e)==SBN3_EBUSY);
        assert(sbn3_arena_trim(a,3*pg,pg,&e)==SBN3_EBUSY);
    }
    sbn3_arena_get_stats(a,&before);
    assert(sbn3_arena_prepare(a,0,8u<<20,&e)==SBN3_ECAPACITY);
    sbn3_arena_get_stats(a,&s);assert(s.payload_resident_bytes==before.payload_resident_bytes);
    assert(*static_cast<uint64_t *>(retained.data)==0x91323f87400ULL);
    sbn3_arena_release(a,&nearby);sbn3_arena_release(a,&retained);
    sbn3_lease slab{};sbn3_arena_acquire(a,0,4*pg,&slab);
    {
        Frame f(*a,slab);sbn3_arena_release(a,&slab); // frame retains its own reference
        const auto mark=f.mark();
        auto *x=f.alloc<uint64_t>(8);memset(x,0x4d,64);
        {
            auto left=f.subframe(2048,128),right=f.subframe(2048,128);
            sbn3_arena_stats child_stats{};sbn3_arena_get_stats(a,&child_stats);
            assert(child_stats.lease_references==1); // retained root owns both children
            assert((reinterpret_cast<uintptr_t>(left.data())&127)==0);
            assert(left.data()+left.capacity()<=right.data());
            memset(left.alloc_zero<uint8_t>(2000),0xab,2000);
            memset(right.alloc<uint8_t>(2048),0xcd,2048);
            assert(x[0]==0x4d4d4d4d4d4d4d4dULL);
            assert(sbn3_arena_trim(a,0,pg,&e)==SBN3_EBUSY);
        }
        f.rewind(mark);assert(f.used()==0 && f.peak()>=4160);
        assert(sbn3_arena_trim(a,0,pg,&e)==SBN3_EBUSY);
    }
    assert(sbn3_arena_trim(a,0,4*pg,&e)==SBN3_OK);
    sbn3_arena_get_stats(a,&s);assert(!s.payload_resident_bytes && !s.lease_references);
    sbn3_arena_destroy(a);
    // Model check: sparse prepare/trim including merges/splits at the same VA.
    a=create();bool resident[48]={};
    uint64_t seed=731357;
    for(int i=0;i<500;++i) {
        seed=seed*6364136223846793005ULL+1;size_t lo=(seed>>32)%48;
        seed=seed*6364136223846793005ULL+1;size_t n=1+(seed>>32)%(48-lo);
        bool set=(seed>>22)&1;
        assert((set?sbn3_arena_prepare(a,lo*pg,n*pg,&e):sbn3_arena_trim(a,lo*pg,n*pg,&e))==SBN3_OK);
        for(size_t j=lo;j<lo+n;++j)resident[j]=set;
        size_t count=0;for(bool v:resident)count+=v;
        sbn3_arena_get_stats(a,&s);assert(s.payload_resident_bytes==count*pg);
    }
    sbn3_arena_destroy(a);
    // Isolated native lock failure: no permission changes to the parent/system.
    pid_t child=fork();assert(child>=0);
    if(!child) {
        auto *q=create(16u<<20);sbn3_error er{};
        assert(sbn3_arena_prepare(q,0,2*pg,&er)==SBN3_OK);
        sbn3_lease live{};sbn3_arena_acquire(q,0,8,&live);*static_cast<uint64_t *>(live.data)=951;
        sbn3_arena_stats old{},now{};sbn3_arena_get_stats(q,&old);
        struct rlimit lim{};assert(getrlimit(RLIMIT_MEMLOCK,&lim)==0);
        lim.rlim_cur=old.control_resident_bytes+old.payload_resident_bytes+pg;
        assert(setrlimit(RLIMIT_MEMLOCK,&lim)==0);
        const auto rc=sbn3_arena_prepare(q,0,1u<<20,&er);
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
        // Sanitizers may intercept mlock as a no-op. Native check is authoritative.
        if(rc==SBN3_OK) _exit(0);
#endif
#endif
        assert(rc==SBN3_ELOCK);
        sbn3_arena_get_stats(q,&now);assert(now.payload_resident_bytes==old.payload_resident_bytes);
        assert(*static_cast<uint64_t *>(live.data)==951);
        sbn3_arena_release(q,&live);sbn3_arena_destroy(q);_exit(0);
    }
    int st;assert(waitpid(child,&st,0)==child && WIFEXITED(st) && WEXITSTATUS(st)==0);
    for(int i=0;i<5;++i) {
        child=fork();assert(child>=0);
        if(!child){struct rlimit z={0,0};setrlimit(RLIMIT_CORE,&z);fatal_case(i);}
        assert(waitpid(child,&st,0)==child && WIFSIGNALED(st) && WTERMSIG(st)==SIGABRT);
    }
    puts("arena: sparse page model, shared-page leases, retained frames, lock rollback and 5 fatal gates OK");
}
