#include "runtime/scratch.hpp"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <initializer_list>
using namespace sbn::v3;

static sbn3_arena *create(size_t budget=4u<<20) {
    sbn3_arena *a=nullptr;sbn3_error e{};
    sbn3_arena_config c{size_t(1)<<40,budget};
    assert(sbn3_arena_create(&c,&a,&e)==SBN3_OK);
    return a;
}
static constexpr size_t slots=Arena::max_leases;
static size_t slot_of(const sbn3_lease &l) {return static_cast<size_t>(l.token&0xffffu);}
static void fatal_case(int which) {
    sbn3_arena *a=create();sbn3_error e{};
    assert(sbn3_arena_prepare(a,0,16384,&e)==SBN3_OK);
    sbn3_lease l{};sbn3_arena_acquire(a,0,16384,&l);
    if(which==0) {Frame f(*a,l);auto m=f.mark();f.reset();f.rewind(m);}
    if(which==1) {Frame f(*a,l);auto child=f.subframe(64);f.reset();}
    if(which==2) {Frame f(*a,l);f.allocate(16385);}
    if(which==3) {auto stale=l;sbn3_arena_release(a,&l);a->retain(stale);}
    if(which==4) sbn3_arena_destroy(a);
    static sbn3_lease many[slots];
    if(which==5) {for(size_t i=1;i<=slots;++i)sbn3_arena_acquire(a,64*(i%256),64,&many[i-1]);} // 513th descriptor
    if(which==6) {                                        // the only overlapping live lease is slot 511
        sbn3_arena_release(a,&l);sbn3_lease low{};sbn3_arena_acquire(a,0,64,&low);
        for(size_t i=1;i<slots-1;++i)sbn3_arena_acquire(a,8192,64,&many[i]);   // released before the claim
        sbn3_lease high{};sbn3_arena_acquire(a,32,64,&high);assert(slot_of(low)==0 && slot_of(high)==slots-1);
        for(size_t i=1;i<slots-1;++i)sbn3_arena_release(a,&many[i]);
        a->claim_unshared(low);
    }
    if(which==7) {                                        // stale token of a reused word-boundary slot
        sbn3_arena_release(a,&l);
        for(size_t i=0;i<=64;++i)sbn3_arena_acquire(a,64*i,64,&many[i]);
        const auto stale=many[64];sbn3_arena_release(a,&many[64]);
        sbn3_arena_acquire(a,64*64,64,&many[64]);assert(slot_of(many[64])==64 && many[64].token!=stale.token);
        a->retain(stale);
    }
    _exit(99);
}
// Live-slot index across its 64-bit words: lowest-free allocation through slots 63/64,
// 127/128 and 511, exhaustion, holes reused lowest first with new tokens, and lifetime,
// trim, guard and overlap checks when only high slots are live.
static void lease_slots(size_t pg) {
    sbn3_arena *a=create();sbn3_error e{};sbn3_arena_stats s{};
    assert(sbn3_arena_prepare(a,0,16*pg,&e)==SBN3_OK);
    static sbn3_lease l[slots];
    for(size_t i=0;i<slots;++i){sbn3_arena_acquire(a,64*i,64,&l[i]);assert(slot_of(l[i])==i);}
    sbn3_arena_get_stats(a,&s);assert(s.active_leases==slots && s.lease_references==slots);
    sbn3_lease g{};assert(a->reserve_guard(20*pg,pg,g)==SBN3_ECAPACITY);
    static uint64_t before[slots];                        // release clears the caller's lease
    for(size_t i:{511,128,127,64,63,5}){before[i]=l[i].token;sbn3_arena_release(a,&l[i]);}
    sbn3_arena_get_stats(a,&s);assert(s.active_leases==slots-6);
    for(size_t i:{5,63,64,127,128,511}) {
        sbn3_arena_acquire(a,64*i,64,&l[i]);
        assert(slot_of(l[i])==i && l[i].token!=before[i]);
    }
    for(size_t i=0;i<slots;++i)if(i!=300 && i!=511)sbn3_arena_release(a,&l[i]);
    sbn3_arena_get_stats(a,&s);assert(s.active_leases==2 && s.lease_references==2);
    const size_t high_page=64*511/pg*pg;
    assert(sbn3_arena_trim(a,high_page,pg,&e)==SBN3_EBUSY);
    assert(!a->unleased(64*511,64) && !a->unleased(64*300,1) && a->unleased(0,64));
    // a guard in the top slot while everything below it is held, then only it and slot 300 live
    sbn3_arena_release(a,&l[511]);
    static sbn3_lease fill[slots];size_t nf=0;
    while(nf<slots-2){sbn3_arena_acquire(a,0,8,&fill[nf]);assert(slot_of(fill[nf])!=300);++nf;}
    sbn3_lease guard{};assert(a->reserve_guard(20*pg,pg,guard)==SBN3_OK && slot_of(guard)==slots-1);
    for(size_t i=0;i<nf;++i)sbn3_arena_release(a,&fill[i]);
    sbn3_arena_get_stats(a,&s);assert(s.active_leases==2);
    assert(sbn3_arena_prepare(a,20*pg,1,&e)==SBN3_EBUSY);
    sbn3_lease second{};assert(a->reserve_guard(20*pg,pg,second)==SBN3_EBUSY);
    assert(sbn3_arena_trim(a,20*pg,pg,&e)==SBN3_EBUSY);
    // retain: one release keeps the lease live, the last one frees the slot for a new token
    a->retain(l[300]);auto once=l[300];sbn3_arena_release(a,&once);
    assert(!a->unleased(64*300,1) && sbn3_arena_trim(a,64*300/pg*pg,pg,&e)==SBN3_EBUSY);
    const auto old300=l[300];sbn3_arena_release(a,&l[300]);
    sbn3_arena_get_stats(a,&s);assert(s.active_leases==1 && a->unleased(64*300,1));
    for(size_t i=0;i<300;++i)sbn3_arena_acquire(a,0,8,&fill[i]);
    sbn3_arena_acquire(a,64*300,64,&l[300]);assert(slot_of(l[300])==300 && l[300].token!=old300.token);
    for(size_t i=0;i<300;++i)sbn3_arena_release(a,&fill[i]);
    sbn3_arena_release(a,&l[300]);sbn3_arena_release(a,&guard);
    sbn3_arena_get_stats(a,&s);assert(!s.active_leases && !s.lease_references);
    sbn3_arena_destroy(a);
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
    lease_slots(pg);
    for(int i=0;i<8;++i) {
        child=fork();assert(child>=0);
        if(!child){struct rlimit z={0,0};setrlimit(RLIMIT_CORE,&z);fatal_case(i);}
        assert(waitpid(child,&st,0)==child && WIFSIGNALED(st) && WTERMSIG(st)==SIGABRT);
    }
    puts("arena: sparse page model, shared-page leases, retained frames, lock rollback, live-slot words and 8 fatal gates OK");
}
