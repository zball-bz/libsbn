#pragma once
#include "sbn3/product.h"
#include "runtime/arena.hpp"
#include <assert.h>
#include "../oracle/oracle.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <atomic>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();

static size_t up(size_t n,size_t a){return (n+a-1)&~(a-1);}
static uint64_t rng=0x382a79e1be224411ULL;
[[maybe_unused]] static uint64_t random_word(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
[[maybe_unused]] static uint64_t tick(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
struct Fixture {
    sbn3_arena *arena=nullptr;
    sbn3_team *team=nullptr;
    std::vector<sbn3_lease> leases;
    std::vector<sbn3_mul_binding *> bindings;
    uintptr_t base=0;size_t cursor=0;
    bool guard_pages=true;
    // resident: the arena's declared (locked) memory budget; larger gates state theirs explicitly.
    explicit Fixture(unsigned workers,bool guards=true,size_t resident=size_t(256)<<20):guard_pages(guards) {
        sbn3_error e{};sbn3_arena_config ac{size_t(1)<<34,resident};
        assert(sbn3_arena_create(&ac,&arena,&e)==SBN3_OK);
        auto control=allocate(sbn3_team_storage_bytes(),64);
        base=reinterpret_cast<uintptr_t>(control.data);
        sbn3_team_config tc{};tc.workers=workers;tc.pin_threads=1;tc.stack_offset=64u<<10;
        for(int &c:tc.cpu_ids)c=-1;
        const auto status=sbn3_team_create(arena,&control,&tc,&team,&e);
        if(status!=SBN3_OK)fprintf(stderr,"extended team: %d %s errno=%d\n",status,e.where,e.system_error);
        assert(status==SBN3_OK);
        cursor=tc.stack_offset+sbn3_team_stack_virtual_bytes(workers)+4096;
    }
    sbn3_lease allocate(size_t bytes,size_t alignment) {
        cursor=up(base+cursor,alignment)-base;
        sbn3_error e{};auto status=sbn3_arena_prepare(arena,cursor,bytes,&e);
        if(status!=SBN3_OK)fprintf(stderr,"extended prepare: %d %s errno=%d bytes=%zu\n",status,e.where,e.system_error,bytes);
        assert(status==SBN3_OK);
        sbn3_lease l{};sbn3_arena_acquire(arena,cursor,bytes,&l);leases.push_back(l);
        cursor=guard_pages?up(cursor+bytes,4096)+4096:cursor+bytes;return l;
    }
    uint64_t *guarded(size_t limbs) {
        if(!guard_pages)return static_cast<uint64_t *>(allocate(limbs*8,64).data);
        const size_t begin=up(cursor,4096),end=up(begin+limbs*8,4096),start=end-limbs*8;
        sbn3_error e{};assert(sbn3_arena_prepare(arena,begin,end-begin,&e)==SBN3_OK);
        sbn3_lease value{},guard{};sbn3_arena_acquire(arena,start,limbs*8,&value);
        assert(arena->reserve_guard(end,4096,guard)==SBN3_OK);
        leases.push_back(value);leases.push_back(guard);cursor=end+4096;
        return static_cast<uint64_t *>(value.data);
    }
    sbn3_mul_binding *bind(size_t an,size_t bn,sbn3_mul_options options,sbn3_mul_info &info) {
        sbn3_product_spec spec{an,bn};sbn3_mul_plan plan{};
        allocation_watch_start();
        assert(sbn3_mul_query(&spec,&options,&plan,&info)==SBN3_SUPPORTED);
        assert(allocation_watch_stop()==0);
        auto tables=allocate(info.table_bytes,128),work=allocate(info.workspace_bytes,info.workspace_alignment);
        sbn3_mul_binding *b=nullptr;allocation_watch_start();sbn3_mul_bind(&plan,arena,&tables,&work,team,&b);
        assert(allocation_watch_stop()==0);bindings.push_back(b);return b;
    }
    sbn3_mul_binding *product(const sbn3_product_request &req,sbn3_mul_options options,sbn3_product_info &info,sbn3_mul_plan &plan,
                              const sbn3_spectrum *a=nullptr,const sbn3_spectrum *a1=nullptr){
        allocation_watch_start();const auto status=sbn3_product_query(&req,&options,&plan,&info);assert(!allocation_watch_stop());
        if(status!=SBN3_SUPPORTED)fprintf(stderr,"query kind=%u np=%u rejected %u sizes=%zu,%zu T=%d\n",req.kind,options.prime_count,status,req.a_limbs,req.b_limbs,options.trunk_bits);
        assert(status==SBN3_SUPPORTED);auto t=allocate(info.mul.table_bytes,128),w=allocate(info.mul.workspace_bytes,info.mul.workspace_alignment);
        sbn3_mul_binding *b=nullptr;allocation_watch_start();sbn3_product_bind(&plan,arena,&t,&w,team,a,a1,&b);assert(!allocation_watch_stop());bindings.push_back(b);return b;
    }
    sbn3_spectrum *cache(sbn3_mul_binding *b,sbn3_const_limbs a,unsigned frontier,const sbn3_product_info &info,sbn3_spectrum_desc &desc){
        auto storage=allocate(info.spectrum_bytes,info.spectrum_alignment);sbn3_spectrum *s=nullptr;
        allocation_watch_start();sbn3_spectrum_prepare(b,a,static_cast<sbn3_spectrum_frontier>(frontier),7,arena,&storage,&s);assert(!allocation_watch_stop());
        sbn3_spectrum_describe(s,&desc);return s;
    }
    void unbind(sbn3_mul_binding *b){for(auto &v:bindings)if(v==b){sbn3_mul_unbind(v);v=nullptr;return;}assert(false);}
    // Setup-only producer resources can be retired once their binding ends.
    // Trimming also proves that reserved spectra own independent tables.
    void retire(size_t index){
        const auto l=leases[index];const size_t offset=reinterpret_cast<uintptr_t>(l.data)-reinterpret_cast<uintptr_t>(arena->base);
        sbn3_arena_release(arena,&leases[index]);sbn3_error e{};
        if(guard_pages)assert(sbn3_arena_trim(arena,offset,l.bytes,&e)==SBN3_OK);
        leases.erase(leases.begin()+index);
    }
    ~Fixture(){
        for(auto *b:bindings)if(b)sbn3_mul_unbind(b);
        sbn3_team_destroy(team);
        for(auto &l:leases)sbn3_arena_release(arena,&l);
        sbn3_arena_stats stats{};sbn3_arena_get_stats(arena,&stats);
        assert(!stats.lease_references && !stats.active_computations);sbn3_arena_destroy(arena);
    }
};
[[maybe_unused]] static void verify_product(const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const uint64_t *r){
    assert(ref_product_equal(a,an,b,bn,r,an+bn));
}
