#include "runtime/team.hpp"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <new>
#include <initializer_list>
using namespace sbn::v3;

static uint64_t now() {timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
struct Gate {
    unsigned width,arrived=0;
    uint64_t bits=0,deadline;
    pthread_t threads[32]{};
};
struct Node {Frame *frame;Gate *gate;bool skew;uint64_t result=0;};
static unsigned split(unsigned n,bool skew) {return skew?1:n/2;}
static size_t tree_bytes(unsigned n,bool skew) {
    if(n<=4)return 512+8192*n;
    unsigned l=split(n,skew);return 512+tree_bytes(l,skew)+tree_bytes(n-l,skew);
}
struct Leaf {Gate *gate;uint64_t *buffer;unsigned first;};
static void leaf_region(void *p,sbn3_team_region *region) {
    auto &c=*static_cast<Leaf *>(p);
    unsigned rank=sbn3_region_rank(region),width=sbn3_region_width(region),global=c.first+rank;
    c.gate->threads[global]=pthread_self();
    uint64_t bit=uint64_t(1)<<global;
    assert(!(__atomic_fetch_or(&c.gate->bits,bit,__ATOMIC_ACQ_REL)&bit));
    __atomic_add_fetch(&c.gate->arrived,1,__ATOMIC_RELEASE);
    while(__atomic_load_n(&c.gate->arrived,__ATOMIC_ACQUIRE)<c.gate->width) {
        assert(now()<c.gate->deadline);sched_yield();
    }
    // Each independent child barrier must synchronize just its own team.
    for(unsigned round=0;round<100;++round) {
        c.buffer[rank*8]=global+1+round;
        sbn3_region_barrier(region);
        uint64_t sum=0;for(unsigned i=0;i<width;++i)sum+=c.buffer[i*8];
        const uint64_t want=uint64_t(width)*(c.first+1+round)+uint64_t(width)*(width-1)/2;
        assert(sum==want);sbn3_region_barrier(region);
    }
    c.buffer[rank*8]=global+1;sbn3_region_barrier(region);
}
static void node_action(void *p,sbn3_team_scope *scope) {
    auto &n=*static_cast<Node *>(p);Frame &f=*n.frame;FrameMark mark(f);
    auto *retained=f.alloc<uint64_t>(8);retained[0]=0xa876453231ULL;
    const unsigned w=sbn3_team_width(scope);
    if(w<=4) {
        auto *buffer=f.alloc<uint64_t>(w*8);
        Leaf leaf{n.gate,buffer,sbn3_team_first_worker(scope)};
        sbn3_team_region_run(scope,leaf_region,&leaf);
        n.result=0;for(unsigned i=0;i<w;++i)n.result+=buffer[i*8];
    } else {
        unsigned l=split(w,n.skew);
        auto left=f.subframe(tree_bytes(l,n.skew)),right=f.subframe(tree_bytes(w-l,n.skew));
        Node a{&left,n.gate,n.skew},b{&right,n.gate,n.skew};
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,l,node_action,&a,node_action,&b);
        n.result=a.result+b.result;
    }
    assert(retained[0]==0xa876453231ULL);
}
struct Ranges {uint32_t hits[10064]{};unsigned width;};
static void range_body(void *p,uint64_t lo,uint64_t hi,unsigned rank) {
    auto &r=*static_cast<Ranges *>(p);assert(rank<r.width);
    for(uint64_t i=lo;i<hi;++i)assert(__atomic_fetch_add(&r.hits[i],1,__ATOMIC_RELAXED)==0);
}
static void ranges_action(void *p,sbn3_team_scope *s) {
    auto &r=*static_cast<Ranges *>(p);r.width=sbn3_team_width(s);
    for(auto mode:{SBN3_STATIC,SBN3_DYNAMIC})for(uint64_t grain:{1u,7u,1000u,20000u}) {
        memset(r.hits,0,sizeof r.hits);
        sbn3_team_for(s,31,10038,grain,mode,range_body,&r);
        for(unsigned i=0;i<10064;++i)assert(r.hits[i]==unsigned(i>=31 && i<10038));
        sbn3_team_for(s,55,55,grain,mode,range_body,&r);
    }
}
struct Serial {unsigned width;unsigned stage=0;uint64_t data[32]{};};
static void serial_fill(void *p,uint64_t lo,uint64_t hi,unsigned) {
    auto &c=*static_cast<Serial *>(p);
    for(uint64_t i=lo;i<hi;++i) {assert(c.data[i]==c.stage);c.data[i]=c.stage+1;}
}
static void serial_child(void *p,sbn3_team_scope *s) {
    auto &c=*static_cast<Serial *>(p);assert(sbn3_team_width(s)==c.width);
    sbn3_team_for(s,0,c.width,1,SBN3_STATIC,serial_fill,&c);++c.stage;
}
static void serial_action(void *p,sbn3_team_scope *s) {
    auto &c=*static_cast<Serial *>(p);c.width=sbn3_team_width(s);
    sbn3_team_invoke2(s,SBN3_SERIAL_CHILDREN,0,serial_child,&c,serial_child,&c);
    for(unsigned i=0;i<c.width;++i)assert(c.data[i]==2);
}
struct Kernels {Frame *frames[32];unsigned width;uint64_t result[32]{};};
static void kernel_body(void *p,uint64_t lo,uint64_t hi,int worker,Frame *f) {
    auto &k=*static_cast<Kernels *>(p);assert(worker>=0 && unsigned(worker)<k.width);
    auto *v=f->alloc<uint64_t>(128);for(size_t i=0;i<128;++i)v[i]=uint64_t(worker)+i;
    uint64_t sum=0;for(size_t i=0;i<128;++i)sum+=v[i];
    k.result[worker]+=sum*(hi-lo);
}
static void kernel_action(void *p,sbn3_team_scope *s) {
    auto &k=*static_cast<Kernels *>(p);
    kernel_for(s,0,1000,3,SBN3_DYNAMIC,kernel_body,&k,k.frames);
    for(unsigned i=0;i<k.width;++i)assert(!k.frames[i]->used());
}
int main() {
    for(unsigned w:{1u,3u,5u,16u,32u}) {
        sbn3_arena *arena=nullptr;sbn3_arena_config ac{size_t(1)<<32,12u<<20};sbn3_error e{};
        assert(sbn3_arena_create(&ac,&arena,&e)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&e)==SBN3_OK);
        sbn3_lease control{};sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
        sbn3_team_config cfg{};cfg.workers=w;cfg.pin_threads=1;cfg.stack_offset=64u<<10;
        for(int &c:cfg.cpu_ids)c=-1;
        sbn3_team *team=nullptr;
        const auto status=sbn3_team_create(arena,&control,&cfg,&team,&e);
        if(status!=SBN3_OK)fprintf(stderr,"team setup: %d %s errno=%d\n",int(status),e.where,e.system_error);
        assert(status==SBN3_OK && sbn3_team_workers(team)==w);
        if(w>1)assert(sbn3_arena_prepare(arena,cfg.stack_offset,4096,&e)==SBN3_EBUSY);
        size_t off=cfg.stack_offset+sbn3_team_stack_virtual_bytes(w),bytes=tree_bytes(w,true)+4096*w+4096;
        assert(sbn3_arena_prepare(arena,off,bytes,&e)==SBN3_OK);
        sbn3_lease memory{};sbn3_arena_acquire(arena,off,bytes,&memory);
        {
            Frame f(*arena,memory);
            for(bool skew:{false,true}) {
                Gate gate{w,0,0,now()+5000000000ULL,{}};Node node{&f,&gate,skew};
                sbn3_team_run(team,node_action,&node);
                assert(node.result==uint64_t(w)*(w+1)/2 && f.used()==0 && f.peak()<=f.capacity());
                assert(gate.bits==(uint64_t(1)<<w)-1);
                for(unsigned i=0;i<w;++i)for(unsigned j=i+1;j<w;++j)assert(!pthread_equal(gate.threads[i],gate.threads[j]));
            }
            Ranges ranges{};sbn3_team_run(team,ranges_action,&ranges);
            Serial serial{};sbn3_team_run(team,serial_action,&serial);
            Kernels kernels{};kernels.width=w;
            alignas(Frame) unsigned char objects[32*sizeof(Frame)];
            for(unsigned i=0;i<w;++i)kernels.frames[i]=::new(objects+i*sizeof(Frame))Frame(f.subframe(2048));
            for(int i=0;i<20;++i)sbn3_team_run(team,kernel_action,&kernels);
            for(unsigned i=0;i<w;++i)kernels.frames[i]->~Frame();
            f.reset();
        }
        sbn3_arena_stats before{};sbn3_arena_get_stats(arena,&before);
        assert(before.payload_resident_bytes>=sbn3_team_stack_resident_bytes(w));
        sbn3_arena_release(arena,&memory);sbn3_team_destroy(team);sbn3_arena_release(arena,&control);
        sbn3_arena_stats after{};sbn3_arena_get_stats(arena,&after);
        assert(!after.lease_references && !after.active_computations);
        sbn3_arena_destroy(arena);
        printf("team: W=%u balanced/skew nested progress, independent barriers, serial full-width, range and scratch gates OK\n",w);
    }
}
