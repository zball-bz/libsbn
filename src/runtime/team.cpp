/* Persistent pool and spin/park design adapted from libsbn/src/team.cpp.
 * Each invocation owns its own descriptor/barrier; no global busy->serial. */
#include "runtime/team.hpp"
#include "runtime/linux/sync.hpp"
#include <limits.h>
#include <new>
using namespace sbn::v3;
namespace {
thread_local Team *current_team;
thread_local unsigned current_worker;
void scope_valid(const sbn3_team_scope *s) {
    require(s && s->team==current_team && s->first==current_worker && !s->active && s->width,
            SBN3_FATAL_TEAM,"scope leader or nested dispatch");
}
sbn3_status error(sbn3_error *e,sbn3_status code,const char *where,int system=0) {
    if(e)*e={code,system,0,0,where};return code;
}
struct Barrier {
    unsigned width;
    uint32_t arrivals=0,generation=0;
    void arrive() noexcept {
        const uint32_t g=__atomic_load_n(&generation,__ATOMIC_ACQUIRE);
        os::publish_stores();
        if(__atomic_add_fetch(&arrivals,1,__ATOMIC_ACQ_REL)==width) {
            __atomic_store_n(&arrivals,0,__ATOMIC_RELAXED);
            __atomic_add_fetch(&generation,1,__ATOMIC_RELEASE);os::wake(&generation,INT_MAX);
        } else os::wait_changed(&generation,g);
    }
};
struct ForCall {
    sbn3_team_scope *scope;
    uint64_t begin,end,grain,chunks;
    sbn3_schedule schedule;
    sbn3_for_fn fn;
    void *argument;
    alignas(64) uint64_t next=0;
};
void for_worker(void *arg,unsigned global) {
    auto &c=*static_cast<ForCall *>(arg);
    const unsigned rank=global-c.scope->first,width=c.scope->width;
    if(c.schedule==SBN3_STATIC) {
        const uint64_t q=c.chunks/width,rem=c.chunks%width;
        const uint64_t lo=q*rank+(rank<rem?rank:rem), hi=lo+q+(rank<rem);
        if(lo<hi) {
            const uint64_t a=c.begin+lo*c.grain;
            const uint64_t b=hi==c.chunks?c.end:c.begin+hi*c.grain;
            c.fn(c.argument,a,b,rank);
        }
    } else {
        for(;;) {
            const uint64_t chunk=__atomic_fetch_add(&c.next,1,__ATOMIC_RELAXED);
            if(chunk>=c.chunks)break;
            const uint64_t lo=c.begin+chunk*c.grain;
            const uint64_t hi=c.end-lo<c.grain?c.end:lo+c.grain;
            c.fn(c.argument,lo,hi,rank);
        }
    }
}
void group_run(sbn3_team_scope *s,Team::Work work,void *arg,unsigned active_width=0) {
    const unsigned width=active_width?active_width:s->width;
    s->active=true;uint32_t tickets[32]{};
    for(unsigned i=1;i<width;++i)tickets[i]=s->team->send(s->first+i,work,arg);
    work(arg,s->first);os::publish_stores();
    for(unsigned i=1;i<width;++i)s->team->join(s->first+i,tickets[i]);
    s->active=false;
}
struct ActionCall {sbn3_team_scope scope;sbn3_action_fn fn;void *arg;};
void action_worker(void *arg,unsigned worker) {
    auto &a=*static_cast<ActionCall *>(arg);
    require(worker==a.scope.first,SBN3_FATAL_TEAM,"action leader");
    a.fn(a.arg,&a.scope);
    require(!a.scope.active,SBN3_FATAL_TEAM,"unfinished child scope");
}
}
struct sbn3_team_region {Barrier *barrier;unsigned rank,width;};
namespace {
struct RegionCall {sbn3_team_scope *scope;Barrier barrier;sbn3_region_fn fn;void *arg;};
void region_worker(void *arg,unsigned global) {
    auto &c=*static_cast<RegionCall *>(arg);
    sbn3_team_region r{&c.barrier,global-c.scope->first,c.scope->width};c.fn(c.arg,&r);
}
}

uint32_t Team::send(unsigned worker,Work fn,void *argument) noexcept {
    require(worker<width && worker!=current_worker,SBN3_FATAL_TEAM,"dispatch worker");
    auto &w=workers[worker];const uint32_t issued=__atomic_load_n(&w.issued,__ATOMIC_ACQUIRE);
    require(__atomic_load_n(&w.completed,__ATOMIC_ACQUIRE)==issued,
            SBN3_FATAL_TEAM,"dispatch occupied worker");
    w.fn=fn;w.argument=argument;
    os::publish_epoch(&w.issued,issued+1,&w.parked_issue);
    return issued+1;
}
void Team::join(unsigned worker,uint32_t ticket) noexcept {
    auto &w=workers[worker];
    for(;;){const uint32_t done=__atomic_load_n(&w.completed,__ATOMIC_ACQUIRE);if(done==ticket)return;os::wait_epoch(&w.completed,done,&w.parked_done);}
}
void *Team::worker_main(void *p) noexcept {
    auto &w=*static_cast<Worker *>(p);current_team=w.team;current_worker=w.index;
    __atomic_store_n(&w.started,1,__ATOMIC_RELEASE);os::wake(&w.started);
    uint32_t last=0;
    for(;;) {
        os::wait_epoch(&w.issued,last,&w.parked_issue);
        const uint32_t next=__atomic_load_n(&w.issued,__ATOMIC_ACQUIRE);
        Work fn=w.fn;void *arg=w.argument;
        if(fn)fn(arg,w.index);
        os::publish_stores();
        // Publishing done is the final access to the caller-owned descriptor.
        os::publish_epoch(&w.completed,next,&w.parked_done);
        last=next;if(!fn)break;
    }
    current_team=nullptr;return nullptr;
}
void Team::stop_workers() noexcept {
    for(unsigned i=1;i<=created;++i)send(i,nullptr,nullptr);
    for(unsigned i=1;i<=created;++i)require(pthread_join(workers[i].thread,nullptr)==0,SBN3_FATAL_TEAM,"pthread_join");
    created=0;
}
void Team::release_stacks() noexcept {
    for(unsigned i=1;i<width;++i) {
        if(stacks[i].bytes){arena->release(stacks[i]);stacks[i]={};}
        if(guards[i].bytes){arena->release(guards[i]);guards[i]={};}
    }
}
extern "C" size_t sbn3_team_storage_bytes(void) {size_t n=0;require(align_size(sizeof(sbn3_team),64,n),SBN3_FATAL_SIZE,"team bytes");return n;}
extern "C" size_t sbn3_team_stack_virtual_bytes(unsigned w) {require(w && w<=32,SBN3_FATAL_ARGUMENT,"stack workers");return (w-1)*(Team::stack_bytes+Team::guard_bytes);}
extern "C" size_t sbn3_team_stack_resident_bytes(unsigned w) {require(w && w<=32,SBN3_FATAL_ARGUMENT,"stack workers");return (w-1)*Team::stack_bytes;}
extern "C" sbn3_status sbn3_team_create(sbn3_arena *a,const sbn3_lease *storage,
                                         const sbn3_team_config *cfg,sbn3_team **out,sbn3_error *e) {
    require(a&&storage&&cfg&&out,SBN3_FATAL_ARGUMENT,"team_create arguments");*out=nullptr;
    if(!cfg->workers || cfg->workers>32 || storage->bytes<sbn3_team_storage_bytes() ||
       (reinterpret_cast<uintptr_t>(storage->data)&63))return error(e,SBN3_ECAPACITY,"team configuration/storage");
    const sbn3_lease saved_storage=*storage;const sbn3_team_config saved_config=*cfg;
    storage=&saved_storage;cfg=&saved_config;
    if(current_team || (cfg->stack_offset&(Team::guard_bytes-1)) ||
       !a->unleased(cfg->stack_offset,sbn3_team_stack_virtual_bytes(cfg->workers)))
        return error(e,SBN3_EBUSY,"team stack range/binding phase");
    a->retain(*storage);
    auto *t=::new(storage->data) sbn3_team{};
    t->arena=a;t->storage=*storage;t->width=cfg->workers;t->creator=pthread_self();
    int rc=pthread_getaffinity_np(t->creator,sizeof(cpu_set_t),&t->original_affinity);
    if(rc){t->~sbn3_team();a->release(*storage);return error(e,SBN3_EOS,"caller affinity query",rc);}
    unsigned n=0;
    if(cfg->cpu_ids[0]<0) {
        for(int c=0;c<CPU_SETSIZE && n<t->width;++c)if(CPU_ISSET(c,&t->original_affinity))t->cpus[n++]=c;
    } else {
        for(unsigned i=0;i<t->width;++i) {
            const int c=cfg->cpu_ids[i];bool duplicate=false;
            for(unsigned j=0;j<i;++j)duplicate|=t->cpus[j]==c;
            if(c<0 || c>=CPU_SETSIZE || !CPU_ISSET(c,&t->original_affinity) || duplicate)break;
            t->cpus[n++]=c;
        }
    }
    if(n<t->width){t->~sbn3_team();a->release(*storage);return error(e,SBN3_ECAPACITY,"team CPUs");}
    for(unsigned i=1;i<t->width;++i) {
        const size_t off=cfg->stack_offset+(i-1)*(Team::stack_bytes+Team::guard_bytes);
        sbn3_status status=a->reserve_guard(off,Team::guard_bytes,t->guards[i]);
        if(status==SBN3_OK)status=a->prepare(off+Team::guard_bytes,Team::stack_bytes,e);
        if(status!=SBN3_OK) {t->release_stacks();t->~sbn3_team();a->release(*storage);return error(e,status,"worker stack preparation");}
        t->stacks[i]=a->acquire(off+Team::guard_bytes,Team::stack_bytes);
    }
    if(cfg->pin_threads) {
        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(t->cpus[0],&mask);
        rc=pthread_setaffinity_np(t->creator,sizeof(mask),&mask);
        if(rc){t->release_stacks();t->~sbn3_team();a->release(*storage);return error(e,SBN3_EOS,"caller affinity bind",rc);}
        t->pinned=true;
    }
    for(unsigned i=1;i<t->width;++i) {
        auto &w=t->workers[i];w.team=t;w.index=i;
        pthread_attr_t attr;rc=pthread_attr_init(&attr);
        if(!rc) {
            rc=pthread_attr_setstack(&attr,t->stacks[i].data,t->stacks[i].bytes);
            if(!rc && cfg->pin_threads) {cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(t->cpus[i],&mask);rc=pthread_attr_setaffinity_np(&attr,sizeof(mask),&mask);}
            if(!rc)rc=pthread_create(&w.thread,&attr,Team::worker_main,&w);
            pthread_attr_destroy(&attr);
        }
        if(rc) {
            t->stop_workers();
            if(t->pinned)require(pthread_setaffinity_np(t->creator,sizeof(cpu_set_t),&t->original_affinity)==0,SBN3_FATAL_TEAM,"affinity rollback");
            t->release_stacks();t->~sbn3_team();a->release(*storage);return error(e,SBN3_EOS,"worker create",rc);
        }
        ++t->created;os::wait_equal(&w.started,1);
    }
    *out=t;return error(e,SBN3_OK,"team create");
}
extern "C" void sbn3_team_destroy(sbn3_team *t) {
    if(!t)return;
    require(pthread_equal(t->creator,pthread_self()) && !t->busy,SBN3_FATAL_TEAM,"team destroy owner/active");
    t->stop_workers();
    if(t->pinned)require(pthread_setaffinity_np(t->creator,sizeof(cpu_set_t),&t->original_affinity)==0,SBN3_FATAL_TEAM,"restore caller affinity");
    t->release_stacks();auto *a=t->arena;auto l=t->storage;t->~sbn3_team();a->release(l);
}
extern "C" unsigned sbn3_team_workers(const sbn3_team *t) {require(t,SBN3_FATAL_ARGUMENT,"null team");return t->width;}
extern "C" void sbn3_team_run(sbn3_team *t,sbn3_action_fn fn,void *arg) {
    require(t&&fn,SBN3_FATAL_ARGUMENT,"team_run arguments");
    require(pthread_equal(t->creator,pthread_self()) && !t->busy,SBN3_FATAL_TEAM,"team root owner/active");
    Team *old=current_team;const unsigned oldw=current_worker;
    current_team=t;current_worker=0;t->busy=true;
    {
        ComputeLease compute(*t->arena);
        require(t->epoch!=UINT64_MAX,SBN3_FATAL_TEAM,"team epoch overflow");
        sbn3_team_scope scope{t,0,t->width,false,++t->epoch};fn(arg,&scope);
        require(!scope.active,SBN3_FATAL_TEAM,"unfinished root scope");os::publish_stores();
    }
    t->busy=false;current_team=old;current_worker=oldw;
}
extern "C" unsigned sbn3_team_width(const sbn3_team_scope *s) {require(s,SBN3_FATAL_ARGUMENT,"null scope");return s->width;}
extern "C" unsigned sbn3_team_first_worker(const sbn3_team_scope *s) {require(s,SBN3_FATAL_ARGUMENT,"null scope");return s->first;}
extern "C" void sbn3_team_invoke2(sbn3_team_scope *s,sbn3_children mode,unsigned left,
                                   sbn3_action_fn lf,void *la,sbn3_action_fn rf,void *ra) {
    scope_valid(s);require(lf&&rf && (mode==SBN3_SERIAL_CHILDREN || mode==SBN3_PARALLEL_CHILDREN),SBN3_FATAL_ARGUMENT,"invoke2 arguments");
    if(mode==SBN3_SERIAL_CHILDREN || s->width==1) {lf(la,s);os::publish_stores();rf(ra,s);os::publish_stores();return;}
    require(left && left<s->width,SBN3_FATAL_TEAM,"invoke2 partition",left,s->width);
    s->active=true;
    ActionCall r{{s->team,s->first+left,s->width-left,false,s->epoch},rf,ra};
    const auto ticket=s->team->send(r.scope.first,action_worker,&r);
    ActionCall l{{s->team,s->first,left,false,s->epoch},lf,la};action_worker(&l,s->first);os::publish_stores();
    s->team->join(r.scope.first,ticket);s->active=false;
}
extern "C" void sbn3_team_for(sbn3_team_scope *s,uint64_t begin,uint64_t end,uint64_t grain,
                                sbn3_schedule schedule,sbn3_for_fn fn,void *arg) {
    scope_valid(s);require(fn && begin<=end && grain && (schedule==SBN3_STATIC||schedule==SBN3_DYNAMIC),SBN3_FATAL_ARGUMENT,"parallel_for arguments");
    if(begin==end)return;
    const uint64_t size=end-begin,chunks=size/grain+(size%grain!=0);
    require(chunks<UINT64_MAX-32,SBN3_FATAL_SIZE,"parallel_for chunk count");
    // At most one worker per chunk can help. Preserve the declared scope and
    // static rank mapping, but do not wake empty ranks (notably SMT siblings).
    const unsigned active=chunks<s->width?unsigned(chunks):s->width;
    ForCall c{s,begin,end,grain,chunks,schedule,fn,arg};group_run(s,for_worker,&c,active);
}
extern "C" void sbn3_team_region_run(sbn3_team_scope *s,sbn3_region_fn fn,void *arg) {
    scope_valid(s);require(fn,SBN3_FATAL_ARGUMENT,"region function");
    RegionCall c{s,{s->width},fn,arg};group_run(s,region_worker,&c);
}
extern "C" unsigned sbn3_region_rank(const sbn3_team_region *r) {require(r,SBN3_FATAL_ARGUMENT,"null region");return r->rank;}
extern "C" unsigned sbn3_region_width(const sbn3_team_region *r) {require(r,SBN3_FATAL_ARGUMENT,"null region");return r->width;}
extern "C" void sbn3_region_barrier(sbn3_team_region *r) {require(r,SBN3_FATAL_ARGUMENT,"null region");r->barrier->arrive();}

namespace sbn::v3 {
void require_scope_leader(const sbn3_team_scope *s) {scope_valid(s);}
void kernel_for(sbn3_team_scope *s,uint64_t lo,uint64_t hi,uint64_t grain,sbn3_schedule schedule,
                 KernelForFn fn,void *arg,Frame *const *frames) {
    struct Context {KernelForFn fn;void *arg;Frame *const *frames;unsigned first;};
    Context c{fn,arg,frames,s->first};
    auto body=[](void *p,uint64_t a,uint64_t b,unsigned rank) {
        auto &c=*static_cast<Context *>(p);Frame &f=*c.frames[rank];FrameMark mark(f);
        c.fn(c.arg,a,b,static_cast<int>(c.first+rank),&f);
    };
    sbn3_team_for(s,lo,hi,grain,schedule,body,&c);
}
}
