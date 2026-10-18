#pragma once
#include "sbn3/team.h"
#include "runtime/scratch.hpp"
#include <sched.h>

namespace sbn::v3 {
class Team {
public:
    static constexpr size_t stack_bytes=256u<<10,guard_bytes=64u<<10;
    using Work = void (*)(void *, unsigned);
    struct alignas(64) Worker {
        Team *team=nullptr;
        unsigned index=0;
        pthread_t thread{};
        Work fn=nullptr;
        void *argument=nullptr;
        uint32_t issued=0, completed=0, started=0;
        uint32_t parked_issue=0, parked_done=0;
    };
    Arena *arena=nullptr;
    sbn3_lease storage{};
    unsigned width=0, created=0;
    int cpus[32]{};
    pthread_t creator{};
    cpu_set_t original_affinity{};
    bool pinned=false,busy=false;
    uint64_t epoch=0;
    uint64_t idle_spin_ns=1000000; // atomic access: idle workers may read during owner policy changes
    Worker workers[32]{};
    sbn3_lease stacks[32]{},guards[32]{};

    uint32_t send(unsigned worker,Work fn,void *argument) noexcept;
    void join(unsigned worker,uint32_t ticket) noexcept;
    static void *worker_main(void *) noexcept;
    void stop_workers() noexcept;
    void release_stacks() noexcept;
};
// The owning controller may shorten idle/join polling around an IO operation.
// Arithmetic region barriers keep their existing wait policy. Restore the
// previous setting on every return; no allocation or additional worker.
class TeamWaitScope {
    Team&team_;uint64_t old_;
public:
    TeamWaitScope(Team&t,uint64_t ns) noexcept:team_(t),old_(__atomic_exchange_n(&t.idle_spin_ns,ns,__ATOMIC_ACQ_REL)){}
    ~TeamWaitScope(){__atomic_store_n(&team_.idle_spin_ns,old_,__ATOMIC_RELEASE);}
    TeamWaitScope(const TeamWaitScope&)=delete;
    TeamWaitScope&operator=(const TeamWaitScope&)=delete;
};
using KernelForFn=void (*)(void *,uint64_t,uint64_t,int,Frame *);
void require_scope_leader(const sbn3_team_scope *);
// Allocation-free W1 root for algorithms using caller-owned raw scratch.
// No arena, page work, worker creation or persistent runtime object.
void serial_team_run(sbn3_action_fn,void *);
/* Private bridge for existing pass signatures; no allocator or TLS scratch. */
void kernel_for(sbn3_team_scope *,uint64_t,uint64_t,uint64_t,sbn3_schedule,
                 KernelForFn,void *,Frame *const *frames);
}
struct sbn3_team : sbn::v3::Team {};
struct sbn3_team_scope {
    sbn::v3::Team *team;
    unsigned first,width;
    bool active;
    uint64_t epoch;
};
