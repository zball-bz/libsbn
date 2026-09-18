#ifndef SBN3_TEAM_H
#define SBN3_TEAM_H
#include "sbn3/arena.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sbn3_team sbn3_team;
typedef struct sbn3_team_scope sbn3_team_scope;
typedef struct sbn3_team_region sbn3_team_region;
typedef struct sbn3_team_config {
    unsigned workers; /* 1..32; includes the creating/calling thread */
    int pin_threads;
    int cpu_ids[32];  /* all -1: ascending allowed CPUs; or explicit distinct CPUs */
    size_t stack_offset; /* 64 KiB aligned, disjoint unused VA for stacks/guards */
} sbn3_team_config;
typedef enum sbn3_schedule { SBN3_STATIC=0, SBN3_DYNAMIC=1 } sbn3_schedule;
typedef enum sbn3_children { SBN3_SERIAL_CHILDREN=0, SBN3_PARALLEL_CHILDREN=1 } sbn3_children;
typedef void (*sbn3_action_fn)(void *, sbn3_team_scope *);
typedef void (*sbn3_for_fn)(void *, uint64_t begin, uint64_t end, unsigned rank);
typedef void (*sbn3_region_fn)(void *, sbn3_team_region *);

size_t sbn3_team_storage_bytes(void);
size_t sbn3_team_stack_virtual_bytes(unsigned workers);
size_t sbn3_team_stack_resident_bytes(unsigned workers);
/* Control storage is caller-prepared, 64-byte aligned and retained by the
 * team. Worker creation is an explicit failable preparation boundary.
 * Stacks are explicitly prepared and pinned inside the same arena, with
 * unmapped guard leases. Setup on an arena is serialized by its controller.
 * Successful page preparations can remain cached if a later setup step fails;
 * failure releases every worker/lease. Control storage is exclusively used
 * by the team until destroy. The creator owns run/destroy; affinity is restored. */
sbn3_status sbn3_team_create(sbn3_arena *, const sbn3_lease *, const sbn3_team_config *,
                              sbn3_team **, sbn3_error *);
void sbn3_team_destroy(sbn3_team *);
unsigned sbn3_team_workers(const sbn3_team *);
void sbn3_team_run(sbn3_team *, sbn3_action_fn, void *);
unsigned sbn3_team_width(const sbn3_team_scope *);
unsigned sbn3_team_first_worker(const sbn3_team_scope *);
void sbn3_team_invoke2(sbn3_team_scope *, sbn3_children, unsigned left_workers,
                      sbn3_action_fn left, void *left_arg, sbn3_action_fn right, void *right_arg);
/* Callback rank is local to this scope. Each scope is used by its leader;
 * parallel-for/region callbacks may not recursively dispatch on that scope. */
void sbn3_team_for(sbn3_team_scope *, uint64_t begin, uint64_t end, uint64_t grain,
                   sbn3_schedule, sbn3_for_fn, void *);
void sbn3_team_region_run(sbn3_team_scope *, sbn3_region_fn, void *);
unsigned sbn3_region_rank(const sbn3_team_region *);
unsigned sbn3_region_width(const sbn3_team_region *);
void sbn3_region_barrier(sbn3_team_region *);
#ifdef __cplusplus
}
#endif
#endif
