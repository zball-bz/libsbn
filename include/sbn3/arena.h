#ifndef SBN3_ARENA_H
#define SBN3_ARENA_H
#include "sbn3/base.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sbn3_arena sbn3_arena;
typedef struct sbn3_arena_config {
    size_t virtual_bytes;
    size_t resident_budget; /* Includes the arena's locked control pages. */
} sbn3_arena_config;
typedef struct sbn3_error {
    sbn3_status code;
    int system_error;
    size_t need, have;
    const char *where; /* Static diagnostic string, no allocation. */
} sbn3_error;
typedef struct sbn3_lease {
    void *data;
    size_t bytes;
    uint64_t token;
} sbn3_lease;
typedef struct sbn3_arena_stats {
    size_t virtual_bytes, resident_budget;
    size_t payload_resident_bytes, control_resident_bytes, peak_resident_bytes;
    size_t active_leases, lease_references, active_computations;
    uint64_t prepare_calls, trim_calls;
} sbn3_arena_stats;

/* Only these explicit OS/resource boundaries return failures. On failed
 * prepare, previously prepared bytes and live data are unchanged. */
sbn3_status sbn3_arena_create(const sbn3_arena_config *, sbn3_arena **, sbn3_error *);
sbn3_status sbn3_arena_prepare(sbn3_arena *, size_t offset, size_t bytes, sbn3_error *);
sbn3_status sbn3_arena_trim(sbn3_arena *, size_t offset, size_t bytes, sbn3_error *);
void sbn3_arena_destroy(sbn3_arena *);
void sbn3_arena_get_stats(sbn3_arena *, sbn3_arena_stats *);

/* Retain a prepared span. Lease references may share pages; trim refuses
 * any page still covered by a lease. Max 512 live lease identities.
 * Acquisition is a checked binding, not an allocation/failable hot path. */
void sbn3_arena_acquire(sbn3_arena *, size_t offset, size_t bytes, sbn3_lease *);
void sbn3_arena_release(sbn3_arena *, sbn3_lease *);

#ifdef __cplusplus
}
#endif
#endif
