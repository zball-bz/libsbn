#ifndef SBN3_CONSTANTS_H
#define SBN3_CONSTANTS_H
#include "sbn3/series.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sbn3_e_options {
    sbn3_series_options series;
    size_t memory_budget; /* complete binding; caller output and Team/arena overhead are separate */
} sbn3_e_options;
typedef struct sbn3_e_plan {
    uint64_t opaque[512];
} sbn3_e_plan;
typedef struct sbn3_e_info {
    size_t fractional_limbs, working_limbs, output_limbs, storage_bytes, storage_alignment;
    uint64_t terms, plan_id;
    size_t series_prepared_bytes, series_workspace_bytes, finish_storage_bytes, value_bytes;
    unsigned workers, serial_prefix;
} sbn3_e_info;
typedef struct sbn3_e_metrics {
    uint64_t series_ns, finish_prepare_ns, finish_ns;
} sbn3_e_metrics;
typedef struct sbn3_e_binding sbn3_e_binding;
/* Returns floor(e * 2^(64*n)), n+1 little-endian limbs. Tail and normalization
 * errors are bounded; guard separation is checked before output is defined.
 * A mathematically inconclusive guard is a fatal certificate failure. */
sbn3_query_result sbn3_e_query(size_t n, const sbn3_e_options *, sbn3_e_plan *, sbn3_e_info *);
/* Caller prepares an unleased aligned arena range. A binding is single use.
 * The series and final division reuse this range at an algorithm boundary;
 * no page allocation/resizing or plan query occurs during execute. As with
 * setup, the arena controller is exclusive through this composite operation;
 * another thread must not trim/reserve the idle pool at its phase boundary. */
void sbn3_e_bind(const sbn3_e_plan *, sbn3_arena *, size_t offset, sbn3_team *, sbn3_e_binding **);
void sbn3_e_execute(sbn3_e_binding *, sbn3_limbs);
/* Read-only result in the caller-provided binding region, valid until unbind.
 * The view may be interior and only limb aligned.
 * Consumes the single-use binding; no separate result allocation. */
sbn3_const_limbs sbn3_e_execute_inplace(sbn3_e_binding *);
void sbn3_e_get_metrics(const sbn3_e_binding *, sbn3_e_metrics *);
void sbn3_e_unbind(sbn3_e_binding *);

typedef struct sbn3_pi_options {
    sbn3_series_options series;
    size_t memory_budget;
    uint64_t minimum_block_terms, maximum_block_terms; /* 0: defaults (128,unlimited) */
    size_t minimum_block_limbs; /* 0: 16384*workers; bit-growth floor for amortizing a block */
} sbn3_pi_options;
typedef struct sbn3_pi_plan {
    uint64_t opaque[1024];
} sbn3_pi_plan;
typedef struct sbn3_pi_info {
    size_t fractional_limbs, working_limbs, output_limbs, storage_bytes, storage_alignment;
    size_t psr_storage_bytes, psr_value_bytes, psr_pool_bytes, terminal_storage_bytes;
    uint64_t terms, plan_id;
    unsigned workers, blocks;
} sbn3_pi_info;
typedef struct sbn3_pi_metrics {
    uint64_t psr_ns, finite_ns, merge_ns, prepare_ns, terminal_ns;
} sbn3_pi_metrics;
typedef struct sbn3_pi_binding sbn3_pi_binding;
/* Chudnovsky T/Q -> one terminal division -> rsqrt(10005) and multiply.
 * Returns floor(pi*2^(64*n)).
 * Same prepared-region, single-use and controller contracts as the e API. */
sbn3_query_result sbn3_pi_query(size_t, const sbn3_pi_options *, sbn3_pi_plan *, sbn3_pi_info *);
void sbn3_pi_bind(const sbn3_pi_plan *, sbn3_arena *, size_t, sbn3_team *, sbn3_pi_binding **);
void sbn3_pi_execute(sbn3_pi_binding *, sbn3_limbs);
/* Compute into a slot in the caller-provided binding region. The returned
 * read-only result remains valid until unbind; no separate output allocation.
 * The view may be interior and only limb aligned.
 * Like execute, this consumes the single-use binding. */
sbn3_const_limbs sbn3_pi_execute_inplace(sbn3_pi_binding *);
void sbn3_pi_get_metrics(const sbn3_pi_binding *, sbn3_pi_metrics *);
void sbn3_pi_unbind(sbn3_pi_binding *);
#ifdef __cplusplus
}
#endif
#endif
