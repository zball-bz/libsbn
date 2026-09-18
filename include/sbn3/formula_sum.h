#ifndef SBN3_FORMULA_SUM_H
#define SBN3_FORMULA_SUM_H
#include "sbn3/formula.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Finite signed linear combinations of nonnegative formula values.
 * Result = floor(sum coefficient[j]*value[j] / divisor * 2^(64*n)).
 * The caller's mathematical result must be nonnegative and below 2^63.
 * Components run sequentially through one reusable arena region. Coefficient
 * amplification is included in the working precision; rounding occurs after
 * the combination. Unsupported output-boundary cases are compute-face fatal.
 * The immutable object and the prepared arena range are caller-owned. */
enum { SBN3_FORMULA_SUM_MAX_TERMS = 64 };
typedef struct sbn3_formula_sum_term {
    sbn3_formula_def formula;
    int64_t coefficient;
} sbn3_formula_sum_term;
typedef struct sbn3_formula_sum_plan {
    uint64_t opaque[256];
} sbn3_formula_sum_plan;
typedef struct sbn3_formula_sum_info {
    size_t fractional_limbs, working_limbs, output_limbs;
    size_t object_bytes, storage_bytes, storage_alignment, component_storage_bytes;
    uint64_t plan_id, divisor;
    unsigned components, workers;
    const char *rejection;
} sbn3_formula_sum_info;
typedef struct sbn3_formula_sum_binding sbn3_formula_sum_binding;
typedef struct sbn3_formula_sum_metrics {
    sbn3_formula_metrics components;
    uint64_t component_bind_ns, combine_ns;
} sbn3_formula_sum_metrics;
/* Worst-case immutable storage for this many input terms, 64-byte aligned.
 * Zero coefficients and identical definitions are combined during query. */
size_t sbn3_formula_sum_object_bytes(unsigned terms);
/* memory_budget covers immutable object_bytes + execution storage_bytes.
 * On capacity rejection, info gives requirements and plan is unchanged. */
sbn3_query_result sbn3_formula_sum_query(const sbn3_formula_sum_term *, unsigned count, uint64_t divisor,
                                         size_t fractional_limbs, const sbn3_formula_options *, void *object,
                                         size_t object_capacity, sbn3_formula_sum_plan *,
                                         sbn3_formula_sum_info *);
void sbn3_formula_sum_bind(const sbn3_formula_sum_plan *, const void *object, sbn3_arena *, size_t offset,
                           sbn3_team *, sbn3_formula_sum_binding **);
void sbn3_formula_sum_execute(sbn3_formula_sum_binding *, sbn3_limbs);
sbn3_const_limbs sbn3_formula_sum_execute_inplace(sbn3_formula_sum_binding *);
void sbn3_formula_sum_get_metrics(const sbn3_formula_sum_binding *, sbn3_formula_sum_metrics *);
void sbn3_formula_sum_unbind(sbn3_formula_sum_binding *);
#ifdef __cplusplus
}
#endif
#endif
