#ifndef SBN3_SERIES_H
#define SBN3_SERIES_H
#include "sbn3/mul.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Finite ordered reductions. Infinite-series tail/rounding and the formula's
 * leaf/merge arithmetic belong to the consumer. No implicit allocation. */
typedef enum sbn3_series_recipe {
    SBN3_SERIES_HYPERDESCENT = 0,
    SBN3_SERIES_COMMON_P2B3 = 1,
    SBN3_SERIES_BINARY_BBP = 2
} sbn3_series_recipe;
enum { SBN3_SERIES_T = 1, SBN3_SERIES_D = 2, SBN3_SERIES_U = 4 };
typedef struct sbn3_series_range {
    uint64_t begin, end;
} sbn3_series_range;
typedef struct sbn3_series_shape {
    size_t limbs[3];
} sbn3_series_shape;
typedef struct sbn3_series_spec {
    sbn3_series_recipe recipe;
    sbn3_series_range range; /* nonempty [begin,end), ordered; no commutation */
    unsigned need;
    uint64_t formula_id, parameter_id; /* caller versions mathematical semantics */
} sbn3_series_spec;
typedef struct sbn3_series_options {
    unsigned workers;           /* 1..32 */
    unsigned leaf_terms;        /* 1..1024; callback evaluates this finite batch */
    unsigned max_serial_prefix; /* 0..4; candidate serial levels above task forks */
    double min_parallel_work;   /* same units as work(); 0 permits all forks */
    size_t workspace_budget;    /* 0: unfiltered; excludes outputs/plan/team */
} sbn3_series_options;
typedef struct sbn3_series_stage {
    size_t index;
    size_t prepared_offset; /* disjoint stage preparation storage; no runtime allocation */
    sbn3_series_range envelope;
    uint64_t max_terms;
    unsigned workers, need, leaf, serial_envelope;
    sbn3_series_shape output, left, right;
} sbn3_series_stage;
typedef struct sbn3_series_resources {
    size_t bytes, alignment; /* power of two, <=2 MiB; bytes excludes alignment slop */
    double work;             /* estimated elapsed merge/leaf work at stage.workers */
    size_t prepared_bytes,
        prepared_alignment; /* tables/control retained across executions; 0 bytes allowed */
} sbn3_series_resources;
typedef struct sbn3_series_oracle {
    const void *context;
    /* Must bound EVERY contiguous subrange of envelope with <=max_terms.
     * Bounds include the sum, signs, exact exponent alignment and carry room.
     * Pure, deterministic, no allocation. Not a floating-point size estimate. */
    sbn3_query_result (*bounds)(const void *, sbn3_series_range, uint64_t max_terms, unsigned need,
                                sbn3_series_shape *);
    /* Positive additive size/work proxy; used only to choose a coarse split.
     * Approximate values cannot weaken bounds(). No scratch guarantee uses it. */
    double (*work)(const void *, sbn3_series_range);
    /* Must cover all operand lengths <= the supplied shapes, including all
     * subranges in a serial envelope. Plans/tables may be prepared by visiting
     * stages before execute. A stage is never concurrently reused. */
    sbn3_query_result (*resources)(const void *, const sbn3_series_stage *, sbn3_series_resources *);
} sbn3_series_oracle;
typedef struct sbn3_series_info {
    size_t plan_bytes, plan_alignment, workspace_bytes, workspace_alignment;
    size_t prepared_bytes, prepared_alignment; /* consumer prepares this separate span via visit() */
    sbn3_series_shape output;
    size_t stages, coarse_nodes, serial_subtrees;
    unsigned serial_prefix, max_depth;
    double estimated_work;
    uint64_t mathematical_id, schedule_id;
} sbn3_series_info;
typedef struct sbn3_series_plan sbn3_series_plan;
typedef struct sbn3_series_value {
    sbn3_int mantissa;
    int64_t exponent2; /* exact mantissa * 2^exponent2; BBP may cross limb boundaries */
} sbn3_series_value;
typedef struct sbn3_series_values {
    sbn3_series_value value[3];
} sbn3_series_values;
typedef struct sbn3_series_executor {
    void *context;
    /* Callbacks finish synchronously, initialize only requested outputs, and
     * may dispatch arithmetic on scope. Inputs are immutable through callback
     * return; scratch and children die immediately afterward. No OS allocation,
     * query fallback, cancellation, or error return on the compute path. */
    void (*leaf)(void *, const sbn3_series_stage *, sbn3_series_range, unsigned need, sbn3_series_values *,
                 void *scratch, size_t, sbn3_team_scope *);
    void (*merge)(void *, const sbn3_series_stage *, sbn3_series_range, uint64_t split, unsigned need,
                  const sbn3_series_values *, const sbn3_series_values *, sbn3_series_values *, void *scratch,
                  size_t, sbn3_team_scope *);
} sbn3_series_executor;

/* Query reports the smallest tested requirement on budget rejection. init is
 * a preparation boundary into caller-owned, aligned storage; callbacks must
 * return the same answers as query. Plan holds no value or oracle pointers. */
sbn3_query_result sbn3_series_query(const sbn3_series_spec *, const sbn3_series_options *,
                                    const sbn3_series_oracle *, sbn3_series_info *);
void sbn3_series_plan_init(const sbn3_series_spec *, const sbn3_series_options *, const sbn3_series_oracle *,
                           void *storage, size_t, sbn3_series_plan **);
void sbn3_series_visit(const sbn3_series_plan *,
                       void (*)(void *, const sbn3_series_stage *, const sbn3_series_resources *), void *);
/* Output slots and scratch are 64-byte aligned, disjoint from each other,
 * plan/team/consumer resources, and sized according to info. Caller keeps
 * their prepared leases alive. Plan is reusable with exclusive RunBindings. */
void sbn3_series_execute(const sbn3_series_plan *, const sbn3_series_executor *, sbn3_team *,
                         sbn3_series_values *, void *scratch, size_t);
void sbn3_series_child_needs(sbn3_series_recipe, unsigned need, unsigned *left, unsigned *right);
#ifdef __cplusplus
}
#endif
#endif
