#ifndef SBN3_MUL_H
#define SBN3_MUL_H
#include "sbn3/team.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sbn3_product_spec {size_t a_limbs,b_limbs;} sbn3_product_spec;
typedef enum sbn3_mul_algorithm {SBN3_MUL_AUTO=0,SBN3_MUL_BAILEY=1,SBN3_MUL_FLAT=2,SBN3_MUL_SCALAR=3,SBN3_MUL_U52=4,SBN3_MUL_PQ16=5} sbn3_mul_algorithm;
typedef struct sbn3_mul_options {
    unsigned workers;       /* 1..32, exact execution width */
    unsigned prime_batch;   /* 0 = task/budget-aware policy; explicit 1..np */
    int trunk_bits;         /* NTT: 0 = widest legal T. Explicit PQ16: 0 = cost policy,
                            * 16..20 = digit bits within that codec's supported band. */
    unsigned borrow_output; /* 0: disabled; 1: E1 must fit the logical output;
                            * 2: permit a larger backing span (see output_capacity).
                            * Borrowed output is disjoint from all live inputs. */
    size_t workspace_budget;/* 0 = no additional query filter */
    unsigned column_log2,row_log2; /* both zero: policy; both nonzero: exact Bailey geometry */
    unsigned prime_count;   /* 0 = AUTO: deep W32 includes NP9/10; other widths
                            * retain NP4..8 plus the small flat NP10 candidate.
                            * Explicit 4..10 pins the prime family. */
    unsigned crt_mode;      /* 0 = default; 1 = Garner; 2 = direct CRT (when supported) */
    unsigned codec_mode;    /* 0 = default; 1 = existing narrow/byte decode; 3 = shared slices (T>88); 2 is retired/unsupported */
    unsigned algorithm;       /* AUTO policy, or explicit BAILEY / FLAT / SCALAR / U52 / PQ16 */
    int fused_start_skew_us;   /* 0 = policy; -1 = disabled; positive = us per item */
} sbn3_mul_options;
typedef enum sbn3_query_result {
    SBN3_SUPPORTED=0, SBN3_UNSUPPORTED=1, SBN3_QUERY_CAPACITY=2
} sbn3_query_result;
/* Opaque value storage; no pointers or vector types. Do not interpret words. */
typedef struct sbn3_mul_plan {uint64_t opaque[384];} sbn3_mul_plan;
typedef struct sbn3_mul_info {
    unsigned np,trunk_bits,digit_words,workers,prime_batch;
    unsigned fused,row_major,borrow_output,full;
    size_t C,M2,lbv,lbw,nat,nyt,transform_trunks;
    size_t output_limbs,output_alignment,table_bytes,workspace_bytes,workspace_alignment;
    size_t per_worker_bytes,plane_pitch,pool_bytes,transpose_bytes;
    size_t table_entries,factor_levels,root_order_log2;
    size_t block_stride,product_row_stride,emit_trunks,emit_limbs,emit_quantum_trunks,emit_quantum_limbs;
    uint64_t rowscale,scale_a[10];
    unsigned row_task_grain,fused_row_grain,format_slot_bytes,format_block_slots;
    unsigned fused_items;
    unsigned crt_mode,codec_mode; /* PQ16 result: codec_mode 0=PFA/PQ, 1=CT/PQ, 2=right-angle; bit 3 (8) = balanced digits.
                                  * Other algorithms retain their own codec-mode definitions. */
    uint64_t basis_id,arithmetic_id,execution_id;
    unsigned algorithm,fused_start_skew_us; /* resolved execution choices */
} sbn3_mul_info;
/* Physical writable capacity, in limbs. output_limbs remains the mathematical
 * result length. With resolved borrow_output==2, the extra tail is scratch,
 * has unspecified contents and must be included in every alias/lifetime check.
 * This helper does not change the size or alignment of the C ABI structures. */
static inline size_t sbn3_mul_output_capacity(const sbn3_mul_info *info) {
    const size_t scratch = info->borrow_output == 2
        ? info->transpose_bytes / 8 + (info->transpose_bytes % 8 != 0) : 0;
    return scratch > info->output_limbs ? scratch : info->output_limbs;
}
typedef struct sbn3_const_limbs {const uint64_t *data;size_t count;} sbn3_const_limbs;
typedef struct sbn3_limbs {uint64_t *data;size_t capacity;} sbn3_limbs;
typedef struct sbn3_mul_binding sbn3_mul_binding;
typedef struct sbn3_mul_metrics {
    size_t worker_peak_bytes,table_used_bytes,workspace_used_bytes;
    uint64_t executions; /* Short kernels: zero unless SBN3_SHORT_STATS=1 in the library build. */
    uint64_t last_stage_ns[4]; /* Bailey: rows(A), rows(B), tile, inverse+emit.
                              * Flat: complete per-prime pipeline, 0, 0, CRT+emit.
                              * Scalar/u52/pq16: zero in normal builds;
                              * SBN3_SHORT_TIMING=1 records product, 0, 0, 0.
                              * Memory peaks remain available; short execution
                              * counts are optional (enabled by diagnostics/timing). */
} sbn3_mul_metrics;

/* Pure query. On capacity rejection info reports the computed requirement;
 * plan is written only on success. No source/output buffers are accessed. */
sbn3_query_result sbn3_mul_query(const sbn3_product_spec *,const sbn3_mul_options *,
                                 sbn3_mul_plan *,sbn3_mul_info *);
/* Table/control/work storage are explicitly prepared leases, disjoint from
 * team control/stacks and each other. Use info.workspace_alignment for work storage;
 * table storage is at least 128-byte aligned for short products (64 for p48). Bind materializes tables with no allocation.
 * Table and workspace spans are exclusively owned until unbind. */
void sbn3_mul_bind(const sbn3_mul_plan *,sbn3_arena *,const sbn3_lease *tables,
                    const sbn3_lease *workspace,sbn3_team *,sbn3_mul_binding **);
/* Output obeys info.output_alignment, has sbn3_mul_output_capacity(info) limbs,
 * and that complete writable span is disjoint from inputs and bound resources.
 * Input/output values are caller-owned; this operation returns no error code.
 * Normal small kernels trust matching bound sizes, capacity, live bindings and
 * exclusive workspace. Query/bind are checked; --checked and sanitizer builds
 * additionally validate execution preconditions. */
void sbn3_mul_execute(sbn3_mul_binding *,sbn3_const_limbs,sbn3_const_limbs,sbn3_limbs);
/* Thin entry for a plain MUL binding. Sizes are fixed by query/bind;
 * provide the queried input/output spans and exclusive live workspace.
 * Pointer arguments avoid passing redundant size/capacity views by value. */
void sbn3_mul_execute_ptrs(sbn3_mul_binding *,const uint64_t *a,const uint64_t *b,uint64_t *out);
/* Cooperative use inside a preplanned recursive team action. Bind before
 * entering the root computation; scope width must equal the plan's width. */
void sbn3_mul_execute_on_scope(sbn3_mul_binding *,sbn3_team_scope *,
                                sbn3_const_limbs,sbn3_const_limbs,sbn3_limbs);
void sbn3_mul_get_metrics(const sbn3_mul_binding *,sbn3_mul_metrics *);
void sbn3_mul_unbind(sbn3_mul_binding *);
/* A MUL binding applied to signed caller-owned values; normalizes sign/length. */
void sbn3_int_mul_execute(sbn3_mul_binding *,sbn3_int *,sbn3_int_view,sbn3_int_view);
#ifdef __cplusplus
}
#endif
#endif
