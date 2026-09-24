#ifndef SBN3_NEWTON_H
#define SBN3_NEWTON_H
#include "sbn3/product.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum sbn3_newton_kind {
    SBN3_NEWTON_INVERSE=0,
    SBN3_NEWTON_RSQRT=1,
    SBN3_NEWTON_DIVIDE=2,
    SBN3_SQRT2_RSQRT=3,
    SBN3_SQRT2_RATIONAL=4
} sbn3_newton_kind;
typedef struct sbn3_newton_options {
    unsigned workers;       /* maximum 1..32; small stages may use fewer */
    unsigned prime_count;   /* 0 selects over NP4..10; explicit value pins NTT */
    size_t memory_budget;   /* 0: no extra filter; includes all binding storage */
    unsigned timing;        /* 1 records whole-operation compute/verify times */
} sbn3_newton_options;
typedef struct sbn3_newton_plan {uint64_t opaque[256];} sbn3_newton_plan;
typedef struct sbn3_newton_info {
    sbn3_newton_kind kind;
    unsigned workers,products,spectra,stages,lease_peak;
    size_t precision_limbs,output_limbs,storage_bytes,storage_alignment;
    size_t control_bytes,table_bytes,product_workspace_bytes,value_bytes;
    size_t spectrum_bytes; /* complete cache storage, including its own tables */
    /* With stage pooling, table/workspace/spectrum report component maxima;
     * they need not coexist. storage_bytes is the actual allocation bound. */
    size_t shared_bytes,setup_bytes;
    uint64_t plan_id;
} sbn3_newton_info;
typedef struct sbn3_newton_inputs {
    sbn3_const_limbs numerator,denominator;
    uint64_t radicand;
} sbn3_newton_inputs;
typedef struct sbn3_newton_metrics {
    uint64_t compute_ns,verify_ns;
    unsigned unit_corrections,products_executed,spectra_computed;
} sbn3_newton_metrics;
typedef struct sbn3_newton_binding sbn3_newton_binding;

/* B=2^64. INVERSE: normalized n-limb D -> U with B^n<=U<2B^n,
 * |U-B^(2n)/D|<3. RSQRT: positive u64 a -> R, high limb zero,
 * |R-B^n/sqrt(a)|<3. DIVIDE (n>=1): normalized n-limb D, n+1-limb A
 * with A[n]<=1 -> |Q-B^n*A/D|<3. Small default requests use word/native
 * quotient kernels; larger requests use a half-precision inverse.
 * SQRT2 kinds return exactly floor(B^n*sqrt(2)), with integer square proof.
 * All outputs have n+1 limbs. Approximate entries do not promise div_qr. */
sbn3_query_result sbn3_newton_query(sbn3_newton_kind,size_t n,const sbn3_newton_options *,
                                    sbn3_newton_plan *,sbn3_newton_info *);
/* Caller prepares an unleased, aligned arena range of info.storage_bytes.
 * The entire supplied range is exclusively assigned until unbind; callers
 * must not reuse temporarily unleased gaps inside a compact stage pool.
 * The binding owns concrete nonoverlapping leases inside that range, not a
 * parent lease covering them. Team storage/stacks and outputs are outside it.
 * Bind prepares immutable recipes. A compact binding may construct stage
 * tables/caches in a reused resident span; there are no heap/page operations
 * or plan searches during execution. Expired spectra are destroyed first.
 * Plan is reusable; a binding is single-use.
 * After unbind the same prepared range may be rebound without arena growth. */
/* Small non-FFT DIVIDE bindings borrow the caller-granted exclusive range:
 * ordinary builds perform no arena lease operation, while checked/sanitized
 * builds register the lease for lifetime diagnostics. The caller keeps the
 * entire prepared range resident and exclusive until unbind. */
void sbn3_newton_bind(const sbn3_newton_plan *,sbn3_arena *,size_t offset,sbn3_team *,sbn3_newton_binding **);
/* INVERSE uses denominator; RSQRT uses radicand; DIVIDE uses both views;
 * SQRT2 ignores inputs and permits NULL. Output is 64-byte aligned and all
 * value spans are disjoint from the binding's storage and from one another.
 * DIVIDE additionally permits output==numerator exactly (consuming A after
 * its last read); partial overlaps and denominator/output overlap are invalid. */
void sbn3_newton_execute(sbn3_newton_binding *,const sbn3_newton_inputs *,sbn3_limbs);
void sbn3_newton_get_metrics(const sbn3_newton_binding *,sbn3_newton_metrics *);
void sbn3_newton_unbind(sbn3_newton_binding *);
#ifdef __cplusplus
}
#endif
#endif
