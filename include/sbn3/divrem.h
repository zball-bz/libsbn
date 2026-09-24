#ifndef SBN3_DIVREM_H
#define SBN3_DIVREM_H
#include "sbn3/newton.h"
#include "sbn3/value.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Exact integer division, B=2^64, little-endian limbs.
 *   unsigned:  N = Q*D + R, 0 <= R < D
 *   signed:    N = Q*D + R, |R| < |D|, truncation toward zero, R has the sign of N.
 * A zero divisor is a fatal SBN3_FATAL_ARGUMENT diagnostic on the value
 * boundary (prepare/basecase); it never produces a result or a SIGFPE. */

/* Thin one-shot entry: schoolbook division without plans, arena or team.
 * Work is O(nn*dn) words. D has dn>=1 limbs, D[dn-1]!=0; N has nn limbs
 * (leading zeros allowed). scratch has nn+dn+1 limbs. Writes q[0..nn-dn+1)
 * when nn>=dn (nothing otherwise; the quotient is then zero) and r[0..dn).
 * All five spans are mutually disjoint. Returns the normalized quotient
 * length (0 for a zero quotient); the normalized remainder length is
 * recoverable by scanning r. Suitable for small operands and short
 * quotients; larger balanced shapes should use the prepared service. */
size_t sbn3_divrem_basecase(uint64_t *q, uint64_t *r, const uint64_t *n, size_t nn,
                            const uint64_t *d, size_t dn, uint64_t *scratch);
/* Signed truncating wrapper. q capacity >= max(n.size-d.size+1,0), r capacity >= d.size,
 * scratch capacity >= n.size+d.size+1; all outputs are normalized (no negative zero). */
void sbn3_int_divrem_basecase(sbn3_int *q, sbn3_int *r, sbn3_int_view n, sbn3_int_view d, sbn3_limbs scratch);

/* One-shot local division: 1<=dn<=32768, nn<=2^40, D[dn-1]!=0.
 * Simple thresholds select word, native two-vector/DC or local Barrett. No external plan,
 * team, arena binding, retained divisor, allocation or page operation.
 * Q/R capacities and returned quotient length match divrem_basecase.
 * All spans are disjoint. Scratch is caller-owned, at least the queried
 * number of bytes, 8-byte aligned; it may be null when the query is zero. */
size_t sbn3_divrem_small_scratch_bytes(size_t nn,size_t dn);
size_t sbn3_divrem_small(uint64_t *q,uint64_t *r,const uint64_t *n,size_t nn,
                         const uint64_t *d,size_t dn,void *scratch);

/* Prepared service: thresholds select word/schoolbook, local u52 D&C,
 * block Barrett, or a fused short-quotient terminal followed by an exact
 * residual. Large recipes use a bounded Newton inverse and planned spectra.
 * One binding serves any
 * numerator of at most numerator_limbs limbs against the prepared divisor,
 * and may be re-prepared with another divisor of the same length. Quotient
 * limbs left over by the block size (the first limb of 2d/d under a full
 * inverse) may use up to four exact O(dn) word steps; their integer buffer
 * capacity is independent of spare transform-ring limbs. */
typedef struct sbn3_divrem_request {
    size_t numerator_limbs;   /* maximum numerator length accepted by execute */
    size_t denominator_limbs; /* exact length: the prepared divisor's top limb is nonzero */
} sbn3_divrem_request;
typedef struct sbn3_divrem_options {
    unsigned workers;       /* 1..32 */
    unsigned prime_count;   /* 0 selects; explicit 4..10 pins NTT product families; seeds retain their local kernels */
    size_t memory_budget;   /* 0: no filter; otherwise the policy passes over plans whose storage_bytes exceed it (docs/division.md) */
    unsigned reuse_hint;    /* expected executions per divisor; 0/1 is fresh. May increase
                               retained inverse precision to amortize preparation. No plan grid is searched. */
    size_t block_limbs;     /* 0: policy (near-equal blocks); otherwise the requested Barrett block size:
                               denominator_limbs asks for complete blocks under a full inverse */
    unsigned residual;      /* 0: policy; 1: linear residual product only; 2: cyclic only (experiments) */
    unsigned algorithm;     /* 0: policy; otherwise force SCHOOLBOOK, BARRETT (dn>=3), or DC (3<=dn<=2^20) */
    unsigned timing;        /* 1 records prepare/execute durations */
} sbn3_divrem_options;
typedef enum sbn3_divrem_algorithm {
    SBN3_DIVREM_WORD=0,       /* dn<=2: single/double-limb division */
    SBN3_DIVREM_SCHOOLBOOK=1, /* 3/2 quotient estimates, submul updates */
    SBN3_DIVREM_BARRETT=2,    /* block quotients from a Newton inverse, cyclic residual */
    SBN3_DIVREM_DC=3          /* native u52 divide-and-conquer with 416-bit quotient-block leaves */
} sbn3_divrem_algorithm;
/* Carries the resolved block inverse (a Newton plan above the schoolbook-inverse block sizes) and product recipes
 * so bind performs no plan search. */
typedef struct sbn3_divrem_plan {uint64_t opaque[2560];} sbn3_divrem_plan;
typedef struct sbn3_divrem_info {
    size_t numerator_limbs,denominator_limbs,quotient_limbs,remainder_limbs;
    unsigned algorithm,workers,products,blocks; /* products per block (2 Barrett, 4 fused); blocks for longest numerator */
    size_t block_limbs,inverse_limbs,ring_limbs; /* Barrett geometry; ring 0 means a linear residual product */
    size_t storage_bytes,storage_alignment;      /* complete binding storage */
    size_t control_bytes,persistent_bytes,shared_bytes; /* control; normalized divisor/U/spectra; phase union */
    size_t table_bytes,product_workspace_bytes,spectrum_bytes,scratch_bytes;
    /* Tables/product workspace/scratch are phase components, not additive to
     * shared_bytes. storage_bytes is the complete maximum across phases. */
    uint64_t plan_id;
    size_t head_limbs; /* longest numerator: leading quotient limbs by O(dn) word division, outside blocks */
} sbn3_divrem_info;
typedef struct sbn3_divrem_result {
    size_t quotient_limbs,remainder_limbs; /* normalized lengths; zero for a zero value */
    uint64_t corrections;                  /* Barrett block corrections; other recipes do not collect this counter */
} sbn3_divrem_result;
typedef struct sbn3_divrem_metrics {
    uint64_t prepare_ns,execute_ns;        /* last prepare / execute, timing option only */
    uint64_t prepares,executes,corrections; /* since bind */
    unsigned products_executed;            /* last execute */
    uint64_t head_limbs,head_corrections;  /* since bind: quotient limbs by word division; their add-backs */
} sbn3_divrem_metrics;
typedef struct sbn3_divrem_binding sbn3_divrem_binding;

/* Pure query. Null options select ordinary W1 defaults. The arithmetic
 * recipe is chosen by thresholds and capacity calculations before layout.
 * Under memory_budget, compact recipes try their fixed smaller-block/DC
 * fallbacks; the general Barrett recipe has a bounded block-halving ladder.
 * A short divisor/quotient may use bounded schoolbook work if needed.
 * Explicit algorithm, block and residual options are constraints, never
 * silently replaced. Failure leaves the plan untouched; CAPACITY reports
 * the least complete storage requirement encountered by this fixed policy. */
sbn3_query_result sbn3_divrem_query(const sbn3_divrem_request *,const sbn3_divrem_options *,
                                    sbn3_divrem_plan *,sbn3_divrem_info *);
/* Caller prepares an unleased, aligned arena range of info.storage_bytes,
 * exclusively assigned until unbind. The team is idle and owned by the
 * caller; its storage/stacks are outside the range. The plan is reusable;
 * after unbind the same prepared range may be rebound without arena growth.
 * Compact word/DC/local bindings borrow this caller-owned range. The caller
 * keeps it resident and exclusive until unbind, including between calls;
 * checked builds additionally register/check its arena lease. */
void sbn3_divrem_bind(const sbn3_divrem_plan *,sbn3_arena *,size_t offset,sbn3_team *,sbn3_divrem_binding **);
/* Prepare the divisor: normalization, the block inverse and cached spectra.
 * denominator has exactly denominator_limbs limbs with a nonzero top limb and
 * is read only during this call (the binding keeps its own copy). Repeated
 * prepare replaces the divisor; nothing is allocated. */
void sbn3_divrem_prepare(sbn3_divrem_binding *,sbn3_const_limbs denominator);
/* Construct a prepared divisor directly in caller-provided resident storage.
 * Performs selection, binding and all divisor preparation in this call, then
 * returns the same reusable binding accepted by execute/prepare/unbind.
 * Query remains available for support/budget admission and storage sizing;
 * no queried plan or operand cache is required as input here. storage_bytes
 * must cover this request/options and the range has the bind lifetime above.
 * Unsupported requests or insufficient supplied storage violate a caller
 * precondition and are fatal, like execute with insufficient output capacity.
 * No allocation, page operation or hidden cross-call cache is introduced. */
sbn3_divrem_binding *sbn3_divrem_prepare_into(const sbn3_divrem_request *,const sbn3_divrem_options *,
                                             sbn3_arena *,size_t offset,size_t storage_bytes,sbn3_team *,
                                             sbn3_const_limbs denominator);
/* Q=floor(N/D), R=N-Q*D for a prepared divisor. numerator has at most
 * numerator_limbs limbs (leading zeros allowed). quotient capacity is at
 * least numerator.count-denominator_limbs+1 (0 when the numerator is
 * shorter), remainder capacity at least denominator_limbs; exactly those
 * many limbs are written, high zeros included. Both outputs are 8-byte
 * aligned. numerator, quotient, remainder, the binding storage and the
 * team storage are mutually disjoint; no aliasing is supported. No plan
 * search, allocation or page operation occurs. Repeated execute is allowed.
 * Compact execution trusts these caller/plan preconditions in normal builds;
 * checked builds audit lifetime, capacity and aliasing. Divisor-zero and
 * arithmetic invariant diagnostics are retained in every build. */
void sbn3_divrem_execute(sbn3_divrem_binding *,sbn3_const_limbs numerator,sbn3_limbs quotient,
                         sbn3_limbs remainder,sbn3_divrem_result *);
/* Signed truncating division over the prepared |d|: q/r are caller-owned
 * with capacities as above; d must be the prepared divisor's magnitude
 * (same limbs) and carries the sign. Outputs are normalized. */
void sbn3_int_divrem_execute(sbn3_divrem_binding *,sbn3_int *q,sbn3_int *r,sbn3_int_view n,sbn3_int_view d);
void sbn3_divrem_get_metrics(const sbn3_divrem_binding *,sbn3_divrem_metrics *);
void sbn3_divrem_unbind(sbn3_divrem_binding *);
#ifdef __cplusplus
}
#endif
#endif
