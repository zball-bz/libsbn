#ifndef SBN3_PRODUCT_H
#define SBN3_PRODUCT_H
#include "sbn3/mul.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Positive integer recipes. TMP returns the certified middle window below. */
typedef enum sbn3_product_kind {
    SBN3_PRODUCT_MUL=0, SBN3_PRODUCT_SQR=1, SBN3_PRODUCT_TMP=2, SBN3_PRODUCT_MAC2=3,
    SBN3_PRODUCT_LOW=4, SBN3_PRODUCT_HIGH=5
} sbn3_product_kind;
typedef enum sbn3_spectrum_frontier { SBN3_SPECTRUM_ROWS=0, SBN3_SPECTRUM_COLUMNS=1 } sbn3_spectrum_frontier;
typedef struct sbn3_spectrum sbn3_spectrum;
typedef struct sbn3_spectrum_desc {
    uint64_t basis_id,instance_id,generation,seal;
    unsigned np,trunk_bits,frontier,format_version; /* 1 blocked48 Bailey; 2 flat lazy64; 3 FFT AoSoA; 4 flat packed48 */
    size_t C,M2,transform_trunks,live_slots,written_slots,source_limbs,source_trunks,block_stride;
    uint64_t scale[10]; /* NTT: per-prime scale; FFT: scale[0]=1, others zero */
    size_t storage_bytes,table_bytes,plane_bytes;
    uint64_t backend_id;       /* producer/consumer arithmetic family */
    unsigned codec_mode;      /* FFT: recipe | balanced(8) | centered(16) */
} sbn3_spectrum_desc;
typedef struct sbn3_product_request {
    sbn3_product_kind kind;
    size_t a_limbs,b_limbs,a1_limbs,b1_limbs;
    /* Optional read-only cached A for term 0/1. SQR uses only a_limbs/A0;
     * MUL/TMP use term 0; MAC2 uses both terms. Descriptors are copied. */
    const sbn3_spectrum_desc *cached_a[2];
    /* MUL/SQR only: nonzero means modulo 2^(64*cyclic_limbs)-1.
     * Each input has at most cyclic_limbs limbs; the all-one input is legal.
     * Output is exactly cyclic_limbs limbs, canonical in [0, modulus).
     * Query supports native transform periods, never silently pads the ring
     * or substitutes a complete linear product. Zero retains linear semantics. */
    size_t cyclic_limbs;
    /* With cyclic_limbs: 0 => minus, 1 => plus modulus. Plus permits the
     * canonical r+1-limb residue 2^(64*r); output then has r+1 limbs. */
    unsigned negacyclic;
    /* LOW/HIGH: exact low/top window of the fixed (a_limbs+b_limbs)
     * word product. 1..a_limbs+b_limbs. Initial native word kernels support
     * <=256 limbs per input; larger requests explicitly query unsupported. */
    /* Also on minus CYC MUL/SQR: nonzero requests an exact low prefix.
     * Query must prove a_limbs+b_limbs (twice a_limbs for SQR) < ring.
     * Only window_limbs output words are writable; the spectrum basis stays
     * full-sized. This is not a prefix of a generally wrapped product. */
    size_t window_limbs;
} sbn3_product_request;
typedef struct sbn3_window_certificate {
    uint64_t offset_bits,width_bits,error_bits;
    /* V is the returned integer. For TMP there exists integer e with
     * |e| < 2^error_bits and V+e == floor(A*B/2^offset_bits) mod 2^width_bits.
     * Extra returned high padding is not part of the window. */
} sbn3_window_certificate;
typedef struct sbn3_product_info {
    sbn3_mul_info mul;
    sbn3_product_kind kind;
    unsigned cached_mask;
    sbn3_window_certificate window;
    size_t spectrum_bytes,spectrum_alignment;
    uint64_t leaf_scale[2][10];
    size_t cyclic_limbs;
    unsigned negacyclic;
} sbn3_product_info;
typedef struct sbn3_product_inputs { sbn3_const_limbs a,b,a1,b1; } sbn3_product_inputs;
typedef struct sbn3_product_metrics {
    sbn3_mul_metrics mul;
    /* Last execution's actual kernel calls, summed over NTT primes. FFT
     * reports complete forwards/inverses in row_forward/row_inverse. */
    uint64_t row_forward,row_mid,column_forward,column_inverse,row_inverse,row_transpose_forward,leaf_products;
} sbn3_product_metrics;

sbn3_query_result sbn3_product_query(const sbn3_product_request *,const sbn3_mul_options *,sbn3_mul_plan *,sbn3_product_info *);
/* Cache arguments must match the descriptors used in query. Bind retains them.
 * The table/workspace requirements in info exclude independent cached storage. */
void sbn3_product_bind(const sbn3_mul_plan *,sbn3_arena *,const sbn3_lease *tables,const sbn3_lease *workspace,
                       sbn3_team *,const sbn3_spectrum *a,const sbn3_spectrum *a1,sbn3_mul_binding **);
void sbn3_product_execute(sbn3_mul_binding *,const sbn3_product_inputs *,sbn3_limbs);
void sbn3_product_execute_on_scope(sbn3_mul_binding *,sbn3_team_scope *,const sbn3_product_inputs *,sbn3_limbs);
void sbn3_product_get_metrics(const sbn3_mul_binding *,sbn3_product_metrics *);

/* Prepare the binding plan's term-0 A in independent caller-provided storage.
 * Uses its idle team's already-resident workspace; input may be released later.
 * The handle owns its data/tables lease until all user and binding refs end. */
void sbn3_spectrum_prepare(sbn3_mul_binding *,sbn3_const_limbs,sbn3_spectrum_frontier,uint64_t generation,
                           sbn3_arena *,const sbn3_lease *storage,sbn3_spectrum **);
/* Pure description of a future term-0 A spectrum. instance_id==0 means a
 * mathematical requirement, not a particular allocation. Suitable for
 * planning cached consumers before the input value has been computed. */
sbn3_query_result sbn3_spectrum_query(const sbn3_mul_plan *,sbn3_spectrum_frontier,uint64_t generation,sbn3_spectrum_desc *);
/* Reserve tables/storage without a forward transform. Cached consumers may
 * bind this handle, but can_apply is false and execution is forbidden until
 * compute completes. compute fills it exactly once, with the producer's
 * geometry/scale and an idle team's preallocated workspace. */
void sbn3_spectrum_reserve(sbn3_mul_binding *,sbn3_spectrum_frontier,uint64_t generation,
                           sbn3_arena *,const sbn3_lease *storage,sbn3_spectrum **);
/* Reserve directly from the producing plan. Table rows are built in place;
 * no temporary producer binding/team or extra workspace is needed. */
void sbn3_spectrum_reserve_plan(const sbn3_mul_plan *,sbn3_spectrum_frontier,uint64_t generation,
                                sbn3_arena *,const sbn3_lease *storage,sbn3_spectrum **);
void sbn3_spectrum_compute(sbn3_mul_binding *,sbn3_spectrum *,sbn3_const_limbs);
/* Fill the reserved cache bound to a cached SQR plan and execute that square
 * in one team episode. Cache becomes immutable on return; its data can be
 * reused by subsequent products. The plan's optional exact low prefix applies. */
void sbn3_spectrum_compute_square(sbn3_mul_binding *,sbn3_spectrum *,sbn3_const_limbs,sbn3_limbs);
/* The descriptor declares the reserved representation; can_apply additionally
 * checks that its computation has completed. */
void sbn3_spectrum_describe(const sbn3_spectrum *,sbn3_spectrum_desc *);
int sbn3_spectrum_can_apply(const sbn3_mul_plan *,const sbn3_spectrum *,unsigned term);
void sbn3_spectrum_retain(const sbn3_spectrum *);
void sbn3_spectrum_release(const sbn3_spectrum *);
#ifdef __cplusplus
}
#endif
#endif
