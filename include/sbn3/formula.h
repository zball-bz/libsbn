#ifndef SBN3_FORMULA_H
#define SBN3_FORMULA_H
#include "sbn3/series.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Data-defined series constants. A definition is plain data: three factored
 * integer products P, Q, R over the index k, a start index, an optional
 * explicit first term and an outer rational coefficient. The service reuses
 * the finite binary-splitting recipes, the limited partial-sum reduction and
 * the ratio terminal; adding a constant adds a definition, not code.
 *
 *   leaf k:  T = P(k), D = Q(k), U = R(k)           (SBN3_SERIES_COMMON_P2B3)
 *            T = P(k), D = Q(k), U = 1 (R must be 1, Q > 0)   (HYPERDESCENT)
 *            T = P(k) 2^(shift - stride k), D = Q(k)  (BINARY_BBP)
 *   F[a,b)(x) = (T + U x)/D composed left over right; value = T/D
 *   result   = floor(numerator_scale * T / (D 2^denominator_exponent) * 2^(64 n))
 * Each product is constant * (-1)^(alternating k) * prod (a k + b)^power,
 * optionally times an integer polynomial (numerators only). */
typedef struct sbn3_formula_factor {
    uint64_t a;     /* >= 0 */
    int64_t b;      /* signed offset */
    unsigned power; /* 1..16 */
} sbn3_formula_factor;
typedef struct sbn3_formula_product {
    uint64_t constant_low, constant_high; /* |constant| < 2^128, nonzero */
    unsigned negative, alternating;       /* 0 or 1 */
    unsigned factor_count;                /* 0..6 */
    sbn3_formula_factor factor[6];
    unsigned degree;                      /* 0: no polynomial; else 1..4 */
    int64_t coefficient[5];               /* sum coefficient[i] k^i */
} sbn3_formula_product;
typedef struct sbn3_formula_def {
    sbn3_series_recipe recipe;
    uint64_t begin; /* first index */
    sbn3_formula_product P, Q, R;
    int64_t shift;   /* BINARY_BBP only */
    uint32_t stride; /* BINARY_BBP only, >= 1 */
    unsigned explicit_first; /* use first_* at k == begin (e.g. Chudnovsky k = 0) */
    uint64_t first_t, first_d, first_u;
    unsigned first_t_negative;
    uint64_t numerator_scale;  /* 0 or 1: none; else multiply T (fits one word) */
    int denominator_exponent;  /* value is scaled by 2^-denominator_exponent */
} sbn3_formula_def;
typedef struct sbn3_formula_options {
    sbn3_series_options series;   /* series.leaf_terms 0: service policy (32) */
    size_t memory_budget;        /* 0: unfiltered; complete binding storage */
    uint64_t minimum_block_terms; /* 0: 128 */
    size_t minimum_block_limbs;   /* bit-growth floor per limited block (a series whose exact prefix fits
                                   * its own demand is planned as one exact tree and has no such blocks,
                                   * see info.exact_blocks). 0: policy — the largest of
                                   * {working precision (while 64 bytes per limb of it fit the
                                   * last-level cache), half of it, 16384*workers} whose complete
                                   * binding exceeds the storage no plan can avoid (value slots plus
                                   * terminal) by at most max(1/8 of it, 2 MiB per worker), or stays
                                   * within memory_budget when one is given. */
} sbn3_formula_options;
typedef struct sbn3_formula_plan {
    uint64_t opaque[1024];
} sbn3_formula_plan;
/* Support boundary of the constant service: the term ratio |R(k)/Q(k)| must
 * tend to a limit below 1 (degree of R below Q, or equal degrees with a
 * smaller leading coefficient); the certified prefix keeps at most 2^40
 * terms; the value must be nonnegative and, after the outer scale, below
 * 2^63; values within the certificate's error of a multiple of 2^-(64 n)
 * cannot be certified (deterministic compute-face fatal), except an exact
 * zero result. Rejections name their reason in info.rejection. */
typedef struct sbn3_formula_info {
    size_t fractional_limbs, working_limbs, output_limbs, storage_bytes, storage_alignment;
    size_t psr_storage_bytes, terminal_storage_bytes;
    uint64_t terms, plan_id;
    unsigned workers, blocks, leaf_words;
    unsigned exact_blocks;        /* blocks planned as exact trees (no nonzero bit dropped): a result of
                                   * the precision rule, 1 of 1 for series whose exact prefix fits its
                                   * own demand; limited blocks otherwise */
    unsigned contraction;         /* attenuation certificate available */
    uint64_t attenuation_bits;    /* certified attenuation of the omitted tail, bits */
    const char *rejection;        /* static text when the definition is outside the supported class */
} sbn3_formula_info;
typedef struct sbn3_formula_binding sbn3_formula_binding;
typedef struct sbn3_formula_metrics {
    uint64_t prepare_ns, series_ns, merge_ns, terminal_ns;
} sbn3_formula_metrics;
/* Immutable prepared definition; caller-owned storage of this many bytes,
 * 64-byte aligned, alive and unmodified through every plan, bind and
 * execute derived from it. No allocation happens on the compute path. */
size_t sbn3_formula_object_bytes(void);
/* Analyze the definition for floor(value * 2^(64 n)) with n fractional limbs:
 * certifies the term count from the contribution facts, plans the limited
 * partial-sum reduction and the terminal. SBN3_UNSUPPORTED with info.rejection
 * set describes the reason; SBN3_QUERY_CAPACITY reports the requirement. */
sbn3_query_result sbn3_formula_query(const sbn3_formula_def *, size_t n, const sbn3_formula_options *, void *object,
                                     sbn3_formula_plan *, sbn3_formula_info *);
/* Same prepared-region, single-use and controller contracts as the e/pi API.
 * The object pointer must be the one passed to query. */
void sbn3_formula_bind(const sbn3_formula_plan *, const void *object, sbn3_arena *, size_t offset, sbn3_team *,
                       sbn3_formula_binding **);
void sbn3_formula_execute(sbn3_formula_binding *, sbn3_limbs);
sbn3_const_limbs sbn3_formula_execute_inplace(sbn3_formula_binding *);
void sbn3_formula_get_metrics(const sbn3_formula_binding *, sbn3_formula_metrics *);
void sbn3_formula_unbind(sbn3_formula_binding *);
#ifdef __cplusplus
}
#endif
#endif
