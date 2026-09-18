#ifndef SBN3_LOG_H
#define SBN3_LOG_H
#include "sbn3/formula_sum.h"
#ifdef __cplusplus
extern "C" {
#endif
/* ArcCoth(m) = atanh(1/m) for m>=2. Use the ordinary sbn3_formula_bind/
 * execute/unbind functions and sbn3_formula_object_bytes() with the plan.
 *
 * Two exact series definitions exist; both are plain formula data.
 *   TAYLOR       sum_{k>=0} 1/((2k+1) m^(2k+1)). CommonP2B3 with P=m,
 *                Q=m^2(2k+1), R=2k+1; powers of two use the equivalent
 *                BinaryBBP form (P=1, Q=2k+1, shift=-s, stride=2s).
 *   ACCELERATED  x=m^2, k>=1, value = (1/4) sum P/Q prod_{j<k} R/Q with
 *                P(k) = m [8(9x^2-15x+4)k - 4x(3x-5)]
 *                Q(k) = 3x(x-1)^2 (6k-1)(6k-5),  R(k) = 8k(2k-1),
 *                all divided by gcd(8, 3x(x-1)^2). Term ratio tends to
 *                4/(27 m^2 (m^2-1)^2). Representable in this data schema
 *                while the linear coefficient of P fits int64 (m <= 31817,
 *                or m <= 26754 when m = 2 mod 4); otherwise UNSUPPORTED.
 * AUTO takes ACCELERATED when representable and TAYLOR otherwise.
 *
 * atanh(a/b) for coprime integers 1 <= a, 2a <= b is the same pair of series
 * with m = b/a cleared of denominators (A = a^2, B = b^2):
 *   TAYLOR       P = a b, Q = B (2k+1), R = A (2k+1)
 *   ACCELERATED  P(k) = a b [8(9B^2-15AB+4A^2)k - 4B(3B-5A)]
 *                Q(k) = 3B(B-A)^2 (6k-1)(6k-5),  R(k) = 8A^3 k(2k-1),
 *                all divided by their common divisor; term ratio tends to
 *                4A^3/(27 B (B-A)^2). Representable while b <= 65536 and the
 *                polynomial coefficients of P fit int64 (b up to about 3.4e4).
 * The convergence is that of the integer b/a, the constants are larger; such
 * components pay when (b+a)/(b-a) is a small smooth comma that shortens a
 * Log relation (128/125, 256/243, 3136/3125, ...). */
typedef enum sbn3_arccoth_series {
    SBN3_ARCCOTH_AUTO = 0,
    SBN3_ARCCOTH_TAYLOR = 1,
    SBN3_ARCCOTH_ACCELERATED = 2
} sbn3_arccoth_series;
sbn3_query_result sbn3_arccoth_series_definition(uint64_t m, sbn3_arccoth_series, sbn3_formula_def *);
/* UNSUPPORTED unless 1 <= numerator, 2*numerator <= denominator and the two are coprime. */
sbn3_query_result sbn3_atanh_series_definition(uint64_t numerator, uint64_t denominator, sbn3_arccoth_series,
                                               sbn3_formula_def *);
/* Same as the AUTO series. */
sbn3_query_result sbn3_arccoth_definition(uint64_t m, sbn3_formula_def *);
sbn3_query_result sbn3_arccoth_query(uint64_t m, size_t fractional_limbs, const sbn3_formula_options *,
                                     void *object, sbn3_formula_plan *, sbn3_formula_info *);
enum { SBN3_LOG_MAX_TERMS = 35 };
typedef struct sbn3_log_term {
    uint64_t argument;   /* b */
    int64_t coefficient;
    uint64_t numerator;  /* a >= 1, coprime to argument; 1 for ArcCoth(argument) */
} sbn3_log_term;
typedef struct sbn3_log_formula {
    /* log(n) = (2/divisor) sum coefficient[j]*atanh(numerator[j]/argument[j]).
     * Empty for n=1. Identity and coefficients are machine-independent. */
    uint64_t divisor, identity;
    unsigned count;
    double lehmer_measure; /* classical proxy, reported only */
    double work_estimate;  /* selection objective: modelled binary-splitting work of
                            * every component with its AUTO series, in units of the
                            * requested precision, at a fixed reference precision;
                            * a ranking device, not a measured execution time */
    sbn3_log_term term[SBN3_LOG_MAX_TERMS];
} sbn3_log_formula;
/* Bounded exact-relation search for {2,3,5,7}-smooth targets over a fixed
 * catalog of integer and rational arguments, compared with a general
 * integer-halving construction. Every accepted relation is verified exactly
 * on all prime rows. Candidates are ranked by work_estimate. All positive
 * uint32_t inputs are supported; this is not a claim of globally optimal
 * formulas. */
sbn3_query_result sbn3_log_formula_query(uint32_t n, sbn3_log_formula *);
size_t sbn3_log_object_bytes(uint32_t n);
/* Execute through sbn3_formula_sum_bind/execute/unbind. */
sbn3_query_result sbn3_log_query(uint32_t n, size_t fractional_limbs, const sbn3_formula_options *,
                                 void *object, size_t object_capacity, sbn3_formula_sum_plan *,
                                 sbn3_formula_sum_info *);
#ifdef __cplusplus
}
#endif
#endif
