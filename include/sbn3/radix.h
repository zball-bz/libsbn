#ifndef SBN3_RADIX_H
#define SBN3_RADIX_H
#include "sbn3/series.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Radix conversion, bases 2..64, of exact binary floating-point (dyadic) numbers
 *     X = M * 2^exponent2,   0 <= M < 2^(64 * limbs),
 * M little-endian u64 limbs; the sign is carried by the caller. A big integer is
 * exponent2 >= 0 (usually 0); a pure fraction is exponent2 = -64 * limbs; a big
 * float (sbn3_series_value, the constant services) is anything in between.
 *
 * Digits are bytes, most significant first: the integer area, then the fraction
 * area; the point is not stored. A byte is a digit value 0..base-1 unless the
 * options carry an alphabet. No allocation: query -> bind -> execute -> unbind
 * on caller-prepared arena storage, like the other services. */
typedef enum sbn3_radix_mode {
    /* X is exact. Fraction digits are floor(frac(X) * base^D): certified truncation. */
    SBN3_RADIX_EXACT = 0,
    /* X is the floor, at its last bit, of an unknown real v in [X, X + 2^exponent2)
     * (the contract of the constant services). Only digits shared by that whole
     * interval are returned; requires exponent2 <= 0. */
    SBN3_RADIX_ENCLOSED = 1
} sbn3_radix_mode;
typedef struct sbn3_radix_options {
    unsigned workers;           /* 1..32 */
    size_t memory_budget;       /* 0: no filter; otherwise an upper bound of storage_bytes */
    unsigned repeated;          /* nonzero: the binding converts many values; plan for execute time alone
                                 * (bind may then cost several conversions). 0: plan for bind + one execute. */
    unsigned use_alphabet;      /* 0: bytes are digit values */
    unsigned char alphabet[64]; /* use_alphabet: byte written/read for digit value j, distinct */
} sbn3_radix_options;
/* The conventional alphabet: 0-9 a-z up to base 36; 0-9 A-Z a-z + / beyond. */
void sbn3_radix_alphabet(unsigned base, unsigned char out[64]);

/* ---- format: dyadic -> digits ---- */
typedef struct sbn3_format_spec {
    unsigned base;
    size_t limbs;
    int64_t exponent2;
    uint64_t fraction_digits; /* requested digits after the point */
    sbn3_radix_mode mode;
} sbn3_format_spec;
typedef struct sbn3_format_plan {uint64_t opaque[64];} sbn3_format_plan;
typedef struct sbn3_format_info {
    unsigned workers, lease_peak;
    uint64_t integer_digits;  /* bytes of the integer area, leading zero digits included (0: X < 1 always) */
    uint64_t fraction_digits; /* digits execute can return: the request, less in ENCLOSED mode when the
                               * input does not determine that many */
    size_t fraction_offset;   /* byte offset of the first fraction digit = padded integer area */
    size_t digit_bytes;       /* required output capacity (both areas are padded to 64 bytes) */
    size_t storage_bytes, storage_alignment;
    size_t control_bytes, table_bytes, value_bytes, workspace_bytes, divide_bytes;
    uint64_t plan_id;
} sbn3_format_info;
typedef struct sbn3_format_result {
    uint64_t integer_first;   /* index of the first nonzero integer digit; integer_digits when the integer part is 0 */
    uint64_t fraction_digits; /* certified digits after the point (== info.fraction_digits in EXACT mode) */
    unsigned exact_fallback;  /* 1: the rare exact resolution of a digit-boundary tie ran */
} sbn3_format_result;
typedef struct sbn3_format_binding sbn3_format_binding;
sbn3_query_result sbn3_format_query(const sbn3_format_spec *, const sbn3_radix_options *, sbn3_format_plan *,
                                    sbn3_format_info *);
/* Storage: an unleased prepared arena range of info.storage_bytes at an offset aligned to
 * info.storage_alignment, exclusively the binding's until unbind. Bind builds the power rail and
 * every product program (it runs products on the team). A binding converts any number of values. */
void sbn3_format_bind(const sbn3_format_plan *, sbn3_arena *, size_t offset, sbn3_team *, sbn3_format_binding **);
/* mantissa.count == spec.limbs; digits has info.digit_bytes bytes; both are disjoint from the storage.
 * Bytes of the two areas beyond the reported digits are unspecified. */
void sbn3_format_execute(sbn3_format_binding *, sbn3_const_limbs mantissa, unsigned char *digits,
                         sbn3_format_result *);
void sbn3_format_unbind(sbn3_format_binding *);
/* Spec of a big float as the series/constant services return it. */
static inline sbn3_format_spec sbn3_format_spec_of(const sbn3_series_value *v, unsigned base,
                                                   uint64_t fraction_digits, sbn3_radix_mode mode) {
    sbn3_format_spec s;
    s.base = base;
    s.limbs = v->mantissa.size;
    s.exponent2 = v->exponent2;
    s.fraction_digits = fraction_digits;
    s.mode = mode;
    return s;
}

/* ---- parse: digits -> dyadic ---- */
typedef struct sbn3_parse_spec {
    unsigned base;
    uint64_t integer_digits;  /* digit bytes before the point (leading zeros allowed) */
    uint64_t fraction_digits; /* digit bytes after the point */
    uint64_t fraction_bits;   /* the result is M = floor(value * 2^fraction_bits), exactly */
} sbn3_parse_spec;
typedef struct sbn3_parse_plan {uint64_t opaque[64];} sbn3_parse_plan;
typedef struct sbn3_parse_info {
    unsigned workers, lease_peak;
    size_t limbs;      /* output capacity: M < 2^(64 * limbs) */
    int64_t exponent2; /* = -fraction_bits: the value is M * 2^exponent2 up to the truncation */
    size_t storage_bytes, storage_alignment;
    size_t control_bytes, table_bytes, value_bytes, workspace_bytes, divide_bytes;
    uint64_t plan_id;
} sbn3_parse_info;
typedef struct sbn3_parse_result {
    unsigned valid;          /* 0: a byte is not a digit of the base; the output is unspecified */
    uint64_t invalid_index;  /* position of the first such byte */
    unsigned exact_fallback; /* 1: the quotient sat on a truncation boundary and was settled by multiplying back */
} sbn3_parse_result;
typedef struct sbn3_parse_binding sbn3_parse_binding;
sbn3_query_result sbn3_parse_query(const sbn3_parse_spec *, const sbn3_radix_options *, sbn3_parse_plan *,
                                   sbn3_parse_info *);
void sbn3_parse_bind(const sbn3_parse_plan *, sbn3_arena *, size_t offset, sbn3_team *, sbn3_parse_binding **);
/* digits: integer_digits + fraction_digits bytes, most significant first, no point. out: info.limbs limbs,
 * 64-byte aligned, disjoint from the storage and from the digits. All info.limbs limbs are written. */
void sbn3_parse_execute(sbn3_parse_binding *, const unsigned char *digits, sbn3_limbs out, sbn3_parse_result *);
void sbn3_parse_unbind(sbn3_parse_binding *);
#ifdef __cplusplus
}
#endif
#endif
