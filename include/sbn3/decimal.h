#ifndef SBN3_DECIMAL_H
#define SBN3_DECIMAL_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Fixed-width, unsigned decimal primitives. No context, plan, allocation or
 * scratch. Sign, whitespace, separators and a terminating NUL are not part of
 * this interface. Pointers must designate valid, disjoint caller-owned spans.
 * No SIMD alignment or readable padding is required.
 *
 * Formatting requires room for the result (20 / 39 chars always suffice).
 * It writes only those chars, including a single '0' for zero, and returns
 * the end pointer.
 * Parsing consumes the entire span, allows leading zeros, and writes the
 * output only on success. Invalid text takes precedence over overflow. */
/* value = lo + hi * 2^64; scalar C ABI, independent of SIMD representation. */
typedef struct sbn3_uint128 { uint64_t lo, hi; } sbn3_uint128;
typedef enum sbn3_decimal_status {
    SBN3_DECIMAL_OK = 0,
    SBN3_DECIMAL_INVALID = 1,
    SBN3_DECIMAL_OVERFLOW = 2
} sbn3_decimal_status;

char *sbn3_decimal_write_u64(char *out, uint64_t value);
char *sbn3_decimal_write_u128(char *out, sbn3_uint128 value);
sbn3_decimal_status sbn3_decimal_read_u64(uint64_t *out, const char *digits, size_t count);
sbn3_decimal_status sbn3_decimal_read_u128(sbn3_uint128 *out, const char *digits, size_t count);

/* Bounded multiword leaves: 1..8 little-endian u64 limbs. All output limbs
 * are written on successful parsing; an error leaves all of them unchanged.
 * Leading zero limbs are allowed. 20*limbs bytes suffice for integer writing.
 * These leaf entries use at most 192 bytes of stack value storage; larger
 * values use radix.h, whose dispatcher also reuses these algorithms. */
#define SBN3_DECIMAL_SMALL_LIMBS 8
char *sbn3_decimal_write_small(char *out, const uint64_t *value, size_t limbs);
sbn3_decimal_status sbn3_decimal_read_small(uint64_t *out, size_t limbs,
                                           const char *digits, size_t count);

/* A pure fraction value / 2^(64*limbs), in [0,1). No "0." is emitted or
 * accepted. Writing emits exactly count digits, truncated toward zero.
 * Reading returns floor(0.digits * 2^(64*limbs)); no rounding to nearest.
 * Empty fractional spans denote zero and are valid. The integer and fractional
 * leaves can be combined; radix.h supports arbitrary binary exponents. */
char *sbn3_decimal_write_fraction(char *out, const uint64_t *value,
                                 size_t limbs, size_t count);
sbn3_decimal_status sbn3_decimal_read_fraction(uint64_t *out, size_t limbs,
                                              const char *digits, size_t count);

#ifdef __cplusplus
}
#endif
#endif
