#ifndef SBN3_TEST_ORACLE_H
#define SBN3_TEST_ORACLE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Test-only signed integer shell. No production headers or allocator contracts
 * are shared with the reference arithmetic. Supported import/export: LE,
 * least-significant word first, 1- or 8-byte words, no nails. */
typedef struct ref_number {
    void *value;
} ref_number;
typedef ref_number ref_int[1];
void ref_init(ref_number *);
void ref_clear(ref_number *);
void ref_inits(ref_number *, ...);
void ref_clears(ref_number *, ...);
void ref_import(ref_number *, size_t, int, size_t, int, size_t, const void *);
void ref_export(void *, size_t *, int, size_t, int, size_t, const ref_number *);
void ref_set(ref_number *, const ref_number *);
void ref_set_ui(ref_number *, uint64_t);
uint64_t ref_get_ui(const ref_number *);
int ref_cmp(const ref_number *, const ref_number *);
int ref_cmp_ui(const ref_number *, uint64_t);
int ref_sgn(const ref_number *);
int ref_fits_ulong_p(const ref_number *);
size_t ref_sizeinbase(const ref_number *, unsigned);
void ref_neg(ref_number *, const ref_number *);
void ref_abs(ref_number *, const ref_number *);
void ref_add(ref_number *, const ref_number *, const ref_number *);
void ref_sub(ref_number *, const ref_number *, const ref_number *);
void ref_mul(ref_number *, const ref_number *, const ref_number *);
void ref_addmul(ref_number *, const ref_number *, const ref_number *);
void ref_add_ui(ref_number *, const ref_number *, uint64_t);
void ref_sub_ui(ref_number *, const ref_number *, uint64_t);
void ref_mul_ui(ref_number *, const ref_number *, uint64_t);
void ref_mul_2exp(ref_number *, const ref_number *, size_t);
void ref_fdiv_q_2exp(ref_number *, const ref_number *, size_t);
void ref_tdiv_q_2exp(ref_number *, const ref_number *, size_t);
void ref_fdiv_r_2exp(ref_number *, const ref_number *, size_t);
void ref_fdiv_q(ref_number *, const ref_number *, const ref_number *);
void ref_mod(ref_number *, const ref_number *, const ref_number *);
uint64_t ref_fdiv_ui(const ref_number *, uint64_t);
void ref_divexact_ui(ref_number *, const ref_number *, uint64_t);
void ref_sqrt(ref_number *, const ref_number *);
uint64_t ref_add_n(uint64_t *, const uint64_t *, const uint64_t *, size_t);
uint64_t ref_sub_n(uint64_t *, const uint64_t *, const uint64_t *, size_t);
uint64_t ref_mul_1(uint64_t *, const uint64_t *, size_t, uint64_t);
/* Exact small reference / two fresh 61-bit prime checks for large products.
 * False-acceptance bounds and the certified-witness protocol are documented. */
int ref_product_equal(const uint64_t *, size_t, const uint64_t *, size_t, const uint64_t *, size_t);
int ref_prime61(uint64_t);
uint64_t ref_mod_words(const uint64_t *, size_t, uint64_t);
#ifdef __cplusplus
}
#endif
#endif
