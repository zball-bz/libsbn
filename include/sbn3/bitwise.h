#ifndef SBN3_BITWISE_H
#define SBN3_BITWISE_H
#include "sbn3/value.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Signed sign/magnitude inputs, infinite two's-complement BIT semantics.
 * Leading zeros and negative zero are accepted; outputs are normalized.
 * Exact output/input alias is allowed; partial overlap is not. No allocation,
 * scratch, arena, plan or team. Inputs have <=SIZE_MAX/64 limbs.
 * max(a.size,b.size)+1 output limbs always suffice for binary operations;
 * max(a.size+1,1) for complement. Only the operation's finite support is
 * written; words beyond its normalized result may have unspecified contents. */
typedef enum sbn3_bit_operation {SBN3_BIT_AND=0,SBN3_BIT_OR=1,SBN3_BIT_XOR=2} sbn3_bit_operation;
void sbn3_int_bitwise(sbn3_int *,sbn3_bit_operation,sbn3_int_view,sbn3_int_view);
void sbn3_int_and(sbn3_int *,sbn3_int_view,sbn3_int_view);
void sbn3_int_or(sbn3_int *,sbn3_int_view,sbn3_int_view);
void sbn3_int_xor(sbn3_int *,sbn3_int_view,sbn3_int_view);
void sbn3_int_complement(sbn3_int *,sbn3_int_view);
/* Arithmetic/floor right shift, distinct from value.h's truncating rshift.
 * Capacity a.size always suffices (1 for a nonzero negative value).
 * A discarded whole-word prefix can round an all-one high part up by a
 * limb, so a.size-bits/64 alone is not always sufficient.
 * A zero result permits zero capacity. Negative nonzero values shifted past
 * all their bits produce -1. Exact alias allowed. */
void sbn3_int_arshift(sbn3_int *,sbn3_int_view,size_t bits);
/* Magnitude bit length and population, including for negative values. */
size_t sbn3_int_bit_length(sbn3_int_view);
size_t sbn3_int_popcount_magnitude(sbn3_int_view);
int sbn3_int_test_bit(sbn3_int_view,size_t bit);
/* Infinite two's-complement scans in the representable index domain
 * [first,SIZE_MAX). SIZE_MAX means not found. which is 0 or 1. */
size_t sbn3_int_scan_bit(sbn3_int_view,size_t first,unsigned which);
/* Population/difference in [first, first+count), with the mathematical end
 * allowed to exceed SIZE_MAX. No work proportional to sign extension. */
size_t sbn3_int_popcount_range(sbn3_int_view,size_t first,size_t count);
size_t sbn3_int_hamdist_range(sbn3_int_view,sbn3_int_view,size_t first,size_t count);
typedef enum sbn3_bit_edit {SBN3_BIT_CLEAR=0,SBN3_BIT_SET=1,SBN3_BIT_FLIP=2} sbn3_bit_edit;
/* Writes one edited signed value, with infinite bit semantics. Universal
 * capacity max(a.size,bit/64+1)+1; unchanged operations need only a.size.
 * The required finite support must fit the limb/index domain above.
 * Exact alias allowed; no temporary two's-complement representation. */
void sbn3_int_edit_bit(sbn3_int *,sbn3_int_view,size_t bit,sbn3_bit_edit);
#ifdef __cplusplus
}
#endif
#endif
