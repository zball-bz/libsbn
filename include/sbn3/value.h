#ifndef SBN3_VALUE_H
#define SBN3_VALUE_H
#include "sbn3/base.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Caller-owned signed magnitudes. Leading zero limbs and negative zero are
 * accepted as inputs; outputs are normalized. No scratch or allocation.
 * Exact output/input alias is allowed; partial overlap is not. Normal builds
 * trust span/capacity preconditions; checked/sanitizer builds diagnose them. */
void sbn3_int_normalize(sbn3_int *);
int sbn3_int_compare(sbn3_int_view,sbn3_int_view);
/* Capacity >= max(input sizes)+1, or zero when both inputs are empty. */
void sbn3_int_add(sbn3_int *,sbn3_int_view,sbn3_int_view);
void sbn3_int_sub(sbn3_int *,sbn3_int_view,sbn3_int_view);
/* Left capacity >= input size + bits/64 + (bits%64 != 0), unless input is
 * zero. Right capacity >= max(input size-bits/64,0). Right shift truncates
 * toward zero for negative values. Bits may be any size_t value. */
void sbn3_int_lshift(sbn3_int *,sbn3_int_view,size_t bits);
void sbn3_int_rshift(sbn3_int *,sbn3_int_view,size_t bits);
#ifdef __cplusplus
}
#endif
#endif
