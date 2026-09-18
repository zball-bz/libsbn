#ifndef SBN3_VERIFY_H
#define SBN3_VERIFY_H
#include "sbn3/mul.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sbn3_bbp_result {
    uint64_t bit_offset;        /* Skip this many fractional bits; extract the next 128. */
    uint64_t bits[2];           /* Most significant word first. */
    uint64_t terms, error_ulps; /* Absolute error in units of 2^-window_bits. */
    unsigned window_bits;       /* Accumulator precision: 192 or 384; result is still 128 bits. */
    unsigned stable;            /* Both endpoints of the error interval have these 128 bits. */
} sbn3_bbp_result;
/* Independent Bellard BBP digit extraction. No Chudnovsky, large products,
 * reference library, or allocation. Existing team, bounded per-worker stack.
 * FE_TONEAREST required. Supports offset <= 2^48-3 and windows 192/384.
 * This checks a window, not integrity of every stored bit of a large value. */
int sbn3_pi_bbp_supported(uint64_t bit_offset, unsigned window_bits);
void sbn3_pi_bbp(sbn3_team *, uint64_t bit_offset, unsigned window_bits, sbn3_bbp_result *);
/* Compare with the last 128 fractional bits of floor(pi*2^(64*n)).
 * value has n+1 little-endian limbs, n>=2. Returns stable && equal.
 * The integer limb must also equal 3. No scan of the intervening limbs. */
int sbn3_pi_bbp_check_tail(sbn3_team *, sbn3_const_limbs value, unsigned window_bits, sbn3_bbp_result *);
#ifdef __cplusplus
}
#endif
#endif
