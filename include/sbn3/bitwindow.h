#ifndef SBN3_BITWINDOW_H
#define SBN3_BITWINDOW_H
#include "sbn3/verify.h"
#ifdef __cplusplus
extern "C" {
#endif

/* A sum of streams, indexed from k=0:
 *   numerator * (-1)^(alternating*k) * 2^(shift-stride*k)
 *   / (denominator_scale * (a*k+b)^power).
 * This factored subset covers Bellard, Huvent and higher denominator powers.
 * No general expression interpreter or prefix-to-window terminal conversion.
 * a>=0, b,scale,stride>0; 1<=power<=8; alternating=0 or 1.
 * Coefficients/streams are borrowed and immutable throughout execution. */
typedef struct sbn3_bbp_stream {
    int64_t numerator;
    uint64_t a, b, denominator_scale;
    int32_t shift;
    uint32_t stride, power, alternating;
} sbn3_bbp_stream;

typedef struct sbn3_bbp_window_info {
    uint64_t terms, error_ulps;
    unsigned max_modulus_bits, max_native_digits;
} sbn3_bbp_window_info;

/* Pure query: 1..32 streams, window 192/384, offset<=2^62, nonzero numerator.
 * All included a*k+b must fit u64 and denominators must fit 256 bits.
 * Checks the finite range needed by a conservative geometric tail bound.
 * Unsupported requests return 0 and do not modify info. */
int sbn3_bbp_window_supported(const sbn3_bbp_stream *, size_t stream_count, uint64_t bit_offset,
                              unsigned window_bits, sbn3_bbp_window_info *info);
/* Existing Team; no allocation, bounded per-worker stack, FE_TONEAREST.
 * Native CIOS u52 instances selected outside the exponent loop.
 * Returns a 128-bit fractional window and its stability/error bound.
 * sbn3_pi_bbp uses the same IFMA word-stream core with its Bellard tail path. */
void sbn3_bbp_window(sbn3_team *, const sbn3_bbp_stream *, size_t stream_count, uint64_t bit_offset,
                     unsigned window_bits, sbn3_bbp_result *);

/* Huvent's rearranged nine-stream formula for Catalan's constant. */
int sbn3_catalan_bbp_supported(uint64_t bit_offset, unsigned window_bits);
void sbn3_catalan_bbp(sbn3_team *, uint64_t bit_offset, unsigned window_bits, sbn3_bbp_result *);
#ifdef __cplusplus
}
#endif
#endif
