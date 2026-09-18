/* Imported from libsbn/include/sbn/detail/u52/mparam.h; arithmetic preserved, private namespace and prepared Frame adapter. */
#pragma once
namespace sbn::v3::u52 {
/* Native v3 thresholds in u52 digits of the smaller operand.
 * 2026-09-07: 112/704 versus donor 96/618 and 112/618; same-process,
 * same-buffer, rotating-order probe, 212 fine-grid shapes at ratios
 * 1, 3/2, 2, 8. The combined change modestly improves all ratio means.
 * Evidence: results/small_alg_2026-09-07/u52-thresholds-interleaved/.
 * Forced build definitions remain available for bounded experiments. */
#ifndef MUL_U52_T22_THRESHOLD
#define MUL_U52_T22_THRESHOLD 112
#endif
#ifndef MUL_U52_T33_THRESHOLD
#define MUL_U52_T33_THRESHOLD 704
#endif
#ifndef MUL_U52_T32_OK_THRESHOLD
#define MUL_U52_T32_OK_THRESHOLD 108
#endif
#ifndef MUL_U52_T42_OK_THRESHOLD
#define MUL_U52_T42_OK_THRESHOLD 132
#endif

} // namespace
