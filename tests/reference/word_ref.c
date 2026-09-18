#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Independent unsigned schoolbook oracle, deliberately no SIMD or donor calls. */
void ref_mul(uint64_t *r, const uint64_t *a, size_t an, const uint64_t *b, size_t bn) {
    memset(r, 0, (an + bn) * sizeof *r);
    for (size_t i = 0; i < an; ++i) {
        unsigned __int128 cy = 0;
        for (size_t j = 0; j < bn; ++j) {
            unsigned __int128 v = (unsigned __int128)a[i] * b[j] + r[i+j] + cy;
            r[i+j] = (uint64_t)v; cy = v >> 64;
        }
        if (bn) r[i+bn] = (uint64_t)cy;
    }
}
