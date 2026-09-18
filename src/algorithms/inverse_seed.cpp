/* Seed framing ported from libsbn/include/sbn/invert.h:sbn_inv_seed.
 * The BMI2/ADX reciprocal body is the unchanged v2 donor with private symbols. */
#include "algorithms/inverse_seed.hpp"
#include "common/checked.hpp"
extern "C" void sbn3i_recip_u1024(const uint64_t *, uint64_t *);
namespace sbn::v3 {
void inverse_seed(uint64_t *U, const uint64_t *D, size_t n) noexcept {
    require(n >= 1 && n <= 15 && U && D && (D[n - 1] >> 63), SBN3_FATAL_ARGUMENT, "inverse seed input");
    uint64_t padded[16]{}, reciprocal[16];
    for (size_t j = 0; j < n; ++j)
        padded[16 - n + j] = D[j];
    bool power = padded[15] == (uint64_t(1) << 63);
    for (size_t j = 0; j < 15 && power; ++j)
        power = padded[j] == 0;
    if (power)
        for (auto &v : reciprocal)
            v = UINT64_MAX;
    else
        sbn3i_recip_u1024(padded, reciprocal);
    const size_t shift = 1023 - 64 * n, word = shift / 64;
    const unsigned bits = shift % 64;
    for (size_t j = 0; j < n; ++j)
        U[j] = (reciprocal[word + j] >> bits) | (reciprocal[word + j + 1] << (64 - bits));
    U[n] = 1;
}
} // namespace sbn::v3
