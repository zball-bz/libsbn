#include "algorithms/sqrt2_finish.hpp"
#include "common/checked.hpp"
namespace sbn::v3 {
void sqrt2_from_rsqrt(uint64_t *z, size_t n, const uint64_t *r, size_t precision) noexcept {
    require(n >= 1 && precision >= n + 2 && !r[precision], SBN3_FATAL_ARGUMENT, "sqrt2 rsqrt precision");
    const size_t drop = precision - n;
    for (size_t j = 0; j <= n; ++j)
        z[j] = (r[drop + j] << 1) | (r[drop + j - 1] >> 63);
}
namespace {
int compare(const uint64_t *s, size_t n) {
    for (size_t j = 2 * n + 2; j-- > 0;) {
        const uint64_t expected = j == 2 * n ? 2 : 0;
        if (s[j] != expected)
            return s[j] > expected ? 1 : -1;
    }
    return 0;
}
// s +=/-= 2z+1, with separate doubling and add/sub carries.
void change(uint64_t *s, const uint64_t *z, size_t n, bool add) {
    uint64_t doubling = 1, carry = 0;
    for (size_t j = 0; j < 2 * n + 2; ++j) {
        const __uint128_t d = __uint128_t(j <= n ? z[j] : 0) * 2 + doubling;
        doubling = uint64_t(d >> 64);
        const __uint128_t amount = uint64_t(d) + __uint128_t(carry);
        if (add) {
            const __uint128_t sum = __uint128_t(s[j]) + amount;
            s[j] = uint64_t(sum);
            carry = uint64_t(sum >> 64);
        } else {
            const uint64_t old = s[j];
            s[j] = old - uint64_t(amount);
            carry = __uint128_t(old) < amount;
        }
    }
    require(!carry && !doubling, SBN3_FATAL_MATH, "sqrt2 square correction overflow");
}
void unit(uint64_t *z, size_t n, bool add) {
    for (size_t j = 0; j <= n; ++j) {
        if (add) {
            if (++z[j])
                return;
        } else if (z[j]--)
            return;
    }
    fatal(SBN3_FATAL_MATH, "sqrt2 candidate overflow");
}
} // namespace
unsigned sqrt2_certify(sbn3_mul_binding *square, uint64_t *z, size_t n, uint64_t *work) noexcept {
    require(n >= 1 && n <= (size_t(1) << 31) && z[n] == 1, SBN3_FATAL_ARGUMENT, "sqrt2 candidate shape");
    sbn3_product_inputs input{};
    input.a = {z, n + 1};
    sbn3_product_execute(square, &input, {work, 2 * n + 2});
    unsigned corrections = 0;
    while (compare(work, n) > 0) {
        require(corrections < 2, SBN3_FATAL_MATH, "sqrt2 candidate error bound");
        unit(z, n, false);
        change(work, z, n, false);
        ++corrections;
    }
    for (;;) {
        change(work, z, n, true);
        if (compare(work, n) > 0) {
            change(work, z, n, false);
            break;
        }
        require(corrections < 2, SBN3_FATAL_MATH, "sqrt2 candidate error bound");
        unit(z, n, true);
        ++corrections;
    }
    return corrections;
}
} // namespace sbn::v3
