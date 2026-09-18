#define CR_NP 10
#define SBN3_P48_NS p48_capacity_test
#include "backend/ntt_p48/geom.hpp"
#include <assert.h>
#include "../oracle/oracle.h"
#include <initializer_list>
#include <stdio.h>

namespace p = sbn::v3::p48_capacity_test;
namespace cb = sbn::v3::coefficient_bound;

int main() {
    ref_int product, digit, square, limit, value;
    ref_inits(product, digit, square, limit, value, nullptr);
    unsigned cases = 0;
    for (unsigned np = 4; np <= 10; ++np) {
        ref_set_ui(product, 1);
        for (unsigned k = 0; k < np; ++k) ref_mul_ui(product, product, p::PR[k]);
        const auto exact = cb::prime_product(p::PR, np);
        ref_import(value, cb::Wide::words, -1, 8, 0, 0, exact.limb);
        assert(ref_cmp(value, product) == 0);
        const int step = np == 4 ? 4 : 8;
        for (int T = np == 4 ? 80 : 24 * int(np) - 32;
             T <= (np == 4 ? 88 : 24 * int(np) - 8); T += step) {
            ref_set_ui(digit, 1); ref_mul_2exp(digit, digit, T); ref_sub_ui(digit, digit, 1);
            ref_mul(square, digit, digit);
            ref_sub_ui(limit, product, 1); ref_fdiv_q(limit, limit, square);
            assert(ref_fits_ulong_p(limit));
            const uint64_t edge = ref_get_ui(limit);
            for (uint64_t terms : {uint64_t(0), uint64_t(1), edge - 1, edge, edge + 1, UINT64_MAX}) {
                ref_mul_ui(value, square, terms);
                const bool expected = ref_cmp(value, product) < 0;
                assert(cb::fits(exact, T, terms) == expected);
                if (np == 10) assert(bool(p::plan_T_ok(T, terms, terms)) == expected);
                ++cases;
            }
        }
    }
    // Equality is rejected, including small T where the subtractions overlap.
    for (unsigned bits = 0; bits <= 256; ++bits) {
        ref_set_ui(digit, 1); ref_mul_2exp(digit, digit, bits); ref_sub_ui(digit, digit, 1);
        ref_mul(square, digit, digit); ref_mul_ui(square, square, UINT64_MAX);
        cb::Wide bound{};size_t count = 0;ref_export(bound.limb, &count, -1, 8, 0, 0, square);
        assert(!cb::fits(bound, bits, UINT64_MAX));
        ref_add_ui(square, square, 1);bound={};ref_export(bound.limb, &count, -1, 8, 0, 0, square);
        assert(cb::fits(bound, bits, UINT64_MAX));
        cases += 2;
    }
    ref_clears(product, digit, square, limit, value, nullptr);
    printf("exact CRT capacity: %u reference/modular oracle boundary checks, NP4..10, all production T PASS\n", cases);
}
