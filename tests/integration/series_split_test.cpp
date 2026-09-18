#include "product_support.hpp"
#include "series/formulas.hpp"
#include <array>
#include <cmath>
using namespace sbn::v3::series;
static const Formula formula{FormulaKind::Chudnovsky};
static long double qbits(uint64_t a, uint64_t b) {
    a = std::max(uint64_t(1), a);
    if (a == b)
        return 0;
    if (b - a <= 4096) {
        long double sum = 0;
        for (uint64_t k = a; k < b; ++k)
            sum += std::log2((long double)10939058860032000ULL) + 3 * std::log2((long double)k);
        return sum;
    }
    return ((b - a) * std::log((long double)10939058860032000ULL) +
            3 * (std::lgamma((long double)b) - std::lgamma((long double)a))) / std::log(2.L);
}
static void check(sbn3_series_range root, sbn3_series_range current, unsigned depth) {
    if (current.end - current.begin <= 8)
        return; // Execution uses the independently bounded complete leaf batch.
    uint64_t terms = 0;
    sbn3_series_shape shape{};
    assert(formula.serial_envelope(root, depth, 7, terms, shape) == SBN3_SUPPORTED);
    assert(current.end - current.begin <= terms);
    const long double q = qbits(current.begin, current.end);
    assert(q < 64.L * shape.limbs[1]);
    sbn3_series_shape normalized{};uint64_t normalized_terms=0;
    assert(formula.serial_envelope(root,depth,7,normalized_terms,normalized,true)==SBN3_SUPPORTED);
    assert(normalized_terms==terms && normalized.limbs[0]==shape.limbs[0] && normalized.limbs[2]==shape.limbs[2]);
    const uint64_t first=std::max(uint64_t(1),current.begin),last=current.end;
    const uint64_t zeros=15*(last-first)+3*((last-1-__builtin_popcountll(last-1))-(first-1-__builtin_popcountll(first-1)));
    assert(q-64.L*(zeros/64)<64.L*normalized.limbs[1]);
    assert(normalized.limbs[1]<=shape.limbs[1]);
    const long double t = q + std::log2((long double)(current.end - current.begin)) +
                         std::log2(13591409.L + 545140134.L * (current.end - 1));
    assert(t < 64.L * shape.limbs[0]);
    if (current.end - current.begin <= 4096) {
        long double u = 0;
        for (uint64_t k = std::max(uint64_t(1), current.begin); k < current.end; ++k)
            u += std::log2(6.L * k - 5) + std::log2(2.L * k - 1) + std::log2(6.L * k - 1);
        assert(u < 64.L * shape.limbs[2]);
    }
}
static void tree(sbn3_series_range root, sbn3_series_range current, unsigned depth) {
    check(root, current, depth);
    if (current.end - current.begin <= 8)
        return;
    const auto m = formula.split_point(current);
    assert(current.begin < m && m < current.end);
    tree(root, {current.begin, m}, depth + 1);
    tree(root, {m, current.end}, depth + 1);
}
int main() {
    // All shorter intervals must fit a normalized envelope too; factors near
    // powers of two exercise large and discontinuous exact valuations.
    for(uint64_t a:{uint64_t(0),uint64_t(1),uint64_t(65520),(uint64_t(1)<<48)-1024})
        for(uint64_t maximum:{1u,3u,5u,17u,64u,257u}){
            sbn3_series_shape shape{};
            assert(formula.bounds({a,a+2*maximum},maximum,7,shape,true)==SBN3_SUPPORTED);
            for(uint64_t count=1;count<=maximum;++count)for(uint64_t start:{a,a+2*maximum-count}){
                const uint64_t first=std::max(uint64_t(1),start),last=start+count;
                const uint64_t zeros=15*(last-first)+3*((last-1-__builtin_popcountll(last-1))-(first-1-__builtin_popcountll(first-1)));
                assert(qbits(start,last)-64.L*(zeros/64)<64.L*shape.limbs[1]);
            }
        }

    for (uint64_t a : {uint64_t(0), uint64_t(1), uint64_t(65530),
                       (uint64_t(1) << 48) - 4096})
        for (uint64_t n : {0u, 1u, 2u, 7u, 31u, 257u, 4096u}) {
            const auto interval = factorial_log_bounds(a, a + n);
            long double exact = 0;
            for (uint64_t k = a + 1; k <= a + n; ++k)
                exact += std::log2((long double)k);
            assert(interval.lower <= exact && exact <= interval.upper);
        }
    // Native YC callback results, after converting (a,b] to [a+1,b+1).
    for (auto c : {std::array<uint64_t, 3>{1, 1001, 520},
                   {1, 70527, 36336}, {63043, 66789, 64917}, {64184, 66301, 65243},
                   {1, 352568360, 180186924}})
        assert(formula.split_point({c[0], c[1]}) == c[2]);
    for (uint64_t a : {uint64_t(0), uint64_t(1), uint64_t(13), uint64_t(63000), uint64_t(65500),
                       (uint64_t(1) << 28) - 1000, (uint64_t(1) << 48) - 4097})
        for (uint64_t n : {9u, 15u, 31u, 40u, 64u, 127u, 257u, 1024u, 4096u})
            tree({a, a + n}, {a, a + n}, 0);
    // Sample deep paths at real 5B and maximum-index scales. No giant values.
    for (sbn3_series_range root : {sbn3_series_range{0, 353396614}, {94615212, 161702851},
                                   {0, uint64_t(1) << 48},
                                   {(uint64_t(1) << 48) - 10000000, uint64_t(1) << 48}})
        for (unsigned trial = 0; trial < 64; ++trial) {
            auto r = root;
            for (unsigned depth = 0; r.end - r.begin > 8; ++depth) {
                check(root, r, depth);
                assert(depth < 96);
                const auto m = formula.split_point(r);
                const bool right = trial == 0 ? false : trial == 1 ? true : bool(random_word() & 1);
                r = right ? sbn3_series_range{m, r.end} : sbn3_series_range{r.begin, m};
            }
        }
    puts("smooth split native cuts / exhaustive small envelopes / deep 5B and 2^48 paths / Q,T,U growth PASS");
}
