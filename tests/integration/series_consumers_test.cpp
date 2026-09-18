// More formula consumers through the same service: alternating Hyperdescent
// (sin 1), polynomial numerators (zeta(3) Amdeberhan-Zeilberger), negative R
// (zeta(3) Apery), quartic factors with a nontrivial terminal (Ramanujan pi:
// ratio times sqrt 2). Two independent zeta(3) formulas must agree limb for
// limb; Ramanujan pi is checked against a Machin reference; sin 1 against the
// exact integer reference and long double.
#include "formula_reference.hpp"
#include "series/terminal.hpp"
#include "sbn3/newton.h"
#include <math.h>
using namespace sbn::v3;
using namespace sbn::v3::series;
using namespace formula_reference;
namespace {
FormulaDef sin_one() { // sum_{k>=0} (-1)^k/(2k+1)!: Q(k)=2k(2k+1) for k>=1, first term 1
    FormulaDef d{};
    d.recipe = SBN3_SERIES_HYPERDESCENT;
    d.begin = 0;
    d.explicit_first = true;
    d.first = {1, 1, 1, false};
    d.P.alternating = true;
    d.Q.count = 2;
    d.Q.factor[0] = {2, 0, 1};
    d.Q.factor[1] = {2, 1, 1};
    return d;
}
FormulaDef zeta3_az() { // 64 zeta(3) = sum (-1)^k (205k^2+250k+77) (k!)^10/((2k+1)!)^5
    FormulaDef d{};
    d.recipe = SBN3_SERIES_COMMON_P2B3;
    d.begin = 0;
    d.explicit_first = true;
    d.first = {77, 1, 1, false};
    d.P.alternating = true;
    d.P.count = 1;
    d.P.factor[0] = {1, 0, 5};
    d.P.degree = 2;
    d.P.coefficient[0] = 77;
    d.P.coefficient[1] = 250;
    d.P.coefficient[2] = 205;
    d.Q.constant_low = 32;
    d.Q.count = 1;
    d.Q.factor[0] = {2, 1, 5};
    d.R.count = 1;
    d.R.factor[0] = {1, 0, 5};
    return d;
}
FormulaDef zeta3_apery() { // (2/5) zeta(3) = sum_{k>=1} (-1)^(k+1)/(k^3 C(2k,k)); ratio -(k-1)^3/(2k^2(2k-1))
    FormulaDef d{};
    d.recipe = SBN3_SERIES_COMMON_P2B3;
    d.begin = 1;
    d.explicit_first = true;
    d.first = {1, 2, 1, false}; // term_1 = 1/2 with U = 1 and D = 2 scaling the rest
    d.P.negative = true;
    d.P.count = 1;
    d.P.factor[0] = {1, -1, 3};
    d.Q.constant_low = 2;
    d.Q.count = 2;
    d.Q.factor[0] = {1, 0, 2};
    d.Q.factor[1] = {2, -1, 1};
    d.R.negative = true;
    d.R.count = 1;
    d.R.factor[0] = {1, -1, 3};
    return d;
}
FormulaDef ramanujan() { // 9801/(2 sqrt2 pi) = sum (4k)!(1103+26390k)/((k!)^4 396^(4k))
    FormulaDef d{};
    d.recipe = SBN3_SERIES_COMMON_P2B3;
    d.begin = 0;
    d.explicit_first = true;
    d.first = {1103, 1, 1, false};
    d.P.count = 5;
    d.P.factor[0] = {4, -3, 1};
    d.P.factor[1] = {4, -2, 1};
    d.P.factor[2] = {4, -1, 1};
    d.P.factor[3] = {4, 0, 1};
    d.P.factor[4] = {26390, 1103, 1};
    d.Q.constant_low = 24591257856ULL; // 396^4
    d.Q.count = 1;
    d.Q.factor[0] = {1, 0, 4};
    d.R.count = 4;
    d.R.factor[0] = {4, -3, 1};
    d.R.factor[1] = {4, -2, 1};
    d.R.factor[2] = {4, -1, 1};
    d.R.factor[3] = {4, 0, 1};
    return d;
}
FormulaDef ramanujan_reduced() { // the same series with P/Q/R divided by 8k for k >= 1
    FormulaDef d{};
    d.recipe = SBN3_SERIES_COMMON_P2B3;
    d.begin = 0;
    d.explicit_first = true;
    d.first = {1103, 1, 1, false};
    d.P.count = 4;
    d.P.factor[0] = {4, -3, 1};
    d.P.factor[1] = {2, -1, 1};
    d.P.factor[2] = {4, -1, 1};
    d.P.factor[3] = {26390, 1103, 1};
    d.Q.constant_low = 3073907232ULL; // 396^4 / 8
    d.Q.count = 1;
    d.Q.factor[0] = {1, 0, 3};
    d.R.count = 3;
    d.R.factor[0] = {4, -3, 1};
    d.R.factor[1] = {2, -1, 1};
    d.R.factor[2] = {4, -1, 1};
    return d;
}
struct Sum {
    std::vector<uint64_t> t, d, u;
    bool t_negative = false;
    int64_t t_exponent = 0, u_exponent = 0;
    uint64_t terms = 0;
};
// Exact finite sum over the tail-certified term count for `fractional` limbs.
Sum evaluate(Fixture &fixture, const FormulaDef &def, size_t working, unsigned workers, bool check_small,
             const char *label = "?") {
    DataFormula probe{};
    const uint64_t domain = std::min<uint64_t>(uint64_t(1) << 40, formula_domain_limit(def));
    if (!data_formula_prepare(def, domain, probe) || !probe.contraction)
        fprintf(stderr, "%s rejected: %s contraction=%d attenuation_minimum=%llu table=%u\n", label,
                probe.rejection ? probe.rejection : "-", probe.contraction, (unsigned long long)probe.attenuation_minimum,
                probe.attenuation.count);
    assert(data_formula_prepare(def, domain, probe) && probe.contraction);
    assert(!probe.tail_rejection);
    const uint64_t terms = probe.infinite_tail_terms(64 * working + 8);
    assert(terms);
    static DataFormula data{};
    assert(data_formula_prepare(def, terms, data));
    const auto f = data_finite_formula(data);
    FinitePlan plan{};
    const sbn3_series_options options{workers, 8, 1, 0, 0};
    allocation_watch_start();
    assert(finite_query(f, {def.begin, terms}, 3, options, plan) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    auto meta = fixture.allocate(plan.info.plan_bytes, plan.info.plan_alignment),
         tables = fixture.allocate(plan.info.prepared_bytes, plan.info.prepared_alignment),
         scratch = fixture.allocate(plan.info.workspace_bytes, plan.info.workspace_alignment);
    sbn3_series_values v{};
    for (unsigned j = 0; j < 2; ++j) {
        auto l = fixture.allocate(plan.info.output.limbs[j] * 8, 64);
        v.value[j].mantissa = {static_cast<uint64_t *>(l.data), plan.info.output.limbs[j], 0, 0};
    }
    FiniteBinding bound{};
    allocation_watch_start();
    finite_prepare(plan, *fixture.arena, *fixture.team, meta, tables, scratch, bound);
    finite_execute(bound, v);
    assert(!allocation_watch_stop());
    Sum s{};
    s.terms = terms;
    s.t.assign(v.value[0].mantissa.data, v.value[0].mantissa.data + v.value[0].mantissa.size);
    s.d.assign(v.value[1].mantissa.data, v.value[1].mantissa.data + v.value[1].mantissa.size);
    s.t_negative = v.value[0].mantissa.negative;
    s.t_exponent = v.value[0].exponent2;
    assert(!v.value[1].mantissa.negative && v.value[1].mantissa.size);
    if (check_small) {
        // Exact reference of the same finite sum and two prime certificates.
        Ref r;
        reference(def, def.begin, terms, r);
        compare(r, 3, v, def.recipe);
        certificate(def, def.begin, terms, 3, v);
    }
    return s;
}
// floor(value * 2^(64 fractional)) with fractional+1 limbs; numerator scale and
// denominator binary exponent express the outer rational coefficient.
std::vector<uint64_t> ratio(Fixture &fixture, Sum &s, size_t fractional, unsigned workers, uint64_t numerator_scale,
                            int denominator_exponent, uint64_t extra_error) {
    const size_t guard = 2;
    RatioTerminalPlan plan{};
    assert(ratio_terminal_query(fractional, guard, workers, 1 + extra_error, plan) == SBN3_SUPPORTED);
    std::vector<uint64_t> t = s.t;
    t.push_back(0);
    if (numerator_scale != 1)
        t.back() = ref_mul_1(t.data(), t.data(), t.size() - 1, numerator_scale);
    while (!t.empty() && !t.back())
        t.pop_back();
    const sbn3_series_value numerator{{t.data(), t.size(), t.size(), s.t_negative}, s.t_exponent};
    const sbn3_series_value denominator{{s.d.data(), s.d.size(), s.d.size(), 0}, denominator_exponent};
    const size_t at = up(fixture.base + fixture.cursor, plan.storage_alignment) - fixture.base;
    fixture.cursor = up(at + plan.storage_bytes, 4096) + 4096;
    sbn3_error error{};
    assert(sbn3_arena_prepare(fixture.arena, at, plan.storage_bytes, &error) == SBN3_OK);
    auto out = fixture.allocate(plan.output_limbs * 8, 64);
    allocation_watch_start();
    ratio_terminal_execute(plan, numerator, denominator, fixture.arena, at, fixture.team,
                           {static_cast<uint64_t *>(out.data), plan.output_limbs});
    assert(!allocation_watch_stop());
    return std::vector<uint64_t>(static_cast<uint64_t *>(out.data), static_cast<uint64_t *>(out.data) + plan.output_limbs);
}
// Machin reference floor(pi * 2^(64 n)) (from pi_test).
void arctan(ref_number *result, unsigned q, size_t words) {
    ref_int power, term, divisor;
    ref_inits(power, term, divisor, nullptr);
    ref_set_ui(power, 1);
    ref_mul_2exp(power, power, 64 * words);
    ref_set_ui(divisor, q);
    ref_fdiv_q(power, power, divisor);
    ref_set_ui(result, 0);
    for (uint64_t k = 0; ref_sgn(power); ++k) {
        ref_set_ui(divisor, 2 * k + 1);
        ref_fdiv_q(term, power, divisor);
        if (k & 1)
            ref_sub(result, result, term);
        else
            ref_add(result, result, term);
        ref_set_ui(divisor, q * q);
        ref_fdiv_q(power, power, divisor);
    }
    ref_clears(power, term, divisor, nullptr);
}
void machin(ref_number *want, size_t n) {
    ref_int a, b;
    ref_inits(a, b, nullptr);
    arctan(a, 5, n + 3);
    arctan(b, 239, n + 3);
    ref_mul_ui(a, a, 16);
    ref_mul_ui(b, b, 4);
    ref_sub(want, a, b);
    ref_fdiv_q_2exp(want, want, 192);
    ref_clears(a, b, nullptr);
}
bool top_bits_match(const std::vector<uint64_t> &v, size_t fractional, long double expected, unsigned bits = 52) {
    const uint64_t integer = uint64_t(floorl(expected));
    const uint64_t fraction = uint64_t(ldexpl(expected - floorl(expected), 64));
    return v[fractional] == integer && ((v[fractional - 1] ^ fraction) >> (64 - bits)) == 0;
}
} // namespace
int main() {
    for (unsigned workers : {1u, 3u, 16u})
        for (size_t fractional : {size_t(3), size_t(40), size_t(300)}) {
            Fixture fixture(workers, false);
            const size_t working = fractional + 2;
            // sin(1): alternating Hyperdescent with a two-factor denominator; exact reference for small sums.
            auto s = evaluate(fixture, sin_one(), working, workers, fractional <= 40, "sin1");
            auto v = ratio(fixture, s, fractional, workers, 1, 0, 0);
            assert(top_bits_match(v, fractional, sinl(1.0L)));
            // zeta(3): polynomial numerator (AZ) versus negative-R Apery form, limb for limb.
            auto az = evaluate(fixture, zeta3_az(), working, workers, fractional <= 40, "zeta3-az");
            auto ap = evaluate(fixture, zeta3_apery(), working, workers, fractional <= 40, "zeta3-apery");
            auto z1 = ratio(fixture, az, fractional, workers, 1, 6, 0);  // T/(64 D)
            auto z2 = ratio(fixture, ap, fractional, workers, 5, 1, 0);  // 5T/(2D)
            assert(z1 == z2 && z1[fractional] == 1 && (z1[fractional - 1] >> 32) == 0x33ba004fULL);
            assert(top_bits_match(z1, fractional, 1.2020569031595942853997L));
            // Ramanujan pi: 9801 sqrt2 / 4 * D/T, ratio terminal then sqrt2 product, against Machin.
            // The reduced definition (P/Q/R over 8k) must give the same T/D and U/D and the same value.
            auto rm = evaluate(fixture, ramanujan(), working, workers, fractional <= 40, "ramanujan");
            {
                auto reduced = evaluate(fixture, ramanujan_reduced(), working, workers, fractional <= 40, "ramanujan reduced");
                assert(reduced.terms == rm.terms);
                Ref a, b;
                reference(ramanujan(), 0, rm.terms, a);
                reference(ramanujan_reduced(), 0, reduced.terms, b);
                ref_int x, y;
                ref_inits(x, y, nullptr);
                ref_mul(x, a.t, b.d);
                ref_mul(y, b.t, a.d);
                assert(!ref_cmp(x, y)); // T/D equal
                ref_mul(x, a.u, b.d);
                ref_mul(y, b.u, a.d);
                assert(!ref_cmp(x, y)); // U/D equal
                assert(ref_sizeinbase(b.d, 2) < ref_sizeinbase(a.d, 2) && ref_sizeinbase(b.u, 2) < ref_sizeinbase(a.u, 2));
                ref_clears(x, y, nullptr);
                Sum inv{};
                inv.t = reduced.d;
                inv.d = reduced.t;
                auto q_reduced = ratio(fixture, inv, fractional + 2, workers, 9801, 2, 0);
                Sum inv_full{};
                inv_full.t = rm.d;
                inv_full.d = rm.t;
                auto q_full = ratio(fixture, inv_full, fractional + 2, workers, 9801, 2, 0);
                assert(q_reduced == q_full);
                printf("ramanujan reduced: D bits %zu -> %zu, U bits %zu -> %zu, same ratios and terminal quotient\n",
                       ref_sizeinbase(a.d, 2), ref_sizeinbase(b.d, 2), ref_sizeinbase(a.u, 2), ref_sizeinbase(b.u, 2));
            }
            Sum inverted{};
            inverted.t = rm.d;
            inverted.d = rm.t;
            inverted.t_exponent = 0;
            assert(!rm.t_negative && rm.t_exponent == 0);
            const size_t n = fractional + 2;
            auto q1 = ratio(fixture, inverted, n, workers, 9801, 2, 0); // floor(9801 D/(4T) 2^(64 n)), n+1 limbs
            sbn3_newton_plan sq{};
            sbn3_newton_info info{};
            const sbn3_newton_options no{workers, 0, 0, 0};
            assert(sbn3_newton_query(SBN3_SQRT2_RSQRT, n, &no, &sq, &info) == SBN3_SUPPORTED);
            const size_t at = up(fixture.base + fixture.cursor, info.storage_alignment) - fixture.base;
            fixture.cursor = up(at + info.storage_bytes, 4096) + 4096;
            sbn3_error error{};
            assert(sbn3_arena_prepare(fixture.arena, at, info.storage_bytes, &error) == SBN3_OK);
            auto root = fixture.allocate((n + 1) * 8, 64);
            sbn3_newton_binding *binding = nullptr;
            sbn3_newton_bind(&sq, fixture.arena, at, fixture.team, &binding);
            sbn3_newton_execute(binding, nullptr, {static_cast<uint64_t *>(root.data), n + 1});
            sbn3_newton_unbind(binding);
            // product of the two n+1-limb approximations: pi 2^(128 n) with error below 16 units of 2^(64 n)
            std::vector<uint64_t> product(2 * (n + 1));
            sbn3_mul_basecase(product.data(), product.size(), q1.data(), n + 1, static_cast<uint64_t *>(root.data), n + 1);
            std::vector<uint64_t> pi_limbs(product.begin() + n, product.begin() + 2 * n + 2);
            // guard: the two words dropped below the requested fractional limbs must be separated from a carry boundary
            assert(!(pi_limbs[1] == 0 && pi_limbs[0] < 32) && !(pi_limbs[1] == UINT64_MAX && pi_limbs[0] > UINT64_MAX - 32));
            std::vector<uint64_t> pi_value(pi_limbs.begin() + 2, pi_limbs.begin() + 2 + fractional + 1);
            ref_int want, got;
            ref_inits(want, got, nullptr);
            machin(want, fractional);
            ref_import(got, pi_value.size(), -1, 8, 0, 0, pi_value.data());
            assert(!ref_cmp(want, got));
            ref_clears(want, got, nullptr);
            printf("consumers W%u fractional=%zu sin1 terms=%llu zeta3 AZ/Apery terms=%llu/%llu agree, Ramanujan pi terms=%llu == Machin, "
                   "zeta3=1.%016llx PASS\n",
                   workers, fractional, (unsigned long long)s.terms, (unsigned long long)az.terms, (unsigned long long)ap.terms,
                   (unsigned long long)rm.terms, (unsigned long long)z1[fractional - 1]);
            fflush(stdout);
        }
    puts("series consumers: alternating Hyperdescent, polynomial numerator, negative R, quartic factors, "
         "ratio and sqrt2 terminals PASS");
}
