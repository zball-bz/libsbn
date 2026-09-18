#include "product_support.hpp"
#include "series/psr.hpp"
#include "series/scaled_ratio.hpp"
#include "series/limited.hpp"
using namespace sbn::v3;
using namespace sbn::v3::series;
// Independent integer oracle for signed window addition, including cancellation,
// low zero runs, unequal exponents, and either input used as the fixed output.
static void window_addition() {
    ref_int a, b, expected, got;
    ref_inits(a, b, expected, got, nullptr);
    for (unsigned trial = 0; trial < 2000; ++trial) {
        uint64_t original_x[128]{}, original_y[128]{};
        size_t xn = 1 + random_word() % 32, yn = 1 + random_word() % 32;
        const size_t words = 1 + random_word() % 24;
        if (!(trial & 1)) { xn = std::min(xn, words + 1); yn = std::min(yn, words + 1); }
        int64_t xe = (int64_t(random_word() % 17) - 8) * 64;
        int64_t ye = (int64_t(random_word() % 17) - 8) * 64;
        unsigned xs = random_word() & 1, ys = random_word() & 1;
        for (size_t i = 0; i < xn; ++i) original_x[i] = random_word();
        for (size_t i = 0; i < yn; ++i) original_y[i] = random_word();
        for (size_t i = 0; i < trial % xn; ++i) original_x[i] = 0;
        if (!(trial % 13)) { yn = xn; ye = xe; ys = !xs; memcpy(original_y, original_x, sizeof original_x); }
        const size_t capacity = std::max({xn, yn, words}) + 3;
        const int64_t high = std::max(xe + int64_t(xn * 64), ye + int64_t(yn * 64));
        const int64_t low = std::max(std::min(xe, ye), high - int64_t((words + 1) * 64));
        ref_import(a, xn, -1, 8, 0, 0, original_x);
        ref_import(b, yn, -1, 8, 0, 0, original_y);
        if (xs) ref_neg(a, a);
        if (ys) ref_neg(b, b);
        if (xe >= low) ref_mul_2exp(a, a, size_t(xe - low));
        else ref_tdiv_q_2exp(a, a, size_t(low - xe));
        if (ye >= low) ref_mul_2exp(b, b, size_t(ye - low));
        else ref_tdiv_q_2exp(b, b, size_t(low - ye));
        ref_add(expected, a, b);
        const size_t used = (ref_sizeinbase(expected, 2) + 63) / 64;
        const size_t drop = used > words ? used - words : 0;
        ref_tdiv_q_2exp(expected, expected, drop * 64);
        const int64_t exponent = low + int64_t(drop * 64);
        for (unsigned mode = 0; mode < 6; ++mode) {
            const unsigned alias = mode % 3;
            uint64_t xd[128], yd[128], od[128]{};
            memcpy(xd, original_x, sizeof xd); memcpy(yd, original_y, sizeof yd);
            xd[capacity] = yd[capacity] = od[capacity] = UINT64_MAX;
            sbn3_series_value x{{xd, capacity, xn, xs}, xe}, y{{yd, capacity, yn, ys}, ye}, out{{od, capacity, 0, 0}, 0};
            auto &r = alias == 1 ? x : alias == 2 ? y : out;
            const auto *base = r.mantissa.data;
            allocation_watch_start();
            if (mode < 3) add_limited(r, x, y, words);
            else add_limited<true>(r, x, y, words);
            assert(!allocation_watch_stop());
            assert(r.mantissa.data == base && r.mantissa.capacity == capacity && r.mantissa.size <= words);
            assert(xd[capacity] == UINT64_MAX && yd[capacity] == UINT64_MAX && od[capacity] == UINT64_MAX);
            ref_import(got, r.mantissa.size, -1, 8, 0, 0, r.mantissa.data);
            if (r.mantissa.negative) ref_neg(got, got);
            ref_set(a, expected);
            if (r.exponent2 >= exponent) ref_mul_2exp(got, got, size_t(r.exponent2 - exponent));
            else ref_mul_2exp(a, a, size_t(exponent - r.exponent2));
            assert(ref_cmp(got, a) == 0);
        }
    }
    ref_clears(a, b, expected, got, nullptr);
}
struct Decay {
    FormulaKind kind;
    uint64_t begin;
};
static uint64_t decay(const void *ptr, uint64_t a) {
    const auto &d = *static_cast<const Decay *>(ptr);
    if (a == d.begin)
        return 0;
    if (d.kind == FormulaKind::Euler)
        return uint64_t(integer_log_sum(d.begin, a) - (a - d.begin));
    return 47 * (a - std::max(uint64_t(1), d.begin));
}
static void reference(const Formula &f, uint64_t a, uint64_t b, size_t precision,
                      const sbn3_series_values &out, uint64_t error) {
    ref_int t, d, u, lt, ld, lu, temp, got, gd, diff, limit;
    ref_inits(t, d, u, lt, ld, lu, temp, got, gd, diff, limit, nullptr);
    ref_set_ui(t, 0);
    ref_set_ui(d, 1);
    ref_set_ui(u, 1);
    for (uint64_t k = a; k < b; ++k) {
        ref_set_ui(lt, 1);
        ref_set_ui(ld, k);
        ref_set_ui(lu, 1);
        if (f.kind == FormulaKind::Chudnovsky) {
            if (!k) {
                ref_set_ui(lt, 13591409);
                ref_set_ui(ld, 1);
            } else {
                ref_set_ui(lu, 6 * k - 5);
                ref_mul_ui(lu, lu, 2 * k - 1);
                ref_mul_ui(lu, lu, 6 * k - 1);
                ref_mul_ui(ld, ld, k);
                ref_mul_ui(ld, ld, k);
                ref_mul_ui(ld, ld, 10939058860032000ULL);
                ref_mul_ui(lt, lu, 545140134 * k + 13591409);
                if (k & 1)
                    ref_neg(lt, lt);
            }
        }
        ref_mul(t, t, ld);
        if (f.kind == FormulaKind::Chudnovsky)
            ref_mul(lt, lt, u);
        ref_add(t, t, lt);
        ref_mul(d, d, ld);
        ref_mul(u, u, lu);
    }
    const auto &nt = out.value[0], &nq = out.value[1];
    ref_import(got, nt.mantissa.size, -1, 8, 0, 0, nt.mantissa.data);
    ref_import(gd, nq.mantissa.size, -1, 8, 0, 0, nq.mantissa.data);
    assert(!nq.mantissa.negative && nq.mantissa.size);
    if (nt.mantissa.negative)
        ref_neg(got, got);
    if (nt.exponent2 >= nq.exponent2)
        ref_mul_2exp(got, got, size_t(nt.exponent2 - nq.exponent2));
    else
        ref_mul_2exp(gd, gd, size_t(nq.exponent2 - nt.exponent2));
    ref_mul(got, got, d);
    ref_mul(t, t, gd);
    ref_sub(diff, got, t);
    ref_abs(diff, diff);
    ref_mul_2exp(diff, diff, 64 * precision);
    ref_mul(d, d, gd);
    ref_mul_ui(limit, d, error);
    assert(ref_cmp(diff, limit) < 0);
    ref_clears(t, d, u, lt, ld, lu, temp, got, gd, diff, limit, nullptr);
}
static void one(FormulaKind kind, uint64_t a, uint64_t b, size_t precision, unsigned workers,
                uint64_t max_terms = 0) {
    Fixture fixture(workers, false);
    Formula formula{kind};
    Decay policy{kind, a};
    PsrSpec spec{finite_formula(formula),
                 {a, b},
                 {workers, 8, 0, 0, 0},
                 {&policy, decay, max_terms ? 4u : 128u, max_terms},
                 precision};
    PsrInfo info{};
    allocation_watch_start();
    const auto status = psr_query(spec, info);
    assert(!allocation_watch_stop());
    assert(status == SBN3_SUPPORTED);
    const size_t offset = up(fixture.base + fixture.cursor, info.storage_alignment) - fixture.base;
    fixture.cursor = up(offset + info.storage_bytes, 4096) + 4096;
    sbn3_error error{};
    assert(sbn3_arena_prepare(fixture.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    sbn3_series_values out{};
    auto result = fixture.allocate(2 * info.output_limbs * 8, 128);
    for (unsigned j = 0; j < 2; ++j) {
        out.value[j].mantissa = {static_cast<uint64_t *>(result.data) + j * info.output_limbs,
                                 info.output_limbs, 0, 0};
    }
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        PsrBinding binding{};
        allocation_watch_start();
        psr_prepare(spec, info, *fixture.arena, *fixture.team, offset, binding);
        assert(!allocation_watch_stop());
        allocation_watch_start();
        psr_execute(binding, out);
        assert(!allocation_watch_stop());
        reference(formula, a, b, precision, out, info.error_units);
        allocation_watch_start();
        psr_release(binding);
        assert(!allocation_watch_stop());
    }
    printf("PSR kind=%u [%llu,%llu) p=%zu W%u blocks=%u error<%llu ulps bytes=%zu PASS\n", unsigned(kind),
           (unsigned long long)a, (unsigned long long)b, precision, workers, info.blocks,
           (unsigned long long)info.error_units, info.storage_bytes);
    fflush(stdout);
}
int main() {
    window_addition();
    for (unsigned workers : {1u, 3u, 16u}) {
        one(FormulaKind::Chudnovsky, 0, 37, 4, workers, 8);
        one(FormulaKind::Chudnovsky, 3, 42, 16, workers, 8);
        one(FormulaKind::Chudnovsky, 0, 1000, 64, workers);
        one(FormulaKind::Euler, 1, 81, 8, workers, 8);
    }
    one(FormulaKind::Chudnovsky, 0, 4096, 2048, 16);
    puts("PSR limited pair/affine precision/reverse merge/error/no division/no allocation gates PASS");
}
