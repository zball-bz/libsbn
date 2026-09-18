// Public C constant service: full outputs against independent references at
// tiny and moderate precisions (F2), rejection of divergent definitions (F4),
// rejection of out-of-domain C fields (F5), outer-scale semantics including
// an exact zero result and the deterministic guard-boundary fatal (review §3),
// capacity rejection, threads, rebinding and zero execution-time allocation.
#include "formula_reference.hpp"
#include "sbn3/formula.h"
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <unistd.h>
using namespace sbn::v3;
using namespace sbn::v3::series;
using namespace formula_reference;
namespace {
sbn3_formula_def exp_reciprocal(uint64_t m) {
    sbn3_formula_def d{};
    d.recipe = SBN3_SERIES_HYPERDESCENT;
    d.P.constant_low = d.R.constant_low = 1;
    d.Q.constant_low = m;
    d.Q.factor_count = 1;
    d.Q.factor[0] = {1, 0, 1};
    d.explicit_first = 1;
    d.first_t = d.first_d = d.first_u = 1;
    return d;
}
sbn3_formula_def sin1() {
    sbn3_formula_def d{};
    d.recipe = SBN3_SERIES_HYPERDESCENT;
    d.P.constant_low = d.R.constant_low = 1;
    d.P.alternating = 1;
    d.Q.constant_low = 2;
    d.Q.factor_count = 2;
    d.Q.factor[0] = {1, 0, 1};
    d.Q.factor[1] = {2, 1, 1};
    d.explicit_first = 1;
    d.first_t = d.first_d = d.first_u = 1;
    return d;
}
sbn3_formula_def zeta3_az() {
    sbn3_formula_def d{};
    d.recipe = SBN3_SERIES_COMMON_P2B3;
    d.P.constant_low = 1;
    d.P.alternating = 1;
    d.P.factor_count = 1;
    d.P.factor[0] = {1, 0, 5};
    d.P.degree = 2;
    d.P.coefficient[0] = 77;
    d.P.coefficient[1] = 250;
    d.P.coefficient[2] = 205;
    d.Q.constant_low = 32;
    d.Q.factor_count = 1;
    d.Q.factor[0] = {2, 1, 5};
    d.R.constant_low = 1;
    d.R.factor_count = 1;
    d.R.factor[0] = {1, 0, 5};
    d.explicit_first = 1;
    d.first_t = 77;
    d.first_d = d.first_u = 1;
    d.denominator_exponent = 6;
    return d;
}
sbn3_formula_def zeta3_apery() {
    sbn3_formula_def d{};
    d.recipe = SBN3_SERIES_COMMON_P2B3;
    d.begin = 1;
    d.P.constant_low = 1;
    d.P.negative = 1;
    d.P.factor_count = 1;
    d.P.factor[0] = {1, -1, 3};
    d.Q.constant_low = 2;
    d.Q.factor_count = 2;
    d.Q.factor[0] = {1, 0, 2};
    d.Q.factor[1] = {2, -1, 1};
    d.R.constant_low = 1;
    d.R.negative = 1;
    d.R.factor_count = 1;
    d.R.factor[0] = {1, -1, 3};
    d.explicit_first = 1;
    d.first_t = 1;
    d.first_d = 2;
    d.first_u = 1;
    d.numerator_scale = 5;
    d.denominator_exponent = 1;
    return d;
}
// Same definition for the reference evaluator.
FormulaDef internal(const sbn3_formula_def &in) {
    FormulaDef d{};
    d.recipe = in.recipe;
    d.begin = in.begin;
    auto product = [](const sbn3_formula_product &p) {
        FactorProduct f{};
        f.constant_low = p.constant_low;
        f.constant_high = p.constant_high;
        f.negative = p.negative;
        f.alternating = p.alternating;
        f.count = p.factor_count;
        for (unsigned i = 0; i < f.count; ++i)
            f.factor[i] = {p.factor[i].a, p.factor[i].b, p.factor[i].power};
        f.degree = p.degree;
        for (unsigned i = 0; i <= f.degree; ++i)
            f.coefficient[i] = p.coefficient[i];
        return f;
    };
    d.P = product(in.P);
    d.Q = product(in.Q);
    d.R = product(in.R);
    d.explicit_first = in.explicit_first;
    d.first = {in.first_t, in.first_d, in.first_u, in.first_t_negative != 0};
    return d;
}
// floor(scale * T / (D 2^e) * 2^(64 n)) from an exact finite sum with far more
// terms than the service certifies; the remainder must stay away from the
// boundaries so the floor is the true one.
void expected(const sbn3_formula_def &def, uint64_t terms, size_t n, ref_number *out) {
    Ref r;
    reference(internal(def), def.begin, terms, r);
    ref_int num, den, rem;
    ref_inits(num, den, rem, nullptr);
    ref_set(num, r.t);
    if (def.numerator_scale > 1)
        ref_mul_ui(num, num, def.numerator_scale);
    ref_mul_2exp(num, num, 64 * n);
    ref_set(den, r.d);
    if (def.denominator_exponent >= 0)
        ref_mul_2exp(den, den, size_t(def.denominator_exponent));
    else
        ref_mul_2exp(num, num, size_t(-def.denominator_exponent));
    ref_fdiv_q(out, num, den);
    ref_mod(rem, num, den);
    ref_mul_2exp(rem, rem, 64);
    assert(ref_cmp(rem, den) > 0); // remainder above 2^-64 of a unit
    ref_sub(rem, rem, den);
    ref_mul_2exp(den, den, 64);
    assert(ref_cmp(rem, den) < 0); // and below (1 - 2^-64) of it
    ref_clears(num, den, rem, nullptr);
}
struct Run {
    sbn3_formula_info info{};
    std::vector<uint64_t> value;
};
Run compute(Fixture &fixture, const sbn3_formula_def &def, size_t n, unsigned workers, bool inplace, void *object) {
    Run r{};
    sbn3_formula_options options{{workers, 8, 0, 0, 0}, 0, 0, 0};
    sbn3_formula_plan plan{};
    allocation_watch_start();
    const auto rc = sbn3_formula_query(&def, n, &options, object, &plan, &r.info);
    assert(!allocation_watch_stop());
    if (rc != SBN3_SUPPORTED)
        fprintf(stderr, "query %u: %s\n", rc, r.info.rejection ? r.info.rejection : "");
    assert(rc == SBN3_SUPPORTED);
    const auto saved = plan;
    options.memory_budget = r.info.storage_bytes - 1;
    sbn3_formula_info rejected{};
    // Capacity rejection reports the requirement and leaves the plan untouched.
    assert(sbn3_formula_query(&def, n, &options, object, &plan, &rejected) == SBN3_QUERY_CAPACITY &&
           rejected.storage_bytes == r.info.storage_bytes && !memcmp(&saved, &plan, sizeof plan));
    assert(sbn3_formula_query(&def, n, &options, object, &plan, &rejected) == SBN3_QUERY_CAPACITY &&
           !memcmp(&saved, &plan, sizeof plan));
    options.memory_budget = 0;
    // A repeated successful query yields the same plan identity and sizes.
    // Opaque bytes are not compared: embedded service plans (Newton) carry
    // padding, and byte identity is not part of the plan contract.
    const auto again = sbn3_formula_query(&def, n, &options, object, &plan, &rejected);
    assert(again == SBN3_SUPPORTED && rejected.plan_id == r.info.plan_id && rejected.storage_bytes == r.info.storage_bytes &&
           rejected.terms == r.info.terms);
    const size_t offset = up(fixture.base + fixture.cursor, r.info.storage_alignment) - fixture.base;
    fixture.cursor = up(offset + r.info.storage_bytes, 4096) + 4096;
    sbn3_error error{};
    assert(sbn3_arena_prepare(fixture.arena, offset, r.info.storage_bytes, &error) == SBN3_OK);
    auto out = fixture.allocate(r.info.output_limbs * 8, 64);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        sbn3_formula_binding *binding = nullptr;
        allocation_watch_start();
        sbn3_formula_bind(&plan, object, fixture.arena, offset, fixture.team, &binding);
        const uint64_t *data = nullptr;
        if (inplace) {
            data = sbn3_formula_execute_inplace(binding).data;
        } else {
            sbn3_formula_execute(binding, {static_cast<uint64_t *>(out.data), r.info.output_limbs});
            data = static_cast<uint64_t *>(out.data);
        }
        assert(!allocation_watch_stop());
        std::vector<uint64_t> got(data, data + r.info.output_limbs);
        if (repeat)
            assert(got == r.value);
        r.value = got;
        allocation_watch_start();
        sbn3_formula_unbind(binding);
        assert(!allocation_watch_stop());
    }
    return r;
}
void check(Fixture &fixture, const char *label, const sbn3_formula_def &def, size_t n, unsigned workers, void *object) {
    auto r = compute(fixture, def, n, workers, (n + workers) & 1, object);
    ref_int want, got;
    ref_inits(want, got, nullptr);
    expected(def, 2 * r.info.terms + 16, n, want);
    ref_import(got, r.value.size(), -1, 8, 0, 0, r.value.data());
    if (ref_cmp(want, got))
        fprintf(stderr, "%s n=%zu W%u mismatch: integer limb %llx\n", label, n, workers, (unsigned long long)r.value[n]);
    assert(!ref_cmp(want, got));
    ref_clears(want, got, nullptr);
    // Planning result, not a mode: a Hyperdescent prefix stopped by its tail bound fits its own
    // demand and is one exact tree; no public option selects this.
    assert(r.info.exact_blocks <= r.info.blocks);
    if (def.recipe == SBN3_SERIES_HYPERDESCENT)
        assert(r.info.blocks == 1 && r.info.exact_blocks == 1);
    printf("C service %s n=%zu W%u terms=%llu working=%zu blocks=%u exact=%u value=%llx.%016llx PASS\n", label, n, workers,
           (unsigned long long)r.info.terms, r.info.working_limbs, r.info.blocks, r.info.exact_blocks,
           (unsigned long long)r.value[n], (unsigned long long)r.value[n - 1]);
    fflush(stdout);
}
void rejected(const char *label, sbn3_formula_def def, void *object, const char *fragment) {
    sbn3_formula_options options{{1, 8, 0, 0, 0}, 0, 0, 0};
    sbn3_formula_plan plan{};
    sbn3_formula_info info{};
    const auto rc = sbn3_formula_query(&def, 4, &options, object, &plan, &info);
    if (rc != SBN3_UNSUPPORTED || !info.rejection || !strstr(info.rejection, fragment))
        fprintf(stderr, "%s: rc=%u rejection=%s\n", label, rc, info.rejection ? info.rejection : "(none)");
    assert(rc == SBN3_UNSUPPORTED && info.rejection && strstr(info.rejection, fragment));
    printf("rejected %s: %s\n", label, info.rejection);
}
} // namespace
int main() {
    void *object = aligned_alloc(64, up(sbn3_formula_object_bytes(), 64));
    assert(object);
    // F5: fields outside the header's domain are rejected, never clamped.
    {
        auto d = exp_reciprocal(1);
        d.P.factor_count = 7;
        for (auto &f : d.P.factor)
            f = {0, 1, 1};
        rejected("factor_count 7", d, object, "factor_count");
        d = exp_reciprocal(1);
        d.P.degree = 5;
        rejected("degree 5", d, object, "degree");
        d = exp_reciprocal(1);
        d.P.alternating = 2;
        rejected("alternating 2", d, object, "0 or 1");
        d = exp_reciprocal(1);
        d.explicit_first = 3;
        rejected("explicit_first 3", d, object, "0 or 1");
        d = exp_reciprocal(1);
        d.Q.factor[0].power = 17;
        rejected("power 17", d, object, "1..16");
        d = exp_reciprocal(1);
        d.Q.factor[0].power = 0;
        rejected("power 0", d, object, "1..16");
        d = exp_reciprocal(1);
        d.recipe = sbn3_series_recipe(3);
        rejected("recipe 3", d, object, "recipe");
    }
    // F4: divergent and non-geometric definitions are rejected with the structural reason.
    {
        sbn3_formula_def d{};
        d.recipe = SBN3_SERIES_COMMON_P2B3;
        d.begin = 1;
        d.P.constant_low = d.R.constant_low = 1;
        d.Q.constant_high = uint64_t(1) << 36; // Q = 2^100, R = k: sum (k-1)!/2^(100k)
        d.R.factor_count = 1;
        d.R.factor[0] = {1, 0, 1};
        rejected("divergent (k-1)!/2^(100k)", d, object, "grows without bound");
        sbn3_formula_def h{};
        h.recipe = SBN3_SERIES_COMMON_P2B3;
        h.begin = 1;
        h.P.constant_low = h.Q.constant_low = h.R.constant_low = 1;
        h.Q.factor_count = 1;
        h.Q.factor[0] = {1, 1, 1};
        h.R.factor_count = 1;
        h.R.factor[0] = {1, 0, 1};
        rejected("ratio -> 1", h, object, "does not tend below 1");
    }
    // Outer scale range: e * 2^62 exceeds the terminal's 2^63 domain; 2^61 is accepted.
    {
        auto d = exp_reciprocal(1);
        d.numerator_scale = uint64_t(1) << 62;
        rejected("scale 2^62", d, object, "2^63");
    }
    const struct { const char *label; sbn3_formula_def def; } constants[]{
        {"e", exp_reciprocal(1)}, {"exp(1/2)", exp_reciprocal(2)}, {"sin(1)", sin1()}, {"zeta(3) AZ /64", zeta3_az()},
        {"zeta(3) Apery 5/2", zeta3_apery()}};
    for (unsigned workers : {1u, 3u, 16u}) {
        Fixture fixture(workers, false);
        // F2: tiny precisions with full-output references; the integer limb of e is 2.
        for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(17), size_t(64)})
            for (const auto &c : constants) {
                if (n < 3 && c.def.recipe == SBN3_SERIES_COMMON_P2B3 && workers == 3)
                    continue; // same coverage at the other worker counts
                check(fixture, c.label, c.def, n, workers, object);
            }
        auto e = exp_reciprocal(1);
        auto r = compute(fixture, e, 1, workers, false, object);
        assert(r.value[1] == 2 && r.value[0] == 0xb7e151628aed2a6aULL);
        // Outer scale semantics: e * 2^61 (fits), e * 8 (negative exponent), e / 2^1024 (exact zero result).
        e.numerator_scale = uint64_t(1) << 61;
        check(fixture, "e * 2^61", e, 5, workers, object);
        e = exp_reciprocal(1);
        e.denominator_exponent = -3;
        check(fixture, "e * 8", e, 5, workers, object);
        e = exp_reciprocal(1);
        e.denominator_exponent = 1024;
        r = compute(fixture, e, 4, workers, true, object);
        assert(std::all_of(r.value.begin(), r.value.end(), [](uint64_t x) { return x == 0; }));
        puts("outer scale: range, negative exponent and exact zero result PASS");
    }
    // Two zeta(3) definitions, one output.
    {
        Fixture fixture(16, false);
        auto a = compute(fixture, zeta3_az(), 300, 16, true, object), b = compute(fixture, zeta3_apery(), 300, 16, false, object);
        assert(a.value == b.value && a.value[300] == 1 && a.value[299] == 0x33ba004f00621383ULL);
    }
    // Deterministic support boundary: a value at an output word boundary
    // (sum 2^-k = 1) cannot be certified; the compute face aborts.
    {
        sbn3_formula_def one{};
        one.recipe = SBN3_SERIES_HYPERDESCENT;
        one.begin = 1;
        one.P.constant_low = one.R.constant_low = 1;
        one.Q.constant_low = 2; // term_k = 2^-k
        const pid_t child = fork();
        assert(child >= 0);
        if (!child) {
            rlimit z{0, 0};
            setrlimit(RLIMIT_CORE, &z);
            Fixture fixture(1, false);
            compute(fixture, one, 2, 1, true, object);
            _exit(0);
        }
        int status = 0;
        assert(waitpid(child, &status, 0) == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
        puts("guard-boundary value aborts deterministically on the compute face PASS");
    }
    free(object);
    puts("public formula service: full outputs at 1..64 limbs, rejections (fields, divergence, range), outer scale, "
         "capacity, threads, rebind, no allocation PASS");
}
