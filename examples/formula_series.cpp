// Runnable consumer of the structured formula service: a constant is defined
// by data (P/Q/R factors, start index, explicit first term), evaluated with
// the shared exact binary-splitting recipes and finished by the generic ratio
// terminal. Changing the data changes the constant; nothing else changes.
//   formula_series [--limbs N] [--workers W] [--leaf-terms K] [--m M]
// Computes e = exp(1/1), exp(1/2), exp(1/3) and exp(1/M) to N fractional limbs.
#include "series/formula_def.hpp"
#include "series/terminal.hpp"
#include "sbn3/team.h"
#include <initializer_list>
#include <algorithm>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
using namespace sbn::v3;
using namespace sbn::v3::series;
namespace {
size_t up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }
uint64_t tick() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
struct Runtime {
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    sbn3_lease control{};
    size_t cursor = 0;
    uintptr_t base = 0;
    bool open(unsigned workers, size_t virtual_bytes) {
        sbn3_error error{};
        const sbn3_arena_config config{virtual_bytes, virtual_bytes};
        if (sbn3_arena_create(&config, &arena, &error) != SBN3_OK)
            return false;
        if (sbn3_arena_prepare(arena, 0, sbn3_team_storage_bytes(), &error) != SBN3_OK)
            return false;
        sbn3_arena_acquire(arena, 0, sbn3_team_storage_bytes(), &control);
        sbn3_team_config tc{};
        tc.workers = workers;
        tc.pin_threads = 1;
        tc.stack_offset = 65536;
        for (int &c : tc.cpu_ids)
            c = -1;
        if (sbn3_team_create(arena, &control, &tc, &team, &error) != SBN3_OK) {
            fprintf(stderr, "team: %s errno=%d\n", error.where, error.system_error);
            return false;
        }
        base = uintptr_t(control.data);
        cursor = tc.stack_offset + sbn3_team_stack_virtual_bytes(workers) + 4096;
        return true;
    }
    // Prepared, unleased range; the caller acquires leases as needed.
    size_t region(size_t bytes, size_t alignment) {
        cursor = up(base + cursor, alignment) - base;
        const size_t at = cursor;
        sbn3_error error{};
        if (sbn3_arena_prepare(arena, at, bytes, &error) != SBN3_OK) {
            fprintf(stderr, "prepare: %s errno=%d need=%zu\n", error.where, error.system_error, error.need);
            exit(1);
        }
        cursor = up(cursor + bytes, 4096);
        return at;
    }
    sbn3_lease lease(size_t bytes, size_t alignment) {
        const size_t at = region(bytes, alignment);
        sbn3_lease l{};
        sbn3_arena_acquire(arena, at, bytes, &l);
        return l;
    }
    void reset(size_t cursor_at) { cursor = cursor_at; }
    void close() {
        if (team)
            sbn3_team_destroy(team);
        if (control.bytes)
            sbn3_arena_release(arena, &control);
        if (arena)
            sbn3_arena_destroy(arena);
    }
};
// exp(1/m) to `fractional` limbs. Returns the integer limb and the top
// fractional limbs; verifies the leading 60 bits against long double.
bool evaluate(Runtime &rt, const char *label, const FormulaDef &def, size_t fractional, unsigned workers,
              unsigned leaf_terms, long double expected) {
    const uint64_t start = tick();
    const size_t guard = 2, working = fractional + guard;
    // Contribution/tail fact: pick N so the omitted tail is below one unit at
    // the working precision; the analysis certifies attenuation, the caller
    // adds the amplitude and a guard (documented in formula_def.hpp).
    DataFormula probe{};
    if (!data_formula_prepare(def, std::min<uint64_t>(uint64_t(1) << 40, formula_domain_limit(def)), probe)) {
        fprintf(stderr, "%s: %s\n", label, probe.rejection);
        return false;
    }
    if (probe.tail_rejection) {
        fprintf(stderr, "%s: %s\n", label, probe.tail_rejection);
        return false;
    }
    const uint64_t terms = probe.infinite_tail_terms(64 * working + 8);
    if (!terms) {
        fprintf(stderr, "%s: the tail cannot be certified within the probe domain\n", label);
        return false;
    }
    DataFormula data{};
    if (!data_formula_prepare(def, terms, data)) {
        fprintf(stderr, "%s: %s\n", label, data.rejection);
        return false;
    }
    const auto formula = data_finite_formula(data);
    FinitePlan plan{};
    const sbn3_series_options options{workers, leaf_terms, 1, 0, 0};
    if (finite_query(formula, {def.begin, terms}, 3, options, plan) != SBN3_SUPPORTED) {
        fprintf(stderr, "%s: finite query rejected\n", label);
        return false;
    }
    RatioTerminalPlan terminal{};
    if (ratio_terminal_query(fractional, guard, workers, 1, terminal) != SBN3_SUPPORTED) {
        fprintf(stderr, "%s: terminal query rejected\n", label);
        return false;
    }
    const size_t mark = rt.cursor;
    auto metadata = rt.lease(plan.info.plan_bytes, plan.info.plan_alignment);
    auto prepared = rt.lease(plan.info.prepared_bytes, plan.info.prepared_alignment);
    auto scratch = rt.lease(plan.info.workspace_bytes, plan.info.workspace_alignment);
    sbn3_series_values values{};
    sbn3_lease slots[2];
    for (unsigned j = 0; j < 2; ++j) {
        slots[j] = rt.lease(plan.info.output.limbs[j] * 8, 64);
        values.value[j].mantissa = {static_cast<uint64_t *>(slots[j].data), plan.info.output.limbs[j], 0, 0};
    }
    const size_t terminal_at = rt.region(terminal.storage_bytes, terminal.storage_alignment);
    auto output = rt.lease(terminal.output_limbs * 8, 64);
    FiniteBinding binding{};
    finite_prepare(plan, *rt.arena, *rt.team, metadata, prepared, scratch, binding);
    const uint64_t ready = tick();
    finite_execute(binding, values);
    const uint64_t summed = tick();
    ratio_terminal_execute(terminal, values.value[0], values.value[1], rt.arena, terminal_at, rt.team,
                           {static_cast<uint64_t *>(output.data), terminal.output_limbs});
    const uint64_t done = tick();
    const auto *v = static_cast<const uint64_t *>(output.data);
    // Leading 60 bits from long double: integer limb and top fractional limb.
    const long double top = ldexpl(expected, 0);
    const uint64_t integer = uint64_t(floorl(top));
    const uint64_t fraction = uint64_t(ldexpl(top - floorl(top), 64));
    const bool ok = v[fractional] == integer && ((v[fractional - 1] ^ fraction) >> 12) == 0;
    printf("{\"kind\":\"formula\",\"label\":\"%s\",\"fractional_limbs\":%zu,\"terms\":%" PRIu64 ",\"workers\":%u,"
           "\"leaf_words\":%u,\"leaf\":\"%s\",\"T_limbs\":%zu,\"D_limbs\":%zu,\"prepare_ns\":%" PRIu64
           ",\"series_ns\":%" PRIu64 ",\"terminal_ns\":%" PRIu64 ",\"value\":\"%" PRIx64 ".%016" PRIx64 "%016" PRIx64
           "\",\"long_double_check\":%s}\n",
           label, fractional, terms, workers, formula.max_leaf_limbs,
           data.leaf_kind == LeafKind::HyperWordBatch ? "batch" : "factored", values.value[0].mantissa.size,
           values.value[1].mantissa.size, ready - start, summed - ready, done - summed, v[fractional],
           v[fractional - 1], fractional >= 2 ? v[fractional - 2] : 0, ok ? "true" : "false");
    fflush(stdout);
    for (auto *l : {&metadata, &prepared, &scratch, &slots[0], &slots[1], &output})
        sbn3_arena_release(rt.arena, l);
    rt.reset(mark);
    return ok;
}
} // namespace
int main(int argc, char **argv) {
    size_t limbs = 4096;
    unsigned workers = 16, leaf_terms = 8;
    uint64_t m = 5;
    for (int j = 1; j + 1 < argc; j += 2) {
        const char *key = argv[j], *value = argv[j + 1];
        if (!strcmp(key, "--limbs"))
            limbs = strtoull(value, nullptr, 10);
        else if (!strcmp(key, "--workers"))
            workers = unsigned(strtoul(value, nullptr, 10));
        else if (!strcmp(key, "--leaf-terms"))
            leaf_terms = unsigned(strtoul(value, nullptr, 10));
        else if (!strcmp(key, "--m"))
            m = strtoull(value, nullptr, 10);
        else
            return 2;
    }
    if (!limbs || limbs > (size_t(1) << 22) || !workers || workers > 32 || !leaf_terms || leaf_terms > 1024 || !m)
        return 2;
    Runtime rt{};
    if (!rt.open(workers, size_t(64) << 30))
        return 1;
    bool ok = true;
    // The same data definition family; only the denominator constant changes.
    ok &= evaluate(rt, "e", exp_reciprocal_definition(1), limbs, workers, leaf_terms, expl(1.0L));
    ok &= evaluate(rt, "exp(1/2)", exp_reciprocal_definition(2), limbs, workers, leaf_terms, expl(0.5L));
    ok &= evaluate(rt, "exp(1/3)", exp_reciprocal_definition(3), limbs, workers, leaf_terms, expl(1.0L / 3));
    char label[32];
    snprintf(label, sizeof label, "exp(1/%" PRIu64 ")", m);
    ok &= evaluate(rt, label, exp_reciprocal_definition(m), limbs, workers, leaf_terms, expl(1.0L / (long double)m));
    // A different recipe from the same service: exp(-1) with alternating signs.
    ok &= evaluate(rt, "exp(-1)", exp_minus_one_definition(), limbs, workers, leaf_terms, expl(-1.0L));
    rt.close();
    return ok ? 0 : 1;
}
