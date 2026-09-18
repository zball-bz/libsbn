/* Thin C service consumer: constants defined by data only.
 *   formula_constants [--limbs N] [--workers W] [--only LABEL] [--leaf-terms T]
 * Computes e, exp(1/2), zeta(3) (Amdeberhan-Zeilberger) and sin(1) to N
 * fractional limbs (or only the one labelled LABEL, so that process-wide
 * memory figures belong to it) and prints their leading hex limbs; exit 1 on
 * a mismatch with the known leading words. */
#define _POSIX_C_SOURCE 200809L
#include "sbn3/formula.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
static size_t up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }
static uint64_t tick(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec; }
static sbn3_formula_def exp_reciprocal(uint64_t m) { /* sum_{k>=0} 1/(m^k k!) */
    sbn3_formula_def d; memset(&d, 0, sizeof d);
    d.recipe = SBN3_SERIES_HYPERDESCENT; d.begin = 0;
    d.P.constant_low = 1; d.R.constant_low = 1;
    d.Q.constant_low = m; d.Q.factor_count = 1; d.Q.factor[0].a = 1; d.Q.factor[0].power = 1;
    d.explicit_first = 1; d.first_t = 1; d.first_d = 1; d.first_u = 1;
    return d;
}
static sbn3_formula_def zeta3(void) { /* 64 zeta(3) = sum (-1)^k (205k^2+250k+77) (k!)^10/((2k+1)!)^5 */
    sbn3_formula_def d; memset(&d, 0, sizeof d);
    d.recipe = SBN3_SERIES_COMMON_P2B3; d.begin = 0;
    d.P.constant_low = 1; d.P.alternating = 1; d.P.factor_count = 1; d.P.factor[0].a = 1; d.P.factor[0].power = 5;
    d.P.degree = 2; d.P.coefficient[0] = 77; d.P.coefficient[1] = 250; d.P.coefficient[2] = 205;
    d.Q.constant_low = 32; d.Q.factor_count = 1; d.Q.factor[0].a = 2; d.Q.factor[0].b = 1; d.Q.factor[0].power = 5;
    d.R.constant_low = 1; d.R.factor_count = 1; d.R.factor[0].a = 1; d.R.factor[0].power = 5;
    d.explicit_first = 1; d.first_t = 77; d.first_d = 1; d.first_u = 1;
    d.denominator_exponent = 6;
    return d;
}
static sbn3_formula_def sin1(void) { /* sum (-1)^k/(2k+1)! */
    sbn3_formula_def d; memset(&d, 0, sizeof d);
    d.recipe = SBN3_SERIES_HYPERDESCENT; d.begin = 0;
    d.P.constant_low = 1; d.P.alternating = 1; d.R.constant_low = 1;
    d.Q.constant_low = 2; d.Q.factor_count = 2;
    d.Q.factor[0].a = 1; d.Q.factor[0].power = 1; d.Q.factor[1].a = 2; d.Q.factor[1].b = 1; d.Q.factor[1].power = 1;
    d.explicit_first = 1; d.first_t = 1; d.first_d = 1; d.first_u = 1;
    return d;
}
int main(int argc, char **argv) {
    size_t n = 4096; unsigned workers = 16, leaf_terms = 8; const char *only = NULL; /* --leaf-terms 0: service policy */
    for (int j = 1; j + 1 < argc; j += 2) {
        if (!strcmp(argv[j], "--limbs")) n = strtoull(argv[j + 1], NULL, 10);
        else if (!strcmp(argv[j], "--workers")) workers = (unsigned)strtoul(argv[j + 1], NULL, 10);
        else if (!strcmp(argv[j], "--only")) only = argv[j + 1];
        else if (!strcmp(argv[j], "--leaf-terms")) leaf_terms = (unsigned)strtoul(argv[j + 1], NULL, 10);
        else return 2;
    }
    if (!n || n > (1u << 22) || !workers || workers > 32) return 2;
    sbn3_arena_config ac = {(size_t)64 << 30, (size_t)64 << 30};
    sbn3_arena *arena = NULL; sbn3_team *team = NULL; sbn3_lease control = {0}; sbn3_error error = {0};
    if (sbn3_arena_create(&ac, &arena, &error) != SBN3_OK) return 1;
    if (sbn3_arena_prepare(arena, 0, sbn3_team_storage_bytes(), &error) != SBN3_OK) return 1;
    sbn3_arena_acquire(arena, 0, sbn3_team_storage_bytes(), &control);
    sbn3_team_config tc; memset(&tc, 0, sizeof tc); tc.workers = workers; tc.pin_threads = 1; tc.stack_offset = 65536;
    for (unsigned j = 0; j < 32; ++j) tc.cpu_ids[j] = -1;
    if (sbn3_team_create(arena, &control, &tc, &team, &error) != SBN3_OK) { fprintf(stderr, "team: %s\n", error.where); return 1; }
    size_t cursor = tc.stack_offset + sbn3_team_stack_virtual_bytes(workers) + 4096;
    const uintptr_t base = (uintptr_t)control.data;
    void *object = aligned_alloc(64, up(sbn3_formula_object_bytes(), 64));
    struct { const char *label; sbn3_formula_def def; uint64_t integer, top; } cases[4] = {
        {"e", exp_reciprocal(1), 2, 0xb7e151628aed2a6aULL},
        {"exp(1/2)", exp_reciprocal(2), 1, 0xa61298e1e069bc97ULL},
        {"zeta(3)", zeta3(), 1, 0x33ba004f00621383ULL}, /* two independent formulas agree in series_consumers_test */
        {"sin(1)", sin1(), 0, 0xd76aa47848677021ULL}};
    int ok = 1;
    for (unsigned c = 0; c < 4; ++c) {
        if (only && strcmp(only, cases[c].label)) continue;
        sbn3_formula_options options = {{workers, leaf_terms, 0, 0, 0}, 0, 0, 0};
        sbn3_formula_plan plan; sbn3_formula_info info;
        const uint64_t q0 = tick();
        const sbn3_query_result rc = sbn3_formula_query(&cases[c].def, n, &options, object, &plan, &info);
        if (rc != SBN3_SUPPORTED) { fprintf(stderr, "%s: query %u %s\n", cases[c].label, rc, info.rejection ? info.rejection : ""); ok = 0; continue; }
        const size_t offset = up(base + cursor, info.storage_alignment) - base;
        if (sbn3_arena_prepare(arena, offset, info.storage_bytes, &error) != SBN3_OK) { fprintf(stderr, "prepare: %s\n", error.where); return 1; }
        sbn3_formula_binding *binding = NULL;
        const uint64_t b0 = tick();
        sbn3_formula_bind(&plan, object, arena, offset, team, &binding);
        const uint64_t e0 = tick();
        const sbn3_const_limbs v = sbn3_formula_execute_inplace(binding);
        const uint64_t e1 = tick();
        struct rusage usage; getrusage(RUSAGE_SELF, &usage);
        const int match = v.data[n] == cases[c].integer && (v.data[n - 1] >> 8) == (cases[c].top >> 8);
        printf("{\"kind\":\"formula_c\",\"label\":\"%s\",\"fractional_limbs\":%zu,\"terms\":%" PRIu64 ",\"blocks\":%u,\"exact_blocks\":%u,\"leaf_words\":%u,"
               "\"query_ns\":%" PRIu64 ",\"bind_ns\":%" PRIu64 ",\"execute_ns\":%" PRIu64 ",\"storage_bytes\":%zu,\"maxrss_bytes\":%" PRIu64 ","
               "\"value\":\"%" PRIx64 ".%016" PRIx64 "%016" PRIx64 "\",\"known_leading_words\":%s}\n",
               cases[c].label, n, info.terms, info.blocks, info.exact_blocks, info.leaf_words, b0 - q0, e0 - b0, e1 - e0, info.storage_bytes,
               (uint64_t)usage.ru_maxrss * 1024,
               v.data[n], v.data[n - 1], n >= 2 ? v.data[n - 2] : 0, match ? "true" : "false");
        fflush(stdout);
        ok &= match;
        sbn3_formula_unbind(binding);
        if (sbn3_arena_trim(arena, offset, info.storage_bytes, &error) != SBN3_OK) return 1;
    }
    free(object);
    sbn3_team_destroy(team); sbn3_arena_release(arena, &control); sbn3_arena_destroy(arena);
    return ok ? 0 : 1;
}
