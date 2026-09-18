/* log_constants log|arccoth ARG [fractional-limbs [workers]]
 * Public C headers, no SIMD flags. The host process needs sufficient
 * RLIMIT_MEMLOCK, as for the other arena-backed examples. */
#define _POSIX_C_SOURCE 200809L
#include "sbn3/log.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static size_t up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }
static uint64_t tick(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}
static int parse(const char *s, uint64_t *value) {
    if (!*s || *s == '-')
        return 0;
    char *end = NULL;
    errno = 0;
    const unsigned long long x = strtoull(s, &end, 10);
    if (errno || *end)
        return 0;
    *value = (uint64_t)x;
    return 1;
}
int main(int argc, char **argv) {
    if (argc < 3 || argc > 6 || (argc == 6 && strcmp(argv[5], "--query")))
        return 2;
    const int is_log = !strcmp(argv[1], "log");
    if (!is_log && strcmp(argv[1], "arccoth"))
        return 2;
    uint64_t argument, requested_limbs = 1024, requested_workers = 16;
    if (!parse(argv[2], &argument) || (argc > 3 && !parse(argv[3], &requested_limbs)) ||
        (argc > 4 && !parse(argv[4], &requested_workers)))
        return 2;
    if (!requested_limbs || requested_limbs > (1u << 24) || !requested_workers || requested_workers > 32 ||
        (is_log ? (!argument || argument > UINT32_MAX) : argument < 2))
        return 2;
    const size_t n = (size_t)requested_limbs;
    const unsigned workers = (unsigned)requested_workers;
    const uint64_t query_start = tick();
    const size_t object_bytes =
        is_log ? sbn3_log_object_bytes((uint32_t)argument) : sbn3_formula_object_bytes();
    void *object = aligned_alloc(64, up(object_bytes, 64));
    if (!object)
        return 1;
    const sbn3_formula_options options = {{workers, 0, 0, 0, 0}, 0, 0, 0}; /* zeros: service policies */
    sbn3_formula_plan arc_plan;
    sbn3_formula_info arc_info;
    sbn3_formula_sum_plan log_plan;
    sbn3_formula_sum_info log_info;
    const sbn3_query_result rc =
        is_log ? sbn3_log_query((uint32_t)argument, n, &options, object, object_bytes, &log_plan, &log_info)
               : sbn3_arccoth_query(argument, n, &options, object, &arc_plan, &arc_info);
    if (rc != SBN3_SUPPORTED) {
        fprintf(stderr, "query %u: %s\n", rc, is_log ? log_info.rejection : arc_info.rejection);
        free(object);
        return 1;
    }
    const size_t bytes = is_log ? log_info.storage_bytes : arc_info.storage_bytes;
    const size_t alignment = is_log ? log_info.storage_alignment : arc_info.storage_alignment;
    const uint64_t query_end = tick();
    if (argc == 6) {
        printf("{\"kind\":\"query\",\"operation\":\"%s\",\"argument\":%" PRIu64
               ",\"fractional_limbs\":%zu,\"workers\":%u,\"storage_bytes\":%zu,\"object_bytes\":%zu,"
               "\"components\":%u,\"terms\":%" PRIu64 ",\"blocks\":%u,\"query_ns\":%" PRIu64 "}\n",
               argv[1], argument, n, workers, bytes, object_bytes, is_log ? log_info.components : 1,
               is_log ? 0 : arc_info.terms, is_log ? 0 : arc_info.blocks, query_end - query_start);
        free(object);
        return 0;
    }
    sbn3_arena_config ac = {(size_t)64 << 30, (size_t)64 << 30};
    sbn3_arena *arena = NULL;
    sbn3_team *team = NULL;
    sbn3_error error = {0};
    sbn3_lease control = {0};
    if (sbn3_arena_create(&ac, &arena, &error) != SBN3_OK ||
        sbn3_arena_prepare(arena, 0, sbn3_team_storage_bytes(), &error) != SBN3_OK)
        return 1;
    sbn3_arena_acquire(arena, 0, sbn3_team_storage_bytes(), &control);
    sbn3_team_config tc;
    memset(&tc, 0, sizeof tc);
    tc.workers = workers;
    tc.pin_threads = 1;
    tc.stack_offset = 65536;
    for (unsigned j = 0; j < 32; ++j)
        tc.cpu_ids[j] = -1;
    if (sbn3_team_create(arena, &control, &tc, &team, &error) != SBN3_OK)
        return 1;
    const uintptr_t base = (uintptr_t)control.data;
    const size_t offset =
        up(base + tc.stack_offset + sbn3_team_stack_virtual_bytes(workers) + 4096, alignment) - base;
    if (sbn3_arena_prepare(arena, offset, bytes, &error) != SBN3_OK) {
        fprintf(stderr, "prepare: %s errno=%d bytes=%zu\n", error.where, error.system_error, bytes);
        return 1;
    }
    sbn3_const_limbs value;
    sbn3_formula_binding *arc = NULL;
    sbn3_formula_sum_binding *logarithm = NULL;
    uint64_t bind_start, execute_start, execute_end;
    if (is_log) {
        sbn3_log_formula formula;
        sbn3_log_formula_query((uint32_t)argument, &formula);
        printf("log(%" PRIu64 ") = (2/%" PRIu64 ") * [", argument, formula.divisor);
        for (unsigned j = 0; j < formula.count; ++j) {
            if (formula.term[j].numerator == 1)
                printf("%s%" PRId64 "*ArcCoth(%" PRIu64 ")", j ? " + " : "", formula.term[j].coefficient,
                       formula.term[j].argument);
            else
                printf("%s%" PRId64 "*atanh(%" PRIu64 "/%" PRIu64 ")", j ? " + " : "", formula.term[j].coefficient,
                       formula.term[j].numerator, formula.term[j].argument);
        }
        puts("]");
        bind_start = tick();
        sbn3_formula_sum_bind(&log_plan, object, arena, offset, team, &logarithm);
        execute_start = tick();
        value = sbn3_formula_sum_execute_inplace(logarithm);
        execute_end = tick();
    } else {
        bind_start = tick();
        sbn3_formula_bind(&arc_plan, object, arena, offset, team, &arc);
        execute_start = tick();
        value = sbn3_formula_execute_inplace(arc);
        execute_end = tick();
    }
    printf("%s(%" PRIu64 ") = %" PRIx64 ".", argv[1], argument, value.data[n]);
    for (size_t j = n; j > (n > 4 ? n - 4 : 0); --j)
        printf("%016" PRIx64, value.data[j - 1]);
    printf("... (hex; %zu fractional limbs; workspace %zu B; object %zu B)\n", n, bytes, object_bytes);
    uint64_t fingerprint = UINT64_C(1469598103934665603);
    for (size_t j = 0; j < value.count; ++j) {
        fingerprint ^= value.data[j];
        fingerprint *= UINT64_C(1099511628211);
    }
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    sbn3_arena_stats stats;
    sbn3_arena_get_stats(arena, &stats);
    printf("{\"kind\":\"run\",\"operation\":\"%s\",\"argument\":%" PRIu64
           ",\"fractional_limbs\":%zu,\"workers\":%u,\"storage_bytes\":%zu,\"object_bytes\":%zu,"
           "\"components\":%u,\"terms\":%" PRIu64 ",\"blocks\":%u,\"query_ns\":%" PRIu64
           ",\"setup_ns\":%" PRIu64 ",\"bind_ns\":%" PRIu64 ",\"execute_ns\":%" PRIu64
           ",\"bind_execute_ns\":%" PRIu64 ",\"maxrss_bytes\":%" PRIu64 ",\"locked_peak_bytes\":%zu,"
           "\"fingerprint\":\"%016" PRIx64 "\"}\n",
           argv[1], argument, n, workers, bytes, object_bytes, is_log ? log_info.components : 1,
           is_log ? 0 : arc_info.terms, is_log ? 0 : arc_info.blocks, query_end - query_start,
           bind_start - query_end, execute_start - bind_start, execute_end - execute_start,
           execute_end - bind_start, (uint64_t)usage.ru_maxrss * 1024, stats.peak_resident_bytes, fingerprint);
    sbn3_formula_metrics phases = {0};
    sbn3_formula_sum_metrics sum_phases = {0};
    if (is_log) {
        sbn3_formula_sum_get_metrics(logarithm, &sum_phases);
        phases = sum_phases.components;
    } else {
        sbn3_formula_get_metrics(arc, &phases);
    }
    printf("{\"kind\":\"phases\",\"prepare_ns\":%" PRIu64 ",\"series_ns\":%" PRIu64
           ",\"merge_ns\":%" PRIu64 ",\"terminal_ns\":%" PRIu64
           ",\"component_bind_ns\":%" PRIu64 ",\"combine_ns\":%" PRIu64 "}\n",
           phases.prepare_ns, phases.series_ns, phases.merge_ns, phases.terminal_ns,
           sum_phases.component_bind_ns, sum_phases.combine_ns);
    if (is_log)
        sbn3_formula_sum_unbind(logarithm);
    else
        sbn3_formula_unbind(arc);
    sbn3_team_destroy(team);
    sbn3_arena_release(arena, &control);
    sbn3_arena_destroy(arena);
    free(object);
    return 0;
}
