#define _POSIX_C_SOURCE 200809L
#include "sbn3/constants.h"
#ifdef SBN3_PI_EXAMPLE
#include "sbn3/verify.h"
#define sbn3_e_options sbn3_pi_options
#define sbn3_e_plan sbn3_pi_plan
#define sbn3_e_info sbn3_pi_info
#define sbn3_e_binding sbn3_pi_binding
#define sbn3_e_metrics sbn3_pi_metrics
#define sbn3_e_query sbn3_pi_query
#define sbn3_e_bind sbn3_pi_bind
#define sbn3_e_execute sbn3_pi_execute
#define sbn3_e_execute_inplace sbn3_pi_execute_inplace
#define sbn3_e_get_metrics sbn3_pi_get_metrics
#define sbn3_e_unbind sbn3_pi_unbind
#endif
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static size_t up(size_t n, size_t a) {
    return (n + a - 1) & ~(a - 1);
}
static uint64_t tick(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}
static size_t number(const char *s) {
    char *end = NULL;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || !*s || *end) {
        fprintf(stderr, "invalid number: %s\n", s);
        exit(2);
    }
    return (size_t)n;
}
static int resource_error(const sbn3_error *e) {
    fprintf(stderr, "resource error: %s errno=%d need=%zu have=%zu\n", e->where, e->system_error, e->need,
            e->have);
    return 1;
}
static uint64_t fingerprint(const uint64_t *p, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t j = 0; j < n; ++j)
        for (unsigned k = 0; k < 8; ++k)
            h = (h ^ ((p[j] >> (8 * k)) & 255)) * UINT64_C(1099511628211);
    return h;
}
int main(int argc, char **argv) {
    size_t n = 65536, budget = 0;
#ifdef SBN3_PI_EXAMPLE
    size_t minimum_block_limbs = 0;
    int verify_bbp = 0;
    unsigned bbp_window = 192;
#endif
    unsigned workers = 16, batch = 8, repeats = 1;
#ifdef SBN3_PI_EXAMPLE
    unsigned serial = 0;
#else
    unsigned serial = 1;
#endif
    int query = 0;
    const char *output = NULL;
    for (int j = 1; j < argc; ++j) {
#ifdef SBN3_PI_EXAMPLE
        if (!strcmp(argv[j], "--verify-bbp")) {
            verify_bbp = 1;
            continue;
        }
#endif
        if (!strcmp(argv[j], "--query")) {
            query = 1;
            continue;
        }
        if (!strcmp(argv[j], "--help")) {
            puts("constant --limbs N | --digits D --workers W [--leaf-terms K] [--serial-levels K] "
                 "[--repeats R] [--budget-mib "
                 "M] [--query] [--output FILE]");
#ifdef SBN3_PI_EXAMPLE
            puts("pi additionally: --verify-bbp [--bbp-window 192|384]; separate verification timing");
#endif
            return 0;
        }
        if (j + 1 == argc)
            return 2;
        const char *key = argv[j], *value = argv[++j];
        if (!strcmp(key, "--limbs"))
            n = number(value);
        else if (!strcmp(key, "--digits")) {
            const __uint128_t bits = (__uint128_t)number(value) * UINT64_C(3321928094887363);
            n = (size_t)((bits + UINT64_C(64000000000000000) - 1) / UINT64_C(64000000000000000));
        }
#ifdef SBN3_PI_EXAMPLE
        else if (!strcmp(key, "--minimum-block-limbs"))
            minimum_block_limbs = number(value);
        else if (!strcmp(key, "--bbp-window")) {
            const size_t bits = number(value);
            if (bits != 192 && bits != 384)
                return 2;
            bbp_window = (unsigned)bits;
        }
#endif
        else if (!strcmp(key, "--workers"))
            workers = (unsigned)number(value);
        else if (!strcmp(key, "--leaf-terms"))
            batch = (unsigned)number(value);
        else if (!strcmp(key, "--serial-levels"))
            serial = (unsigned)number(value);
        else if (!strcmp(key, "--repeats"))
            repeats = (unsigned)number(value);
        else if (!strcmp(key, "--budget-mib")) {
            const size_t m = number(value);
            if (m > SIZE_MAX / (1u << 20))
                return 2;
            budget = m * (1u << 20);
        } else if (!strcmp(key, "--output"))
            output = value;
        else
            return 2;
    }
    if (!workers || workers > 32 || !repeats || repeats > 1000)
        return 2;
#ifdef SBN3_PI_EXAMPLE
    if (verify_bbp && (n < 2 || n > ((uint64_t)1 << 48) / 64))
        return 2;
#endif
#ifdef SBN3_PI_EXAMPLE
    sbn3_e_options options = {{workers, batch, serial, 0, 0}, budget, 128, 0, minimum_block_limbs};
#else
    sbn3_e_options options = {{workers, batch, serial, 0, 0}, budget};
#endif
    sbn3_e_plan plan;
    sbn3_e_info info;
    const uint64_t query_start = tick();
    const sbn3_query_result rc = sbn3_e_query(n, &options, &plan, &info);
    const uint64_t query_ns = tick() - query_start;
    if (rc != SBN3_SUPPORTED) {
        fprintf(stderr, "constant query=%u required=%zu\n", rc, info.storage_bytes);
        return 1;
    }
    const size_t control_bytes = sbn3_team_storage_bytes();
    const size_t result_bytes = 0; // result lives in the binding until after output
    const size_t admission = up(info.storage_bytes, 4096) + up(result_bytes, 4096) + up(control_bytes, 4096) +
                             sbn3_team_stack_resident_bytes(workers) + (1u << 20);
#ifdef SBN3_PI_EXAMPLE
    printf("{\"kind\":\"plan\",\"constant\":\"pi\",\"fractional_limbs\":%zu,\"workers\":%u,\"terms\":%" PRIu64
           ",\"blocks\":%u,\"binding_bytes\":%zu,\"psr_value_bytes\":%zu,\"psr_pool_bytes\":%zu,\"terminal_"
           "storage_bytes\":%zu,\"required_lock_limit_bytes\":%zu,\"query_ns\":%" PRIu64
           ",\"plan_id\":\"%016" PRIx64 "\"}\n",
           n, workers, info.terms, info.blocks, info.storage_bytes, info.psr_value_bytes, info.psr_pool_bytes,
           info.terminal_storage_bytes, admission, query_ns, info.plan_id);
#else
    printf("{\"kind\":\"plan\",\"constant\":\"e\",\"fractional_limbs\":%zu,\"workers\":%u,\"terms\":%" PRIu64
           ",\"leaf_terms\":%u,\"serial_prefix\":%u,\"binding_bytes\":%zu,\"series_prepared_bytes\":%zu,"
           "\"series_workspace_bytes\":%zu,\"finish_storage_bytes\":%zu,\"required_lock_limit_bytes\":%zu,"
           "\"query_ns\":%" PRIu64 ",\"plan_id\":\"%016" PRIx64 "\"}\n",
           n, workers, info.terms, batch, info.serial_prefix, info.storage_bytes, info.series_prepared_bytes,
           info.series_workspace_bytes, info.finish_storage_bytes, admission, query_ns, info.plan_id);
#endif
    fflush(stdout);
    if (budget && admission > budget) {
        fprintf(stderr, "overall budget: need=%zu have=%zu\n", admission, budget);
        return 1;
    }
    if (query)
        return 0;
    sbn3_arena_config ac = {info.storage_bytes + result_bytes + sbn3_team_stack_virtual_bytes(workers) +
                                2 * info.storage_alignment + (16u << 20),
                            admission};
    sbn3_arena *arena = NULL;
    sbn3_team *team = NULL;
    sbn3_error error = {0};
    sbn3_lease control = {0}, result = {0};
    if (sbn3_arena_create(&ac, &arena, &error) != SBN3_OK)
        return resource_error(&error);
    if (sbn3_arena_prepare(arena, 0, control_bytes, &error) != SBN3_OK)
        return resource_error(&error);
    sbn3_arena_acquire(arena, 0, control_bytes, &control);
    sbn3_team_config tc = {0};
    tc.workers = workers;
    tc.pin_threads = 1;
    tc.stack_offset = 65536;
    for (unsigned j = 0; j < 32; ++j)
        tc.cpu_ids[j] = -1;
    if (sbn3_team_create(arena, &control, &tc, &team, &error) != SBN3_OK)
        return resource_error(&error);
    const uintptr_t base = (uintptr_t)control.data;
    const size_t offset =
        up(base + tc.stack_offset + sbn3_team_stack_virtual_bytes(workers) + 4096, info.storage_alignment) -
        base;
    if (sbn3_arena_prepare(arena, offset, info.storage_bytes, &error) != SBN3_OK)
        return resource_error(&error);
    sbn3_e_binding *held = NULL;
    for (unsigned rep = 0; rep < repeats; ++rep) {
        if (held) {
            sbn3_e_unbind(held);
            held = NULL;
        }
        const uint64_t start = tick();
        sbn3_e_binding *binding = NULL;
        sbn3_e_bind(&plan, arena, offset, team, &binding);
        const uint64_t ready = tick();
        const sbn3_const_limbs inside = sbn3_e_execute_inplace(binding);
        result.data = (void *)inside.data;
        result.bytes = inside.count * 8;
        held = binding;
        const uint64_t end = tick();
        sbn3_e_metrics metrics;
        sbn3_e_get_metrics(binding, &metrics);
        sbn3_arena_stats stats;
        sbn3_arena_get_stats(arena, &stats);
        const uint64_t *v = result.data;
#ifdef SBN3_PI_EXAMPLE
        printf("{\"kind\":\"run\",\"repeat\":%u,\"fractional_limbs\":%zu,\"workers\":%u,\"bind_ns\":%" PRIu64
               ",\"compute_ns\":%" PRIu64 ",\"bind_execute_ns\":%" PRIu64 ",\"psr_ns\":%" PRIu64
               ",\"finite_ns\":%" PRIu64 ",\"merge_ns\":%" PRIu64 ",\"prepare_ns\":%" PRIu64
               ",\"terminal_ns\":%" PRIu64 ",\"locked_peak_bytes\":%zu,\"prefix\":\"%" PRIx64 ".%016" PRIx64
               "\",\"fnv1a64\":\"%016" PRIx64 "\"}\n",
               rep, n, workers, ready - start, end - ready, end - start, metrics.psr_ns, metrics.finite_ns,
               metrics.merge_ns, metrics.prepare_ns, metrics.terminal_ns, stats.peak_resident_bytes, v[n],
               v[n - 1], fingerprint(v, n + 1));
#else
        printf("{\"kind\":\"run\",\"repeat\":%u,\"fractional_limbs\":%zu,\"workers\":%u,\"bind_ns\":%" PRIu64
               ",\"compute_ns\":%" PRIu64 ",\"bind_execute_ns\":%" PRIu64 ",\"series_ns\":%" PRIu64
               ",\"finish_prepare_ns\":%" PRIu64 ",\"finish_ns\":%" PRIu64
               ",\"locked_peak_bytes\":%zu,\"prefix\":\"%" PRIx64 ".%016" PRIx64
               "\",\"fnv1a64\":\"%016" PRIx64 "\"}\n",
               rep, n, workers, ready - start, end - ready, end - start, metrics.series_ns,
               metrics.finish_prepare_ns, metrics.finish_ns, stats.peak_resident_bytes, v[n], v[n - 1],
               fingerprint(v, n + 1));
#endif
        fflush(stdout);

    }
#ifdef SBN3_PI_EXAMPLE
    int bbp_ok = 1;
    if (verify_bbp) {
        sbn3_bbp_result checked;
        const uint64_t before = tick();
        bbp_ok = sbn3_pi_bbp_check_tail(team, (sbn3_const_limbs){result.data, n + 1}, bbp_window, &checked);
        const uint64_t elapsed = tick() - before;
        const uint64_t *v = result.data;
        printf("{\"kind\":\"bbp_verify\",\"bit_offset\":%" PRIu64 ",\"window_bits\":%u,"
               "\"checked_bits\":128,\"terms\":%" PRIu64 ",\"error_ulps\":%" PRIu64 ",\"stable\":%u,"
               "\"expected\":\"%016" PRIx64 "%016" PRIx64 "\",\"actual\":\"%016" PRIx64 "%016" PRIx64 "\","
               "\"passed\":%s,\"verify_ns\":%" PRIu64 "}\n", checked.bit_offset, checked.window_bits,
               checked.terms, checked.error_ulps, checked.stable, checked.bits[0], checked.bits[1],
               v[1], v[0], bbp_ok ? "true" : "false", elapsed);
        fflush(stdout);
    }
#endif
    if (output) {
        FILE *f = fopen(output, "wb");
        if (!f) {
            perror(output);
            return 1;
        }
#ifdef SBN3_PI_EXAMPLE
        const unsigned char magic[8] = {'S', 'B', 'N', '3', 'P', 'I', 0, 0};
#else
        const unsigned char magic[8] = {'S', 'B', 'N', '3', 'E', 0, 0, 0};
#endif
        const uint64_t size = n;
        const int ok = fwrite(magic, 1, 8, f) == 8 && fwrite(&size, 8, 1, f) == 1 &&
                       fwrite(result.data, 8, n + 1, f) == n + 1;
        const int closed = fclose(f);
        if (!ok || closed)
            return 1;
    }
    if (held)
        sbn3_e_unbind(held);
    sbn3_team_destroy(team);
    sbn3_arena_release(arena, &control);
    sbn3_arena_destroy(arena);
#ifdef SBN3_PI_EXAMPLE
    return bbp_ok ? 0 : 1;
#else
    return 0;
#endif
}
