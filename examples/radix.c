// Radix conversion example and timing driver: random or file input -> digits (and back).
//   radix --limbs N [--kind fraction|integer|float] [--base B] [--digits D] [--mode exact|enclosed]
//         [--workers W] [--repeats R] [--alphabet] [--output FILE] [--query]
#define _POSIX_C_SOURCE 200809L
#include "sbn3/radix.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static size_t up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }
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
    fprintf(stderr, "resource error: %s errno=%d need=%zu have=%zu\n", e->where, e->system_error, e->need, e->have);
    return 1;
}
static uint64_t fingerprint(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t j = 0; j < n; ++j)
        h = (h ^ p[j]) * UINT64_C(1099511628211);
    return h;
}
int main(int argc, char **argv) {
    size_t limbs = 65536, digits = SIZE_MAX;
    unsigned base = 10, workers = 16, repeats = 1, alphabet = 0, roundtrip = 0, repeated = 0;
    const char *kind = "fraction", *mode_name = "exact", *output = NULL, *input = NULL, *pattern = "random";
    int query_only = 0;
    for (int j = 1; j < argc; ++j) {
        if (!strcmp(argv[j], "--query")) { query_only = 1; continue; }
        if (!strcmp(argv[j], "--alphabet")) { alphabet = 1; continue; }
        if (!strcmp(argv[j], "--roundtrip")) { roundtrip = 1; continue; }
        if (!strcmp(argv[j], "--repeated")) { repeated = 1; continue; }
        if (!strcmp(argv[j], "--help") || j + 1 == argc) {
            puts("radix --limbs N [--kind fraction|integer|float] [--base B] [--digits D] [--mode exact|enclosed] "
                 "[--workers W] [--repeats R] [--alphabet] [--roundtrip] [--repeated] [--pattern P] [--output FILE] [--query]");
            return j + 1 == argc ? 2 : 0;
        }
        const char *key = argv[j], *value = argv[++j];
        if (!strcmp(key, "--limbs")) limbs = number(value);
        else if (!strcmp(key, "--base")) base = (unsigned)number(value);
        else if (!strcmp(key, "--digits")) digits = number(value);
        else if (!strcmp(key, "--workers")) workers = (unsigned)number(value);
        else if (!strcmp(key, "--repeats")) repeats = (unsigned)number(value);
        else if (!strcmp(key, "--kind")) kind = value;
        else if (!strcmp(key, "--mode")) mode_name = value;
        else if (!strcmp(key, "--output")) output = value;
        else if (!strcmp(key, "--input")) input = value;
        else if (!strcmp(key, "--pattern")) pattern = value; /* random | zero | ones | half: structured inputs */
        else return 2;
    }
    // --input FILE: a constant as the e/pi examples write it (16 byte header, then the n + 1 limbs of
    // floor(c * 2^(64 n))); converted as the big float it is.
    uint64_t *file_limbs = NULL;
    if (input) {
        FILE *f = fopen(input, "rb");
        if (!f) {
            perror(input);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        const long size = ftell(f);
        if (size < 32 || (size - 16) % 8)
            return 2;
        limbs = (size_t)(size - 16) / 8;
        file_limbs = aligned_alloc(64, up(limbs * 8, 64));
        fseek(f, 16, SEEK_SET);
        if (!file_limbs || fread(file_limbs, 8, limbs, f) != limbs)
            return 1;
        fclose(f);
        kind = "float";
    }
    if (!limbs || base < 2 || base > 64 || !workers || workers > 32 || !repeats)
        return 2;
    sbn3_format_spec spec = {0};
    spec.base = base;
    spec.limbs = limbs;
    spec.mode = !strcmp(mode_name, "enclosed") ? SBN3_RADIX_ENCLOSED : SBN3_RADIX_EXACT;
    const double digits_per_limb = 64.0 / log2((double)base);
    if (!strcmp(kind, "integer")) {
        spec.exponent2 = 0;
        spec.fraction_digits = digits == SIZE_MAX ? 0 : digits;
    } else if (!strcmp(kind, "float")) { // one integer limb, the rest fraction: a constant
        spec.exponent2 = -(int64_t)(64 * (limbs - 1));
        spec.fraction_digits = digits == SIZE_MAX ? (uint64_t)((double)(limbs - 1) * digits_per_limb) : digits;
    } else {
        spec.exponent2 = -(int64_t)(64 * limbs);
        spec.fraction_digits = digits == SIZE_MAX ? (uint64_t)((double)limbs * digits_per_limb) : digits;
    }
    sbn3_radix_options options = {0};
    options.workers = workers;
    options.repeated = repeated;
    if (alphabet) {
        options.use_alphabet = 1;
        sbn3_radix_alphabet(base, options.alphabet);
    }
    sbn3_format_plan plan;
    sbn3_format_info info;
    const uint64_t query_start = tick();
    const sbn3_query_result rc = sbn3_format_query(&spec, &options, &plan, &info);
    const uint64_t query_ns = tick() - query_start;
    if (rc != SBN3_SUPPORTED) {
        fprintf(stderr, "format query=%u required=%zu\n", rc, info.storage_bytes);
        return 1;
    }
    // --roundtrip: parse the digits back (integer digits then fraction digits) to the input's bit count.
    sbn3_parse_spec pspec = {0};
    sbn3_parse_plan pplan;
    sbn3_parse_info pinfo = {0};
    size_t format_storage = info.storage_bytes;
    if (roundtrip) {
        pspec.base = base;
        pspec.integer_digits = info.integer_digits;
        pspec.fraction_digits = info.fraction_digits;
        pspec.fraction_bits = spec.exponent2 < 0 ? (uint64_t)-spec.exponent2 : 0;
        const uint64_t parse_query_start = tick();
        const sbn3_query_result prc = sbn3_parse_query(&pspec, &options, &pplan, &pinfo);
        const uint64_t parse_query_ns = tick() - parse_query_start;
        if (prc != SBN3_SUPPORTED) {
            fprintf(stderr, "parse query=%u required=%zu\n", prc, pinfo.storage_bytes);
            return 1;
        }
        printf("{\"kind\":\"plan\",\"conversion\":\"parse\",\"base\":%u,\"integer_digits\":%" PRIu64 ",\"fraction_digits\":%" PRIu64
               ",\"fraction_bits\":%" PRIu64 ",\"limbs\":%zu,\"storage_bytes\":%zu,\"table_bytes\":%zu,\"workspace_bytes\":%zu,"
               "\"divide_bytes\":%zu,\"query_ns\":%" PRIu64 "}\n",
               base, pspec.integer_digits, pspec.fraction_digits, pspec.fraction_bits, pinfo.limbs, pinfo.storage_bytes,
               pinfo.table_bytes, pinfo.workspace_bytes, pinfo.divide_bytes, parse_query_ns);
        if (pinfo.storage_bytes > info.storage_bytes)
            info.storage_bytes = pinfo.storage_bytes;
        if (pinfo.storage_alignment > info.storage_alignment)
            info.storage_alignment = pinfo.storage_alignment;
    }
    (void)format_storage;
    const size_t control_bytes = sbn3_team_storage_bytes();
    const size_t admission = up(info.storage_bytes, 4096) + up(control_bytes, 4096) +
                             sbn3_team_stack_resident_bytes(workers) + (1u << 20);
    printf("{\"kind\":\"plan\",\"conversion\":\"format\",\"input\":\"%s\",\"base\":%u,\"limbs\":%zu,\"exponent2\":%" PRId64
           ",\"mode\":\"%s\",\"workers\":%u,\"integer_digits\":%" PRIu64 ",\"fraction_digits\":%" PRIu64
           ",\"storage_bytes\":%zu,\"table_bytes\":%zu,\"value_bytes\":%zu,\"workspace_bytes\":%zu,\"divide_bytes\":%zu,"
           "\"required_lock_limit_bytes\":%zu,\"query_ns\":%" PRIu64 ",\"plan_id\":\"%016" PRIx64 "\"}\n",
           kind, base, limbs, spec.exponent2, mode_name, workers, info.integer_digits, info.fraction_digits,
           info.storage_bytes, info.table_bytes, info.value_bytes, info.workspace_bytes, info.divide_bytes, admission,
           query_ns, info.plan_id);
    fflush(stdout);
    if (query_only)
        return 0;
    sbn3_arena_config ac = {info.storage_bytes + sbn3_team_stack_virtual_bytes(workers) + 2 * info.storage_alignment +
                                (16u << 20),
                            admission};
    sbn3_arena *arena = NULL;
    sbn3_team *team = NULL;
    sbn3_error error = {0};
    sbn3_lease control = {0};
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
    const uintptr_t arena_base = (uintptr_t)control.data;
    const size_t offset =
        up(arena_base + tc.stack_offset + sbn3_team_stack_virtual_bytes(workers) + 4096, info.storage_alignment) - arena_base;
    if (sbn3_arena_prepare(arena, offset, info.storage_bytes, &error) != SBN3_OK)
        return resource_error(&error);
    uint64_t *m = file_limbs ? file_limbs : aligned_alloc(64, up(limbs * 8, 64));
    unsigned char *out = aligned_alloc(64, up(info.digit_bytes + 64, 64));
    if (!m || !out)
        return 1;
    uint64_t x = UINT64_C(0x9e3779b97f4a7c15);
    for (size_t j = 0; j < limbs && !file_limbs; ++j) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        m[j] = !strcmp(pattern, "random") ? x : !strcmp(pattern, "ones") ? ~UINT64_C(0) : 0;
    }
    if (!file_limbs && !strcmp(pattern, "half"))
        m[limbs - 1] = UINT64_C(1) << 63;
    memset(out, 0, info.digit_bytes);
    for (unsigned rep = 0; rep < repeats; ++rep) {
        const uint64_t start = tick();
        sbn3_format_binding *binding = NULL;
        sbn3_format_bind(&plan, arena, offset, team, &binding);
        const uint64_t ready = tick();
        sbn3_format_result result;
        sbn3_format_execute(binding, (sbn3_const_limbs){m, limbs}, out, &result);
        const uint64_t end = tick();
        sbn3_format_execute(binding, (sbn3_const_limbs){m, limbs}, out, &result);
        const uint64_t again = tick();
        sbn3_format_unbind(binding);
        sbn3_arena_stats stats;
        sbn3_arena_get_stats(arena, &stats);
        printf("{\"kind\":\"run\",\"repeat\":%u,\"bind_ns\":%" PRIu64 ",\"execute_ns\":%" PRIu64 ",\"bind_execute_ns\":%" PRIu64
               ",\"second_execute_ns\":%" PRIu64 ",\"integer_first\":%" PRIu64 ",\"certified_fraction_digits\":%" PRIu64
               ",\"exact_fallback\":%u,\"locked_peak_bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}\n",
               rep, ready - start, end - ready, end - start, again - end, result.integer_first, result.fraction_digits,
               result.exact_fallback, stats.peak_resident_bytes,
               fingerprint(out + result.integer_first, info.fraction_offset - result.integer_first) ^
                   fingerprint(out + info.fraction_offset, result.fraction_digits));
        fflush(stdout);
        if (roundtrip) {
            uint64_t *back = aligned_alloc(64, up(pinfo.limbs * 8, 64));
            if (!back)
                return 1;
            const uint64_t pstart = tick();
            sbn3_parse_binding *pb = NULL;
            sbn3_parse_bind(&pplan, arena, offset, team, &pb);
            const uint64_t pready = tick();
            sbn3_parse_result presult;
            sbn3_parse_execute(pb, out, (sbn3_limbs){back, pinfo.limbs}, &presult);
            const uint64_t pend = tick();
            sbn3_parse_unbind(pb);
            // back <= input, and they agree to within the truncation of the digits (exactly for integers)
            size_t differing = 0;
            int ordered = 1;
            if (presult.valid) {
                size_t top = limbs < pinfo.limbs ? limbs : pinfo.limbs;
                for (size_t j = top; j < pinfo.limbs; ++j)
                    ordered &= back[j] == 0;
                size_t j = top;
                while (j-- > 0 && back[j] == m[j]) {}
                differing = j + 1 == 0 ? 0 : j + 1; // limbs at and below the first difference
                if (differing)
                    ordered &= back[differing - 1] < m[differing - 1];
            }
            printf("{\"kind\":\"parse\",\"repeat\":%u,\"bind_ns\":%" PRIu64 ",\"execute_ns\":%" PRIu64 ",\"bind_execute_ns\":%" PRIu64
                   ",\"valid\":%u,\"exact_fallback\":%u,\"differing_low_limbs\":%zu,\"ordered\":%d}\n",
                   rep, pready - pstart, pend - pready, pend - pstart, presult.valid, presult.exact_fallback, differing, ordered);
            fflush(stdout);
            free(back);
            if (!presult.valid || !ordered || (spec.exponent2 >= 0 && differing))
                return 1;
        }
        if (output && rep + 1 == repeats) {
            FILE *f = fopen(output, "wb");
            if (!f) {
                perror(output);
                return 1;
            }
            const unsigned char zero = alphabet ? options.alphabet[0] : 0;
            if (result.integer_first == info.integer_digits)
                fputc(alphabet ? zero : '0', f);
            for (size_t j = result.integer_first; j < info.integer_digits; ++j)
                fputc(alphabet ? out[j] : "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/"[out[j]], f);
            if (result.fraction_digits)
                fputc('.', f);
            for (size_t j = 0; j < result.fraction_digits; ++j) {
                const unsigned char d = out[info.fraction_offset + j];
                fputc(alphabet ? d : "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/"[d], f);
            }
            fputc('\n', f);
            fclose(f);
        }
    }
    sbn3_team_destroy(team);
    sbn3_arena_release(arena, &control);
    sbn3_arena_destroy(arena);
    free(m);
    free(out);
    return 0;
}
