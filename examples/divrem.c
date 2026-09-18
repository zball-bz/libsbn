/* Exact integer division: the thin schoolbook entry for a small one-shot
 * division, then the prepared service dividing several numerators by one
 * prepared divisor. Build: clang -std=c11 -O2 -Iinclude examples/divrem.c
 * build/native/libsbn_v3.a -pthread -lm -o build/native/divrem */
#include "sbn3/divrem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t state = 0x9e3779b97f4a7c15ULL;
static uint64_t word(void) { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
static size_t round_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }
static void *prepare(sbn3_arena *arena, size_t offset, size_t bytes, sbn3_lease *lease) {
    sbn3_error e = {0};
    if (sbn3_arena_prepare(arena, offset, bytes, &e) != SBN3_OK) {
        fprintf(stderr, "resource preparation: %s (OS error %d)\n", e.where, e.system_error);
        exit(1);
    }
    if (lease) { sbn3_arena_acquire(arena, offset, bytes, lease); return lease->data; }
    return NULL;
}
/* Check N == Q*D + R modulo a 61-bit prime with plain word arithmetic. */
static uint64_t mod_words(const uint64_t *a, size_t n, uint64_t p) {
    unsigned __int128 acc = 0;
    for (size_t k = n; k-- > 0;) acc = ((acc << 64) | a[k]) % p;
    return (uint64_t)acc;
}
static int identity_holds(const uint64_t *n, size_t nn, const uint64_t *d, size_t dn, const uint64_t *q, size_t qn,
                          const uint64_t *r, size_t rn) {
    const uint64_t p = 0x1fffffffffffffffULL; /* 2^61-1 */
    unsigned __int128 lhs = (unsigned __int128)mod_words(q, qn, p) * mod_words(d, dn, p) + mod_words(r, rn, p);
    return (uint64_t)(lhs % p) == mod_words(n, nn, p);
}
int main(void) {
    /* 1. One-shot small division: 40 limbs by 3 limbs, caller-owned scratch. */
    uint64_t n[40], d[3], q[38], r[3], scratch[44];
    for (size_t k = 0; k < 40; ++k) n[k] = word();
    for (size_t k = 0; k < 3; ++k) d[k] = word();
    d[2] |= 1;
    const size_t qn = sbn3_divrem_basecase(q, r, n, 40, d, 3, scratch);
    printf("basecase: quotient %zu limbs, identity %s\n", qn, identity_holds(n, 40, d, 3, q, qn, r, 3) ? "ok" : "BROKEN");

    /* 2. Prepared service: one divisor of 4096 limbs, numerators up to 8192 limbs. */
    const size_t dn = 4096, nn = 8192;
    sbn3_divrem_request request = {nn, dn};
    sbn3_divrem_options options = {0};
    options.workers = 4;
    options.reuse_hint = 4;
    options.timing = 1;
    sbn3_divrem_plan plan;
    sbn3_divrem_info info;
    if (sbn3_divrem_query(&request, &options, &plan, &info) != SBN3_SUPPORTED) return 1;
    printf("service: algorithm %u, block %zu limbs, ring %zu, storage %zu KiB (spectra %zu KiB, shared %zu KiB)\n",
           info.algorithm, info.block_limbs, info.ring_limbs, info.storage_bytes >> 10, info.spectrum_bytes >> 10,
           info.shared_bytes >> 10);
    sbn3_arena_config config = {(size_t)1 << 32, (size_t)256 << 20};
    sbn3_error error = {0};
    sbn3_arena *arena = NULL;
    if (sbn3_arena_create(&config, &arena, &error) != SBN3_OK) { fprintf(stderr, "%s\n", error.where); return 1; }
    sbn3_lease control;
    prepare(arena, 0, sbn3_team_storage_bytes(), &control);
    sbn3_team_config tc = {0};
    tc.workers = options.workers;
    tc.pin_threads = 1;
    for (int k = 0; k < 32; ++k) tc.cpu_ids[k] = -1;
    tc.stack_offset = 65536;
    sbn3_team *team = NULL;
    if (sbn3_team_create(arena, &control, &tc, &team, &error) != SBN3_OK) { fprintf(stderr, "%s\n", error.where); return 1; }
    const uintptr_t base = (uintptr_t)control.data;
    size_t offset = round_up(base + tc.stack_offset + sbn3_team_stack_virtual_bytes(tc.workers) + 4096, info.storage_alignment) - base;
    prepare(arena, offset, info.storage_bytes, NULL);
    sbn3_divrem_binding *bound = NULL;
    sbn3_divrem_bind(&plan, arena, offset, team, &bound);
    /* Caller-owned values, outside the binding's storage. */
    uint64_t *D = malloc(dn * 8), *N = malloc(nn * 8), *Q = malloc((nn - dn + 1) * 8), *R = malloc(dn * 8);
    for (size_t k = 0; k < dn; ++k) D[k] = word();
    D[dn - 1] |= 1;
    sbn3_divrem_prepare(bound, (sbn3_const_limbs){D, dn});
    int ok = 1;
    for (unsigned round = 0; round < 4; ++round) {
        const size_t count = round == 3 ? dn + 100 : nn; /* long and short numerators */
        for (size_t k = 0; k < count; ++k) N[k] = word();
        sbn3_divrem_result result;
        sbn3_divrem_execute(bound, (sbn3_const_limbs){N, count}, (sbn3_limbs){Q, nn - dn + 1}, (sbn3_limbs){R, dn}, &result);
        const int good = identity_holds(N, count, D, dn, Q, result.quotient_limbs, R, result.remainder_limbs);
        ok &= good;
        printf("execute %u: %zu limbs / %zu -> quotient %zu limbs, remainder %zu limbs, %llu corrections, identity %s\n",
               round, count, dn, result.quotient_limbs, result.remainder_limbs, (unsigned long long)result.corrections,
               good ? "ok" : "BROKEN");
    }
    sbn3_divrem_metrics metrics;
    sbn3_divrem_get_metrics(bound, &metrics);
    printf("last prepare %.3f ms, last execute %.3f ms\n", metrics.prepare_ns / 1e6, metrics.execute_ns / 1e6);
    sbn3_divrem_unbind(bound);
    free(D); free(N); free(Q); free(R);
    sbn3_team_destroy(team);
    sbn3_arena_release(arena, &control);
    sbn3_arena_destroy(arena);
    return ok ? 0 : 1;
}
