// Shared support of the exact-division gates: the independent certificate
// (R<D and N==Q*D+R modulo two 61-bit primes, optionally the exact reference
// integers) and an allocation-watched service wrapper.
#pragma once
#include "product_support.hpp"
#include "sbn3/divrem.h"
#include <algorithm>
static uint64_t prime_after(uint64_t p) {
    p |= 1;
    while (!ref_prime61(p))
        p += 2;
    return p;
}
static const uint64_t primes[2] = {prime_after(0x1fffffffffff0001ULL), prime_after(0x1ffffffff0000123ULL)};
static void import(ref_int x, const uint64_t *a, size_t n) {
    if (n)
        ref_import(x, n, -1, 8, 0, 0, a);
    else
        ref_set_ui(x, 0);
}
// Independent certificate: N == Q*D + R and 0 <= R < D.
static void certify(const uint64_t *N, size_t nn, const uint64_t *D, size_t dn, const uint64_t *Q, size_t qn,
                    const uint64_t *R, size_t rn, bool exact) {
    assert(rn <= dn);
    // R < D by limb comparison with zero extension.
    int cmp = 0;
    for (size_t k = dn; k-- > 0 && !cmp;) {
        const uint64_t r = k < rn ? R[k] : 0;
        if (r != D[k])
            cmp = r > D[k] ? 1 : -1;
    }
    assert(cmp < 0);
    for (uint64_t p : primes) {
        const unsigned __int128 q = ref_mod_words(Q, qn, p), d = ref_mod_words(D, dn, p), r = ref_mod_words(R, rn, p),
                                n = ref_mod_words(N, nn, p);
        assert((q * d + r) % p == n);
    }
    if (exact) {
        ref_int a, b, x, y;
        ref_inits(a, b, x, y, nullptr);
        import(a, N, nn);
        import(b, D, dn);
        ref_fdiv_q(x, a, b);
        import(y, Q, qn);
        assert(ref_cmp(x, y) == 0);
        ref_mul(x, x, b);
        ref_sub(x, a, x);
        import(y, R, rn);
        assert(ref_cmp(x, y) == 0);
        ref_clears(a, b, x, y, nullptr);
    }
}
static size_t trim(const uint64_t *a, size_t n) {
    while (n && !a[n - 1])
        --n;
    return n;
}
struct Service {
    Fixture &f;
    sbn3_divrem_plan plan{};
    sbn3_divrem_info info{};
    sbn3_divrem_binding *bound = nullptr;
    size_t offset = 0;
    Service(Fixture &fixture, size_t nn, size_t dn, size_t block = 0, unsigned reuse = 0, unsigned residual = 0, unsigned algorithm = 0,
            unsigned prime_count = 0) : f(fixture) {
        sbn3_divrem_request request{nn, dn};
        sbn3_divrem_options options{sbn3_team_workers(f.team), prime_count, 0, reuse, block, residual, algorithm, 1};
        allocation_watch_start();
        const auto rc = sbn3_divrem_query(&request, &options, &plan, &info);
        assert(!allocation_watch_stop());
        if (rc != SBN3_SUPPORTED)
            fprintf(stderr, "divrem query nn=%zu dn=%zu: %u\n", nn, dn, rc);
        assert(rc == SBN3_SUPPORTED);
        const auto saved = plan;
        options.memory_budget = info.storage_bytes - 1;
        sbn3_divrem_info need{};
        assert(sbn3_divrem_query(&request, &options, &plan, &need) == SBN3_QUERY_CAPACITY &&
               need.storage_bytes == info.storage_bytes && !memcmp(&saved, &plan, sizeof plan));
        offset = up(f.base + f.cursor, info.storage_alignment) - f.base;
        f.cursor = up(offset + info.storage_bytes, 4096) + 4096;
        sbn3_error e{};
        assert(sbn3_arena_prepare(f.arena, offset, info.storage_bytes, &e) == SBN3_OK);
        bind();
    }
    void bind() {
        allocation_watch_start();
        sbn3_divrem_bind(&plan, f.arena, offset, f.team, &bound);
        assert(!allocation_watch_stop());
    }
    void prepare(const std::vector<uint64_t> &d) {
        allocation_watch_start();
        sbn3_divrem_prepare(bound, {d.data(), d.size()});
        assert(!allocation_watch_stop());
    }
    sbn3_divrem_result execute(const std::vector<uint64_t> &n, std::vector<uint64_t> &q, std::vector<uint64_t> &r) {
        sbn3_divrem_result out{};
        allocation_watch_start();
        sbn3_divrem_execute(bound, {n.data(), n.size()}, {q.data(), q.size()}, {r.data(), r.size()}, &out);
        assert(!allocation_watch_stop());
        return out;
    }
    void close() {
        if (bound) {
            allocation_watch_start();
            sbn3_divrem_unbind(bound);
            assert(!allocation_watch_stop());
            bound = nullptr;
        }
    }
    ~Service() { close(); }
};
