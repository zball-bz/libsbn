#include "product_support.hpp"
#include "sbn3/constants.h"
static void exact(const uint64_t *out, size_t n, uint64_t terms) {
    ref_int t, d, a, q, r, tmp, bits, got;
    ref_inits(t, d, a, q, r, tmp, bits, got, nullptr);
    ref_set_ui(t, 0);
    ref_set_ui(d, 1);
    for (uint64_t k = 1; k <= terms; ++k) {
        ref_mul_ui(t, t, k);
        ref_add_ui(t, t, 1);
        ref_mul_ui(d, d, k);
    }
    ref_add(a, t, d);
    ref_mul_2exp(a, a, 64 * n);
    ref_fdiv_q(q, a, d);
    ref_import(got, n + 1, -1, 8, 0, 0, out);
    assert(ref_cmp(got, q) == 0);
    // e-e_N < 1/(N*N!). Certify that this entire tail stays below the
    // next output integer, not just that the finite quotient agrees.
    ref_add_ui(tmp, q, 1);
    ref_mul(r, tmp, d);
    ref_sub(r, r, a);
    ref_mul_ui(r, r, terms);
    ref_set_ui(bits, 1);
    ref_mul_2exp(bits, bits, 64 * n);
    assert(ref_cmp(r, bits) > 0);
    ref_clears(t, d, a, q, r, tmp, bits, got, nullptr);
}
static void one(size_t n, unsigned workers, bool reference = true) {
    Fixture f(workers);
    sbn3_e_options options{{workers, 8, 0, 0, 0}, 0};
    sbn3_e_plan plan{};
    sbn3_e_info info{};
    allocation_watch_start();
    assert(sbn3_e_query(n, &options, &plan, &info) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    const auto saved = plan;
    options.memory_budget = info.storage_bytes - 1;
    sbn3_e_info rejected{};
    assert(sbn3_e_query(n, &options, &plan, &rejected) == SBN3_QUERY_CAPACITY &&
           !memcmp(&saved, &plan, sizeof plan));
    const size_t offset = up(f.base + f.cursor, info.storage_alignment) - f.base;
    f.cursor = up(offset + info.storage_bytes, 4096) + 4096;
    sbn3_error err{};
    assert(sbn3_arena_prepare(f.arena, offset, info.storage_bytes, &err) == SBN3_OK);
    auto *out = f.guarded(up(n + 1, 8));
    std::vector<uint64_t> previous(n + 1);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        sbn3_e_binding *binding = nullptr;
        allocation_watch_start();
        sbn3_e_bind(&plan, f.arena, offset, f.team, &binding);
        assert(!allocation_watch_stop());
        allocation_watch_start();
        if (repeat) {
            const auto inside = sbn3_e_execute_inplace(binding);
            assert(inside.count == n + 1);
            out = const_cast<uint64_t *>(inside.data);
        } else
            sbn3_e_execute(binding, {out, n + 1});
        assert(!allocation_watch_stop());
        sbn3_e_metrics metrics{};
        sbn3_e_get_metrics(binding, &metrics);
        assert(out[n] == 2 && out[n - 1] == 0xb7e151628aed2a6aULL);
        if (reference)
            exact(out, n, info.terms);
        if (repeat)
            assert(!memcmp(out, previous.data(), (n + 1) * 8));
        else
            memcpy(previous.data(), out, (n + 1) * 8);
        allocation_watch_start();
        sbn3_e_unbind(binding);
        assert(!allocation_watch_stop());
        printf(
            "Euler n=%zu W%u N=%llu bytes=%zu tree=%llu finish=%llu: floor/tail/no allocation/rebind PASS\n",
            n, workers, (unsigned long long)info.terms, info.storage_bytes,
            (unsigned long long)metrics.series_ns,
            (unsigned long long)(metrics.finish_prepare_ns + metrics.finish_ns));
        fflush(stdout);
    }
}
int main() {
    for (unsigned w : {1u, 3u, 16u})
        for (size_t n : {1u, 4u, 31u, 256u})
            one(n, w);
    one(2048, 16);
    one(32768, 16, false);
    one(1048576, 16, false); // large no-compaction layout and interior result view
    puts("Euler constant service gates PASS");
}
