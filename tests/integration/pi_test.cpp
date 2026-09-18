#include "product_support.hpp"
#include "sbn3/constants.h"
static void arctan(ref_number *result, unsigned q, size_t words) {
    ref_int power, term, divisor;
    ref_inits(power, term, divisor, nullptr);
    ref_set_ui(power, 1);
    ref_mul_2exp(power, power, 64 * words);
    ref_set_ui(divisor, q);
    ref_fdiv_q(power, power, divisor);
    ref_set_ui(result, 0);
    for (uint64_t k = 0; ref_sgn(power); ++k) {
        ref_set_ui(divisor, 2 * k + 1);
        ref_fdiv_q(term, power, divisor);
        if (k & 1)
            ref_sub(result, result, term);
        else
            ref_add(result, result, term);
        ref_set_ui(divisor, q * q);
        ref_fdiv_q(power, power, divisor);
    }
    ref_clears(power, term, divisor, nullptr);
}
static void machin(const uint64_t *data, size_t n) {
    ref_int a, b, want, got;
    ref_inits(a, b, want, got, nullptr);
    arctan(a, 5, n + 3);
    arctan(b, 239, n + 3);
    ref_mul_ui(a, a, 16);
    ref_mul_ui(b, b, 4);
    ref_sub(want, a, b);
    ref_fdiv_q_2exp(want, want, 192);
    ref_import(got, n + 1, -1, 8, 0, 0, data);
    assert(ref_cmp(want, got) == 0);
    ref_clears(a, b, want, got, nullptr);
}
static void one(size_t n, unsigned workers, bool reference = true) {
    Fixture f(workers, false);
    sbn3_pi_options options{{workers, 8, 0, 0, 0}, 0, 128, 0, 0};
    sbn3_pi_plan plan{};
    sbn3_pi_info info{};
    allocation_watch_start();
    assert(sbn3_pi_query(n, &options, &plan, &info) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    const auto saved = plan;
    options.memory_budget = 1; // every complete recipe exceeds this budget
    sbn3_pi_info rejected{};
    assert(sbn3_pi_query(n, &options, &plan, &rejected) == SBN3_QUERY_CAPACITY &&
           !memcmp(&saved, &plan, sizeof plan));
    const size_t offset = up(f.base + f.cursor, info.storage_alignment) - f.base;
    f.cursor = up(offset + info.storage_bytes, 4096) + 4096;
    sbn3_error error{};
    assert(sbn3_arena_prepare(f.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    auto output = f.allocate((n + 1) * 8, 64);
    auto *out = static_cast<uint64_t *>(output.data);
    std::vector<uint64_t> previous(n + 1);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        sbn3_pi_binding *binding = nullptr;
        allocation_watch_start();
        sbn3_pi_bind(&plan, f.arena, offset, f.team, &binding);
        assert(!allocation_watch_stop());
        allocation_watch_start();
        if (repeat) {
            const auto result = sbn3_pi_execute_inplace(binding);
            assert(result.count == n + 1 && !(uintptr_t(result.data)&7));
            const auto address=reinterpret_cast<uintptr_t>(result.data),begin=f.base+offset;
            assert(address>=begin && address-begin<=info.storage_bytes && result.count*8<=info.storage_bytes-(address-begin));
            assert(!f.arena->unleased(address-f.base,result.count*8));
            out = const_cast<uint64_t *>(result.data);
        } else
            sbn3_pi_execute(binding, {out, n + 1});
        assert(!allocation_watch_stop());
        assert(out[n] == 3 && out[n - 1] == 0x243f6a8885a308d3ULL);
        if (reference)
            machin(out, n);
        if (repeat)
            assert(!memcmp(out, previous.data(), (n + 1) * 8));
        else
            memcpy(previous.data(), out, (n + 1) * 8);
        sbn3_pi_metrics m{};
        sbn3_pi_get_metrics(binding, &m);
        allocation_watch_start();
        sbn3_pi_unbind(binding);
        assert(!allocation_watch_stop());
        if(repeat)assert(f.arena->unleased(reinterpret_cast<uintptr_t>(out)-f.base,(n+1)*8));
        printf("pi n=%zu W%u N=%llu blocks=%u bytes=%zu PSR=%llu terminal=%llu reference=%u PASS\n", n,
               workers, (unsigned long long)info.terms, info.blocks, info.storage_bytes,
               (unsigned long long)m.psr_ns, (unsigned long long)m.terminal_ns, unsigned(reference));
        fflush(stdout);
    }
}
int main() {
    for (unsigned w : {1u, 3u, 16u})
        for (size_t n : {1u, 7u, 64u, 512u})
            one(n, w);
    one(2048, 16);
    one(16384, 16, false);
    one((1u << 20) + 7, 32, false); // parallel terminal value passes and exact rebinding
    puts("pi complete PSR/rsqrt/division/Machin/no allocation/rebind gates PASS");
}
