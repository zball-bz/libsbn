#include "product_support.hpp"
#include "series/arithmetic.hpp"
#include "series/schedule.hpp"
#include <algorithm>
#include <sys/random.h>
using namespace sbn::v3;
using namespace sbn::v3::series;
static uint64_t challenge() {
    uint64_t x;
    do {
        assert(getrandom(&x, sizeof x, 0) == sizeof x);
        x = (x & ((uint64_t(1) << 60) - 1)) | (uint64_t(1) << 60) | 1;
    } while (!ref_prime61(x));
    return x;
}
static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) {
    return __uint128_t(a) * b % q;
}
static uint64_t powmod(uint64_t a, uint64_t n, uint64_t q) {
    uint64_t r = 1;
    for (; n; n >>= 1, a = mm(a, a, q))
        if (n & 1)
            r = mm(r, a, q);
    return r;
}
static void verify(const Formula &formula, uint64_t a, uint64_t b, unsigned need,
                   const sbn3_series_values &out) {
    for (unsigned trial = 0; trial < 2; ++trial) {
        const uint64_t prime = challenge();
        uint64_t t = 0, d = 1, u = 1;
        for (uint64_t k = a; k < b; ++k) {
            uint64_t lt = 1, ld = k % prime, lu = 1;
            if (formula.kind == FormulaKind::Chudnovsky) {
                if (!k) {
                    lt = 13591409;
                    ld = 1;
                } else {
                    lu = mm(mm((6 * k - 5) % prime, (2 * k - 1) % prime, prime), (6 * k - 1) % prime, prime);
                    ld = mm(mm(mm(k % prime, k % prime, prime), k % prime, prime),
                            10939058860032000ULL % prime, prime);
                    lt = mm(lu, (__uint128_t(545140134) * k + 13591409) % prime, prime);
                    if (k & 1)
                        lt = lt ? prime - lt : 0;
                }
            }
            t = mm(t, ld, prime);
            if (formula.kind == FormulaKind::BinaryLog) {
                if (k > a)
                    t = mm(t, powmod(2, formula.radix_bits, prime), prime);
                lt = mm(lt, d, prime);
            } else if (formula.kind == FormulaKind::Chudnovsky)
                lt = mm(lt, u, prime);
            t = (t + lt) % prime;
            d = mm(d, ld, prime);
            u = mm(u, lu, prime);
        }
        const uint64_t expected[]{t, d, u};
        for (unsigned j = 0; j < 3; ++j)
            if (need & (1u << j)) {
                const auto &v = out.value[j];
                uint64_t got = ref_mod_words(v.mantissa.data, v.mantissa.size, prime);
                if (v.mantissa.negative)
                    got = got ? prime - got : 0;
                assert(got == expected[j]);
                assert(v.exponent2 == (j == 0 && formula.kind == FormulaKind::BinaryLog
                                           ? -int64_t(formula.radix_bits * (b - 1))
                                           : 0));
            }
    }
}
static void one(FormulaKind kind, uint64_t begin, uint64_t end, unsigned workers, unsigned need = 3,
                unsigned batch = 8) {
    Formula f{kind, 3};
    Fixture fixture(workers, false);
    FinitePlan plan{};
    sbn3_series_options options{workers, batch, 0, 0, 0};
    const auto formula = finite_formula(f);
    allocation_watch_start();
    assert(finite_query(formula, {begin, end}, need, options, plan) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    auto meta = fixture.allocate(plan.info.plan_bytes, plan.info.plan_alignment),
         tables = fixture.allocate(plan.info.prepared_bytes, plan.info.prepared_alignment),
         scratch = fixture.allocate(plan.info.workspace_bytes, plan.info.workspace_alignment);
    FiniteBinding bound{};
    allocation_watch_start();
    finite_prepare(plan, *fixture.arena, *fixture.team, meta, tables, scratch, bound);
    assert(!allocation_watch_stop());
    sbn3_series_values result{};
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j)) {
            auto l = fixture.allocate(plan.info.output.limbs[j] * 8, 64);
            result.value[j].mantissa = {static_cast<uint64_t *>(l.data), plan.info.output.limbs[j], 0, 0};
        }
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        allocation_watch_start();
        finite_execute(bound, result);
        assert(!allocation_watch_stop());
        verify(f, begin, end, need, result);
    }
    struct Inventory {
        const sbn3_series_plan *schedule;
        size_t requested = 0, unique = 0, references = 0, owners = 0;
    } inventory{bound.schedule};
    sbn3_series_visit(bound.schedule, [](void *ptr, const sbn3_series_stage *s, const sbn3_series_resources *) {
        auto &v = *static_cast<Inventory *>(ptr);
        for(unsigned slot=0;slot<2;++slot){
            const auto place = shared_placement(v.schedule, s->index,slot);
            v.requested += place.bytes;
            if (place.bytes) ++v.references;
            if (place.owner) { v.unique += place.bytes; ++v.owners; }
        }
    }, &inventory);
    printf("engine tables requested=%zu unique=%zu references=%zu owners=%zu\n",
           inventory.requested, inventory.unique, inventory.references, inventory.owners);
    if (kind == FormulaKind::Euler && end == 262145) {
        // An engine-owned pool outlives either algorithm binding. Construct it
        // from the first schedule's demand, then bind a fresh schedule through
        // the same generic directory used by independent ProductPrograms.
        ProductTablePool<> engine;
        auto storage = fixture.allocate(plan.info.prepared_bytes, 128);
        {
            Frame frame(*fixture.arena, storage);
            struct Prepare { ProductTablePool<> *engine; Frame *frame; } prepare{&engine, &frame};
            sbn3_series_visit(bound.schedule, [](void *ptr, const sbn3_series_stage *s, const sbn3_series_resources *) {
                if (s->leaf) return;
                const auto an = std::max({s->left.limbs[0], s->left.limbs[1], s->left.limbs[2]});
                const auto bn = std::max({s->right.limbs[0], s->right.limbs[1], s->right.limbs[2]});
                if (std::max(an, bn) <= 6) return;
                sbn3_mul_options options{}; options.workers = s->workers; options.borrow_output = 1;
                ProductProgramPlan product{};
                assert(product_program_query(an, bn, options, product) == SBN3_SUPPORTED);
                if (product_program_tables(product).bytes) {
                    auto &p = *static_cast<Prepare *>(ptr);p.engine->prepare(product, *p.frame);
                }
            }, &prepare);
        }
        auto external = formula; external.table_pool = engine.seal();
        FinitePlan second{};
        assert(finite_query(external, {begin, end}, need, options, second) == SBN3_SUPPORTED);
        assert(second.info.prepared_bytes < plan.info.prepared_bytes);
        assert(second.info.workspace_bytes == plan.info.workspace_bytes);
        auto metadata = fixture.allocate(second.info.plan_bytes, second.info.plan_alignment);
        auto local = fixture.allocate(second.info.prepared_bytes, second.info.prepared_alignment);
        FiniteBinding next{};
        finite_prepare(second, *fixture.arena, *fixture.team, metadata, local, scratch, next);
        const std::vector<unsigned char> saved(static_cast<unsigned char *>(storage.data),
                                              static_cast<unsigned char *>(storage.data) + storage.bytes);
        allocation_watch_start();finite_execute(next, result);assert(!allocation_watch_stop());
        verify(f, begin, end, need, result);
        assert(!memcmp(saved.data(), storage.data, storage.bytes));
        printf("engine pool reused by independent series binding: local=%zu versus %zu PASS\n",
               second.info.prepared_bytes, plan.info.prepared_bytes);
    }
    printf("finite arithmetic kind=%u [%llu,%llu) W%u mask=%u batch=%u T/D/U=%zu/%zu/%zu prepared=%zu "
           "scratch=%zu PASS\n",
           unsigned(kind), (unsigned long long)begin, (unsigned long long)end, workers, need, batch,
           result.value[0].mantissa.size, result.value[1].mantissa.size, result.value[2].mantissa.size,
           plan.info.prepared_bytes, plan.info.workspace_bytes);
    fflush(stdout);
}
int main() {
    for (unsigned w : {1u, 3u, 16u}) {
        for (auto kind : {FormulaKind::Euler, FormulaKind::Chudnovsky, FormulaKind::BinaryLog}) {
            const uint64_t first = kind == FormulaKind::Chudnovsky ? 0 : 1;
            for (unsigned mask = 1; mask <= (kind == FormulaKind::Chudnovsky ? 7u : 3u); ++mask)
                one(kind, first, first + 37, w, mask, 3);
            one(kind, 13, 1000, w);
            one(kind, 13, 1000, w, 3, 1);
        }
    }
    one(FormulaKind::Euler, 1, 262145, 16);
    one(FormulaKind::Chudnovsky, 0, 65536, 16);
    // Enough work per serial leaf to cross the shared-LLC FFT/Flat choice.
    // Full finite results are checked against two fresh prime challenges.
    one(FormulaKind::Chudnovsky, 0, 524288, 32);
    one(FormulaKind::BinaryLog, 1, 8193, 16);
    puts("finite exact arithmetic and fresh modular certificates PASS");
}
