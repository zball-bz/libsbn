#include "product_support.hpp"
#include "product/program.hpp"
#include "product/cost_model.hpp"
using namespace sbn::v3;
static uint64_t digest(const sbn3_lease &l) {
    uint64_t h = 0;
    for (size_t j = 0; j < l.bytes; ++j)
        h = (h ^ static_cast<const unsigned char *>(l.data)[j]) * 1099511628211ULL;
    return h;
}
struct Task {
    ProductProgram *program;
    Fixture *fixture;
    sbn3_lease work;
    const uint64_t *a, *b;
    uint64_t *out;
    size_t an = 0, bn = 0;
    bool bounded = false, consume = false;
};
static void call(void *ptr, sbn3_team_scope *scope) {
    auto &t = *static_cast<Task *>(ptr);
    Frame f(*t.fixture->arena, t.work);
    if (t.bounded)
        product_program_bounded(*t.program, f, scope, {t.a, t.an}, {t.b, t.bn},
                                {t.out, t.program->an + t.program->bn}, t.consume);
    else
        product_program_execute(*t.program, f, scope, {t.a, t.program->an}, {t.b, t.program->bn},
                                {t.out, t.program->an + t.program->bn});
}
static void one(size_t an, size_t bn, unsigned workers, unsigned algorithm, unsigned np = 0,
                bool shared = false) {
    Fixture fixture(shared ? 2 : workers);
    sbn3_mul_options o{};
    o.workers = workers;
    o.algorithm = algorithm;
    o.prime_count = np;
    o.borrow_output = 1;
    ProductProgramPlan p{};
    allocation_watch_start();
    assert(product_program_query(an, bn, o, p) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    auto prepared = fixture.allocate(p.prepared_bytes, 128);
    memset(prepared.data, 0x6d, prepared.bytes);
    ProductProgram program{};
    allocation_watch_start();
    {
        Frame f(*fixture.arena, prepared);
        program = product_program_prepare(p, f);
    }
    assert(!allocation_watch_stop());
    const uint64_t hash = digest(prepared);
    const unsigned count = shared ? 2 : 1;
    Task tasks[2]{};
    for (unsigned j = 0; j < count; ++j) {
        auto a = fixture.allocate(an * 8, 64), b = fixture.allocate(bn * 8, 64),
             r = fixture.allocate((an + bn) * 8, 64);
        auto *ap = static_cast<uint64_t *>(a.data), *bp = static_cast<uint64_t *>(b.data);
        for (size_t k = 0; k < an; ++k)
            ap[k] = random_word();
        for (size_t k = 0; k < bn; ++k)
            bp[k] = random_word();
        tasks[j] = {&program, &fixture, fixture.allocate(p.info.workspace_bytes, p.info.workspace_alignment),
                    ap,       bp,       static_cast<uint64_t *>(r.data)};
    }
    auto action = [](void *ptr, sbn3_team_scope *scope) {
        auto *t = static_cast<Task *>(ptr);
        sbn3_team_invoke2(scope, SBN3_PARALLEL_CHILDREN, 1, call, t, call, t + 1);
    };
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        allocation_watch_start();
        sbn3_team_run(fixture.team, shared ? action : call, tasks);
        assert(!allocation_watch_stop());
        assert(digest(prepared) == hash);
        for (unsigned j = 0; j < count; ++j)
            verify_product(tasks[j].a, an, tasks[j].b, bn, tasks[j].out);
    }
    printf("prepared product an=%zu bn=%zu W%u alg=%u np=%u shared=%u: correct/readonly/no allocation PASS\n",
           an, bn, workers, p.info.algorithm, p.info.np, shared);
}
static void bounded(size_t an, size_t bn, unsigned workers, unsigned algorithm, unsigned np = 0) {
    Fixture f(workers);
    sbn3_mul_options o{};
    o.workers = workers;
    o.algorithm = algorithm;
    o.prime_count = np;
    o.borrow_output = 0;
    ProductProgramPlan plan{};
    assert(product_program_query(an, bn, o, plan) == SBN3_SUPPORTED);
    assert(plan.contract & program_bounded_inputs);
    auto table = f.allocate(plan.prepared_bytes, 128),
         work = f.allocate(plan.info.workspace_bytes, plan.info.workspace_alignment);
    ProductProgram p{};
    {
        Frame frame(*f.arena, table);
        p = product_program_prepare(plan, frame);
    }
    const auto hash = digest(table);
    const size_t na = an - 3, nb = bn - 7;
    auto result = f.allocate((an + bn + 8) * 8, 128);
    auto *out = static_cast<uint64_t *>(result.data);
    const bool consume = bool(plan.contract & program_consume_inputs);
    auto input_a = f.allocate(an * 8, 128), input_b = f.allocate(bn * 8, 128);
    auto *a = consume ? out : static_cast<uint64_t *>(input_a.data);
    auto *b = consume ? out + up(na, 8) : static_cast<uint64_t *>(input_b.data);
    std::vector<uint64_t> saved_a(na), saved_b(nb);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        if (consume && repeat) {
            a = out + up(nb, 8);
            b = out;
        }
        std::fill(out, out + an + bn + 8, UINT64_C(0x85c217ab36e490df));
        for (size_t i = 0; i < na; ++i)
            a[i] = saved_a[i] = random_word();
        for (size_t i = 0; i < nb; ++i)
            b[i] = saved_b[i] = random_word();
        Task task{&p, &f, work, a, b, out, na, nb, true, consume};
        allocation_watch_start();
        sbn3_team_run(f.team, call, &task);
        assert(!allocation_watch_stop());
        verify_product(saved_a.data(), na, saved_b.data(), nb, out);
        for (size_t i = an + bn; i < an + bn + 8; ++i)
            assert(out[i] == UINT64_C(0x85c217ab36e490df));
        assert(digest(table) == hash);
    }
    printf("bounded program an=%zu bn=%zu W%u alg=%u np=%u consume=%u PASS\n", an, bn, workers,
           unsigned(p.backend->id), np, consume);
}
int main() {
    sbn3_mul_options smt{}; smt.workers = 32;
    assert(cost_model::small_domain({65536, 65536}, smt));
    ProductProgramPlan smt_plan{};
    assert(product_program_query(65536, 65536, smt, smt_plan) == SBN3_SUPPORTED);
    assert(smt_plan.info.workers == 32);
    assert(cost_model::small_ntt(smt_plan.info).evidence == cost_model::Evidence::Transferred);
    bounded(32, 23, 1, SBN3_MUL_SCALAR);
    bounded(256, 137, 1, SBN3_MUL_U52);
    bounded(8193, 513, 1, SBN3_MUL_U52); // streamed, disjoint and shortened runtime inputs
    bounded(8192, 6845, 1, SBN3_MUL_PQ16);
    bounded(32768, 28001, 16, SBN3_MUL_PQ16);
    for (unsigned np : {4u, 5u, 6u, 7u, 8u, 9u, 10u})
        bounded(8192, 7903, 1, SBN3_MUL_FLAT, np);
    bounded(131072, 98305, 16, SBN3_MUL_BAILEY, 6);
    one(3, 2, 1, SBN3_MUL_SCALAR, 0, true);
    one(256, 137, 1, SBN3_MUL_U52, 0, true);
    one(8192, 6845, 1, SBN3_MUL_PQ16, 0, true);
    one(8192, 8192, 16, SBN3_MUL_PQ16);
    one(32768, 28001, 1, SBN3_MUL_FLAT, 6, true);
    for (unsigned np : {4u, 5u, 6u, 7u, 8u, 9u, 10u})
        one(8192, 7903, 1, SBN3_MUL_FLAT, np);
    one(131072, 98305, 16, SBN3_MUL_BAILEY, 6);
    one(524288, 400009, 16, SBN3_MUL_AUTO);
    one(65536, 65536, 32, SBN3_MUL_AUTO);
    one(262144, 196609, 32, SBN3_MUL_AUTO);
    one(1048576, 786433, 32, SBN3_MUL_AUTO);
    puts("prepared product service gates PASS");
}
