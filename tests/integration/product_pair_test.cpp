#include "product_support.hpp"
#include "product/table_pool.hpp"
#include <algorithm>
using namespace sbn::v3;
struct PairCall {
    const ProductProgram *program;
    Fixture *fixture;
    sbn3_lease work;
    sbn3_const_limbs common, x, y;
    sbn3_limbs out0, out1;
    bool common_on_right;
};
static void apply(void *ptr, sbn3_team_scope *scope) {
    auto &t = *static_cast<PairCall *>(ptr);
    Frame f(*t.fixture->arena, t.work);
    product_program_pair(*t.program, f, scope, t.common, t.x, t.y, t.out0, t.out1, t.common_on_right);
}
static void one(size_t an, size_t bn, unsigned workers, unsigned algorithm, unsigned np = 0, unsigned batch = 0) {
    Fixture f(workers, false);
    sbn3_mul_options options{};
    options.workers = workers;
    options.algorithm = algorithm;
    options.prime_count = np;
    options.prime_batch = batch;
    ProductProgramPlan plan{};
    assert(product_program_query(an, bn, options, plan) == SBN3_SUPPORTED);
    assert(plan.pair_workspace_bytes);
    auto tables = f.allocate(plan.prepared_bytes, 128);
    ProductProgram program{};
    { Frame frame(*f.arena, tables); program = product_program_prepare(plan, frame); }
    ProductProgramPlan second{};
    assert(product_program_query(an - 1, bn - 1, options, second) == SBN3_SUPPORTED);
    if (!(product_program_tables(second) == product_program_tables(plan))) second = plan;
    if (np == 6) {
        auto other_options = options;
        other_options.trunk_bits = plan.info.trunk_bits - 8;
        ProductProgramPlan other{};
        if (product_program_query(an - 1, bn - 1, other_options, other) == SBN3_SUPPORTED &&
            product_program_tables(other) == product_program_tables(plan)) second = other;
    }
    assert(product_program_tables(plan).bytes);
    ProductTablePool<2> engine;
    auto roots = f.allocate(product_program_tables(plan).bytes, product_program_tables(plan).alignment);
    const void *root = nullptr;
    {
        Frame frame(*f.arena, roots);
        assert(engine.additional_bytes(plan) == product_program_tables(plan).bytes);
        allocation_watch_start();
        root = engine.prepare(plan, frame);
        const size_t used = frame.used();
        assert(engine.prepare(second, frame) == root && frame.used() == used);
        assert(!allocation_watch_stop());
        assert(!engine.additional_bytes(second));
    }
    const auto pool = engine.seal();
    auto local = f.allocate(product_program_local_bytes(second), 128);
    ProductProgram pooled{};
    { Frame frame(*f.arena, local); pooled = product_program_prepare_pooled(second, frame, pool); }
    assert(pooled.shared_prepared == root);
    const std::vector<unsigned char> saved_roots(static_cast<unsigned char *>(roots.data),
                                                static_cast<unsigned char *>(roots.data) + roots.bytes);
    const std::vector<unsigned char> saved(static_cast<unsigned char *>(tables.data),
                                         static_cast<unsigned char *>(tables.data) + tables.bytes);
    auto work = f.allocate(std::max(plan.pair_workspace_bytes, second.pair_workspace_bytes),
                           std::max(plan.info.workspace_alignment, second.info.workspace_alignment));
    auto *out0 = static_cast<uint64_t *>(f.allocate((an + bn + 8) * 8, 64).data);
    auto *out1 = static_cast<uint64_t *>(f.allocate((an + bn + 8) * 8, 64).data);
    for (unsigned right = 0; right < 2; ++right) {
        const size_t cn = (right ? bn : an) - 7, xn = (right ? an : bn) - 9, yn = xn - 3;
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            auto *common = out0;
            auto *x = out0 + up(cn, 8);
            auto *y = out1;
            for (size_t j = 0; j < cn; ++j) common[j] = random_word();
            for (size_t j = 0; j < xn; ++j) x[j] = random_word();
            for (size_t j = 0; j < yn; ++j) y[j] = random_word();
            std::vector<uint64_t> cv(common, common + cn), xv(x, x + xn), yv(y, y + yn);
            for (unsigned j = 0; j < 8; ++j) out0[an + bn + j] = out1[an + bn + j] = 0x37fd831f;
            PairCall task{repeat ? &pooled : &program, &f, work, {common, cn}, {x, xn}, {y, yn},
                          {out0, an + bn}, {out1, an + bn}, bool(right)};
            allocation_watch_start();
            sbn3_team_run(f.team, apply, &task);
            assert(!allocation_watch_stop());
            verify_product(cv.data(), cn, xv.data(), xn, out0);
            verify_product(cv.data(), cn, yv.data(), yn, out1);
            assert(!memcmp(saved.data(), tables.data, tables.bytes));
            assert(!memcmp(saved_roots.data(), roots.data, roots.bytes));
            for (unsigned j = 0; j < 8; ++j)
                assert(out0[an + bn + j] == 0x37fd831f && out1[an + bn + j] == 0x37fd831f);
        }
    }
    printf("product pair an=%zu bn=%zu W%u alg=%u np=%u bytes=%zu: consume both orientations / exact / readonly / no allocation PASS\n",
           an, bn, workers, plan.info.algorithm, plan.info.np, plan.pair_workspace_bytes);
    printf("engine pool T%u/T%u, table=%zu local=%zu bytes PASS\n", plan.info.trunk_bits, second.info.trunk_bits,
           product_program_tables(plan).bytes, product_program_local_bytes(second));
}
int main() {
    {
        sbn3_mul_options o{};o.workers=16;o.algorithm=SBN3_MUL_BAILEY;o.prime_count=5;o.trunk_bits=104;
        ProductProgramPlan p{},q{},replayed{};
        assert(product_program_query(2626496,4074157,o,p)==SBN3_SUPPORTED && p.info.full);
        sbn3_mul_options chosen{};
        assert(product_program_rebatch(p,1,q,&chosen)==SBN3_SUPPORTED);
        assert(q.info.full && q.info.arithmetic_id==p.info.arithmetic_id);
        assert(product_program_query(p.an,p.bn,chosen,replayed)==SBN3_SUPPORTED &&
               replayed.info.arithmetic_id==q.info.arithmetic_id && replayed.info.execution_id==q.info.execution_id);
        puts("near-full lattice execution-only rebatch preserves full/rowscale PASS");
    }

    for(unsigned w:{1u,16u})for(unsigned lg=15;lg<=27;++lg){
        const size_t bn=(size_t(1)<<lg)-123,an=bn*5/8;
        sbn3_mul_options o{};o.workers=w;ProductProgramPlan p{},q{};
        assert(product_program_query(an,bn,o,p)==SBN3_SUPPORTED);
        const auto *backend=backend_lookup(p.plan.opaque[1]);
        if(backend->program_options){
            const auto replay=backend->program_options(p.plan);
            assert(product_program_replay(an,bn,replay,p.plan.opaque[1],q)==SBN3_SUPPORTED);
            assert(p.info.arithmetic_id==q.info.arithmetic_id && p.info.execution_id==q.info.execution_id);
        }
    }
    puts("backend-owned replay parameters across large balanced/unbalanced towers PASS");
    for (size_t n : {128u, 310u, 4096u, 16000u, 33000u}) one(n, n - 17, 1, SBN3_MUL_PQ16);
    one(33000, 28000, 4, SBN3_MUL_PQ16);
    for (unsigned np : {4u, 5u, 6u, 7u, 8u, 9u, 10u}) {
        one(2048, 1800, 4, SBN3_MUL_FLAT, np);
        one(33000, 28000, 4, SBN3_MUL_BAILEY, np);
    }
    one(33000, 28000, 16, SBN3_MUL_BAILEY, 6);
    one(100000, 90000, 1, SBN3_MUL_FLAT, 6);
    one(125000, 110000, 4, SBN3_MUL_FLAT, 8);
    for (unsigned np : {4u, 6u, 9u, 10u}) one(131075, 95000, 32, SBN3_MUL_BAILEY, np, 1);
}
