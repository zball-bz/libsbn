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
static void one(size_t an, size_t bn, unsigned workers, unsigned algorithm, unsigned np = 0, unsigned batch = 0,int T=0,unsigned lgC=0,unsigned lgM=0) {
    Fixture f(workers, false);
    sbn3_mul_options options{};
    options.workers = workers;
    options.algorithm = algorithm;
    options.prime_count = np;
    options.prime_batch = batch;
    options.trunk_bits=T;options.column_log2=lgC;options.row_log2=lgM;
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
        const size_t cn = (right ? bn : an) - 7, xn = (right ? an : bn) - 9;
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            const size_t yn = repeat ? xn * 5 / 8 : xn - 3;
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
            const bool ok0=ref_product_equal(cv.data(),cn,xv.data(),xn,out0,cn+xn);
            const bool ok1=ref_product_equal(cv.data(),cn,yv.data(),yn,out1,cn+yn);
            if(!ok0||!ok1)fprintf(stderr,"pair mismatch an%zu bn%zu W%u alg%u NP%u repeat%u right%u first%d second%d live%zu/%zu/%zu T%u C%zu M%zu L%zu full%u\n",
                an,bn,workers,algorithm,np,repeat,right,ok0,ok1,cn,xn,yn,plan.info.trunk_bits,plan.info.C,plan.info.M2,plan.info.lbv,plan.info.full);
            assert(ok0&&ok1);
            const auto&selected=repeat?pooled:program;
            if(selected.backend->id>=4&&selected.backend->id<=10)
                for(size_t j=cn+yn;j<selected.an+selected.bn;++j)assert(out1[j]==0);
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
static void program_recipes(){
    for(bool cyclic:{false,true})for(unsigned pattern:{0u,1u}){
        const size_t an=cyclic?32768:3079,bn=cyclic?991:an;
        Fixture f(4,false);sbn3_mul_options o{};o.workers=4;o.algorithm=SBN3_MUL_BAILEY;o.prime_count=6;o.trunk_bits=128;
        o.column_log2=5;o.row_log2=6;
        sbn3_product_request r{};r.kind=cyclic?SBN3_PRODUCT_MUL:SBN3_PRODUCT_SQR;r.a_limbs=an;r.b_limbs=cyclic?bn:0;r.cyclic_limbs=cyclic?an:0;
        ProductProgramPlan plan;sbn3_product_info info{};
        assert(sbn3_product_query(&r,&o,&plan.plan,&info)==SBN3_SUPPORTED);plan.info=info.mul;
        assert(product_program_describe(an,bn,plan)==SBN3_SUPPORTED);
        auto tables=f.allocate(plan.prepared_bytes,128),work=f.allocate(plan.info.workspace_bytes,plan.info.workspace_alignment);
        auto out=f.allocate(sbn3_mul_output_capacity(&plan.info)*8,64);
        std::vector<uint64_t>a(an),b(bn),expected(sbn3_mul_output_capacity(&plan.info));
        for(auto&v:a)v=pattern?UINT64_MAX:random_word();for(auto&v:b)v=random_word();if(!cyclic)b=a;
        ref_int x,y,h;ref_inits(x,y,h,nullptr);ref_import(x,an,-1,8,0,0,a.data());ref_import(y,bn,-1,8,0,0,b.data());ref_mul(x,x,y);
        if(cyclic){ref_set_ui(h,1);ref_mul_2exp(h,h,64*an);ref_sub_ui(h,h,1);ref_mod(x,x,h);}
        size_t used=0;ref_export(expected.data(),&used,-1,8,0,0,x);ref_clears(x,y,h,nullptr);
        ProductProgram p;{auto prepared=Frame(*f.arena,tables);p=product_program_prepare(plan,prepared);}
        struct Call {const ProductProgram*p;Fixture*f;sbn3_lease work;sbn3_const_limbs a,b;sbn3_limbs out;} call{&p,&f,work,{a.data(),an},{b.data(),bn},{static_cast<uint64_t*>(out.data),expected.size()}};
        allocation_watch_start();sbn3_team_run(f.team,[](void*ptr,sbn3_team_scope*scope){auto&c=*static_cast<Call*>(ptr);Frame frame(*c.f->arena,c.work);
            product_program_execute(*c.p,frame,scope,c.a,c.b,c.out);},&call);assert(!allocation_watch_stop());
        assert(!memcmp(expected.data(),out.data,info.mul.output_limbs*8));
    }
    puts("Frame-owned native SQR and CYC: independent exact value and canonical wrap PASS");
}
int main() {
    setvbuf(stdout,nullptr,_IOLBF,0);
    program_recipes();
    one(4096,4096,4,SBN3_MUL_FLAT,6,0,128);
    one(4096,4096,4,SBN3_MUL_BAILEY,6,0,128,4,5);
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
