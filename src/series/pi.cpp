#include "sbn3/constants.h"
#include "series/psr.hpp"
#include "series/scaled_ratio.hpp"
#include "algorithms/mul_rsqrt.hpp"
#include "product/program.hpp"
#include "product/cost_model.hpp"
#include "common/identity.hpp"
#include "runtime/team.hpp"
#include "core/x86_64/word.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <cstring>
#include <new>
#include <time.h>
namespace sbn::v3::series {
namespace {
constexpr uint64_t magic = 0x53424e3350493031ULL, coefficient = 4270934400ULL;
static_assert(__uint128_t(72) * (uint64_t(1) << 47) < 10939058860032000ULL);
struct TerminalInfo {
    uint64_t plan_id = 0;
    size_t storage_bytes = 0, storage_alignment = 0;
};
struct SeparateTerminal {
    sbn3_newton_plan root{};
    TerminalInfo root_info{};
    sbn3_mul_options options{};
    sbn3_mul_info multiply{};
    size_t prepared_bytes = 0, work_at = 0, product = 0, pool = 0, value_bytes = 0;
    unsigned contract = 0;
};
struct Plan {
    uint64_t marker = magic, seal = 0;
    sbn3_pi_options options{};
    sbn3_pi_info info{};
    PsrInfo psr{};
    sbn3_newton_plan division{};
    TerminalInfo division_info{};
    MulRsqrtPlan terminal{};
    SeparateTerminal separate{};
    bool fused_terminal = false;
    size_t control = 0, values = 0, value_bytes = 0, S = 0, R = 0, A = 0, D = 0, Q = 0, pool = 0, pair_D = 0,
           slot_bytes = 0, division_pool = 0, terminal_pool = 0, terminal_value_bytes = 0;
};
static_assert(sizeof(Plan) <= sizeof(sbn3_pi_plan));
struct Binding {
    Plan plan{};
    Formula formula{FormulaKind::Chudnovsky};
    PsrBinding psr{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, values{};
    bool used = false, psr_live = false;
    sbn3_pi_metrics metrics{};
};
size_t up(size_t n, size_t a = 128) {
    size_t r;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "pi layout alignment");
    return r;
}
uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
uint64_t decay(const void *, uint64_t a) {
    return a ? 47 * (a - 1) : 0;
}
PsrSpec specification(const Plan &p, const Formula &formula) {
    return {finite_formula(formula),
            {0, p.info.terms},
            p.options.series,
            {nullptr, decay, p.options.minimum_block_terms ? p.options.minimum_block_terms : 128,
             p.options.maximum_block_terms,
             64.0 * (p.options.minimum_block_limbs ? p.options.minimum_block_limbs
                                                   : size_t(16384) * p.options.series.workers)},
            p.info.working_limbs};
}
uint64_t seal(const Plan &p) {
    uint64_t h = identity::word(identity::fnv_seed, magic);
    for (uint64_t v : {p.info.fractional_limbs,
                       p.info.working_limbs,
                       p.info.terms,
                       p.info.storage_bytes,
                       p.info.storage_alignment,
                       p.psr.schedule_id,
                       p.terminal.plan_id,
                       p.division_info.plan_id,
                       p.control,
                       p.values,
                       p.value_bytes,
                       p.S,
                       p.R,
                       p.A,
                       p.D,
                       p.Q,
                       p.pool,
                       p.options.memory_budget,
                       p.pair_D,
                       p.terminal_value_bytes,
                       p.slot_bytes,
                       p.division_pool,
                       p.terminal_pool,
                       uint64_t(p.fused_terminal),
                       p.separate.root_info.plan_id,
                       p.separate.multiply.arithmetic_id,
                       p.separate.multiply.execution_id,
                       p.separate.prepared_bytes,
                       p.separate.work_at,
                       p.separate.product,
                       p.separate.pool,
                       p.separate.value_bytes,
                       uint64_t(p.separate.contract)})
        h = identity::word(h, v);
    return h;
}
Plan load(const sbn3_pi_plan &in) {
    Plan p{};
    std::memcpy(&p, in.opaque, sizeof p);
    require(p.marker == magic && p.seal == seal(p), SBN3_FATAL_ARGUMENT, "pi plan identity");
    return p;
}
uint64_t *at(Binding &b, size_t offset) {
    return reinterpret_cast<uint64_t *>(b.arena->base + b.offset + offset);
}
Binding &get(sbn3_pi_binding *p) {
    require(p, SBN3_FATAL_ARGUMENT, "pi binding");
    auto &b = *reinterpret_cast<Binding *>(p);
    require(b.plan.marker == magic && pthread_equal(b.team->creator, pthread_self()) && !b.team->busy,
            SBN3_FATAL_LIFETIME, "pi owner/state");
    return b;
}
struct FinalProduct {
    ProductProgram program;
    Frame *work;
    sbn3_const_limbs a,b;
    sbn3_limbs out;
    bool consume;
};
void final_product(void *ptr,sbn3_team_scope *scope) {
    auto &c=*static_cast<FinalProduct *>(ptr);
    if(c.consume) product_program_bounded(c.program,*c.work,scope,c.a,c.b,c.out,true);
    else product_program_execute(c.program,*c.work,scope,c.a,c.b,c.out);
}
void resize_value_lease(Binding &b, size_t bytes) {
    b.arena->release(b.values);
    b.values = b.arena->acquire(b.offset + b.plan.values, bytes);
}
} // namespace
} // namespace sbn::v3::series
using namespace sbn::v3;
using namespace sbn::v3::series;
extern "C" sbn3_query_result sbn3_pi_query(size_t n, const sbn3_pi_options *options, sbn3_pi_plan *out,
                                           sbn3_pi_info *info) {
    require(out && info, SBN3_FATAL_ARGUMENT, "pi query output");
    *info = {};
    if (!n || n > (size_t(1) << 28) - 4)
        return SBN3_UNSUPPORTED;
    Plan p{};
    p.options = options ? *options : sbn3_pi_options{{16, 8, 0, 0, 0}, 0, 128, 0, 0};
    p.info.fractional_limbs = n;
    p.info.working_limbs = n + 4;
    p.info.output_limbs = n + 1;
    p.info.workers = p.options.series.workers;
    // Tail < 2^62 * (2^-47)^N, N<2^31. This leaves <1/4 root-work ulp.
    p.info.terms = (uint64_t(p.info.working_limbs) * 64 + 64 + 46) / 47;
    require(p.info.terms < (uint64_t(1) << 31), SBN3_FATAL_MATH, "pi term bound domain");
    Formula formula{FormulaKind::Chudnovsky};
    auto spec = specification(p, formula);
    auto rc = psr_query(spec, p.psr);
    if (rc != SBN3_SUPPORTED)
        return rc;
    const sbn3_newton_options no{p.info.workers, 0, 0, 0};
    sbn3_newton_info ni{};
    rc = sbn3_newton_query(SBN3_NEWTON_DIVIDE, p.info.working_limbs + 2, &no, &p.division, &ni);
    if (rc != SBN3_SUPPORTED)
        return rc;
    p.division_info = {ni.plan_id, ni.storage_bytes, ni.storage_alignment};
    rc = mul_rsqrt_query(p.info.working_limbs, p.info.workers, p.terminal, p.psr.output_limbs + 1);
    if (rc != SBN3_SUPPORTED)
        return rc;
    auto &separate=p.separate;
    rc=sbn3_newton_query(SBN3_NEWTON_RSQRT,p.info.working_limbs,&no,&separate.root,&ni);
    if(rc!=SBN3_SUPPORTED)return rc;
    separate.root_info={ni.plan_id,ni.storage_bytes,ni.storage_alignment};
    sbn3_mul_options mo{};mo.workers=p.info.workers;mo.borrow_output=1;
    ProductProgramPlan multiply{};
    rc=product_program_query(p.info.working_limbs+1,p.info.working_limbs+1,mo,multiply);
    if(rc!=SBN3_SUPPORTED)return rc;
    if(!(multiply.contract&program_consume_inputs)) {
        auto alternative=mo;alternative.borrow_output=0;ProductProgramPlan next{};
        if(product_program_query(p.info.working_limbs+1,p.info.working_limbs+1,alternative,next)==SBN3_SUPPORTED &&
           (next.contract&program_consume_inputs)){multiply=next;mo=alternative;}
    }
    separate.options=mo;separate.multiply=multiply.info;separate.prepared_bytes=multiply.prepared_bytes;
    separate.contract=multiply.contract;
    // Pin NTT family/codec for replay. FFT preserves its established query.
    if(multiply.info.np){separate.options.prime_count=multiply.info.np;separate.options.trunk_bits=multiply.info.trunk_bits;separate.options.algorithm=multiply.info.algorithm;}
    separate.work_at=up(separate.prepared_bytes,separate.multiply.workspace_alignment);
    p.info.storage_alignment = std::max({p.psr.storage_alignment, p.division_info.storage_alignment,
                                        p.terminal.alignment,separate.root_info.storage_alignment,
                                        separate.multiply.workspace_alignment});
    p.control = up(sizeof(Binding));
    p.values = up(p.control, p.info.storage_alignment);
    p.slot_bytes = 8 * p.psr.output_limbs;
    p.S = p.D = p.values;
    p.pair_D = p.Q = p.values + p.slot_bytes;
    p.R = p.values + 8 * p.terminal.input_at;
    p.A = p.values + 2 * p.slot_bytes;
    p.value_bytes = 2 * p.slot_bytes;
    p.pool = up(p.A, p.info.storage_alignment);
    p.division_pool = up(p.values + 3 * p.slot_bytes, p.info.storage_alignment);
    p.terminal_value_bytes = std::max(2 * p.slot_bytes, 8 * p.terminal.value_words);
    p.terminal_pool = up(p.values + p.terminal_value_bytes, p.info.storage_alignment);
    const size_t m=p.info.working_limbs;
    separate.product=(separate.contract&program_consume_inputs)?p.values:p.values+2*p.slot_bytes;
    separate.value_bytes=std::max(2*p.slot_bytes,separate.product+16*(m+1)-p.values);
    separate.pool=up(p.values+separate.value_bytes,p.info.storage_alignment);
    const size_t common_peak=std::max(p.pool+p.psr.storage_bytes,p.division_pool+p.division_info.storage_bytes);
    const size_t old_peak=std::max({common_peak,p.pool+separate.root_info.storage_bytes,
                                   separate.pool+separate.work_at+separate.multiply.workspace_bytes});
    const size_t fused_peak=std::max(common_peak,p.terminal_pool+p.terminal.storage_bytes);
    const cost_model::RecipeCost candidates[]{
        {p.terminal.standalone_rung_ns+cost_model::linear_product(separate.multiply,m+1,m+1).nanoseconds,old_peak},
        {p.terminal.estimated_tail_ns,fused_peak}};
    unsigned selected=cost_model::choose_recipe(candidates,2,p.options.memory_budget);
    const bool rejected=selected==2;
    if(rejected)selected=cost_model::choose_recipe(candidates,2);
    p.fused_terminal=selected==1;
    if(!p.fused_terminal)p.R=p.Q+8;
    p.info.terminal_storage_bytes=std::max(p.division_info.storage_bytes,p.fused_terminal?p.terminal.storage_bytes:
        std::max(separate.root_info.storage_bytes,separate.work_at+separate.multiply.workspace_bytes));
    p.info.storage_bytes=candidates[selected].peak_bytes;
    p.info.psr_storage_bytes = p.psr.storage_bytes;
    p.info.psr_value_bytes = p.psr.value_bytes;
    p.info.psr_pool_bytes = p.psr.pool_bytes;
    p.info.blocks = p.psr.blocks;
    p.seal = seal(p);
    p.info.plan_id = p.seal;
    *info = p.info;
    if (rejected)
        return SBN3_QUERY_CAPACITY;
    std::memset(out, 0, sizeof *out);
    std::memcpy(out->opaque, &p, sizeof p);
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_pi_bind(const sbn3_pi_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                             sbn3_pi_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "pi bind arguments");
    const auto p = load(*opaque);
    require(team->arena == arena && team->width == p.info.workers && !team->busy &&
                pthread_equal(team->creator, pthread_self()) && offset <= arena->virtual_bytes &&
                p.info.storage_bytes <= arena->virtual_bytes - offset &&
                arena->contains(offset, offset + p.info.storage_bytes) &&
                arena->unleased(offset, p.info.storage_bytes) &&
                !(uintptr_t(arena->base + offset) & (p.info.storage_alignment - 1)),
            SBN3_FATAL_WORKSPACE, "pi prepared region");
    auto control = arena->acquire(offset, p.control);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->control = control;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->values = arena->acquire(offset + p.values, p.value_bytes);
    psr_prepare(specification(p, b->formula), p.psr, *arena, *team, offset + p.pool, b->psr);
    b->psr_live = true;
    *out = reinterpret_cast<sbn3_pi_binding *>(b);
}
static sbn3_const_limbs execute_pi(Binding &b, sbn3_limbs out) {
    const auto &p = b.plan;
    const size_t m = p.info.working_limbs;
    require(!b.used,SBN3_FATAL_LIFETIME,"pi single-use binding");
    b.used = true;
    sbn3_series_values sum{};
    sum.value[0].mantissa = {at(b, p.S), p.psr.output_limbs, 0, 0};
    sum.value[1].mantissa = {at(b, p.pair_D), p.psr.output_limbs, 0, 0};
    uint64_t start = now();
    psr_execute(b.psr, sum);
    b.metrics.psr_ns = now() - start;
    b.metrics.finite_ns = b.psr.metrics.finite_ns;
    b.metrics.merge_ns = b.psr.metrics.merge_ns;
    b.metrics.prepare_ns = b.psr.metrics.prepare_ns;
    const auto &t = sum.value[0], &q = sum.value[1];
    require(t.mantissa.size && !t.mantissa.negative && q.mantissa.size && !q.mantissa.negative,
            SBN3_FATAL_MATH, "Chudnovsky pair signs");
    psr_release(b.psr);
    b.psr_live = false;
    start = now();
    resize_value_lease(b, 3 * p.slot_bytes);
    // Normalize before multiplying by C<2^32. Its input-truncation error
    // contributes <2C/B ulps after ratio_result excludes the low guard word.
    // T is dead after this exact-alias normalization into the D slot.
    ratio_inputs({{q.mantissa.data, q.mantissa.size, 0}, q.exponent2},
                 {{t.mantissa.data, t.mantissa.size, 0}, t.exponent2}, m, at(b, p.A), at(b, p.D), b.team);
    const uint64_t carry = parallel_limbs::mul_1(b.team, at(b, p.A), at(b, p.A), m + 3, coefficient);
    require(!carry && !at(b, p.A)[m + 2], SBN3_FATAL_MATH, "pi normalized numerator");
    sbn3_newton_binding *division = nullptr;
    sbn3_newton_bind(&p.division, b.arena, b.offset + p.division_pool, b.team, &division);
    const sbn3_newton_inputs di{{at(b, p.A), m + 3}, {at(b, p.D), m + 2}, 0};
    sbn3_newton_execute(division, &di, {at(b, p.Q), m + 3});
    sbn3_newton_unbind(division);
    const auto quotient=ratio_result(at(b,p.Q),m);
    // Small layouts may need a later input slot to keep the half root disjoint.
    // Otherwise the consumer reads Q+1 directly, without shifting the quotient.
    if(quotient.data!=at(b,p.R))std::memmove(at(b,p.R),quotient.data,quotient.count*8);
    require(at(b, p.R)[m] == 314, SBN3_FATAL_MATH, "pi terminal quotient range");
    sbn3_const_limbs answer{};
    uint64_t error=4096;
    if(p.fused_terminal) {
        resize_value_lease(b, p.terminal_value_bytes);
        answer=mul_rsqrt_execute(p.terminal,10005,at(b,p.S),p.terminal_value_bytes/8,
                                 *b.arena,b.offset+p.terminal_pool,*b.team);
        error+=mul_rsqrt_error;
    } else {
        const auto &separate=p.separate;
        resize_value_lease(b,2*p.slot_bytes);
        sbn3_newton_binding *root=nullptr;
        sbn3_newton_bind(&separate.root,b.arena,b.offset+p.pool,b.team,&root);
        const sbn3_newton_inputs ri{{},{},10005};
        sbn3_newton_execute(root,&ri,{at(b,p.S),m+1});sbn3_newton_unbind(root);
        resize_value_lease(b,separate.value_bytes);
        ProductProgramPlan product{};
        require(product_program_query(m+1,m+1,separate.options,product)==SBN3_SUPPORTED &&
                    product.info.execution_id==separate.multiply.execution_id &&
                    product.info.arithmetic_id==separate.multiply.arithmetic_id &&
                    product.prepared_bytes==separate.prepared_bytes,SBN3_FATAL_MATH,"pi product replay");
        auto tables=b.arena->acquire(b.offset+separate.pool,separate.prepared_bytes);
        auto work=b.arena->acquire(b.offset+separate.pool+separate.work_at,separate.multiply.workspace_bytes);
        {
            Frame tf(*b.arena,tables),wf(*b.arena,work);
            FinalProduct c{product_program_prepare(product,tf),&wf,{at(b,p.R),m+1},{at(b,p.S),m+1},
                           {at(b,separate.product),2*(m+1)},bool(separate.contract&program_consume_inputs)};
            sbn3_team_run(b.team,final_product,&c);
        }
        b.arena->release(work);b.arena->release(tables);
        answer={at(b,separate.product)+m,m+1};
    }
    guard_separated(answer.data,m-p.info.fractional_limbs,error);
    require(answer.data[m] == 3, SBN3_FATAL_MATH, "pi integer part");
    const sbn3_const_limbs result{answer.data+m-p.info.fractional_limbs,p.info.output_limbs};
    if(out.data)std::memmove(out.data,result.data,result.count*8);
    b.metrics.terminal_ns = now() - start;
    return result;
}
extern "C" void sbn3_pi_execute(sbn3_pi_binding *ptr,sbn3_limbs out){
    auto &b=get(ptr);const auto &p=b.plan;
    require(out.data && !(uintptr_t(out.data)&63) && out.capacity>=p.info.output_limbs &&
                (out.data==at(b,p.S) || !overlaps(out.data,p.info.output_limbs*8,
                                               b.arena->base+b.offset,p.info.storage_bytes)),
            SBN3_FATAL_ARGUMENT,"pi output/lifetime");
    (void)execute_pi(b,out);
}
extern "C" sbn3_const_limbs sbn3_pi_execute_inplace(sbn3_pi_binding *ptr){
    return execute_pi(get(ptr),{});
}
extern "C" void sbn3_pi_get_metrics(const sbn3_pi_binding *p, sbn3_pi_metrics *m) {
    require(m, SBN3_FATAL_ARGUMENT, "pi metrics");
    *m = get(const_cast<sbn3_pi_binding *>(p)).metrics;
}
extern "C" void sbn3_pi_unbind(sbn3_pi_binding *p) {
    auto &b = get(p);
    if (b.psr_live)
        psr_release(b.psr);
    auto *a = b.arena;
    auto values = b.values, control = b.control;
    b.plan.marker = 0;
    b.~Binding();
    a->release(values);
    a->release(control);
}
