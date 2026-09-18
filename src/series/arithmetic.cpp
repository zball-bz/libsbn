#include "product/cohort_pair_policy.hpp"
#include "series/arithmetic.hpp"
#include "common/identity.hpp"
#include "product/cohort_policy.hpp"
#include "series/schedule.hpp"
#include "series/limited.hpp"
#include "product/program.hpp"
#include "product/cost_model.hpp"
#include "product/root_prepare_cost.hpp"
#include "sbn3/value.h"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
namespace sbn::v3::series {
namespace {
size_t align(size_t n, size_t a = 128) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "finite layout alignment");
    return r;
}
size_t largest(sbn3_series_shape s) {
    return std::max({s.limbs[0], s.limbs[1], s.limbs[2]});
}
// Single-term leaf values up to fast_leaf_words live on the executing stack;
// wider leaves use three reserved scratch slots sized at query time. No
// formula may exceed max_leaf_words per value.
constexpr unsigned fast_leaf_words = 8, max_leaf_words = 64;
// One precondition set for query, compile and prepare. A formula that fails
// here is rejected as unsupported before any schedule work; prepare treats it
// as a caller contract violation because it cannot return a status.
bool finite_valid(const FiniteFormula &f) {
    return f.bounds && f.work && f.leaf && f.max_leaf_limbs && f.max_leaf_limbs <= max_leaf_words &&
           f.recipe >= SBN3_SERIES_HYPERDESCENT && f.recipe <= SBN3_SERIES_BINARY_BBP &&
           (!f.limit_words || f.limit_context) && (!f.serial_envelope || f.split_point) &&
           (!(f.split_point || f.serial_envelope) || f.split_policy_id) &&
           (!f.normalized_serial_envelope || f.normalized_bounds);
}
struct ProductGroup {
    ProductProgram program{};
    size_t an=0,bn=0;
    bool pair=false;
};
struct Stage {
    ProductGroup groups[2]{};
    unsigned count=1;
    size_t an=0,bn=0,temp=0,work_offset=0;
    bool reuse=false,padded=true;
    const ProductGroup &group(unsigned j) const { return groups[count==1?0:j]; }
};
struct Layout {
    Stage stage{};
    ProductProgramPlan product[2]{};
    ProductPlanMemo::Derived derived[2]{}; // query path only
    size_t prepared=0,work=0,alignment=128;
    bool accelerated[2]{};
};
void store_choice(const ProductProgramPlan &p,uint64_t *w) {
    const auto &i=p.info;
    // Small backends replay their own selector at the chosen worker count;
    // NTT fixes the entire geometry and codec, avoiding the NP/T candidate grid.
    sbn3_mul_options o{};o.workers=i.workers;o.algorithm=i.algorithm;o.borrow_output=i.borrow_output;
    const auto *backend=backend_lookup(p.plan.opaque[1]);
    if(backend && backend->program_options)o=backend->program_options(p.plan);
    const uint64_t fields[]{o.workers,o.prime_batch,uint64_t(o.trunk_bits),o.borrow_output,o.workspace_budget,
        o.column_log2,o.row_log2,o.prime_count,o.crt_mode,o.codec_mode,o.algorithm,uint64_t(int64_t(o.fused_start_skew_us)),
        i.arithmetic_id,i.execution_id};
    std::copy_n(fields,sizeof(fields)/sizeof(fields[0]),w);
    w[14]=p.plan.opaque[1];
}
sbn3_mul_options load_choice(const uint64_t *w) {
    sbn3_mul_options o{};o.workers=w[0];o.prime_batch=w[1];o.trunk_bits=int(w[2]);o.borrow_output=w[3];
    o.workspace_budget=w[4];o.column_log2=w[5];o.row_log2=w[6];o.prime_count=w[7];o.crt_mode=w[8];
    o.codec_mode=w[9];o.algorithm=w[10];o.fused_start_skew_us=int(int64_t(w[11]));return o;
}
void derive(const ProductProgramPlan &p,ProductPlanMemo::Derived &d) {
    d={};d.tables=product_program_tables(p);d.local_bytes=product_program_local_bytes(p);store_choice(p,d.choice);
}
sbn3_query_result planned(ProductPlanMemo *memo,size_t an,size_t bn,const sbn3_mul_options &o,ProductProgramPlan &p,
                          ProductPlanMemo::Derived &d) {
    if(memo)return memo->query(an,bn,o,p,&d);
    const auto rc=product_program_query(an,bn,o,p);
    if(rc==SBN3_SUPPORTED)derive(p,d);
    return rc;
}
// A program that runs only a few times pays for its own root tables. The
// product selector ranks by execution time alone, which is right for a
// repeatedly used MUL. Compare the complete prices with the product layer's
// own calibrated estimates (root_prepare_cost). With the scalar, uncompressed
// tower a flat table cost several products of its own size and this switched
// most one-shot nodes to the blocked (Bailey) form; since the compressed,
// vector-built tower (2026-09-17 calibration) it is a few percent of one
// product and the comparison rarely changes the selector's choice. Pairing and
// the input contracts the caller relies on must not be lost.
void price_few_uses(ProductPlanMemo *memo,size_t an,size_t bn,const sbn3_mul_options &o,double uses,bool pair,
                    ProductProgramPlan &product,ProductPlanMemo::Derived &derived) {
    if(!product.info.np || product.info.algorithm!=SBN3_MUL_FLAT)return;
    auto options=o;options.algorithm=SBN3_MUL_BAILEY;
    ProductProgramPlan alternative{};ProductPlanMemo::Derived facts{};
    if(planned(memo,an,bn,options,alternative,facts)!=SBN3_SUPPORTED)return;
    constexpr unsigned kept=program_consume_inputs|program_bounded_inputs;
    if((pair && product.pair_workspace_bytes && !alternative.pair_workspace_bytes) ||
       ((product.contract&kept)&~alternative.contract))return;
    const double current=uses*cost_model::linear_product(product.info,an,bn).nanoseconds+root_prepare_cost(product.info);
    const double candidate=uses*cost_model::linear_product(alternative.info,an,bn).nanoseconds+root_prepare_cost(alternative.info);
    if(candidate<current){product=alternative;derived=facts;}
}
sbn3_query_result layout(const sbn3_series_stage &s,Layout &v,bool reuse,sbn3_series_recipe recipe,unsigned leaf_words,const uint64_t *decisions=nullptr,unsigned cohort_workers=0,ProductPlanMemo *memo=nullptr) {
    v={};v.stage.reuse=reuse && !s.leaf;
    const size_t output=largest(s.output);
    v.stage.an=s.leaf?output:largest(s.left);
    v.stage.bn=s.leaf?output:largest(s.right);
    const bool common=!s.leaf && recipe==SBN3_SERIES_COMMON_P2B3;
    if(common){
        v.stage.count=2;
        // One common Q_R feeds T and Q. A separate common U_L feeds
        // the secondary T term and U; it must not inherit T's input width.
        v.stage.groups[0].an=std::max((s.need&1)?s.left.limbs[0]:0,(s.need&2)?s.left.limbs[1]:0);
        v.stage.groups[0].bn=(s.need&3)?s.right.limbs[1]:0;
        v.stage.groups[0].pair=(s.need&3)==3;
        v.stage.groups[1].an=(s.need&5)?s.left.limbs[2]:0;
        v.stage.groups[1].bn=std::max((s.need&1)?s.right.limbs[0]:0,(s.need&4)?s.right.limbs[2]:0);
        v.stage.groups[1].pair=(s.need&5)==5;
    }else v.stage.groups[0]={ {},v.stage.an,v.stage.bn,bool((s.need&1)&&(s.need&6)) };
    size_t sum=0;
    if(!add_size(v.stage.an,v.stage.bn,sum) || sum>SIZE_MAX/64)return SBN3_QUERY_CAPACITY;
    v.stage.temp=std::max(sum+2,output+2);
    v.prepared=align(sizeof(Stage));
    // Limited-precision leaves may word-align a bit-granular exponent in
    // place, which needs one spare word beyond the widest single-term value.
    const size_t single=s.leaf && leaf_words+1>fast_leaf_words?align((size_t(leaf_words)+1)*8):0;
    const __uint128_t slots=__uint128_t(3)*align(s.leaf?output*8:0)+__uint128_t(3)*single+
        2*__uint128_t(align(v.stage.temp*8))+align(v.stage.an*8)+align(v.stage.bn*8);
    if(slots>SIZE_MAX)return SBN3_QUERY_CAPACITY;
    v.work=size_t(slots);
    if(v.stage.reuse)v.work-=align(v.stage.temp*8);
    bool all_consume=v.stage.reuse;
    size_t arithmetic_bytes=0;
    for(unsigned j=0;j<v.stage.count;++j){
        auto &g=v.stage.groups[j];auto &product=v.product[j];auto &derived=v.derived[j];
        if(!g.an || !g.bn)continue;
        if(!s.leaf && std::max(g.an,g.bn)>6){
            sbn3_mul_options o{};o.workers=s.workers;o.borrow_output=v.stage.reuse?0:1;
            if(decisions)o=load_choice(decisions+16*j);
            const auto rc=decisions?product_program_replay(g.an,g.bn,o,decisions[16*j+14],product)
                                   :planned(memo,g.an,g.bn,o,product,derived);
            if(rc!=SBN3_SUPPORTED)return rc;
            // Preserve the measured paired centered-FFT policy for each
            // family. Its dispatch belongs to this episode, not fresh MUL.
            if(!decisions && v.stage.reuse && g.pair && product.info.algorithm==SBN3_MUL_PQ16 &&
               !(product.contract&program_consume_inputs)){
                auto options=o;options.algorithm=SBN3_MUL_FLAT;ProductProgramPlan alternative{};
                ProductPlanMemo::Derived facts{};
                if(planned(memo,g.an,g.bn,options,alternative,facts)==SBN3_SUPPORTED && alternative.pair_workspace_bytes){
                    product=alternative;derived=facts;
                }
            }
            // Two complex FFT buffers. Concurrent serial products share the
            // machine LLC; tables are already shared separately by the engine.
            if(!decisions && product.info.algorithm==SBN3_MUL_PQ16 && s.workers==1 && cohort_workers>1) {
                const size_t cache_share=cohort_policy::serial_cache_share(cohort_workers);
                if(cache_share && product.info.transform_trunks>cache_share/16) {
                    auto options=o;options.algorithm=SBN3_MUL_FLAT;ProductProgramPlan alternative{};
                    ProductPlanMemo::Derived facts{};
                    if(planned(memo,g.an,g.bn,options,alternative,facts)==SBN3_SUPPORTED &&
                       (!(v.stage.reuse&&g.pair)||alternative.pair_workspace_bytes) &&
                       alternative.info.per_worker_bytes<=cache_share){
                        product=alternative;derived=facts;
                    }
                }
            }
            // Coarse nodes run once; siblings of the same width may share a table.
            // (Counting a paired program as two products was measured: slightly
            // better trees, but the larger flat tables cost more storage than the
            // block policy then saves; see the round's report.)
            if(!decisions && !s.serial_envelope)
                price_few_uses(memo,g.an,g.bn,o,double(std::max(1u,cohort_workers/std::max(1u,s.workers))),
                               v.stage.reuse&&g.pair,product,derived);
            if(!decisions && v.stage.reuse && g.pair){
                const uint64_t before=product.info.execution_id;
                cohort_pair::select(product,cohort_workers);
                if(product.info.execution_id!=before)derive(product,derived);
            }
            if(decisions)require(product.info.arithmetic_id==decisions[16*j+12] &&
                                  product.info.execution_id==decisions[16*j+13],
                                  SBN3_FATAL_MATH,"finite product decision replay");
            v.accelerated[j]=true;
            all_consume &= (product.contract&(program_consume_inputs|program_bounded_inputs))==
                           (program_consume_inputs|program_bounded_inputs);
            v.alignment=std::max(v.alignment,product.info.workspace_alignment);
            const size_t bytes=v.stage.reuse && g.pair?std::max(product.info.workspace_bytes,product.pair_workspace_bytes)
                                                            :product.info.workspace_bytes;
            arithmetic_bytes=std::max(arithmetic_bytes,bytes);
            if(!add_size(v.prepared,decisions?product_program_local_bytes(product):derived.local_bytes,v.prepared))
                return SBN3_QUERY_CAPACITY;
        }else all_consume=false;
    }
    v.stage.padded=!all_consume;
    if(!v.stage.padded)v.work-=align(v.stage.an*8)+align(v.stage.bn*8);
    v.stage.work_offset=align(v.work,v.alignment);
    if(!add_size(v.stage.work_offset,arithmetic_bytes,v.work))return SBN3_QUERY_CAPACITY;
    return SBN3_SUPPORTED;
}
static sbn3_query_result finish_bounds(const FiniteFormula &f, sbn3_series_range r, unsigned need,
                                      sbn3_series_shape *out) {
    // Value add/shift APIs require a spare limb even when a stronger formula
    // bound proves its final carry zero. Preserve that implementation contract.
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j)) {
            if (f.limit_words)
                out->limbs[j] = std::min(out->limbs[j], f.limit_words(f.limit_context, r.begin));
            if (!add_size(out->limbs[j], 2, out->limbs[j]))
                return SBN3_QUERY_CAPACITY;
        }
    return SBN3_SUPPORTED;
}
sbn3_query_result bounds(const void *ptr, sbn3_series_range r, uint64_t n, unsigned need,
                         sbn3_series_shape *out) {
    const auto &f = *static_cast<const FiniteFormula *>(ptr);
    const auto rc = f.bounds(f.context, r, n, need, out);
    return rc == SBN3_SUPPORTED ? finish_bounds(f, r, need, out) : rc;
}
double work(const void *ptr, sbn3_series_range r) {
    const auto &f = *static_cast<const FiniteFormula *>(ptr);
    return f.work(f.context, r);
}
sbn3_query_result prepared_resources_impl(const void *ptr, const sbn3_series_stage *s,
                                      sbn3_series_resources *r, StagePreparation *prepared,unsigned cohort_workers,
                                      ProductPlanMemo *memo=nullptr) {
    const auto &f = *static_cast<const FiniteFormula *>(ptr);
    Layout v{};
    const auto rc = layout(*s, v, f.reuse_values, f.recipe,f.max_leaf_limbs,nullptr,cohort_workers,memo);
    if (rc != SBN3_SUPPORTED)
        return rc;
    double cost=0;
    for(unsigned j=0;j<v.stage.count;++j){
        const auto &g=v.stage.groups[j];
        const double size=double(g.an+g.bn);
        const unsigned count=v.stage.count==2
            ? unsigned(bool(s->need&1))+unsigned(bool(s->need&(j?4:2)))
            : unsigned(bool(s->need&2))+unsigned(bool(s->need&4))+
              unsigned(bool(s->need&1))*(f.recipe==SBN3_SERIES_HYPERDESCENT?1:2);
        const unsigned width=v.accelerated[j]?v.product[j].info.workers:1;
        cost+=s->leaf?double(s->max_terms)*size:count*size*std::log2(size+1)/width;
        const auto request=v.accelerated[j]?v.derived[j].tables:SharedPreparation{};
        if(v.accelerated[j])std::copy_n(v.derived[j].choice,16,prepared->decisions+16*j);
        prepared->shared.requests[j]=f.table_pool.find(request)?SharedPreparation{}:request;
    }
    *r={v.work,v.alignment,cost,v.prepared,128};
    return SBN3_SUPPORTED;
}
struct PlanningResources { const FiniteFormula *formula; unsigned cohort_workers; ProductPlanMemo *memo; };
sbn3_query_result prepared_resources(const void *ptr,const sbn3_series_stage *s,
                                    sbn3_series_resources *r,StagePreparation *prepared) {
    return prepared_resources_impl(ptr,s,r,prepared,0);
}
sbn3_query_result contextual_resources(const void *ptr,const sbn3_series_stage *s,
                                       sbn3_series_resources *r,StagePreparation *prepared) {
    const auto &context=*static_cast<const PlanningResources *>(ptr);
    return prepared_resources_impl(context.formula,s,r,prepared,context.cohort_workers,context.memo);
}
sbn3_query_result resources(const void *ptr, const sbn3_series_stage *s, sbn3_series_resources *r) {
    StagePreparation shared{};
    return prepared_resources(ptr, s, r, &shared);
}
sbn3_series_oracle oracle(const FiniteFormula &f) {
    return {&f, bounds, work, resources};
}
ScheduleRefinement refinement(const FiniteFormula &f,const PlanningResources *resources=nullptr) {
    ScheduleRefinement policy{&f, nullptr, bool(f.limit_words) || f.reuse_values, f.reuse_values};
    policy.split_policy_id = f.split_policy_id;
    policy.prepared_resources = resources?contextual_resources:prepared_resources;
    policy.prepared_context=resources;
    if (f.split_point)
        policy.split_point = [](const void *p, sbn3_series_range r, double fraction) {
            const auto &f = *static_cast<const FiniteFormula *>(p);
            return f.split_point(f.context, r, fraction);
        };
    if (f.serial_envelope)
        policy.serial_envelope = [](const void *p, sbn3_series_range r, unsigned depth, unsigned need,
                                    uint64_t *max_terms, sbn3_series_shape *out) {
            const auto &f = *static_cast<const FiniteFormula *>(p);
            const auto rc = f.serial_envelope(f.context, r, depth, need, max_terms, out);
            return rc == SBN3_SUPPORTED ? finish_bounds(f, r, need, out) : rc;
        };
    // A truncated node is split explicitly so that its right part is planned
    // with its own, smaller limit instead of the limit at the node's start. That
    // only pays where the limit falls appreciably inside the node: otherwise
    // the shared serial envelope (limit at the start, capacity for all) plans
    // the same products within a few percent, and every explicit node is a
    // stage of its own to plan, bind and prepare. With a tight contribution
    // certificate the late blocks of a chain are truncated almost everywhere,
    // and splitting all of it tripled their schedules for no measurable gain.
    if (f.limit_words)
        policy.split_serial = [](const void *p, sbn3_series_range r, unsigned need) {
            const auto &f = *static_cast<const FiniteFormula *>(p);
            sbn3_series_shape exact{};
            if (f.bounds(f.context, r, r.end - r.begin, need, &exact) != SBN3_SUPPORTED)
                return false;
            const size_t limit = f.limit_words(f.limit_context, r.begin);
            if (largest(exact) <= limit)
                return false;
            // An explicit node costs a stage (planned in query and again in bind, tables prepared at
            // execution): a few microseconds each. A tenth of the products of a node pays for that
            // only once they are transform-sized.
            if (limit < 1024)
                return false;
            const size_t later = f.limit_words(f.limit_context, r.begin + (r.end - r.begin) / 2);
            return later + limit / 8 < limit;
        };
    return policy;
}
sbn3_int_view view(const sbn3_series_value &v) {
    return {v.mantissa.data, v.mantissa.size, v.mantissa.negative};
}
void copy(sbn3_series_value &out, const sbn3_series_value &in) {
    require(in.mantissa.size <= out.mantissa.capacity, SBN3_FATAL_MATH, "finite output bound",
            in.mantissa.size, out.mantissa.capacity);
    if (in.mantissa.size)
        std::memmove(out.mantissa.data, in.mantissa.data, in.mantissa.size * 8);
    out.mantissa.size = in.mantissa.size;
    out.mantissa.negative = in.mantissa.negative;
    out.exponent2 = in.exponent2;
}
struct Merge {
    FiniteBinding &binding;
    const Stage &stage;
    sbn3_team_scope *scope;
    unsigned char *base;
    sbn3_series_value temp[2]{};
    uint64_t *a = nullptr, *b = nullptr;
    size_t cursor = 0;
    size_t precision = 0;
    sbn3_series_value slot(size_t words) {
        auto *p = reinterpret_cast<uint64_t *>(base + cursor);
        cursor += align(words * 8);
        return {{p, words, 0, 0}, 0};
    }
    Merge(FiniteBinding &bnd, const Stage &s, sbn3_team_scope *sc, void *ptr, size_t output, uint64_t begin)
        : binding(bnd), stage(s), scope(sc), base(static_cast<unsigned char *>(ptr)) {
        const auto &f = binding.plan.formula;
        precision = f.limit_words ? f.limit_words(f.limit_context, begin) : 0;
        cursor = 3 * align(output * 8);
        temp[0] = slot(s.temp);
        if (!s.reuse)
            temp[1] = slot(s.temp);
        if (s.padded) {
            a = slot(s.an).mantissa.data;
            b = slot(s.bn).mantissa.data;
        }
        require(cursor <= stage.work_offset, SBN3_FATAL_WORKSPACE, "finite scratch layout");
    }
    void multiply(sbn3_series_value &out, const sbn3_series_value &x, const sbn3_series_value &y, unsigned group=0) {
        const auto &g=stage.group(group);
        const unsigned negative = x.mantissa.negative ^ y.mantissa.negative;
        require(!__builtin_add_overflow(x.exponent2, y.exponent2, &out.exponent2), SBN3_FATAL_SIZE,
                "finite product exponent");
        if (!x.mantissa.size || !y.mantissa.size) {
            out.mantissa.size = out.mantissa.negative = 0;
            return;
        }
        const size_t xn = x.mantissa.size, yn = y.mantissa.size;
        require(xn <= g.an && yn <= g.bn, SBN3_FATAL_WORKSPACE, "finite operand capacities");
        if (!g.program.prepared || (stage.padded && std::min(xn, yn) <= 4)) {
            require(xn + yn <= out.mantissa.capacity, SBN3_FATAL_WORKSPACE, "finite base product");
            if (stage.reuse) {
                std::memcpy(a, x.mantissa.data, xn * 8);
                std::memcpy(b, y.mantissa.data, yn * 8);
            }
            sbn3_mul_basecase(out.mantissa.data, out.mantissa.capacity, stage.reuse ? a : x.mantissa.data, xn,
                              stage.reuse ? b : y.mantissa.data, yn);
            out.mantissa.size = xn + yn;
        } else {
            if (stage.padded) {
                std::memcpy(a, x.mantissa.data, xn * 8);
                std::memset(a + xn, 0, (g.an - xn) * 8);
                std::memcpy(b, y.mantissa.data, yn * 8);
                std::memset(b + yn, 0, (g.bn - yn) * 8);
            }
            auto frame = binding.scratch_roots[sbn3_team_first_worker(scope)]->borrowed_view(
                base + stage.work_offset, g.program.workspace_bytes);
            if (stage.reuse && (g.program.contract & program_bounded_inputs)) {
                product_program_bounded(g.program, frame, scope, {stage.padded ? a : x.mantissa.data, xn},
                                        {stage.padded ? b : y.mantissa.data, yn},
                                        {out.mantissa.data, out.mantissa.capacity}, !stage.padded);
                out.mantissa.size = xn + yn;
            } else {
                product_program_execute(g.program, frame, scope, {a, g.an}, {b, g.bn},
                                        {out.mantissa.data, out.mantissa.capacity});
                out.mantissa.size = g.an + g.bn;
            }
        }
        out.mantissa.negative = negative;
        sbn3_int_normalize(&out.mantissa);
        if (precision)
            limit_value(out, precision);
    }
    void add(sbn3_series_value &out, sbn3_series_value &x, sbn3_series_value &y) {
        if (precision) {
            add_limited(out, x, y, precision);
            return;
        }
        const int64_t e = std::min(x.exponent2, y.exponent2);
        for (auto *v : {&x, &y})
            if (v->exponent2 != e) {
                const __int128 difference = __int128(v->exponent2) - e;
                require(difference <= SIZE_MAX, SBN3_FATAL_SIZE, "finite exponent alignment");
                sbn3_int_lshift(&v->mantissa, view(*v), size_t(difference));
                v->exponent2 = e;
            }
        if (!x.mantissa.negative && !y.mantissa.negative && out.mantissa.data == x.mantissa.data &&
            x.mantissa.size && y.mantissa.size) {
            const size_t n = std::max(x.mantissa.size, y.mantissa.size);
            require(out.mantissa.capacity > n, SBN3_FATAL_WORKSPACE, "finite in-place sum capacity");
            if (n > x.mantissa.size)
                std::memset(out.mantissa.data + x.mantissa.size, 0, (n - x.mantissa.size) * 8);
            const uint64_t carry = parallel_limbs::add_to_scope(scope, out.mantissa.data, n,
                                                                y.mantissa.data, y.mantissa.size);
            out.mantissa.size = n;
            if (carry)
                out.mantissa.data[out.mantissa.size++] = carry;
            out.mantissa.negative = 0;
        } else
            sbn3_int_add(&out.mantissa, view(x), view(y));
        out.exponent2 = e;
    }
    void copy_addend(sbn3_series_value &out, const sbn3_series_value &in) {
        require(in.mantissa.size <= out.mantissa.capacity, SBN3_FATAL_WORKSPACE, "finite addend copy");
        parallel_limbs::copy_scope(scope, out.mantissa.data, in.mantissa.data, in.mantissa.size);
        out.mantissa.size = in.mantissa.size;
        out.mantissa.negative = in.mantissa.negative;
        out.exponent2 = in.exponent2;
    }
    bool paired(sbn3_series_value &out0, sbn3_series_value &out1, const sbn3_series_value &common,
                const sbn3_series_value &x, const sbn3_series_value &y, bool right, unsigned group=0) {
        const auto &g=stage.group(group);
        if (!g.program.pair_workspace_bytes || !common.mantissa.size ||
            !x.mantissa.size || !y.mantissa.size)
            return false;
        const unsigned negative0 = common.mantissa.negative ^ x.mantissa.negative;
        const unsigned negative1 = common.mantissa.negative ^ y.mantissa.negative;
        int64_t exponent0 = 0, exponent1 = 0;
        require(!__builtin_add_overflow(common.exponent2, x.exponent2, &exponent0) &&
                    !__builtin_add_overflow(common.exponent2, y.exponent2, &exponent1),
                SBN3_FATAL_SIZE, "finite pair exponent");
        const size_t n0 = common.mantissa.size + x.mantissa.size;
        const size_t n1 = common.mantissa.size + y.mantissa.size;
        auto frame = binding.scratch_roots[sbn3_team_first_worker(scope)]->borrowed_view(
            base + stage.work_offset, g.program.pair_workspace_bytes);
        product_program_pair(g.program, frame, scope, {common.mantissa.data, common.mantissa.size},
                             {x.mantissa.data, x.mantissa.size}, {y.mantissa.data, y.mantissa.size},
                             {out0.mantissa.data, out0.mantissa.capacity},
                             {out1.mantissa.data, out1.mantissa.capacity}, right);
        out0.mantissa.size = n0; out0.mantissa.negative = negative0; out0.exponent2 = exponent0;
        out1.mantissa.size = n1; out1.mantissa.negative = negative1; out1.exponent2 = exponent1;
        for (auto *out : {&out0, &out1}) {
            sbn3_int_normalize(&out->mantissa);
            if (precision)
                limit_value(*out, precision);
        }
        return true;
    }
    void apply(const sbn3_series_values &l, const sbn3_series_values &r, sbn3_series_values &out,
               unsigned need) {
        const auto recipe = binding.plan.formula.recipe;
        if (stage.reuse) {
            bool q_done = false, u_done = false;
            if (need & 1) {
                // TR dies after the secondary product. The primary product
                // then consumes TL/QR and may overwrite the whole parent T lane.
                if (recipe == SBN3_SERIES_HYPERDESCENT)
                    copy_addend(temp[0], r.value[0]);
                else if (recipe == SBN3_SERIES_COMMON_P2B3 && (need & 4) &&
                         paired(temp[0], out.value[2], l.value[2], r.value[0], r.value[2], false, 1))
                    u_done = true;
                else
                    multiply(temp[0], l.value[recipe == SBN3_SERIES_COMMON_P2B3 ? 2 : 1], r.value[0], recipe == SBN3_SERIES_COMMON_P2B3 ? 1 : 0);
                if ((need & 2) && paired(out.value[0], out.value[1], r.value[1], l.value[0], l.value[1], true))
                    q_done = true;
                else
                    multiply(out.value[0], l.value[0], r.value[1]);
                add(out.value[0], out.value[0], temp[0]);
            }
            if ((need & 2) && !q_done)
                multiply(out.value[1], l.value[1], r.value[1]);
            if ((need & 4) && !u_done)
                multiply(out.value[2], l.value[2], r.value[2], 1);
            return;
        }
        if (need & 1) {
            multiply(temp[0], l.value[0], r.value[1]);
            if (recipe == SBN3_SERIES_HYPERDESCENT)
                copy(temp[1], r.value[0]);
            else
                multiply(temp[1], l.value[recipe == SBN3_SERIES_COMMON_P2B3 ? 2 : 1], r.value[0], recipe == SBN3_SERIES_COMMON_P2B3 ? 1 : 0);
            add(out.value[0], temp[0], temp[1]);
        }
        if (need & 2) {
            multiply(temp[0], l.value[1], r.value[1]);
            copy(out.value[1], temp[0]);
        }
        if (need & 4) {
            multiply(temp[0], l.value[2], r.value[2], 1);
            copy(out.value[2], temp[0]);
        }
    }
};
void leaf(void *ptr, const sbn3_series_stage *s, sbn3_series_range range, unsigned need,
          sbn3_series_values *out, void *scratch, size_t, sbn3_team_scope *scope) {
    auto &binding = *static_cast<FiniteBinding *>(ptr);
    const auto &f = binding.plan.formula;
    const size_t terms=range.end-range.begin;
    if (f.batch && (!f.batch_max_terms || terms<=f.batch_max_terms) && !f.limit_words) {
        f.batch(f.context, range, need, out);
        return;
    }
    constexpr size_t batch_limbs = 64 * 3 + 8; // 64 terms of at most three words, carries and spare
    if (f.batch && f.limit_words && f.batch_words_per_term && terms <= f.batch_max_terms &&
        terms * f.batch_words_per_term + 2 <= batch_limbs - 4) {
        uint64_t words[3][batch_limbs];
        sbn3_series_values exact{};
        for (unsigned j = 0; j < 3; ++j)
            exact.value[j].mantissa = {words[j], batch_limbs, 0, 0};
        f.batch(f.context, range, need, &exact);
        const size_t precision = f.limit_words(f.limit_context, range.begin);
        for (unsigned j = 0; j < 3; ++j)
            if (need & (1u << j)) {
                limit_value(exact.value[j], precision);
                copy(out->value[j], exact.value[j]);
            }
        return;
    }
    const auto &stage = *reinterpret_cast<const Stage *>(static_cast<const char *>(binding.prepared.data) +
                                                         s->prepared_offset);
    Merge m(binding, stage, scope, scratch, largest(s->output), range.begin);
    sbn3_series_values acc{}, single{};
    uint64_t small[3][fast_leaf_words]{};
    unsigned required_left = 0, required_right = 0;
    sbn3_series_child_needs(f.recipe, need, &required_left, &required_right);
    const unsigned all = required_left | required_right;
    const size_t single_words = size_t(f.max_leaf_limbs) + 1;
    const bool wide = single_words > fast_leaf_words;
    if (wide)
        require(m.cursor + 3 * align(single_words * 8) <= stage.work_offset, SBN3_FATAL_WORKSPACE,
                "finite wide leaf slots");
    for (unsigned j = 0; j < 3; ++j) {
        acc.value[j] = {{reinterpret_cast<uint64_t *>(m.base + j * align(largest(s->output) * 8)),
                         largest(s->output), 0, 0},
                        0};
        single.value[j] = wide ? sbn3_series_value{{reinterpret_cast<uint64_t *>(
                                                        m.base + m.cursor + j * align(single_words * 8)),
                                                    single_words, 0, 0},
                                                   0}
                               : sbn3_series_value{{small[j], fast_leaf_words, 0, 0}, 0};
    }
    // Limited arithmetic aligns values by whole words. Exact evaluation keeps
    // bit-granular exponents (BinaryBBP); limited blocks fold the sub-word part
    // of each single-term exponent into the mantissa losslessly.
    auto word_align = [&](sbn3_series_values &v) {
        if (!m.precision)
            return;
        for (unsigned j = 0; j < 3; ++j)
            if (all & (1u << j)) {
                auto &x = v.value[j];
                if (!x.mantissa.size) {
                    x.exponent2 = 0;
                    continue;
                }
                const unsigned r = unsigned(((x.exponent2 % 64) + 64) % 64);
                if (!r)
                    continue;
                require(x.mantissa.size < x.mantissa.capacity, SBN3_FATAL_WORKSPACE, "finite leaf alignment");
                sbn3_int_lshift(&x.mantissa, view(x), r);
                x.exponent2 -= int64_t(r);
            }
    };
    f.leaf(f.context, range.begin, all, &single);
    word_align(single);
    for (unsigned j = 0; j < 3; ++j)
        if (all & (1u << j))
            copy(acc.value[j], single.value[j]);
    for (uint64_t k = range.begin + 1; k < range.end; ++k) {
        f.leaf(f.context, k, all, &single);
        word_align(single);
        m.apply(acc, single, acc, all);
    }
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j))
            copy(out->value[j], acc.value[j]);
}
void merge(void *ptr, const sbn3_series_stage *s, sbn3_series_range range, uint64_t, unsigned need,
           const sbn3_series_values *l, const sbn3_series_values *r, sbn3_series_values *out, void *scratch,
           size_t, sbn3_team_scope *scope) {
    auto &binding = *static_cast<FiniteBinding *>(ptr);
    const auto &stage = *reinterpret_cast<const Stage *>(static_cast<const char *>(binding.prepared.data) +
                                                         s->prepared_offset);
    Merge m(binding, stage, scope, scratch, 0, range.begin);
    m.apply(*l, *r, *out, need);
}
void prepare_tables(void *ptr,const sbn3_series_stage *s,const sbn3_series_resources *) {
    auto &binding=*static_cast<FiniteBinding *>(ptr);
    const auto &f=binding.plan.formula;
    bool owner=false;
    for(unsigned j=0;j<2;++j)owner |= shared_placement(binding.schedule,s->index,j).owner;
    if(!owner)return;
    Layout v{};require(layout(*s,v,f.reuse_values,f.recipe,f.max_leaf_limbs,stage_decisions(binding.schedule,s->index))==SBN3_SUPPORTED,SBN3_FATAL_MATH,"finite shared table query");
    for(unsigned j=0;j<v.stage.count;++j){
        const auto place=shared_placement(binding.schedule,s->index,j);
        if(!place.owner)continue;
        const auto &product=v.product[j];
        require(product_program_tables(product).bytes==place.bytes,SBN3_FATAL_MATH,"finite shared table size");
        auto frame=Frame::borrow(*binding.arena,binding.prepared,static_cast<char *>(binding.prepared.data)+place.offset,place.bytes);
        const auto *backend=backend_lookup(product.plan.opaque[1]);
        const void *data=backend->program_tables_prepare(product.plan,frame);
        require(data==frame.data(),SBN3_FATAL_MATH,"finite shared table object base");
    }
}
void prepare_stage(void *ptr,const sbn3_series_stage *s,const sbn3_series_resources *r) {
    auto &binding=*static_cast<FiniteBinding *>(ptr);const auto &f=binding.plan.formula;
    Layout v{};
    require(layout(*s,v,f.reuse_values,f.recipe,f.max_leaf_limbs,stage_decisions(binding.schedule,s->index))==SBN3_SUPPORTED && v.prepared==r->prepared_bytes && v.work==r->bytes,
            SBN3_FATAL_MATH,"finite preparation/query agreement");
    auto frame=Frame::borrow(*binding.arena,binding.prepared,static_cast<char *>(binding.prepared.data)+s->prepared_offset,r->prepared_bytes);
    auto *stage=::new(frame.allocate(align(sizeof(Stage)),128))Stage(v.stage);
    for(unsigned j=0;j<v.stage.count;++j)if(v.accelerated[j]){
        const auto &product=v.product[j];
        const auto place=shared_placement(binding.schedule,s->index,j);
        const void *shared=f.table_pool.find(product_program_tables(product));
        if(!shared && place.bytes)shared=static_cast<char *>(binding.prepared.data)+place.offset;
        stage->groups[j].program=product_program_prepare(product,frame,shared);
    }
}
} // namespace
FiniteFormula finite_formula(const Formula &f) noexcept {
    FiniteFormula result{&f,
            f.recipe(),
            0x53424e33464f5231ULL,
            uint64_t(f.kind) * 65536 + f.radix_bits,
            5,
            [](const void *p, sbn3_series_range r, uint64_t n, unsigned need, sbn3_series_shape *s) {
                return static_cast<const Formula *>(p)->bounds(r, n, need, *s);
            },
            [](const void *p, sbn3_series_range r) { return static_cast<const Formula *>(p)->work(r); },
            [](const void *p, uint64_t k, unsigned need, sbn3_series_values *v) {
                static_cast<const Formula *>(p)->leaf(k, need, *v);
            },
            f.kind == FormulaKind::Euler
                ? +[](const void *p, sbn3_series_range r, unsigned need,
                      sbn3_series_values *v) { static_cast<const Formula *>(p)->euler_batch(r, need, *v); }
                : nullptr};
    if(result.batch){result.batch_words_per_term=1;result.batch_max_terms=64;}
    if (f.kind == FormulaKind::Chudnovsky) {
        result.normalized_bounds=[](const void *p,sbn3_series_range r,uint64_t n,unsigned need,sbn3_series_shape *shape){
            return static_cast<const Formula *>(p)->bounds(r,n,need,*shape,true);
        };
        result.normalized_serial_envelope=[](const void *p,sbn3_series_range r,unsigned depth,unsigned need,
                                             uint64_t *n,sbn3_series_shape *shape){
            return static_cast<const Formula *>(p)->serial_envelope(r,depth,need,*n,*shape,true);
        };
        result.split_policy_id = 0x434855444d415331ULL; // CHUDMAS1: tree mass envelope, PSR unchanged
        result.split_point = [](const void *p, sbn3_series_range r, double fraction) {
            return static_cast<const Formula *>(p)->split_point(r, fraction);
        };
        result.serial_envelope = [](const void *p, sbn3_series_range r, unsigned depth, unsigned need,
                                    uint64_t *max_terms, sbn3_series_shape *shape) {
            return static_cast<const Formula *>(p)->serial_envelope(r, depth, need, *max_terms, *shape);
        };
    }
    return result;
}
sbn3_query_result finite_value_shape(const FiniteFormula &f, sbn3_series_range range, unsigned need,
                                     sbn3_series_shape &out) noexcept {
    return bounds(&f, range, range.end - range.begin, need, &out);
}
static bool same_options(const sbn3_mul_options &a,const sbn3_mul_options &b) {
    // Field-wise: padding is not part of the request. A new option field must be added here.
    static_assert(sizeof(sbn3_mul_options)==56,"ProductPlanMemo key must cover every sbn3_mul_options field");
    return a.workers==b.workers && a.prime_batch==b.prime_batch && a.trunk_bits==b.trunk_bits &&
           a.borrow_output==b.borrow_output && a.workspace_budget==b.workspace_budget &&
           a.column_log2==b.column_log2 && a.row_log2==b.row_log2 && a.prime_count==b.prime_count &&
           a.crt_mode==b.crt_mode && a.codec_mode==b.codec_mode && a.algorithm==b.algorithm &&
           a.fused_start_skew_us==b.fused_start_skew_us;
}
sbn3_query_result ProductPlanMemo::query(size_t an,size_t bn,const sbn3_mul_options &o,ProductProgramPlan &p,
                                         Derived *facts) noexcept {
    Entry *victim=&entries[0];
    for(auto &e:entries){
        if(e.stamp && e.an==an && e.bn==bn && same_options(e.options,o)){
            e.stamp=++clock;++hits;
            if(e.result==SBN3_SUPPORTED){p=e.plan;if(facts)*facts=e.derived;}
            return e.result;
        }
        if(e.stamp<victim->stamp)victim=&e;
    }
    ++misses;
    victim->an=an;victim->bn=bn;victim->options=o;victim->stamp=++clock;
    victim->plan=ProductProgramPlan{};victim->derived={};
    // A decision evicted from this small table is still in the pass's own
    // record (or in the replayed one): rebuild it from its canonical options
    // instead of searching the candidate families again. The record then holds
    // each distinct decision once, so its capacity covers larger plans.
    bool known=false,recorded=false;
    auto short_key=[](size_t x,size_t y,const sbn3_mul_options &q){
        return uint32_t(identity::word(identity::word(identity::word(x,y),q.workers),q.prime_count)>>16);
    };
    const uint32_t key=short_key(an,bn,o);
    for(unsigned side=0;side<2;++side){
        const ProductDecisionLog *log=side?static_cast<const ProductDecisionLog *>(record):replay;
        if(!log || known)continue;
        for(;keyed[side]<log->count;++keyed[side]){
            const auto &e=log->entries[keyed[side]];
            keys[side][keyed[side]]=short_key(e.an,e.bn,e.options);
        }
        for(unsigned j=0;j<log->count;++j){
            if(keys[side][j]!=key)continue;
            const auto &e=log->entries[j];
            if(e.an!=an || e.bn!=bn || !same_options(e.options,o))continue;
            ProductProgramPlan again{};
            if(product_program_replay(an,bn,load_choice(e.choice),e.choice[14],again)==SBN3_SUPPORTED &&
               again.info.arithmetic_id==e.choice[12] && again.info.execution_id==e.choice[13]){
                victim->plan=again;victim->result=SBN3_SUPPORTED;known=true;++replayed;
            }
            recorded=log==record;
            break;
        }
    }
    if(!known)victim->result=product_program_query(an,bn,o,victim->plan);
    if(victim->result==SBN3_SUPPORTED){
        derive(victim->plan,victim->derived);
        p=victim->plan;if(facts)*facts=victim->derived;
        if(record && !recorded && record->count<ProductDecisionLog::capacity){
            auto &e=record->entries[record->count++];
            e.an=an;e.bn=bn;e.options=o;std::copy_n(victim->derived.choice,16,e.choice);
        }
    }
    return victim->result;
}
void product_price_few_uses(ProductPlanMemo &memo,size_t an,size_t bn,const sbn3_mul_options &o,double uses,
                            bool pair,ProductProgramPlan &product) noexcept {
    ProductPlanMemo::Derived unused{};
    price_few_uses(&memo,an,bn,o,uses,pair,product,unused);
}
sbn3_query_result finite_query(const FiniteFormula &f, sbn3_series_range range, unsigned need,
                               const sbn3_series_options &options, FinitePlan &out,
                               ProductPlanMemo *memo) noexcept {
    if (!finite_valid(f))
        return SBN3_UNSUPPORTED;
    FinitePlan p{};
    p.formula = f;
    p.options = options;
    p.spec = {f.recipe, range, need, f.formula_id, f.parameter_id};
    const auto q = oracle(p.formula);
    const PlanningResources resources{&p.formula,options.workers,memo};
    const auto rc = query_schedule(p.spec, p.options, q, p.info, refinement(p.formula,&resources));
    if (rc == SBN3_SUPPORTED)
        out = p;
    return rc;
}
sbn3_query_result finite_compile(const FiniteFormula &f,sbn3_series_range range,unsigned need,
                                  const sbn3_series_options &options,void *storage,size_t bytes,
                                  sbn3_series_info &info,sbn3_series_plan *&out,ProductPlanMemo *memo) noexcept {
    if(!finite_valid(f))return SBN3_UNSUPPORTED;
    const sbn3_series_spec spec{f.recipe,range,need,f.formula_id,f.parameter_id};
    const auto q=oracle(f);
    const PlanningResources resources{&f,options.workers,memo};
    return compile_schedule(spec,options,q,storage,bytes,info,out,refinement(f,&resources));
}
void finite_prepare(const FinitePlan &p, sbn3_arena &arena, sbn3_team &team, const sbn3_lease &metadata,
                    const sbn3_lease &prepared, const sbn3_lease &scratch, FiniteBinding &b, sbn3_series_plan *compiled) noexcept {
    require(finite_valid(p.formula), SBN3_FATAL_ARGUMENT, "finite formula preconditions");
    require(metadata.bytes >= p.info.plan_bytes && prepared.bytes >= p.info.prepared_bytes &&
                scratch.bytes >= p.info.workspace_bytes && !(uintptr_t(prepared.data) & 127),
            SBN3_FATAL_WORKSPACE, "finite binding storage");
    b = {p, nullptr, &arena, &team, prepared, scratch};
    if(compiled){
        const uintptr_t begin=uintptr_t(metadata.data),at=uintptr_t(compiled);
        require(at>=begin && at-begin<=metadata.bytes && p.info.plan_bytes<=metadata.bytes-(at-begin),
                SBN3_FATAL_WORKSPACE,"finite compiled metadata span");
        b.schedule=compiled;
    }else{
        const auto q=oracle(b.plan.formula);
        const PlanningResources resources{&b.plan.formula,b.plan.options.workers,nullptr};
        b.schedule=prepare_schedule(b.plan.spec,b.plan.options,q,b.plan.info,metadata.data,metadata.bytes,
                                    refinement(b.plan.formula,&resources));
    }
    // Engine tables publish before any consumer is constructed. Distinct
    // table owners can prepare in parallel; the stage boundary is the barrier.
    prepare_stages(b.schedule, &team, prepare_tables, &b);
    prepare_stages(b.schedule, &team, prepare_stage, &b);
}
void finite_execute(FiniteBinding &binding, sbn3_series_values &out) noexcept {
    struct alignas(64) RootStorage {
        unsigned char bytes[(sizeof(Frame) + 63) & ~size_t(63)];
    };
    RootStorage storage[32];
    const unsigned workers = sbn3_team_workers(binding.team);
    for (unsigned j = 0; j < workers; ++j) {
        auto *root = ::new (storage[j].bytes) Frame(*binding.arena, binding.scratch);
        root->allocate(binding.scratch.bytes);
        binding.scratch_roots[j] = root;
    }
    const sbn3_series_executor executor{&binding, leaf, merge};
    sbn3_series_execute(binding.schedule, &executor, binding.team, &out, binding.scratch.data,
                        binding.scratch.bytes);
    for (unsigned j = 0; j < workers; ++j) {
        binding.scratch_roots[j]->~Frame();
        binding.scratch_roots[j] = nullptr;
    }
}
} // namespace sbn::v3::series
