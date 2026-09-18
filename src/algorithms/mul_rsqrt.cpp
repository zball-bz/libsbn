#include "algorithms/mul_rsqrt.hpp"
#include "algorithms/newton_planner.hpp"
#include "algorithms/newton_contract.hpp"
#include "product/table_pool.hpp"
#include "product/cost_model.hpp"
#include "common/identity.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
namespace sbn::v3 {
namespace {
size_t up(size_t n, size_t a = 128) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "mul rsqrt alignment");
    return r;
}
MulRsqrtChoice choice(sbn3_mul_options options, const sbn3_mul_info &i, size_t tables) {
    // Pin the arithmetic family and codec, leaving deterministic geometry
    // construction to that backend. Short FFT keeps its existing policy.
    if (i.np) {
        options.prime_count = i.np;
        options.algorithm = i.algorithm;
        options.trunk_bits = int(i.trunk_bits);
    }
    return {options, i.arithmetic_id, i.execution_id, tables, i.workspace_bytes, i.workspace_alignment};
}
bool same(const MulRsqrtChoice &c, const sbn3_mul_info &i, size_t tables) {
    return c.arithmetic_id == i.arithmetic_id && c.execution_id == i.execution_id &&
           c.table_bytes == tables && c.workspace_bytes == i.workspace_bytes && c.alignment == i.workspace_alignment;
}
sbn3_product_request square_request(const MulRsqrtPlan &p) {
    sbn3_product_request r{};
    r.kind = SBN3_PRODUCT_SQR;
    r.a_limbs = p.m;
    r.cyclic_limbs = p.square_ring;
    r.window_limbs = p.m + 2;
    return r;
}
bool consuming_product(size_t an, size_t bn, unsigned workers, ProductProgramPlan &p,
                       sbn3_mul_options &o) {
    o = {}; o.workers = workers;
    if (product_program_query(an, bn, o, p) != SBN3_SUPPORTED) return false;
    if (p.contract & program_consume_inputs) return true;
    // These implementations convert every input before writing output.
    // No size threshold: accept only a backend with the required contract.
    o.algorithm = p.info.algorithm == SBN3_MUL_SCALAR ? SBN3_MUL_U52 : SBN3_MUL_FLAT;
    return product_program_query(an, bn, o, p) == SBN3_SUPPORTED && (p.contract & program_consume_inputs);
}
uint64_t plan_seal(const MulRsqrtPlan &p) {
    uint64_t h = identity::word(identity::fnv_seed, 0x4d52535152543031ULL);
    for (uint64_t x : {p.n,p.m,p.root_at,p.input_at,p.result_at,p.value_words,p.half_bytes,p.square_ring,
                       p.program_at,p.local_at[0],p.local_at[1],p.shared_at,p.tables_bytes,
                       p.workspace_at,p.workspace_bytes,p.storage_bytes,p.alignment}) h=identity::word(h,x);
    for (auto x : p.half_root.opaque) h=identity::word(h,x);
    for (const auto &c : p.choices)
        for (uint64_t x : {c.arithmetic_id,c.execution_id,c.table_bytes,c.workspace_bytes,c.alignment})
            h=identity::word(h,x);
    return h;
}
struct Multiply {
    const ProductProgram *p;
    Frame *work;
    sbn3_const_limbs a,b;
    sbn3_limbs out;
};
void multiply(void *ptr, sbn3_team_scope *scope) {
    auto &c=*static_cast<Multiply *>(ptr);
    product_program_bounded(*c.p,*c.work,scope,c.a,c.b,c.out,true);
}
} // namespace
sbn3_query_result mul_rsqrt_query(size_t n, unsigned workers, MulRsqrtPlan &out, size_t input_at) noexcept {
    if (n < 4 || n > (size_t(1)<<28) || !workers || workers > 32) return SBN3_UNSUPPORTED;
    MulRsqrtPlan p{}; p.n=n; p.m=n/2+1;
    p.root_at=up(8*(p.m+2))/8;
    p.input_at=std::max(input_at,up(8*(p.root_at+p.m+1))/8);
    p.result_at=p.root_at+p.m;
    p.value_words=up(8*std::max(p.input_at+n+1,p.root_at+n+p.m+1))/8;
    const sbn3_newton_options no{workers,0,0,0}; sbn3_newton_info half{};
    if (sbn3_newton_query(SBN3_NEWTON_RSQRT,p.m,&no,&p.half_root,&half)!=SBN3_SUPPORTED) return SBN3_UNSUPPORTED;
    p.half_bytes=half.storage_bytes;p.alignment=half.storage_alignment;
    // Reuse the existing legal low-square period/codec enumeration. The
    // other two products have their own geometry; no larger shared basis is forced.
    newton_detail::Plan selector{}; selector.options=no; newton_detail::Bundle square{};
    if (!newton_detail::choose_cycle(selector,newton_detail::Cycle::Rsqrt,p.m,n,0,false,square)) return SBN3_UNSUPPORTED;
    const auto c=selector.choices[0]; p.square_ring=c.ring;
    sbn3_mul_options so{};so.workers=c.workers;so.algorithm=c.algorithm;so.prime_count=c.np;
    so.trunk_bits=c.T;so.borrow_output=1;
    p.choices[0]=choice(so,square.producer_info.mul,square.producer_info.mul.table_bytes);
    ProductProgramPlan products[2]{}; sbn3_mul_options po[2]{};
    if (!consuming_product(p.m,n+1,workers,products[0],po[0]) ||
        !consuming_product(p.m+1,p.m+1,workers,products[1],po[1])) return SBN3_UNSUPPORTED;
    const bool deep=p.m+1>native_policy::small_model_max_words;
    p.standalone_rung_ns=newton_detail::cycle_cost(newton_detail::Cycle::Rsqrt,square,deep);
    const auto &si=square.producer_info.mul;
    const double sq=si.algorithm==SBN3_MUL_SCALAR?scalar_product_cost(p.m,p.m,si.workers)
                      :cost_model::cyclic_product(si,deep).nanoseconds*2/3;
    p.estimated_tail_ns=sq+cost_model::linear_product(products[0].info,p.m,n+1).nanoseconds+
                           cost_model::linear_product(products[1].info,p.m+1,p.m+1).nanoseconds;
    for (unsigned j=0;j<2;++j) p.choices[j+1]=choice(po[j],products[j].info,products[j].prepared_bytes);
    p.program_at=up(p.choices[0].table_bytes);
    size_t cursor=p.program_at;
    for (unsigned j=0;j<2;++j) {p.local_at[j]=cursor;cursor=up(cursor+product_program_local_bytes(products[j]));}
    p.shared_at=cursor;
    const auto t0=product_program_tables(products[0]),t1=product_program_tables(products[1]);
    cursor=up(cursor+t0.bytes);
    if (!(t0==t1)) cursor=up(cursor+t1.bytes);
    p.tables_bytes=cursor;
    p.workspace_bytes=p.half_bytes;
    for (const auto &c : p.choices) {p.alignment=std::max(p.alignment,c.alignment);p.workspace_bytes=std::max(p.workspace_bytes,c.workspace_bytes);}
    p.workspace_at=up(p.tables_bytes,p.alignment);
    p.storage_bytes=up(p.workspace_at+p.workspace_bytes);
    p.plan_id=plan_seal(p);out=p;return SBN3_SUPPORTED;
}
sbn3_const_limbs mul_rsqrt_execute(const MulRsqrtPlan &p, uint64_t a, uint64_t *v, size_t capacity,
                                   sbn3_arena &arena, size_t offset, sbn3_team &team) noexcept {
    require(p.plan_id==plan_seal(p) && a && v && !(uintptr_t(v)&127) && capacity>=p.value_words &&
                !team.busy && team.arena==&arena,
            SBN3_FATAL_ARGUMENT,"mul rsqrt bindings");
    require(offset<=arena.virtual_bytes && p.storage_bytes<=arena.virtual_bytes-offset &&
                !(uintptr_t(arena.base+offset)&(p.alignment-1)) &&
                arena.contains(offset,offset+p.storage_bytes) && arena.unleased(offset,p.storage_bytes) &&
                !overlaps(v,capacity*8,arena.base+offset,p.storage_bytes),
            SBN3_FATAL_WORKSPACE,"mul rsqrt resident arena");
    sbn3_mul_plan square_plan{};sbn3_product_info square_info{};const auto sr=square_request(p);
    require(sbn3_product_query(&sr,&p.choices[0].options,&square_plan,&square_info)==SBN3_SUPPORTED &&
                same(p.choices[0],square_info.mul,square_info.mul.table_bytes),
            SBN3_FATAL_MATH,"mul rsqrt square replay");
    ProductProgramPlan programs[2]{};
    for (unsigned j=0;j<2;++j)
        require(product_program_query(j?p.m+1:p.m,j?p.m+1:p.n+1,p.choices[j+1].options,programs[j])==SBN3_SUPPORTED &&
                    same(p.choices[j+1],programs[j].info,programs[j].prepared_bytes),
                SBN3_FATAL_MATH,"mul rsqrt product replay");
    const auto table=arena.acquire(offset+p.program_at,p.tables_bytes-p.program_at);
    ProductProgram prepared[2]{};
    {
        auto roots=Frame::borrow(arena,table,arena.base+offset+p.shared_at,p.tables_bytes-p.shared_at);
        ProductTablePool<2> pool;
        for (const auto &q : programs) if (product_program_tables(q).bytes) pool.prepare(q,roots);
        const auto view=pool.seal();
        for (unsigned j=0;j<2;++j) {
            auto local=Frame::borrow(arena,table,arena.base+offset+p.local_at[j],product_program_local_bytes(programs[j]));
            prepared[j]=product_program_prepare_pooled(programs[j],local,view);
        }
    }
    sbn3_newton_binding *half=nullptr;
    sbn3_newton_bind(&p.half_root,&arena,offset+p.workspace_at,&team,&half);
    const sbn3_newton_inputs input{{},{},a};
    sbn3_newton_execute(half,&input,{v+p.root_at,p.m+1});sbn3_newton_unbind(half);
    {
        auto st=arena.acquire(offset,p.choices[0].table_bytes);
        auto sw=arena.acquire(offset+p.workspace_at,p.choices[0].workspace_bytes);
        sbn3_mul_binding *square=nullptr;
        sbn3_product_bind(&square_plan,&arena,&st,&sw,&team,nullptr,nullptr,&square);
        const sbn3_product_inputs in{{v+p.root_at,p.m},{},{},{}};
        sbn3_product_execute(square,&in,{v,p.m+2});
        sbn3_mul_unbind(square);arena.release(sw);arena.release(st);
    }
    // Unique signed lift modulo B^(m+2); B^(2m) vanishes in this window.
    (void)parallel_limbs::mul_1(&team,v,v,p.m+2,a);
    const bool negative=v[p.m+1]>>63;
    if (negative) {parallel_limbs::complement(&team,v,p.m+2);parallel_limbs::add_word(v,p.m+2,1);}
    require(v[p.m]<newton_contract::rsqrt_residual_limit && !v[p.m+1],SBN3_FATAL_MATH,"mul rsqrt residual certificate");
    auto run=[&](unsigned j,sbn3_const_limbs x,sbn3_const_limbs y,sbn3_limbs out) {
        auto lease=arena.acquire(offset+p.workspace_at,p.choices[j+1].workspace_bytes);
        {
            Frame frame(arena,lease);Multiply call{prepared+j,&frame,x,y,out};
            sbn3_team_run(&team,multiply,&call);
        }
        arena.release(lease);
    };
    // P overwrites r and Q after their last reads. X=floor(P/B^m) is an
    // interior view, and stays disjoint from the final correction product.
    run(0,{v+p.root_at,p.m},{v+p.input_at,p.n+1},{v+p.root_at,p.n+p.m+1});
    auto *x=v+p.result_at;
    run(1,{x+p.n-p.m,p.m+1},{v,p.m+1},{v,2*p.m+2});
    const size_t first=3*p.m-p.n,count=p.n-p.m+2;
    parallel_limbs::each(&team,count,parallel_limbs::parts(&team,count),[&](size_t lo,size_t hi,unsigned) {
        for (size_t j=lo;j<hi;++j) v[j]=(v[first+j]>>1) | (first+j+1<2*p.m+2?v[first+j+1]<<63:0);
    });
    const auto spill=negative?parallel_limbs::add_to(&team,x,p.n+1,v,count):parallel_limbs::sub_from(&team,x,p.n+1,v,count);
    require(!spill,SBN3_FATAL_MATH,"mul rsqrt correction overflow");
    arena.release(table);
    return {x,p.n+1};
}
} // namespace sbn::v3
