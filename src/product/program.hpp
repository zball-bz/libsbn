#pragma once
#include "product/backend.hpp"
#include "runtime/team.hpp"
namespace sbn::v3 {
struct ProductProgramPlan {
    sbn3_mul_plan plan{};
    sbn3_mul_info info{};
    size_t prepared_bytes = 0;
    size_t an = 0, bn = 0;
    unsigned contract = 0;
    size_t pair_workspace_bytes = 0;
};
struct ProductProgram {
    const Backend *backend = nullptr;
    const void *prepared = nullptr;
    size_t an = 0, bn = 0, workspace_bytes = 0, workspace_alignment = 0;
    unsigned workers = 0;
    size_t prepared_bytes = 0;
    unsigned contract = 0;
    size_t pair_workspace_bytes = 0;
    const void *shared_prepared = nullptr;
    size_t shared_prepared_bytes = 0;
    size_t output_capacity = 0;
};
inline sbn3_query_result product_program_describe(size_t an,size_t bn,ProductProgramPlan &p) {
    const auto *backend=backend_lookup(p.plan.opaque[1]);
    if(!backend || !backend->program_prepare || !backend->program_execute || !backend->program_bytes)
        return SBN3_UNSUPPORTED;
    p.prepared_bytes=backend->program_bytes(p.plan);p.an=an;p.bn=bn;
    p.contract=backend->program_contract?backend->program_contract(p.plan):0;
    p.pair_workspace_bytes=backend->program_pair_bytes?backend->program_pair_bytes(p.plan):0;
    return p.prepared_bytes?SBN3_SUPPORTED:SBN3_UNSUPPORTED;
}
inline sbn3_query_result product_program_query(size_t an,size_t bn,const sbn3_mul_options &options,ProductProgramPlan &p) {
    const sbn3_product_spec spec{an,bn};
    const auto rc=sbn3_mul_query(&spec,&options,&p.plan,&p.info);
    return rc==SBN3_SUPPORTED?product_program_describe(an,bn,p):rc;
}
inline sbn3_query_result product_program_replay(size_t an,size_t bn,const sbn3_mul_options &options,
                                               uint64_t backend_id,ProductProgramPlan &p) {
    const auto *backend=backend_lookup(backend_id);
    if(!backend)return SBN3_UNSUPPORTED;
    const sbn3_product_spec spec{an,bn};
    const auto rc=backend->query(spec,options,p.plan,p.info);
    return rc==SBN3_SUPPORTED?product_program_describe(an,bn,p):rc;
}
inline SharedPreparation product_program_tables(const ProductProgramPlan &p) {
    const auto *backend = backend_lookup(p.plan.opaque[1]);
    return backend && backend->program_tables ? backend->program_tables(p.plan) : SharedPreparation{};
}
// An execution-only alternative of a compiled plain product. Keep the
// arithmetic geometry/codec fixed; do not repeat the NP/T search to fit RAM.
inline sbn3_mul_options product_program_batch_options(const ProductProgramPlan &p, unsigned batch) {
    const auto &i=p.info;
    const auto *backend=backend_lookup(p.plan.opaque[1]);
    if(backend && backend->program_options){auto o=backend->program_options(p.plan);o.prime_batch=batch;o.workspace_budget=0;return o;}
    sbn3_mul_options o{};
    o.workers=i.workers; o.prime_count=i.np; o.trunk_bits=int(i.trunk_bits);
    o.algorithm=i.algorithm; o.prime_batch=batch; o.borrow_output=i.borrow_output;
    o.crt_mode=i.crt_mode; o.codec_mode=i.codec_mode;
    o.column_log2=unsigned(__builtin_ctzll(i.C));
    o.row_log2=unsigned(__builtin_ctzll(i.M2));
    o.fused_start_skew_us=i.fused_start_skew_us?int(i.fused_start_skew_us):-1;
    return o;
}
inline sbn3_query_result product_program_rebatch(const ProductProgramPlan &p, unsigned batch,
                                                 ProductProgramPlan &out, sbn3_mul_options *used = nullptr) {
    if (p.info.algorithm!=SBN3_MUL_BAILEY || !p.info.fused || !batch || batch>p.info.np)
        return SBN3_UNSUPPORTED;
    auto o=product_program_batch_options(p,batch);
    for(unsigned attempt=0;attempt<2;++attempt){
        ProductProgramPlan next{};
        if(product_program_replay(p.an,p.bn,o,p.plan.opaque[1],next)==SBN3_SUPPORTED &&
           next.info.arithmetic_id==p.info.arithmetic_id && next.contract==p.contract &&
           next.prepared_bytes==p.prepared_bytes){
            out=next;if(used)*used=o;return SBN3_SUPPORTED;
        }
        // The default lattice policy can add full/rowscale. Geometry alone
        // is not its entire arithmetic identity; retry that policy and check.
        o.column_log2=o.row_log2=0;
    }
    return SBN3_UNSUPPORTED;
}
inline size_t product_program_local_bytes(const ProductProgramPlan &p) {
    const auto *backend = backend_lookup(p.plan.opaque[1]);
    return product_program_tables(p).bytes ? backend->program_local_bytes(p.plan) : p.prepared_bytes;
}
inline ProductProgram product_program_prepare(const ProductProgramPlan &p, Frame &tables,
                                               const void *shared = nullptr) {
    const auto *backend = backend_lookup(p.plan.opaque[1]);
    const size_t before = tables.used();
    require(!shared || (product_program_tables(p).bytes && backend->program_prepare_shared &&
                       !(uintptr_t(shared) & (product_program_tables(p).alignment - 1))),
            SBN3_FATAL_ARGUMENT, "product program shared tables");
    const void *data = shared ? backend->program_prepare_shared(p.plan, tables, shared)
                              : backend->program_prepare(p.plan, tables);
    require(tables.used() - before <= (shared ? product_program_local_bytes(p) : p.prepared_bytes) + 127, SBN3_FATAL_WORKSPACE,
            "product program preparation");
    const size_t constructed_bytes =
        size_t(tables.data() + tables.used() - static_cast<const unsigned char *>(data));
    return {backend,
            data,
            p.an,
            p.bn,
            p.info.workspace_bytes,
            p.info.workspace_alignment,
            p.info.workers,
            constructed_bytes,
            p.contract,
            p.pair_workspace_bytes,
            shared,
            shared ? product_program_tables(p).bytes : 0,
            sbn3_mul_output_capacity(&p.info)};
}
inline void product_program_pair(const ProductProgram &p, Frame &work, sbn3_team_scope *scope,
                                 sbn3_const_limbs common, sbn3_const_limbs fresh0,
                                 sbn3_const_limbs fresh1, sbn3_limbs out0, sbn3_limbs out1,
                                 bool common_on_right = false) {
    require(p.backend && p.prepared && p.pair_workspace_bytes && p.backend->program_pair_execute && scope &&
                common.count <= (common_on_right ? p.bn : p.an) &&
                fresh0.count <= (common_on_right ? p.an : p.bn) &&
                fresh1.count <= (common_on_right ? p.an : p.bn) && common.count && fresh0.count && fresh1.count &&
                out0.capacity >= p.an + p.bn && out1.capacity >= p.an + p.bn &&
                sbn3_team_width(scope) >= p.workers,
            SBN3_FATAL_ARGUMENT, "product pair shape");
    require(!(uintptr_t(work.data()) & (p.workspace_alignment - 1)) &&
                work.capacity() >= p.pair_workspace_bytes &&
                !overlaps(work.data(), work.capacity(), p.shared_prepared, p.shared_prepared_bytes),
            SBN3_FATAL_WORKSPACE, "product pair scratch", p.pair_workspace_bytes, work.capacity());
    const size_t written = (p.an + p.bn) * 8;
    require(!overlaps(out0.data, written, out1.data, written) &&
                !overlaps(out0.data, written, fresh1.data, fresh1.count * 8),
            SBN3_FATAL_ARGUMENT, "product pair live second input/output alias");
    const sbn3_const_limbs spans[]{common, fresh0, fresh1, {out0.data, p.an + p.bn}, {out1.data, p.an + p.bn}};
    for (const auto v : spans) {
        valid_span(v.data, v.count * 8, "product pair span");
        require(!overlaps(v.data, v.count * 8, work.data(), work.capacity()) &&
                    !overlaps(v.data, v.count * 8, p.prepared, p.prepared_bytes) &&
                    !overlaps(v.data, v.count * 8, p.shared_prepared, p.shared_prepared_bytes),
                SBN3_FATAL_ARGUMENT, "product pair resource alias");
    }
    require_scope_leader(scope);
    sbn3_team_scope subset{scope->team, scope->first, p.workers, false, scope->epoch};
    FrameMark mark(work);
    p.backend->program_pair_execute(p.prepared, work, &subset, common, fresh0, fresh1, out0, out1);
}
inline void product_program_run(const ProductProgram &p, Frame &work, sbn3_team_scope *scope,
                                sbn3_const_limbs a, sbn3_const_limbs b, sbn3_limbs out, bool bounded,
                                bool consume) {
    const bool lengths = bounded ? (p.contract & program_bounded_inputs) && a.count <= p.an && b.count <= p.bn
                                 : a.count == p.an && b.count == p.bn;
    require(p.backend && p.prepared && lengths && out.capacity >= p.output_capacity && scope &&
                sbn3_team_width(scope) >= p.workers,
            SBN3_FATAL_ARGUMENT, "product program execution");
    require(!(uintptr_t(work.data()) & (p.workspace_alignment - 1)) && work.capacity() >= p.workspace_bytes &&
                !overlaps(work.data(), work.capacity(), p.shared_prepared, p.shared_prepared_bytes),
            SBN3_FATAL_WORKSPACE, "product program scratch", p.workspace_bytes, work.capacity());
    require(consume ? bool(p.contract & program_consume_inputs)
                    : !overlaps(a.data, a.count * 8, out.data, p.output_capacity * 8) &&
                          !overlaps(b.data, b.count * 8, out.data, p.output_capacity * 8),
            SBN3_FATAL_ARGUMENT, "product program alias");
    const sbn3_const_limbs spans[]{a, b, {out.data, p.output_capacity}};
    for (const auto &v : spans)
        require(!overlaps(v.data, v.count * 8, work.data(), work.capacity()) &&
                    !overlaps(v.data, v.count * 8, p.prepared, p.prepared_bytes) &&
                    !overlaps(v.data, v.count * 8, p.shared_prepared, p.shared_prepared_bytes),
                SBN3_FATAL_ARGUMENT, "product program resource alias");
    FrameMark mark(work);
    require_scope_leader(scope);
    sbn3_team_scope subset{scope->team, scope->first, p.workers, false, scope->epoch};
    p.backend->program_execute(p.prepared, work, &subset, a, b, out);
}
inline void product_program_execute(const ProductProgram &p, Frame &work, sbn3_team_scope *scope,
                                    sbn3_const_limbs a, sbn3_const_limbs b, sbn3_limbs out) {
    product_program_run(p, work, scope, a, b, out, false, false);
}
inline void product_program_bounded(const ProductProgram &p, Frame &work, sbn3_team_scope *scope,
                                    sbn3_const_limbs a, sbn3_const_limbs b, sbn3_limbs out,
                                    bool consume = false) {
    product_program_run(p, work, scope, a, b, out, true, consume);
}
} // namespace sbn::v3
