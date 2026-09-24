#pragma once
#include "product/window.hpp"
#include "product/program.hpp"

namespace sbn::v3::product {
// Adapter from the existing ordinary-product program machinery. Any backend
// supplying those callbacks also obtains exact windows/cancellation through
// FullWindowFactory; it need not implement a reciprocal or division kernel.
// An active caller-owned scope covers the complete dependent operation group.
struct ProgramMultiply {
    sbn3_mul_options options{};
    sbn3_team_scope *scope=nullptr;
    size_t workspace_bytes(size_t a,size_t b) const noexcept {
        ProductProgramPlan p;
        require(product_program_query(a,b,options,p)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"full window program support");
        const size_t capacity=sbn3_mul_output_capacity(&p.info);
        return p.prepared_bytes+p.info.workspace_bytes+(capacity>a+b?8*capacity:0)+
               2*std::max<size_t>(128,p.info.workspace_alignment)+512;
    }
    void operator()(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,Frame &space) const noexcept {
        FrameMark mark(space);ProductProgramPlan plan;
        require(product_program_query(an,bn,options,plan)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"full window program support");
        auto table=space.subframe(plan.prepared_bytes+128,128);
        const auto program=product_program_prepare(plan,table);
        auto work=space.subframe(plan.info.workspace_bytes+128,plan.info.workspace_alignment);
        auto *value=program.output_capacity>an+bn?space.alloc<uint64_t>(program.output_capacity):out;
        product_program_execute(program,work,scope,{a,an},{b,bn},{value,program.output_capacity});
        if(value!=out)memcpy(out,value,(an+bn)*8);
    }
};
} // namespace sbn::v3::product
