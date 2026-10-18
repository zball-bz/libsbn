// Original libsbn code; LGPL-3.0-or-later.
#include "float/constant_bank.hpp"
#include "float/constant_bank_internal.hpp"
#include "float/constants.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <new>

namespace sbn::v3::floating {
namespace {
using constant_bank_detail::Values;
using constant_bank_detail::checked;
using constant_bank_detail::radius_good;
using constant_bank_detail::value_magic;
constexpr std::uint64_t work_magic=UINT64_C(0x53424e3343425731);
struct Builder {
    std::uint64_t marker=0;std::uint32_t active=0;unsigned workers=1;
    std::size_t bytes=0;Values* values=nullptr;ColdConstantPrepared pi{},log2{};ConstantBankView view{};
};
}
ConstantBankQuote constant_bank_requirements(ConstantBankDemand demand,unsigned workers) noexcept {
    ConstantBankQuote q;q.workers=workers;
    if((!demand.pi_bits&&!demand.log2_bits)||demand.pi_bits>float_precision_limit||demand.log2_bits>float_precision_limit||!workers||workers>32){
        q.status=support::unsupported;q.reason="constant bank demand";return q;}
    q.pi_bits=demand.pi_bits;q.log2_bits=demand.log2_bits;
    if(q.log2_bits){const auto dependency=cold_constant_working_precision(bits{q.log2_bits});
        if(!dependency){q.status=support::unsupported;q.reason="constant bank log2 precision";return q;}
        q.pi_bits=std::max(q.pi_bits,dependency);}
    bool ok=real_append(q.value_bytes,sizeof(Values),128);
    for(auto bits:{q.pi_bits,q.log2_bits})if(bits)ok=ok&&real_append(q.value_bytes,sizeof(float_buffer),64)&&real_append(q.value_bytes,((bits+63)/64)*8,64);
    const auto pi=cold_constant_requirements(bits{q.pi_bits},workers,ConstantKind::pi);
    // Precision-only internal binding is handled by the producer's quote;
    // it cannot read a value or perform numerical work during admission.
    const auto log=q.log2_bits?cold_constant_requirements(bits{q.log2_bits},workers,ConstantKind::log2,
        constant_bank_query_binding({q.pi_bits,0})):requirements{};
    if(!pi.supported()||!log.supported()){q.status=support::unsupported;q.reason="constant bank producer support";return q;}
    q.workspace_alignment=std::max({std::size_t(128),pi.scratch_alignment,log.scratch_alignment});
    ok=ok&&real_append(q.workspace_bytes,sizeof(Builder),128)&&real_append(q.workspace_bytes,pi.scratch_bytes,pi.scratch_alignment)&&
        real_append(q.workspace_bytes,log.scratch_bytes,log.scratch_alignment);
    if(!ok){q.status=support::unsupported;q.reason="constant bank storage extent";}return q;
}
ConstantBankPrepared constant_bank_prepare(const ConstantBankQuote& supplied,std::span<std::byte> values,std::span<std::byte> workspace) noexcept {
    const auto q=constant_bank_requirements({supplied.pi_bits,supplied.log2_bits},supplied.workers);
    require(q.supported()&&q.pi_bits==supplied.pi_bits&&q.log2_bits==supplied.log2_bits&&q.value_bytes==supplied.value_bytes&&
        q.workspace_bytes==supplied.workspace_bytes,SBN3_FATAL_ARGUMENT,"constant bank quote identity");
    require(values.size()>=q.value_bytes&&workspace.size()>=q.workspace_bytes&&
        !(std::uintptr_t(values.data())&(q.value_alignment-1))&&!(std::uintptr_t(workspace.data())&(q.workspace_alignment-1)),
        SBN3_FATAL_WORKSPACE,"constant bank storage");
    require(!overlaps(values.data(),values.size(),workspace.data(),workspace.size()),SBN3_FATAL_ARGUMENT,"constant bank value/work overlap");
    auto value_frame=Frame::external(values.data(),q.value_bytes);auto* v=::new(value_frame.allocate(sizeof(Values),128))Values{};
    v->bytes=q.value_bytes;v->pi_bits=q.pi_bits;v->log2_bits=q.log2_bits;
    auto make=[&](std::size_t bits){auto* raw=value_frame.allocate(sizeof(float_buffer),64);const auto limbs=(bits+63)/64;
        auto* data=value_frame.alloc<std::uint64_t>(limbs);return ::new(raw)float_buffer({data,limbs},sbn::bits{bits});};
    v->pi=make(q.pi_bits);if(q.log2_bits)v->log2=make(q.log2_bits);v->marker=value_magic;
    auto work_frame=Frame::external(workspace.data(),q.workspace_bytes);auto* b=::new(work_frame.allocate(sizeof(Builder),128))Builder{};
    b->values=v;b->workers=q.workers;b->bytes=q.workspace_bytes;b->view={v,values.first(q.value_bytes),q.pi_bits,q.log2_bits};b->marker=work_magic;
    ConstantBankPrepared result{b,workspace.first(q.workspace_bytes),b->view};
    const auto pi=cold_constant_requirements(bits{q.pi_bits},q.workers,ConstantKind::pi);
    auto* pi_work=static_cast<std::byte*>(work_frame.allocate(pi.scratch_bytes,pi.scratch_alignment));
    b->pi=cold_constant_prepare(bits{q.pi_bits},q.workers,ConstantKind::pi,{pi_work,pi.scratch_bytes});
    if(q.log2_bits){const auto log=cold_constant_requirements(bits{q.log2_bits},q.workers,ConstantKind::log2,
            constant_bank_query_binding({q.pi_bits,0}));
        auto* log_work=static_cast<std::byte*>(work_frame.allocate(log.scratch_bytes,log.scratch_alignment));
        b->log2=cold_constant_prepare(bits{q.log2_bits},q.workers,ConstantKind::log2,{log_work,log.scratch_bytes},constant_bank_pending_binding(result));}
    return result;
}
ConstantBankBinding constant_bank_pending_binding(const ConstantBankPrepared& prepared) noexcept {
    require(prepared.state&&prepared.state==prepared.workspace.data()&&prepared.workspace.size()>=sizeof(Builder),SBN3_FATAL_LIFETIME,"constant bank pending builder");
    const auto& b=*static_cast<const Builder*>(prepared.state);require(b.marker==work_magic&&checked(prepared.pending)==b.values,SBN3_FATAL_LIFETIME,"constant bank pending identity");
    return ConstantBankBinding(prepared.pending,true);
}
namespace {
void build(void* opaque,sbn3_team_scope* scope){auto& b=*static_cast<Builder*>(opaque);auto& v=*b.values;const execution ex{{},nullptr,scope};
    v.pi_report=cold_constant_execute(b.pi,v.pi->ref(),ex);
    require(radius_good(v.pi->view(),v.pi_report.error,v.pi_bits,ConstantKind::pi),SBN3_FATAL_MATH,"constant bank pi publication radius");
    __atomic_store_n(&v.pi_ready,1,__ATOMIC_RELEASE);
    if(v.log2){v.log2_report=cold_constant_execute(b.log2,v.log2->ref(),ex);
        require(radius_good(v.log2->view(),v.log2_report.error,v.log2_bits,ConstantKind::log2),SBN3_FATAL_MATH,"constant bank log2 publication radius");
        __atomic_store_n(&v.log2_ready,1,__ATOMIC_RELEASE);}
    __atomic_store_n(&v.ready,1,__ATOMIC_RELEASE);
}
}
ConstantBankView constant_bank_execute(const ConstantBankPrepared& prepared,execution ex) noexcept {
    require(prepared.state&&prepared.state==prepared.workspace.data()&&prepared.workspace.size()>=sizeof(Builder),SBN3_FATAL_LIFETIME,"constant bank builder");
    auto& b=*static_cast<Builder*>(prepared.state);require(b.marker==work_magic&&b.bytes==prepared.workspace.size()&&checked(prepared.pending)==b.values&&ex.workers()>=b.workers,
        SBN3_FATAL_ARGUMENT,"constant bank execution shape");
    std::uint32_t idle=0;require(__atomic_compare_exchange_n(&b.active,&idle,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"constant bank concurrent construction");
    if(!__atomic_load_n(&b.values->ready,__ATOMIC_ACQUIRE)){
        if(ex.scope)build(&b,ex.scope);else if(ex.team)sbn3_team_run(ex.team,build,&b);else serial_team_run(build,&b);}
    __atomic_store_n(&b.active,0,__ATOMIC_RELEASE);return b.view;
}
}
