// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/job.hpp"
#include "external/shared.hpp"
#include "common/checked.hpp"
#include "common/identity.hpp"
#include "radix/power_bank.hpp"
#include "runtime/arena.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <new>
#include <utility>
namespace sbn::v3::external {
namespace {
constexpr size_t controller_bytes=1u<<20,series_controller=512u<<10,series_io=(1u<<20)+index_page_bytes;
bool add(uint64_t a,uint64_t b,uint64_t&out){return !__builtin_add_overflow(a,b,&out);}
unsigned bits(uint64_t x){return x?64-unsigned(__builtin_clzll(x)):0;}
int64_t input_error_shift(const TerminalSpec&t){return t.reciprocal?int64_t(t.upper_bits)-2*int64_t(t.lower_bits)+3:2-int64_t(t.lower_bits);}
bool allocation(size_t words,uint64_t&bytes){size_t n;RegionLayout layout;
    if(!mul_size(words,8,n)||region_layout(n,0,layout))return false;bytes=layout.allocation_bytes;return true;}
SourceWindow normalized_envelope(size_t storage,size_t words){
    return {storage,storage>=words?storage-words:0,words,0,storage,0,storage<words?words-storage:0};
}
uint64_t cookie(const JobPlan&p){
    const auto&s=p.spec;const auto&t=s.terminal;uint64_t h=identity::fnv_seed;
    const uint64_t words[]={s.series.formula.formula_id,s.series.formula.parameter_id,uint64_t(s.series.formula.recipe),
        s.series.range.begin,s.series.range.end,s.series.fractional_limbs,s.base,s.digits,s.tail_error_units,
        uint64_t(t.lower_bits),uint64_t(t.upper_bits),t.scalar,t.radicand,t.integer_addend,uint64_t(t.reciprocal),
        p.precision.fractional,p.precision.working,p.series.error_units,p.storage_bytes,p.write_bytes,p.peak_backing_bytes};
    for(auto v:words)h=identity::word(h,v);return h;
}
}
bool query_job_precision(unsigned base,uint64_t digits,const TerminalSpec&t,size_t maximum_blocks,uint64_t tail,JobPrecision&out) noexcept {
    radix::BaseInfo b;
    if(!radix::base_info(base,b)||!digits||digits>(uint64_t(1)<<40)||!maximum_blocks||maximum_blocks>(UINT64_MAX-1)/2||
       t.lower_bits>t.upper_bits||!t.scalar||!t.radicand)return false;
    uint64_t errors;if(!add(2*uint64_t(maximum_blocks)+1,tail,errors))return false;
    const int64_t high=t.reciprocal?-int64_t(t.lower_bits):t.upper_bits;
    const int64_t output_bits=high+bits(t.scalar)-int64_t((bits(t.radicand)-1)/2);
    if(output_bits>=64)return false;
    const uint64_t upper=output_bits>0?uint64_t(1)<<unsigned(output_bits):1;
    if(__uint128_t(upper)+t.integer_addend>UINT64_MAX)return false;
    const bool fused=t.scalar!=1||t.radicand!=1;const __uint128_t finish=__uint128_t(8)*t.scalar+(fused?(uint64_t(1)<<38):0);
    if(finish>UINT64_MAX||__uint128_t(4)*t.scalar>(__uint128_t(1)<<64))return false;
    JobPrecision p{};p.fractional=radix::node_limbs(b,(digits+63)/64)+2;
    const int64_t integer_guard=std::max(int64_t(0),high+2);
    const int64_t series_guard=std::max(int64_t(0),int64_t(bits(errors))+input_error_shift(t));
    if(!add_size(p.fractional,4+size_t((integer_guard+63)/64),p.working)||
       !add_size(p.working,std::max(size_t(4),size_t((series_guard+63)/64)),p.series)||p.series>(size_t(1)<<31))return false;
    p.finishing_error=uint64_t(finish);out=p;return true;
}
planning::ExternalPlanStatus compile_job(const JobSpec&spec,const JobBudget&budget,const planning::ExternalProductProfile&profile,
    const SeriesMetadata&metadata,std::span<uint8_t>memory,JobPlan&out) noexcept {
    using planning::ExternalPlanStatus;
    ::new(&out)JobPlan{};auto&p=out;p.spec=spec;
    const unsigned workers=spec.series.series.workers;
    if(!workers||workers>32||!query_job_precision(spec.base,spec.digits,spec.terminal,metadata.block_capacity,spec.tail_error_units,p.precision)||
       spec.series.fractional_limbs<p.precision.series)return ExternalPlanStatus::invalid;
    p.spec.format.radix.workers=workers;p.spec.newton.workers=workers;
    p.fused_root=spec.terminal.scalar!=1||spec.terminal.radicand!=1;p.write_limit=budget.write_limit;
    const size_t engine=engine_shared_bytes(),radix_shared=radix::fixed_power_bytes()+radix::digit_constant_bytes();
    size_t fixed;
    if(!add_size(budget.other_live_bytes,sizeof(JobPlan),fixed)||!add_size(fixed,metadata.bytes,fixed)||
       !add_size(fixed,engine+radix_shared+controller_bytes,p.common_fixed_bytes)||p.common_fixed_bytes>=budget.resident_bytes)
        return ExternalPlanStatus::no_admitted_plan;
    const size_t available=(budget.resident_bytes-p.common_fixed_bytes)&~(p.alignment-1);
    const size_t total=p.common_fixed_bytes+available;p.series_work=p.alignment;
    SeriesBudget series_budget{total,p.common_fixed_bytes-metadata.bytes-engine-series_controller+p.series_work-series_io,
                               UINT64_MAX,budget.other_owned_disk_bytes,UINT64_MAX};
    auto status=compile_series(p.spec.series,series_budget,{true,spec.force_stored_depth},profile,metadata,memory,p.series);
    if(status!=ExternalPlanStatus::ready)return status;
    if(p.series.work_alignment>p.alignment||!add_size(p.series_work,p.series.work_bytes,p.storage_bytes))return ExternalPlanStatus::unsupported;
    const size_t n=p.precision.working;size_t capacity[2];
    for(unsigned j=0;j<2;++j){auto source=p.series.result_shape[j].source;
        if(!resolve_source_window(p.series.result_shape[j].limbs,source)||!source.storage_limbs)return ExternalPlanStatus::unsupported;
        capacity[j]=source.storage_limbs;
    }
    const auto numerator=normalized_envelope(capacity[spec.terminal.reciprocal?1:0],n),
               denominator=normalized_envelope(capacity[spec.terminal.reciprocal?0:1],n);
    InverseBudget quotient_budget{total,p.common_fixed_bytes-sizeof(QuotientPlan)-engine,UINT64_MAX,budget.other_owned_disk_bytes,UINT64_MAX};
    status=query_quotient_sources(numerator,denominator,n,p.spec.newton,quotient_budget,profile,p.quotient);
    if(status!=ExternalPlanStatus::ready)return status;
    p.storage_bytes=std::max(p.storage_bytes,p.quotient.storage_bytes);
    if(p.fused_root){InverseBudget root_budget{total,p.common_fixed_bytes-sizeof(MulRsqrtPlan)-engine,UINT64_MAX,budget.other_owned_disk_bytes,UINT64_MAX};
        status=query_mul_rsqrt_source({n+1,0,n+1},n,4,spec.terminal.radicand,spec.terminal.scalar,p.spec.newton,root_budget,profile,p.root);
        if(status!=ExternalPlanStatus::ready)return status;p.storage_bytes=std::max(p.storage_bytes,p.root.storage_bytes);
    }
    const int64_t hi=spec.terminal.reciprocal?-int64_t(spec.terminal.lower_bits):spec.terminal.upper_bits;
    if(!query_fixed_output_source({n+1,0,n+1},hi+2-int64_t(64*n),p.precision.fractional,p.precision.finishing_error,p.fixed,spec.terminal.integer_addend))return ExternalPlanStatus::unsupported;
    p.storage_bytes=std::max(p.storage_bytes,p.fixed.storage_bytes);
    FormatBudget format_budget{total,p.common_fixed_bytes-sizeof(FormatPlan)-engine-radix_shared-65536,UINT64_MAX,budget.other_owned_disk_bytes,UINT64_MAX};
    status=query_format_source({p.precision.fractional+1,0,p.precision.fractional+1},p.precision.fractional,spec.base,spec.digits,p.spec.format,format_budget,profile,p.format);
    if(status!=ExternalPlanStatus::ready)return status;p.storage_bytes=std::max(p.storage_bytes,p.format.storage_bytes);
    p.phase_writes[0]=p.series.write_bytes;p.phase_writes[1]=p.quotient.worst_write_bytes;
    p.phase_writes[2]=p.fused_root?p.root.worst_write_bytes:0;p.phase_writes[3]=p.fixed.write_bytes;p.phase_writes[4]=p.format.write_bytes;
    p.phase_backing[0]=p.series.peak_backing_bytes;p.phase_backing[1]=p.quotient.peak_backing_bytes;
    p.phase_backing[2]=p.fused_root?p.root.peak_backing_bytes:0;p.phase_backing[4]=p.format.peak_backing_bytes;
    uint64_t input;
    if(!allocation(n+1,input)||!add(input,p.fixed.output_allocation,p.phase_backing[3])||
       !add(p.phase_backing[3],budget.other_owned_disk_bytes,p.phase_backing[3]))return ExternalPlanStatus::unsupported;
    for(unsigned i=0;i<5;++i){if(!add(p.write_bytes,p.phase_writes[i],p.write_bytes))return ExternalPlanStatus::unsupported;
        p.peak_backing_bytes=std::max(p.peak_backing_bytes,p.phase_backing[i]);}
    if(!align_size(p.storage_bytes,p.alignment,p.storage_bytes)||!add_size(p.common_fixed_bytes,p.storage_bytes,p.total_resident_bytes))return ExternalPlanStatus::unsupported;
    p.minimum_regions=std::max(p.series.minimum_free_regions,2*size_t(p.format.powers)+64);
    if(p.total_resident_bytes>budget.resident_bytes||p.peak_backing_bytes>budget.owned_disk_limit||p.write_bytes>budget.write_limit)
        return ExternalPlanStatus::no_admitted_plan;
    p.plan_id=cookie(p);p.ready=true;return ExternalPlanStatus::ready;
}
IoError advance_job(JobPlan&p,JobState&s,Transport&t,ValueStore&store,sbn3_arena&arena,size_t offset,sbn3_team&team) noexcept {
    if(!p.ready||team.arena!=&arena||team.width!=p.spec.series.series.workers||team.busy||
       p.storage_bytes>SIZE_MAX-offset||!arena.contains(offset,offset+p.storage_bytes)||!arena.unleased(offset,p.storage_bytes)||
       (uintptr_t(arena.base+offset)&(p.alignment-1))||(s.plan_id&&s.plan_id!=p.plan_id))return {IoCode::invalid,"constant job binding"};
    if(!s.plan_id){if(s.next!=JobPhase::series||s.working.value||s.binary.value||s.text.digits)return {IoCode::invalid,"constant job initial state"};
        for(const auto&v:s.pair)if(v.value)return {IoCode::invalid,"constant job initial pair"};s.plan_id=p.plan_id;}
    if(s.next==JobPhase::complete)return {};
    if(unsigned(s.next)>unsigned(JobPhase::complete))return {IoCode::invalid,"constant job phase"};
    uint64_t remaining=0;for(unsigned i=unsigned(s.next);i<5;++i)require(add(remaining,p.phase_writes[i],remaining),SBN3_FATAL_SIZE,"constant job write sum");
    if(s.charged_write_bytes>p.write_limit||remaining>p.write_limit-s.charged_write_bytes)
        return {IoCode::capacity,"constant job cumulative writes",0,remaining,s.charged_write_bytes<=p.write_limit?p.write_limit-s.charged_write_bytes:0};
    if(t.counters.write_submitted>t.write_limit||remaining>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"constant job transport writes",0,remaining,t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    const auto usage=store.usage();
    if(usage.live_bytes+usage.free_bytes<p.peak_backing_bytes)
        return {IoCode::capacity,"constant job Store admission",0,p.peak_backing_bytes,usage.live_bytes+usage.free_bytes};
    if(store.available_regions()<p.minimum_regions)
        return {IoCode::capacity,"constant job Store slots",0,p.minimum_regions,store.available_regions()};
    if(t.cancel.test())return {IoCode::cancelled,"constant job safe point"};
    const auto saved=t.write_limit,before=t.counters.write_submitted;
    const uint64_t room=p.write_limit-s.charged_write_bytes;
    t.write_limit=std::min(saved,room>UINT64_MAX-before?UINT64_MAX:before+room);
    struct Charge {Transport&t;JobState&s;uint64_t saved,before;~Charge(){t.write_limit=saved;
        require(add(s.charged_write_bytes,t.counters.write_submitted-before,s.charged_write_bytes),SBN3_FATAL_SIZE,"constant job write charge");}} charge{t,s,saved,before};
    if(s.next==JobPhase::series){
        auto io=arena.acquire(offset,p.series.io_bytes),work=arena.acquire(offset+p.series_work,p.series.work_bytes);
        struct Release {sbn3_arena&a;sbn3_lease&io,&work;~Release(){a.release(work);a.release(io);}} release{arena,io,work};
        SeriesResult result;if(auto e=run_series(p.series,t,store,arena,team,work,{static_cast<uint8_t *>(io.data),io.bytes},result))return e;
        require(result.in_store,SBN3_FATAL_LIFETIME,"constant job stored series");s.pair=std::move(result.stored);s.next=JobPhase::quotient;return {};
    }
    if(s.next==JobPhase::quotient){const unsigned a=p.spec.terminal.reciprocal?1:0,d=a^1;
        require(rebind_stored_quotient(p.quotient,s.pair[a],s.pair[d]),SBN3_FATAL_MATH,"constant job quotient source envelope");
        const auto&terminal=p.spec.terminal;uint64_t error;
        require(add(p.series.error_units,p.spec.tail_error_units,error),SBN3_FATAL_MATH,"constant job series error");
        const __int128 power=-64*__int128(p.spec.series.fractional_limbs)+(terminal.reciprocal?1-2*__int128(terminal.lower_bits):0)-p.quotient.result_exponent2;
        require(__int128(bits(error))+power<=0,SBN3_FATAL_MATH,"constant job error scale certificate");
        if(terminal.reciprocal)require(__int128(bits(error))-64*__int128(p.spec.series.fractional_limbs)<=__int128(terminal.lower_bits)-1,
            SBN3_FATAL_MATH,"constant job reciprocal domain certificate");
        StoredInteger result;if(auto e=run_stored_quotient(p.quotient,t,store,s.pair[a],s.pair[d],result,arena,offset,team))return e;
        s.working=std::move(result);s.pair={};s.next=p.fused_root?JobPhase::root:JobPhase::fixed;return {};
    }
    if(s.next==JobPhase::root){StoredInteger result;
        if(auto e=run_stored_mul_rsqrt(p.root,t,store,s.working,result,arena,offset,team))return e;
        s.working=std::move(result);s.next=JobPhase::fixed;return {};
    }
    if(s.next==JobPhase::fixed){FixedOutputPlan actual;
        require(query_fixed_output(s.working,p.precision.fractional,p.precision.finishing_error,actual,p.spec.terminal.integer_addend)&&
            actual.storage_bytes<=p.fixed.storage_bytes&&actual.write_bytes==p.fixed.write_bytes,SBN3_FATAL_MATH,"constant job fixed output envelope");
        auto lease=arena.acquire(offset,actual.storage_bytes);struct Release {sbn3_arena&a;sbn3_lease&l;~Release(){a.release(l);}} release{arena,lease};
        StoredInteger result;if(auto e=run_fixed_output(actual,t,store,s.working,result,{static_cast<uint8_t *>(lease.data),lease.bytes}))return e;
        s.binary=std::move(result);s.working.value.reset();s.next=JobPhase::radix;return {};
    }
    if(auto e=run_stored_format(p.format,t,store,s.binary,s.text,arena,offset,team))return e;
    s.next=JobPhase::complete;return {};
}
IoError run_job(JobPlan&p,JobState&s,Transport&t,ValueStore&store,sbn3_arena&arena,size_t offset,sbn3_team&team) noexcept {
    do{if(auto e=advance_job(p,s,t,store,arena,offset,team))return e;}while(s.next!=JobPhase::complete);return {};
}
} // namespace sbn::v3::external
