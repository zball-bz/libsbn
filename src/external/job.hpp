// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/series.hpp"
#include "external/newton.hpp"
#include "external/format.hpp"
namespace sbn::v3::external {
// True positive series sum S is in [2^lower_bits,2^upper_bits]. The result is
// integer_addend + scalar*(reciprocal?1/S:S)/sqrt(radicand). Bounds and tail
// error are mathematical certificates supplied with the custom formula.
struct TerminalSpec {
    int lower_bits=0,upper_bits=1;
    uint64_t scalar=1,radicand=1,integer_addend=0;
    bool reciprocal=false;
};
struct JobPrecision {size_t fractional=0,working=0,series=0;uint64_t finishing_error=0;};
bool query_job_precision(unsigned base,uint64_t digits,const TerminalSpec &,size_t maximum_blocks,
                         uint64_t tail_error_units,JobPrecision &) noexcept;
struct JobSpec {
    series::PsrSpec series{}; // stable formula contexts; range already chosen
    TerminalSpec terminal{};
    unsigned base=10;
    uint64_t digits=0,tail_error_units=1; // |true S - finite S| <= tail_units*B^-series.precision
    FormatOptions format{};
    InverseOptions newton{};
    unsigned force_stored_depth=0; // correctness-placement override
};
using JobBudget=InverseBudget;
enum class JobPhase { series,quotient,root,fixed,radix,complete };
struct JobPlan {
    JobSpec spec{};
    JobPrecision precision{};
    SeriesPlan series{};
    QuotientPlan quotient{};
    MulRsqrtPlan root{};
    FixedOutputPlan fixed{};
    FormatPlan format{};
    bool fused_root=false,ready=false;
    size_t storage_bytes=0,alignment=size_t(2)<<20,total_resident_bytes=0,common_fixed_bytes=0,series_work=0,minimum_regions=0;
    uint64_t phase_writes[5]{},phase_backing[5]{},write_bytes=0,peak_backing_bytes=0,write_limit=UINT64_MAX,plan_id=0;
};
// Compile at a stable address. The plan, series metadata and formula contexts
// stay live. Every backend choice is fixed before any operands/inverse exist.
planning::ExternalPlanStatus compile_job(const JobSpec &,const JobBudget &,const planning::ExternalProductProfile &,
    const SeriesMetadata &,std::span<uint8_t>metadata,JobPlan &) noexcept;
struct JobState {
    uint64_t plan_id=0,charged_write_bytes=0;
    JobPhase next=JobPhase::series;
    StoredTriple pair{};
    StoredInteger working{},binary{};
    FormatResult text{};
};
// One atomic numerical phase. IO error/cancellation retains the previous state;
// failed submitted writes remain charged. Success moves to the next safe point.
// A caller can checkpoint the current values between calls. Any older root must
// be released after the new root commits and before the next numerical phase;
// retaining arbitrary history requires an additional declared backing budget.
// No durable publication is performed by this state machine itself.
IoError advance_job(JobPlan &,JobState &,Transport &,ValueStore &,sbn3_arena &,size_t offset,sbn3_team &) noexcept;
IoError run_job(JobPlan &,JobState &,Transport &,ValueStore &,sbn3_arena &,size_t offset,sbn3_team &) noexcept;
} // namespace sbn::v3::external
