// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
namespace sbn::v3::floating {
// Read-only, private query descriptions. Never called by arithmetic. Optional
// components use SIZE_MAX for unknown; they are subsets of total scratch, not
// additional allocations. Descriptions come from the production quote helpers.
struct FloatDiagnostic {
    requirements need{};
    const char* recipe="unknown";
    std::size_t working_bits=SIZE_MAX,metadata_bytes=SIZE_MAX;
    std::size_t constant_value_bytes=SIZE_MAX,constant_work_bytes=SIZE_MAX,table_bytes=SIZE_MAX;
};
FloatDiagnostic division_diagnostic(float_ref,float_view,float_view,bool,unsigned) noexcept;
FloatDiagnostic exp_series_diagnostic(bits,bool,bool,unsigned) noexcept;
FloatDiagnostic trig_series_diagnostic(bits,unsigned) noexcept;
FloatDiagnostic exponential_diagnostic(float_ref,float_view,bool,unsigned) noexcept;
FloatDiagnostic logarithm_diagnostic(bits,unsigned) noexcept;
FloatDiagnostic elliptic_diagnostic(bits,unsigned) noexcept;
FloatDiagnostic agm_diagnostic(bits,unsigned) noexcept;
FloatDiagnostic trig_diagnostic(bits,bits,float_view,bool,unsigned) noexcept;
FloatDiagnostic root_diagnostic(float_ref,float_view,bool,unsigned) noexcept;
FloatDiagnostic rootn_diagnostic(bits,std::uint64_t,unsigned) noexcept;
}
