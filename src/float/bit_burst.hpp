// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/trig_agm.hpp"
namespace sbn::v3::floating {
enum class BitBurstKind {exp,expm1,sincos};
struct BitBurstLayout {
    requirements need{};
    std::size_t absolute_bits=0,working_bits=0,halvings=0,slices=0;
    std::size_t metadata_bytes=0,shared_table_budget=0,live_value_bytes=0,phase_bytes=0;
    std::size_t max_leaf_limbs=0,max_exact_limbs=0;
};
struct BitBurstStats {std::size_t nonzero_slices=0,max_runtime_leaf_limbs=0,shared_table_bytes=0;};
struct FloatBitBurstPrepared {void* state=nullptr;std::span<std::byte> storage{};};
// Absolute evaluation target on |r|<=1. Output quantization is added to the
// report separately; a relative target near zero belongs to the caller.
// All slice/product geometry is frozen in metadata. Equal-key immutable
// tables use a bounded shared directory; uncached tables are rebuilt from
// those selected plans in ONE quoted slice phase, without geometry queries.
BitBurstLayout bit_burst_program_layout(bits absolute_bits,unsigned,BitBurstKind) noexcept;
requirements bit_burst_program_requirements(bits absolute_bits,unsigned,BitBurstKind) noexcept;
FloatBitBurstPrepared bit_burst_program_prepare(bits absolute_bits,unsigned,BitBurstKind,std::span<std::byte>) noexcept;
BitBurstStats bit_burst_statistics(const FloatBitBurstPrepared&) noexcept;
float_report exp_bit_burst_execute(const FloatBitBurstPrepared&,float_ref,float_view,
    execution={},rounding=rounding::nearest_even) noexcept;
TrigSincosReport sincos_bit_burst_execute(const FloatBitBurstPrepared&,float_ref,float_ref,float_view,
    execution={},rounding=rounding::nearest_even) noexcept;
// Magnitude-aware point adapters. Tiny analytic leaves are selected before
// an extreme virtual exponent can enlarge the absolute working precision.
requirements bit_burst_requirements(float_ref,float_view,BitBurstKind,unsigned=1) noexcept;
float_report bit_burst_into(float_ref,float_view,BitBurstKind,execution={},rounding=rounding::nearest_even) noexcept;
requirements bit_burst_sincos_requirements(float_ref,float_ref,float_view,unsigned=1) noexcept;
TrigSincosReport bit_burst_sincos_into(float_ref,float_ref,float_view,execution={},rounding=rounding::nearest_even) noexcept;
}
