#pragma once
#include <stddef.h>
namespace sbn::v3::divrem_tuning {
// Fixed resource ladder. These are policy limits, not timing estimates.
inline constexpr unsigned budget_halvings=2;
inline constexpr size_t schoolbook_budget_divisor=64;
inline constexpr size_t schoolbook_budget_quotient=8;
inline constexpr double schoolbook_budget_work=360000.0;
// Derived from the bounded reciprocal / quotient-estimate contracts.
inline constexpr unsigned correction_limit=8;
inline constexpr unsigned head_correction_limit=2;
}
