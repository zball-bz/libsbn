// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "product/program.hpp"
#include "sbn3/series.h"
namespace sbn::v3::series {
// Prepared consuming merge shared by the resident PSR and external planner.
// Tables/work and all value capacities are admitted before dispatch. x/y may
// consume the block value lanes; result is one contiguous numerator/denominator
// pair. Rounding boundaries are part of the existing PSR numerical policy.
struct PrecisionMergeCall {
    const ProductProgram *program, *contraction;
    Frame *work;
    const sbn3_series_values *block;
    sbn3_series_values *result;
    sbn3_series_value x, y;
    uint64_t *a, *b, *fallback;
    size_t pair_words;
    size_t precision;
    sbn3_series_recipe recipe;
};
void precision_merge_action(void *, sbn3_team_scope *);
} // namespace sbn::v3::series
