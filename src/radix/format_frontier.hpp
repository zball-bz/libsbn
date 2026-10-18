// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "sbn3/radix.h"
namespace sbn::v3::radix {
// Private SRT frontier: exactly 64*fragments digits and the tree's existing
// eight-digit overlap. The digit prefix is exact for the supplied dyadic;
// overlap is exact or one low. No extra 64-digit fragment is constructed.
// Query retains the ordinary exact-prefix tie witness. Small/non-tree recipes
// report unsupported and keep the public formatter path.
sbn3_query_result format_frontier_query(unsigned base,size_t limbs,uint64_t fragments,
    const sbn3_radix_options&,sbn3_format_plan&,sbn3_format_info&) noexcept;
uint64_t format_frontier_execute(sbn3_format_binding*,sbn3_const_limbs,uint8_t*) noexcept;
}
