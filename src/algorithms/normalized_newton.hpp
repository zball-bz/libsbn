// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/newton.h"
#include <cstddef>

namespace sbn::v3 {
class Frame;
struct NormalizedNewtonInfo {
    sbn3_newton_kind kind=SBN3_NEWTON_INVERSE;
    std::size_t n=0;
    unsigned workers=1;
    std::size_t metadata_bytes=0,workspace_bytes=0,storage_bytes=0,alignment=128;
    bool local=false;
};
struct NormalizedNewtonPrepared {
    const void* state=nullptr;
    NormalizedNewtonInfo info{};
};
// Uses the existing public Newton selection policy, preserving its local
// prefixes, native cyclic groups and half-inverse quotient terminal.
sbn3_query_result normalized_newton_query(sbn3_newton_kind,std::size_t,unsigned,
                                         NormalizedNewtonInfo&) noexcept;
// Immutable records live in caller metadata. All groups, local-prefix and
// low-witness choices are compiled before any numerical input is evaluated.
NormalizedNewtonPrepared normalized_newton_prepare(sbn3_newton_kind,std::size_t,unsigned,Frame& metadata) noexcept;
// Execute/close all transient bindings inside this admitted scope epoch.
// Metadata is reusable with distinct exclusive work Frames. D has n
// normalized words; divide A has n+1 words and A[n]<=1. Inverse takes empty A.
// Q has capacity n+1; divide allows Q==A. The error remains <3 output units.
void normalized_newton_execute(const NormalizedNewtonPrepared&,sbn3_const_limbs A,
    sbn3_const_limbs D,sbn3_limbs Q,Frame& work,sbn3_team_scope*) noexcept;
}
