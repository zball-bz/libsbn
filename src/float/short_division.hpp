// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/newton.h"
#include <cstddef>
#include <cstdint>

namespace sbn::v3 {
class Frame;
namespace floating {
struct ShortDivisionInfo {
    sbn3_newton_kind kind=SBN3_NEWTON_DIVIDE;
    std::size_t n=0,k=0,metadata_bytes=0,workspace_bytes=0,storage_bytes=0,alignment=128;
    unsigned workers=1;
    bool local=false;
};
struct ShortDivisionPrepared {const void* state=nullptr;ShortDivisionInfo info{};};
sbn3_query_result short_division_query(sbn3_newton_kind,std::size_t n,std::size_t k,
                                      unsigned workers,ShortDivisionInfo&) noexcept;
ShortDivisionPrepared short_division_prepare(sbn3_newton_kind,std::size_t n,std::size_t k,
                                             unsigned workers,Frame& metadata) noexcept;
// 3<=k<n/2. D has k raw limbs with a nonzero high word; Dnorm=D<<clz(D[k-1]).
// DIV stages A=floor(A_raw*2^a_shift) as n normalized limbs and returns
// floor(A*B^k/Dnorm). The default shift admits an already normalized A.
// INVERSE takes empty A and returns floor(B^(n+k)/Dnorm). B=2^64.
// Q has n+2 limbs. This is <1 raw unit from the normalized real quotient.
// Metadata is immutable/reusable. Read-only A/D may overlap. Q is disjoint
// from both; all values are disjoint from metadata and the unused work suffix.
// Execution uses an existing admitted scope, without queries.
void short_division_execute(const ShortDivisionPrepared&,sbn3_const_limbs A,
                             sbn3_const_limbs D,sbn3_limbs Q,Frame& work,
                             sbn3_team_scope*,std::int64_t a_shift=0) noexcept;
}
}
