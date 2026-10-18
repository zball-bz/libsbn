// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/mul.h"
#include <cstddef>

namespace sbn::v3 {
class Frame;
enum class NormalizedLocalKind { inverse, divide };
struct NormalizedLocalInfo {
    NormalizedLocalKind kind=NormalizedLocalKind::inverse;
    std::size_t n=0,metadata_bytes=0,workspace_bytes=0,storage_bytes=0,alignment=128;
    unsigned record_count=0,products=0,seed_words=0;
};
// Query/prepare select every local, middle, peeled and compact group, including
// their otherwise-uncached products and low witnesses. No such query occurs
// in execute. This is the local subset of the normalized Newton service;
// larger/nonlocal requests report unsupported for its native provider.
sbn3_query_result normalized_local_query(NormalizedLocalKind,std::size_t,
                                        NormalizedLocalInfo&) noexcept;
struct NormalizedLocalPrepared {
    const void* state=nullptr;
    NormalizedLocalInfo info{};
};
// Allocates only within metadata, whose owner retains the region. No numerical
// input is read. The resulting metadata is immutable and may be reused with
// independent work Frames, including on active Team scope leaders.
NormalizedLocalPrepared normalized_local_prepare(NormalizedLocalKind,std::size_t,Frame& metadata) noexcept;
// D: n normalized limbs. Inverse ignores empty A, writes n+1 words U with
// U[n]==1 and |U-B^(2n)/D|<3. Divide takes n+1-word A, A[n]<=1, and writes
// |Q-B^n*A/D|<3, using only a half-precision inverse. Q may equal A exactly;
// all other value/work/metadata overlaps are invalid. The unused work suffix
// starts at info.alignment and has at least info.workspace_bytes. Execution is serial
// within the caller's admitted scope and never starts a Team episode.
void normalized_local_execute(const NormalizedLocalPrepared&,sbn3_const_limbs A,
                              sbn3_const_limbs D,sbn3_limbs Q,Frame& work) noexcept;
}
