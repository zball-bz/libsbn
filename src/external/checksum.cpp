// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/checksum.hpp"
#include <algorithm>
#include <cstring>
#define XXH_INLINE_ALL
#define XXH_NO_STDLIB
#define XXH_VECTOR XXH_AVX512
// Keep enough independent loads ahead of the accumulator on this native tier.
// Cached and streaming-buffer experiments include this setting in provenance.
#define XXH_PREFETCH_DIST 2048
#include "../../third_party/xxhash/xxhash.h"

namespace sbn::v3::external {
uint64_t chunk_checksum(RegionId id, uint64_t chunk, std::span<const uint8_t> payload) noexcept {
    // Exact XXH3(header || payload), using upstream's unchanged accumulate,
    // scramble and finalization primitives (xxHash 0.8.3, BSD-2-Clause).
    // The only discontiguous stripe is the first: 48 header bytes + 16 input
    // bytes. No full payload copy, streaming state, seed or format change.
    static_assert(XXH_VERSION_NUMBER==803 && XXH_STRIPE_LEN==64 && XXH3_MIDSIZE_MAX==240);
    constexpr size_t header_bytes=48,secret_limit=sizeof(XXH3_kSecret)-XXH_STRIPE_LEN;
    constexpr size_t block_stripes=secret_limit/XXH_SECRET_CONSUME_RATE;
    alignas(64) uint8_t header[XXH3_MIDSIZE_MAX];
    put64(header, 0x314b48434e4253ULL); // versioned chunk domain, little endian
    put64(header+8, id.uuid_low); put64(header+16, id.uuid_high);
    put64(header+24, id.incarnation); put64(header+32, chunk); put64(header+40, payload.size());
    if(payload.size()<=XXH3_MIDSIZE_MAX-header_bytes){
        if(!payload.empty())std::memcpy(header+header_bytes,payload.data(),payload.size());
        return XXH3_64bits(header,header_bytes+payload.size());
    }
    std::memcpy(header+header_bytes,payload.data(),XXH_STRIPE_LEN-header_bytes);
    XXH_ALIGN(XXH_ACC_ALIGN) xxh_u64 acc[XXH_ACC_NB]=XXH3_INIT_ACC;
    const auto*secret=XXH3_kSecret;
    XXH3_accumulate_512(acc,header,secret);
    size_t stripes=(payload.size()+header_bytes-1)/XXH_STRIPE_LEN-1;
    const auto*cursor=payload.data()+XXH_STRIPE_LEN-header_bytes;
    const size_t first=std::min(stripes,block_stripes-1);
    XXH3_accumulate(acc,cursor,secret+XXH_SECRET_CONSUME_RATE,first);
    cursor+=first*XXH_STRIPE_LEN;stripes-=first;
    if(first==block_stripes-1)XXH3_scrambleAcc(acc,secret+secret_limit);
    while(stripes>=block_stripes){
        XXH3_accumulate(acc,cursor,secret,block_stripes);
        XXH3_scrambleAcc(acc,secret+secret_limit);
        cursor+=block_stripes*XXH_STRIPE_LEN;stripes-=block_stripes;
    }
    if(stripes)XXH3_accumulate(acc,cursor,secret,stripes);
    // As in the upstream one-shot algorithm, the final stripe is selected
    // from the end and may overlap the previous regular stripe.
    XXH3_accumulate_512(acc,payload.data()+payload.size()-XXH_STRIPE_LEN,
                        secret+secret_limit-XXH_SECRET_LASTACC_START);
    return XXH3_finalizeLong_64b(acc,secret,header_bytes+payload.size());
}
uint64_t metadata_checksum(std::span<const uint8_t> payload) noexcept {
    return XXH3_64bits(payload.data(), payload.size());
}
} // namespace sbn::v3::external
