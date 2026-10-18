// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "sbn3/team.h"
#include "product/crt.hpp"
#include <span>
namespace sbn::v3::product {
struct OuterTransformSpec {
    size_t outer_cells=0,inner_cells=0,live_rows=0,stripe_cells=0;
    uint64_t input_cells=0;
    unsigned trunk_bits=0,prime_count=0,prime_batch=0,workers=1;
};
struct OuterTransformInfo {
    size_t prepared_bytes=0,worker_bytes=0,workspace_bytes=0,shared_bytes=0;
    size_t raw_stride=0,raw_bytes=0,fragment_bytes=0,packed_bytes=0,coefficient_bytes=0;
    unsigned workers=0;
};
bool p48_outer_query(const OuterTransformSpec &,OuterTransformInfo &) noexcept;
void p48_outer_prepare(const OuterTransformSpec &,const OuterTransformInfo &,std::span<uint8_t>) noexcept;
// One verified raw stripe, zero-filled past the source and with quoted readable
// guards. The wide codec shares each byte funnel across the active prime batch.
// Output: canonical packed48, [prime-in-batch][BR row][stripe column].
void p48_outer_forward(std::span<const uint8_t> prepared,uint64_t column,
                        unsigned prime_begin,unsigned prime_count,std::span<const uint8_t> raw,
                        std::span<uint8_t> packed,std::span<uint8_t> workspace,sbn3_team_scope *) noexcept;
// Contiguous normalized input, padded to input_cells*trunk_bits + 64 bytes.
// Writes one prime directly at the named column in a complete packed row plane.
void p48_outer_forward_resident(std::span<const uint8_t> prepared,uint64_t column,unsigned prime,
                        std::span<const uint8_t> source,std::span<uint8_t> plane,size_t row_stride,
                        std::span<uint8_t> workspace,sbn3_team_scope *) noexcept;
// Cell-aligned normalized prefix plus a staged suffix stripe. Suffix row zero
// corresponds to floor(prefix_cells/inner_cells); each staged row has raw_stride.
// Prefix has 64 readable guard bytes. One prime is written directly to its plane.
void p48_outer_forward_cached(std::span<const uint8_t> prepared,uint64_t column,unsigned prime,
    std::span<const uint8_t> prefix,size_t prefix_cells,std::span<const uint8_t> suffix,
    std::span<uint8_t> plane,size_t row_stride,std::span<uint8_t> workspace,sbn3_team_scope *) noexcept;
// Each input row starts at a cells*64 stride, with cells*48 packed bytes at its
// start. Expansion is in place; inverse consumes a Fourier prefix and a known
// zero coefficient tail. Output is normalized lazy64 in the same row slots.
void p48_outer_inverse(std::span<const uint8_t> prepared,unsigned prime_begin,unsigned prime_count,
                        std::span<uint8_t> coefficient_bank,std::span<uint8_t> workspace,
                        sbn3_team_scope *) noexcept;
struct OuterEmitInfo {
    size_t tile_cells=0,tiles=0,worker_bytes=0,workspace_bytes=0;
    size_t coefficients=0,packed=0,tails=0;
    unsigned workers=0;
};
bool p48_outer_emit_query(const OuterTransformSpec &,OuterEmitInfo &) noexcept;
// Contiguous packed48 input [prime][row][column], with 64 readable guard bytes,
// left immutable. Complete a cache-sized
// tile across all primes, reconstruct its integers, then resolve tile carries.
// Output is one zero-incoming-carry stripe per row, plus its exact carry tail.
void p48_outer_inverse_emit(std::span<const uint8_t> prepared,std::span<const uint8_t> bank,
    const CrtBackend &,std::span<const uint8_t> crt_prepared,std::span<uint8_t> output,size_t row_stride,
    std::span<CrtCarry> row_tails,std::span<uint8_t> workspace,sbn3_team_scope *) noexcept;
} // namespace sbn::v3::product
