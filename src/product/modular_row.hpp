// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "sbn3/team.h"
#include <span>
namespace sbn::v3::product {
enum class RowOperation { forward, inverse, product };
struct ModularRowSpec {
    RowOperation operation = RowOperation::product;
    unsigned prime = 0, global_log = 0, workers = 1;
    size_t cells = 0; // power of two; each cell has eight polynomial coefficients
    uint64_t global_prefix = 0;
    uint64_t local_tau = 1, product_scale = 1;
    bool a_transformed=false,b_transformed=false; // product inputs already in the named inner Fourier basis
};
// Local Fourier evaluations are local_tau * z_cells^k, in bit-reversed order.
// Polynomial leaf wrap is z_global^(global_prefix + (global_N/cells)*k).
// These contexts are distinct. Inverse is normalized; product_scale multiplies
// the product of the provided inputs (the adapter cancels IFMA REDC itself).
struct ModularRowInfo {
    size_t prepared_bytes = 0, scratch_bytes = 0, array_bytes = 0;
    size_t leaf_cells = 0, leaf_count = 0, worker_bytes = 0;
    unsigned workers = 0, depth = 0;
    size_t shared_bytes = 0; // published root bank, charged once by the context
};
bool p48_row_query(const ModularRowSpec &, ModularRowInfo &) noexcept;
uint64_t p48_row_prime(unsigned) noexcept;
uint64_t p48_row_coset(unsigned prime, unsigned global_log, uint64_t natural_prefix) noexcept;
void p48_row_prepare(const ModularRowSpec &, const ModularRowInfo &, std::span<uint8_t> prepared) noexcept;
// Inputs/outputs are lazy [0,4p), 64-byte aligned and caller-owned. Product
// consumes both arrays, returns B; exact A==B is SQR. Unary operations ignore A.
// No row-sized allocation per worker, IO, allocation or planning during execute.
void p48_row_execute(std::span<const uint8_t> prepared, std::span<uint8_t> scratch,
                     std::span<uint64_t> a, std::span<uint64_t> b, sbn3_team_scope *) noexcept;
// Exact-base in-place expand/compact is supported; other overlaps are rejected.
// A packed source includes 16 readable padding bytes beyond cells*48.
void p48_row_unpack(std::span<const uint8_t> prepared, std::span<const uint8_t> packed,
                    std::span<uint64_t> unpacked) noexcept;
void p48_row_pack(std::span<const uint8_t> prepared, std::span<const uint64_t> unpacked,
                  std::span<uint8_t> packed) noexcept;
// Private IO staging inside expanded arrays. Each block reserves cells*64
// bytes, initially containing cells*48 packed bytes at its beginning. Workers
// expand backward/compact forward within disjoint blocks. Output B has the same
// blocked packing; A is consumed. This is not a persistent plane format.
inline constexpr size_t p48_row_io_cells=32768;
void p48_row_blocked_product(std::span<const uint8_t> prepared,std::span<uint8_t> scratch,
                            std::span<uint64_t> a,std::span<uint64_t> b,sbn3_team_scope *) noexcept;
struct PackedRowBatchSpec {
    unsigned prime=0,global_log=0,workers=1;
    size_t cells=0;
    uint64_t first_row=0,rows=0,row_stride=0,product_scale=1;
    bool square=false;
    RowOperation operation=RowOperation::product;
    bool a_transformed=false,b_transformed=false;
};
// One joined region covers all rows. Workers unpack directly from immutable
// packed sources, run the existing row arithmetic, then pack back into B.
// Exact packed bounds require no readable overrun; one unpacked row pair and
// the existing prepared/worker grants suffice independently of the row count.
// The unpacked arrays are scratch, not outputs. Deep rows may read packed
// inputs in the first butterflies and emit packed B in the final butterflies.
void p48_rows_packed(const PackedRowBatchSpec &,const ModularRowInfo &,
    std::span<const uint8_t> packed_a,std::span<uint8_t> packed_b,
    std::span<uint8_t> prepared,std::span<uint8_t> scratch,
    std::span<uint64_t> a,std::span<uint64_t> b,sbn3_team_scope *) noexcept;
} // namespace sbn::v3::product
