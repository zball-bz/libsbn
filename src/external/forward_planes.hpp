// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/chunk_writer.hpp"
#include "product/outer_transform.hpp"
#include "external/source_window.hpp"
namespace sbn::v3::external {
struct ForwardPlanesSpec {
    size_t input_limbs=0,outer_cells=0,inner_cells=0,live_rows=0,stripe_cells=0;
    unsigned trunk_bits=0,prime_count=0,prime_batch=0,workers=1;
    SourceWindow source{};
    bool ring_checksum=false; // m64 for a general minus-ring middle window
};
struct ForwardPlanesPlan {
    ForwardPlanesSpec spec{};
    product::OuterTransformSpec kernel{};
    product::OuterTransformInfo info{};
    size_t storage_bytes=0,prepared=0,work=0,raw=0,packed=0,transfer=0,index_page=0,input_index=0;
    size_t transfer_bytes=0,input_index_bytes=0;
    uint64_t input_bytes=0,row_stride=0,plane_bytes=0;
    uint64_t source_bytes=0,source_offset=0,valid_input_bytes=0,source_read_bytes=0;
    uint64_t source_capacity_bytes=0;
    size_t serial_storage_bytes=0,pipeline_storage_bytes=0,single_raw_pipeline_bytes=0,second_raw=0,second_packed=0;
    bool pipeline=false,pipeline_double_raw=false;
    bool parallel_writes=false; // independent complete prime fragments, provider permitting
};
bool query_forward_planes(const ForwardPlanesSpec &,ForwardPlanesPlan &) noexcept;
// Two packed prime-batch slots and one/two raw-stripe lifetimes. The one-raw
// variant waits for its last compute reader before loading the next stripe.
// Every source stripe is loaded exactly once across all its prime batches.
bool enable_forward_pipeline(ForwardPlanesPlan &,size_t available) noexcept;
// Execution-only binding adjustment within the already admitted input/index
// capacity. Arithmetic roots/shape and workspace offsets remain unchanged.
bool rebind_forward_source(ForwardPlanesPlan &,SourceWindow) noexcept;
// Input is immutable. Output ChunkWriters are empty, scratch-kind, disjoint
// allocations on the same transport. The caller owns their index/coverage and
// IO grants. All outputs stay unpublished; failure abandons them together.
IoError run_forward_planes(const ForwardPlanesPlan &,Transport &,const Region &input,
                           std::span<ChunkWriter> outputs,std::span<uint8_t>workspace,sbn3_team &,
                           uint64_t *input_m61=nullptr) noexcept;
inline size_t forward_memory_workspace(const ForwardPlanesPlan&p) noexcept {return p.work+p.info.workspace_bytes;}
// Contiguous resident source after applying source windows/shifts, zero-padded
// through kernel.input_cells*trunk_bits+64. Generate one prime directly into
// canonical packed48. All three grants are disjoint; no stripe copy or bank.
IoError run_forward_memory_prime(const ForwardPlanesPlan &,unsigned prime,
    std::span<const uint8_t> source,std::span<uint8_t> plane,
    std::span<uint8_t> workspace,sbn3_team &) noexcept;
struct PrefixForwardPlan {
    product::OuterTransformSpec kernel{};
    product::OuterTransformInfo info{};
    size_t cached_bytes=0,first_row=0,raw_bytes=0,storage_bytes=0,prepared=0,work=0,raw[2]{};
    bool pipeline=false;
};
bool query_prefix_forward(const ForwardPlanesPlan &,size_t cached_bytes,PrefixForwardPlan &,size_t max_stripe_cells=0) noexcept;
// Data traffic for one logical slice, with optional stripe-word cuts. The
// pinned index is separate and read only once by the containing product.
bool source_slice_reads(const SourceWindow &,size_t offset,size_t count,size_t stripe_words,uint64_t &bytes) noexcept;
IoError run_forward_prefix_prime(const ForwardPlanesPlan &,const PrefixForwardPlan &,unsigned prime,ValueReader &,
    std::span<const uint8_t> prefix,std::span<uint8_t> plane,std::span<uint8_t> workspace,sbn3_team &,
    uint64_t *suffix_m61=nullptr) noexcept;
} // namespace sbn::v3::external
