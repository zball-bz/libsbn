// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/forward_planes.hpp"
#include "external/values.hpp"
#include "product/modular_row.hpp"
namespace sbn::v3::external {
struct SpilledProductPlan;
enum class RowCacheKind { outer_rows, completed_columns };
struct RowCachePlan {
    ForwardPlanesPlan forward{};
    size_t storage_bytes=0,control=0,indexes=0,index_stride=0,bits=0,bits_stride=0,io_data=0,io_page=0;
    uint64_t owned_disk_bytes=0,normal_read_bytes=0,normal_write_bytes=0;
    RowCacheKind kind=RowCacheKind::outer_rows;
    size_t build_bytes=0,raw=0,raw_bytes=0,source_index=0,prime=0,phase=0,phase_bytes=0;
    size_t row_prepared=0,row_scratch=0,row_array=0;
    product::ModularRowInfo column{};
};
// Ephemeral immutable outer rows, before the cross twist. Index expectations
// stay in admitted RAM; disposal needs no index writeback. It is not a durable
// cache format. Producer and all consumers use the same private codec/scale.
bool query_row_cache(const ForwardPlanesPlan &,RowCachePlan &,RowCacheKind=RowCacheKind::outer_rows) noexcept;
struct RowCacheBinding {void *control=nullptr;};
IoError build_row_cache(const RowCachePlan &,Transport &,ValueStore &,const StoredValue &,SourceWindow,
                       std::span<uint8_t> retained,std::span<uint8_t> work,sbn3_team &,RowCacheBinding &) noexcept;
void release_row_cache(RowCacheBinding &) noexcept;
bool row_cache_compatible(const RowCacheBinding &,const SpilledProductPlan &,const Transport &,std::span<uint8_t>) noexcept;
bool row_cache_disjoint(const RowCacheBinding &,const Region &) noexcept;
SourceWindow row_cache_source(const RowCacheBinding &) noexcept;
uint64_t row_cache_residue(const RowCacheBinding &) noexcept;
std::span<ScratchPlane> row_cache_planes(RowCacheBinding &) noexcept;
} // namespace sbn::v3::external
