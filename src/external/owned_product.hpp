// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/values.hpp"
#include "external/spilled_product.hpp"
namespace sbn::v3::external {
// Reserves every output/scratch extent before dispatch. Immutable inputs remain
// owned by the caller; only a successfully sealed result is published. Every
// temporary reservation is returned on success/error, after synchronous joins.
IoError run_owned_product(const SpilledProductPlan &,Transport &,ValueStore &,
                         const StoredValue &,const StoredValue &,StoredValue &,
                         std::span<uint8_t>,sbn3_team &,WordExtent * = nullptr,CyclicResidualResult * = nullptr,
                         const StoredValue *subtract=nullptr,MiddleWindowResult *middle=nullptr) noexcept;
IoError run_owned_cached_product(const SpilledProductPlan &,Transport &,ValueStore &,RowCacheBinding &,
                         const StoredValue &fresh,StoredValue &,std::span<uint8_t>,sbn3_team &,WordExtent * = nullptr,CyclicResidualResult * = nullptr,
                         const StoredValue *subtract=nullptr,MiddleWindowResult *middle=nullptr) noexcept;
} // namespace sbn::v3::external
