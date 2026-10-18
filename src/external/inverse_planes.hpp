// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/scratch_plane.hpp"
#include "product/outer_transform.hpp"
namespace sbn::v3::external {
struct InversePlanesPlan {
    product::OuterTransformSpec spec{};
    product::OuterTransformInfo info{};
    product::OuterEmitInfo fused{};
    size_t storage_bytes=0,prepared=0,work=0,coefficients=0,coefficient_bytes=0;
    uint64_t row_stride=0,plane_bytes=0;
    bool pipeline=false; // independent prime batches share the already quoted bank
};
bool query_inverse_planes(const product::OuterTransformSpec &,InversePlanesPlan &,bool fused_emit=false) noexcept;
struct InversePlanesBinding {
    const InversePlanesPlan *plan=nullptr;
    std::span<uint8_t> workspace{};
    sbn3_team *team=nullptr;
};
// No IO during prepare. Workspace and plan remain caller-owned and exclusive.
void prepare_inverse_planes(const InversePlanesPlan &,std::span<uint8_t>,sbn3_team &,InversePlanesBinding &) noexcept;
struct CoefficientStripe {
    std::span<const uint64_t> data{}; // [prime][coefficient row][column][8 lanes], lazy [0,4p)
    size_t primes=0,rows=0,columns=0;
    uint64_t first_column=0;
};
// Inputs are completed inner-product rows before inverse outer transform. Time
// rows beyond live_rows are mathematically zero. No plane is rewritten; a
// successful borrowed stripe lasts until the next call reuses the workspace.
// Multi-batch parallel plans overlap checked reads and inverse computation in
// disjoint prime spans of that bank. All batches join before exposing the stripe.
IoError read_inverse_stripe(InversePlanesBinding &,uint64_t column,std::span<ScratchPlane> results,
                            CoefficientStripe &) noexcept;
// Split stages for an outer pipeline. Load writes packed data into the bank;
// only transform exposes a completed CoefficientStripe. The bank is exclusive
// through both stages, with all IO joined before the numeric stage begins.
IoError load_inverse_stripe(InversePlanesBinding &,uint64_t column,std::span<ScratchPlane>,sbn3_team_scope *read_scope=nullptr) noexcept;
void transform_inverse_stripe(InversePlanesBinding &,uint64_t column,sbn3_team_scope *,CoefficientStripe &) noexcept;
} // namespace sbn::v3::external
