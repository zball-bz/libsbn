#pragma once
#include "sbn3/product.h"
namespace sbn::v3 {
struct ProductStage;
/* One normalized reciprocal rung, B=2^64, 3<=m<n<=2m-1.
 * D has n limbs with its top bit set. U has m+1 limbs, U[m]==1,
 * and differs by at most 8 from B^(2m)/floor(D/B^(n-m)).
 * Output has n+1 limbs: B^n <= V < 2B^n, |V-B^(2n)/D| < 3.
 *
 * product is a prebound CYC MUL (A=m+1, B=min(n,ring)). A smaller
 * ring requires a preplanned low-product witness using dead product scratch.
 * u is its reserved, not yet computed A spectrum. All tables exist before
 * entry. correction holds the bound input span, or is unused when
 * consume_residual is supported by the cached product. residual holds ring limbs when
 * ring>=n+3, otherwise max(ring,n)+3 for the exact low-word-assisted lift.
 * Output, U, D and these buffers are mutually disjoint, except output==U is
 * allowed when that allocation has n+1 words. An optional preplanned stage controller may
 * rebind resident workspace; no heap/page allocation or plan search occurs. */
struct InverseRung {
    size_t m, n, ring;
    sbn3_mul_binding *product;
    sbn3_spectrum *u;
    uint64_t *residual, *correction;
    sbn3_team *team = nullptr; // idle team for large value passes; optional in clients
    const ProductStage *stage = nullptr;
    sbn3_mul_binding *correction_product = nullptr;
    size_t correction_input_words = 0; // zero preserves the legacy n-word binding
    bool consume_residual = false;
    size_t repair_bytes = 0;
};
// Native residual metrics include the initial cache-build/apply episode;
// correction metrics describe only the subsequent cached product.
void inverse_rung(const InverseRung &, const uint64_t *D, const uint64_t *U, uint64_t *V,
                  sbn3_product_metrics *residual_counts = nullptr,
                  sbn3_product_metrics *correction_counts = nullptr) noexcept;
} // namespace sbn::v3
