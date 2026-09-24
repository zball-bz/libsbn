#pragma once
#include "sbn3/product.h"
namespace sbn::v3 {
struct ProductStage;
/* Fused quotient terminal, 3<=m<n<=2m-1, B=2^64.
 * D: n normalized limbs. A: n+1 limbs, A[n]<=1.
 * U: m+1 limbs, U[m]==1, error <=8 against B^(2m)/D_hi.
 * Output Q has n+1 limbs, |Q-B^n*A/D|<3.
 * No n-limb inverse is produced. cached_inverse is CYC MUL(m+1,input_words),
 * input_words>=max(m+1,n-m+3); zero retains the legacy n+1 input span.
 * residual is CYC MUL(m+1,min(n,ring)). A smaller ring requires the
 * preplanned low-word lift, borrowing idle product scratch. u is reserved.
 * work[0] has input_words limbs (n+1 if zero); work[1] has output_capacity
 * limbs (ring if zero). A ring<n+3 needs output_capacity>=max(ring,n)+3 for
 * exact wrap reconstruction and a nonzero input_words<=ring. Other tail
 * storage beyond the reconstructed integer may serve E1 scratch.
 * Q may equal A exactly; no output is written before both inputs' last read.
 * Work buffers are disjoint from inputs/output/bound resources. An optional stage
 * controller reuses product storage and rebuilds the cache for correction. */
struct DivideTerminal {
    size_t m, n, ring;
    sbn3_mul_binding *cached_inverse, *residual;
    sbn3_spectrum *u;
    uint64_t *work[2];
    sbn3_team *team = nullptr;
    size_t input_words = 0;
    const ProductStage *stage = nullptr;
    size_t output_capacity = 0;
    bool retain_u_across_residual = false;
    bool fresh_residual_in_cached = false;
    size_t repair_bytes = 0;
};
void divide_terminal(const DivideTerminal &, const uint64_t *A, const uint64_t *D, const uint64_t *U,
                     uint64_t *Q) noexcept;
} // namespace sbn::v3
