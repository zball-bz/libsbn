#pragma once
#include "series/arithmetic.hpp"
namespace sbn::v3::series {
struct PrecisionPolicy {
    const void *context = nullptr;
    // Root-anchored contraction certificate: |U(begin,a)/D(begin,a)| <=
    // 2^-decay_bits(a), with |U/D| <= 1 on every subinterval. This is the
    // weight by which an error injected at a node starting at a reaches the
    // root value, which is all the precision rule needs (no increment
    // property between two interior indices is assumed). Monotone, zero at
    // the root begin. Local |T/D| must be <2^63. Precision and work balance
    // are separate.
    // BinaryBBP (absolute contribution convention): decay_bits(a) certifies
    // that every term at index >= a has magnitude < 2^(63-decay_bits(a))
    // relative to the root scale, so the tail state [a,end) keeps
    // fractional-decay_bits(a)/64 words; the block's own D replaces U in the
    // merge. Block exponents stay absolute; no relative block rescaling.
    uint64_t (*decay_bits)(const void *, uint64_t a) = nullptr;
    uint64_t minimum_block_terms = 128, maximum_block_terms = 0;
    double minimum_block_bits = 0; // Formula.work() is a bit-growth estimate for PSR cutting.
};
struct PsrSpec {
    FiniteFormula formula{};
    sbn3_series_range range{};
    sbn3_series_options series{};
    PrecisionPolicy precision{};
    size_t fractional_limbs = 0;
    // Bytes of storage this reduction may occupy without raising its caller's
    // resident peak (for example the terminal that follows in the same
    // region). Zero plans the minimal pool: every merge then runs in its
    // compact batch whenever the fast one does not fit the finite stages'
    // floor. A larger allowance only admits faster merge batches; it never
    // changes cuts, products or values.
    size_t storage_allowance = 0;
    // Planning rule, not a mode: a remaining range whose exact values fit the
    // words its demand keeps becomes one final exact block. False keeps
    // limited blocks everywhere (smaller peak for a tight memory budget).
    bool exact_suffix = true;
    // Optional product decision record (see ProductDecisionLog): written by
    // psr_query, replayed by psr_prepare. Caller-owned, outlives the binding.
    ProductDecisionLog *record = nullptr;
    const ProductDecisionLog *replay = nullptr;
};
struct PsrInfo {
    size_t storage_bytes = 0, storage_alignment = 128, plan_bytes = 0, value_bytes = 0, pool_bytes = 0;
    size_t T = 0, D = 0, U = 0, numerator = 0, product = 0, pool = 0;
    size_t t_limbs = 0, d_limbs = 0, u_limbs = 0, numerator_limbs = 0, product_limbs = 0, output_limbs = 0;
    unsigned blocks = 0, exact_blocks = 0; // exact: no nonzero bit is dropped anywhere in the block's tree
    uint64_t schedule_id = 0, error_units = 0;
};
struct PsrMetrics {
    uint64_t finite_ns = 0, merge_ns = 0, prepare_ns = 0;
};
struct PsrBinding {
    PsrSpec spec{};
    PsrInfo info{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease plan{}, values{};
    bool used = false;
    PsrMetrics metrics{};
};
sbn3_query_result psr_query(const PsrSpec &, PsrInfo &) noexcept;
void psr_prepare(const PsrSpec &, const PsrInfo &, sbn3_arena &, sbn3_team &, size_t offset,
                 PsrBinding &) noexcept;
// Limited numerator/positive denominator in value[0]/value[1]. Each external
// output has info.output_limbs capacity; both are adjacent halves of one
// writable caller-owned array (N first, D second), surviving pool reuse. No division
// occurs here. The exact finite service remains a separate interface.
void psr_execute(PsrBinding &, sbn3_series_values &) noexcept;
void psr_release(PsrBinding &) noexcept;
} // namespace sbn::v3::series
