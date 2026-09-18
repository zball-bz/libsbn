#pragma once
#include "series/formulas.hpp"
#include "runtime/arena.hpp"
#include "product/table_pool.hpp"
namespace sbn::v3 {
class Frame;
}
namespace sbn::v3::series {
// User-defined formulas use the same adapter. The context is immutable and
// remains alive through query/prepare/execute; there is no per-value vtable.
struct FiniteFormula {
    const void *context = nullptr;
    sbn3_series_recipe recipe = SBN3_SERIES_HYPERDESCENT;
    uint64_t formula_id = 0, parameter_id = 0;
    unsigned max_leaf_limbs = 0;
    sbn3_query_result (*bounds)(const void *, sbn3_series_range, uint64_t, unsigned,
                                sbn3_series_shape *) = nullptr;
    double (*work)(const void *, sbn3_series_range) = nullptr;
    void (*leaf)(const void *, uint64_t, unsigned, sbn3_series_values *) = nullptr;
    void (*batch)(const void *, sbn3_series_range, unsigned, sbn3_series_values *) = nullptr;
    // Optional private limited-value adapter. Monotone nonincreasing with the
    // interval start; includes guard words. Null preserves exact finite BSR.
    const void *limit_context = nullptr;
    size_t (*limit_words)(const void *, uint64_t begin) = nullptr;
    bool reuse_values = false;
    // Independent tree policy: does not change work(), PSR blocks or values.
    uint64_t split_policy_id = 0;
    uint64_t (*split_point)(const void *, sbn3_series_range, double fraction) = nullptr;
    sbn3_query_result (*serial_envelope)(const void *, sbn3_series_range, unsigned depth,
                                        unsigned need, uint64_t *max_terms, sbn3_series_shape *) = nullptr;
    // Optional engine-wide table directory. The engine owns its lifetime;
    // this algorithm neither builds nor releases its entries.
    ProductTablePoolView table_pool{};
    // Optional bounds for canonical whole-word-normalized mantissas. Only
    // the limited-value adapter opts in; exact integer views keep bounds().
    sbn3_query_result (*normalized_bounds)(const void *, sbn3_series_range, uint64_t, unsigned,
                                          sbn3_series_shape *) = nullptr;
    sbn3_query_result (*normalized_serial_envelope)(const void *, sbn3_series_range, unsigned, unsigned,
                                                   uint64_t *, sbn3_series_shape *) = nullptr;
    // Optional exact-span fact: words of the exact values of the whole range
    // in the representation limited execution uses (canonical whole-word
    // mantissas where the formula has them). An upper bound like bounds(), but
    // as tight as the formula can state it (capacities may be looser to stay
    // cheap and shareable); the precision planner compares it with the words
    // a node must keep. Null: bounds() of the whole range is used.
    sbn3_query_result (*exact_span)(const void *, sbn3_series_range, unsigned need, sbn3_series_shape *) = nullptr;
    // Exact short-batch bound: every raw output fits terms*words_per_term+2
    // limbs, including intermediate sums. Limited execution can compute a
    // bounded batch exactly in local storage and round only its final values.
    unsigned batch_words_per_term = 0, batch_max_terms = 0;
};
// Bounded memo of pure product planning results. product_program_query is a
// pure function of (an, bn, options), so a hit returns the identical plan. The
// memo is caller-owned, lives only through one planning pass on the calling
// thread and is never retained by a plan, schedule or binding. Eviction is
// LRU; exhaustion only repeats queries and never changes a decision.
// Append-only record of the distinct product choices of one planning pass,
// kept in caller-owned storage that outlives the plan (for example the
// immutable formula object). A later pass over the same request answers its
// memo misses by replaying the recorded canonical options instead of searching
// the candidate families again; the replayed plan must reproduce the recorded
// arithmetic and execution identities or the ordinary query is used. A full
// record only stops recording. Decisions are never taken from it blindly: the
// caller's schedule identity check still covers every replayed plan.
struct ProductDecisionLog {
    static constexpr unsigned capacity = 1024;
    struct Entry {
        size_t an = 0, bn = 0;
        sbn3_mul_options options{};
        uint64_t choice[16]{};
    };
    unsigned count = 0;
    Entry entries[capacity]{};
};
struct ProductPlanMemo {
    static constexpr unsigned capacity = 48;
    // Facts every scheduled stage reads from its chosen plan. Each costs a
    // backend plan-identity check, so they are derived once per distinct plan.
    struct Derived {
        SharedPreparation tables{};
        size_t local_bytes = 0;
        uint64_t choice[16]{};
    };
    struct Entry {
        size_t an = 0, bn = 0;
        sbn3_mul_options options{};
        sbn3_query_result result = SBN3_UNSUPPORTED;
        uint64_t stamp = 0; // zero: empty
        ProductProgramPlan plan{};
        Derived derived{};
    };
    Entry entries[capacity]{};
    uint64_t clock = 0, hits = 0, misses = 0, replayed = 0;
    ProductDecisionLog *record = nullptr;       // first pass: remember each distinct decision
    const ProductDecisionLog *replay = nullptr; // later pass: replay them on a miss
    // Short keys of the log entries (replay, record), filled lazily: a miss
    // scans two kilobytes instead of walking the 200-byte log entries.
    uint32_t keys[2][ProductDecisionLog::capacity]{};
    unsigned keyed[2]{};
    sbn3_query_result query(size_t an, size_t bn, const sbn3_mul_options &, ProductProgramPlan &,
                            Derived * = nullptr) noexcept;
};
// Re-price an already selected plan for a program that executes only `uses`
// times: execution plus its own root-table preparation, by the product
// layer's calibrated estimates. May replace a flat NTT plan by the blocked
// one; keeps pairing and the consume/bounded input contracts.
void product_price_few_uses(ProductPlanMemo &, size_t an, size_t bn, const sbn3_mul_options &, double uses,
                            bool pair, ProductProgramPlan &) noexcept;
FiniteFormula finite_formula(const Formula &) noexcept;
// Logical result bound, excluding extra storage needed to reuse parent lanes.
sbn3_query_result finite_value_shape(const FiniteFormula &, sbn3_series_range, unsigned need,
                                     sbn3_series_shape &) noexcept;
struct FinitePlan {
    FiniteFormula formula{};
    sbn3_series_spec spec{};
    sbn3_series_options options{};
    sbn3_series_info info{};
};
struct FiniteBinding {
    FinitePlan plan{};
    sbn3_series_plan *schedule = nullptr;
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    sbn3_lease prepared{}, scratch{};
    Frame *scratch_roots[32]{}; // populated only during synchronous execute
};
// The optional memo only removes repeated identical product queries inside
// this planning pass; results are identical with or without it.
sbn3_query_result finite_query(const FiniteFormula &, sbn3_series_range, unsigned need,
                               const sbn3_series_options &, FinitePlan &,
                               ProductPlanMemo * = nullptr) noexcept;
sbn3_query_result finite_compile(const FiniteFormula &, sbn3_series_range, unsigned need,
                                 const sbn3_series_options &, void *, size_t,
                                 sbn3_series_info &, sbn3_series_plan *&,
                                 ProductPlanMemo * = nullptr) noexcept;
// These three caller-owned prepared leases remain alive while the binding is
// in use. Only scratch is mutable during execute; no per-stage lease IDs.
void finite_prepare(const FinitePlan &, sbn3_arena &, sbn3_team &, const sbn3_lease &metadata,
                    const sbn3_lease &prepared, const sbn3_lease &scratch, FiniteBinding &,
                    sbn3_series_plan *compiled = nullptr) noexcept;
void finite_execute(FiniteBinding &, sbn3_series_values &) noexcept;
} // namespace sbn::v3::series
