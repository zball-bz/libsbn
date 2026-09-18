#pragma once
// Exact product programs and the power rail shared by the radix conversion trees.
#include "radix/geometry.hpp"
#include "product/program.hpp"
#include "product/table_pool.hpp"
namespace sbn::v3::radix {
inline constexpr unsigned max_rail = 36, max_shared_tables = 48;
// Schedule policy of the conversion trees (measured on Zen 5, see docs/radix-conversion-results).
struct TreePolicy {
    // Smaller nodes are frontier tasks (one worker runs the subtree), but never so large that the
    // frontier has fewer than about frontier_tasks_per_worker tasks per worker.
    size_t team_node_limbs = 32768;
    size_t frontier_tasks_per_worker = 1;
    size_t frontier_floor_limbs = 512;
    // A perfect node halves its worker group when it is at least fork_node_limbs large and the halves
    // keep fork_min_workers, or when it is below team_node_limbs. In between, sixteen concurrent
    // single-worker products fall out of the shared cache (measured 2-5x slower per product at
    // 50K-250K limbs), four-worker groups do not.
    size_t fork_node_limbs = 800000;
    unsigned fork_min_workers = 4;
    size_t wide_product_limbs = 600000; // smaller products run on at most eight workers (one chiplet)
    // Integer parts below this many limbs are formatted by schoolbook division: its execute time crosses the
    // tree's at 32 limbs, the tree's bind (reciprocal, programs) is repaid by one conversion from about 256.
    size_t integer_tree_limbs = 256, integer_tree_limbs_repeated = 32;
    bool cyclic_products = true;        // frontier splits as wrap-around products with cached rail spectra
    bool ring_products = true;          // staged splits in the NTT band as wrap-around products (product service)
    size_t ring_node_limbs = 40000;     // from this node size on
    unsigned ring_min_count = 8;        // and only for stages with this many nodes (the spectrum is built per run)
};
// Process-wide policy; probes may change it before planning (not during).
TreePolicy &tree_policy() noexcept;
inline constexpr size_t rail_basecase_limbs = 12;
// Plan-time facts of one exact product (an x bn limbs on `workers` workers).
struct ProductShape {
    size_t an = 0, bn = 0;
    unsigned workers = 1, requested = 1; // resolved execution width; width asked of the query
    size_t prepared_bytes = 0;            // complete private preparation
    size_t local_bytes = 0;    // preparation beside the shareable tables
    SharedPreparation tables{};
    size_t work_bytes = 0, work_alignment = 64;
    size_t output_limbs = 0;
    uint64_t arithmetic_id = 0;
    // Bytes of [output][padding][workspace] when the output starts 64-byte aligned.
    size_t episode_bytes() const noexcept {
        return ((output_limbs * 8 + 63) & ~size_t(63)) + (work_alignment > 64 ? work_alignment - 64 : 0) + work_bytes;
    }
    // The same plus a program prepared next to it (one-time products).
    size_t temporary_bytes() const noexcept { return ((prepared_bytes + 127) & ~size_t(127)) + 512 + episode_bytes(); }
};
sbn3_query_result product_shape(size_t an, size_t bn, unsigned workers, ProductShape &, ProductProgramPlan *keep = nullptr);
// Immutable preparation of a set of programs: [shared tables, each built once][per program state].
struct ProgramSetPlan {
    SharedPreparation shared[max_shared_tables]{};
    size_t shared_offset[max_shared_tables]{};
    unsigned shared_count = 0;
    size_t shared_bytes = 0, local_bytes = 0, alignment = 128;
    void account(const ProductShape &) noexcept;
    size_t bytes() const noexcept { return ((shared_bytes + 127) & ~size_t(127)) + local_bytes; }
};
class ProgramSetBuilder {
    const ProgramSetPlan &plan_;
    Arena &arena_;
    sbn3_lease lease_;
    Frame local_;
    const void *data_[max_shared_tables]{};
public:
    ProgramSetBuilder(const ProgramSetPlan &, Arena &, const sbn3_lease &prepared) noexcept;
    // Replays the query of `expected` (fatal when it no longer matches) and prepares the program.
    ProductProgram prepare(const ProductShape &expected) noexcept;
};
struct TeamProduct {
    const ProductProgram *program;
    Frame *work;
    sbn3_const_limbs a, b;
    sbn3_limbs out;
};
void run_product(sbn3_team *, TeamProduct &) noexcept;
// One exact product on the whole team with a program built inside `scratch` and discarded.
void temporary_product(sbn3_team *, unsigned workers, Frame &scratch, const uint64_t *a, size_t an, const uint64_t *b,
                       size_t bn, uint64_t *out, size_t out_capacity) noexcept;
// ---- rail: odd^(64 * 2^k), k < count, each zero padded to its capacity -------------------------------
struct RailPlan {
    unsigned count = 0;
    size_t limbs[max_rail]{}, offset[max_rail]{};
    size_t total_limbs = 0;
    uint64_t square_id[max_rail]{}; // arithmetic identity of rail[k]^2 when it runs as a program
    size_t setup_bytes = 0;         // scratch of the largest squaring
};
sbn3_query_result rail_finish(const BaseInfo &, unsigned workers, RailPlan &) noexcept;
// The scratch lease is released around squarings that run through the product service (its leases are
// exclusive) and acquired again: the caller's lease is updated in place.
void rail_build(const BaseInfo &, const RailPlan &, unsigned workers, Arena &, sbn3_team &, sbn3_lease &scratch,
                uint64_t *storage, const uint64_t **entries) noexcept;
// ---- odd^(64 F) for any F, from the rail (rail.count > top bit of F) ----------------------------------
inline constexpr size_t chain_basecase_limbs = 24;
struct Chain {
    unsigned first = 0, steps = 0;
    struct Step {
        unsigned level;
        size_t acc_limbs, out_limbs;
    } step[max_rail]{};
    size_t limbs = 0, widest = 0; // capacity of the result; largest intermediate product
};
Chain chain_of(const BaseInfo &, uint64_t fragments) noexcept;
sbn3_query_result chain_bytes(const BaseInfo &, const Chain &, unsigned workers, size_t &bytes) noexcept;
// The result (c.limbs limbs, zero padded) lives in `scratch`, below whatever the caller allocates next.
const uint64_t *chain_evaluate(const Chain &, const RailPlan &, const uint64_t *const *rail, sbn3_team *, unsigned workers,
                               Frame &scratch) noexcept;
} // namespace sbn::v3::radix
