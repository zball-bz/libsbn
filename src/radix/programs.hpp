#pragma once
// Exact product programs and the power rail shared by the radix conversion trees.
#include "radix/geometry.hpp"
#include "algorithms/inverse_tuning.hpp"
#include "algorithms/local_inverse.hpp"
#include "product/program.hpp"
#include "product/table_pool.hpp"
#include <cstring>
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
    size_t wide_product_limbs = 0; // full available team; nonzero is a legacy probe override
    // Integer parts use maximal-word preinverse division below the SRT
    // crossover. Fresh work = input limbs times maximal radix-word chunks;
    // the complete-call threshold includes fused root normalization.
    // A retained tree has a lower, execution-oriented crossover.
    size_t integer_tree_work = 2600, integer_tree_limbs_repeated = 32;
    size_t integer_scaled_work = 160; // maximal-word work before direct fractional normalization wins
    unsigned fraction_recipe = 0; // 0: work policy; probes: 1 scalar, 2 tree, 3 IFMA
    // Serial fresh calls use a scaled ladder in the coupled FFT regime;
    // local-prefix and parallel trees retain the regular grid. Probes may
    // pin 1: grid or 2: scaled in either regime.
    unsigned format_split_policy = 0;
    unsigned format_seed_fragments = 0; // 0: serial 16 / parallel 31; otherwise a probe override
    bool reuse_fft_squares = true;       // consume an existing eligible FFT cache while constructing powers
    size_t cyclic_min_fresh = 400;
    bool block_leaves = true;
    unsigned block_tree_limbs=0;        // 0: widen only with two vector batches per worker
    bool short_tail_prefix = true;
    // Bounded leaf products: pay for U52 conversion only above this many
    // scalar limb products (Zen5 native, r12-leaf-split-volume evidence).
    size_t group_u52_work = 128;
    bool cyclic_products = true;        // frontier splits as wrap-around products with cached rail spectra
    bool ring_products = true;          // staged splits in the NTT band as wrap-around products (product service)
    unsigned ring_prime_batch=0;        // 0: backend policy; explicit probe override for Bailey stages
    bool prefer_short_ring=false;       // experimental: prefer a period shorter than the linear product
    bool staged_serial_fft=true;        // keep a cache-sized window kernel serial; parallelize independent nodes
    size_t ring_node_limbs = 40000;     // from this node size on
    unsigned ring_min_count = 8;        // and only for stages with this many nodes (the spectrum is built per run)
};
// Process-wide policy; probes may change it before planning (not during).
TreePolicy &tree_policy() noexcept;
inline constexpr size_t rail_basecase_limbs = 12;
// Transcript of the product searches of one plan, in planning order. The tree plans are far too large for the
// services' plan values, so bind assembles the plan again; the public query records the winning candidate of
// every search here (the transcript travels inside the plan value) and the assembly at bind repeats each
// winner's single backend query instead of the search. An entry carries a check of its request and of the
// identity of the plan it stands for: an entry that does not reproduce it, an entry without a choice and a
// transcript that ran out of room all fall back to the search, and the owning service still compares the
// identity of the whole replayed plan with the sealed one.
struct PlanTranscript {
    static constexpr unsigned capacity = 176;
    uint64_t entry[capacity];
    uint32_t count = 0, cursor = 0;
    uint32_t searched = 0; // replay: products that had to be searched after all
    bool replay = false;   // false: the query records; true: bind consumes
    // Only the recorded prefix exists. Small direct queries have no entries
    // and should not clear the capacity-sized transcript buffer.
    PlanTranscript() noexcept {}
    PlanTranscript(const PlanTranscript &r) noexcept { *this=r; }
    PlanTranscript &operator=(const PlanTranscript &r) noexcept {
        if(this!=&r){count=r.count;cursor=r.cursor;searched=r.searched;replay=r.replay;
            if(count)std::memcpy(entry,r.entry,size_t(count)*sizeof(uint64_t));}
        return *this;
    }
    static uint64_t check(uint64_t a, uint64_t b, uint64_t c, uint64_t identity) noexcept;
    void record(ProductChoice choice, uint64_t check) noexcept {
        if (count < capacity)
            entry[count] = choice | check << product_choice_bits;
        count += count < capacity; // a full transcript keeps its prefix; the rest is searched again
    }
    // The next recorded entry, 0 when there is none.
    uint64_t next() noexcept { return cursor < count ? entry[cursor++] : 0; }
};
// Plan-time facts of one exact product (an x bn limbs on `workers` workers).
struct ProductShape {
    size_t an = 0, bn = 0;
    unsigned workers = 1, requested = 1; // resolved execution width; width asked of the query
    ProductChoice choice = 0;            // winner of the search: preparation repeats its one backend query
    size_t prepared_bytes = 0;            // complete private preparation
    size_t local_bytes = 0;    // preparation beside the shareable tables
    SharedPreparation tables{};
    size_t work_bytes = 0, work_alignment = 64;
    size_t output_limbs = 0;
    uint64_t arithmetic_id = 0;
    double prepare_ns = 0, apply_ns = 0; // cost of this recipe, not of its caller's search
    // Bytes of [output][padding][workspace] when the output starts 64-byte aligned.
    size_t episode_bytes() const noexcept {
        return ((output_limbs * 8 + 63) & ~size_t(63)) + (work_alignment > 64 ? work_alignment - 64 : 0) + work_bytes;
    }
    // The same plus a program prepared next to it (one-time products).
    size_t temporary_bytes() const noexcept { return ((prepared_bytes + 127) & ~size_t(127)) + 512 + episode_bytes(); }
    // A one-time product whose output has already been allocated by the
    // caller (power-chain ping-pong or exact-boundary resolution).
    size_t temporary_work_bytes() const noexcept {
        return ((prepared_bytes + 127) & ~size_t(127)) + 512 +
               (work_alignment > 64 ? work_alignment - 64 : 0) + work_bytes;
    }
};
sbn3_query_result product_shape(size_t an, size_t bn, unsigned workers, ProductShape &, ProductProgramPlan *keep = nullptr,
                                PlanTranscript * = nullptr, uint64_t applications = 0);
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
// One exact product on the whole team with a program built inside `scratch` and discarded. `choice`: the
// planned winner of this product's search (0: search here).
void temporary_product(sbn3_team *, unsigned workers, Frame &scratch, const uint64_t *a, size_t an, const uint64_t *b,
                       size_t bn, uint64_t *out, size_t out_capacity, ProductChoice choice = 0) noexcept;
// Bounded local products used by setup chains and exceptional digit witnesses.
// Their scratch query and execution share the multiplication layer's compact
// word/FFT recipe; no complete service plan is constructed just to price them.
bool local_temporary_supported(size_t an,size_t bn,unsigned workers) noexcept;
size_t local_temporary_bytes(size_t an,size_t bn) noexcept;
void local_temporary_product(Frame &,const uint64_t *,size_t,const uint64_t *,size_t,uint64_t *) noexcept;
// ---- rail: odd^(64 * 2^k), k < count, each zero padded to its capacity -------------------------------
struct RailPlan {
    unsigned count = 0;
    unsigned fixed_levels = 0;
    size_t limbs[max_rail]{}, offset[max_rail]{};
    size_t total_limbs = 0;
    // rail[k]^2: through the product service's SQR recipe (service) or as a temporary program; its search winner.
    bool service[max_rail]{};
    ProductChoice square_choice[max_rail]{};
    size_t setup_bytes = 0; // scratch of the largest squaring
};
sbn3_query_result rail_finish(const BaseInfo &, unsigned workers, RailPlan &, PlanTranscript * = nullptr) noexcept;
// The scratch lease is released around squarings that run through the product service (its leases are
// exclusive) and acquired again: the caller's lease is updated in place.
void rail_build(const BaseInfo &, const RailPlan &, unsigned workers, Arena &, sbn3_team &, sbn3_lease &scratch,
                uint64_t *storage, const uint64_t **entries) noexcept;
// ---- reciprocal of a scaling power ----------------------------------------------------------------------
// U = floor((B^(2n) - 1) / D) for a normalized n-limb D (top bit set), n + 1 limbs with U[n] = 1: inside the
// Newton INVERSE contract (|U - B^(2n)/D| < 3). Local u52 recurrence with exact correction, cheaper than
// the complete Newton route (query, bind, execute) below the shared crossover
// in inverse_tuning.hpp. Such plans carry no Newton plan/storage.
inline constexpr size_t reciprocal_basecase_limbs = inverse_tuning::basecase_limbs;
inline bool reciprocal_is_basecase(size_t n) noexcept { return n && n <= reciprocal_basecase_limbs; }
inline size_t reciprocal_basecase_bytes(size_t n) noexcept { return local_inverse_bytes(n); }
void reciprocal_basecase(const uint64_t *d, size_t n, uint64_t *out, Frame &scratch) noexcept;
// ---- odd^(64 F) for any F, from the rail (rail.count > top bit of F) ----------------------------------
inline constexpr size_t chain_basecase_limbs = 24;
struct Chain {
    unsigned first = 0, steps = 0;
    size_t first_limbs=0;
    struct Step {
        unsigned level;
        size_t acc_limbs, out_limbs, power_limbs;
        ProductChoice choice; // set by chain_bytes
        int cache_slot=-1;
        size_t cache_bytes=0;
    } step[max_rail]{};
    size_t limbs = 0, widest = 0; // capacity of the result; largest intermediate product
    size_t resident_bytes() const noexcept {
        return steps ? 2 * (((widest + 8) * 8 + 63) & ~size_t(63)) : 0;
    }
};
Chain chain_of(const BaseInfo &, uint64_t fragments) noexcept;
// Scratch bytes of chain_evaluate; records the products' search winners in the chain's steps.
sbn3_query_result chain_bytes(const BaseInfo &, Chain &, unsigned workers, size_t &bytes, PlanTranscript * = nullptr) noexcept;
// The result (c.limbs limbs, zero padded) lives in `scratch`, below whatever the caller allocates next.
const uint64_t *chain_evaluate(const Chain &, const RailPlan &, const uint64_t *const *rail, sbn3_team *, unsigned workers,
                               Frame &scratch,const struct RailProduct *caches=nullptr) noexcept;
} // namespace sbn::v3::radix
