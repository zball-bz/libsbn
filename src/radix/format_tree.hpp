#pragma once
// Division-free scaled remainder tree: binary fraction -> digits
// (docs/radix-conversion-design-2026-09-18.md section 2).
//
// A node of n fragments holds `limbs` limbs of a fraction in [0,1), binary
// point above the top limb, approximating its digits from below. The left
// child (the largest power of two below n fragments) is the top limbs of the
// node; the right child is a limb window of the exact product with the rail
// power odd^(64 * 2^k). Nodes of at most eight fragments are converted by one
// eight-lane leaf call. Every fragment also produces one overlap word; the
// flat right-to-left seam pass reconciles neighbours.
//
// Schedule. Large nodes ("top") are processed in stages, one per node size in
// descending order: all nodes of one size are split together by a power of
// two of equal worker groups (parents are always larger than children, so the
// order respects the tree; the perfect subtrees hanging off the ragged right
// spine join the main subtree's nodes of the same size). Group widths that
// are not powers of two are several times slower in the NTT products, and
// sixteen concurrent single-worker products of 50K-250K limbs fall out of the
// shared cache, so groups keep at least four workers until nodes are small.
// Every subtree below the frontier size is one task; the tasks run last, in
// one dynamically scheduled loop over all workers, each serially inside its
// worker's region.
#include "radix/programs.hpp"
#include "radix/rail_product.hpp"
#include "radix/ring_product.hpp"
namespace sbn::v3::radix {
inline constexpr unsigned max_classes = 192, max_stages = 96, max_trees = 2;
inline constexpr size_t group_product_limbs = 96, group_right_limbs = 32;
// How the right child window of one node size is produced.
struct SplitPlan {
    ProductShape product{};   // exact product program, or
    RailProductPlan cyclic{}; // wrap-around product with the cached rail spectrum, FFT kernels (single worker), or
    RingPlan ring{};          // wrap-around product through the product service (NTT band, staged nodes only)
    size_t gap_limbs = 0;     // wrap-around: limbs between the wrapped part and the window (at least one)
    size_t low_limbs = 0, low_rail_limbs = 0, low_bytes = 0; // wrap-around tie-break: exact low product and its scratch
    unsigned low_workers = 1;
    size_t middle_words = 0, middle_bytes = 0;
    uint64_t middle_shift = 0;
    bool wraps() const noexcept { return cyclic.enabled || ring.enabled; }
    size_t episode_bytes() const noexcept {
        if (cyclic.enabled)
            return ((cyclic.ring * 8 + 63) & ~size_t(63)) + 64 + std::max(cyclic.scratch_bytes, low_bytes);
        if (ring.enabled)
            return ((ring.output_limbs * 8 + 63) & ~size_t(63)) + 64 + low_bytes;
        const size_t middle=((middle_words*8+63)&~size_t(63))+middle_bytes;
        return middle_words?std::max(product.temporary_bytes(),middle):product.episode_bytes();
    }
};
struct NodeClass {
    uint64_t fragments = 0;
    size_t limbs = 0;
    bool group = false;   // converted by one leaf call, no storage
    bool frontier = true; // run as a whole by one worker
    unsigned level = 0;   // the split multiplies by rail[level]
    int left = -1, right = -1;
    size_t right_limbs = 0, window_limbs = 0;
    // The top twos * 2^level limbs of the node only reach limbs above the window (base = 2^twos * odd): the
    // product uses the low split_limbs = limbs - twos * 2^level limbs, and its window ends at its operand's top.
    size_t split_limbs = 0;
    // Frontier class: single-worker split; private storage below the node, [right child fraction][rest]; rest
    // holds first the product episode, then the regions of the children (one after the other).
    SplitPlan split{};
    size_t rest_offset = 0, region_bytes = 0;
    // Top class: right child fractions of the whole top subtree persist until the frontier ran
    // ([own right child][left subtree][right subtree]).
    size_t persist_bytes = 0;
    uint32_t tasks = 1, top_nodes = 0; // frontier tasks / top nodes in the subtree (a frontier class is one task)
};
struct Stage {
    int node_class = -1;
    uint32_t count = 0, groups = 1, first = 0; // nodes of this size; concurrent worker groups; first instance
    unsigned workers = 1;                       // workers of one group
    int ring_stage = -1;                        // index of the stage's replayed ring plans
    SplitPlan split{};
};
struct TreeShape {
    int root = -1;
    uint32_t stage_count = 0, top_nodes = 0, tasks = 0;
    Stage stages[max_stages]{};
    size_t episode_bytes = 0; // largest stage: groups * split episode
    size_t pool_bytes = 0;    // largest ring stage: spectrum and one bound consumer per group
};
struct FormatInstance {
    const uint64_t *y;
    uint64_t *right;
};
struct FormatTask {
    int node;
    const uint64_t *y;
    uint64_t fragment;
};
struct FormatTreePlan {
    BaseInfo base{};
    unsigned workers = 1, top_workers = 1; // top_workers: the largest power of two <= workers
    unsigned class_count = 0, tree_count = 0;
    bool repeated = false; // repeated bindings price execution; one-shot plans price the whole tree
    NodeClass classes[max_classes]{};
    TreeShape trees[max_trees]{};
    RailPlan rail{};
    ProductShape extra[2]{}; // products of the owning service, prepared with the tree's programs
    unsigned extra_count = 0;
    size_t group_limbs[group_fragments + 1]{};
    unsigned fragment_u52 = 0;
    ProgramSetPlan programs{}; // immutable preparation of every exact product program
    size_t cyclic_bytes = 0;   // tables and cached spectra of the wrap-around classes, after the programs
    size_t frontier_region_bytes = 0;
    size_t frontier_limbs = 0; // nodes below this size are frontier tasks (set by format_tree_begin)
    sbn3_query_result status = SBN3_SUPPORTED;
    // Product searches of the assembly (programs.hpp); set after format_tree_begin, used until finish().
    PlanTranscript *transcript = nullptr;
    // Adds the tree of `fragments` fragments; returns its index in trees[] or -1.
    int add_tree(uint64_t fragments) noexcept;
    // A team product of the owning service (before finish()); returns its index or -1.
    int add_product(size_t an, size_t bn) noexcept;
    // Builds the stages, the rail and the preparation accounting after every tree was added.
    sbn3_query_result finish() noexcept;
    size_t prepared_bytes() const noexcept { return ((programs.bytes() + 127) & ~size_t(127)) + cyclic_bytes; }
    size_t prepared_alignment() const noexcept { return programs.alignment; }
    size_t root_limbs(int tree) const noexcept { return classes[trees[tree].root].limbs; }
    uint64_t root_fragments(int tree) const noexcept { return classes[trees[tree].root].fragments; }
    // An eight-fragment slot carries its own guard. Every split above the
    // leaf group has a power-of-two left child divisible by eight, so the
    // two child slot capacities partition their parent's capacity exactly.
    size_t fraction_storage_words(uint64_t fragments) const noexcept {
        return size_t((fragments+group_fragments-1)/group_fragments)*group_limbs[group_fragments];
    }
    size_t root_storage_words(int tree) const noexcept {
        const auto &c=classes[trees[tree].root];
        return c.frontier?c.limbs:fraction_storage_words(c.fragments);
    }
    // Program slots the binder needs: one per class, then max_stages per tree.
    unsigned program_slots() const noexcept { return class_count + tree_count * max_stages; }
    unsigned ring_stages = 0; // replayed ring plans the binder needs
    size_t pool_bytes() const noexcept { return std::max(trees[0].pool_bytes, trees[1].pool_bytes); }
    // Work lease: [instances][tasks][one fraction slab][max(stage episodes, frontier regions)].
    size_t work_bytes(int tree) const noexcept;
private:
    int classify(uint64_t fragments) noexcept;
    sbn3_query_result split_plan(const NodeClass &, unsigned workers, uint64_t count, SplitPlan &) const noexcept;
};
// `largest_fragments`: the largest tree that will be added (it sets the frontier granularity).
sbn3_query_result format_tree_begin(unsigned base, unsigned workers, uint64_t largest_fragments, FormatTreePlan &) noexcept;
// Bound engine. All storage is caller-provided and outlives the object.
struct FormatTree {
    const FormatTreePlan *plan = nullptr;
    DigitPlan digits{};
    Arena *arena = nullptr;
    sbn3_team *team = nullptr;
    sbn3_lease work{};
    ProductProgram *programs = nullptr; // plan.program_slots() entries
    RailProduct *cyclic = nullptr;      // one per class (used by wrap-around splits)
    RingStage *rings = nullptr;         // plan.ring_stages entries
    sbn3_arena *ring_arena = nullptr;   // pool of the ring stages: an unleased prepared range of plan.pool_bytes()
    size_t pool_offset = 0;
    const uint64_t *rail[max_rail]{};
    ProductProgram extra[2]{};
    Frame *roots[32]{};
};
// The caller may form an initial root here, avoiding a separate padded copy.
// It is overwritten by a subsequent run or use of the tree's work lease.
uint64_t *format_tree_root_buffer(FormatTree &,int tree) noexcept;
// Builds the rail in `rail_storage` (plan.rail.total_limbs limbs) and prepares every program in `prepared`
// (plan.prepared_bytes()); `work` is scratch for the squarings (at least plan.rail.setup_bytes) and later
// the run storage. `programs` has plan.program_slots() entries, `cyclic` class_count entries, `rings`
// plan.ring_stages entries; pool_offset is the arena offset of plan.pool_bytes() unleased prepared bytes
// (2 MiB aligned) when the plan has ring stages.
void format_tree_bind(FormatTree &, const FormatTreePlan &, const uint8_t *alphabet, sbn3_arena &, sbn3_team &,
                      const sbn3_lease &prepared, sbn3_lease &work, uint64_t *rail_storage,
                      ProductProgram *programs, RailProduct *cyclic, RingStage *rings, size_t pool_offset) noexcept;
// Converts the fraction `y` (root_limbs(tree) limbs) to 64 * fragments digit bytes at `out`, reconciles the
// seams, and returns the overlap word after the last digit. `first`/`overlap` are scratch arrays of
// `fragments` words. The work lease holds at least plan.work_bytes(tree).
uint64_t format_tree_run(FormatTree &, int tree, const uint64_t *y, uint8_t *out, uint64_t *first,
                         uint64_t *overlap) noexcept;
// Adds one to the digit string out[0..count) (carry out dropped). Returns the index of the
// most significant byte changed.
size_t increment_digits(uint8_t *out, size_t count, const DigitPlan &) noexcept;
} // namespace sbn::v3::radix
