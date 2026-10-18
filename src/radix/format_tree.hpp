#pragma once
// Division-free scaled remainder tree: binary fraction -> digits
// (docs/radix-conversion-design-2026-09-18.md section 2).
//
// A node of n fragments holds `limbs` limbs of a fraction in [0,1), binary
// point above the top limb, approximating its digits from below. The left
// child is the top limbs of the
// node; the right child is a limb window of the exact product with the rail
// power odd^(64 * left_fragments). Parallel/reused calls use a power-of-two
// grid; serial fresh calls scale a dyadic ladder to the request. Nodes
// of at most sixteen fragments use bounded products and up to two eight-lane
// leaf calls. Every fragment also produces one overlap word; the
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
#include "radix/scale.hpp"
#include "radix/rail_product.hpp"
#include "radix/ring_product.hpp"
#include "radix/plan_slots.hpp"
#include "radix/balanced_powers.hpp"
#include <algorithm>
namespace sbn::v3::radix {
inline constexpr unsigned max_classes = 192, max_stages = 96, max_trees = 2;
inline constexpr uint64_t format_group_fragments=16;
inline constexpr size_t group_product_limbs = 192, group_right_limbs = 64;
// How the right child window of one node size is produced.
struct SplitPlan {
    ProductShape product{};   // exact product program, or dimensions and direct-repair scratch for middle_words
    RailProductPlan cyclic{}; // wrap-around product with the cached rail spectrum, FFT kernels (single worker), or
    RingPlan ring{};          // wrap-around product through the product service (NTT band, staged nodes only)
    int cache_slot=-1;        // shared level cache, or this split's own class
    size_t gap_limbs = 0;     // wrap-around: limbs between the wrapped part and the window (at least one)
    size_t low_limbs = 0, low_rail_limbs = 0, low_bytes = 0; // wrap-around tie-break: exact low product and its scratch
    unsigned low_workers = 1;
    ProductChoice low_choice=0;
    size_t middle_words = 0, middle_bytes = 0;
    unsigned middle_guards=4;
    bool direct=false,direct_native=false;
    uint64_t middle_shift = 0;
    bool wraps() const noexcept { return cyclic.enabled || ring.enabled; }
    size_t episode_bytes() const noexcept {
        if (cyclic.enabled)
            return ((cyclic.output_limbs() * 8 + 63) & ~size_t(63)) + 64 + std::max(cyclic.scratch_bytes, low_bytes);
        if (ring.enabled)
            return ((ring.output_limbs * 8 + 63) & ~size_t(63)) + 64;
        return middle_words?((middle_words*8+63)&~size_t(63))+std::max(middle_bytes,low_bytes):product.episode_bytes();
    }
};
struct NodeClass {
    uint64_t fragments = 0;
    size_t limbs = 0;
    bool group = false;   // bounded local splitting and leaf calls
    bool frontier = true; // run as a whole by one worker
    unsigned level = 0;   // the split multiplies by rail[level]
    uint64_t left_fragments=0;
    size_t slot_limbs=0; // sum of leaf-group capacities, including their guards
    uint64_t seams=0;
    size_t block_batch_offset=0;
    int left = -1, right = -1;
    size_t right_limbs = 0, window_limbs = 0;
    // The top twos * left_fragments limbs only reach above the window
    // (base = 2^twos * odd); the product omits those input limbs.
    size_t split_limbs = 0;
    // Frontier class: one product episode reused by both children. Fraction
    // values occupy disjoint low/right and high/left views of the root slab.
    SplitPlan split{};
    size_t region_bytes = 0;
    // Top class: right child fractions of the whole top subtree persist until the frontier ran
    // ([own right child][left subtree][right subtree]).
    size_t persist_bytes = 0;
    uint32_t tasks = 1, top_nodes = 0; // frontier tasks / top nodes in the subtree (a frontier class is one task)
};
struct Stage {
    int node_class = -1;
    int partner_class=-1;
    uint32_t main_count=0;
    uint32_t count = 0, groups = 1, first = 0; // nodes of this size; concurrent worker groups; first instance
    unsigned workers = 1;                       // workers of one group
    int ring_stage = -1;                        // index of the stage's replayed ring plans
    bool borrow_output=false;                   // numeric result dies before digit emission
    bool borrow_result=false;                   // ordinary result only; its workspace stays in the arena
    SplitPlan split{};
    size_t episode_bytes()const noexcept {
        return split.episode_bytes()-(borrow_result?((split.product.output_limbs*8+63)&~size_t(63)):0);
    }
};
struct TreeShape {
    int root = -1;
    uint32_t stage_count = 0, top_nodes = 0, tasks = 0;
    PlanView<Stage,max_stages> stages;
    size_t episode_bytes = 0; // largest stage: groups * split episode
    size_t pool_bytes = 0;    // largest ring stage: spectrum and one bound consumer per group
};
struct FormatInstance {
    const uint64_t *y;
    uint64_t *right;
    int node_class;
    unsigned repair=0; // 1: unresolved; 2/3: exact witness says no carry/carry
};
struct FormatTask {
    int node;
    const uint64_t *y;
    uint64_t *storage;
    uint64_t fragment;
    uint64_t seam;
};
struct alignas(64) FragmentBatch {
    uint64_t value[8][max_fragment_limbs];
    uint64_t position[8];
    unsigned count=0;
    FragmentBatch() noexcept {}
};
struct alignas(64) BlockBatch {
    uint64_t position[8],seam[8];
    unsigned count=0;
    BlockBatch() noexcept {}
};
struct FormatTreeView {
    BaseInfo base{};
    unsigned workers = 1, top_workers = 1; // top_workers: the largest power of two <= workers
    unsigned class_count = 0, tree_count = 0;
    // Retained power-of-two-grid recipe with apply-oriented product prices.
    // A fixed-shape reused binding may instead retain a coupled balanced
    // recipe; this field is not permission to execute a binding repeatedly.
    bool repeated = false;
    bool balanced = false; // separate geometry choice; no implication about binding reuse
    bool block_leaves=false;
    unsigned block_max_limbs=32;
    size_t block_work_bytes=0;
    size_t block_batch_bytes=0;
    PlanView<NodeClass,max_classes> classes;
    TreeShape trees[max_trees]{};
    RailPlan rail{};
    BalancedPowers powers;
    ScaleProductPlan extra[2]{}; // products of the owning service, prepared with the tree's programs
    unsigned extra_count = 0;
    size_t group_limbs[format_group_fragments + 1]{};
    size_t group_work[format_group_fragments + 1]{};
    unsigned group_u52 = 0; // bit n: the split of n fragments uses native U52
    unsigned fragment_u52 = 0;
    ProgramSetPlan programs{}; // immutable preparation of every exact product program
    size_t cyclic_bytes = 0;   // tables and cached spectra of the wrap-around classes, after the programs
    size_t frontier_region_bytes = 0;
    size_t frontier_limbs = 0; // nodes below this size are frontier tasks (set by format_tree_begin)
    sbn3_query_result status = SBN3_SUPPORTED;
    // Product searches of the assembly (programs.hpp); set after format_tree_begin, used until finish().
    PlanTranscript *transcript = nullptr;
    Chain chain(int tree)const noexcept;
    size_t prepared_bytes() const noexcept { return ((programs.bytes() + 127) & ~size_t(127)) + cyclic_bytes; }
    size_t prepared_alignment() const noexcept { return programs.alignment; }
    size_t root_limbs(int tree) const noexcept { return classes[trees[tree].root].limbs; }
    uint64_t root_fragments(int tree) const noexcept { return classes[trees[tree].root].fragments; }
    uint64_t seam_count(int tree)const noexcept {return classes[trees[tree].root].seams;}
    size_t seam_bytes(int tree)const noexcept {return block_leaves?(8*seam_count(tree)+63)&~size_t(63):0;}
    // Child capacities partition the parent even for balanced non-power-of-two
    // splits. Their guards are counted at the actual leaf-group boundaries.
    size_t root_storage_words(int tree) const noexcept {
        const auto &c=classes[trees[tree].root];
        return c.slot_limbs;
    }
    size_t batch_bytes(int tree)const noexcept {
        if(classes[trees[tree].root].group)return 0;
        return block_leaves?size_t(workers)*block_batch_bytes:repeated||!balanced?0:size_t(workers)*sizeof(FragmentBatch);
    }
    unsigned stage_program(unsigned tree,unsigned stage)const noexcept {
        unsigned at=class_count;for(unsigned j=0;j<tree;++j)at+=trees[j].stage_count;return at+stage;
    }
    unsigned program_slots() const noexcept {return stage_program(tree_count,0);}
    unsigned ring_stages = 0; // replayed ring plans the binder needs
    size_t pool_bytes() const noexcept { return std::max(trees[0].pool_bytes, trees[1].pool_bytes); }
    size_t pool_alignment()const noexcept {
        size_t alignment=4096;
        for(unsigned t=0;t<tree_count;++t)for(unsigned j=0;j<trees[t].stage_count;++j){const auto &r=trees[t].stages[j].split.ring;
            if(r.enabled)alignment=std::max({alignment,r.work_alignment,r.spectrum_alignment});}
        return alignment;
    }
    // Work lease: [instances][tasks][one fraction slab][max(stage episodes, frontier regions)].
    size_t work_bytes(int tree) const noexcept;
    size_t work_prefix_bytes(int tree) const noexcept;
    // Stage-local end of the retained fraction/episode region, followed by
    // that stage's unleased product pool. Absolute arena offsets preserve
    // the product backend's alignment even when the work prefix is smaller.
    size_t stage_work_bytes(int tree,unsigned stage)const noexcept;
    size_t stage_pool_offset(int tree,unsigned stage,size_t work_offset)const noexcept;
    size_t runtime_bytes(int tree,size_t work_offset)const noexcept;
    size_t packed_bytes()const noexcept;
    FormatTreeView *pack(void *storage,size_t bytes)const noexcept;
};
// Query/assembly owns maximum bounded capacity on the control stack. A bound
// operation uses FormatTreeView::pack, not this capacity-sized object.
struct FormatTreePlan : FormatTreeView {
    PlanSlots<NodeClass,max_classes> class_storage;
    PlanSlots<Stage,max_stages> stage_storage[max_trees];
    PlanSlots<BalancedPowers::Entry,max_balanced_powers> power_storage;
    FormatTreePlan() noexcept {
        classes.reset(class_storage.storage());powers.entries.reset(power_storage.storage());
        for(unsigned i=0;i<max_trees;++i)trees[i].stages.reset(stage_storage[i].storage());
    }
    void select_reuse_recipe(bool reuse) noexcept {
        // Coupled FFT eligibility is established by format_tree_begin.
        // The binding fixes the shape, so reuse does not require a grid.
        // Reciprocal caching is independent. Parallel fixed-shape bindings
        // can retain the same balanced stages and their bounded work pools.
        repeated=reuse&&!balanced;
        if(repeated)balanced=false;
    }
    int add_tree(uint64_t fragments) noexcept;
    int add_product(size_t an,size_t bn,bool reuse) noexcept;
    int add_product(size_t an,size_t bn) noexcept {return add_product(an,bn,repeated);}
    sbn3_query_result finish() noexcept;
private:
    int classify(uint64_t fragments,uint64_t seed) noexcept;
    sbn3_query_result split_plan(const NodeClass &, unsigned workers, uint64_t count, SplitPlan &) const noexcept;
};
// `largest_fragments`: the largest tree that will be added (it sets the frontier granularity).
sbn3_query_result format_tree_begin(unsigned base, unsigned workers, uint64_t largest_fragments, FormatTreePlan &) noexcept;
// Bound engine. All storage is caller-provided and outlives the object.
struct FormatTree {
    const FormatTreeView *plan = nullptr;
    DigitPlan digits{};
    const FractionBlockPlan *block_plan=nullptr;
    Arena *arena = nullptr;
    sbn3_team *team = nullptr;
    sbn3_lease work{};
    ProductProgram *programs = nullptr; // plan.program_slots() entries
    RailProduct *cyclic = nullptr;      // one per class (used by wrap-around splits)
    RingStage *rings = nullptr;         // plan.ring_stages entries
    sbn3_arena *ring_arena = nullptr;   // pool of the ring stages: an unleased prepared range of plan.pool_bytes()
    size_t pool_offset = 0;
    bool shared_pool=false;
    const uint64_t *rail[max_balanced_powers]{};
    ScaleProduct extra[2]{};
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
void format_tree_bind(FormatTree &, const FormatTreeView &, const uint8_t *alphabet, sbn3_arena &, sbn3_team &,
                      const sbn3_lease &prepared, sbn3_lease &work, uint64_t *rail_storage,
                      ProductProgram *programs, RailProduct *cyclic, RingStage *rings, size_t pool_offset,
                      bool shared_pool=false) noexcept;
// Converts the fraction `y` (root_limbs(tree) limbs) to 64 * fragments digit bytes at `out`, reconciles the
// seams, and returns the overlap word after the last digit. `first`/`overlap` are scratch arrays of
// plan.seam_count(tree) words. The work lease holds at least plan.work_bytes(tree).
uint64_t format_tree_run(FormatTree &, int tree, const uint64_t *y, uint8_t *out, uint64_t *first,
                         uint64_t *overlap) noexcept;
// Adds one to the digit string out[0..count) (carry out dropped). Returns the index of the
// most significant byte changed.
size_t increment_digits(uint8_t *out, size_t count, const DigitPlan &) noexcept;
} // namespace sbn::v3::radix
