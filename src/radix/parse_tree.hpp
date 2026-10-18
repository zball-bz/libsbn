#pragma once
// Divide-and-conquer evaluation: digits -> binary integer, on the same rail as the format tree.
//
// A node of n fragments (64 digits each) evaluates to an integer below base^(64 n). The split is the
// mirror image of the format tree: the perfect child (the largest power of two below n fragments) is
// the LOW part, so every merge multiplies by a rail power:
//     X = H * odd^(64 * 2^k) * 2^(twos * 64 * 2^k) + L,
// and the power of two is a whole-limb offset. The low child is evaluated in place in the node's
// output slab; its high child occupies the next disjoint slot. Once the
// product has consumed that high child, the merge overwrites it with the
// completed parent. Only the product episode needs separate scratch.
//
// Schedule: the mirror of the format tree (format_tree.hpp). Frontier subtrees, one worker each, run
// first in one dynamically scheduled loop; the top of the tree then merges over power-of-two worker
// groups. Frontier merges in the FFT band use the cached spectrum of the rail power.
#include "radix/programs.hpp"
#include "radix/scale.hpp"
#include "radix/rail_product.hpp"
#include "radix/leaf.hpp"
#include "radix/plan_slots.hpp"
#include "radix/power_bank.hpp"
namespace sbn::v3::radix {
inline constexpr unsigned max_parse_classes = 448;
inline constexpr size_t parse_leaf_limbs=56;
static_assert(group_fragments==8);
// At most eight fragments; also the shared leaf of the larger parse tree.
// out has plan.limbs[ceil(count/64)] words, including its zero padding.
bool parse_leaf(const ParseLeafPlan &,const DigitPlan &,const uint8_t *,size_t count,uint64_t *out) noexcept;
// Same leaf with an exact caller output width; the padded leaf result lives
// in a bounded, aligned frame. Reused by the one-shot path and local recursion.
bool parse_leaf_exact(const ParseLeafPlan &,const DigitPlan &,const uint8_t *,size_t count,uint64_t *out,size_t words) noexcept;
// Small integer recursion: only immutable radix powers and local multiply
// kernels, without a product-program directory or operation-level search.
size_t parse_local_bytes(size_t words) noexcept;
bool parse_local(const ParseLeafPlan &,const DigitPlan &,const uint8_t *,size_t count,
                 uint64_t *out,Frame &) noexcept;
struct ParseClass {
    uint64_t fragments = 0;
    unsigned workers = 1;
    size_t limbs = 0;
    bool group = false, frontier = true, parallel = false;
    bool direct=false,direct_native=false;
    unsigned level = 0;
    int low = -1, high = -1;
    size_t low_limbs = 0, high_limbs = 0, shift_limbs = 0;
    ProductShape product{};
    RailProductPlan cached{}; // enabled: linear product with the cached rail spectrum
    // Parallel child values partition one output slab; a merge consumes
    // the high child before overwriting it. Serial subtrees keep the high
    // child beside their reusable product episode for cache locality.
    size_t slot_limbs=0,high_offset=0,rest_offset=0,region_bytes=0;
    size_t episode_bytes = 0, high_episode_offset = 0;
    uint32_t tasks = 1;
};
struct ParseTask {
    int node;
    uint64_t fragment;
    uint64_t *out;
};
struct ParseTreeView {
    BaseInfo base{};
    unsigned workers = 1, top_workers = 1;
    unsigned class_count = 0;
    bool repeated=false;
    unsigned root_count=0;
    int roots[2]{-1,-1};
    PlanView<ParseClass,max_parse_classes> classes;
    RailPlan rail{};
    ScaleProductPlan extra[2]{}; // products of the owning service, prepared with the tree's programs
    unsigned extra_count = 0;
    ProgramSetPlan programs{};
    size_t cached_bytes = 0;
    size_t group_limbs[group_fragments + 1]{};
    size_t frontier_region_bytes = 0, frontier_limbs = 0;
    sbn3_query_result status = SBN3_SUPPORTED;
    // Product searches of the assembly (programs.hpp); set after parse_tree_begin, used until finish().
    PlanTranscript *transcript = nullptr;
    size_t prepared_bytes() const noexcept { return ((programs.bytes() + 127) & ~size_t(127)) + cached_bytes; }
    size_t task_bytes(int root) const noexcept;
    size_t work_bytes(int root) const noexcept;
    size_t packed_bytes() const noexcept;
    ParseTreeView *pack(void *,size_t) const noexcept;
};
// Query owns bounded capacity; a binding stores only the live directory.
struct ParseTreePlan : ParseTreeView {
    PlanSlots<ParseClass,max_parse_classes> class_storage;
    ParseTreePlan() noexcept { classes.reset(class_storage.storage()); }
    int add_tree(uint64_t fragments) noexcept;
    int add_product(size_t an,size_t bn) noexcept;
    sbn3_query_result finish() noexcept;
private:
    int classify(uint64_t fragments, unsigned workers) noexcept;
};
sbn3_query_result parse_tree_begin(unsigned base, unsigned workers, uint64_t largest_fragments, ParseTreePlan &) noexcept;
struct ParseTree {
    const ParseTreeView *plan = nullptr;
    DigitPlan digits{};
    ParseLeafPlan leaf{};
    Arena *arena = nullptr;
    sbn3_team *team = nullptr;
    sbn3_lease work{};
    ProductProgram *programs = nullptr;
    RailProduct *cached = nullptr;
    const uint64_t *rail[max_rail]{};
    ScaleProduct extra[2]{};
    Frame *roots[32]{};
};
void parse_tree_bind(ParseTree &, const ParseTreeView &, const uint8_t *alphabet, Arena &, sbn3_team &,
                     const sbn3_lease &prepared, sbn3_lease &work, uint64_t *rail_storage,
                     ProductProgram *programs, RailProduct *cached) noexcept;
// The output slab has classes[root].slot_limbs words; the completed value
// occupies its first classes[root].limbs words. False on an invalid digit.
bool parse_tree_run(ParseTree &, int root, const uint8_t *digits, uint64_t count, uint64_t *out,size_t capacity) noexcept;
} // namespace sbn::v3::radix
