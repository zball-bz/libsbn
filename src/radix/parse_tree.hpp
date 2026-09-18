#pragma once
// Divide-and-conquer evaluation: digits -> binary integer, on the same rail as the format tree.
//
// A node of n fragments (64 digits each) evaluates to an integer below base^(64 n). The split is the
// mirror image of the format tree: the perfect child (the largest power of two below n fragments) is
// the LOW part, so every merge multiplies by a rail power:
//     X = H * odd^(64 * 2^k) * 2^(twos * 64 * 2^k) + L,
// and the power of two is a whole-limb offset. The low child is evaluated in place in the node's
// output; the high child and the product live beside it.
//
// Schedule: the mirror of the format tree (format_tree.hpp). Frontier subtrees, one worker each, run
// first in one dynamically scheduled loop; the top of the tree then merges over power-of-two worker
// groups. Frontier merges in the FFT band use the cached spectrum of the rail power.
#include "radix/programs.hpp"
#include "radix/rail_product.hpp"
#include "radix/leaf.hpp"
namespace sbn::v3::radix {
inline constexpr unsigned max_parse_classes = 448;
struct ParseClass {
    uint64_t fragments = 0;
    unsigned workers = 1;
    size_t limbs = 0;
    bool group = false, frontier = true, parallel = false;
    unsigned level = 0;
    int low = -1, high = -1;
    size_t low_limbs = 0, high_limbs = 0, shift_limbs = 0;
    ProductShape product{};
    RailProductPlan cached{}; // enabled: linear product with the cached rail spectrum
    // Frontier class: [high child value][rest]; rest holds the children's regions, then the product episode.
    size_t rest_offset = 0, region_bytes = 0;
    // Top class: the high child values of the whole top subtree persist ([own][low subtree][high subtree]).
    size_t persist_bytes = 0, episode_bytes = 0, high_episode_offset = 0;
    uint32_t tasks = 1;
};
struct ParseTask {
    int node;
    uint64_t fragment;
    uint64_t *out;
};
struct ParseTreePlan {
    BaseInfo base{};
    unsigned workers = 1, top_workers = 1;
    unsigned class_count = 0;
    ParseClass classes[max_parse_classes]{};
    RailPlan rail{};
    ProductShape extra[2]{}; // products of the owning service, prepared with the tree's programs
    unsigned extra_count = 0;
    ProgramSetPlan programs{};
    size_t cached_bytes = 0;
    size_t group_limbs[group_fragments + 1]{};
    size_t frontier_region_bytes = 0, frontier_limbs = 0;
    sbn3_query_result status = SBN3_SUPPORTED;
    int add_tree(uint64_t fragments) noexcept;
    // A team product of the owning service (before finish()); returns its index or -1.
    int add_product(size_t an, size_t bn) noexcept;
    sbn3_query_result finish() noexcept;
    size_t prepared_bytes() const noexcept { return ((programs.bytes() + 127) & ~size_t(127)) + cached_bytes; }
    size_t task_bytes(int root) const noexcept;
    size_t work_bytes(int root) const noexcept;
private:
    int classify(uint64_t fragments, unsigned workers) noexcept;
};
sbn3_query_result parse_tree_begin(unsigned base, unsigned workers, uint64_t largest_fragments, ParseTreePlan &) noexcept;
struct ParseTree {
    const ParseTreePlan *plan = nullptr;
    DigitPlan digits{};
    Arena *arena = nullptr;
    sbn3_team *team = nullptr;
    sbn3_lease work{};
    ProductProgram *programs = nullptr;
    RailProduct *cached = nullptr;
    const uint64_t *rail[max_rail]{};
    ProductProgram extra[2]{};
    Frame *roots[32]{};
};
void parse_tree_bind(ParseTree &, const ParseTreePlan &, const uint8_t *alphabet, Arena &, sbn3_team &,
                     const sbn3_lease &prepared, sbn3_lease &work, uint64_t *rail_storage,
                     ProductProgram *programs, RailProduct *cached) noexcept;
// digits[0..count), MSD first, count <= 64 * fragments of the root: out receives classes[root].limbs
// limbs. False when some byte is not a digit (out is then unspecified).
bool parse_tree_run(ParseTree &, int root, const uint8_t *digits, uint64_t count, uint64_t *out) noexcept;
} // namespace sbn::v3::radix
