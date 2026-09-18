#include "radix/format_tree.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <new>
#include <string.h>
// -DSBN3_RADIX_TRACE: per class wall time of products, windows and leaf groups on stderr (probe builds only).
#ifdef SBN3_RADIX_TRACE
#include <stdio.h>
#include <time.h>
namespace {
struct TraceRow { uint64_t product_ns, window_ns, leaf_ns, count; };
TraceRow trace_rows[512];
uint64_t trace_now() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec; }
void trace_add(uint64_t &slot, uint64_t v) { __atomic_add_fetch(&slot, v, __ATOMIC_RELAXED); }
struct TraceSpan {
    uint64_t *slot, start;
    explicit TraceSpan(uint64_t &s) : slot(&s), start(trace_now()) {}
    ~TraceSpan() { trace_add(*slot, trace_now() - start); }
};
}
#define SBN3_RADIX_SPAN(name, slot) TraceSpan name(slot)
#else
#define SBN3_RADIX_SPAN(name, slot) ((void)0)
#endif
namespace sbn::v3::radix {
namespace {
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
unsigned pow2_floor(unsigned v) noexcept { return 1u << (31 - unsigned(__builtin_clz(v))); }
} // namespace
sbn3_query_result format_tree_begin(unsigned base, unsigned workers, uint64_t largest_fragments, FormatTreePlan &p) noexcept {
    p.~FormatTreePlan();
    ::new (&p) FormatTreePlan{};
    if (!base_info(base, p.base) || p.base.odd == 1 || !workers || workers > 32)
        return p.status = SBN3_UNSUPPORTED;
    p.workers = workers;
    p.top_workers = pow2_floor(workers);
    const auto &policy = tree_policy();
    const size_t share = node_limbs(p.base, std::max<uint64_t>(largest_fragments, 1)) /
                         std::max<size_t>(1, policy.frontier_tasks_per_worker * workers);
    p.frontier_limbs = std::min(policy.team_node_limbs, std::max(policy.frontier_floor_limbs, share));
    for (uint64_t n = 1; n <= group_fragments; ++n)
        p.group_limbs[n] = node_limbs(p.base, n);
    const uint64_t leaf_bits = power_bits(p.base.log2_base, fragment_digits + word_digits) + guard_bits;
    p.fragment_u52 = unsigned((leaf_bits + 51) / 52);
    // The leaf kernel reads whole u52 digits from the top of the fragment fraction.
    if (p.group_limbs[1] > max_fragment_limbs || p.fragment_u52 > max_fragment_u52 ||
        uint64_t(52) * p.fragment_u52 >= uint64_t(64) * p.group_limbs[1] + 52 ||
        p.group_limbs[group_fragments] + rail_limbs(p.base, 2) > group_product_limbs ||
        p.group_limbs[group_fragments / 2] > group_right_limbs)
        return p.status = SBN3_UNSUPPORTED;
    p.rail.count = 3; // leaf groups use rail[0..2]
    return p.status = SBN3_SUPPORTED;
}
// The split of one node size on `workers` workers: the wrap-around product with the cached rail spectrum
// where the FFT kernels cover it (one worker), the exact product program otherwise.
sbn3_query_result FormatTreePlan::split_plan(const NodeClass &c, unsigned w, uint64_t count, SplitPlan &out) const noexcept {
    const auto &policy = tree_policy();
    out = {};
    const size_t rail_size = rail_limbs(base, c.level);
    // Everything above the window wraps at least one limb below it when the ring holds the rest of the product.
    const size_t wrapped_ring = c.split_limbs + rail_size - c.window_limbs + 1;
    bool cyclic = w == 1 && policy.cyclic_products &&
                  rail_product_plan(c.split_limbs, rail_size, std::max(c.split_limbs, wrapped_ring), out.cyclic);
    if (cyclic) {
        // The exact tie-break: limb h of the full product from the low h + 1 limbs of the operands.
        out.gap_limbs = out.cyclic.ring - wrapped_ring + 1;
        out.low_limbs = c.split_limbs + rail_size - out.cyclic.ring + 1;
        out.low_rail_limbs = std::min(out.low_limbs, rail_size);
        out.low_bytes = align_to((out.low_limbs + out.low_rail_limbs) * 8, 64);
        if (std::max(out.low_limbs, out.low_rail_limbs) > chain_basecase_limbs) {
            ProductShape low{};
            cyclic = product_shape(out.low_limbs, out.low_rail_limbs, 1, low, nullptr, transcript) == SBN3_SUPPORTED;
            out.low_bytes = align_to(low.output_limbs * 8, 64) + low.temporary_bytes();
        }
    }
    if (cyclic) {
        out.product.an = c.split_limbs;
        out.product.bn = rail_size;
        return SBN3_SUPPORTED;
    }
    out = {};
    const unsigned product_workers = c.limbs < policy.wide_product_limbs ? std::min(w, 8u) : w;
    if (!c.frontier && policy.ring_products && c.limbs >= policy.ring_node_limbs && count >= policy.ring_min_count &&
        ring_plan(c.split_limbs, rail_size, std::max(c.split_limbs, wrapped_ring), product_workers, out.ring, transcript)) {
        bool ok = true;
        out.gap_limbs = out.ring.ring - wrapped_ring + 1;
        out.low_limbs = c.split_limbs + rail_size - out.ring.ring + 1;
        out.low_rail_limbs = std::min(out.low_limbs, rail_size);
        out.low_workers = product_workers;
        out.low_bytes = align_to((out.low_limbs + out.low_rail_limbs) * 8, 64);
        if (std::max(out.low_limbs, out.low_rail_limbs) > chain_basecase_limbs) {
            ProductShape low{};
            ok = product_shape(out.low_limbs, out.low_rail_limbs, product_workers, low, nullptr, transcript) == SBN3_SUPPORTED;
            out.low_bytes = align_to(low.output_limbs * 8, 64) + low.temporary_bytes();
        }
        if (ok) {
            out.product.an = c.split_limbs;
            out.product.bn = rail_size;
            out.product.workers = product_workers;
            return SBN3_SUPPORTED;
        }
        out = {};
    }
    return product_shape(c.split_limbs, rail_size, product_workers, out.product, nullptr, transcript);
}
int FormatTreePlan::classify(uint64_t n) noexcept {
    for (unsigned j = 0; j < class_count; ++j)
        if (classes[j].fragments == n)
            return int(j);
    if (status != SBN3_SUPPORTED)
        return -1;
    NodeClass c{};
    c.fragments = n;
    c.limbs = node_limbs(base, n);
    c.frontier = top_workers == 1 || n <= group_fragments || c.limbs < frontier_limbs;
    if (n <= group_fragments) {
        c.group = true;
    } else {
        c.level = split_level(n);
        const uint64_t left = uint64_t(1) << c.level, right = n - left;
        c.left = classify(left);
        c.right = classify(right);
        if (c.left < 0 || c.right < 0)
            return -1;
        c.right_limbs = classes[c.right].limbs;
        const uint64_t window = window_bit(base, c.limbs, c.right_limbs, c.level);
        const size_t shift = size_t(base.twos) << c.level;
        require(!(window & 63) && shift < c.limbs && window / 64 + c.right_limbs == c.limbs - shift, SBN3_FATAL_MATH,
                "radix window geometry");
        c.window_limbs = size_t(window / 64);
        c.split_limbs = c.limbs - shift;
        rail.count = std::max(rail.count, c.level + 1);
        const auto &l = classes[c.left], &r = classes[c.right];
        const size_t own = align_to(c.right_limbs * 8, 64);
        if (c.frontier) {
            const auto rc = split_plan(c, 1, 0, c.split);
            if (rc != SBN3_SUPPORTED) {
                status = rc;
                return -1;
            }
            c.rest_offset = own;
            c.region_bytes = own + align_to(std::max({c.split.episode_bytes(), l.region_bytes, r.region_bytes}), 64);
        } else {
            // A frontier child's fraction is this node's right child or a view: nothing of its own persists.
            c.persist_bytes = own + l.persist_bytes + r.persist_bytes;
            c.tasks = l.tasks + r.tasks;
            c.top_nodes = 1 + l.top_nodes + r.top_nodes;
        }
    }
    if (c.frontier)
        frontier_region_bytes = std::max(frontier_region_bytes, c.region_bytes);
    if (class_count == max_classes) {
        status = SBN3_QUERY_CAPACITY;
        return -1;
    }
    classes[class_count] = c;
    return int(class_count++);
}
int FormatTreePlan::add_tree(uint64_t fragments) noexcept {
    if (status != SBN3_SUPPORTED)
        return -1;
    if (!fragments || fragments > max_fragments || tree_count == max_trees) {
        status = SBN3_QUERY_CAPACITY;
        return -1;
    }
    const int root = classify(fragments);
    if (root < 0)
        return -1;
    trees[tree_count] = {};
    trees[tree_count].root = root;
    return int(tree_count++);
}
int FormatTreePlan::add_product(size_t an, size_t bn) noexcept {
    if (status != SBN3_SUPPORTED || extra_count == 2)
        return -1;
    const auto rc = product_shape(an, bn, workers, extra[extra_count], nullptr, transcript);
    if (rc != SBN3_SUPPORTED) {
        status = rc;
        return -1;
    }
    return int(extra_count++);
}
sbn3_query_result FormatTreePlan::finish() noexcept {
    if (status != SBN3_SUPPORTED)
        return status;
    status = rail_finish(base, workers, rail, transcript);
    if (status != SBN3_SUPPORTED)
        return status;
    const auto &policy = tree_policy();
    programs = {};
    ring_stages = 0;
    for (unsigned j = 0; j < extra_count; ++j)
        programs.account(extra[j]);
    // One prepared wrap-around product per class: the first split plan that asks for it (bind prepares the same one).
    bool wrapped[max_classes]{};
    cyclic_bytes = 0;
    auto wrap = [&](unsigned j, const RailProductPlan &plan) {
        if (!wrapped[j])
            cyclic_bytes += align_to(plan.prepared_bytes(), 128);
        wrapped[j] = true;
    };
    for (unsigned j = 0; j < class_count; ++j) {
        const auto &c = classes[j];
        if (c.group || !c.frontier)
            continue;
        if (c.split.cyclic.enabled)
            wrap(j, c.split.cyclic);
        else
            programs.account(c.split.product);
    }
    // Stages of every tree: node counts flow from each top class to its children; classes were created
    // children first, so descending index order visits parents before children... but not by size.
    for (unsigned t = 0; t < tree_count; ++t) {
        auto &tree = trees[t];
        const auto &root = classes[tree.root];
        tree.tasks = root.tasks;
        tree.top_nodes = root.top_nodes;
        uint64_t count[max_classes]{};
        count[tree.root] = 1;
        // order the top classes by descending size
        int order[max_classes];
        unsigned tops = 0;
        for (unsigned j = 0; j < class_count; ++j)
            if (!classes[j].frontier)
                order[tops++] = int(j);
        std::sort(order, order + tops, [&](int a, int b) { return classes[a].fragments > classes[b].fragments; });
        uint32_t first = 0;
        for (unsigned k = 0; k < tops; ++k) {
            const int j = order[k];
            if (!count[j])
                continue;
            const auto &c = classes[j];
            count[c.left] += count[j];
            count[c.right] += count[j];
            if (tree.stage_count == max_stages)
                return status = SBN3_QUERY_CAPACITY;
            auto &s = tree.stages[tree.stage_count++];
            s.node_class = j;
            s.count = uint32_t(count[j]);
            s.first = first;
            first += s.count;
            // Concurrent groups: as many as there are nodes, but groups keep fork_min_workers while nodes are large.
            const unsigned floor_workers = c.limbs < policy.team_node_limbs ? 1u : std::min(policy.fork_min_workers, top_workers);
            s.groups = std::min<uint32_t>({pow2_floor(unsigned(std::min<uint64_t>(count[j], 32))), top_workers / floor_workers,
                                           c.limbs < policy.team_node_limbs ? 32u : ring_max_groups});
            s.workers = top_workers / s.groups;
            const auto rc = split_plan(c, s.workers, count[j], s.split);
            if (rc != SBN3_SUPPORTED)
                return status = rc;
            if (s.split.cyclic.enabled)
                wrap(unsigned(j), s.split.cyclic);
            else if (s.split.ring.enabled) {
                s.ring_stage = int(ring_stages++);
                tree.pool_bytes = std::max(tree.pool_bytes, s.split.ring.pool_bytes(s.groups));
            } else
                programs.account(s.split.product);
            tree.episode_bytes = std::max(tree.episode_bytes, size_t(s.groups) * align_to(s.split.episode_bytes(), 64));
        }
        require(first == tree.top_nodes, SBN3_FATAL_MATH, "radix stage instances", first, tree.top_nodes);
    }
    return status;
}
size_t FormatTreePlan::work_bytes(int t) const noexcept {
    const auto &tree = trees[t];
    const auto &c = classes[tree.root];
    return align_to(size_t(tree.top_nodes) * sizeof(FormatInstance), 64) + align_to(size_t(tree.tasks) * sizeof(FormatTask), 64) +
           (c.frontier ? 0 : c.persist_bytes) + tree.episode_bytes + size_t(workers) * align_to(frontier_region_bytes, 64);
}
void format_tree_bind(FormatTree &t, const FormatTreePlan &plan, const uint8_t *alphabet, sbn3_arena &arena,
                      sbn3_team &team, const sbn3_lease &prepared, sbn3_lease &work,
                      uint64_t *rail_storage, ProductProgram *programs, RailProduct *cyclic, RingStage *rings,
                      size_t pool_offset) noexcept {
    t = {};
    t.rings = rings;
    t.ring_arena = &arena;
    t.pool_offset = pool_offset;
    t.plan = &plan;
    t.arena = &arena;
    t.team = &team;
    t.work = work;
    t.programs = programs;
    t.cyclic = cyclic;
    require(digit_plan_init(t.digits, plan.base.base, alphabet), SBN3_FATAL_ARGUMENT, "radix alphabet");
    require(work.bytes >= plan.rail.setup_bytes && !(uintptr_t(work.data) & 63), SBN3_FATAL_WORKSPACE,
            "radix tree storage");
#ifdef SBN3_RADIX_TRACE
    const uint64_t bind_start = trace_now();
#endif
    rail_build(plan.base, plan.rail, plan.workers, arena, team, work, rail_storage, t.rail);
    t.work = work; // (re-acquired by the rail build)
#ifdef SBN3_RADIX_TRACE
    const uint64_t rail_done = trace_now();
#endif
    require(prepared.bytes >= plan.prepared_bytes(), SBN3_FATAL_WORKSPACE, "radix prepared storage", plan.prepared_bytes(),
            prepared.bytes);
    ProgramSetBuilder builder(plan.programs, arena, prepared);
    for (unsigned j = 0; j < plan.extra_count; ++j)
        t.extra[j] = builder.prepare(plan.extra[j]);
    const size_t cyclic_at = align_to(plan.programs.bytes(), 128);
    Frame spectra = Frame::borrow(arena, prepared, static_cast<uint8_t *>(prepared.data) + cyclic_at, prepared.bytes - cyclic_at);
    for (unsigned j = 0; j < plan.program_slots(); ++j)
        ::new (&programs[j]) ProductProgram{};
    for (unsigned j = 0; j < plan.class_count; ++j)
        ::new (&cyclic[j]) RailProduct{};
    auto prepare = [&](const NodeClass &c, const SplitPlan &split, ProductProgram &slot) {
        const unsigned index = unsigned(&c - plan.classes);
        if (split.ring.enabled)
            return; // replayed below, bound stage by stage at run time
        if (!split.cyclic.enabled)
            slot = builder.prepare(split.product);
        else if (!cyclic[index].tables)
            rail_product_prepare(cyclic[index], split.cyclic, spectra, t.rail[c.level]);
    };
    for (unsigned j = 0; j < plan.class_count; ++j)
        if (!plan.classes[j].group && plan.classes[j].frontier)
            prepare(plan.classes[j], plan.classes[j].split, programs[j]);
    for (unsigned k = 0; k < plan.tree_count; ++k)
        for (unsigned s = 0; s < plan.trees[k].stage_count; ++s) {
            const auto &stage = plan.trees[k].stages[s];
            prepare(plan.classes[stage.node_class], stage.split, programs[plan.class_count + k * max_stages + s]);
            if (stage.ring_stage >= 0) {
                ::new (&rings[stage.ring_stage]) RingStage{};
                ring_replay(stage.split.ring, uint64_t(stage.ring_stage) + 1, rings[stage.ring_stage]);
            }
        }
#ifdef SBN3_RADIX_TRACE
    fprintf(stderr, "bind: rail %.3f ms, programs and spectra %.3f ms (%u classes)\n", (rail_done - bind_start) / 1e6,
            (trace_now() - rail_done) / 1e6, plan.class_count);
#endif
}
namespace {
// ---- execute ----------------------------------------------------------------------------------------
struct Run {
    FormatTree *tree;
    uint8_t *out;
    uint64_t *first, *overlap;
};
void leaf_group(Run &run, const NodeClass &c, const uint64_t *y, uint64_t fragment) noexcept {
    auto &t = *run.tree;
#ifdef SBN3_RADIX_TRACE
    trace_add(trace_rows[&c - t.plan->classes].count, 1);
#endif
    SBN3_RADIX_SPAN(span, trace_rows[&c - t.plan->classes].leaf_ns);
    const auto &p = *t.plan;
    const uint64_t *lane[8]{};
    uint64_t pool[group_fragments - 1][group_right_limbs];
    uint64_t z[group_product_limbs];
    unsigned used = 0;
    struct Item {
        const uint64_t *y;
        size_t limbs;
        unsigned n, slot;
    } stack[4];
    unsigned depth = 0;
    Item it{y, c.limbs, unsigned(c.fragments), 0};
    for (;;) {
        while (it.n > 1) {
            const unsigned k = split_level(it.n), left = 1u << k, right = it.n - left;
            const size_t right_limbs = p.group_limbs[right], left_limbs = p.group_limbs[left];
            const size_t qn = p.rail.limbs[k];
            // the top twos * 2^k limbs only reach limbs above the window
            const size_t used_limbs = it.limbs - (size_t(p.base.twos) << k);
            sbn3_mul_basecase(z, used_limbs + qn, it.y, used_limbs, t.rail[k], qn);
            uint64_t *r = pool[used++];
            take_window(r, right_limbs, z, used_limbs + qn, window_bit(p.base, it.limbs, right_limbs, k));
            stack[depth++] = {r, right_limbs, right, it.slot + left};
            it = {it.y + (it.limbs - left_limbs), left_limbs, left, it.slot};
        }
        lane[it.slot] = it.y;
        if (!depth)
            break;
        it = stack[--depth];
    }
    alignas(64) uint64_t words[8 * (fragment_words + 1)];
    extract_words(words, lane, unsigned(p.group_limbs[1]), p.fragment_u52, fragment_words + 1, t.digits);
    for (unsigned u = 0; u < c.fragments; ++u) {
        const uint64_t *w = words + u * (fragment_words + 1);
        emit_words(run.out + (fragment + u) * fragment_digits, w, fragment_words, t.digits);
        run.first[fragment + u] = w[0];
        run.overlap[fragment + u] = w[fragment_words];
    }
}
// The right child of a node: the window of y * rail[level].
void split(Run &run, int index, const SplitPlan &plan, const ProductProgram &program, sbn3_mul_binding *consumer,
           const uint64_t *y, uint64_t *right, uint8_t *episode, sbn3_team_scope *scope) noexcept {
    auto &t = *run.tree;
    const auto &c = t.plan->classes[index];
    auto *z = reinterpret_cast<uint64_t *>(episode);
    const size_t window = c.window_limbs;
    if (plan.wraps()) {
        // Wrap-around product with the cached rail spectrum (rail_product.hpp, ring_product.hpp): exact window.
        const size_t residue = plan.cyclic.enabled ? plan.cyclic.ring : plan.ring.output_limbs;
        auto *work = episode + align_to(residue * 8, 64) + 64;
        auto frame = t.roots[sbn3_team_first_worker(scope)]->borrowed_view(
            work, plan.cyclic.enabled ? std::max(plan.cyclic.scratch_bytes, plan.low_bytes) : plan.low_bytes);
        {
            SBN3_RADIX_SPAN(span, trace_rows[index].product_ns);
            if (plan.cyclic.enabled) {
                rail_product_execute(t.cyclic[index], frame, scope, y, z);
            } else {
                sbn3_product_inputs in{};
                in.b = {y, c.split_limbs};
                sbn3_product_execute_on_scope(consumer, scope, &in, {z, plan.ring.output_limbs});
            }
        }
#ifdef SBN3_RADIX_TRACE
        trace_add(trace_rows[index].count, 1);
#endif
        SBN3_RADIX_SPAN(window_span, trace_rows[index].window_ns);
        memcpy(right, z + window, c.right_limbs * 8);
        uint64_t clean = 0;
        for (size_t j = window - plan.gap_limbs; j < window; ++j)
            clean |= z[j];
        if (!clean) {
            // Gap all zero: either nothing carried, or a carry ran through a gap of ones. Limb h of the
            // exact product of the low limbs tells which.
            const size_t h = plan.low_limbs - 1;
            uint64_t any = 0;
            for (size_t j = 0; j < plan.low_limbs && !any; ++j)
                any = y[j];
            bool carried = false;
            if (any) { // (zero low limbs of the fraction make limb h zero)
                FrameMark mark(frame);
                uint64_t *low = nullptr;
                if (std::max(plan.low_limbs, plan.low_rail_limbs) <= chain_basecase_limbs) {
                    low = frame.alloc<uint64_t>(plan.low_limbs + plan.low_rail_limbs);
                    sbn3_mul_basecase(low, plan.low_limbs + plan.low_rail_limbs, y, plan.low_limbs, t.rail[c.level],
                                      plan.low_rail_limbs);
                } else {
                    ProductProgramPlan pp{};
                    ProductShape shape{};
                    require(product_shape(plan.low_limbs, plan.low_rail_limbs, plan.low_workers, shape, &pp) == SBN3_SUPPORTED,
                            SBN3_FATAL_MATH, "radix tie product");
                    low = frame.alloc<uint64_t>(shape.output_limbs);
                    auto tables = frame.subframe(align_to(shape.prepared_bytes, 128) + 256, 128);
                    const auto tie = product_program_prepare(pp, tables);
                    auto scratch = frame.subframe(shape.work_bytes, shape.work_alignment);
                    product_program_execute(tie, scratch, scope, {y, plan.low_limbs}, {t.rail[c.level], plan.low_rail_limbs},
                                            {low, shape.output_limbs});
                }
                carried = low[h] != 0;
            }
            if (carried)
                for (size_t j = 0; j < c.right_limbs && right[j]-- == 0; ++j) {
                }
        }
        return;
    }
    const auto &s = plan.product;
    const uintptr_t after = reinterpret_cast<uintptr_t>(episode) + align_to(s.output_limbs * 8, 64);
    auto *work = reinterpret_cast<uint8_t *>((after + s.work_alignment - 1) & ~uintptr_t(s.work_alignment - 1));
    auto frame = t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work, s.work_bytes);
    {
        SBN3_RADIX_SPAN(span, trace_rows[index].product_ns);
        product_program_execute(program, frame, scope, {y, s.an}, {t.rail[c.level], s.bn}, {z, s.output_limbs});
    }
#ifdef SBN3_RADIX_TRACE
    trace_add(trace_rows[index].count, 1);
#endif
    SBN3_RADIX_SPAN(window_span, trace_rows[index].window_ns);
    parallel_limbs::each_scope(scope, c.right_limbs, parallel_limbs::scope_parts(scope, c.right_limbs),
                               [&](size_t b, size_t e, unsigned) { memcpy(right + b, z + window + b, (e - b) * 8); });
}
// A frontier subtree, serially inside `region`.
void serial_node(Run &run, int index, const uint64_t *y, uint8_t *region, uint64_t fragment, sbn3_team_scope *scope) noexcept {
    auto &t = *run.tree;
    const auto &p = *t.plan;
    const auto &c = p.classes[index];
    if (c.group) {
        leaf_group(run, c, y, fragment);
        return;
    }
    auto *right = reinterpret_cast<uint64_t *>(region);
    uint8_t *rest = region + c.rest_offset;
    split(run, index, c.split, t.programs[index], nullptr, y, right, rest, scope);
    serial_node(run, c.left, y + (c.limbs - p.classes[c.left].limbs), rest, fragment, scope);
    serial_node(run, c.right, right, rest, fragment + (uint64_t(1) << c.level), scope);
}
// Instances of every stage and the frontier tasks, in depth-first order.
struct Collector {
    const FormatTreePlan *plan;
    FormatInstance *instances;
    FormatTask *tasks;
    uint32_t next[max_classes];
    uint32_t task = 0;
    void visit(int index, const uint64_t *y, uint8_t *persist, uint64_t fragment) noexcept {
        const auto &c = plan->classes[index];
        if (c.frontier) {
            tasks[task++] = {index, y, fragment};
            return;
        }
        auto *right = reinterpret_cast<uint64_t *>(persist);
        instances[next[index]++] = {y, right};
        const auto &l = plan->classes[c.left];
        uint8_t *below = persist + align_to(c.right_limbs * 8, 64);
        visit(c.left, y + (c.limbs - l.limbs), below, fragment);
        visit(c.right, right, below + l.persist_bytes, fragment + (uint64_t(1) << c.level));
    }
};
struct RootTask {
    Run *run;
    int tree;
    FormatInstance *instances;
    FormatTask *tasks;
    uint8_t *episodes, *regions;
    size_t region_bytes;
    sbn3_team_scope *scope;
    // the stage being fanned out
    const Stage *stage;
    const ProductProgram *program;
    const RingBound *bound;
    unsigned first_stage, last_stage; // stages [first, last) of one team episode
    bool frontier;
};
struct GroupTask {
    RootTask *root;
    uint32_t low, high; // groups [low, high) of the current stage
};
void fan(void *v, sbn3_team_scope *scope) {
    auto &g = *static_cast<GroupTask *>(v);
    auto &k = *g.root;
    const auto &stage = *k.stage;
    if (g.high - g.low > 1) {
        const uint32_t mid = g.low + (g.high - g.low) / 2;
        GroupTask a{g.root, g.low, mid}, b{g.root, mid, g.high};
        sbn3_team_invoke2(scope, SBN3_PARALLEL_CHILDREN, sbn3_team_width(scope) / 2, fan, &a, fan, &b);
        return;
    }
    // one group: a contiguous share of the stage's nodes, its own episode memory
    const uint32_t q = stage.count / stage.groups, r = stage.count % stage.groups;
    const uint32_t begin = g.low * q + std::min(g.low, r), end = begin + q + (g.low < r);
    uint8_t *episode = k.episodes + size_t(g.low) * align_to(stage.split.episode_bytes(), 64);
    for (uint32_t j = begin; j < end; ++j) {
        const auto &node = k.instances[stage.first + j];
        split(*k.run, stage.node_class, stage.split, *k.program, k.bound ? k.bound->consumers[g.low] : nullptr, node.y,
              node.right, episode, scope);
    }
}
void frontier_tasks(void *v, uint64_t begin, uint64_t end, unsigned rank) {
    auto &k = *static_cast<RootTask *>(v);
    // One worker of the running loop: a scope of its own for the serial products.
    sbn3_team_scope self{k.scope->team, k.scope->first + rank, 1, false, k.scope->epoch};
    uint8_t *region = k.regions + size_t(rank) * k.region_bytes;
    for (uint64_t j = begin; j < end; ++j)
        serial_node(*k.run, k.tasks[j].node, k.tasks[j].y, region, k.tasks[j].fragment, &self);
}
} // namespace
size_t increment_digits(uint8_t *out, size_t count, const DigitPlan &p) noexcept {
    const uint8_t top = uint8_t(p.base - 1);
    size_t j = count;
    while (j-- > 0) {
        const uint8_t v = p.identity ? out[j] : p.decode[out[j]];
        if (v != top) {
            out[j] = p.encode[v + 1];
            return j;
        }
        out[j] = p.encode[0];
    }
    return 0;
}
uint64_t format_tree_run(FormatTree &t, int tree_index, const uint64_t *y, uint8_t *out, uint64_t *first,
                         uint64_t *overlap) noexcept {
    const auto &p = *t.plan;
    const auto &tree = p.trees[tree_index];
    const auto &c = p.classes[tree.root];
    require(t.work.bytes >= p.work_bytes(tree_index), SBN3_FATAL_WORKSPACE, "radix tree storage", p.work_bytes(tree_index),
            t.work.bytes);
    Run run{&t, out, first, overlap};
    if (c.group) {
        leaf_group(run, c, y, 0);
    } else {
        struct alignas(64) RootStorage {
            unsigned char bytes[(sizeof(Frame) + 63) & ~size_t(63)];
        };
        RootStorage storage[32];
        const unsigned workers = sbn3_team_workers(t.team);
        for (unsigned j = 0; j < workers; ++j) {
            auto *frame = ::new (storage[j].bytes) Frame(*t.arena, t.work);
            frame->allocate(t.work.bytes);
            t.roots[j] = frame;
        }
        auto *base = static_cast<uint8_t *>(t.work.data);
        RootTask task{};
        task.run = &run;
        task.tree = tree_index;
        task.instances = reinterpret_cast<FormatInstance *>(base);
        base += align_to(size_t(tree.top_nodes) * sizeof(FormatInstance), 64);
        task.tasks = reinterpret_cast<FormatTask *>(base);
        base += align_to(size_t(tree.tasks) * sizeof(FormatTask), 64);
        uint8_t *persist = base;
        base += c.frontier ? 0 : c.persist_bytes;
        task.episodes = base;
        task.regions = base + tree.episode_bytes;
        task.region_bytes = align_to(p.frontier_region_bytes, 64);
        Collector collect{&p, task.instances, task.tasks, {}, 0};
        for (unsigned s = 0; s < tree.stage_count; ++s)
            collect.next[tree.stages[s].node_class] = tree.stages[s].first;
        collect.visit(tree.root, y, persist, 0);
        require(collect.task == tree.tasks, SBN3_FATAL_MATH, "radix frontier tasks");
        // Episodes: runs of stages that need no binding share one team run; a ring stage is bound by this
        // (owner) thread around its own run. The frontier loop joins the last episode.
        auto episode = [](void *v, sbn3_team_scope *scope) {
            auto &k = *static_cast<RootTask *>(v);
            const auto &plan = *k.run->tree->plan;
            const auto &shape = plan.trees[k.tree];
            k.scope = scope;
            // The staged part runs on the power-of-two prefix of the team.
            sbn3_team_scope top{scope->team, scope->first, std::min(plan.top_workers, sbn3_team_width(scope)), false,
                                scope->epoch};
            for (unsigned s = k.first_stage; s < k.last_stage; ++s) {
                k.stage = &shape.stages[s];
                k.program = &k.run->tree->programs[plan.class_count + unsigned(k.tree) * max_stages + s];
                GroupTask all{&k, 0, k.stage->groups};
                fan(&all, &top);
            }
            if (k.frontier)
                sbn3_team_for(scope, 0, shape.tasks, 1, SBN3_DYNAMIC, frontier_tasks, &k);
        };
#ifdef SBN3_RADIX_TRACE
        fprintf(stderr, "work breakdown: instances+tasks %.1f MB, persist %.1f MB, stage episodes %.1f MB, frontier regions %u x %.1f MB, pool %.1f MB\n",
                (size_t(tree.top_nodes) * sizeof(FormatInstance) + size_t(tree.tasks) * sizeof(FormatTask)) / 1e6,
                (c.frontier ? 0 : c.persist_bytes) / 1e6, tree.episode_bytes / 1e6, p.workers, p.frontier_region_bytes / 1e6,
                p.pool_bytes() / 1e6);
        for (unsigned s = 0; s < tree.stage_count; ++s)
            fprintf(stderr, "  stage %2u class %3d limbs %9zu count %5u groups %2u x w %2u episode %.1f MB each (%s)\n", s,
                    tree.stages[s].node_class, p.classes[tree.stages[s].node_class].limbs, tree.stages[s].count,
                    tree.stages[s].groups, tree.stages[s].workers, tree.stages[s].split.episode_bytes() / 1e6,
                    tree.stages[s].split.ring.enabled ? "ring" : tree.stages[s].split.cyclic.enabled ? "fft wrap" : "program");
#endif
        RingBound bound{};
        for (unsigned s = 0; s <= tree.stage_count;) {
            unsigned e = s;
            if (s < tree.stage_count && tree.stages[s].ring_stage >= 0) {
                const auto &stage = tree.stages[s];
#ifdef SBN3_RADIX_TRACE
                const uint64_t bind_begin = trace_now(); // ring stage trace
#endif
                ring_bind(bound, stage.split.ring, t.rings[stage.ring_stage], stage.groups, t.ring_arena, t.pool_offset,
                          p.pool_bytes(), t.team, t.rail[p.classes[stage.node_class].level]);
#ifdef SBN3_RADIX_TRACE
                fprintf(stderr, "ring stage class %d: np %u T %d alg %u ring %zu groups %u x w %u, bind+spectrum %.3f ms\n", stage.node_class,
                        stage.split.ring.np, stage.split.ring.trunk_bits, stage.split.ring.algorithm, stage.split.ring.ring,
                        stage.groups, stage.workers, (trace_now() - bind_begin) / 1e6);
#endif
                task.bound = &bound;
                e = s + 1;
            } else {
                task.bound = nullptr;
                while (e < tree.stage_count && tree.stages[e].ring_stage < 0)
                    ++e;
            }
            task.first_stage = s;
            task.last_stage = e;
            task.frontier = e == tree.stage_count;
#ifdef SBN3_RADIX_TRACE
            const uint64_t episode_begin = trace_now();
#endif
            sbn3_team_run(t.team, episode, &task);
#ifdef SBN3_RADIX_TRACE
            const uint64_t episode_end = trace_now();
#endif
            if (task.bound)
                ring_unbind(bound, t.ring_arena);
#ifdef SBN3_RADIX_TRACE
            fprintf(stderr, "episode stages [%u,%u)%s: run %.3f ms, unbind %.3f ms\n", s, e, task.frontier ? " + frontier" : "",
                    (episode_end - episode_begin) / 1e6, (trace_now() - episode_end) / 1e6);
#endif
            if (task.frontier)
                break;
            s = e;
        }
        for (unsigned j = 0; j < workers; ++j) {
            t.roots[j]->~Frame();
            t.roots[j] = nullptr;
        }
    }
#ifdef SBN3_RADIX_TRACE
    for (unsigned j = 0; j < p.class_count; ++j) {
        const auto &k = p.classes[j];
        if (!trace_rows[j].count)
            continue;
        unsigned groups = 1, width = 1, product_workers = 1;
        bool wrapped = k.split.cyclic.enabled;
        for (unsigned s = 0; s < tree.stage_count; ++s)
            if (tree.stages[s].node_class == int(j)) {
                groups = tree.stages[s].groups;
                width = tree.stages[s].workers;
                product_workers = tree.stages[s].split.product.workers;
                wrapped = tree.stages[s].split.cyclic.enabled;
            }
        fprintf(stderr, "class %3u frag %8llu %s groups %2u x w %2u (product w %2u%s) limbs %8zu count %6llu product %9.3f window %8.3f leaf %9.3f ms\n",
                j, (unsigned long long)k.fragments, k.frontier ? "frontier" : "stage   ", groups, width, product_workers,
                wrapped ? ", wrap" : "", k.limbs, (unsigned long long)trace_rows[j].count, trace_rows[j].product_ns / 1e6,
                trace_rows[j].window_ns / 1e6, trace_rows[j].leaf_ns / 1e6);
        trace_rows[j] = {};
    }
    const uint64_t seam_start = trace_now();
#endif
    // Seams, right to left: fragment i's overlap word against the (final) first word of fragment i + 1.
    const uint64_t b8 = t.digits.b8;
    for (uint64_t i = c.fragments - 1; i-- > 0;) {
        const uint64_t f = first[i + 1], o = overlap[i];
        if (o == f)
            continue;
        require((o + 1 == b8 ? 0 : o + 1) == f, SBN3_FATAL_MATH, "radix seam", o, f);
        if (f)
            continue;
        // Fragment i borrowed across the seam: add one to its 64 digits (carry out dropped).
        const size_t changed = increment_digits(out + i * fragment_digits, fragment_digits, t.digits);
        if (changed < word_digits)
            first[i] = first[i] + 1 == b8 ? 0 : first[i] + 1;
    }
#ifdef SBN3_RADIX_TRACE
    fprintf(stderr, "seam pass %.3f ms\n", (trace_now() - seam_start) / 1e6);
#endif
    return overlap[c.fragments - 1];
}
} // namespace sbn::v3::radix
