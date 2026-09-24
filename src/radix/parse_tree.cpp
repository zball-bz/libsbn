#include "radix/parse_tree.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <new>
#include <string.h>
namespace sbn::v3::radix {
namespace {
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
constexpr size_t parse_group_limbs = 56; // value of eight fragments of base 63, and its largest product
unsigned pow2_floor(unsigned v) noexcept { return 1u << (31 - unsigned(__builtin_clz(v))); }
} // namespace
sbn3_query_result parse_tree_begin(unsigned base, unsigned workers, uint64_t largest_fragments, ParseTreePlan &p) noexcept {
    static_assert(__is_trivially_destructible(ParseTreePlan)); // may be raw storage
    ::new (&p) ParseTreePlan{};
    if (!base_info(base, p.base) || p.base.odd == 1 || !workers || workers > 32)
        return p.status = SBN3_UNSUPPORTED;
    p.workers = workers;
    p.top_workers = pow2_floor(workers);
    for (uint64_t n = 1; n <= group_fragments; ++n)
        p.group_limbs[n] = integer_limbs(p.base, n);
    if (p.group_limbs[group_fragments] > parse_group_limbs ||
        p.group_limbs[group_fragments / 2] + rail_limbs(p.base, 2) > parse_group_limbs)
        return p.status = SBN3_UNSUPPORTED;
    const auto &policy = tree_policy();
    const size_t share = integer_limbs(p.base, std::max<uint64_t>(largest_fragments, 1)) /
                         std::max<size_t>(1, policy.frontier_tasks_per_worker * workers);
    p.frontier_limbs = std::min(policy.team_node_limbs, std::max(policy.frontier_floor_limbs, share));
    p.rail.count = 3;
    return p.status = SBN3_SUPPORTED;
}
int ParseTreePlan::classify(uint64_t n, unsigned w) noexcept {
    const auto &policy = tree_policy();
    const size_t limbs = integer_limbs(base, n);
    if (w == 1 || n <= group_fragments || limbs < frontier_limbs)
        w = 1;
    for (unsigned j = 0; j < class_count; ++j)
        if (classes[j].fragments == n && classes[j].workers == w)
            return int(j);
    if (status != SBN3_SUPPORTED)
        return -1;
    ParseClass c{};
    c.fragments = n;
    c.workers = w;
    c.limbs = limbs;
    c.frontier = w == 1;
    if (n <= group_fragments) {
        c.group = true;
    } else {
        c.level = split_level(n);
        const uint64_t low = uint64_t(1) << c.level, high = n - low;
        c.parallel = !c.frontier && low == high &&
                     ((limbs >= policy.fork_node_limbs && w >= 2 * policy.fork_min_workers) ||
                      limbs < policy.team_node_limbs);
        const unsigned child = c.parallel ? w / 2 : w;
        c.low = classify(low, child);
        c.high = classify(high, child);
        if (c.low < 0 || c.high < 0)
            return -1;
        c.low_limbs = classes[c.low].limbs;
        c.high_limbs = classes[c.high].limbs;
        c.shift_limbs = size_t(base.twos) << c.level;
        rail.count = std::max(rail.count, c.level + 1);
        require(c.shift_limbs <= c.low_limbs && c.low_limbs <= c.limbs, SBN3_FATAL_MATH, "radix merge geometry");
        const size_t rail_size = rail_limbs(base, c.level);
        if (!(c.frontier && policy.cyclic_products && rail_linear_plan(c.high_limbs, rail_size, c.cached))) {
            c.cached = {};
            const unsigned product_workers = c.limbs < policy.wide_product_limbs ? std::min(w, 8u) : w;
            const auto rc = product_shape(c.high_limbs, rail_size, product_workers, c.product, nullptr, transcript);
            if (rc != SBN3_SUPPORTED) {
                status = rc;
                return -1;
            }
        } else {
            c.product = {};
            c.product.an = c.high_limbs;
            c.product.bn = rail_size;
        }
        const auto &l = classes[c.low], &h = classes[c.high];
        const size_t own = align_to(c.high_limbs * 8, 64);
        const size_t episode = c.cached.enabled ? c.cached.episode_bytes() : c.product.episode_bytes();
        if (c.frontier) {
            c.rest_offset = own;
            c.region_bytes = own + align_to(std::max({episode, l.region_bytes, h.region_bytes}), 64);
        } else {
            c.persist_bytes = own + l.persist_bytes + h.persist_bytes;
            const size_t le = l.frontier ? 0 : l.episode_bytes, he = h.frontier ? 0 : h.episode_bytes;
            c.high_episode_offset = c.parallel ? le : 0;
            c.episode_bytes = align_to(std::max(episode, c.parallel ? le + he : std::max(le, he)), 64);
            c.tasks = l.tasks + h.tasks;
        }
    }
    if (c.frontier)
        frontier_region_bytes = std::max(frontier_region_bytes, c.region_bytes);
    if (class_count == max_parse_classes) {
        status = SBN3_QUERY_CAPACITY;
        return -1;
    }
    classes.emplace(class_count,c);
    return int(class_count++);
}
int ParseTreePlan::add_tree(uint64_t fragments) noexcept {
    if (status != SBN3_SUPPORTED)
        return -1;
    if (!fragments || fragments > max_fragments) {
        status = SBN3_QUERY_CAPACITY;
        return -1;
    }
    return classify(fragments, top_workers);
}
int ParseTreePlan::add_product(size_t an, size_t bn) noexcept {
    if (status != SBN3_SUPPORTED || extra_count == 2)
        return -1;
    const auto rc = product_shape(an, bn, workers, extra[extra_count], nullptr, transcript);
    if (rc != SBN3_SUPPORTED) {
        status = rc;
        return -1;
    }
    return int(extra_count++);
}
sbn3_query_result ParseTreePlan::finish() noexcept {
    if (status != SBN3_SUPPORTED)
        return status;
    status = rail_finish(base, workers, rail, transcript);
    if (status != SBN3_SUPPORTED)
        return status;
    programs = {};
    for (unsigned j = 0; j < extra_count; ++j)
        programs.account(extra[j]);
    cached_bytes = 0;
    for (unsigned j = 0; j < class_count; ++j) {
        if (classes[j].cached.enabled)
            cached_bytes += align_to(classes[j].cached.prepared_bytes(), 128);
        else if (!classes[j].group)
            programs.account(classes[j].product);
    }
    return status;
}
size_t ParseTreePlan::task_bytes(int root) const noexcept {
    return align_to(size_t(classes[root].tasks) * sizeof(ParseTask), 64);
}
size_t ParseTreePlan::work_bytes(int root) const noexcept {
    const auto &c = classes[root];
    return task_bytes(root) + (c.frontier ? 0 : c.persist_bytes + c.episode_bytes) +
           size_t(workers) * align_to(frontier_region_bytes, 64);
}
void parse_tree_bind(ParseTree &t, const ParseTreePlan &plan, const uint8_t *alphabet, Arena &arena, sbn3_team &team,
                     const sbn3_lease &prepared, sbn3_lease &work, uint64_t *rail_storage,
                     ProductProgram *programs, RailProduct *cached) noexcept {
    t = {};
    t.plan = &plan;
    t.arena = &arena;
    t.team = &team;
    t.work = work;
    t.programs = programs;
    t.cached = cached;
    require(digit_plan_init(t.digits, plan.base.base, alphabet), SBN3_FATAL_ARGUMENT, "radix alphabet");
    require(work.bytes >= plan.rail.setup_bytes && !(uintptr_t(work.data) & 63) && prepared.bytes >= plan.prepared_bytes(),
            SBN3_FATAL_WORKSPACE, "radix tree storage");
    rail_build(plan.base, plan.rail, plan.workers, arena, team, work, rail_storage, t.rail);
    t.work = work; // (re-acquired by the rail build)
    ProgramSetBuilder builder(plan.programs, arena, prepared);
    for (unsigned j = 0; j < plan.extra_count; ++j)
        t.extra[j] = builder.prepare(plan.extra[j]);
    const size_t cached_at = align_to(plan.programs.bytes(), 128);
    Frame spectra = Frame::borrow(arena, prepared, static_cast<uint8_t *>(prepared.data) + cached_at, prepared.bytes - cached_at);
    for (unsigned j = 0; j < plan.class_count; ++j) {
        const auto &c = plan.classes[j];
        ::new (&programs[j]) ProductProgram{};
        ::new (&cached[j]) RailProduct{};
        if (c.cached.enabled)
            rail_product_prepare(cached[j], c.cached, spectra, t.rail[c.level]);
        else if (!c.group)
            programs[j] = builder.prepare(c.product);
    }
}
namespace {
struct Run {
    ParseTree *tree;
    const uint8_t *digits;
    uint64_t count, fragments; // digits present; fragments of the root (the first may be partial)
    bool valid;
};
// Value of fragments [first, first + n) of `values` (l1 limbs each) into out[0..group_limbs[n]).
void combine(const ParseTree &t, const uint64_t *values, size_t l1, unsigned first, unsigned n, uint64_t *out) noexcept {
    const auto &p = *t.plan;
    if (n == 1) {
        memcpy(out, values + size_t(first) * l1, l1 * 8);
        return;
    }
    const unsigned k = split_level(n), low = 1u << k, high = n - low;
    const size_t low_limbs = p.group_limbs[low], high_limbs = p.group_limbs[high], limbs = p.group_limbs[n];
    const size_t qn = p.rail.limbs[k], shift = size_t(p.base.twos) << k;
    combine(t, values, l1, first + high, low, out);
    uint64_t h[parse_group_limbs], z[2 * parse_group_limbs];
    combine(t, values, l1, first, high, h);
    sbn3_mul_basecase(z, high_limbs + qn, h, high_limbs, t.rail[k], qn);
    for (size_t j = low_limbs; j < limbs; ++j)
        out[j] = 0;
    const size_t count = std::min(high_limbs + qn, limbs - shift);
    uint64_t carry = 0;
    for (size_t j = 0; j < count; ++j) {
        const __uint128_t sum = (__uint128_t)out[shift + j] + z[j] + carry;
        out[shift + j] = uint64_t(sum);
        carry = uint64_t(sum >> 64);
    }
    for (size_t j = shift + count; carry && j < limbs; ++j)
        carry = ++out[j] == 0;
}
void leaf_group(Run &run, const ParseClass &c, uint64_t fragment, uint64_t *out) noexcept {
    auto &t = *run.tree;
    const auto &p = *t.plan;
    const size_t l1 = p.group_limbs[1];
    const unsigned n = unsigned(c.fragments);
    uint64_t values[group_fragments * 8];
    const uint64_t b8 = t.digits.b8;
    const uint64_t missing = run.fragments * fragment_digits - run.count; // leading zero digits of fragment 0
    bool valid = true;
    for (unsigned u = 0; u < n; ++u) {
        const uint64_t f = fragment + u;
        uint64_t words[fragment_words];
        if (f == 0 && missing) {
            uint8_t padded[fragment_digits];
            memset(padded, t.digits.encode[0], missing);
            memcpy(padded + missing, run.digits, fragment_digits - missing);
            valid &= parse_words(words, padded, fragment_words, t.digits);
        } else {
            valid &= parse_words(words, run.digits + (f * fragment_digits - missing), fragment_words, t.digits);
        }
        // Horner by b^8 over the eight words
        uint64_t *v = values + size_t(u) * l1;
        for (size_t j = 0; j < l1; ++j)
            v[j] = 0;
        v[0] = words[0];
        size_t used = 1;
        for (unsigned w = 1; w < fragment_words; ++w) {
            uint64_t carry = words[w];
            for (size_t j = 0; j < used; ++j) {
                const __uint128_t x = (__uint128_t)v[j] * b8 + carry;
                v[j] = uint64_t(x);
                carry = uint64_t(x >> 64);
            }
            if (carry && used < l1)
                v[used++] = carry;
        }
    }
    if (!valid)
        __atomic_store_n(&run.valid, false, __ATOMIC_RELAXED);
    combine(t, values, l1, 0, n, out);
}
// out = low value (already in out[0..low_limbs)) + (high * rail[level]) << (64 * shift_limbs)
void merge(Run &run, int index, const uint64_t *high, uint64_t *out, uint8_t *episode, sbn3_team_scope *scope) noexcept {
    auto &t = *run.tree;
    const auto &c = t.plan->classes[index];
    auto *z = reinterpret_cast<uint64_t *>(episode);
    const size_t product_limbs = c.high_limbs + c.product.bn;
    if (c.cached.enabled) {
        const auto &rp = t.cached[index];
        auto *work = episode + align_to(rp.plan.output_limbs() * 8, 64) + 64;
        auto frame = t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work, rp.plan.scratch_bytes);
        rail_product_execute(rp, frame, scope, high, z);
    } else {
        const auto &s = c.product;
        const uintptr_t after = reinterpret_cast<uintptr_t>(episode) + align_to(s.output_limbs * 8, 64);
        auto *work = reinterpret_cast<uint8_t *>((after + s.work_alignment - 1) & ~uintptr_t(s.work_alignment - 1));
        auto frame = t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work, s.work_bytes);
        product_program_execute(t.programs[index], frame, scope, {high, s.an}, {t.rail[c.level], s.bn}, {z, s.output_limbs});
    }
    const size_t count = std::min(product_limbs, c.limbs - c.shift_limbs);
    const size_t overlap = c.low_limbs - c.shift_limbs; // limbs of z that meet the low value
    const uint64_t carry = parallel_limbs::add_to_scope(scope, out + c.shift_limbs, overlap, z, std::min(overlap, count));
    if (count > overlap)
        parallel_limbs::copy_scope(scope, out + c.low_limbs, z + overlap, count - overlap);
    for (size_t j = std::max(c.shift_limbs + count, c.low_limbs); j < c.limbs; ++j)
        out[j] = 0;
    if (carry)
        parallel_limbs::add_word(out + c.low_limbs, c.limbs - c.low_limbs, carry);
}
// A frontier subtree, serially inside `region`.
void serial_node(Run &run, int index, uint64_t fragment, uint64_t *out, uint8_t *region, sbn3_team_scope *scope) noexcept {
    const auto &c = run.tree->plan->classes[index];
    if (c.group) {
        leaf_group(run, c, fragment, out);
        return;
    }
    auto *high = reinterpret_cast<uint64_t *>(region);
    uint8_t *rest = region + c.rest_offset;
    const uint64_t high_fragments = c.fragments - (uint64_t(1) << c.level);
    serial_node(run, c.low, fragment + high_fragments, out, rest, scope);
    serial_node(run, c.high, fragment, high, rest, scope);
    merge(run, index, high, out, rest, scope);
}
// The frontier tasks below a top node, in the order of the top recursion.
void collect(const ParseTreePlan &p, int index, uint64_t fragment, uint64_t *out, uint8_t *persist, ParseTask *tasks) noexcept {
    const auto &c = p.classes[index];
    if (c.frontier) {
        *tasks = {index, fragment, out};
        return;
    }
    const auto &l = p.classes[c.low];
    const uint64_t high_fragments = c.fragments - (uint64_t(1) << c.level);
    uint8_t *below = persist + align_to(c.high_limbs * 8, 64);
    collect(p, c.low, fragment + high_fragments, out, below, tasks);
    collect(p, c.high, fragment, reinterpret_cast<uint64_t *>(persist), below + l.persist_bytes, tasks + l.tasks);
}
struct TopTask {
    Run *run;
    int node;
    uint64_t *out;
    uint8_t *persist, *episode;
};
void top_node(Run &run, int index, uint64_t *out, uint8_t *persist, uint8_t *episode, sbn3_team_scope *scope) noexcept;
void top_action(void *v, sbn3_team_scope *scope) {
    auto &k = *static_cast<TopTask *>(v);
    top_node(*k.run, k.node, k.out, k.persist, k.episode, scope);
}
void top_node(Run &run, int index, uint64_t *out, uint8_t *persist, uint8_t *episode, sbn3_team_scope *scope) noexcept {
    const auto &p = *run.tree->plan;
    const auto &c = p.classes[index];
    if (c.frontier)
        return; // evaluated by its frontier task
    const auto &l = p.classes[c.low];
    auto *high = reinterpret_cast<uint64_t *>(persist);
    uint8_t *below = persist + align_to(c.high_limbs * 8, 64);
    TopTask lt{&run, c.low, out, below, episode};
    TopTask ht{&run, c.high, high, below + l.persist_bytes, episode + c.high_episode_offset};
    if (c.parallel) {
        sbn3_team_invoke2(scope, SBN3_PARALLEL_CHILDREN, sbn3_team_width(scope) / 2, top_action, &lt, top_action, &ht);
    } else {
        top_action(&lt, scope);
        top_action(&ht, scope);
    }
    merge(run, index, high, out, episode, scope);
}
struct RootTask {
    Run *run;
    int root;
    uint64_t *out;
    ParseTask *tasks;
    uint8_t *persist, *episode, *regions;
    size_t region_bytes;
    sbn3_team_scope *scope;
};
void frontier_tasks(void *v, uint64_t begin, uint64_t end, unsigned rank) {
    auto &k = *static_cast<RootTask *>(v);
    sbn3_team_scope self{k.scope->team, k.scope->first + rank, 1, false, k.scope->epoch};
    uint8_t *region = k.regions + size_t(rank) * k.region_bytes;
    for (uint64_t j = begin; j < end; ++j)
        serial_node(*k.run, k.tasks[j].node, k.tasks[j].fragment, k.tasks[j].out, region, &self);
}
} // namespace
bool parse_tree_run(ParseTree &t, int root, const uint8_t *digits, uint64_t count, uint64_t *out) noexcept {
    const auto &p = *t.plan;
    const auto &c = p.classes[root];
    require(count && count <= c.fragments * fragment_digits && count > (c.fragments - 1) * fragment_digits,
            SBN3_FATAL_ARGUMENT, "radix parse digit count");
    require(t.work.bytes >= p.work_bytes(root), SBN3_FATAL_WORKSPACE, "radix tree storage", p.work_bytes(root), t.work.bytes);
    Run run{&t, digits, count, c.fragments, true};
    if (c.group) {
        leaf_group(run, c, 0, out);
        return run.valid;
    }
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
    RootTask task{&run, root, out, reinterpret_cast<ParseTask *>(base), base + p.task_bytes(root), nullptr, nullptr,
                  align_to(p.frontier_region_bytes, 64), nullptr};
    task.episode = task.persist + (c.frontier ? 0 : c.persist_bytes);
    task.regions = task.episode + (c.frontier ? 0 : c.episode_bytes);
    collect(p, root, 0, out, task.persist, task.tasks);
    sbn3_team_run(t.team, [](void *v, sbn3_team_scope *scope) {
        auto &k = *static_cast<RootTask *>(v);
        const auto &plan = *k.run->tree->plan;
        k.scope = scope;
        sbn3_team_for(scope, 0, plan.classes[k.root].tasks, 1, SBN3_DYNAMIC, frontier_tasks, &k);
        sbn3_team_scope top{scope->team, scope->first, std::min(plan.top_workers, sbn3_team_width(scope)), false, scope->epoch};
        top_node(*k.run, k.root, k.out, k.persist, k.episode, &top);
    }, &task);
    for (unsigned j = 0; j < workers; ++j) {
        t.roots[j]->~Frame();
        t.roots[j] = nullptr;
    }
    return run.valid;
}
} // namespace sbn::v3::radix
