#include "sbn3/series.h"
#include "series/schedule.hpp"
#include "common/identity.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

namespace sbn::v3::series {
constexpr uint64_t magic = 0x53424e3353455231ULL;
constexpr size_t no_child = SIZE_MAX, max_stages = 1u << 18;
struct Node {
    sbn3_series_stage stage{};
    sbn3_series_resources resource{};
    size_t left = no_child, right = no_child, workspace = 0;
    uint64_t split = 0;
    bool parallel = false;
    sbn3_series_shape lanes{};
    SharedPlacement shared[2]{};
    uint64_t decisions[32]{};
};
struct Plan {
    uint64_t marker = magic;
    sbn3_series_spec spec{};
    sbn3_series_info info{};
    unsigned workers = 1;
    bool right_inline = false;
    bool value_lanes = false;
    bool balanced_serial = false;
    unsigned leaf_terms = 1;
    const void *split_context = nullptr;
    uint64_t (*split_point)(const void *, sbn3_series_range, double) = nullptr;
};
static size_t aligned(size_t n) {
    size_t r;
    require(align_size(n, 64, r), SBN3_FATAL_SIZE, "series alignment");
    return r;
}
static Node *nodes(Plan *p) {
    return reinterpret_cast<Node *>(reinterpret_cast<char *>(p) + aligned(sizeof(Plan)));
}
static const Node *nodes(const Plan *p) {
    return nodes(const_cast<Plan *>(p));
}
static void needs(sbn3_series_recipe recipe, unsigned n, unsigned &l, unsigned &r) {
    l = r = n;
    if (n & SBN3_SERIES_T) {
        r |= SBN3_SERIES_D;
        if (recipe == SBN3_SERIES_COMMON_P2B3)
            l |= SBN3_SERIES_U;
        if (recipe == SBN3_SERIES_BINARY_BBP)
            l |= SBN3_SERIES_D;
    }
}
static unsigned closure(sbn3_series_recipe recipe, unsigned n) {
    unsigned l, r;
    needs(recipe, n, l, r);
    return l | r;
}
struct Summary {
    size_t index = 0, workspace = 0;
    sbn3_series_shape shape{};
    double work = 0;
    sbn3_series_shape lanes{};
};
struct Builder {
    const sbn3_series_spec &spec;
    const sbn3_series_options &options;
    const sbn3_series_oracle &oracle;
    Node *destination = nullptr;
    size_t destination_count = 0;
    ScheduleRefinement refinement{};
    size_t count = 0, coarse = 0, subtrees = 0;
    size_t prepared = 0, prepared_alignment = 64;
    unsigned depth = 0;
    uint64_t hash = 1469598103934665603ULL;
    sbn3_query_result status = SBN3_SUPPORTED;
    SharedPreparationIndex<> shared_index{};
    size_t add(size_t a, size_t b) {
        size_t r = 0;
        if (!add_size(a, b, r))
            status = SBN3_QUERY_CAPACITY;
        return r;
    }
    size_t pad(size_t n) {
        size_t r = 0;
        if (!align_size(n, 64, r))
            status = SBN3_QUERY_CAPACITY;
        return r;
    }
    size_t payload(const sbn3_series_shape &s) {
        size_t n = 0;
        for (size_t k : s.limbs) {
            size_t b = 0;
            if (!mul_size(k, 8, b))
                status = SBN3_QUERY_CAPACITY;
            n = add(n, pad(b));
        }
        return n;
    }
    size_t lane_words(size_t n) {
        size_t bytes = 0;
        if (!mul_size(n, 8, bytes)) {
            status = SBN3_QUERY_CAPACITY;
            return 0;
        }
        return pad(bytes) / 8;
    }
    void lane_shape(Node &n, const Summary *l = nullptr, const Summary *r = nullptr) {
        if (!refinement.value_lanes)
            return;
        const size_t raw = l ? add(add(std::max({l->shape.limbs[0], l->shape.limbs[1], l->shape.limbs[2]}),
                                       std::max({r->shape.limbs[0], r->shape.limbs[1], r->shape.limbs[2]})),
                                   2)
                             : 0;
        for (unsigned j = 0; j < 3; ++j) {
            size_t cap = lane_words(n.stage.output.limbs[j]);
            if (l) {
                const size_t right_at = n.parallel ? l->lanes.limbs[j] : lane_words(l->shape.limbs[j]);
                cap = std::max({cap, l->lanes.limbs[j], add(right_at, r->lanes.limbs[j])});
                if (n.stage.need & (1u << j))
                    cap = std::max(cap, lane_words(raw));
            }
            n.lanes.limbs[j] = cap;
        }
    }
    sbn3_series_shape bound(sbn3_series_range r, uint64_t n, unsigned need) {
        sbn3_series_shape s{};
        if (status == SBN3_SUPPORTED)
            status = oracle.bounds(oracle.context, r, n, need, &s);
        if (status != SBN3_SUPPORTED)
            return s;
        for (unsigned j = 0; j < 3; ++j) {
            if (!(need & (1u << j)))
                s.limbs[j] = 0;
            else if (!s.limbs[j])
                status = SBN3_UNSUPPORTED;
        }
        (void)payload(s);
        return s;
    }
    double cost(sbn3_series_range r) {
        double c = oracle.work(oracle.context, r);
        if (!(c > 0) || !std::isfinite(c)) {
            status = SBN3_UNSUPPORTED;
            return 1;
        }
        return c;
    }
    uint64_t split(sbn3_series_range r, double fraction) {
        if (refinement.split_point) {
            const uint64_t m = refinement.split_point(refinement.context, r, fraction);
            if (m <= r.begin || m >= r.end)
                status = SBN3_UNSUPPORTED;
            return m;
        }
        const double target = cost(r) * fraction;
        uint64_t lo = r.begin + 1, hi = r.end - 1;
        while (lo < hi) {
            const uint64_t m = lo + (hi - lo) / 2;
            if (cost({r.begin, m}) < target)
                lo = m + 1;
            else
                hi = m;
        }
        if (lo > r.begin + 1 &&
            std::abs(cost({r.begin, lo - 1}) - target) < std::abs(cost({r.begin, lo}) - target))
            --lo;
        // Bound recursion depth even for a badly conditioned performance proxy.
        const uint64_t margin = (r.end - r.begin) / 4;
        return std::clamp(lo, r.begin + std::max(uint64_t(1), margin), r.end - std::max(uint64_t(1), margin));
    }
    size_t reserve(unsigned d) {
        depth = std::max(depth, d);
        if (count >= max_stages || d > 96)
            status = SBN3_QUERY_CAPACITY;
        return count++;
    }
    size_t resource(Node &n) {
        StagePreparation preparation{};
        if (status == SBN3_SUPPORTED)
            status = refinement.prepared_resources
                         ? refinement.prepared_resources(refinement.prepared_context?refinement.prepared_context:refinement.context, &n.stage, &n.resource, &preparation)
                         : oracle.resources(oracle.context, &n.stage, &n.resource);
        if (status != SBN3_SUPPORTED)
            return 0;
        const auto &r = n.resource;
        if (!r.alignment || (r.alignment & (r.alignment - 1)) || r.alignment > (2u << 20) || !(r.work >= 0) ||
            !std::isfinite(r.work))
            status = SBN3_UNSUPPORTED;
        if (r.prepared_bytes) {
            size_t at = 0;
            if (!r.prepared_alignment || r.prepared_alignment > (2u << 20) ||
                !align_size(prepared, r.prepared_alignment, at)) {
                status = SBN3_UNSUPPORTED;
                return 0;
            }
            prepared_alignment = std::max(prepared_alignment, r.prepared_alignment);
            n.stage.prepared_offset = at;
            prepared = add(at, r.prepared_bytes);
        }
        if(refinement.prepared_resources)
            std::copy_n(preparation.decisions,32,n.decisions);
        if (refinement.prepared_resources) for(unsigned slot=0;slot<2;++slot) {
            const auto &shared=preparation.shared.requests[slot];
            if (shared.bytes) {
                size_t at = 0;
                if (!shared.alignment || shared.alignment > (2u << 20) ||
                    !align_size(prepared, shared.alignment, at)) {
                    status = SBN3_UNSUPPORTED;
                    return 0;
                }
                const bool found = shared_index.find(shared, at);
                if (!found) {
                    prepared = add(at, shared.bytes);
                    prepared_alignment = std::max(prepared_alignment, shared.alignment);
                    shared_index.insert(shared, at);
                }
                n.shared[slot] = {at, shared.bytes, !found};
                for (auto x : shared.key) hash = identity::word(hash, x);
                hash = identity::word(hash, shared.alignment);
            }
        }
        return add(r.bytes, r.alignment ? r.alignment - 1 : 0);
    }
    void save(const Node &n) {
        if (status != SBN3_SUPPORTED)
            return;
        // Hash fields, never object padding, function addresses, or value pointers.
        auto feed = [&](uint64_t v) { hash = identity::word(hash, v); };
        for (uint64_t x : {uint64_t(n.stage.index), uint64_t(n.stage.prepared_offset), n.stage.envelope.begin,
                           n.stage.envelope.end, n.stage.max_terms, uint64_t(n.stage.workers),
                           uint64_t(n.stage.need), uint64_t(n.stage.leaf), uint64_t(n.stage.serial_envelope),
                           uint64_t(n.workspace), uint64_t(n.left), uint64_t(n.right), n.split,
                           uint64_t(n.parallel), uint64_t(n.resource.bytes), uint64_t(n.resource.alignment),
                           uint64_t(n.resource.prepared_bytes), uint64_t(n.resource.prepared_alignment)})
            feed(x);
        for (const auto &s : {n.stage.output, n.stage.left, n.stage.right, n.lanes})
            for (auto x : s.limbs)
                feed(x);
        if(refinement.prepared_resources)for(auto decision : n.decisions)feed(decision);
        if (refinement.prepared_resources) for(const auto &shared : n.shared)
            for (uint64_t x : {uint64_t(shared.offset), uint64_t(shared.bytes), uint64_t(shared.owner)})
                feed(x);
        if (destination) {
            require(n.stage.index < destination_count, SBN3_FATAL_MATH, "series oracle changed stage count");
            destination[n.stage.index] = n;
        }
    }
    Summary serial(sbn3_series_range r, uint64_t length, unsigned n, unsigned w, unsigned d,
                   unsigned serial_depth = 0) {
        Node node{};
        node.stage.index = reserve(d);
        node.stage.envelope = r;
        node.stage.workers = w;
        node.stage.need = closure(spec.recipe, n);
        node.stage.serial_envelope = 1;
        if (refinement.serial_envelope && status == SBN3_SUPPORTED) {
            status = refinement.serial_envelope(refinement.context, r, serial_depth, node.stage.need,
                                                 &length, &node.stage.output);
            if (!length || length > r.end - r.begin)
                status = SBN3_UNSUPPORTED;
            if (length <= options.leaf_terms) {
                // A small actual subtree may jump straight to this leaf stage.
                // Reserve a complete configured batch, not just the narrow
                // depth-envelope remainder (which could be e.g. five terms).
                length = std::min(uint64_t(options.leaf_terms), r.end - r.begin);
                node.stage.output = bound(r, length, node.stage.need);
            }
        } else
            node.stage.output = bound(r, length, node.stage.need);
        node.stage.max_terms = length;
        double time = 0;
        if (length <= options.leaf_terms) {
            node.stage.leaf = 1;
            node.workspace = resource(node);
            lane_shape(node);
            time = node.resource.work;
        } else if (status == SBN3_SUPPORTED) {
            const auto child = serial(r, length / 2 + length % 2, node.stage.need, w, d + 1, serial_depth + 1);
            node.left = node.right = child.index;
            node.stage.left = node.stage.right = child.shape;
            const size_t merge = resource(node);
            node.workspace = refinement.value_lanes ? std::max(child.workspace, merge)
                                                    : add(add(payload(child.shape), payload(child.shape)),
                                                          std::max(child.workspace, merge));
            lane_shape(node, &child, &child);
            time = 2 * child.work + node.resource.work;
        }
        if (!std::isfinite(time))
            status = SBN3_UNSUPPORTED;
        save(node);
        return {node.stage.index, node.workspace, node.stage.output, time, node.lanes};
    }
    Summary tree(sbn3_series_range r, unsigned n, unsigned w, unsigned prefix, unsigned d) {
        if (status != SBN3_SUPPORTED)
            return {};
        const uint64_t length = r.end - r.begin;
        const bool parallel = prefix == 0 && w > 1 && cost(r) >= options.min_parallel_work;
        const bool refine = refinement.split_serial && refinement.split_serial(refinement.context, r, n);
        if (length <= options.leaf_terms || (prefix == 0 && !parallel && !refine)) {
            ++subtrees;
            return serial(r, length, n, w, d);
        }
        Node node{};
        node.stage.index = reserve(d);
        ++coarse;
        node.stage.envelope = r;
        node.stage.max_terms = length;
        node.stage.workers = w;
        node.stage.need = n;
        node.stage.output = bound(r, length, n);
        node.parallel = parallel;
        unsigned ln, rn;
        needs(spec.recipe, n, ln, rn);
        const unsigned lw = node.parallel ? w / 2 : w, rw = node.parallel ? w - lw : w;
        node.split = split(r, node.parallel ? double(lw) / w : 0.5);
        auto l = tree({r.begin, node.split}, ln, lw, prefix ? prefix - 1 : 0, d + 1);
        auto rr = tree({node.split, r.end}, rn, rw, prefix ? prefix - 1 : 0, d + 1);
        node.left = l.index;
        node.right = rr.index;
        node.stage.left = l.shape;
        node.stage.right = rr.shape;
        const size_t merge = resource(node);
        const size_t children =
            node.parallel ? add(pad(l.workspace), rr.workspace) : std::max(l.workspace, rr.workspace);
        node.workspace = refinement.value_lanes
                             ? std::max(children, merge)
                             : add(add(payload(l.shape), payload(rr.shape)), std::max(children, merge));
        lane_shape(node, &l, &rr);
        const double time =
            (node.parallel ? std::max(l.work, rr.work) : l.work + rr.work) + node.resource.work;
        if (!std::isfinite(time))
            status = SBN3_UNSUPPORTED;
        save(node);
        return {node.stage.index, node.workspace, node.stage.output, time, node.lanes};
    }
};
static bool valid(const sbn3_series_spec *s, const sbn3_series_options *o, const sbn3_series_oracle *q) {
    return s && o && q && q->bounds && q->work && q->resources && s->recipe >= SBN3_SERIES_HYPERDESCENT &&
           s->recipe <= SBN3_SERIES_BINARY_BBP && s->range.begin < s->range.end && s->need &&
           !(s->need & ~(s->recipe == SBN3_SERIES_COMMON_P2B3 ? 7u : 3u)) && o->workers >= 1 &&
           o->workers <= 32 && o->leaf_terms >= 1 && o->leaf_terms <= 1024 && o->max_serial_prefix <= 4 &&
           o->min_parallel_work >= 0 && std::isfinite(o->min_parallel_work);
}
static sbn3_query_result candidate(const sbn3_series_spec &s, const sbn3_series_options &o,
                                   const sbn3_series_oracle &q, unsigned prefix, sbn3_series_info &info,
                                   Node *dst = nullptr, size_t destination_count = 0,
                                   const ScheduleRefinement &refinement = {}) {
    Builder b{s, o, q, dst, destination_count, refinement};
    if (refinement.right_inline)
        b.hash = identity::word(b.hash, 0x5249474854494e4cULL);
    if (refinement.value_lanes)
        b.hash = identity::word(b.hash, 0x56414c55454c414eULL);
    if (refinement.split_point || refinement.serial_envelope) {
        if (!refinement.split_policy_id || (refinement.serial_envelope && !refinement.split_point))
            return SBN3_UNSUPPORTED;
        b.hash = identity::word(b.hash, refinement.split_policy_id);
        b.hash = identity::word(b.hash, bool(refinement.serial_envelope));
    }
    const auto v = b.tree(s.range, s.need, o.workers, prefix, 0);
    info = {};
    info.output = b.bound(s.range, s.range.end - s.range.begin, s.need);
    size_t bytes = 0;
    if (!mul_size(b.count, sizeof(Node), bytes))
        b.status = SBN3_QUERY_CAPACITY;
    info.plan_bytes = b.add(aligned(sizeof(Plan)), bytes);
    info.plan_alignment = 64;
    info.workspace_bytes = v.workspace;
    if (refinement.value_lanes) {
        for (unsigned j = 0; j < 3; ++j) {
            if (s.need & (1u << j))
                info.output.limbs[j] = v.lanes.limbs[j];
            else
                info.workspace_bytes = b.add(info.workspace_bytes, v.lanes.limbs[j] * 8);
        }
    }
    info.workspace_alignment = 64;
    info.prepared_bytes = b.prepared;
    info.prepared_alignment = b.prepared_alignment;
    info.stages = b.count;
    info.coarse_nodes = b.coarse;
    info.serial_subtrees = b.subtrees;
    info.serial_prefix = prefix;
    info.max_depth = b.depth;
    info.estimated_work = v.work;
    uint64_t h = identity::word(1469598103934665603ULL, s.formula_id);
    for (uint64_t x : {s.parameter_id, uint64_t(s.recipe), s.range.begin, s.range.end, uint64_t(s.need)})
        h = identity::word(h, x);
    info.mathematical_id = h;
    info.schedule_id = identity::word(identity::word(h, b.hash), info.plan_bytes);
    return b.status;
}
static const Plan &get(const sbn3_series_plan *p) {
    require(p, SBN3_FATAL_ARGUMENT, "series plan");
    const auto &r = *reinterpret_cast<const Plan *>(p);
    require(r.marker == magic, SBN3_FATAL_LIFETIME, "series plan identity");
    return r;
}
struct Run {
    const Plan &plan;
    const sbn3_series_executor &executor;
    size_t index;
    sbn3_series_range range;
    unsigned need;
    sbn3_series_values *out;
    unsigned char *scratch;
    static size_t payload(const sbn3_series_shape &s) {
        size_t n = 0;
        for (size_t x : s.limbs)
            n += aligned(x * 8);
        return n;
    }
    static sbn3_series_values values(unsigned char *p, const sbn3_series_shape &shape, unsigned need) {
        sbn3_series_values v{};
        for (unsigned j = 0; j < 3; ++j) {
            if (need & (1u << j))
                v.value[j].mantissa = {reinterpret_cast<uint64_t *>(p), shape.limbs[j], 0, 0};
            p += aligned(shape.limbs[j] * 8);
        }
        return v;
    }
    static void validate(const sbn3_series_values &v, unsigned need) {
        for (unsigned j = 0; j < 3; ++j)
            if (need & (1u << j)) {
                const auto &m = v.value[j].mantissa;
                require(m.size <= m.capacity && m.negative <= 1, SBN3_FATAL_MATH, "series callback output",
                        m.size, m.capacity);
            }
    }
    static void action(void *p, sbn3_team_scope *scope) { static_cast<Run *>(p)->execute(scope); }
    void execute(sbn3_team_scope *scope) {
        const Node *node = &nodes(&plan)[index];
        // A ceil-sized serial envelope can have a singleton on its short arm.
        // It needs only the deepest batch stage, without creating empty ranges.
        if (range.end - range.begin == 1 || (plan.balanced_serial && node->stage.serial_envelope &&
                                             range.end - range.begin <= plan.leaf_terms))
            while (!node->stage.leaf)
                node = &nodes(&plan)[node->left];
        const auto &stage = node->stage;
        require(range.end - range.begin <= stage.max_terms, SBN3_FATAL_MATH, "series term envelope");
        const auto &resource = node->resource;
        if (stage.leaf) {
            const uintptr_t at =
                (uintptr_t(scratch) + resource.alignment - 1) & ~uintptr_t(resource.alignment - 1);
            executor.leaf(executor.context, &stage, range, need, out, reinterpret_cast<void *>(at),
                          resource.bytes, scope);
        } else {
            unsigned ln, rn;
            needs(plan.spec.recipe, need, ln, rn);
            auto l = values(scratch, stage.left, ln);
            unsigned char *rp = scratch + payload(stage.left);
            auto r = values(rp, stage.right, rn);
            unsigned char *work = rp + payload(stage.right);
            const uint64_t split = stage.serial_envelope
                                       ? (plan.balanced_serial ? plan.split_point(plan.split_context, range, .5)
                                                               : range.begin + (range.end - range.begin) / 2)
                                       : node->split;
            Run left{plan, executor, node->left, {range.begin, split}, ln, &l, work};
            const auto left_bytes = nodes(&plan)[node->left].workspace;
            Run right{plan,
                      executor,
                      node->right,
                      {split, range.end},
                      rn,
                      &r,
                      work + (node->parallel ? aligned(left_bytes) : 0)};
            const unsigned lw = nodes(&plan)[node->left].stage.workers;
            if (node->parallel && plan.right_inline) {
                // Logical operand order stays L/R. Only physical worker
                // placement changes: queue L and descend through R inline.
                const unsigned rw = nodes(&plan)[node->right].stage.workers;
                sbn3_team_invoke2(scope, SBN3_PARALLEL_CHILDREN, rw, action, &right, action, &left);
            } else {
                sbn3_team_invoke2(scope, node->parallel ? SBN3_PARALLEL_CHILDREN : SBN3_SERIAL_CHILDREN, lw,
                                  action, &left, action, &right);
            }
            const uintptr_t at =
                (uintptr_t(work) + resource.alignment - 1) & ~uintptr_t(resource.alignment - 1);
            executor.merge(executor.context, &stage, range, split, need, &l, &r, out,
                           reinterpret_cast<void *>(at), resource.bytes, scope);
        }
        validate(*out, need);
    }
};
struct LaneRun {
    const Plan &plan;
    const sbn3_series_executor &executor;
    size_t index;
    sbn3_series_range range;
    unsigned need;
    sbn3_series_values *out;
    unsigned char *scratch;
    static void action(void *p, sbn3_team_scope *scope) { static_cast<LaneRun *>(p)->execute(scope); }
    sbn3_series_values child(const Node &n, const size_t offsets[3]) {
        sbn3_series_values v{};
        for (unsigned j = 0; j < 3; ++j) {
            const auto cap = n.lanes.limbs[j];
            if (!cap)
                continue;
            const auto &parent = out->value[j].mantissa;
            require(parent.data && offsets[j] <= parent.capacity && cap <= parent.capacity - offsets[j],
                    SBN3_FATAL_WORKSPACE, "series child lane capacity");
            v.value[j].mantissa = {parent.data + offsets[j], cap, 0, 0};
        }
        return v;
    }
    void execute(sbn3_team_scope *scope) {
        const Node *node = &nodes(&plan)[index];
        if (range.end - range.begin == 1 || (plan.balanced_serial && node->stage.serial_envelope &&
                                             range.end - range.begin <= plan.leaf_terms))
            while (!node->stage.leaf)
                node = &nodes(&plan)[node->left];
        const auto &stage = node->stage;
        require(range.end - range.begin <= stage.max_terms, SBN3_FATAL_MATH, "series term envelope");
        const auto &resource = node->resource;
        const uintptr_t at =
            (uintptr_t(scratch) + resource.alignment - 1) & ~uintptr_t(resource.alignment - 1);
        if (stage.leaf) {
            executor.leaf(executor.context, &stage, range, need, out, reinterpret_cast<void *>(at),
                          resource.bytes, scope);
        } else {
            unsigned ln, rn;
            needs(plan.spec.recipe, need, ln, rn);
            const auto &nl = nodes(&plan)[node->left], &nr = nodes(&plan)[node->right];
            const uint64_t split = stage.serial_envelope
                                       ? (plan.balanced_serial ? plan.split_point(plan.split_context, range, .5)
                                                               : range.begin + (range.end - range.begin) / 2)
                                       : node->split;
            size_t offsets[3]{};
            auto l = child(nl, offsets);
            LaneRun left{plan, executor, node->left, {range.begin, split}, ln, &l, scratch};
            if (node->parallel) {
                for (unsigned j = 0; j < 3; ++j)
                    offsets[j] = nl.lanes.limbs[j];
                auto r = child(nr, offsets);
                LaneRun right{
                    plan, executor, node->right, {split, range.end}, rn, &r, scratch + aligned(nl.workspace)};
                if (plan.right_inline)
                    sbn3_team_invoke2(scope, SBN3_PARALLEL_CHILDREN, nr.stage.workers, action, &right, action,
                                      &left);
                else
                    sbn3_team_invoke2(scope, SBN3_PARALLEL_CHILDREN, nl.stage.workers, action, &left, action,
                                      &right);
                executor.merge(executor.context, &stage, range, split, need, &l, &r, out,
                               reinterpret_cast<void *>(at), resource.bytes, scope);
            } else {
                left.execute(scope);
                // Only the left result remains live. Reclaim its unused lane
                // tail for the right recursion, retaining logical L/R order.
                for (unsigned j = 0; j < 3; ++j)
                    offsets[j] = aligned(l.value[j].mantissa.size * 8) / 8;
                auto r = child(nr, offsets);
                LaneRun right{plan, executor, node->right, {split, range.end}, rn, &r, scratch};
                right.execute(scope);
                executor.merge(executor.context, &stage, range, split, need, &l, &r, out,
                               reinterpret_cast<void *>(at), resource.bytes, scope);
            }
        }
        for (unsigned j = 0; j < 3; ++j)
            if (need & (1u << j))
                require(out->value[j].mantissa.size <= stage.output.limbs[j] &&
                            out->value[j].mantissa.negative <= 1,
                        SBN3_FATAL_MATH, "series lane logical result bound");
    }
};
} // namespace sbn::v3::series

using namespace sbn::v3;
using namespace sbn::v3::series;
extern "C" void sbn3_series_child_needs(sbn3_series_recipe r, unsigned n, unsigned *l, unsigned *rr) {
    require(l && rr && r >= SBN3_SERIES_HYPERDESCENT && r <= SBN3_SERIES_BINARY_BBP && n &&
                !(n & ~(r == SBN3_SERIES_COMMON_P2B3 ? 7u : 3u)),
            SBN3_FATAL_ARGUMENT, "series needs");
    needs(r, n, *l, *rr);
}
extern "C" sbn3_query_result sbn3_series_query(const sbn3_series_spec *s, const sbn3_series_options *o,
                                               const sbn3_series_oracle *q, sbn3_series_info *info) {
    if (!info || !valid(s, o, q))
        return SBN3_UNSUPPORTED;
    return query_schedule(*s, *o, *q, *info);
}
sbn3_query_result sbn::v3::series::query_schedule(const sbn3_series_spec &spec,
                                                  const sbn3_series_options &options,
                                                  const sbn3_series_oracle &oracle, sbn3_series_info &result,
                                                  const ScheduleRefinement &refinement) {
    const auto *s = &spec;
    const auto *o = &options;
    const auto *q = &oracle;
    auto *info = &result;
    if (!valid(s, o, q))
        return SBN3_UNSUPPORTED;
    bool have = false, fit = false;
    sbn3_query_result rejection = SBN3_UNSUPPORTED;
    for (unsigned prefix = 0; prefix <= o->max_serial_prefix; ++prefix) {
        sbn3_series_info c{};
        const auto rc = candidate(*s, *o, *q, prefix, c, nullptr, 0, refinement);
        if (rc != SBN3_SUPPORTED) {
            if (rc == SBN3_QUERY_CAPACITY)
                rejection = rc;
            if (!have)
                *info = c;
            continue;
        }
        const bool fits = !o->workspace_budget || c.workspace_bytes <= o->workspace_budget;
        if (!have || (fits && (!fit || c.estimated_work < info->estimated_work)) ||
            (!fit && !fits && c.workspace_bytes < info->workspace_bytes)) {
            *info = c;
            have = true;
            fit = fits;
        }
    }
    return have ? (fit ? SBN3_SUPPORTED : SBN3_QUERY_CAPACITY) : rejection;
}
extern "C" void sbn3_series_plan_init(const sbn3_series_spec *s, const sbn3_series_options *o,
                                      const sbn3_series_oracle *q, void *storage, size_t bytes,
                                      sbn3_series_plan **out) {
    sbn3_series_info info{};
    require(out && sbn3_series_query(s, o, q, &info) == SBN3_SUPPORTED, SBN3_FATAL_ARGUMENT,
            "series plan query");
    *out = prepare_schedule(*s, *o, *q, info, storage, bytes);
}
sbn3_series_plan *sbn::v3::series::prepare_schedule(const sbn3_series_spec &spec,
                                                    const sbn3_series_options &options,
                                                    const sbn3_series_oracle &oracle,
                                                    const sbn3_series_info &info, void *storage, size_t bytes,
                                                    const ScheduleRefinement &refinement) {
    require(storage && !(uintptr_t(storage) & 63) && bytes >= info.plan_bytes, SBN3_FATAL_WORKSPACE,
            "series plan storage", info.plan_bytes, bytes);
    auto *p = new (storage) Plan{};
    p->spec = spec;
    p->workers = options.workers;
    p->right_inline = refinement.right_inline;
    p->value_lanes = refinement.value_lanes;
    p->balanced_serial = refinement.serial_envelope != nullptr;
    p->leaf_terms = options.leaf_terms;
    p->split_context = refinement.context;
    p->split_point = refinement.split_point;
    sbn3_series_info built{};
    require(candidate(spec, options, oracle, info.serial_prefix, built, nodes(p), info.stages, refinement) ==
                    SBN3_SUPPORTED &&
                built.schedule_id == info.schedule_id && built.plan_bytes == info.plan_bytes &&
                built.prepared_bytes == info.prepared_bytes &&
                built.workspace_bytes == info.workspace_bytes,
            SBN3_FATAL_MATH, "series oracle changed during preparation");
    p->info = info;
    return reinterpret_cast<sbn3_series_plan *>(p);
}
sbn3_query_result sbn::v3::series::compile_schedule(const sbn3_series_spec &spec,
    const sbn3_series_options &options,const sbn3_series_oracle &oracle,void *storage,size_t bytes,
    sbn3_series_info &info,sbn3_series_plan *&out,const ScheduleRefinement &refinement) {
    require(storage && !(uintptr_t(storage)&63) && bytes>=aligned(sizeof(Plan)),
            SBN3_FATAL_WORKSPACE,"series compilation storage");
    if(options.max_serial_prefix){
        const auto rc=query_schedule(spec,options,oracle,info,refinement);
        if(rc!=SBN3_SUPPORTED)return rc;
        out=prepare_schedule(spec,options,oracle,info,storage,bytes,refinement);return SBN3_SUPPORTED;
    }
    if(!valid(&spec,&options,&oracle))return SBN3_UNSUPPORTED;
    auto *p=::new(storage)Plan{};
    p->spec=spec;p->workers=options.workers;p->right_inline=refinement.right_inline;
    p->value_lanes=refinement.value_lanes;p->balanced_serial=refinement.serial_envelope!=nullptr;
    p->leaf_terms=options.leaf_terms;p->split_context=refinement.context;p->split_point=refinement.split_point;
    const size_t capacity=(bytes-aligned(sizeof(Plan)))/sizeof(Node);
    const auto rc=candidate(spec,options,oracle,0,info,nodes(p),capacity,refinement);
    if(rc!=SBN3_SUPPORTED)return rc;
    if(options.workspace_budget && info.workspace_bytes>options.workspace_budget)return SBN3_QUERY_CAPACITY;
    p->info=info;out=reinterpret_cast<sbn3_series_plan *>(p);return SBN3_SUPPORTED;
}
const uint64_t *sbn::v3::series::stage_decisions(const sbn3_series_plan *plan,size_t index) {
    const auto &p=get(plan);require(index<p.info.stages,SBN3_FATAL_ARGUMENT,"series decision index");
    return nodes(&p)[index].decisions;
}
SharedPlacement sbn::v3::series::shared_placement(const sbn3_series_plan *plan, size_t index, unsigned slot) {
    const auto &p = get(plan);
    require(index < p.info.stages && slot < 2, SBN3_FATAL_ARGUMENT, "shared preparation stage index");
    return nodes(&p)[index].shared[slot];
}
extern "C" void
sbn3_series_visit(const sbn3_series_plan *plan,
                  void (*fn)(void *, const sbn3_series_stage *, const sbn3_series_resources *), void *arg) {
    const auto &p = get(plan);
    require(fn, SBN3_FATAL_ARGUMENT, "series visitor");
    for (size_t j = 0; j < p.info.stages; ++j)
        fn(arg, &nodes(&p)[j].stage, &nodes(&p)[j].resource);
}
void sbn::v3::series::prepare_stages(const sbn3_series_plan *plan, sbn3_team *team,
                                     void (*fn)(void *, const sbn3_series_stage *,
                                                const sbn3_series_resources *),
                                     void *arg) {
    const auto &p = get(plan);
    struct Visit {
        const Plan *plan;
        decltype(fn) function;
        void *arg;
    };
    Visit visit{&p, fn, arg};
    auto action = [](void *ptr, sbn3_team_scope *scope) {
        auto &v = *static_cast<Visit *>(ptr);
        sbn3_team_for(
            scope, 0, v.plan->info.stages, 1, SBN3_DYNAMIC,
            [](void *a, uint64_t lo, uint64_t hi, unsigned) {
                auto &v = *static_cast<Visit *>(a);
                for (auto j = lo; j < hi; ++j)
                    v.function(v.arg, &nodes(v.plan)[j].stage, &nodes(v.plan)[j].resource);
            },
            &v);
    };
    sbn3_team_run(team, action, &visit);
}
extern "C" void sbn3_series_execute(const sbn3_series_plan *plan, const sbn3_series_executor *executor,
                                    sbn3_team *team, sbn3_series_values *out, void *scratch, size_t bytes) {
    const auto &p = get(plan);
    require(executor && executor->leaf && executor->merge && out && team &&
                sbn3_team_workers(team) == p.workers,
            SBN3_FATAL_ARGUMENT, "series run bindings");
    require(scratch && !(uintptr_t(scratch) & 63) && bytes >= p.info.workspace_bytes &&
                !overlaps(plan, p.info.plan_bytes, scratch, bytes),
            SBN3_FATAL_WORKSPACE, "series run scratch", p.info.workspace_bytes, bytes);
    for (unsigned j = 0; j < 3; ++j)
        if (p.spec.need & (1u << j)) {
            const auto &v = out->value[j].mantissa;
            require(v.data && !(uintptr_t(v.data) & 63) && v.capacity >= p.info.output.limbs[j],
                    SBN3_FATAL_WORKSPACE, "series output");
            const size_t vb = bytes_for(v.capacity, 8);
            require(!overlaps(v.data, vb, scratch, bytes) && !overlaps(v.data, vb, plan, p.info.plan_bytes),
                    SBN3_FATAL_ARGUMENT, "series output alias");
            for (unsigned k = 0; k < j; ++k)
                if (p.spec.need & (1u << k))
                    require(!overlaps(v.data, vb, out->value[k].mantissa.data,
                                      bytes_for(out->value[k].mantissa.capacity, 8)),
                            SBN3_FATAL_ARGUMENT, "series output overlap");
        }
    if (p.value_lanes) {
        auto lane_values = *out;
        auto *work = static_cast<unsigned char *>(scratch);
        const auto &root = nodes(&p)[0];
        for (unsigned j = 0; j < 3; ++j)
            if (!(p.spec.need & (1u << j)) && root.lanes.limbs[j]) {
                lane_values.value[j].mantissa = {reinterpret_cast<uint64_t *>(work), root.lanes.limbs[j], 0,
                                                 0};
                work += root.lanes.limbs[j] * 8;
            }
        LaneRun run{p, *executor, 0, p.spec.range, p.spec.need, &lane_values, work};
        sbn3_team_run(team, LaneRun::action, &run);
        for (unsigned j = 0; j < 3; ++j)
            if (p.spec.need & (1u << j))
                out->value[j] = lane_values.value[j];
    } else {
        Run run{p, *executor, 0, p.spec.range, p.spec.need, out, static_cast<unsigned char *>(scratch)};
        sbn3_team_run(team, Run::action, &run);
    }
}
