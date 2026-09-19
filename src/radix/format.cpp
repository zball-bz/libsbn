// Format service: dyadic number -> digits (include/sbn3/radix.h).
//
//   integer part   a = floor(X):  root of the tree is (a + 3/4) / b^(64 F): the numerator times the
//                  reciprocal of b^(64 F), a Newton INVERSE computed once at bind (error below four
//                  units of the root). The last fragment is exact because the true value sits 3/4
//                  above a digit boundary.
//   fraction part  f = frac(X):   root is the top limbs of f, no division. The tree returns floor or
//                  floor - 1 of f b^(64 F + 8); only a guard of all (b-1) digits leaves the requested
//                  prefix open, and then it is settled exactly (see resolve()).
//   base 2^s       linear bit extraction.
#include "sbn3/radix.h"
#include "sbn3/newton.h"
#include "radix/format_tree.hpp"
#include "radix/bits.hpp"
#include "common/identity.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <new>
#include <string.h>
namespace sbn::v3::radix {
namespace {
constexpr uint64_t plan_magic = 0x53424e3352465031ULL, binding_magic = 0x53424e3352464231ULL;
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
enum IntegerPath : unsigned { integer_none = 0, integer_schoolbook = 1, integer_tree = 2 };
struct Plan {
    uint64_t marker = 0;
    sbn3_format_spec spec{};
    sbn3_radix_options options{};
    sbn3_format_info info{};
    unsigned shift = 0; // base == 2^shift, else 0
    int64_t point = 0;  // bit index of the binary point inside the mantissa
    uint64_t integer_bits = 0, fraction_bits = 0;
    unsigned integer_path = integer_none;
    size_t integer_limbs = 0;
    uint64_t integer_fragments = 0, fraction_fragments = 0;
    uint64_t tree_digits = 0; // fraction digits the tree decides (the rest of an EXACT request are zeros)
    size_t divide_limbs = 0;
    size_t pool_offset = 0, pool_bytes = 0;
    size_t prepared_offset = 0, prepared_bytes = 0, values_offset = 0, values_bytes = 0, divide_offset = 0,
           divide_bytes = 0, work_offset = 0, work_bytes = 0;
    size_t rail_at = 0, side_at = 0, numerator_at = 0, inverse_at = 0, root_at = 0, small_at = 0; // limbs in values
    int root_product = -1; // tree.extra index of numerator * reciprocal
    size_t divide_alignment = 0;   // of the Newton INVERSE storage (integer tree)
    unsigned divide_lease_peak = 0;
    uint64_t tree_id = 0, seal = 0;
};
// The plan value: [sealed layout][Newton plan of the reciprocal][number of recorded searches][the transcript].
// Bind assembles the tree plan again (it is far larger than a plan value) from these: no search runs twice for
// one conversion. Only the parts a plan uses are written and read (a plan without a tree is a few hundred bytes
// of a zeroed value).
constexpr size_t value_divide_at = (sizeof(Plan) + 7) & ~size_t(7), value_count_at = value_divide_at + sizeof(sbn3_newton_plan),
                 value_entries_at = value_count_at + 8;
static_assert(value_entries_at + PlanTranscript::capacity * 8 <= sizeof(sbn3_format_plan));
uint64_t seal(const Plan &p, const sbn3_newton_plan &divide, const PlanTranscript &t) noexcept {
    uint64_t h = identity::fnv_seed;
    const uint64_t fields[] = {p.marker, p.spec.base, p.spec.limbs, uint64_t(p.spec.exponent2), p.spec.fraction_digits,
                               uint64_t(p.spec.mode), p.options.workers, p.options.memory_budget, p.options.use_alphabet, p.options.repeated,
                               p.shift, uint64_t(p.point), p.integer_bits, p.fraction_bits, p.integer_path, p.integer_limbs,
                               p.integer_fragments, p.fraction_fragments, p.tree_digits, p.divide_limbs, p.prepared_offset,
                               p.prepared_bytes, p.values_offset, p.values_bytes, p.divide_offset, p.divide_bytes,
                               p.work_offset, p.work_bytes, p.pool_offset, p.pool_bytes, p.rail_at, p.side_at, p.numerator_at, p.inverse_at, uint64_t(p.root_product),
                               p.root_at, p.small_at, p.tree_id, p.info.storage_bytes, p.info.digit_bytes,
                               p.info.fraction_digits, p.info.integer_digits, p.info.control_bytes,
                               p.divide_alignment, p.divide_lease_peak, t.count,
                               p.info.storage_alignment, p.info.lease_peak, p.info.workers, p.info.fraction_offset};
    for (uint64_t f : fields)
        h = identity::word(h, f);
    for (unsigned j = 0; j < 64; j += 8) {
        uint64_t w = 0;
        memcpy(&w, p.options.alphabet + j, 8);
        h = identity::word(h, w);
    }
    for (uint32_t j = 0; j < t.count; ++j)
        h = identity::word(h, t.entry[j]);
    if (p.divide_limbs)
        for (uint64_t w : divide.opaque)
            h = identity::word(h, w);
    return h;
}
uint64_t tree_identity(const FormatTreePlan &t) noexcept {
    uint64_t h = identity::fnv_seed;
    h = identity::word(h, t.class_count);
    h = identity::word(h, t.rail.count);
    h = identity::word(h, t.prepared_bytes());
    for (unsigned j = 0; j < t.class_count; ++j) {
        h = identity::word(h, t.classes[j].fragments);
        h = identity::word(h, t.classes[j].region_bytes + t.classes[j].persist_bytes);
        h = identity::word(h, t.classes[j].split.product.arithmetic_id + t.classes[j].split.cyclic.ring);
    }
    for (unsigned k = 0; k < t.tree_count; ++k)
        for (unsigned s = 0; s < t.trees[k].stage_count; ++s) {
            const auto &stage = t.trees[k].stages[s];
            h = identity::word(h, uint64_t(stage.node_class) << 32 | stage.groups << 8 | stage.workers);
            h = identity::word(h, stage.split.product.arithmetic_id + stage.split.cyclic.ring);
        }
    return h;
}
unsigned top_bit(uint64_t v) noexcept { return 63 - unsigned(__builtin_clzll(v)); }
// ---- binding -----------------------------------------------------------------------------------------
struct Binding {
    uint64_t marker = 0;
    Plan plan{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, prepared{}, values{}, work{};
    // The tree plan (175 KB) is constructed by format_tree_begin, once, and only for plans that have a tree; nothing
    // reads it otherwise. The user-provided constructor keeps value-initialisation of the binding from
    // zero-filling it first.
    alignas(FormatTreePlan) unsigned char tree_plan_storage[sizeof(FormatTreePlan)];
    FormatTreePlan &tree_plan() noexcept { return *reinterpret_cast<FormatTreePlan *>(tree_plan_storage); }
    const FormatTreePlan &tree_plan() const noexcept { return *reinterpret_cast<const FormatTreePlan *>(tree_plan_storage); }
    Binding() noexcept {}
    FormatTree tree{};
    int integer_root = -1, fraction_root = -1;
    sbn3_newton_plan divide{};
    unsigned denominator_shift = 0; // numerator = (4a + 3) << denominator_shift
    uint8_t encode[64]{};           // base 2^s digits
    uint64_t *limbs(size_t at) const noexcept { return static_cast<uint64_t *>(values.data) + at; }
};
// ---- plan --------------------------------------------------------------------------------------------
struct Assembly {
    Plan &p;
    FormatTreePlan &tree;
    PlanTranscript &transcript; // records (query) or replays (bind) the product searches
    const Plan *sealed = nullptr; // bind: the sealed plan, whose Newton plan is in `divide` already
    int integer_root = -1, fraction_root = -1;
    sbn3_newton_plan divide{};
    sbn3_newton_info divide_info{};
    Chain integer_chain{}; // odd^(64 integer_fragments) with the winners of its products
    bool tree_begun = false; // `tree` is raw storage until *_tree_begin constructed it
    sbn3_query_result run() noexcept;
    sbn3_query_result assemble() noexcept;
};
sbn3_query_result Assembly::run() noexcept {
    const auto rc = assemble();
    if (tree_begun)
        tree.transcript = nullptr; // the tree plan outlives the transcript
    return rc;
}
sbn3_query_result Assembly::assemble() noexcept {
    const auto &s = p.spec;
    BaseInfo base{};
    if (!base_info(s.base, base) || !s.limbs || unsigned(s.mode) > SBN3_RADIX_ENCLOSED || !p.options.workers ||
        p.options.workers > 32 || p.options.use_alphabet > 1)
        return SBN3_UNSUPPORTED;
    if (s.limbs > (size_t(1) << 40) || s.exponent2 > (int64_t(1) << 46) || s.exponent2 < -(int64_t(1) << 46) ||
        s.fraction_digits > (uint64_t(1) << 40))
        return SBN3_QUERY_CAPACITY;
    if (s.mode == SBN3_RADIX_ENCLOSED && s.exponent2 > 0)
        return SBN3_UNSUPPORTED;
    if (p.options.use_alphabet)
        for (unsigned i = 0; i < s.base; ++i)
            for (unsigned j = 0; j < i; ++j)
                if (p.options.alphabet[i] == p.options.alphabet[j])
                    return SBN3_UNSUPPORTED;
    p.shift = base.odd == 1 ? base.twos : 0;
    p.point = -s.exponent2;
    const int64_t width = int64_t(64) * int64_t(s.limbs);
    p.integer_bits = p.point >= width ? 0 : uint64_t(width - p.point);
    p.fraction_bits = p.point > 0 ? uint64_t(p.point) : 0;
    p.integer_limbs = limbs_for_bits(p.integer_bits);
    auto &info = p.info;
    info = {};
    info.workers = p.options.workers;
    uint64_t integer_digits = 0;
    if (p.integer_bits)
        integer_digits = p.shift ? (p.integer_bits + p.shift - 1) / p.shift
                                 : uint64_t(floorl((long double)p.integer_bits / base.log2_base * (1 + 1e-15L))) + 1;
    p.integer_fragments = (integer_digits + fragment_digits - 1) / fragment_digits;
    info.integer_digits = p.integer_fragments * fragment_digits;
    uint64_t digits = s.fraction_digits;
    if (s.mode == SBN3_RADIX_ENCLOSED) {
        // Digits that an input known to 2^-fraction_bits can determine, keeping a guard of one word.
        uint64_t cap = 0;
        if (p.shift)
            cap = p.fraction_bits / p.shift;
        else if (p.fraction_bits > 2) {
            const long double d = floorl((long double)(p.fraction_bits - 2) / base.log2_base * (1 - 1e-15L));
            cap = d > word_digits ? uint64_t(d) - word_digits : 0;
        }
        digits = std::min(digits, cap);
    }
    info.fraction_digits = digits;
    p.tree_digits = digits;
    if (!p.fraction_bits)
        p.tree_digits = 0;
    else if (base.twos) // a dyadic fraction ends after ceil(bits / twos) digits in an even base
        p.tree_digits = std::min(digits, (p.fraction_bits + base.twos - 1) / base.twos);
    p.fraction_fragments = p.shift ? 0 : (p.tree_digits + fragment_digits - 1) / fragment_digits;
    info.fraction_offset = size_t(info.integer_digits);
    info.digit_bytes = info.fraction_offset + size_t(align_to(digits, fragment_digits));
    if (p.fraction_fragments > max_fragments || p.integer_fragments > max_fragments)
        return SBN3_QUERY_CAPACITY;
    // Schoolbook or tree (TreePolicy): by execute time for repeated conversions, by the whole call for one.
    const auto &policy = tree_policy();
    const bool integer_by_tree = p.options.repeated ? p.integer_limbs >= policy.integer_tree_limbs_repeated
                                 : p.integer_limbs >= (size_t(1) << 20) ||
                                       p.integer_limbs * size_t(fragment_words * p.integer_fragments) >= policy.integer_tree_work;
    p.integer_path = !p.integer_bits || p.shift ? integer_none : integer_by_tree ? integer_tree : integer_schoolbook;
    size_t values = 0, work = 0;
    auto take = [&](size_t limbs) {
        const size_t at = values;
        values += align_to(limbs, 8);
        return at;
    };
    info.lease_peak = 1;
    const bool trees = p.fraction_fragments || p.integer_path == integer_tree;
    unsigned classes = 0;
    if (trees) {
        auto rc = format_tree_begin(s.base, p.options.workers,
                                    std::max(p.integer_path == integer_tree ? p.integer_fragments : 0, p.fraction_fragments), tree);
        tree_begun = true;
        if (rc != SBN3_SUPPORTED)
            return rc;
        tree.transcript = &transcript;
        tree.repeated = p.options.repeated != 0;
        if (p.integer_path == integer_tree) {
            integer_root = tree.add_tree(p.integer_fragments);
            tree.rail.count = std::max(tree.rail.count, top_bit(p.integer_fragments) + 1);
        }
        if (p.fraction_fragments) {
            fraction_root = tree.add_tree(p.fraction_fragments);
            tree.rail.count = std::max(tree.rail.count, top_bit(p.fraction_fragments) + 1);
        }
        if (p.integer_path == integer_tree) {
            const size_t n = tree.root_limbs(integer_root);
            p.root_product = tree.add_product(n, n + 1);
        }
        rc = tree.finish();
        if (rc != SBN3_SUPPORTED)
            return rc;
        classes = tree.class_count;
        p.tree_id = tree_identity(tree);
        p.rail_at = take(tree.rail.total_limbs);
        p.side_at = take(2 * std::max(p.integer_path == integer_tree ? p.integer_fragments : 0, p.fraction_fragments));
        work = tree.rail.setup_bytes;
    }
    if (p.integer_path == integer_tree) {
        p.divide_limbs = tree.root_limbs(integer_root);
        if (reciprocal_is_basecase(p.divide_limbs)) {
            work = std::max(work, reciprocal_basecase_bytes(p.divide_limbs)); // one schoolbook division at bind
        } else if (sealed) {
            // The Newton plan came with the plan value; its resources are sealed.
            divide_info.storage_bytes = sealed->divide_bytes;
            divide_info.storage_alignment = sealed->divide_alignment;
            divide_info.lease_peak = sealed->divide_lease_peak;
        } else {
            sbn3_newton_options o{};
            o.workers = p.options.workers;
            const auto rc = sbn3_newton_query(SBN3_NEWTON_INVERSE, p.divide_limbs, &o, &divide, &divide_info);
            if (rc != SBN3_SUPPORTED)
                return rc;
        }
        p.divide_alignment = divide_info.storage_alignment;
        p.divide_lease_peak = divide_info.lease_peak;
        p.numerator_at = take(p.divide_limbs + 1);
        p.inverse_at = take(p.divide_limbs + 1);
        size_t chain = 0;
        integer_chain = chain_of(tree.base, p.integer_fragments);
        const auto crc = chain_bytes(tree.base, integer_chain, p.options.workers, chain, &transcript);
        if (crc != SBN3_SUPPORTED)
            return crc;
        // The reciprocal is computed inside the work range at bind; execute multiplies there, then runs the tree.
        const auto &product = tree.extra[p.root_product];
        work = std::max({work, chain, tree.work_bytes(integer_root), divide_info.storage_bytes, product.episode_bytes()});
        info.lease_peak += divide_info.lease_peak;
    } else if (p.integer_path == integer_schoolbook) {
        p.small_at = take(p.integer_limbs + 8 * p.integer_fragments);
    }
    if (p.fraction_fragments) {
        p.root_at = take(tree.root_limbs(fraction_root));
        work = std::max(work, tree.work_bytes(fraction_root));
        {
            // Exact tie resolution: odd^(64 F), a copy of the low fraction limbs, and their product.
            auto chain = chain_of(tree.base, p.fraction_fragments);
            size_t bytes = 0;
            auto rc = chain_bytes(tree.base, chain, p.options.workers, bytes, &transcript);
            if (rc != SBN3_SUPPORTED)
                return rc;
            const uint64_t scale = uint64_t(tree.base.twos) * fragment_digits * p.fraction_fragments;
            if (p.fraction_bits > scale) {
                const size_t low = limbs_for_bits(p.fraction_bits - scale + 64);
                const size_t an = std::min(low, limbs_for_bits(p.fraction_bits)), bn = std::min(low, chain.limbs);
                size_t episode = 0, out = an + bn;
                if (std::max(an, bn) > chain_basecase_limbs) {
                    ProductShape shape{};
                    rc = product_shape(an, bn, p.options.workers, shape, nullptr, &transcript);
                    if (rc != SBN3_SUPPORTED)
                        return rc;
                    episode = shape.temporary_bytes();
                    out = shape.output_limbs;
                }
                bytes += align_to(an * 8, 64) + align_to(out * 8, 64) + episode + 256;
            }
            work = std::max(work, bytes);
        }
    }
    // Layout: [control][prepared][values][divide][work], all relative to a base aligned to storage_alignment.
    const size_t prepared_alignment = trees ? std::max<size_t>(4096, tree.prepared_alignment()) : 4096;
    const size_t divide_alignment = p.divide_limbs ? std::max<size_t>(4096, divide_info.storage_alignment) : 4096;
    const size_t pool_alignment = trees && tree.pool_bytes() ? size_t(1) << 21 : 4096;
    info.storage_alignment = std::max({pool_alignment, prepared_alignment, divide_alignment});
    info.control_bytes = align_to(align_to(sizeof(Binding), 64) + size_t(trees ? tree.program_slots() : 0) * sizeof(ProductProgram) +
                                      size_t(classes) * sizeof(RailProduct) + size_t(trees ? tree.ring_stages : 0) * sizeof(RingStage) + 192,
                                  4096);
    p.prepared_offset = align_to(info.control_bytes, prepared_alignment);
    p.prepared_bytes = trees ? align_to(tree.prepared_bytes() + 128, 4096) : 0;
    p.values_offset = align_to(p.prepared_offset + p.prepared_bytes, 4096);
    p.values_bytes = align_to(values * 8, 4096);
    p.divide_offset = 0;
    p.divide_bytes = p.divide_limbs ? divide_info.storage_bytes : 0; // bind-time use of the work range
    // The pool of the ring stages stays unleased: the product service leases inside it stage by stage.
    p.pool_bytes = trees ? align_to(tree.pool_bytes(), 4096) : 0;
    p.pool_offset = align_to(p.values_offset + p.values_bytes, pool_alignment);
    p.work_offset = align_to(p.pool_offset + p.pool_bytes, divide_alignment);
    if (p.pool_bytes)
        info.lease_peak += 1 + 2 * ring_max_groups;
    p.work_bytes = trees ? align_to(work + 64, 4096) : 0;
    info.storage_bytes = p.work_offset + p.work_bytes;
    info.table_bytes = p.prepared_bytes;
    info.value_bytes = p.values_bytes;
    info.workspace_bytes = p.work_bytes;
    info.divide_bytes = p.divide_bytes;
    info.lease_peak += (p.prepared_bytes != 0) + (p.values_bytes != 0) + (p.work_bytes != 0);
    return SBN3_SUPPORTED;
}
// The sealed plan; the Newton plan and the transcript (ready to replay) of the plan value. `divide` and
// `transcript` come in value-initialised.
Plan load(const sbn3_format_plan &opaque, sbn3_newton_plan &divide, PlanTranscript &transcript) noexcept {
    const auto *bytes = reinterpret_cast<const unsigned char *>(opaque.opaque);
    Plan p{};
    memcpy(static_cast<void *>(&p), bytes, sizeof p);
    uint64_t count = 0;
    memcpy(&count, bytes + value_count_at, 8);
    require(p.marker == plan_magic && count <= PlanTranscript::capacity, SBN3_FATAL_ARGUMENT, "format plan");
    if (p.divide_limbs)
        memcpy(&divide, bytes + value_divide_at, sizeof divide);
    transcript.count = uint32_t(count);
    memcpy(transcript.entry, bytes + value_entries_at, size_t(count) * 8);
    require(p.seal == seal(p, divide, transcript), SBN3_FATAL_ARGUMENT, "format plan");
    transcript.replay = true;
    return p;
}
Binding &binding(sbn3_format_binding *opaque) noexcept {
    auto *b = reinterpret_cast<Binding *>(opaque);
    require(b && b->marker == binding_magic, SBN3_FATAL_LIFETIME, "format binding");
    require(pthread_equal(b->team->creator, pthread_self()) && !b->team->busy, SBN3_FATAL_TEAM, "format owner/idle");
    return *b;
}
// ---- execute pieces ----------------------------------------------------------------------------------
template <class F> void for_digits(sbn3_team *team, uint64_t count, F &&fn) noexcept {
    const unsigned parts = count >= (uint64_t(1) << 16) ? sbn3_team_workers(team) : 1;
    parallel_limbs::each(team, size_t(count), parts, [&](size_t b, size_t e, unsigned) { fn(b, e); });
}
void window(sbn3_team *team, uint64_t *out, size_t n, const uint64_t *m, size_t count, int64_t position,
            uint64_t low, uint64_t high) noexcept {
    parallel_limbs::each(team, n, parallel_limbs::parts(team, n), [&](size_t b, size_t e, unsigned) {
        dyadic_window(out + b, e - b, m, count, position + int64_t(64) * int64_t(b), low, high);
    });
}
void power_of_two(Binding &b, const uint64_t *m, unsigned char *digits) noexcept {
    const auto &p = b.plan;
    const unsigned s = p.shift;
    const size_t count = p.spec.limbs;
    const int64_t point = p.point;
    const uint64_t mask = (uint64_t(1) << s) - 1;
    const uint8_t *encode = b.encode;
    const uint64_t area = p.info.integer_digits;
    // integer digit i (weight base^i) sits at byte area - 1 - i; bits outside the mantissa are zero
    for_digits(b.team, area, [&](size_t begin, size_t end) {
        for (size_t at = begin; at < end; ++at)
            digits[at] = encode[bits_at(m, count, point + int64_t((area - 1 - at) * s)) & mask];
    });
    unsigned char *fraction = digits + p.info.fraction_offset;
    for_digits(b.team, p.info.fraction_digits, [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j)
            fraction[j] = encode[bits_at(m, count, point - int64_t((j + 1) * s)) & mask];
    });
}
void schoolbook(Binding &b, const uint64_t *m, unsigned char *digits) noexcept {
    const auto &p = b.plan;
    uint64_t *a = b.limbs(p.small_at), *words = a + p.integer_limbs;
    const size_t total = size_t(8 * p.integer_fragments);
    dyadic_window(a, p.integer_limbs, m, p.spec.limbs, p.point, p.point > 0 ? uint64_t(p.point) : 0,
                  uint64_t(64) * p.spec.limbs);
    const uint64_t b8 = b.tree.digits.b8;
    size_t top = p.integer_limbs, produced = 0;
    while (top && !a[top - 1])
        --top;
    while (top) {
        uint64_t r = 0;
        for (size_t j = top; j-- > 0;)
            a[j] = divide_step(r, a[j], b8, r);
        while (top && !a[top - 1])
            --top;
        require(produced < total, SBN3_FATAL_MATH, "radix integer digit capacity");
        words[total - 1 - produced++] = r;
    }
    for (size_t j = 0; j < total - produced; ++j)
        words[j] = 0;
    emit_words(digits, words, total, b.tree.digits);
}
void integer_tree_run(Binding &b, const uint64_t *m, unsigned char *digits) noexcept {
    const auto &p = b.plan;
    const size_t n = p.divide_limbs;
    uint64_t *numerator = b.limbs(p.numerator_at);
    const unsigned sh = b.denominator_shift;
    // (4 a + 3) << sh, a = the mantissa bits from the point upward
    window(b.team, numerator, n + 1, m, p.spec.limbs, p.point - int64_t(sh + 2), p.point > 0 ? uint64_t(p.point) : 0,
           uint64_t(64) * p.spec.limbs);
    numerator[sh / 64] |= uint64_t(3) << (sh % 64);
    if (sh % 64 == 63)
        numerator[sh / 64 + 1] |= 1;
    // root = floor(numerator * reciprocal / B^n): within four units of B^n (a + 3/4) / b^(64 F)
    {
        const auto &shape = b.tree_plan().extra[p.root_product];
        Frame scratch = Frame::borrow(*b.arena, b.work, b.work.data, b.work.bytes);
        auto *z = scratch.alloc<uint64_t>(shape.output_limbs);
        auto work = scratch.subframe(shape.work_bytes, shape.work_alignment);
        TeamProduct job{&b.tree.extra[p.root_product], &work, {numerator, n}, {b.limbs(p.inverse_at), n + 1},
                        {z, shape.output_limbs}};
        run_product(b.team, job);
        require(!z[2 * n], SBN3_FATAL_MATH, "radix integer root");
        parallel_limbs::copy(b.team, numerator, z + n, n);
    }
    uint64_t *side = b.limbs(p.side_at);
    format_tree_run(b.tree, b.integer_root, numerator, digits, side, side + p.integer_fragments);
}
// The tree left the requested prefix open: every computed digit after it is b - 1. True when the
// prefix must be incremented.
bool resolve(Binding &b, const uint64_t *m, const unsigned char *digits, sbn3_format_result &result) noexcept {
    const auto &p = b.plan;
    const auto &base = b.tree_plan().base;
    const size_t count = p.spec.limbs;
    const uint64_t bits = p.fraction_bits;
    // frac(X) * base^tree_digits is an integer: the computed string is that integer minus one.
    const uint64_t zeros = trailing_zero_bits(m, count, std::min<uint64_t>(bits, uint64_t(64) * count));
    if (zeros + uint64_t(base.twos) * p.tree_digits >= bits)
        return true;
    // Otherwise compare, modulo 2^64, the computed 64 F digit string S with J = floor(f b^(64F)):
    // S is J or J - 1.
    result.exact_fallback = 1;
    const uint64_t fragments = p.fraction_fragments;
    const auto &dp = b.tree.digits;
    uint64_t computed = 0;
    for (uint64_t f = 0; f < fragments; ++f) {
        uint64_t w[8];
        require(parse_words(w, digits + f * fragment_digits, 8, dp), SBN3_FATAL_MATH, "radix digit readback");
        for (unsigned j = 0; j < 8; ++j)
            computed = computed * dp.b8 + w[j];
    }
    const uint64_t scale = uint64_t(base.twos) * fragment_digits * fragments;
    Frame scratch = Frame::borrow(*b.arena, b.work, b.work.data, b.work.bytes);
    const auto chain = chain_of(base, fragments);
    const uint64_t *power = chain_evaluate(chain, b.tree_plan().rail, b.tree.rail, b.team, p.options.workers, scratch);
    uint64_t exact = 0;
    if (bits <= scale) {
        uint64_t low = 0;
        dyadic_window(&low, 1, m, count, 0, 0, bits);
        exact = scale - bits >= 64 ? 0 : (low * power[0]) << (scale - bits);
    } else {
        const uint64_t t = bits - scale;
        const size_t low = limbs_for_bits(t + 64);
        const size_t an = std::min(low, limbs_for_bits(bits)), bn = std::min(low, chain.limbs);
        auto *a = scratch.alloc<uint64_t>(an);
        window(b.team, a, an, m, count, 0, 0, bits);
        if (std::max(an, bn) <= chain_basecase_limbs) {
            auto *out = scratch.alloc<uint64_t>(an + bn);
            sbn3_mul_basecase(out, an + bn, a, an, power, bn);
            exact = bits_at(out, an + bn, int64_t(t));
        } else {
            ProductShape shape{};
            require(product_shape(an, bn, p.options.workers, shape) == SBN3_SUPPORTED, SBN3_FATAL_MATH,
                    "radix exact product");
            auto *out = scratch.alloc<uint64_t>(shape.output_limbs);
            temporary_product(b.team, p.options.workers, scratch, a, an, power, bn, out, shape.output_limbs);
            exact = bits_at(out, an + bn, int64_t(t));
        }
    }
    require(exact == computed || exact == computed + 1, SBN3_FATAL_MATH, "radix exact resolution", exact, computed);
    return exact != computed;
}
void fraction_tree_run(Binding &b, const uint64_t *m, unsigned char *out, sbn3_format_result &result) noexcept {
    const auto &p = b.plan;
    const size_t n = b.tree_plan().root_limbs(b.fraction_root), count = p.spec.limbs;
    const int64_t position = p.point - int64_t(64) * int64_t(n);
    const uint64_t *y = nullptr;
    if (position >= 0 && !(p.point & 63) && uint64_t(p.point) <= uint64_t(64) * count) {
        y = m + size_t(position >> 6); // the fraction's top limbs are the root as they stand
    } else {
        uint64_t *copy = b.limbs(p.root_at);
        window(b.team, copy, n, m, count, position, 0, std::min<uint64_t>(p.fraction_bits, uint64_t(64) * count));
        y = copy;
    }
    uint64_t *side = b.limbs(p.side_at);
    const uint64_t fragments = p.fraction_fragments;
    uint64_t overlap = format_tree_run(b.tree, b.fraction_root, y, out, side, side + fragments);
    const auto &dp = b.tree.digits;
    const uint8_t top = dp.encode[dp.base - 1];
    const uint64_t computed = fragments * fragment_digits;
    // The computed 64 F + 8 digits are the exact ones or one below (in the last place). Only a tail of
    // all b - 1 after the requested prefix leaves the prefix open; settle it exactly.
    bool open = overlap == dp.b8 - 1;
    for (uint64_t j = computed; open && j-- > p.tree_digits;)
        open = out[j] == top;
    if (open && resolve(b, m, out, result)) {
        increment_digits(out, size_t(computed), dp);
        overlap = 0;
    }
    if (p.spec.mode == SBN3_RADIX_EXACT) {
        if (p.tree_digits < p.info.fraction_digits)
            memset(out + p.tree_digits, dp.encode[0], size_t(p.info.fraction_digits - p.tree_digits));
        result.fraction_digits = p.info.fraction_digits;
        return;
    }
    // ENCLOSED: the digits are now those of X itself whenever that matters. The unknown real is below
    // X + 2^-bits, less than a quarter of the last guard digit above: its truncation to digits + 8 places
    // is X's or one more, so the prefix is shared unless the eight guard digits are all b - 1.
    const uint64_t digits = p.info.fraction_digits;
    unsigned char tail[word_digits];
    uint64_t w = overlap;
    for (unsigned j = word_digits; j-- > 0; w /= dp.base)
        tail[j] = dp.encode[w % dp.base];
    auto at = [&](uint64_t j) { return j < computed ? out[j] : tail[j - computed]; };
    bool shared = false;
    for (uint64_t j = digits; !shared && j < digits + word_digits; ++j)
        shared = at(j) != top;
    uint64_t certified = digits;
    if (!shared) {
        while (certified && out[certified - 1] == top)
            --certified;
        certified = certified ? certified - 1 : 0;
    }
    result.fraction_digits = certified;
}
} // namespace
} // namespace sbn::v3::radix
using namespace sbn::v3;
using namespace sbn::v3::radix;
extern "C" void sbn3_radix_alphabet(unsigned base, unsigned char out[64]) {
    require(out, SBN3_FATAL_ARGUMENT, "radix alphabet output");
    static const char lower[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    static const char wide[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/";
    memset(out, 0, 64);
    if (base <= 36)
        memcpy(out, lower, 36);
    else
        memcpy(out, wide, 64);
}
extern "C" sbn3_query_result sbn3_format_query(const sbn3_format_spec *spec, const sbn3_radix_options *options,
                                               sbn3_format_plan *out, sbn3_format_info *info) {
    require(spec && out && info, SBN3_FATAL_ARGUMENT, "format query arguments");
    *info = {};
    Plan p{};
    p.marker = plan_magic;
    p.spec = *spec;
    if (options)
        p.options = *options;
    else
        p.options.workers = 1;
    if (!p.options.use_alphabet)
        memset(p.options.alphabet, 0, sizeof p.options.alphabet);
    // Raw storage: format_tree_begin constructs the tree plan, for plans that have a tree.
    alignas(FormatTreePlan) unsigned char tree_storage[sizeof(FormatTreePlan)];
    auto &tree = *reinterpret_cast<FormatTreePlan *>(tree_storage);
    PlanTranscript transcript{};
    Assembly a{p, tree, transcript};
    const auto rc = a.run();
    *info = p.info;
    if (rc != SBN3_SUPPORTED)
        return rc;
    if (p.options.memory_budget && p.info.storage_bytes > p.options.memory_budget)
        return SBN3_QUERY_CAPACITY;
    // The identity covers the transcript and the Newton plan; info.plan_id is the seal, so it is not part of it.
    p.seal = seal(p, a.divide, transcript);
    p.info.plan_id = info->plan_id = p.seal;
    memset(out, 0, sizeof *out);
    auto *bytes = reinterpret_cast<unsigned char *>(out->opaque);
    memcpy(bytes, static_cast<const void *>(&p), sizeof p);
    if (p.divide_limbs)
        memcpy(bytes + value_divide_at, &a.divide, sizeof a.divide);
    const uint64_t count = transcript.count;
    memcpy(bytes + value_count_at, &count, 8);
    memcpy(bytes + value_entries_at, transcript.entry, size_t(count) * 8);
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_format_bind(const sbn3_format_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                                 sbn3_format_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "format bind arguments");
    sbn3_newton_plan divide{};
    PlanTranscript transcript{};
    const Plan p = load(*opaque, divide, transcript);
    require(team->arena == arena && team->width >= p.options.workers && pthread_equal(team->creator, pthread_self()) &&
                !team->busy,
            SBN3_FATAL_TEAM, "format bind team");
    size_t end = 0;
    require(add_size(offset, p.info.storage_bytes, end) && end <= arena->virtual_bytes && arena->contains(offset, end) &&
                !(reinterpret_cast<uintptr_t>(arena->base + offset) & (p.info.storage_alignment - 1)) &&
                arena->unleased(offset, p.info.storage_bytes),
            SBN3_FATAL_WORKSPACE, "format prepared unleased range");
    sbn3_arena_stats stats{};
    sbn3_arena_get_stats(arena, &stats);
    require(stats.active_leases + p.info.lease_peak <= Arena::max_leases, SBN3_FATAL_WORKSPACE, "format lease capacity");
    auto control = arena->acquire(offset, p.info.control_bytes);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->control = control;
    // The assembly again, product by product from the transcript; its identity must be the sealed one.
    Plan replay = p;
    Assembly a{replay, b->tree_plan(), transcript, &p};
    a.divide = divide;
    require(a.run() == SBN3_SUPPORTED && replay.tree_id == p.tree_id && replay.info.storage_bytes == p.info.storage_bytes &&
                replay.work_bytes == p.work_bytes && replay.prepared_bytes == p.prepared_bytes && replay.pool_bytes == p.pool_bytes &&
                replay.values_bytes == p.values_bytes && replay.info.control_bytes == p.info.control_bytes,
            SBN3_FATAL_MATH, "format bind plan replay");
    b->integer_root = a.integer_root;
    b->fraction_root = a.fraction_root;
    b->divide = divide;
    if (p.prepared_bytes)
        b->prepared = arena->acquire(offset + p.prepared_offset, p.prepared_bytes);
    if (p.values_bytes)
        b->values = arena->acquire(offset + p.values_offset, p.values_bytes);
    if (p.work_bytes)
        b->work = arena->acquire(offset + p.work_offset, p.work_bytes);
    if (p.shift) {
        for (unsigned j = 0; j < 64; ++j)
            b->encode[j] = p.options.use_alphabet ? p.options.alphabet[j] : uint8_t(j);
    } else if (!p.prepared_bytes) {
        // schoolbook only: the digit plan alone
        require(digit_plan_init(b->tree.digits, p.spec.base, p.options.use_alphabet ? p.options.alphabet : nullptr),
                SBN3_FATAL_ARGUMENT, "radix alphabet");
    } else {
        auto *programs = reinterpret_cast<ProductProgram *>(static_cast<uint8_t *>(control.data) +
                                                            align_to(sizeof(Binding), 64));
        auto *cyclic = reinterpret_cast<RailProduct *>(programs + b->tree_plan().program_slots());
        auto *rings = reinterpret_cast<RingStage *>(
            static_cast<uint8_t *>(control.data) +
            align_to(size_t(reinterpret_cast<uint8_t *>(cyclic + b->tree_plan().class_count) - static_cast<uint8_t *>(control.data)), 64));
        format_tree_bind(b->tree, b->tree_plan(), p.options.use_alphabet ? p.options.alphabet : nullptr, *arena, *team,
                         b->prepared, b->work, b->limbs(p.rail_at), programs, cyclic, rings, offset + p.pool_offset);
    }
    if (p.integer_path == integer_tree) {
        // Normalized divisor: odd^(64 F) shifted to the top of divide_limbs limbs; the numerator shift
        // absorbs the power of two of the base.
        const auto &base = b->tree_plan().base;
        const size_t n = p.divide_limbs;
        Frame scratch = Frame::borrow(*arena, b->work, b->work.data, b->work.bytes);
        const auto &chain = a.integer_chain;
        const uint64_t *power = chain_evaluate(chain, b->tree_plan().rail, b->tree.rail, team, p.options.workers, scratch);
        size_t top = chain.limbs;
        while (top && !power[top - 1])
            --top;
        require(top && top <= n, SBN3_FATAL_MATH, "radix divisor size");
        const uint64_t length = uint64_t(64) * (top - 1) + (64 - uint64_t(__builtin_clzll(power[top - 1])));
        const uint64_t z = uint64_t(64) * n - length;
        const uint64_t twos = uint64_t(base.twos) * fragment_digits * p.integer_fragments;
        require(z >= twos + 2 && z - twos - 2 < (uint64_t(1) << 31), SBN3_FATAL_MATH, "radix divisor normalization");
        b->denominator_shift = unsigned(z - twos - 2);
        // The normalized divisor goes through the numerator buffer; its reciprocal stays.
        uint64_t *divisor = b->limbs(p.numerator_at);
        window(team, divisor, n, power, top, -int64_t(z), 0, uint64_t(64) * top);
    }
    if (p.integer_path == integer_tree && reciprocal_is_basecase(p.divide_limbs)) {
        Frame scratch = Frame::borrow(*arena, b->work, b->work.data, b->work.bytes);
        reciprocal_basecase(b->limbs(p.numerator_at), p.divide_limbs, b->limbs(p.inverse_at), scratch);
    } else if (p.integer_path == integer_tree) {
        // Newton INVERSE inside the (still unused) work range: release it for the episode.
        const size_t n = p.divide_limbs;
        arena->release(b->work);
        sbn3_newton_binding *inverse = nullptr;
        sbn3_newton_bind(&b->divide, arena, offset + p.work_offset, team, &inverse);
        const sbn3_newton_inputs in{{}, {b->limbs(p.numerator_at), n}, 0};
        sbn3_newton_execute(inverse, &in, {b->limbs(p.inverse_at), n + 1});
        sbn3_newton_unbind(inverse);
        b->work = arena->acquire(offset + p.work_offset, p.work_bytes);
        b->tree.work = b->work;
    }
    b->marker = binding_magic;
    *out = reinterpret_cast<sbn3_format_binding *>(b);
}
extern "C" void sbn3_format_execute(sbn3_format_binding *opaque, sbn3_const_limbs mantissa, unsigned char *digits,
                                    sbn3_format_result *out) {
    auto &b = binding(opaque);
    const auto &p = b.plan;
    require(out && mantissa.data && mantissa.count == p.spec.limbs && (digits || !p.info.digit_bytes) &&
                !(reinterpret_cast<uintptr_t>(mantissa.data) & 7),
            SBN3_FATAL_ARGUMENT, "format execute arguments");
    const auto *storage = b.arena->base + b.offset;
    require(!overlaps(mantissa.data, mantissa.count * 8, storage, p.info.storage_bytes) &&
                !overlaps(digits, p.info.digit_bytes, storage, p.info.storage_bytes) &&
                !overlaps(digits, p.info.digit_bytes, mantissa.data, mantissa.count * 8),
            SBN3_FATAL_ARGUMENT, "format value overlap");
    sbn3_format_result result{};
    const uint64_t *m = mantissa.data;
    unsigned char *fraction = digits + p.info.fraction_offset;
    if (p.shift) {
        power_of_two(b, m, digits);
        result.fraction_digits = p.info.fraction_digits;
    } else {
        if (p.integer_path == integer_schoolbook)
            schoolbook(b, m, digits);
        else if (p.integer_path == integer_tree)
            integer_tree_run(b, m, digits);
        if (p.fraction_fragments) {
            fraction_tree_run(b, m, fraction, result);
        } else {
            // no fraction bits (EXACT: zeros), or nothing certifiable
            const uint8_t zero = p.options.use_alphabet ? p.options.alphabet[0] : 0;
            if (p.info.fraction_digits)
                memset(fraction, zero, size_t(p.info.fraction_digits));
            result.fraction_digits = p.info.fraction_digits;
        }
    }
    const uint8_t zero = p.options.use_alphabet ? p.options.alphabet[0] : 0;
    uint64_t first = 0;
    while (first < p.info.integer_digits && digits[first] == zero)
        ++first;
    result.integer_first = first;
    *out = result;
}
extern "C" void sbn3_format_unbind(sbn3_format_binding *opaque) {
    auto &b = binding(opaque);
    auto *arena = b.arena;
    const auto control = b.control;
    if (b.plan.work_bytes)
        arena->release(b.work);
    if (b.plan.values_bytes)
        arena->release(b.values);
    if (b.plan.prepared_bytes)
        arena->release(b.prepared);
    b.marker = 0;
    b.~Binding();
    arena->release(control);
}
