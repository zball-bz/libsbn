// Parse service: digits -> dyadic number (include/sbn3/radix.h).
//
//   N = the digit string as an integer (divide-and-conquer evaluation, exact).
//   No fraction digits:  M = N << fraction_bits.
//   Fraction digits Df:  M = floor(N 2^p / b^Df). N times the reciprocal of b^Df (a Newton INVERSE computed
//                        once at bind) gives the quotient with g >= 64 extra bits and an error below 4;
//                        unless those bits sit on a boundary, M is its top part. On a boundary (exactly
//                        representable inputs like 0.5 do that every time) the candidate is settled by
//                        multiplying back.
//   base 2^s:            bit packing.
#include "sbn3/radix.h"
#include "sbn3/newton.h"
#include "radix/parse_tree.hpp"
#include "radix/bits.hpp"
#include "common/identity.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <new>
#include <string.h>
namespace sbn::v3::radix {
namespace {
constexpr uint64_t plan_magic = 0x53424e3352505031ULL, binding_magic = 0x53424e3352504231ULL;
constexpr uint64_t guard_quotient_bits = 64;
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
unsigned top_bit(uint64_t v) noexcept { return 63 - unsigned(__builtin_clzll(v)); }
struct Plan {
    uint64_t marker = 0;
    sbn3_parse_spec spec{};
    sbn3_radix_options options{};
    sbn3_parse_info info{};
    unsigned shift = 0;
    uint64_t digits = 0, fragments = 0;
    size_t number_limbs = 0, divide_limbs = 0;
    uint64_t power_fragments = 0;
    unsigned power_rest = 0;
    size_t prepared_offset = 0, prepared_bytes = 0, values_offset = 0, values_bytes = 0, divide_offset = 0,
           divide_bytes = 0, work_offset = 0, work_bytes = 0;
    size_t rail_at = 0, number_at = 0, denominator_at = 0, inverse_at = 0;
    int quotient_product = -1; // tree.extra index of N * reciprocal
    size_t divide_alignment = 0;   // of the Newton INVERSE storage (fraction digits)
    unsigned divide_lease_peak = 0;
    uint64_t tree_id = 0, seal = 0;
};
// The plan value: the sealed layout, the Newton plan of the reciprocal and the transcript of the product
// searches (see format.cpp): bind assembles the tree plan again without searching again.
struct PlanValue {
    Plan plan{};
    sbn3_newton_plan divide{};
    uint32_t searches = 0;
    uint64_t search[PlanTranscript::capacity]{};
};
static_assert(sizeof(PlanValue) <= sizeof(sbn3_parse_plan));
uint64_t seal(const Plan &p, const sbn3_newton_plan &divide, const PlanTranscript &t) noexcept {
    uint64_t h = identity::fnv_seed;
    const uint64_t fields[] = {p.marker, p.spec.base, p.spec.integer_digits, p.spec.fraction_digits, p.spec.fraction_bits,
                               p.options.workers, p.options.memory_budget, p.options.use_alphabet, p.options.repeated, p.shift, p.digits,
                               p.fragments, p.number_limbs, p.divide_limbs, p.power_fragments, p.power_rest,
                               p.prepared_offset, p.prepared_bytes, p.values_offset, p.values_bytes, p.divide_offset,
                               p.divide_bytes, p.work_offset, p.work_bytes, p.rail_at, p.number_at, p.denominator_at, p.inverse_at,
                               uint64_t(p.quotient_product),
                               p.tree_id, p.info.storage_bytes, p.info.limbs, p.info.control_bytes,
                               p.divide_alignment, p.divide_lease_peak, t.count,
                               p.info.storage_alignment, p.info.lease_peak, p.info.workers, uint64_t(p.info.exponent2)};
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
uint64_t tree_identity(const ParseTreePlan &t) noexcept {
    uint64_t h = identity::fnv_seed;
    h = identity::word(h, t.class_count);
    h = identity::word(h, t.rail.count);
    h = identity::word(h, t.prepared_bytes());
    for (unsigned j = 0; j < t.class_count; ++j) {
        h = identity::word(h, t.classes[j].fragments * 64 + t.classes[j].workers);
        h = identity::word(h, t.classes[j].region_bytes + t.classes[j].persist_bytes + t.classes[j].episode_bytes);
        h = identity::word(h, t.classes[j].product.arithmetic_id);
    }
    return h;
}
struct Binding {
    uint64_t marker = 0;
    Plan plan{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, prepared{}, values{}, work{};
    ParseTreePlan tree_plan{};
    ParseTree tree{};
    int root = -1;
    sbn3_newton_plan divide{};
    uint64_t quotient_shift = 0; // g: M = quotient >> g
    int64_t compare_shift = 0;   // candidate * denominator <=> N << compare_shift
    uint8_t decode[256]{};       // base 2^s
    uint64_t *limbs(size_t at) const noexcept { return static_cast<uint64_t *>(values.data) + at; }
};
struct Assembly {
    Plan &p;
    ParseTreePlan &tree;
    PlanTranscript &transcript; // records (query) or replays (bind) the product searches
    const Plan *sealed = nullptr; // bind: the sealed plan, whose Newton plan is in `divide` already
    int root = -1;
    sbn3_newton_plan divide{};
    sbn3_newton_info divide_info{};
    Chain power_chain{}; // odd^(64 power_fragments) with the winners of its products
    sbn3_query_result run() noexcept;
    sbn3_query_result assemble() noexcept;
};
sbn3_query_result Assembly::run() noexcept {
    const auto rc = assemble();
    tree.transcript = nullptr; // the tree plan outlives the transcript
    return rc;
}
sbn3_query_result Assembly::assemble() noexcept {
    const auto &s = p.spec;
    BaseInfo base{};
    if (!base_info(s.base, base) || !p.options.workers || p.options.workers > 32 || p.options.use_alphabet > 1)
        return SBN3_UNSUPPORTED;
    const uint64_t limit = uint64_t(1) << 40;
    if (s.integer_digits > limit || s.fraction_digits > limit || s.fraction_bits > (uint64_t(1) << 46))
        return SBN3_QUERY_CAPACITY;
    p.digits = s.integer_digits + s.fraction_digits;
    if (!p.digits)
        return SBN3_UNSUPPORTED;
    if (p.options.use_alphabet)
        for (unsigned i = 0; i < s.base; ++i)
            for (unsigned j = 0; j < i; ++j)
                if (p.options.alphabet[i] == p.options.alphabet[j])
                    return SBN3_UNSUPPORTED;
    p.shift = base.odd == 1 ? base.twos : 0;
    auto &info = p.info;
    info = {};
    info.workers = p.options.workers;
    info.exponent2 = -int64_t(s.fraction_bits);
    const uint64_t integer_bits = s.integer_digits ? (p.shift ? s.integer_digits * p.shift
                                                              : power_bits(base.log2_base, s.integer_digits))
                                                   : 0;
    info.limbs = std::max<size_t>(1, limbs_for_bits(integer_bits + s.fraction_bits));
    info.lease_peak = 1;
    size_t values = 0, work = 0;
    auto take = [&](size_t limbs) {
        const size_t at = values;
        values += align_to(limbs, 8);
        return at;
    };
    unsigned classes = 0;
    size_t prepared_alignment = 4096, divide_alignment = 4096;
    if (!p.shift) {
        p.fragments = (p.digits + fragment_digits - 1) / fragment_digits;
        if (p.fragments > max_fragments)
            return SBN3_QUERY_CAPACITY;
        auto rc = parse_tree_begin(s.base, p.options.workers, p.fragments, tree);
        if (rc != SBN3_SUPPORTED)
            return rc;
        tree.transcript = &transcript;
        root = tree.add_tree(p.fragments);
        p.power_fragments = s.fraction_digits / fragment_digits;
        p.power_rest = unsigned(s.fraction_digits % fragment_digits);
        if (p.power_fragments)
            tree.rail.count = std::max(tree.rail.count, top_bit(p.power_fragments) + 1);
        p.number_limbs = integer_limbs(tree.base, p.fragments);
        size_t n = 0, power_limbs = 0;
        if (s.fraction_digits) {
            // Quotient precision: N 2^(a + |d|) / d with a = 64 (n - number_limbs); M takes all but g bits.
            const uint64_t power_low = uint64_t(floorl(base.log2_odd * (long double)s.fraction_digits * (1 - 1e-15L)));
            const uint64_t twos = uint64_t(base.twos) * s.fraction_digits;
            power_limbs = limbs_for_bits(power_bits(base.log2_odd, s.fraction_digits));
            n = std::max<size_t>({4, p.number_limbs, power_limbs});
            const uint64_t have = uint64_t(64) * (n - p.number_limbs) + power_low + twos;
            const uint64_t need = s.fraction_bits + guard_quotient_bits + 2;
            if (have < need)
                n += limbs_for_bits(need - have);
            p.divide_limbs = n;
            p.quotient_product = tree.add_product(p.number_limbs, n + 1);
        }
        rc = tree.finish();
        if (rc != SBN3_SUPPORTED)
            return rc;
        classes = tree.class_count;
        p.tree_id = tree_identity(tree);
        require(p.number_limbs == tree.classes[root].limbs, SBN3_FATAL_MATH, "radix parse root size");
        p.rail_at = take(tree.rail.total_limbs);
        p.number_at = take(p.number_limbs);
        work = std::max(tree.rail.setup_bytes, tree.work_bytes(root));
        prepared_alignment = std::max<size_t>(4096, tree.programs.alignment);
        if (s.fraction_digits) {
            if (reciprocal_is_basecase(n)) {
                work = std::max(work, reciprocal_basecase_bytes(n)); // one schoolbook division at bind
            } else if (sealed) {
                // The Newton plan came with the plan value; its resources are sealed.
                divide_info.storage_bytes = sealed->divide_bytes;
                divide_info.storage_alignment = sealed->divide_alignment;
                divide_info.lease_peak = sealed->divide_lease_peak;
            } else {
                sbn3_newton_options o{};
                o.workers = p.options.workers;
                rc = sbn3_newton_query(SBN3_NEWTON_INVERSE, n, &o, &divide, &divide_info);
                if (rc != SBN3_SUPPORTED)
                    return rc;
            }
            p.divide_alignment = divide_info.storage_alignment;
            p.divide_lease_peak = divide_info.lease_peak;
            divide_alignment = std::max<size_t>(4096, divide_info.storage_alignment);
            p.denominator_at = take(n);
            p.inverse_at = take(n + 1);
            info.lease_peak += divide_info.lease_peak;
            // bind: the power of the odd part and its reciprocal; execute: N * reciprocal; fallback: candidate * denominator
            size_t chain = 0;
            if (p.power_fragments) {
                power_chain = chain_of(base, p.power_fragments);
                rc = chain_bytes(base, power_chain, p.options.workers, chain, &transcript);
                if (rc != SBN3_SUPPORTED)
                    return rc;
            }
            chain += align_to((power_limbs + 16) * 8, 64);
            size_t product = align_to((info.limbs + n) * 8, 64) + 256;
            if (std::max(info.limbs, n) > chain_basecase_limbs) {
                ProductShape shape{};
                rc = product_shape(info.limbs, n, p.options.workers, shape, nullptr, &transcript);
                if (rc != SBN3_SUPPORTED)
                    return rc;
                product = shape.temporary_bytes() + align_to(shape.output_limbs * 8, 64) + 256;
            }
            work = std::max({work, chain, product, divide_info.storage_bytes, tree.extra[p.quotient_product].episode_bytes()});
        }
        info.lease_peak += 3;
    }
    info.storage_alignment = std::max<size_t>({size_t(1) << 21, prepared_alignment, divide_alignment});
    info.control_bytes = align_to(align_to(sizeof(Binding), 64) + size_t(classes) * (sizeof(ProductProgram) + sizeof(RailProduct)) + 64, 4096);
    p.prepared_offset = align_to(info.control_bytes, prepared_alignment);
    p.prepared_bytes = p.shift ? 0 : align_to(tree.prepared_bytes() + 128, 4096);
    p.values_offset = align_to(p.prepared_offset + p.prepared_bytes, 4096);
    p.values_bytes = align_to(values * 8, 4096);
    p.divide_offset = 0;
    p.divide_bytes = p.divide_limbs ? divide_info.storage_bytes : 0; // bind-time use of the work range
    p.work_offset = align_to(p.values_offset + p.values_bytes, std::max<size_t>(size_t(1) << 21, divide_alignment));
    p.work_bytes = p.shift ? 0 : align_to(work + 64, 4096);
    info.storage_bytes = p.work_offset + p.work_bytes;
    info.table_bytes = p.prepared_bytes;
    info.value_bytes = p.values_bytes;
    info.workspace_bytes = p.work_bytes;
    info.divide_bytes = p.divide_bytes;
    return SBN3_SUPPORTED;
}
// The sealed plan; the Newton plan and the transcript (ready to replay) of the plan value.
Plan load(const sbn3_parse_plan &opaque, sbn3_newton_plan &divide, PlanTranscript &transcript) noexcept {
    PlanValue v{};
    memcpy(static_cast<void *>(&v), opaque.opaque, sizeof v);
    const Plan p = v.plan;
    require(p.marker == plan_magic && v.searches <= PlanTranscript::capacity, SBN3_FATAL_ARGUMENT, "parse plan");
    divide = v.divide;
    transcript = {};
    transcript.count = v.searches;
    memcpy(transcript.entry, v.search, size_t(v.searches) * 8);
    require(p.seal == seal(p, divide, transcript), SBN3_FATAL_ARGUMENT, "parse plan");
    transcript.replay = true;
    return p;
}
Binding &binding(sbn3_parse_binding *opaque) noexcept {
    auto *b = reinterpret_cast<Binding *>(opaque);
    require(b && b->marker == binding_magic, SBN3_FATAL_LIFETIME, "parse binding");
    require(pthread_equal(b->team->creator, pthread_self()) && !b->team->busy, SBN3_FATAL_TEAM, "parse owner/idle");
    return *b;
}
void window(sbn3_team *team, uint64_t *out, size_t n, const uint64_t *m, size_t count, int64_t position) noexcept {
    parallel_limbs::each(team, n, parallel_limbs::parts(team, n), [&](size_t b, size_t e, unsigned) {
        dyadic_window(out + b, e - b, m, count, position + int64_t(64) * int64_t(b), 0, uint64_t(64) * count);
    });
}
int64_t floor_div(int64_t a, int64_t b) noexcept { return a >= 0 ? a / b : -((-a + b - 1) / b); }
// Base 2^s: digit j (0 = most significant) holds the bits [pos, pos + s), pos = s (integer - 1 - j) + bits,
// of the result; output limb i collects the digits that reach into [64 i, 64 i + 64). Bits below 0 are
// the truncation.
bool power_of_two(Binding &b, const unsigned char *digits, uint64_t *out) noexcept {
    const auto &p = b.plan;
    const int64_t s = p.shift;
    const int64_t integer = int64_t(p.spec.integer_digits), bits = int64_t(p.spec.fraction_bits);
    const int64_t total = int64_t(p.digits);
    const uint8_t *decode = b.decode;
    bool valid = true;
    const unsigned parts = total >= (int64_t(1) << 16) ? sbn3_team_workers(b.team) : 1;
    parallel_limbs::each(b.team, size_t(total), parts, [&](size_t begin, size_t end, unsigned) {
        uint8_t any = 0;
        for (size_t j = begin; j < end; ++j)
            any |= decode[digits[j]];
        if (any == 0xff) // digit values are below 64: only the invalid marker has the top bits set
            __atomic_store_n(&valid, false, __ATOMIC_RELAXED);
    });
    if (!valid)
        return false;
    parallel_limbs::each(b.team, p.info.limbs, p.info.limbs >= (size_t(1) << 13) ? sbn3_team_workers(b.team) : 1,
                         [&](size_t begin, size_t end, unsigned) {
        for (size_t i = begin; i < end; ++i) {
            const int64_t lo = int64_t(64) * int64_t(i), hi = lo + 64;
            const int64_t first = std::max<int64_t>(integer - 1 - floor_div(hi - bits - 1, s), 0);
            const int64_t last = std::min<int64_t>(integer - 2 - floor_div(lo - bits - s, s), total - 1);
            uint64_t v = 0;
            for (int64_t j = first; j <= last; ++j) {
                const uint64_t d = decode[digits[j]];
                const int64_t pos = s * (integer - 1 - j) + bits - lo;
                v |= pos >= 0 ? d << pos : d >> unsigned(-pos);
            }
            out[i] = v;
        }
    });
    return true;
}
// 0: the low g bits of q are away from a truncation boundary; 1: just above one; 2: just below one.
int boundary(const uint64_t *q, uint64_t g) noexcept {
    const uint64_t first = q[0] >> 2;
    const bool zeros = first == 0, ones = first == (~uint64_t(0) >> 2);
    if (!zeros && !ones)
        return 0;
    const uint64_t want = zeros ? 0 : ~uint64_t(0);
    const size_t full = size_t(g >> 6);
    for (size_t j = 1; j < full; ++j)
        if (q[j] != want)
            return 0;
    const unsigned rest = unsigned(g & 63);
    if (rest && full >= 1 && ((q[full] ^ want) & ((uint64_t(1) << rest) - 1)))
        return 0;
    return zeros ? 1 : 2;
}
// sign of x - (y << shift) for shift >= 0, or of (x << -shift) - y
int compare_shifted(const uint64_t *x, size_t xn, const uint64_t *y, size_t yn, int64_t shift) noexcept {
    const int64_t xs = shift < 0 ? -shift : 0, ys = shift > 0 ? shift : 0;
    const int64_t top = std::max(int64_t(64) * int64_t(xn) + xs, int64_t(64) * int64_t(yn) + ys);
    for (int64_t limb = (top + 63) / 64; limb-- > 0;) {
        const uint64_t a = bits_at(x, xn, int64_t(64) * limb - xs), b = bits_at(y, yn, int64_t(64) * limb - ys);
        if (a != b)
            return a > b ? 1 : -1;
    }
    return 0;
}
uint64_t first_invalid(const unsigned char *digits, uint64_t count, const uint8_t *decode) noexcept {
    for (uint64_t j = 0; j < count; ++j)
        if (decode[digits[j]] == 0xff)
            return j;
    return count;
}
} // namespace
} // namespace sbn::v3::radix
using namespace sbn::v3;
using namespace sbn::v3::radix;
extern "C" sbn3_query_result sbn3_parse_query(const sbn3_parse_spec *spec, const sbn3_radix_options *options,
                                              sbn3_parse_plan *out, sbn3_parse_info *info) {
    require(spec && out && info, SBN3_FATAL_ARGUMENT, "parse query arguments");
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
    ParseTreePlan tree{};
    PlanTranscript transcript{};
    Assembly a{p, tree, transcript};
    const auto rc = a.run();
    *info = p.info;
    if (rc != SBN3_SUPPORTED)
        return rc;
    if (p.options.memory_budget && p.info.storage_bytes > p.options.memory_budget)
        return SBN3_QUERY_CAPACITY;
    p.seal = seal(p, a.divide, transcript);
    p.info.plan_id = info->plan_id = p.seal;
    PlanValue v{};
    v.plan = p;
    if (p.divide_limbs)
        v.divide = a.divide;
    v.searches = transcript.count;
    memcpy(v.search, transcript.entry, size_t(transcript.count) * 8);
    memset(out, 0, sizeof *out);
    memcpy(out->opaque, static_cast<const void *>(&v), sizeof v);
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_parse_bind(const sbn3_parse_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                                sbn3_parse_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "parse bind arguments");
    sbn3_newton_plan divide{};
    PlanTranscript transcript{};
    const Plan p = load(*opaque, divide, transcript);
    require(team->arena == arena && team->width >= p.options.workers && pthread_equal(team->creator, pthread_self()) &&
                !team->busy,
            SBN3_FATAL_TEAM, "parse bind team");
    size_t end = 0;
    require(add_size(offset, p.info.storage_bytes, end) && end <= arena->virtual_bytes && arena->contains(offset, end) &&
                !(reinterpret_cast<uintptr_t>(arena->base + offset) & (p.info.storage_alignment - 1)) &&
                arena->unleased(offset, p.info.storage_bytes),
            SBN3_FATAL_WORKSPACE, "parse prepared unleased range");
    sbn3_arena_stats stats{};
    sbn3_arena_get_stats(arena, &stats);
    require(stats.active_leases + p.info.lease_peak <= Arena::max_leases, SBN3_FATAL_WORKSPACE, "parse lease capacity");
    auto control = arena->acquire(offset, p.info.control_bytes);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->control = control;
    // The assembly again, product by product from the transcript; its identity must be the sealed one.
    Plan replay = p;
    Assembly a{replay, b->tree_plan, transcript, &p};
    a.divide = divide;
    require(a.run() == SBN3_SUPPORTED && replay.tree_id == p.tree_id && replay.info.storage_bytes == p.info.storage_bytes &&
                replay.work_bytes == p.work_bytes && replay.prepared_bytes == p.prepared_bytes &&
                replay.values_bytes == p.values_bytes && replay.info.control_bytes == p.info.control_bytes,
            SBN3_FATAL_MATH, "parse bind plan replay");
    b->root = a.root;
    b->divide = divide;
    if (p.shift) {
        memset(b->decode, 0xff, sizeof b->decode);
        for (unsigned j = 0; j < p.spec.base; ++j)
            b->decode[p.options.use_alphabet ? p.options.alphabet[j] : uint8_t(j)] = uint8_t(j);
        b->marker = binding_magic;
        *out = reinterpret_cast<sbn3_parse_binding *>(b);
        return;
    }
    b->prepared = arena->acquire(offset + p.prepared_offset, p.prepared_bytes);
    b->values = arena->acquire(offset + p.values_offset, p.values_bytes);
    b->work = arena->acquire(offset + p.work_offset, p.work_bytes);
    auto *programs = reinterpret_cast<ProductProgram *>(static_cast<uint8_t *>(control.data) + align_to(sizeof(Binding), 64));
    auto *cached = reinterpret_cast<RailProduct *>(programs + b->tree_plan.class_count);
    parse_tree_bind(b->tree, b->tree_plan, p.options.use_alphabet ? p.options.alphabet : nullptr, *arena, *team,
                    b->prepared, b->work, b->limbs(p.rail_at), programs, cached);
    if (p.divide_limbs) {
        // denominator = odd^fraction_digits, normalized to divide_limbs limbs
        const auto &base = b->tree_plan.base;
        const size_t n = p.divide_limbs;
        Frame scratch = Frame::borrow(*arena, b->work, b->work.data, b->work.bytes);
        const size_t capacity = limbs_for_bits(power_bits(base.log2_odd, p.spec.fraction_digits)) + 8;
        uint64_t small[8]{}, t[16]{};
        small[0] = 1;
        size_t small_limbs = 1;
        for (unsigned j = 0; j < p.power_rest; ++j) {
            uint64_t carry = 0;
            for (size_t i = 0; i < small_limbs; ++i) {
                const __uint128_t x = (__uint128_t)small[i] * base.odd + carry;
                small[i] = uint64_t(x);
                carry = uint64_t(x >> 64);
            }
            if (carry)
                small[small_limbs++] = carry;
        }
        (void)t;
        uint64_t *power = nullptr;
        size_t limbs = 0;
        if (p.power_fragments) {
            const auto &chain = a.power_chain;
            const uint64_t *big = chain_evaluate(chain, b->tree_plan.rail, b->tree.rail, team, p.options.workers, scratch);
            power = scratch.alloc<uint64_t>(capacity + 8);
            limbs = chain.limbs + small_limbs;
            require(limbs <= capacity + 8, SBN3_FATAL_MATH, "radix divisor capacity");
            // (big) x (at most six limbs): one mul_1 pass per limb
            memset(power, 0, (capacity + 8) * 8);
            for (size_t i = 0; i < small_limbs; ++i) {
                uint64_t carry = 0;
                for (size_t j = 0; j < chain.limbs; ++j) {
                    const __uint128_t x = (__uint128_t)big[j] * small[i] + power[i + j] + carry;
                    power[i + j] = uint64_t(x);
                    carry = uint64_t(x >> 64);
                }
                power[i + chain.limbs] = carry;
            }
        } else {
            power = scratch.alloc<uint64_t>(8);
            memcpy(power, small, 64);
            limbs = small_limbs;
        }
        while (limbs && !power[limbs - 1])
            --limbs;
        require(limbs && limbs <= n, SBN3_FATAL_MATH, "radix divisor size");
        const uint64_t length = uint64_t(64) * (limbs - 1) + (64 - uint64_t(__builtin_clzll(power[limbs - 1])));
        const uint64_t z = uint64_t(64) * n - length;
        window(team, b->limbs(p.denominator_at), n, power, limbs, -int64_t(z));
        const uint64_t twos = uint64_t(base.twos) * p.spec.fraction_digits;
        const uint64_t have = uint64_t(64) * (n - p.number_limbs) + length + twos;
        require(have >= p.spec.fraction_bits + guard_quotient_bits, SBN3_FATAL_MATH, "radix quotient guard");
        b->quotient_shift = have - p.spec.fraction_bits;
        // candidate * (d 2^z) <=> N 2^(bits - twos + z)
        b->compare_shift = int64_t(p.spec.fraction_bits) - int64_t(twos) + int64_t(z);
    }
    if (reciprocal_is_basecase(p.divide_limbs)) {
        Frame scratch = Frame::borrow(*arena, b->work, b->work.data, b->work.bytes);
        reciprocal_basecase(b->limbs(p.denominator_at), p.divide_limbs, b->limbs(p.inverse_at), scratch);
    } else if (p.divide_limbs) {
        // Newton INVERSE of the normalized denominator inside the (still unused) work range.
        const size_t n = p.divide_limbs;
        arena->release(b->work);
        sbn3_newton_binding *inverse = nullptr;
        sbn3_newton_bind(&b->divide, arena, offset + p.work_offset, team, &inverse);
        const sbn3_newton_inputs in{{}, {b->limbs(p.denominator_at), n}, 0};
        sbn3_newton_execute(inverse, &in, {b->limbs(p.inverse_at), n + 1});
        sbn3_newton_unbind(inverse);
        b->work = arena->acquire(offset + p.work_offset, p.work_bytes);
        b->tree.work = b->work;
    }
    b->marker = binding_magic;
    *out = reinterpret_cast<sbn3_parse_binding *>(b);
}
extern "C" void sbn3_parse_execute(sbn3_parse_binding *opaque, const unsigned char *digits, sbn3_limbs out,
                                   sbn3_parse_result *result_out) {
    auto &b = binding(opaque);
    const auto &p = b.plan;
    require(digits && result_out && out.data && out.capacity >= p.info.limbs && !(reinterpret_cast<uintptr_t>(out.data) & 63),
            SBN3_FATAL_ARGUMENT, "parse execute arguments");
    const auto *storage = b.arena->base + b.offset;
    require(!overlaps(out.data, p.info.limbs * 8, storage, p.info.storage_bytes) &&
                !overlaps(digits, p.digits, storage, p.info.storage_bytes) &&
                !overlaps(digits, p.digits, out.data, p.info.limbs * 8),
            SBN3_FATAL_ARGUMENT, "parse value overlap");
    sbn3_parse_result result{};
    result.valid = 1;
    if (p.shift) {
        if (!power_of_two(b, digits, out.data)) {
            result.valid = 0;
            result.invalid_index = first_invalid(digits, p.digits, b.decode);
        }
        *result_out = result;
        return;
    }
    const size_t n = p.divide_limbs;
    uint64_t *number = b.limbs(p.number_at);
    if (!parse_tree_run(b.tree, b.root, digits, p.digits, number)) {
        result.valid = 0;
        result.invalid_index = first_invalid(digits, p.digits, b.tree.digits.decode);
        *result_out = result;
        return;
    }
    if (!n) {
        window(b.team, out.data, p.info.limbs, number, p.number_limbs, -int64_t(p.spec.fraction_bits));
        *result_out = result;
        return;
    }
    // q = floor(N * reciprocal / B^number_limbs): within four units of N 2^(bits + g) / b^Df.
    const uint64_t g = b.quotient_shift;
    int state = 0;
    {
        const auto &shape = b.tree_plan.extra[p.quotient_product];
        Frame scratch = Frame::borrow(*b.arena, b.work, b.work.data, b.work.bytes);
        auto *z = scratch.alloc<uint64_t>(shape.output_limbs);
        auto work = scratch.subframe(shape.work_bytes, shape.work_alignment);
        TeamProduct job{&b.tree.extra[p.quotient_product], &work, {number, p.number_limbs}, {b.limbs(p.inverse_at), n + 1},
                        {z, shape.output_limbs}};
        run_product(b.team, job);
        const uint64_t *q = z + p.number_limbs;
        window(b.team, out.data, p.info.limbs, q, n + 1, int64_t(g));
        state = boundary(q, g);
    }
    if (state) {
        // The true quotient is within a few units of candidate * 2^g: decide candidate or candidate - 1 exactly.
        result.exact_fallback = 1;
        if (state == 2)
            parallel_limbs::add_word(out.data, p.info.limbs, 1);
        Frame scratch = Frame::borrow(*b.arena, b.work, b.work.data, b.work.bytes);
        const size_t an = p.info.limbs;
        uint64_t *product = nullptr;
        if (std::max(an, n) <= chain_basecase_limbs) {
            product = scratch.alloc<uint64_t>(an + n);
            sbn3_mul_basecase(product, an + n, out.data, an, b.limbs(p.denominator_at), n);
        } else {
            ProductShape shape{};
            require(product_shape(an, n, p.options.workers, shape) == SBN3_SUPPORTED, SBN3_FATAL_MATH, "radix exact product");
            product = scratch.alloc<uint64_t>(shape.output_limbs);
            temporary_product(b.team, p.options.workers, scratch, out.data, an, b.limbs(p.denominator_at), n, product,
                              shape.output_limbs);
        }
        if (compare_shifted(product, an + n, number, p.number_limbs, b.compare_shift) > 0)
            parallel_limbs::sub_word(out.data, p.info.limbs, 1);
    }
    *result_out = result;
}
extern "C" void sbn3_parse_unbind(sbn3_parse_binding *opaque) {
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
