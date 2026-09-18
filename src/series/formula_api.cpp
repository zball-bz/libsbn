#include "sbn3/formula.h"
#include "series/formula_api.hpp"
#include "series/formula_def.hpp"
#include "series/psr.hpp"
#include "series/terminal.hpp"
#include "product/cohort_policy.hpp"
#include "common/identity.hpp"
#include "runtime/team.hpp"
#include "core/x86_64/word.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <time.h>
namespace sbn::v3::series {
namespace {
constexpr uint64_t magic = 0x53424e33464d4c41ULL; // SBN3FMLA
constexpr size_t guard = 2;
struct Plan {
    uint64_t marker = magic, seal = 0;
    sbn3_formula_options options{};
    sbn3_formula_info info{};
    PsrInfo psr{};
    RatioTerminalPlan terminal{};
    uint64_t formula_id = 0, terms = 0, scale = 1;
    int exponent = 0;
    size_t psr_fractional = 0, control = 0, values = 0, slot_limbs = 0, psr_pool = 0, terminal_pool = 0, output = 0;
    size_t psr_allowance = 0; // storage the reduction may fill below the terminal's own peak
    unsigned exact_suffix = 1; // the reduction may plan a fitting remaining range as one exact tree
};
static_assert(sizeof(Plan) <= sizeof(sbn3_formula_plan));
struct Object {
    uint64_t marker = magic;
    DataFormula data{};
    uint64_t seal = 0;
    // Product choices made by query; bind replays them instead of searching again.
    ProductDecisionLog decisions{};
};
struct Binding {
    Plan plan{};
    const Object *object = nullptr;
    PsrBinding psr{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, values{};
    bool used = false, psr_live = false;
    sbn3_formula_metrics metrics{};
};
uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
size_t up(size_t n, size_t a = 128) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "formula layout alignment");
    return r;
}
uint64_t seal(const Plan &p) {
    uint64_t h = identity::word(identity::fnv_seed, magic);
    for (uint64_t v : {p.info.fractional_limbs, p.info.working_limbs, p.info.storage_bytes, p.info.storage_alignment,
                       p.psr.schedule_id, p.terminal.info.plan_id, p.formula_id, p.terms, p.psr_fractional, p.control,
                       p.values, p.slot_limbs, p.psr_pool, p.terminal_pool, p.output, p.psr_allowance,
                       p.options.memory_budget, uint64_t(p.exact_suffix),
                       uint64_t(p.options.series.workers), p.scale, uint64_t(int64_t(p.exponent))})
        h = identity::word(h, v);
    return h;
}
Plan load(const sbn3_formula_plan &in) {
    Plan p{};
    std::memcpy(&p, in.opaque, sizeof p);
    require(p.marker == magic && p.seal == seal(p), SBN3_FATAL_ARGUMENT, "formula plan identity");
    return p;
}
uint64_t decay(const void *ptr, uint64_t a) {
    return static_cast<const DataFormula *>(ptr)->attenuation_bits(a);
}
// Every field must lie in the domain the header declares; nothing is clamped
// into a different definition.
const char *validate(const sbn3_formula_def &in) {
    if (in.recipe < SBN3_SERIES_HYPERDESCENT || in.recipe > SBN3_SERIES_BINARY_BBP)
        return "recipe outside the supported enumeration";
    const sbn3_formula_product *products[]{&in.P, &in.Q, &in.R};
    for (const auto *p : products) {
        if (p->factor_count > 6)
            return "factor_count exceeds 6";
        if (p->negative > 1 || p->alternating > 1)
            return "negative/alternating must be 0 or 1";
        if (p->degree > 4)
            return "polynomial degree exceeds 4";
        for (unsigned i = 0; i < p->factor_count; ++i)
            if (!p->factor[i].power || p->factor[i].power > 16)
                return "factor power outside 1..16";
    }
    if (in.explicit_first > 1 || in.first_t_negative > 1)
        return "explicit_first/first_t_negative must be 0 or 1";
    return nullptr;
}
FormulaDef convert(const sbn3_formula_def &in) {
    FormulaDef d{};
    d.recipe = in.recipe;
    d.begin = in.begin;
    auto product = [](const sbn3_formula_product &p) {
        FactorProduct f{};
        f.constant_low = p.constant_low;
        f.constant_high = p.constant_high;
        f.negative = p.negative != 0;
        f.alternating = p.alternating != 0;
        f.count = p.factor_count;
        for (unsigned i = 0; i < f.count; ++i)
            f.factor[i] = {p.factor[i].a, p.factor[i].b, p.factor[i].power};
        f.degree = p.degree;
        for (unsigned i = 0; i <= f.degree; ++i)
            f.coefficient[i] = p.coefficient[i];
        return f;
    };
    d.P = product(in.P);
    d.Q = product(in.Q);
    d.R = product(in.R);
    d.shift = in.shift;
    d.stride = in.stride;
    d.explicit_first = in.explicit_first != 0;
    d.first = {in.first_t, in.first_d, in.first_u, in.first_t_negative != 0};
    return d;
}
// Policy values are resolved once at query and stored in the plan, so bind
// replays exactly the same specification.
constexpr unsigned policy_leaf_terms = 32;
size_t parallel_block_floor(const Plan &p) { return size_t(16384) * p.options.series.workers; }
constexpr size_t block_margin_per_worker = size_t(2) << 20;
// Precision-sized blocks pay while the workers' serial subtrees run from cache:
// a subtree's workspace is 8-11 values of its output (block/workers), so the
// candidate is offered only while 64 bytes per limb of the block fit the
// last-level cache. Measured on the calibrated machine (64 MiB, 16 workers):
// -6..-12% at 2^19-2^20 limbs, no time gain from 2^22 on, where it only adds
// 10-15% storage. One sixteenth of slack keeps a power-of-two request with
// its guard words inside the regime.
constexpr size_t cached_block_limbs = cohort_policy::llc_bytes / 64 + cohort_policy::llc_bytes / 1024;
PsrSpec specification(const Plan &p, const Object &o, ProductDecisionLog *record = nullptr) {
    PsrSpec s{data_finite_formula(o.data),
              {o.data.def.begin, p.terms},
              p.options.series,
              {&o.data, decay, p.options.minimum_block_terms ? p.options.minimum_block_terms : 128, 0,
               64.0 * double(p.options.minimum_block_limbs)},
              p.psr_fractional,
              p.psr_allowance};
    s.exact_suffix = p.exact_suffix != 0;
    if (record) {
        record->count = 0; // one record per planning attempt
        s.record = record;
    } else
        s.replay = &o.decisions;
    return s;
}
uint64_t *at(Binding &b, size_t offset) {
    return reinterpret_cast<uint64_t *>(b.arena->base + b.offset + offset);
}
Binding &get(sbn3_formula_binding *p) {
    require(p, SBN3_FATAL_ARGUMENT, "formula binding");
    auto &b = *reinterpret_cast<Binding *>(p);
    require(b.plan.marker == magic && pthread_equal(b.team->creator, pthread_self()) && !b.team->busy,
            SBN3_FATAL_LIFETIME, "formula binding owner/state");
    return b;
}
} // namespace
} // namespace sbn::v3::series
using namespace sbn::v3;
using namespace sbn::v3::series;
extern "C" size_t sbn3_formula_object_bytes(void) {
    return (sizeof(Object) + 63) & ~size_t(63);
}
extern "C" sbn3_query_result sbn3_formula_query(const sbn3_formula_def *def, size_t n, const sbn3_formula_options *options,
                                                void *storage, sbn3_formula_plan *out, sbn3_formula_info *info) {
    return formula_query_shared(def, n, options, storage, out, info, nullptr);
}
sbn3_query_result sbn::v3::series::formula_query_shared(const sbn3_formula_def *def, size_t n,
                                                        const sbn3_formula_options *options, void *storage,
                                                        sbn3_formula_plan *out, sbn3_formula_info *info,
                                                        FormulaTerminalCache *shared) noexcept {
    require(def && storage && out && info && !(uintptr_t(storage) & 63), SBN3_FATAL_ARGUMENT, "formula query arguments");
    *info = {};
    if (const char *why = validate(*def)) {
        info->rejection = why;
        return SBN3_UNSUPPORTED;
    }
    if (!n || n > (size_t(1) << 28) || def->numerator_scale > (uint64_t(1) << 62) ||
        def->denominator_exponent < -1024 || def->denominator_exponent > 1024) {
        info->rejection = "n, numerator_scale or denominator_exponent outside the supported range";
        return SBN3_UNSUPPORTED;
    }
    // The opaque plan is compared byte for byte by callers: construct it in
    // zeroed storage so padding holes (unsigned/int members) are deterministic.
    alignas(Plan) unsigned char raw[sizeof(Plan)];
    std::memset(raw, 0, sizeof raw);
    Plan &p = *::new (raw) Plan{};
    p.options = options ? *options : sbn3_formula_options{{16, 0, 0, 0, 0}, 0, 0, 0};
    // Leaf batches below a few dozen terms spend more in node bookkeeping than
    // in arithmetic; measured flat between 24 and 64 for word and wide leaves.
    if (!p.options.series.leaf_terms)
        p.options.series.leaf_terms = policy_leaf_terms;
    p.info.fractional_limbs = n;
    p.info.output_limbs = n + 1;
    p.info.workers = p.options.series.workers;
    p.scale = def->numerator_scale ? def->numerator_scale : 1;
    p.exponent = def->denominator_exponent;
    // The terminal fixes the actual working precision (at least 4 limbs);
    // the reduction is planned one limb finer so its error is below one
    // terminal unit. Its error units are patched once the reduction is known.
    sbn3_query_result rc = SBN3_SUPPORTED;
    if (shared && shared->valid && shared->fractional_limbs == n && shared->workers == p.info.workers)
        p.terminal = shared->plan; // identical to a fresh query: the terminal does not depend on the formula
    else {
        rc = ratio_terminal_query(n, guard, p.info.workers, 0, p.terminal, true);
        if (rc == SBN3_SUPPORTED && shared) {
            shared->plan = p.terminal; // before the formula-specific error units are patched in
            shared->fractional_limbs = n;
            shared->workers = p.info.workers;
            shared->valid = true;
        }
    }
    if (rc != SBN3_SUPPORTED)
        return rc;
    p.info.working_limbs = p.terminal.working;
    p.psr_fractional = p.terminal.working + 1;
    // Finite analysis on a probe domain gives the attenuation function; the
    // structural tail certificate covers everything beyond the chosen N.
    auto *object = ::new (storage) Object{};
    const FormulaDef converted = convert(*def);
    const uint64_t probe = std::min<uint64_t>(uint64_t(1) << 40, formula_domain_limit(converted));
    if (probe <= converted.begin + 1) {
        info->rejection = "numerator polynomial leaves no evaluation domain";
        return SBN3_UNSUPPORTED;
    }
    // The probe only searches the stopping index: structure, the structural
    // tail verdict and the cumulative attenuation integral. Its domain is huge
    // and almost never used, so the contraction certificate (a table over the
    // whole domain) is built once, on the certified prefix.
    if (!data_formula_prepare(converted, probe, object->data, true)) {
        info->rejection = object->data.rejection;
        return SBN3_UNSUPPORTED;
    }
    // Structural verdict first (it names the mathematical reason), then the
    // finite contraction certificate from the first index.
    if (object->data.tail_rejection) {
        info->rejection = object->data.tail_rejection;
        return SBN3_UNSUPPORTED;
    }
    const uint64_t bits_needed = 64 * uint64_t(p.psr_fractional) + 8;
    p.terms = object->data.infinite_tail_terms(bits_needed);
    if (!p.terms) {
        info->rejection = "the infinite tail cannot be certified below the requested precision within the probe domain (2^40 terms, or the numerator polynomial's evaluation limit)";
        return SBN3_UNSUPPORTED;
    }
    if (!data_formula_prepare(converted, p.terms, object->data)) {
        info->rejection = object->data.rejection;
        return SBN3_UNSUPPORTED;
    }
    if (!object->data.contraction) {
        info->rejection = "no attenuation certificate: |R(k)/Q(k)| <= 1 could not be established from the first index";
        return SBN3_UNSUPPORTED;
    }
    // The prefix object bounds the same tail at least as tightly as the probe did.
    if (!(object->data.infinite_tail_bits(p.terms) >= (long double)bits_needed)) {
        info->rejection = "the infinite tail certificate was not reproduced on the certified prefix";
        return SBN3_UNSUPPORTED;
    }
    // Local block value under contraction: |T/D| <= sum_k |P(k)/Q(k)| 2^-(att(k)-att(a)).
    // With at least one attenuation bit per term the sum is below twice its
    // first term; otherwise count the terms. The outer scale must keep the
    // terminal's |numerator/denominator| < 2^63 as well.
    unsigned term_bits = 0;
    for (uint64_t x = p.terms; x; x >>= 1)
        ++term_bits;
    const unsigned growth = object->data.attenuation_minimum >= 1 ? 1 : term_bits;
    // The terminal needs the scaled value below 2^63: use the real-valued
    // series bound (sum of |P/Q| attenuated), not the per-term bit count.
    const long double magnitude_bits = (long double)object->data.ratio_bits + growth + 1;
    const long double scaled_log2 = object->data.value_log2_bound() + log2l((long double)p.scale) - (long double)p.exponent;
    if (magnitude_bits > 62 || !(scaled_log2 < 63)) {
        info->rejection = "local |T/D| (after the outer scale) cannot be certified below 2^63";
        return SBN3_UNSUPPORTED;
    }
    p.formula_id = object->data.formula_id;
    p.info.terms = p.terms;
    p.info.leaf_words = object->data.leaf_words;
    p.info.contraction = 1;
    p.info.attenuation_bits = object->data.attenuation_bits(p.terms);
    // Value lifetime: the reduced pair lives in two slots of this binding; the
    // reduction's region is released before the terminal starts at the same
    // offset, normalizes both slots in place and writes the quotient over the
    // numerator. The reduction may therefore fill exactly what the terminal
    // needs anyway without raising the peak (or what the budget leaves).
    p.psr_allowance = p.terminal.storage_bytes;
    if (p.options.memory_budget) {
        // Upper bound of the shared start: control, two slots of at most
        // psr_fractional + 6 limbs rounded to 2 MiB, 2 MiB alignment.
        const size_t big = size_t(2) << 20;
        const size_t start = up(up(up(sizeof(Binding)), big) + 2 * (up(8 * (p.psr_fractional + 6), big) + 64), big);
        p.psr_allowance = p.options.memory_budget > start ? std::min(p.psr_allowance, p.options.memory_budget - start) : 0;
    }
    // Block policy. Every outer merge works at the remaining precision, so a
    // block carrying less than one full precision of exact growth buys little
    // for its merge, its planning and its tables (nodes inside a block are
    // limited individually, so a large block is not a full-precision tree).
    // Candidates, largest first: blocks of at least the working precision
    // (only within the cache regime, see cached_block_limbs), half of it, and
    // the parallel-efficiency floor 16384*workers. The first
    // plan whose complete binding stays within the block-structure margin of
    // the storage no plan can avoid (value slots plus terminal) is kept; the
    // floor is kept regardless unless an earlier rejected plan is smaller.
    // The margin is one eighth of that storage, but not less than 2 MiB per
    // worker: what larger blocks cost is the concurrent per-worker workspace
    // of the finite tree, which does not shrink with the terminal. (Measured
    // after the root tables shrank the terminal by 11%: a purely relative
    // margin then flipped Log(2) at 2^20 limbs to half-precision blocks, +8%
    // time for -6% storage; above 2^21 limbs larger blocks buy no time and
    // the relative part governs.) The margin is for the block structure only:
    // merge batches are widened up to the terminal's storage, not beyond
    // (spending the margin there cost 4-12% RSS for 1-3% time). An explicit
    // minimum_block_limbs fixes the blocks; a memory budget replaces the
    // margin. At most three planning passes, query time only.
    const size_t strict_allowance = p.psr_allowance;
    size_t candidates[3]{p.options.minimum_block_limbs, 0, 0};
    unsigned candidate_count = 1;
    if (!p.options.minimum_block_limbs) {
        candidate_count = 0;
        for (size_t floor_limbs : {p.psr_fractional, p.psr_fractional / 2, size_t(0)}) {
            if (floor_limbs == p.psr_fractional && floor_limbs > cached_block_limbs)
                continue; // beyond the cache regime the half-precision block is the largest candidate
            const size_t limbs = std::max(parallel_block_floor(p), floor_limbs);
            if (!candidate_count || candidates[candidate_count - 1] != limbs)
                candidates[candidate_count++] = limbs;
        }
    }
    // The exact-suffix rule comes first: a series whose exact tree fits its
    // own demand (every Hyperdescent prefix stopped by its tail bound) is one
    // exact block whatever the block floor, so its first attempt is final when
    // it fits. When it does not fit the margin or the budget, the same floors
    // are tried again as limited chains, whose blocks have a smaller peak.
    rc = SBN3_UNSUPPORTED;
    bool have = false;
    size_t best_limbs = 0, best_allowance = 0;
    unsigned best_exact = 1;
    PsrInfo best{};
    p.exact_suffix = 1;
    for (unsigned j = 0; j < candidate_count;) {
        p.options.minimum_block_limbs = candidates[j];
        p.psr_allowance = strict_allowance;
        PsrInfo attempt{};
        const auto status = psr_query(specification(p, *object, &object->decisions), attempt);
        if (status != SBN3_SUPPORTED) {
            if (!have)
                rc = status;
            ++j;
            continue;
        }
        const bool whole = p.exact_suffix && attempt.blocks == 1 && attempt.exact_blocks == 1;
        // The retry exists for chains, whose blocks have a smaller peak. A retry that is still one
        // block is the same tree (its limits never bind), not an alternative to the exact plan.
        if (!p.exact_suffix && attempt.blocks == 1 && have) {
            object->decisions.count = 0;
            ++j;
            continue;
        }
        const size_t alignment = std::max<size_t>({128, attempt.storage_alignment, p.terminal.storage_alignment});
        const size_t begin = up(up(up(sizeof(Binding)), alignment) + 2 * 8 * (attempt.output_limbs + 8), alignment);
        const size_t unavoidable = begin + p.terminal.storage_bytes;
        const size_t total = begin + std::max(attempt.storage_bytes, p.terminal.storage_bytes);
        const size_t margin = std::max(unavoidable / 8, block_margin_per_worker * p.options.series.workers);
        const bool fits = p.options.memory_budget ? total <= p.options.memory_budget : total - unavoidable <= margin;
        if (fits || !have || attempt.storage_bytes < best.storage_bytes) {
            best = attempt; // the decision record now describes this attempt
            best_limbs = p.options.minimum_block_limbs;
            best_allowance = p.psr_allowance;
            best_exact = p.exact_suffix;
            have = true;
            rc = SBN3_SUPPORTED;
            if (fits)
                break;
        } else
            object->decisions.count = 0; // the record belongs to an attempt that was not kept
        if (whole)
            p.exact_suffix = 0; // the same floor again, as a limited chain
        else
            ++j;
    }
    if (have) {
        p.options.minimum_block_limbs = best_limbs;
        p.psr_allowance = best_allowance;
        p.exact_suffix = best_exact;
        p.psr = best;
    }
    if (rc != SBN3_SUPPORTED)
        return rc;
    // Reduction error (units of 2^-(64 psr_fractional), one limb below the
    // terminal's unit) is amplified by the outer scale; the tail is below one unit.
    const long double amplified = ldexpl((long double)p.psr.error_units * (long double)p.scale, -p.exponent - 64) + 2;
    if (!(amplified < 1.0e12L)) {
        info->rejection = "the outer scale amplifies the reduction error beyond the terminal certificate";
        return SBN3_UNSUPPORTED;
    }
    p.terminal.error_units = 6 + uint64_t(ceill(amplified));
    p.info.storage_alignment = std::max<size_t>({128, p.psr.storage_alignment, p.terminal.storage_alignment});
    p.control = up(sizeof(Binding));
    p.values = up(p.control, p.info.storage_alignment);
    // Spare words for the numerator's scale carry; a multiple of eight keeps
    // the denominator slot 64-byte aligned for the consuming terminal.
    p.slot_limbs = p.psr.output_limbs + 8;
    require(p.slot_limbs >= ratio_terminal_numerator_limbs(p.terminal) + 1, SBN3_FATAL_MATH, "formula value slots");
    p.psr_pool = p.terminal_pool = up(p.values + 2 * 8 * p.slot_limbs, p.info.storage_alignment);
    p.output = p.values; // the quotient replaces the numerator slot
    p.info.storage_bytes = p.psr_pool + std::max(p.psr.storage_bytes, p.terminal.storage_bytes);
    p.info.psr_storage_bytes = p.psr.storage_bytes;
    p.info.terminal_storage_bytes = p.terminal.storage_bytes;
    p.info.blocks = p.psr.blocks;
    p.info.exact_blocks = p.psr.exact_blocks;
    p.seal = seal(p);
    p.info.plan_id = p.seal;
    object->seal = p.seal;
    *info = p.info;
    if (p.options.memory_budget && p.info.storage_bytes > p.options.memory_budget)
        return SBN3_QUERY_CAPACITY;
    std::memset(out, 0, sizeof *out);
    std::memcpy(out->opaque, &p, sizeof p);
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_formula_bind(const sbn3_formula_plan *opaque, const void *storage, sbn3_arena *arena, size_t offset,
                                  sbn3_team *team, sbn3_formula_binding **out) {
    require(opaque && storage && arena && team && out, SBN3_FATAL_ARGUMENT, "formula bind arguments");
    const auto p = load(*opaque);
    const auto &object = *static_cast<const Object *>(storage);
    require(object.marker == magic && object.seal == p.seal && !object.data.rejection, SBN3_FATAL_ARGUMENT,
            "formula object identity");
    require(team->arena == arena && team->width == p.info.workers && !team->busy &&
                pthread_equal(team->creator, pthread_self()) && offset <= arena->virtual_bytes &&
                p.info.storage_bytes <= arena->virtual_bytes - offset &&
                arena->contains(offset, offset + p.info.storage_bytes) && arena->unleased(offset, p.info.storage_bytes) &&
                !(uintptr_t(arena->base + offset) & (p.info.storage_alignment - 1)),
            SBN3_FATAL_WORKSPACE, "formula prepared region");
    auto control = arena->acquire(offset, p.control);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->object = &object;
    b->control = control;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->values = arena->acquire(offset + p.values, 2 * 8 * p.slot_limbs);
    psr_prepare(specification(p, object), p.psr, *arena, *team, offset + p.psr_pool, b->psr);
    b->psr_live = true;
    *out = reinterpret_cast<sbn3_formula_binding *>(b);
}
static sbn3_const_limbs execute_formula(Binding &b, sbn3_limbs out, RatioOutput mode = RatioOutput::Certified) {
    const auto &p = b.plan;
    require(!b.used, SBN3_FATAL_LIFETIME, "formula single-use binding");
    b.used = true;
    // psr_execute wants the pair as adjacent halves (N first, D second) with
    // output_limbs spacing inside the values lease.
    sbn3_series_values reduced{};
    reduced.value[0].mantissa = {at(b, p.values), p.psr.output_limbs, 0, 0};
    reduced.value[1].mantissa = {at(b, p.values) + p.psr.output_limbs, p.psr.output_limbs, 0, 0};
    psr_execute(b.psr, reduced);
    b.metrics.prepare_ns=b.psr.metrics.prepare_ns;
    b.metrics.series_ns=b.psr.metrics.finite_ns;
    b.metrics.merge_ns=b.psr.metrics.merge_ns;
    psr_release(b.psr);
    b.psr_live = false;
    const uint64_t terminal_start=now();
    auto &t = reduced.value[0], &d = reduced.value[1];
    require(d.mantissa.size && !d.mantissa.negative, SBN3_FATAL_MATH, "formula denominator");
    // Move D up to its own aligned slot so the numerator gains spare words
    // for the outer scale's carry; the values lease covers both slots.
    std::memmove(at(b, p.values) + p.slot_limbs, d.mantissa.data, d.mantissa.size * 8);
    d.mantissa.data = at(b, p.values) + p.slot_limbs;
    t.mantissa.capacity = d.mantissa.capacity = p.slot_limbs;
    if (p.scale != 1 && t.mantissa.size) {
        const uint64_t carry = sbn3i_mul_1(t.mantissa.data, t.mantissa.data, long(t.mantissa.size), p.scale);
        if (carry)
            t.mantissa.data[t.mantissa.size++] = carry;
    }
    require(!__builtin_add_overflow(d.exponent2, int64_t(p.exponent), &d.exponent2), SBN3_FATAL_SIZE,
            "formula denominator exponent");
    const auto result = ratio_terminal_execute_consuming(p.terminal, t, d, b.arena, b.offset + p.terminal_pool,
                                                         b.team, mode);
    require(result.data == at(b, p.output) && result.count == p.info.output_limbs, SBN3_FATAL_MATH,
            "formula result slot");
    if (out.data)
        std::memmove(out.data, result.data, result.count * 8);
    b.metrics.terminal_ns=now()-terminal_start;
    return result;
}
namespace sbn::v3::series {
sbn3_const_limbs formula_execute_approximate(sbn3_formula_binding *ptr) noexcept {
    auto &b = get(ptr);
    // The formula service retains >=2 guard words and bounds its input error
    // by <2^40 working units. After dropping the guards and quantizing to the
    // requested fractional width, the complete absolute error is <2 units.
    require(b.plan.terminal.dropped_words >= 2 && b.plan.terminal.error_units < (uint64_t(1) << 40),
            SBN3_FATAL_MATH, "formula working-value error contract");
    return execute_formula(b, {}, RatioOutput::Approximate);
}
sbn3_limbs formula_execute_consumable(sbn3_formula_binding *ptr) noexcept {
    const auto value = formula_execute_approximate(ptr);
    auto &b = *reinterpret_cast<Binding *>(ptr);
    // The result sits at the base of this binding's numerator slot.
    require(value.data == at(b, b.plan.output), SBN3_FATAL_MATH, "formula consumable slot");
    return {at(b, b.plan.output), b.plan.slot_limbs};
}
} // namespace sbn::v3::series
extern "C" void sbn3_formula_execute(sbn3_formula_binding *ptr, sbn3_limbs out) {
    auto &b = get(ptr);
    const auto &p = b.plan;
    require(out.data && !(uintptr_t(out.data) & 63) && out.capacity >= p.info.output_limbs &&
                !overlaps(out.data, p.info.output_limbs * 8, b.arena->base + b.offset, p.info.storage_bytes),
            SBN3_FATAL_ARGUMENT, "formula output/lifetime");
    (void)execute_formula(b, out);
}
extern "C" sbn3_const_limbs sbn3_formula_execute_inplace(sbn3_formula_binding *ptr) {
    return execute_formula(get(ptr), {});
}
extern "C" void sbn3_formula_get_metrics(const sbn3_formula_binding *ptr, sbn3_formula_metrics *out) {
    require(out, SBN3_FATAL_ARGUMENT, "formula metrics output");
    *out = get(const_cast<sbn3_formula_binding *>(ptr)).metrics;
}
extern "C" void sbn3_formula_unbind(sbn3_formula_binding *ptr) {
    auto &b = get(ptr);
    if (b.psr_live)
        psr_release(b.psr);
    auto *arena = b.arena;
    auto values = b.values, control = b.control;
    b.plan.marker = 0;
    b.~Binding();
    arena->release(values);
    arena->release(control);
}
