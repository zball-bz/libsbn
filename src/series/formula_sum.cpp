#include "sbn3/formula_sum.h"
#include "common/identity.hpp"
#include "runtime/team.hpp"
#include "series/formula_api.hpp"
#include "sbn3/value.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <time.h>

extern "C" uint64_t sbn3i_mul_1(uint64_t *, const uint64_t *, long, uint64_t);
namespace sbn::v3::series {
namespace {
using u128 = __uint128_t;
using i128 = __int128_t;
constexpr uint64_t magic = 0x53424e3353554d31ULL;
struct Record {
    int64_t coefficient = 0;
    sbn3_formula_plan plan{};
    sbn3_formula_info info{};
};
struct Object {
    uint64_t marker = magic, identity = 0;
    unsigned count = 0;
};
struct Plan {
    uint64_t marker = magic, seal = 0, object_id = 0;
    sbn3_formula_sum_info info{};
    size_t control = 0, values = 0, values_bytes = 0, output = 0, pool = 0, drop = 0;
    uint64_t error_low = 0, error_high = 0;
};
static_assert(sizeof(Plan) <= sizeof(sbn3_formula_sum_plan));
struct Binding {
    Plan plan{};
    const Object *object = nullptr;
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, values{};
    bool used = false;
    // Single-component sums finish in the component's own value slot; that
    // component stays bound (its slot leased) until this binding is released.
    sbn3_formula_binding *retained = nullptr;
    sbn3_formula_sum_metrics metrics{};
};
uint64_t now() {timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
size_t up(size_t n, size_t a = 128) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "formula sum alignment");
    return r;
}
size_t record_header() { return up(sizeof(Record), 64); }
size_t stride() { return record_header() + sbn3_formula_object_bytes(); }
Record &record(Object *o, unsigned i) {
    return *reinterpret_cast<Record *>(reinterpret_cast<unsigned char *>(o) + 64 + i * stride());
}
const Record &record(const Object *o, unsigned i) { return record(const_cast<Object *>(o), i); }
void *child_object(Record &r) { return reinterpret_cast<unsigned char *>(&r) + record_header(); }
const void *child_object(const Record &r) { return child_object(const_cast<Record &>(r)); }
uint64_t abs_word(int64_t v) { return v < 0 ? uint64_t(-(v + 1)) + 1 : uint64_t(v); }
bool same(const sbn3_formula_product &a, const sbn3_formula_product &b) {
    if (a.constant_low != b.constant_low || a.constant_high != b.constant_high || a.negative != b.negative ||
        a.alternating != b.alternating || a.factor_count != b.factor_count || a.degree != b.degree)
        return false;
    if (a.factor_count > 6 || a.degree > 4)
        return false;
    for (unsigned j = 0; j < a.factor_count; ++j)
        if (a.factor[j].a != b.factor[j].a || a.factor[j].b != b.factor[j].b ||
            a.factor[j].power != b.factor[j].power)
            return false;
    if (a.degree)
        for (unsigned j = 0; j <= a.degree; ++j)
            if (a.coefficient[j] != b.coefficient[j])
                return false;
    return true;
}
bool same(const sbn3_formula_def &a, const sbn3_formula_def &b) {
    return a.recipe == b.recipe && a.begin == b.begin && same(a.P, b.P) && same(a.Q, b.Q) && same(a.R, b.R) &&
           a.shift == b.shift && a.stride == b.stride && a.explicit_first == b.explicit_first &&
           a.first_t == b.first_t && a.first_d == b.first_d && a.first_u == b.first_u &&
           a.first_t_negative == b.first_t_negative && a.numerator_scale == b.numerator_scale &&
           a.denominator_exponent == b.denominator_exponent;
}
uint64_t object_hash(const Object &o) {
    uint64_t h = identity::word(identity::fnv_seed, o.count);
    for (unsigned j = 0; j < o.count; ++j) {
        const auto &r = record(&o, j);
        h = identity::word(identity::word(h, uint64_t(r.coefficient)), r.info.plan_id);
    }
    return h;
}
uint64_t hash(const Plan &p) {
    uint64_t h = identity::word(identity::fnv_seed, magic);
    for (uint64_t x : {p.object_id,
                       p.info.fractional_limbs,
                       p.info.working_limbs,
                       p.info.output_limbs,
                       p.info.object_bytes,
                       p.info.storage_bytes,
                       p.info.storage_alignment,
                       p.info.component_storage_bytes,
                       p.info.divisor,
                       uint64_t(p.info.components),
                       uint64_t(p.info.workers),
                       p.control,
                       p.values,
                       p.values_bytes,
                       p.output,
                       p.pool,
                       p.drop,
                       p.error_low,
                       p.error_high})
        h = identity::word(h, x);
    return h;
}
Plan load(const sbn3_formula_sum_plan &in) {
    Plan p{};
    std::memcpy(&p, in.opaque, sizeof p);
    require(p.marker == magic && p.seal == hash(p), SBN3_FATAL_ARGUMENT, "formula sum plan identity");
    return p;
}
Binding &get(sbn3_formula_sum_binding *ptr) {
    require(ptr, SBN3_FATAL_ARGUMENT, "formula sum binding");
    auto &b = *reinterpret_cast<Binding *>(ptr);
    require(b.plan.marker == magic && pthread_equal(b.team->creator, pthread_self()) && !b.team->busy,
            SBN3_FATAL_LIFETIME, "formula sum controller");
    return b;
}
uint64_t *at(Binding &b, size_t offset) {
    return reinterpret_cast<uint64_t *>(b.arena->base + b.offset + offset);
}
int compare_low(const sbn3_int &v, size_t words, u128 error, bool complement) {
    for (size_t j = words; j-- > 0;) {
        uint64_t x = j < v.size ? v.data[j] : 0;
        if (complement)
            x = ~x;
        const uint64_t y = j < 2 ? uint64_t(error >> (64 * j)) : 0;
        if (x != y)
            return x < y ? -1 : 1;
    }
    return 0;
}
sbn3_const_limbs execute(Binding &b, sbn3_limbs out) {
    const auto &p = b.plan;
    require(!b.used, SBN3_FATAL_LIFETIME, "formula sum single-use binding");
    b.used = true;
    const size_t w = p.info.working_limbs;
    const bool single = b.object->count == 1;
    // One component: no accumulator exists. The scaled value is divided,
    // certified and moved down inside the component's slot.
    sbn3_int accumulator{single ? nullptr : at(b, p.values), w + 4, 0, 0};
    if (!single)
        std::memset(accumulator.data, 0, (w + 4) * 8);
    for (unsigned j = 0; j < b.object->count; ++j) {
        const auto &r = record(b.object, j);
        sbn3_formula_binding *child = nullptr;
        const uint64_t bind_start=now();
        sbn3_formula_bind(&r.plan, child_object(r), b.arena, b.offset + p.pool, b.team, &child);
        b.metrics.component_bind_ns+=now()-bind_start;
        // The component's value is consumed where it lies: scaled by its
        // coefficient in its own slot and accumulated before the component is
        // released. No second full-precision copy exists.
        const auto value = formula_execute_consumable(child);
        require(value.capacity >= w + 2, SBN3_FATAL_WORKSPACE, "formula sum component slot");
        sbn3_formula_metrics metrics{};sbn3_formula_get_metrics(child,&metrics);
        b.metrics.components.prepare_ns+=metrics.prepare_ns;
        b.metrics.components.series_ns+=metrics.series_ns;
        b.metrics.components.merge_ns+=metrics.merge_ns;
        b.metrics.components.terminal_ns+=metrics.terminal_ns;
        const uint64_t combine_start=now();
        const uint64_t carry = sbn3i_mul_1(value.data, value.data, long(w + 1), abs_word(r.coefficient));
        value.data[w + 1] = carry;
        sbn3_int term{value.data, w + 2, w + 2, unsigned(r.coefficient < 0)};
        sbn3_int_normalize(&term);
        if (single) {
            accumulator = term;
            accumulator.capacity = value.capacity;
            b.retained = child;
            b.metrics.combine_ns+=now()-combine_start;
            break;
        }
        sbn3_int_add(&accumulator, {accumulator.data, accumulator.size, accumulator.negative},
                     {term.data, term.size, term.negative});
        b.metrics.combine_ns+=now()-combine_start;
        sbn3_formula_unbind(child);
    }
    const uint64_t finish_start=now();
    require(!accumulator.negative, SBN3_FATAL_MATH, "formula sum nonnegative result");
    if (p.info.divisor != 1 && accumulator.size) {
        uint64_t remainder = 0;
        for (size_t j = accumulator.size; j-- > 0;) {
            const u128 x = (u128(remainder) << 64) | accumulator.data[j];
            accumulator.data[j] = uint64_t(x / p.info.divisor);
            remainder = uint64_t(x % p.info.divisor);
        }
        sbn3_int_normalize(&accumulator);
    }
    // The result is the accumulator itself, moved down over its dropped words.
    auto *result = single ? accumulator.data : at(b, p.output);
    require(!single || accumulator.capacity >= p.info.output_limbs, SBN3_FATAL_WORKSPACE, "formula sum result slot");
    if (b.object->count) {
        const u128 error = (u128(p.error_high) << 64) | p.error_low;
        const bool zero = accumulator.size <= p.drop;
        // Since the true result is nonnegative, zero only needs the upper
        // boundary check. All other cells need both boundaries separated.
        require((zero || compare_low(accumulator, p.drop, error, false) > 0) &&
                    compare_low(accumulator, p.drop, error, true) >= 0,
                SBN3_FATAL_MATH, "formula sum output guard separation");
        require(accumulator.size <= w + 1 &&
                    (accumulator.size <= w || accumulator.data[w] < (uint64_t(1) << 63)),
                SBN3_FATAL_MATH, "formula sum integer range");
        const size_t kept = zero ? 0 : accumulator.size - p.drop;
        if (kept)
            std::memmove(result, accumulator.data + p.drop, kept * 8);
        std::memset(result + kept, 0, (p.info.output_limbs - kept) * 8);
    }
    if (out.data)
        std::memmove(out.data, result, p.info.output_limbs * 8);
    b.metrics.combine_ns+=now()-finish_start;
    return {result, p.info.output_limbs};
}
} // namespace
} // namespace sbn::v3::series
using namespace sbn::v3;
using namespace sbn::v3::series;
extern "C" size_t sbn3_formula_sum_object_bytes(unsigned count) {
    return count <= SBN3_FORMULA_SUM_MAX_TERMS ? 64 + count * stride() : 0;
}
extern "C" sbn3_query_result sbn3_formula_sum_query(const sbn3_formula_sum_term *terms, unsigned count,
                                                    uint64_t divisor, size_t n,
                                                    const sbn3_formula_options *options, void *object,
                                                    size_t capacity, sbn3_formula_sum_plan *out,
                                                    sbn3_formula_sum_info *info) {
    require(out && info && (!count || terms), SBN3_FATAL_ARGUMENT, "formula sum query arguments");
    *info = {};
    if (!n || n > (size_t(1) << 28) - 3 || !divisor || count > SBN3_FORMULA_SUM_MAX_TERMS) {
        info->rejection = "formula sum precision, divisor or count outside domain";
        return SBN3_UNSUPPORTED;
    }
    unsigned indices[SBN3_FORMULA_SUM_MAX_TERMS]{}, groups = 0;
    i128 coefficients[SBN3_FORMULA_SUM_MAX_TERMS]{};
    for (unsigned i = 0; i < count; ++i)
        if (terms[i].coefficient) {
            unsigned j = 0;
            for (; j < groups; ++j)
                if (same(terms[i].formula, terms[indices[j]].formula))
                    break;
            if (j == groups)
                indices[groups++] = i;
            coefficients[j] += terms[i].coefficient;
        }
    unsigned active = 0;
    u128 sum = 0;
    for (unsigned j = 0; j < groups; ++j)
        if (coefficients[j]) {
            if (coefficients[j] < INT64_MIN || coefficients[j] > INT64_MAX) {
                info->rejection = "combined coefficient exceeds int64";
                return SBN3_UNSUPPORTED;
            }
            indices[active] = indices[j];
            coefficients[active] = coefficients[j];
            sum += abs_word(int64_t(coefficients[active++]));
        }
    Plan p{};
    p.info.fractional_limbs = n;
    p.info.output_limbs = n + 1;
    p.info.components = active;
    p.info.divisor = divisor;
    p.info.object_bytes = sbn3_formula_sum_object_bytes(active);
    p.drop = sum >> 64 ? 3 : 2;
    p.info.working_limbs = n + p.drop;
    *info = p.info;
    if (!object || capacity < p.info.object_bytes)
        return SBN3_QUERY_CAPACITY;
    require(!(uintptr_t(object) & 63), SBN3_FATAL_ARGUMENT, "formula sum object alignment");
    auto component = options ? *options : sbn3_formula_options{{16, 0, 0, 0, 0}, 0, 0, 0};
    p.info.workers = component.series.workers;
    if (!p.info.workers || p.info.workers > 32) {
        info->rejection = "formula sum workers outside 1..32";
        return SBN3_UNSUPPORTED;
    }
    const size_t budget = component.memory_budget;
    component.memory_budget = 0;
    auto *o = ::new (object) Object{};
    o->count = active;
    p.info.storage_alignment = 128;
    FormulaTerminalCache terminal{}; // every component ends in the same division plan
    for (unsigned j = 0; j < active; ++j) {
        auto &r = *::new (&record(o, j)) Record{};
        r.coefficient = int64_t(coefficients[j]);
        const auto rc = formula_query_shared(&terms[indices[j]].formula, p.info.working_limbs, &component,
                                             child_object(r), &r.plan, &r.info, &terminal);
        if (rc != SBN3_SUPPORTED) {
            info->rejection = r.info.rejection;
            return rc;
        }
        p.info.component_storage_bytes = std::max(p.info.component_storage_bytes, r.info.storage_bytes);
        p.info.storage_alignment = std::max(p.info.storage_alignment, r.info.storage_alignment);
    }
    o->identity = object_hash(*o);
    p.object_id = o->identity;
    // Component floor plus its certificate is <2 working units. Dividing
    // the integer sum adds <1 unit; divisor>=1 can only reduce prior error.
    const u128 error = 2 * sum + 2;
    p.error_low = uint64_t(error);
    p.error_high = uint64_t(error >> 64);
    p.control = up(sizeof(Binding));
    p.values = p.output = p.control; // accumulator, then the result in place
    // A single component needs no accumulator: its own slot holds the result.
    p.pool = up(p.values + (active == 1 ? 0 : 8 * (p.info.working_limbs + 4)), p.info.storage_alignment);
    p.values_bytes = p.pool - p.values;
    p.info.storage_bytes = p.pool + p.info.component_storage_bytes;
    p.seal = hash(p);
    p.info.plan_id = p.seal;
    *info = p.info;
    if (budget && (p.info.object_bytes > budget || p.info.storage_bytes > budget - p.info.object_bytes))
        return SBN3_QUERY_CAPACITY;
    std::memset(out, 0, sizeof *out);
    std::memcpy(out->opaque, &p, sizeof p);
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_formula_sum_bind(const sbn3_formula_sum_plan *opaque, const void *object,
                                      sbn3_arena *arena, size_t offset, sbn3_team *team,
                                      sbn3_formula_sum_binding **out) {
    require(opaque && object && arena && team && out, SBN3_FATAL_ARGUMENT, "formula sum bind arguments");
    const auto p = load(*opaque);
    const auto &o = *static_cast<const Object *>(object);
    require(o.marker == magic && o.count == p.info.components && o.identity == p.object_id &&
                object_hash(o) == p.object_id,
            SBN3_FATAL_ARGUMENT, "formula sum object identity");
    require(team->arena == arena && team->width == p.info.workers &&
                pthread_equal(team->creator, pthread_self()) && !team->busy &&
                offset <= arena->virtual_bytes && p.info.storage_bytes <= arena->virtual_bytes - offset &&
                arena->contains(offset, offset + p.info.storage_bytes) &&
                arena->unleased(offset, p.info.storage_bytes) &&
                !(uintptr_t(arena->base + offset) & (p.info.storage_alignment - 1)) &&
                !overlaps(object, p.info.object_bytes, arena->base + offset, p.info.storage_bytes),
            SBN3_FATAL_WORKSPACE, "formula sum prepared region");
    const auto control = arena->acquire(offset, p.control);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->object = &o;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->control = control;
    if (p.values_bytes)
        b->values = arena->acquire(offset + p.values, p.values_bytes);
    *out = reinterpret_cast<sbn3_formula_sum_binding *>(b);
}
extern "C" void sbn3_formula_sum_execute(sbn3_formula_sum_binding *ptr, sbn3_limbs out) {
    auto &b = get(ptr);
    const auto &p = b.plan;
    require(out.data && !(uintptr_t(out.data) & 63) && out.capacity >= p.info.output_limbs &&
                !overlaps(out.data, p.info.output_limbs * 8, b.arena->base + b.offset, p.info.storage_bytes),
            SBN3_FATAL_ARGUMENT, "formula sum output/lifetime");
    (void)execute(b, out);
}
extern "C" sbn3_const_limbs sbn3_formula_sum_execute_inplace(sbn3_formula_sum_binding *ptr) {
    return execute(get(ptr), {});
}
extern "C" void sbn3_formula_sum_get_metrics(const sbn3_formula_sum_binding *ptr,
                                             sbn3_formula_sum_metrics *out) {
    require(out, SBN3_FATAL_ARGUMENT, "formula sum metrics output");
    *out = get(const_cast<sbn3_formula_sum_binding *>(ptr)).metrics;
}
extern "C" void sbn3_formula_sum_unbind(sbn3_formula_sum_binding *ptr) {
    auto &b = get(ptr);
    auto *arena = b.arena;
    const auto control = b.control, values = b.values;
    if (b.retained)
        sbn3_formula_unbind(b.retained);
    b.plan.marker = 0;
    b.~Binding();
    if (values.bytes)
        arena->release(values);
    arena->release(control);
}
