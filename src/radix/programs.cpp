#include "radix/programs.hpp"
#include "value/parallel_limbs.hpp"
#include "common/identity.hpp"
#include "sbn3/product.h"
#include "sbn3/divrem.h"
#include <algorithm>
#include <string.h>
#include <stdlib.h>
#ifdef SBN3_RADIX_TRACE
#include <stdio.h>
#include <time.h>
#endif
namespace sbn::v3::radix {
namespace {
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
constexpr size_t program_slack = 256; // product_program_prepare may round its objects to 128 bytes
size_t program_bytes(size_t bytes) noexcept { return align_to(bytes, 128) + program_slack; }
} // namespace
TreePolicy &tree_policy() noexcept {
    static TreePolicy policy{};
#ifdef SBN3_RADIX_TRACE // probe builds: policy sweeps without recompiling
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        if (const char *v = getenv("SBN3_RADIX_TEAM_NODE")) policy.team_node_limbs = strtoull(v, nullptr, 10);
        if (const char *v = getenv("SBN3_RADIX_FORK_NODE")) policy.fork_node_limbs = strtoull(v, nullptr, 10);
        if (const char *v = getenv("SBN3_RADIX_WIDE_PRODUCT")) policy.wide_product_limbs = strtoull(v, nullptr, 10);
        if (const char *v = getenv("SBN3_RADIX_CYCLIC")) policy.cyclic_products = atoi(v) != 0;
        if (const char *v = getenv("SBN3_RADIX_TASKS")) policy.frontier_tasks_per_worker = strtoull(v, nullptr, 10);
        if (const char *v = getenv("SBN3_RADIX_FORK_MIN")) policy.fork_min_workers = unsigned(strtoul(v, nullptr, 10));
        if (const char *v = getenv("SBN3_RADIX_RING")) policy.ring_products = atoi(v) != 0;
        if (const char *v = getenv("SBN3_RADIX_INTEGER_TREE")) { // limbs; one conversion: the base 10 equivalent in limb-words
            policy.integer_tree_limbs_repeated = strtoull(v, nullptr, 10);
            policy.integer_tree_work = 5 * policy.integer_tree_limbs_repeated * policy.integer_tree_limbs_repeated / 2;
        }
        if (const char *v = getenv("SBN3_RADIX_RING_NODE")) policy.ring_node_limbs = strtoull(v, nullptr, 10);
        if (const char *v = getenv("SBN3_RADIX_RING_COUNT")) policy.ring_min_count = unsigned(strtoul(v, nullptr, 10));
    }
#endif
    return policy;
}
uint64_t PlanTranscript::check(uint64_t a, uint64_t b, uint64_t c, uint64_t identity) noexcept {
    uint64_t h = identity::fnv_seed;
    for (uint64_t v : {a, b, c, identity})
        h = identity::word(h, v);
    return (h ^ h >> 28) & ((uint64_t(1) << (64 - product_choice_bits)) - 1);
}
sbn3_query_result product_shape(size_t an, size_t bn, unsigned workers, ProductShape &s,
                                ProductProgramPlan *keep, PlanTranscript *transcript) {
    ProductProgramPlan local{};
    auto &p = keep ? *keep : local;
    sbn3_mul_options o{};
    o.workers = workers;
    ProductChoice choice = 0;
    bool replayed = false;
    if (transcript && transcript->replay) {
        // The recorded winner: one backend query. Anything that does not reproduce the recorded request and
        // identity is planned by the search below, like a product without a transcript.
        const uint64_t e = transcript->next();
        choice = e & ((uint64_t(1) << product_choice_bits) - 1);
        replayed = choice && product_program_chosen(an, bn, o, choice, p) == SBN3_SUPPORTED &&
                   PlanTranscript::check(an, bn, workers, p.info.arithmetic_id) == e >> product_choice_bits;
    }
    if (!replayed) {
        p = {};
        const auto rc = product_program_choose(an, bn, o, p, choice);
        if (transcript && transcript->replay)
            ++transcript->searched;
        if (transcript && !transcript->replay)
            transcript->record(rc == SBN3_SUPPORTED ? choice : 0, PlanTranscript::check(an, bn, workers, p.info.arithmetic_id));
        if (rc != SBN3_SUPPORTED)
            return rc;
    }
    s = {};
    s.an = an;
    s.bn = bn;
    s.workers = p.info.workers;
    s.requested = workers;
    s.choice = choice;
    s.prepared_bytes = p.prepared_bytes;
    s.tables = product_program_tables(p);
    s.local_bytes = product_program_local_bytes(p);
    s.work_bytes = p.info.workspace_bytes;
    s.work_alignment = std::max<size_t>(64, p.info.workspace_alignment);
    s.output_limbs = sbn3_mul_output_capacity(&p.info);
    s.arithmetic_id = p.info.arithmetic_id;
    if (p.info.output_alignment > 64 || s.output_limbs < an + bn)
        return SBN3_UNSUPPORTED;
    return SBN3_SUPPORTED;
}
void ProgramSetPlan::account(const ProductShape &s) noexcept {
    bool share = s.tables.bytes != 0;
    if (share) {
        bool known = false;
        for (unsigned j = 0; j < shared_count && !known; ++j)
            known = shared[j] == s.tables;
        if (!known && shared_count == max_shared_tables)
            share = false; // no directory slot: this program keeps private tables
        else if (!known) {
            shared_bytes = align_to(shared_bytes, std::max<size_t>(128, s.tables.alignment));
            alignment = std::max(alignment, s.tables.alignment);
            shared[shared_count] = s.tables;
            shared_offset[shared_count++] = shared_bytes;
            shared_bytes += s.tables.bytes;
        }
    }
    local_bytes += program_bytes(share ? s.local_bytes : s.prepared_bytes);
}
ProgramSetBuilder::ProgramSetBuilder(const ProgramSetPlan &plan, Arena &arena, const sbn3_lease &prepared) noexcept
    : plan_(plan), arena_(arena), lease_(prepared),
      local_(Frame::borrow(arena, prepared, static_cast<uint8_t *>(prepared.data) + align_to(plan.shared_bytes, 128),
                           prepared.bytes - align_to(plan.shared_bytes, 128))) {
    require(prepared.bytes >= plan.bytes() && !(uintptr_t(prepared.data) & (plan.alignment - 1)), SBN3_FATAL_WORKSPACE,
            "radix program storage", plan.bytes(), prepared.bytes);
}
namespace {
// The product plan of a planned shape: its winner's one backend query, the search when that does not give the
// planned identity (or the shape carries no choice).
bool planned_product(const ProductShape &expected, ProductProgramPlan &pp) noexcept {
    sbn3_mul_options o{};
    o.workers = expected.requested;
    auto matches = [&] {
        return pp.info.arithmetic_id == expected.arithmetic_id && pp.info.workspace_bytes == expected.work_bytes &&
               sbn3_mul_output_capacity(&pp.info) == expected.output_limbs && pp.prepared_bytes == expected.prepared_bytes;
    };
    if (expected.choice && product_program_chosen(expected.an, expected.bn, o, expected.choice, pp) == SBN3_SUPPORTED && matches())
        return true;
    pp = {};
    return product_program_query(expected.an, expected.bn, o, pp) == SBN3_SUPPORTED && matches();
}
} // namespace
ProductProgram ProgramSetBuilder::prepare(const ProductShape &expected) noexcept {
    ProductProgramPlan pp{};
    require(planned_product(expected, pp), SBN3_FATAL_MATH, "radix product replay");
    ProductShape shape{};
    shape.tables = product_program_tables(pp);
    const void *shared = nullptr;
    if (shape.tables.bytes)
        for (unsigned s = 0; s < plan_.shared_count; ++s)
            if (plan_.shared[s] == shape.tables) {
                if (!data_[s]) {
                    auto frame = Frame::borrow(arena_, lease_, static_cast<uint8_t *>(lease_.data) + plan_.shared_offset[s],
                                               shape.tables.bytes);
                    const auto *backend = backend_lookup(pp.plan.opaque[1]);
                    data_[s] = backend->program_tables_prepare(pp.plan, frame);
                    require(data_[s] == frame.data(), SBN3_FATAL_MATH, "radix shared table base");
                }
                shared = data_[s];
                break;
            }
    return product_program_prepare(pp, local_, shared);
}
void run_product(sbn3_team *team, TeamProduct &job) noexcept {
    sbn3_team_run(team, [](void *v, sbn3_team_scope *scope) {
        auto &j = *static_cast<TeamProduct *>(v);
        product_program_execute(*j.program, *j.work, scope, j.a, j.b, j.out);
    }, &job);
}
void temporary_product(sbn3_team *team, unsigned workers, Frame &scratch, const uint64_t *a, size_t an,
                       const uint64_t *b, size_t bn, uint64_t *out, size_t out_capacity, ProductChoice choice) noexcept {
#ifdef SBN3_RADIX_TRACE // rail trace
    timespec t0{}, t1{}, t2{};
    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
    ProductProgramPlan plan{};
    ProductShape shape{};
    sbn3_mul_options o{};
    o.workers = workers;
    // The planned winner needs no search; its scratch was sized from the same plan.
    if (choice && product_program_chosen(an, bn, o, choice, plan) == SBN3_SUPPORTED) {
        shape.prepared_bytes = plan.prepared_bytes;
        shape.work_bytes = plan.info.workspace_bytes;
        shape.work_alignment = std::max<size_t>(64, plan.info.workspace_alignment);
        shape.output_limbs = sbn3_mul_output_capacity(&plan.info);
        require(plan.info.output_alignment <= 64 && shape.output_limbs >= an + bn && shape.output_limbs <= out_capacity,
                SBN3_FATAL_MATH, "radix temporary product");
    } else {
        plan = {};
        require(product_shape(an, bn, workers, shape, &plan) == SBN3_SUPPORTED && shape.output_limbs <= out_capacity,
                SBN3_FATAL_MATH, "radix temporary product");
    }
    FrameMark mark(scratch);
    auto tables = scratch.subframe(align_to(shape.prepared_bytes, 128) + 256, 128);
    const auto program = product_program_prepare(plan, tables);
    auto work = scratch.subframe(shape.work_bytes, shape.work_alignment);
#ifdef SBN3_RADIX_TRACE
    clock_gettime(CLOCK_MONOTONIC, &t1);
#endif
    TeamProduct job{&program, &work, {a, an}, {b, bn}, {out, out_capacity}};
    run_product(team, job);
#ifdef SBN3_RADIX_TRACE
    clock_gettime(CLOCK_MONOTONIC, &t2);
    fprintf(stderr, "temporary product %zu x %zu: query+prepare %.3f ms, execute %.3f ms\n", an, bn,
            ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1e6, ((t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec)) / 1e6);
#endif
}
namespace {
// Rail squarings from this size on go through the product service's SQR recipe (one forward transform).
constexpr size_t rail_square_service_limbs = 4096;
struct SquarePlan {
    sbn3_mul_plan plan{};
    sbn3_product_info info{};
    size_t bytes = 0; // [tables][workspace][output], page aligned parts
};
// `choice`: in, the planned winner of this squaring's search (0: search); out, the winner. `searched`: set
// when the search ran.
bool square_plan(size_t limbs, unsigned workers, SquarePlan &out, ProductChoice &choice, bool *searched = nullptr) noexcept {
    sbn3_mul_options o{};
    o.workers = workers;
    if (!choice || square_query_chosen(limbs, o, choice, out.plan, out.info) != SBN3_SUPPORTED) {
        if (searched)
            *searched = true;
        out = {};
        if (square_query_choose(limbs, o, out.plan, out.info, choice) != SBN3_SUPPORTED)
            return false;
    }
    const auto &i = out.info.mul;
    if (i.output_alignment > 64 || sbn3_mul_output_capacity(&i) < 2 * limbs)
        return false;
    const size_t alignment = std::max<size_t>(i.workspace_alignment, 4096);
    out.bytes = align_to(std::max<size_t>(i.table_bytes, 64), 4096) + alignment + align_to(std::max<size_t>(i.workspace_bytes, 64), 4096) +
                align_to(sbn3_mul_output_capacity(&i) * 8, 4096) + 4096;
    return true;
}
// Squarings below the wide band run on at most eight workers, like the tree's products.
unsigned square_workers(unsigned workers, size_t limbs) noexcept {
    return 2 * limbs < tree_policy().wide_product_limbs ? std::min(workers, 8u) : workers;
}
} // namespace
sbn3_query_result rail_finish(const BaseInfo &base, unsigned workers, RailPlan &rail, PlanTranscript *transcript) noexcept {
    if (rail.count > max_rail)
        return SBN3_QUERY_CAPACITY;
    rail.total_limbs = 0;
    rail.setup_bytes = 0;
    for (unsigned k = 0; k < rail.count; ++k) {
        rail.limbs[k] = rail_limbs(base, k);
        rail.offset[k] = rail.total_limbs;
        rail.total_limbs += align_to(rail.limbs[k], 8);
        rail.service[k] = false;
        rail.square_choice[k] = 0;
    }
    for (unsigned k = 0; k + 1 < rail.count; ++k) {
        if (rail.limbs[k] <= rail_basecase_limbs)
            continue;
        const unsigned w = square_workers(workers, rail.limbs[k]);
        if (rail.limbs[k] >= rail_square_service_limbs) {
            // One squaring, one plan: the service's SQR recipe where it exists, else the temporary program below.
            const uint64_t mask = (uint64_t(1) << product_choice_bits) - 1;
            auto check = [&](const SquarePlan &plan) { return PlanTranscript::check(rail.limbs[k], 0, w, plan.info.mul.arithmetic_id); };
            const uint64_t recorded = transcript && transcript->replay ? transcript->next() : 0;
            ProductChoice choice = recorded & mask;
            SquarePlan service{};
            bool searched = false;
            rail.service[k] = square_plan(rail.limbs[k], w, service, choice, &searched);
            if (recorded && !searched && rail.service[k] && check(service) != recorded >> product_choice_bits) {
                choice = 0; // not the recorded plan: search
                service = {};
                rail.service[k] = square_plan(rail.limbs[k], w, service, choice, &searched);
            }
            if (transcript && transcript->replay)
                transcript->searched += searched;
            if (transcript && !transcript->replay)
                transcript->record(rail.service[k] ? choice : 0, check(service));
            if (rail.service[k]) {
                rail.square_choice[k] = choice;
                rail.setup_bytes = std::max(rail.setup_bytes, service.bytes + (size_t(1) << 21));
                continue;
            }
        }
        ProductShape s{};
        const auto rc = product_shape(rail.limbs[k], rail.limbs[k], w, s, nullptr, transcript);
        if (rc != SBN3_SUPPORTED)
            return rc;
        rail.square_choice[k] = s.choice;
        rail.setup_bytes = std::max(rail.setup_bytes, s.temporary_bytes() + 64);
    }
    return SBN3_SUPPORTED;
}
void rail_build(const BaseInfo &base, const RailPlan &rail, unsigned workers, Arena &arena, sbn3_team &team,
                sbn3_lease &scratch_lease, uint64_t *storage, const uint64_t **entries) noexcept {
    for (unsigned k = 0; k < rail.count; ++k)
        entries[k] = storage + rail.offset[k];
    if (!rail.count)
        return;
    // odd^64 by six squarings of a value below 2^6.
    uint64_t a[8]{}, b[16]{};
    a[0] = base.odd;
    size_t n = 1;
    for (unsigned j = 0; j < 6; ++j) {
        sbn3_mul_basecase(b, 2 * n, a, n, a, n);
        n = std::min<size_t>(2 * n, 8);
        memcpy(a, b, n * 8);
    }
    require(rail.limbs[0] <= 8, SBN3_FATAL_MATH, "radix rail seed");
    memcpy(storage + rail.offset[0], a, rail.limbs[0] * 8);
    auto pad = [&](unsigned k, size_t written) { // zero padding up to the capacity
        for (size_t j = written; j < rail.limbs[k]; ++j)
            storage[rail.offset[k] + j] = 0;
    };
    pad(0, rail.limbs[0]);
    for (unsigned k = 0; k + 1 < rail.count; ++k) {
        const size_t m = rail.limbs[k], next = rail.limbs[k + 1];
        const uint64_t *q = entries[k];
        uint64_t *out = storage + rail.offset[k + 1];
        if (m <= rail_basecase_limbs) {
            uint64_t z[2 * rail_basecase_limbs];
            sbn3_mul_basecase(z, 2 * m, q, m, q, m);
            for (size_t j = next; j < 2 * m; ++j)
                require(!z[j], SBN3_FATAL_MATH, "radix rail capacity");
            memcpy(out, z, std::min(next, 2 * m) * 8);
            pad(k + 1, std::min(next, 2 * m));
            continue;
        }
        SquarePlan service{};
        if (rail.service[k]) {
            // The product service binds its own leases: carve them out of the scratch range for this one squaring.
            ProductChoice choice = rail.square_choice[k];
            require(square_plan(m, square_workers(workers, m), service, choice), SBN3_FATAL_MATH, "radix rail squaring plan");
            const auto &i = service.info.mul;
            const uintptr_t base = reinterpret_cast<uintptr_t>(arena.base);
            const size_t scratch_offset = size_t(reinterpret_cast<uintptr_t>(scratch_lease.data) - base);
            const size_t scratch_bytes = scratch_lease.bytes;
            arena.release(scratch_lease);
            size_t at = align_to(base + scratch_offset, 4096) - base;
            auto tables = arena.acquire(at, align_to(std::max<size_t>(i.table_bytes, 64), 4096));
            at += tables.bytes;
            at = align_to(base + at, std::max<size_t>(i.workspace_alignment, 4096)) - base;
            auto work = arena.acquire(at, align_to(std::max<size_t>(i.workspace_bytes, 64), 4096));
            at += work.bytes;
            const size_t capacity = sbn3_mul_output_capacity(&i);
            auto *z = reinterpret_cast<uint64_t *>(base + at);
            require(at + capacity * 8 <= scratch_offset + scratch_bytes, SBN3_FATAL_WORKSPACE, "radix rail squaring scratch");
            sbn3_mul_binding *binding = nullptr;
            sbn3_product_bind(&service.plan, static_cast<sbn3_arena *>(&arena), &tables, &work, &team, nullptr, nullptr, &binding);
            sbn3_product_inputs in{};
            in.a = {q, m};
            sbn3_product_execute(binding, &in, {z, capacity});
            sbn3_mul_unbind(binding);
            arena.release(work);
            arena.release(tables);
            scratch_lease = arena.acquire(scratch_offset, scratch_bytes);
            for (size_t j = next; j < 2 * m; ++j)
                require(!z[j], SBN3_FATAL_MATH, "radix rail capacity");
            parallel_limbs::copy(&team, out, z, std::min(next, 2 * m));
            pad(k + 1, std::min(next, 2 * m));
            continue;
        }
        Frame scratch = Frame::borrow(arena, scratch_lease, scratch_lease.data, scratch_lease.bytes);
        auto *z = scratch.alloc<uint64_t>(2 * m);
        temporary_product(&team, square_workers(workers, m), scratch, q, m, q, m, z, 2 * m, rail.square_choice[k]);
        for (size_t j = next; j < 2 * m; ++j)
            require(!z[j], SBN3_FATAL_MATH, "radix rail capacity");
        parallel_limbs::copy(&team, out, z, std::min(next, 2 * m));
        pad(k + 1, std::min(next, 2 * m));
    }
}
void reciprocal_basecase(const uint64_t *d, size_t n, uint64_t *out, Frame &scratch) noexcept {
    local_inverse(out,d,n,scratch);
}
Chain chain_of(const BaseInfo &b, uint64_t fragments) noexcept {
    Chain c{};
    c.first = unsigned(__builtin_ctzll(fragments));
    uint64_t done = uint64_t(1) << c.first;
    c.limbs = c.widest = rail_limbs(b, c.first);
    for (unsigned k = c.first + 1; k < 64 && (fragments >> k); ++k) {
        if (!((fragments >> k) & 1))
            continue;
        done += uint64_t(1) << k;
        const size_t next = limbs_for_bits(power_bits(b.log2_odd, done * fragment_digits));
        c.step[c.steps++] = {k, c.limbs, next, 0};
        c.widest = std::max(c.widest, c.limbs + rail_limbs(b, k));
        c.limbs = next;
    }
    return c;
}
sbn3_query_result chain_bytes(const BaseInfo &b, Chain &c, unsigned workers, size_t &bytes, PlanTranscript *transcript) noexcept {
    size_t episode = 0;
    for (unsigned j = 0; j < c.steps; ++j) {
        auto &s = c.step[j];
        const size_t bn = rail_limbs(b, s.level);
        if (std::max(s.acc_limbs, bn) <= chain_basecase_limbs)
            continue;
        ProductShape shape{};
        const auto rc = product_shape(s.acc_limbs, bn, workers, shape, nullptr, transcript);
        if (rc != SBN3_SUPPORTED)
            return rc;
        s.choice = shape.choice;
        episode = std::max(episode, shape.temporary_bytes());
    }
    bytes = 2 * align_to((c.widest + 8) * 8, 64) + episode + 256;
    return SBN3_SUPPORTED;
}
const uint64_t *chain_evaluate(const Chain &c, const RailPlan &rail, const uint64_t *const *entries, sbn3_team *team,
                               unsigned workers, Frame &scratch) noexcept {
    auto *x = scratch.alloc<uint64_t>(c.widest + 8), *y = scratch.alloc<uint64_t>(c.widest + 8);
    memset(x, 0, (c.widest + 8) * 8);
    memset(y, 0, (c.widest + 8) * 8);
    memcpy(x, entries[c.first], rail.limbs[c.first] * 8);
    for (unsigned j = 0; j < c.steps; ++j) {
        const auto &s = c.step[j];
        const size_t bn = rail.limbs[s.level];
        if (std::max(s.acc_limbs, bn) <= chain_basecase_limbs)
            sbn3_mul_basecase(y, s.acc_limbs + bn, x, s.acc_limbs, entries[s.level], bn);
        else
            temporary_product(team, workers, scratch, x, s.acc_limbs, entries[s.level], bn, y, c.widest + 8, s.choice);
        for (size_t i = s.out_limbs; i < s.acc_limbs + bn; ++i)
            require(!y[i], SBN3_FATAL_MATH, "radix power capacity");
        std::swap(x, y);
    }
    return x;
}
} // namespace sbn::v3::radix
