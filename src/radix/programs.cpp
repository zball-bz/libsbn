#include "radix/programs.hpp"
#include "value/parallel_limbs.hpp"
#include "sbn3/product.h"
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
        if (const char *v = getenv("SBN3_RADIX_INTEGER_TREE")) policy.integer_tree_limbs = policy.integer_tree_limbs_repeated = strtoull(v, nullptr, 10);
        if (const char *v = getenv("SBN3_RADIX_RING_NODE")) policy.ring_node_limbs = strtoull(v, nullptr, 10);
        if (const char *v = getenv("SBN3_RADIX_RING_COUNT")) policy.ring_min_count = unsigned(strtoul(v, nullptr, 10));
    }
#endif
    return policy;
}
sbn3_query_result product_shape(size_t an, size_t bn, unsigned workers, ProductShape &s,
                                ProductProgramPlan *keep) {
    ProductProgramPlan local{};
    auto &p = keep ? *keep : local;
    sbn3_mul_options o{};
    o.workers = workers;
    const auto rc = product_program_query(an, bn, o, p);
    if (rc != SBN3_SUPPORTED)
        return rc;
    s = {};
    s.an = an;
    s.bn = bn;
    s.workers = p.info.workers;
    s.requested = workers;
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
ProductProgram ProgramSetBuilder::prepare(const ProductShape &expected) noexcept {
    ProductProgramPlan pp{};
    ProductShape shape{};
    require(product_shape(expected.an, expected.bn, expected.requested, shape, &pp) == SBN3_SUPPORTED &&
                shape.arithmetic_id == expected.arithmetic_id && shape.work_bytes == expected.work_bytes &&
                shape.output_limbs == expected.output_limbs,
            SBN3_FATAL_MATH, "radix product replay");
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
                       const uint64_t *b, size_t bn, uint64_t *out, size_t out_capacity) noexcept {
#ifdef SBN3_RADIX_TRACE // rail trace
    timespec t0{}, t1{}, t2{};
    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
    ProductProgramPlan plan{};
    ProductShape shape{};
    require(product_shape(an, bn, workers, shape, &plan) == SBN3_SUPPORTED && shape.output_limbs <= out_capacity,
            SBN3_FATAL_MATH, "radix temporary product");
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
bool square_plan(size_t limbs, unsigned workers, SquarePlan &out) noexcept {
    sbn3_product_request r{};
    r.kind = SBN3_PRODUCT_SQR;
    r.a_limbs = limbs;
    sbn3_mul_options o{};
    o.workers = workers;
    if (sbn3_product_query(&r, &o, &out.plan, &out.info) != SBN3_SUPPORTED)
        return false;
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
sbn3_query_result rail_finish(const BaseInfo &base, unsigned workers, RailPlan &rail) noexcept {
    if (rail.count > max_rail)
        return SBN3_QUERY_CAPACITY;
    rail.total_limbs = 0;
    rail.setup_bytes = 0;
    for (unsigned k = 0; k < rail.count; ++k) {
        rail.limbs[k] = rail_limbs(base, k);
        rail.offset[k] = rail.total_limbs;
        rail.total_limbs += align_to(rail.limbs[k], 8);
        rail.square_id[k] = 0;
    }
    for (unsigned k = 0; k + 1 < rail.count; ++k) {
        if (rail.limbs[k] <= rail_basecase_limbs)
            continue;
        ProductShape s{};
        const auto rc = product_shape(rail.limbs[k], rail.limbs[k], square_workers(workers, rail.limbs[k]), s);
        if (rc != SBN3_SUPPORTED)
            return rc;
        rail.square_id[k] = s.arithmetic_id;
        rail.setup_bytes = std::max(rail.setup_bytes, s.temporary_bytes() + 64);
        SquarePlan service{};
        if (rail.limbs[k] >= rail_square_service_limbs && square_plan(rail.limbs[k], square_workers(workers, rail.limbs[k]), service))
            rail.setup_bytes = std::max(rail.setup_bytes, service.bytes + (size_t(1) << 21));
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
        if (m >= rail_square_service_limbs && square_plan(m, square_workers(workers, m), service)) {
            // The product service binds its own leases: carve them out of the scratch range for this one squaring.
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
        temporary_product(&team, square_workers(workers, m), scratch, q, m, q, m, z, 2 * m);
        for (size_t j = next; j < 2 * m; ++j)
            require(!z[j], SBN3_FATAL_MATH, "radix rail capacity");
        parallel_limbs::copy(&team, out, z, std::min(next, 2 * m));
        pad(k + 1, std::min(next, 2 * m));
    }
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
        c.step[c.steps++] = {k, c.limbs, next};
        c.widest = std::max(c.widest, c.limbs + rail_limbs(b, k));
        c.limbs = next;
    }
    return c;
}
sbn3_query_result chain_bytes(const BaseInfo &b, const Chain &c, unsigned workers, size_t &bytes) noexcept {
    size_t episode = 0;
    for (unsigned j = 0; j < c.steps; ++j) {
        const auto &s = c.step[j];
        const size_t bn = rail_limbs(b, s.level);
        if (std::max(s.acc_limbs, bn) <= chain_basecase_limbs)
            continue;
        ProductShape shape{};
        const auto rc = product_shape(s.acc_limbs, bn, workers, shape);
        if (rc != SBN3_SUPPORTED)
            return rc;
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
            temporary_product(team, workers, scratch, x, s.acc_limbs, entries[s.level], bn, y, c.widest + 8);
        for (size_t i = s.out_limbs; i < s.acc_limbs + bn; ++i)
            require(!y[i], SBN3_FATAL_MATH, "radix power capacity");
        std::swap(x, y);
    }
    return x;
}
} // namespace sbn::v3::radix
