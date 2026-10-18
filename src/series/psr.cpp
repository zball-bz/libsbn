#include "series/psr.hpp"
#include "series/precision_policy.hpp"
#include "series/precision_merge.hpp"
#include "series/limited.hpp"
#include "product/program.hpp"
#include "product/cost_model.hpp"
#include "product/root_prepare_cost.hpp"
#include "common/identity.hpp"
#include <algorithm>
#include <cstring>
#include <new>
#include <time.h>
namespace sbn::v3::series {
namespace {
constexpr unsigned max_blocks = 128;
// Four mantissa guard words cover word rounding, <=2^48 terms, a <2^63
// local affine value, and propagation through the contraction certificate.
constexpr size_t guard_words = precision_guard_words;
// Relative cost of a merge that runs its np primes in several passes because
// only `batch` of them fit the pool at a time: 1 + penalty (passes - 1),
// passes = ceil(np/batch). See the merge planning in assemble().
constexpr double extra_pass_penalty = 0.125;
struct Block {
    sbn3_series_range range{};
    size_t precision = 0, next_precision = 0, metadata = 0, prepared = 0, scratch = 0, multiply_work = 0;
    size_t exact_words = 0; // nonzero: exact block, the constant limit of all of its nodes
    sbn3_series_info finite{};
    FiniteFormula formula{}; // stable context for the stored schedule
    sbn3_series_plan *schedule=nullptr;
    ProductProgramPlan multiply{}, contraction{};
};
size_t up(size_t n, size_t a = 128) {
    size_t r;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "PSR layout alignment");
    return r;
}
constexpr size_t block_stride = (sizeof(Block) + 127) & ~size_t(127);
uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
struct MergeBudget {
    size_t an=0,bn=0,offset=0,fast=0,compact=0,pitch=0;
    uint64_t arithmetic=0,execution=0,backend=0;
    size_t pair_bytes=0,prepared_bytes=0;
    sbn3_mul_options options{};
    unsigned batch=1;
    bool adjustable=false, paired=true;
};
size_t product_work(const ProductProgramPlan &p) {
    return std::max(p.info.workspace_bytes,p.pair_workspace_bytes);
}
MergeBudget make_budget(const ProductProgramPlan &p, size_t offset, bool paired) {
    MergeBudget r{};
    r.paired=paired;r.an=p.an;r.bn=p.bn;r.offset=offset;
    r.fast=r.compact=paired?product_work(p):p.info.workspace_bytes;
    r.arithmetic=p.info.arithmetic_id;r.execution=p.info.execution_id;r.backend=p.plan.opaque[1];
    r.pair_bytes=p.pair_workspace_bytes;r.prepared_bytes=p.prepared_bytes;
    r.pitch=p.info.plane_pitch;r.batch=p.info.prime_batch;
    ProductProgramPlan compact{};
    if(r.batch>1 && product_program_rebatch(p,1,compact,&r.options)==SBN3_SUPPORTED) {
        r.compact=paired?product_work(compact):compact.info.workspace_bytes;
        r.adjustable=true;
    }
    return r;
}
sbn3_query_result assemble(const PsrSpec &s, PsrInfo &out, void *destination = nullptr,
                           unsigned expected = 0, size_t destination_bytes = 0) {
    out = {};
    if (!precision_spec_supported(s))
        return SBN3_UNSUPPORTED;
    uint64_t ends[max_blocks]{};bool exact[max_blocks]{};unsigned total_blocks=0;
    for(uint64_t at=s.range.begin;at<s.range.end;){
        if(total_blocks==max_blocks)return SBN3_QUERY_CAPACITY;
        PrecisionBlock c{};
        const auto rc=precision_block_query(s,at,c);
        if(rc!=SBN3_SUPPORTED)return rc;
        at=c.range.end;exact[total_blocks]=c.exact_words!=0;ends[total_blocks++]=at;
    }
    size_t metadata_cursor=up(total_blocks*block_stride);
    if(destination)require(total_blocks==expected && metadata_cursor<=destination_bytes,SBN3_FATAL_WORKSPACE,"PSR metadata headers");
    MergeBudget budgets[2*max_blocks]{};
    struct MergeShape {
        size_t an = 0, un = 0, next_precision = 0; // un: separate U support (Common), zero when U shares the pair
    } merges[max_blocks]{};
    // One bounded memo for the whole pass: neighbouring subtrees and blocks
    // ask for the same product shapes many times over.
    ProductPlanMemo memo;
    memo.record=destination?nullptr:s.record;
    memo.replay=destination?s.replay:nullptr;
    size_t pool_floor=0;
    uint64_t hash = identity::word(identity::fnv_seed, s.formula.formula_id), a = s.range.begin,
             previous_decay = 0;
    for (; a < s.range.end;) {
        if (out.blocks == max_blocks)
            return SBN3_QUERY_CAPACITY;
        Block temporary{};
        auto &b=destination?*::new(static_cast<char *>(destination)+out.blocks*block_stride)Block{}:temporary;
        b.range = {a, ends[out.blocks]};
        out.exact_blocks += exact[out.blocks];
        b.precision = precision_words(s, a);
        b.exact_words = exact[out.blocks] ? b.precision + guard_words : 0;
        b.formula=precision_formula(s,exact[out.blocks]?&b.exact_words:nullptr); // the block is its own stable limit context
        const uint64_t decay = s.precision.decay_bits(s.precision.context, b.range.end);
        if (decay < previous_decay)
            return SBN3_UNSUPPORTED;
        previous_decay = decay;
        const bool has_next = b.range.end < s.range.end;
        b.next_precision = has_next ? precision_words(s, b.range.end) : 0;
        const unsigned need = has_next && s.formula.recipe == SBN3_SERIES_COMMON_P2B3 ? 7 : 3;
        sbn3_query_result rc;
        b.metadata=metadata_cursor;
        if(destination){
            require(metadata_cursor<=destination_bytes,SBN3_FATAL_WORKSPACE,"PSR schedule storage");
            rc=finite_compile(b.formula,b.range,need,s.series,static_cast<char *>(destination)+metadata_cursor,
                              destination_bytes-metadata_cursor,b.finite,b.schedule,&memo);
        }else{
            FinitePlan fp{};rc=finite_query(b.formula,b.range,need,s.series,fp,&memo);b.finite=fp.info;
        }
        if(rc!=SBN3_SUPPORTED)return rc;
        metadata_cursor=up(metadata_cursor+b.finite.plan_bytes);
        b.prepared=0;
        b.scratch=up(b.finite.prepared_bytes,b.finite.workspace_alignment);
        pool_floor=std::max(pool_floor,b.scratch+b.finite.workspace_bytes);
        out.storage_alignment = std::max(out.storage_alignment, b.finite.workspace_alignment);
        const size_t tn = b.finite.output.limbs[0], dn = b.finite.output.limbs[1],
                     un = b.finite.output.limbs[2];
        out.t_limbs = std::max(out.t_limbs, tn);
        out.d_limbs = std::max(out.d_limbs, dn);
        out.u_limbs = std::max(out.u_limbs, std::max(size_t(1), un));
        sbn3_series_shape logical{};
        rc = finite_value_shape(b.formula, b.range, need, logical);
        if (rc != SBN3_SUPPORTED)
            return rc;
        const size_t an = std::max({logical.limbs[0], logical.limbs[1], logical.limbs[2]});
        const size_t pn = has_next ? an + b.next_precision + guard_words : 1;
        out.numerator_limbs = std::max(out.numerator_limbs, pn + 2);
        if (!has_next)
            out.product_limbs = std::max(out.product_limbs, pn + 2);
        merges[out.blocks] = {an, s.formula.recipe == SBN3_SERIES_COMMON_P2B3 && logical.limbs[2] < an ? logical.limbs[2] : 0,
                              b.next_precision};
        for (uint64_t x : {b.range.begin, b.range.end, b.precision, b.next_precision, uint64_t(exact[out.blocks]),
                           b.finite.schedule_id})
            hash = identity::word(hash, x);
        ++out.blocks;
        a = b.range.end;
    }
    // Merges are planned once every finite stage is known: they run inside
    // the pool those stages need anyway. The product selector ranks plans at
    // their full prime batch; under the pool, a plan with few primes (a coarse
    // plane pitch) can fall to one prime per pass, which costs about twice the
    // time, while a plan with more primes still runs most of its batch. A
    // merge whose selected plan does not fit is therefore re-ranked over the
    // prime counts by the product model's one-use price scaled by the measured
    // cost of extra passes (results/planner_2026-09-18/merge-plan-time/, 1.0M x
    // 0.67M words, 16 workers: two passes cost 4-16% over the full batch,
    // three 9-46%, four 26-38%, one prime at a time 30-91%; the product model
    // itself separates the prime counts by only about 10%). Only alternatives
    // that need no larger pool, keep the pairing and the input contracts, and
    // produce no larger output are admitted.
    const size_t finite_floor = pool_floor;
    for (unsigned k = 0; k + 1 < out.blocks; ++k) {
        const auto &m = merges[k];
        const size_t bn = m.next_precision + guard_words, pn = m.an + bn;
        sbn3_mul_options mo{};
        mo.workers = s.series.workers;
        mo.borrow_output = 0;
        ProductProgramPlan multiply{}, contraction{};
        auto rc = memo.query(m.an, bn, mo, multiply);
        if (rc != SBN3_SUPPORTED)
            return rc;
        // Each merge program runs once: its table preparation is part of its price.
        product_price_few_uses(memo, m.an, bn, mo, 1, true, multiply);
        // U has its own support and does not share the paired denominator spectrum.
        if (m.un && memo.query(m.un, bn, mo, contraction) == SBN3_SUPPORTED)
            product_price_few_uses(memo, m.un, bn, mo, 1, false, contraction);
        else
            contraction = {};
        auto table_bytes = [&](const ProductProgramPlan &x, const ProductProgramPlan &u, size_t &alignment) {
            const bool single = u.prepared_bytes != 0;
            const unsigned common_flags = x.contract & (single ? u.contract : ~0u);
            const bool direct = (common_flags & (program_bounded_inputs | program_consume_inputs)) ==
                                (program_bounded_inputs | program_consume_inputs);
            const size_t pads = (common_flags & program_bounded_inputs) ? 0 : up(m.an * 8) + up(bn * 8) + 128;
            const size_t padding = pads + (direct ? 0 : up(pn * 8) + 128);
            alignment = std::max(x.info.workspace_alignment, single ? u.info.workspace_alignment : size_t(1));
            return up(x.prepared_bytes + (single ? u.prepared_bytes : 0) + padding, alignment);
        };
        // Primes of the full batch that run concurrently inside `room` bytes; zero when even the compact form does not fit.
        auto fitting = [](const MergeBudget &r, size_t room) -> unsigned {
            if (r.fast <= room)
                return r.batch;
            if (!r.adjustable || r.compact > room)
                return 0;
            return r.pitch && r.fast - r.compact == r.pitch * (r.batch - 1)
                       ? unsigned(std::min(size_t(r.batch), 1 + (room - r.compact) / r.pitch))
                       : 1;
        };
        auto price = [](const ProductProgramPlan &x, size_t xn, size_t yn, unsigned fit) {
            const double full = cost_model::linear_product(x.info, xn, yn).nanoseconds + root_prepare_cost(x.info);
            const unsigned passes = (x.info.prime_batch + fit - 1) / fit;
            return full * (1 + extra_pass_penalty * (passes - 1));
        };
        auto rerank = [&](bool paired) {
            auto &chosen = paired ? multiply : contraction;
            const size_t xn = paired ? m.an : m.un;
            size_t alignment = 0;
            const size_t offset = table_bytes(multiply, contraction, alignment);
            if (offset >= finite_floor)
                return;
            const auto current = make_budget(chosen, offset, paired);
            unsigned fit = fitting(current, finite_floor - offset);
            // Fits as selected, the pool has to grow anyway, or two passes: that costs about as much
            // as the product model's own uncertainty between prime counts, so the selection stands.
            if (!fit || (current.batch + fit - 1) / fit < 3)
                return;
            double best = price(chosen, xn, bn, fit);
            constexpr unsigned kept = program_consume_inputs | program_bounded_inputs;
            const auto selected = chosen;
            // More primes mean thinner planes and a smaller compact form, so only larger prime
            // counts can run in fewer passes inside the same pool.
            for (unsigned np = selected.info.np + 1; selected.info.np && np <= 8; ++np) {
                auto pinned = mo;
                pinned.prime_count = np;
                ProductProgramPlan candidate{};
                if (memo.query(xn, bn, pinned, candidate) != SBN3_SUPPORTED)
                    continue;
                if ((paired && selected.pair_workspace_bytes && !candidate.pair_workspace_bytes) ||
                    ((selected.contract & kept) & ~candidate.contract) ||
                    candidate.info.output_limbs > selected.info.output_limbs)
                    continue;
                const auto &x = paired ? candidate : multiply;
                const auto &u = paired ? contraction : candidate;
                const size_t at = table_bytes(x, u, alignment);
                if (at >= finite_floor)
                    continue;
                fit = fitting(make_budget(candidate, at, paired), finite_floor - at);
                if (!fit)
                    continue;
                const double cost = price(candidate, xn, bn, fit);
                if (cost < best) {
                    best = cost;
                    chosen = candidate;
                }
            }
        };
        rerank(true);
        if (contraction.prepared_bytes)
            rerank(false);
        size_t alignment = 0;
        const size_t multiply_work = table_bytes(multiply, contraction, alignment);
        const bool single = contraction.prepared_bytes != 0;
        auto &budget = budgets[2 * k];
        budget = make_budget(multiply, multiply_work, true);
        auto &ub = budgets[2 * k + 1];
        if (single)
            ub = make_budget(contraction, multiply_work, false);
        pool_floor = std::max(pool_floor, multiply_work + std::max(budget.compact, ub.compact));
        out.storage_alignment = std::max(out.storage_alignment, alignment);
        // The product slot holds U*N_tail: the contraction's output when U has its own program.
        out.product_limbs = std::max(out.product_limbs, single ? contraction.info.output_limbs + 2 : pn + 2);
        if (destination) {
            auto &b = *reinterpret_cast<Block *>(static_cast<char *>(destination) + k * block_stride);
            b.multiply = multiply;
            b.contraction = contraction;
            b.multiply_work = multiply_work;
        }
        for (uint64_t x : {uint64_t(k), multiply.info.arithmetic_id, contraction.info.arithmetic_id, uint64_t(multiply_work)})
            hash = identity::word(hash, x);
    }
    out.plan_bytes = metadata_cursor;
    size_t cursor = up(out.plan_bytes);
    auto words = [&](size_t n) {
        const size_t at = cursor;
        cursor = up(cursor + 8 * n);
        return at;
    };
    // A single block has no merge: its tree writes the caller's pair directly
    // (a last block never needs U), so no value or product slot exists.
    const bool direct = out.blocks == 1;
    const size_t R = up(8 * std::max({out.t_limbs, out.d_limbs, out.u_limbs}), 128) / 8;
    const size_t V = direct ? 0 : up(8 * std::max(2 * R, out.numerator_limbs),128)/8;
    const size_t Y = direct ? 0 : up(8 * std::max(R, out.product_limbs),128)/8;
    out.t_limbs = out.d_limbs = out.u_limbs = R;
    out.numerator_limbs=V;out.product_limbs=Y;
    out.T = out.numerator = words(V);
    out.D = out.T + 8 * R;
    out.U = out.product = words(Y);
    // The direct root lane can need spare words beyond the value it leaves
    // (lane reuse of the finite tree); the caller's pair then provides them.
    out.output_limbs = up(8 * std::max(s.fractional_limbs + guard_words + 2, direct ? R : size_t(0)), out.storage_alignment) / 8;
    out.value_bytes = cursor - up(out.plan_bytes);
    out.pool = up(cursor, out.storage_alignment);
    // A single pool is shared by every block. Its floor already covers
    // all finite stages and the compact execution of each merge. Retain a
    // faster/larger merge batch whenever it fits this SAME resident pool.
    // Only these few execution alternatives are replayed; finite planning
    // is not run a second time, and no new size threshold is introduced.
    // The caller's allowance (memory it holds resident anyway) may widen
    // the pool up to what the fast merges can use, never beyond.
    size_t pool=pool_floor;
    if(s.storage_allowance>out.pool){
        size_t wanted=pool_floor;
        for(unsigned j=0;j<2*out.blocks;++j)
            if(budgets[j].an)wanted=std::max(wanted,budgets[j].offset+budgets[j].fast);
        pool=std::max(pool_floor,std::min(wanted,s.storage_allowance-out.pool));
    }
    hash=identity::word(identity::word(hash,0x505352504f4f4c31ULL),pool);
    out.pool_bytes=pool;
    for(unsigned j=0;j<2*out.blocks;++j){
        auto &r=budgets[j];
        if(!r.an)continue;
        if(r.fast>pool-r.offset){
            require(r.adjustable && r.compact<=pool-r.offset,SBN3_FATAL_MATH,"PSR execution pool floor");
            unsigned batch=1;
            if(r.pitch && r.fast-r.compact==r.pitch*(r.batch-1))
                batch=unsigned(std::min(size_t(r.batch),1+(pool-r.offset-r.compact)/r.pitch));
            auto options=r.options;options.prime_batch=batch;
            ProductProgramPlan selected{};
            require(product_program_replay(r.an,r.bn,options,r.backend,selected)==SBN3_SUPPORTED &&
                        selected.info.arithmetic_id==r.arithmetic && selected.prepared_bytes==r.prepared_bytes &&
                        (r.paired?product_work(selected):selected.info.workspace_bytes)<=pool-r.offset,
                    SBN3_FATAL_MATH,"PSR budgeted execution replay");
            r.execution=selected.info.execution_id;r.pair_bytes=selected.pair_workspace_bytes;
            if(destination){auto &b=*reinterpret_cast<Block *>(static_cast<char *>(destination)+(j/2)*block_stride);
                (r.paired?b.multiply:b.contraction)=selected;}
        }
        hash=identity::word(identity::word(hash,r.execution),r.pair_bytes);
    }
    out.storage_bytes = out.pool + out.pool_bytes;
    out.schedule_id = identity::word(identity::word(identity::word(hash, 0x50535234534c4f54ULL), R), V);
    for(uint64_t x:{out.plan_bytes,out.value_bytes,out.pool,out.pool_bytes,out.storage_bytes,out.storage_alignment,
                    out.T,out.D,out.U,out.numerator,out.product,out.output_limbs})
        out.schedule_id=identity::word(out.schedule_id,x);
    out.error_units = 2 * out.blocks + 1;
    return SBN3_SUPPORTED;
}
uint64_t *at(PsrBinding &b, size_t offset) {
    return reinterpret_cast<uint64_t *>(b.arena->base + b.offset + offset);
}
sbn3_lease lease(PsrBinding &b, size_t offset, size_t bytes) {
    return b.arena->acquire(b.offset + b.info.pool + offset, bytes);
}
void evaluate(PsrBinding &b, const Block &block, sbn3_series_values &values) {
    auto prepared = lease(b, block.prepared, block.finite.prepared_bytes),
         scratch = lease(b, block.scratch, block.finite.workspace_bytes);
    FinitePlan fp{};
    fp.formula = block.formula;
    fp.options = b.spec.series;
    fp.info = block.finite;
    const unsigned need = block.finite.output.limbs[2] ? 7 : 3;
    fp.spec = {fp.formula.recipe, block.range, need, fp.formula.formula_id, fp.formula.parameter_id};
    FiniteBinding run{};
    uint64_t start = now();
    finite_prepare(fp, *b.arena, *b.team, b.plan, prepared, scratch, run, block.schedule);
    b.metrics.prepare_ns += now() - start;
    start = now();
    finite_execute(run, values);
    b.metrics.finite_ns += now() - start;
    for (auto *l : {&prepared, &scratch})
        sbn3_arena_release(b.arena, l);
}
void combine(PsrBinding &b, const Block &block, const sbn3_series_values &v, sbn3_series_values &result) {
    const uint64_t start = now();
    const size_t cap = block.precision + guard_words;
    if (!block.next_precision) {
        copy_limited(result.value[0], v.value[0], cap);
        copy_limited(result.value[1], v.value[1], cap);
    } else {
        auto table = lease(b, 0, block.multiply_work),
             work = lease(b, block.multiply_work,
                          std::max({block.multiply.info.workspace_bytes, block.multiply.pair_workspace_bytes,
                                    block.contraction.info.workspace_bytes}));
        {
            Frame tf(*b.arena, table), wf(*b.arena, work);
            const auto program = product_program_prepare(block.multiply, tf);
            ProductProgram contraction{};
            const bool single=block.contraction.prepared_bytes!=0;
            if(single)contraction=product_program_prepare(block.contraction,tf);
            const unsigned common_flags=program.contract & (single?contraction.contract:~0u);
            auto *a = (common_flags & program_bounded_inputs)
                          ? nullptr
                          : static_cast<uint64_t *>(tf.allocate(program.an * 8, 128));
            auto *d = (common_flags & program_bounded_inputs)
                          ? nullptr
                          : static_cast<uint64_t *>(tf.allocate(program.bn * 8, 128));
            const bool direct = (common_flags & (program_bounded_inputs | program_consume_inputs)) ==
                                (program_bounded_inputs | program_consume_inputs);
            auto *fallback =
                direct ? nullptr : static_cast<uint64_t *>(tf.allocate((program.an + program.bn) * 8, 128));
            PrecisionMergeCall call{&program,
                           single?&contraction:nullptr,
                           &wf,
                           &v,
                           &result,
                           {{at(b, b.info.numerator), b.info.numerator_limbs, 0, 0}, 0},
                           {{at(b, b.info.product), b.info.product_limbs, 0, 0}, 0},
                           a,
                           d,
                           fallback,
                           b.info.output_limbs,
                           cap,
                           b.spec.formula.recipe};
            sbn3_team_run(b.team, precision_merge_action, &call);
        }
        sbn3_arena_release(b.arena, &table);
        sbn3_arena_release(b.arena, &work);
    }
    require(result.value[1].mantissa.size && !result.value[1].mantissa.negative, SBN3_FATAL_MATH,
            "PSR positive denominator");
    b.metrics.merge_ns += now() - start;
}
} // namespace
sbn3_query_result psr_query(const PsrSpec &s, PsrInfo &info) noexcept {
    return assemble(s, info);
}
void psr_prepare(const PsrSpec &spec, const PsrInfo &info, sbn3_arena &arena, sbn3_team &team, size_t offset,
                 PsrBinding &b) noexcept {
    require(team.arena == &arena && team.width == spec.series.workers && !team.busy &&
                arena.contains(offset, offset + info.storage_bytes) &&
                arena.unleased(offset, info.storage_bytes) &&
                !(uintptr_t(arena.base + offset) & (info.storage_alignment - 1)),
            SBN3_FATAL_WORKSPACE, "PSR prepared region");
    b = {spec, info, &arena, &team, offset};
    b.plan = arena.acquire(offset, info.plan_bytes);
    if (info.value_bytes) // a single block writes the caller's pair: no value slots
        b.values = arena.acquire(offset + up(info.plan_bytes), info.value_bytes);
    PsrInfo replay{};
    require(assemble(b.spec, replay, b.plan.data, info.blocks, info.plan_bytes) == SBN3_SUPPORTED &&
                replay.schedule_id == info.schedule_id && replay.storage_bytes == info.storage_bytes,
            SBN3_FATAL_MATH, "PSR plan replay");
}
void psr_execute(PsrBinding &b, sbn3_series_values &out) noexcept {
    require(!b.used, SBN3_FATAL_LIFETIME, "PSR one-shot execution");
    for (unsigned j = 0; j < 2; ++j) {
        const auto &m = out.value[j].mantissa;
        require(m.data && m.capacity >= b.info.output_limbs && !(uintptr_t(m.data) & 63) &&
                    !overlaps(m.data, m.capacity * 8, b.arena->base + b.offset, b.info.storage_bytes),
                SBN3_FATAL_ARGUMENT, "PSR pair output/lifetime");
    }
    require(!overlaps(out.value[0].mantissa.data, out.value[0].mantissa.capacity * 8,
                      out.value[1].mantissa.data, out.value[1].mantissa.capacity * 8),
            SBN3_FATAL_ARGUMENT, "PSR disjoint pair");
    require(out.value[1].mantissa.data == out.value[0].mantissa.data + b.info.output_limbs,
            SBN3_FATAL_ARGUMENT, "PSR contiguous pair storage");
    b.used = true;
    for (unsigned j = b.info.blocks; j-- > 0;) {
        const auto &block =
            *reinterpret_cast<const Block *>(static_cast<const char *>(b.plan.data) + j * block_stride);
        sbn3_series_values v{};
        if (b.info.blocks == 1) {
            v.value[0].mantissa = {out.value[0].mantissa.data, out.value[0].mantissa.capacity, 0, 0};
            v.value[1].mantissa = {out.value[1].mantissa.data, out.value[1].mantissa.capacity, 0, 0};
        } else {
            v.value[0].mantissa = {at(b, b.info.T), b.info.t_limbs, 0, 0};
            v.value[1].mantissa = {at(b, b.info.D), b.info.d_limbs, 0, 0};
            v.value[2].mantissa = {at(b, b.info.U), b.info.u_limbs, 0, 0};
        }
        evaluate(b, block, v);
        combine(b, block, v, out);
    }
}
void psr_release(PsrBinding &b) noexcept {
    if (b.info.value_bytes)
        sbn3_arena_release(b.arena, &b.values);
    sbn3_arena_release(b.arena, &b.plan);
}
} // namespace sbn::v3::series
