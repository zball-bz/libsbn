// SPDX-License-Identifier: LGPL-3.0-or-later
#include "planning/precision.hpp"
#include "series/precision_merge.hpp"
#include "series/limited.hpp"
#include "common/checked.hpp"
#include <algorithm>

namespace sbn::v3::planning {
namespace {
bool add(size_t a, size_t b, size_t &out) { return add_size(a, b, out); }
bool rounded_words(size_t n, size_t &out) { return align_size(n, 16, out); }
bool bytes(size_t words, size_t &out) { return mul_size(words, sizeof(uint64_t), out); }
size_t product_work(const ProductProgramPlan &p) {
    return std::max(p.info.workspace_bytes, p.pair_workspace_bytes);
}
struct BatchRange {
    ProductProgramPlan fast{},compact{};
    size_t fast_bytes=0,compact_bytes=0,pitch=0;
    unsigned batch=0;
    bool paired=false,adjustable=false;
};
BatchRange batch_range(const ProductProgramPlan&p,bool paired){
    BatchRange r{};r.fast=r.compact=p;r.paired=paired;r.batch=p.info.prime_batch;
    r.fast_bytes=r.compact_bytes=paired?product_work(p):p.info.workspace_bytes;
    if(r.batch>1&&product_program_rebatch(p,1,r.compact)==SBN3_SUPPORTED){
        r.compact_bytes=paired?product_work(r.compact):r.compact.info.workspace_bytes;
        r.adjustable=r.compact_bytes<=r.fast_bytes;
        size_t span;
        if(r.adjustable&&mul_size(p.info.plane_pitch,r.batch-1,span)&&r.fast_bytes-r.compact_bytes==span)r.pitch=p.info.plane_pitch;
    }
    if(!r.adjustable){r.compact=p;r.compact_bytes=r.fast_bytes;}
    return r;
}
ProductProgramPlan fit_batch(const BatchRange&r,size_t room){
    if(r.fast_bytes<=room)return r.fast;
    require(r.compact_bytes<=room,SBN3_FATAL_WORKSPACE,"precision merge compact floor");
    if(!r.adjustable||!r.pitch)return r.compact;
    const unsigned batch=unsigned(std::min(size_t(r.batch),1+(room-r.compact_bytes)/r.pitch));
    if(batch<=1)return r.compact;
    ProductProgramPlan selected{};
    if(product_program_rebatch(r.fast,batch,selected)!=SBN3_SUPPORTED)return r.compact;
    const size_t bytes=r.paired?product_work(selected):selected.info.workspace_bytes;
    if(bytes>room||selected.info.workspace_alignment>std::max(r.fast.info.workspace_alignment,r.compact.info.workspace_alignment))return r.compact;
    return selected;
}
SummaryStatus merge_quote(const series::PsrSpec &s, const series::FiniteFormula &formula,
                          PrecisionEpisode &e, series::ProductPlanMemo *memo) {
    sbn3_series_shape logical{};
    if (series::finite_value_shape(formula, e.block.range, e.block.need, logical) != SBN3_SUPPORTED)
        return SummaryStatus::unsupported;
    const size_t an = std::max({logical.limbs[0], logical.limbs[1], logical.limbs[2]});
    size_t bn, pn;
    if (!add(e.block.next_precision, series::precision_guard_words, bn) || !add(an, bn, pn))
        return SummaryStatus::overflow;
    sbn3_mul_options options{};
    options.workers = s.series.workers;
    auto query = [&](size_t n, ProductProgramPlan &p) {
        return memo ? memo->query(n, bn, options, p) : product_program_query(n, bn, options, p);
    };
    ProductProgramPlan multiply{},contraction{};
    auto rc = query(an, multiply);
    if (rc != SBN3_SUPPORTED) return rc == SBN3_QUERY_CAPACITY ? SummaryStatus::capacity : SummaryStatus::unsupported;
    if (s.formula.recipe == SBN3_SERIES_COMMON_P2B3 && logical.limbs[2] && logical.limbs[2] < an) {
        // Optional shorter contraction shares the same consuming merge recipe.
        if (query(logical.limbs[2], contraction) != SBN3_SUPPORTED) contraction = {};
    }
    const auto pair=batch_range(multiply,true),one=batch_range(contraction,false);
    const bool single = contraction.prepared_bytes != 0;
    const unsigned flags = multiply.contract & (single ? contraction.contract : ~0u);
    const size_t alignment=std::max({size_t(128),multiply.info.workspace_alignment,pair.compact.info.workspace_alignment,
                                    single?contraction.info.workspace_alignment:size_t(1),one.compact.info.workspace_alignment});
    size_t table;
    if (!add(multiply.prepared_bytes, contraction.prepared_bytes, table)) return SummaryStatus::overflow;
    auto reserve = [&](size_t words) {
        size_t n;
        return bytes(words, n) && align_size(n, 128, n) && add(table, n, table);
    };
    if (!(flags & program_bounded_inputs) && (!reserve(an) || !reserve(bn) || !add(table, 128, table)))
        return SummaryStatus::overflow;
    const bool direct = (flags & (program_bounded_inputs | program_consume_inputs)) ==
                        (program_bounded_inputs | program_consume_inputs);
    if (!direct && (!reserve(pn) || !add(table, 128, table))) return SummaryStatus::overflow;
    size_t offset;if(!align_size(table,alignment,offset))return SummaryStatus::overflow;
    for(unsigned d=0;d<2;++d)if(e.finite.summary.candidate[d].supported){
        const auto&info=e.finite.plans[d].info;auto&m=e.merge[d];
        const size_t finite_alignment=std::max({size_t(128),info.prepared_alignment,info.workspace_alignment});
        size_t finite_work,min_pool;
        if(!align_size(info.prepared_bytes,finite_alignment,finite_work)||!add(finite_work,info.workspace_bytes,m.finite_pool_bytes)||
           !add(offset,std::max(pair.compact_bytes,one.compact_bytes),min_pool))return SummaryStatus::overflow;
        const size_t pool=std::max(m.finite_pool_bytes,min_pool),room=pool-offset;
        m.multiply=fit_batch(pair,room);m.contraction=fit_batch(one,room);
        m.work_offset=offset;m.alignment=alignment;
        m.work_bytes=std::max(product_work(m.multiply),m.contraction.info.workspace_bytes);
        m.fast_work_bytes=std::max(pair.fast_bytes,one.fast_bytes);m.compact_work_bytes=std::max(pair.compact_bytes,one.compact_bytes);
        m.original_batch[0]=pair.batch;m.original_batch[1]=one.batch;
    }
    return SummaryStatus::ok;
}
SummaryStatus episode_layout(PrecisionEpisode &e, unsigned which) {
    const auto &p = e.finite.plans[which];
    const auto &i = p.info;
    auto &d = e.storage[which];
    const auto&m=e.merge[which];
    size_t r = std::max({i.output.limbs[0], i.output.limbs[1], i.output.limbs[2]});
    if (!rounded_words(r, d.lane_words)) return SummaryStatus::overflow;
    if (!add(e.block.precision, series::precision_guard_words + 2, d.tail_words)) return SummaryStatus::overflow;
    if (e.direct_root) d.tail_words = std::max(d.tail_words, d.lane_words);
    else {
        size_t twice;
        if (!mul_size(d.lane_words, 2, twice)) return SummaryStatus::overflow;
        size_t pair_output = 0, contraction_output = 0;
        if (e.block.next_precision) {
            if (!add(m.multiply.an, m.multiply.bn, pair_output)) return SummaryStatus::overflow;
            pair_output = std::max(pair_output, m.multiply.info.output_limbs);
            contraction_output = m.contraction.prepared_bytes ? m.contraction.info.output_limbs : pair_output;
            size_t half = pair_output/2 + pair_output%2;
            d.tail_words = std::max(d.tail_words, half);
        }
        if (!add(pair_output, 2, pair_output) || !add(contraction_output, 2, contraction_output))
            return SummaryStatus::overflow;
        if (!rounded_words(std::max(twice, pair_output), d.x_words) ||
            !rounded_words(std::max(d.lane_words, contraction_output), d.y_words)) return SummaryStatus::overflow;
    }
    if (!rounded_words(d.tail_words, d.tail_words)) return SummaryStatus::overflow;
    const size_t align = std::max(size_t(128), std::max(i.prepared_alignment, i.workspace_alignment));
    size_t finite_bytes, merge_bytes;
    if (!align_size(i.prepared_bytes, align, d.finite_work_offset) ||
        !add(d.finite_work_offset, i.workspace_bytes, finite_bytes) ||
        !add(m.work_offset, m.work_bytes, merge_bytes)) return SummaryStatus::overflow;
    SlotEnvelope slots{};
    size_t xbytes, ybytes, tbytes;
    if (!bytes(d.x_words, xbytes) || !bytes(d.y_words, ybytes) || !mul_size(d.tail_words, 16, tbytes))
        return SummaryStatus::overflow;
    slots[Slot::result0] = {xbytes, 128};
    slots[Slot::result1] = {ybytes, 128};
    slots[Slot::result2] = {tbytes, 128};
    slots[Slot::metadata] = {i.plan_bytes, i.plan_alignment};
    slots[Slot::workspace] = {std::max(finite_bytes, merge_bytes), std::max(align, m.alignment)};
    SlotLayout check{};
    const auto rc = layout(slots, check);
    if (rc == SummaryStatus::ok) e.summary.candidate[which] = {true, slots, i.schedule_id};
    return rc;
}
uint8_t *slot(const PrecisionBinding &b, Slot s) {
    return static_cast<uint8_t *>(b.region.data) + b.layout.offsets[unsigned(s)];
}
sbn3_lease acquire(PrecisionBinding &b, Slot s, size_t offset, size_t count) {
    const size_t at = size_t(slot(b, s) - b.arena->base);
    return b.arena->acquire(at + offset, count);
}
void drop(sbn3_arena &a, sbn3_lease &l) {
    if (l.token) sbn3_arena_release(&a, &l);
}
}
SummaryStatus summarize_precision_episode(const series::PsrSpec &spec, uint64_t begin,
                                          PrecisionEpisode &out, series::ProductPlanMemo *memo) noexcept {
    out = {};
    if (series::precision_block_query(spec, begin, out.block) != SBN3_SUPPORTED) return SummaryStatus::unsupported;
    out.spec = &spec;
    out.summary.range = out.block.range;
    out.direct_root = begin == spec.range.begin && out.block.range.end == spec.range.end;
    const auto formula = series::precision_formula(spec, out.block.exact_words ? &out.block.exact_words : nullptr);
    auto status = summarize_native_subtree(formula, out.block.range, out.block.need, spec.series, out.finite, memo);
    if (status != SummaryStatus::ok) return status;
    if (out.block.next_precision) {
        status = merge_quote(spec, formula, out, memo);
        if (status != SummaryStatus::ok) return status;
    }
    for (unsigned d = 0; d < 2; ++d) if (out.finite.summary.candidate[d].supported) {
        status = episode_layout(out, d);
        if (status != SummaryStatus::ok) return status;
    }
    return SummaryStatus::ok;
}
void realize_precision_episode(PrecisionEpisode&e,Decomposition chosen,const SlotEnvelope&capacity) noexcept {
    const unsigned d=unsigned(chosen);
    require(d<2&&e.summary.candidate[d].supported&&contains(capacity,e.summary.candidate[d].slots),SBN3_FATAL_WORKSPACE,"precision realization envelope");
    if(!e.block.next_precision)return;
    auto&m=e.merge[d];const size_t room=capacity[Slot::workspace].bytes-m.work_offset;
    auto expand=[&](ProductProgramPlan&p,unsigned original,bool paired){
        if(!p.prepared_bytes||original<=p.info.prime_batch)return;
        ProductProgramPlan full{};
        if(product_program_rebatch(p,original,full)!=SBN3_SUPPORTED)return;
        const auto selected=fit_batch(batch_range(full,paired),room);
        if(selected.info.workspace_alignment<=capacity[Slot::workspace].alignment)p=selected;
    };
    expand(m.multiply,m.original_batch[0],true);expand(m.contraction,m.original_batch[1],false);
    m.work_bytes=std::max(product_work(m.multiply),m.contraction.info.workspace_bytes);
    size_t used;require(add(m.work_offset,m.work_bytes,used)&&used<=capacity[Slot::workspace].bytes,SBN3_FATAL_WORKSPACE,"precision realized merge fits");
    e.summary.candidate[d].slots[Slot::workspace].bytes=std::max(m.finite_pool_bytes,used);
}
void prepare_precision_episode(const PrecisionEpisode &e, Decomposition chosen, const SlotEnvelope &envelope,
                               sbn3_arena &arena, sbn3_team &team, const sbn3_lease &region,
                               PrecisionBinding &b) noexcept {
    const unsigned d = unsigned(chosen);
    require(d < 2 && e.spec && e.summary.candidate[d].supported &&
            contains(envelope, e.summary.candidate[d].slots), SBN3_FATAL_ARGUMENT, "precision episode choice");
    require(team.arena == &arena && team.width == e.spec->series.workers && !team.busy,
            SBN3_FATAL_TEAM, "precision episode team");
    SlotLayout physical{};
    require(layout(envelope, physical) == SummaryStatus::ok && physical.bytes <= region.bytes &&
            !(uintptr_t(region.data) & (physical.alignment-1)), SBN3_FATAL_WORKSPACE, "precision episode region");
    b = {};
    b.episode = &e; b.decomposition = d; b.arena = &arena; b.team = &team;
    b.region = region; b.envelope = envelope; b.layout = physical;
    const auto &plan = e.finite.plans[d];
    b.metadata = acquire(b, Slot::metadata, 0, plan.info.plan_bytes);
    b.finite_prepared = acquire(b, Slot::workspace, 0, plan.info.prepared_bytes);
    b.finite_work = acquire(b, Slot::workspace, e.storage[d].finite_work_offset, plan.info.workspace_bytes);
    series::finite_prepare(plan, arena, team, b.metadata, b.finite_prepared, b.finite_work, b.finite);
}
void execute_precision_episode(PrecisionBinding &b, PrecisionState &tail) noexcept {
    require(b.episode && !b.used, SBN3_FATAL_LIFETIME, "precision episode one shot");
    const auto &e = *b.episode;
    const auto &d = e.storage[b.decomposition];
    const auto &m = e.merge[b.decomposition];
    const size_t pair_words = b.envelope[Slot::result2].bytes/16;
    auto *pair = reinterpret_cast<uint64_t *>(slot(b, Slot::result2));
    if (e.block.next_precision) {
        require(tail.ready && tail.owner == e.spec && tail.begin == e.block.range.end &&
                tail.values.value[0].mantissa.data == pair && tail.values.value[1].mantissa.data == pair+pair_words &&
                tail.values.value[0].mantissa.capacity == pair_words && tail.values.value[1].mantissa.capacity == pair_words,
                SBN3_FATAL_LIFETIME, "precision episode tail dependency");
    } else {
        require(!tail.ready, SBN3_FATAL_LIFETIME, "precision episode tail initialized twice");
        tail = {};
        tail.owner = e.spec;
        for (unsigned j = 0; j < 2; ++j) tail.values.value[j].mantissa = {pair+j*pair_words, pair_words, 0, 0};
    }
    b.used = true;
    auto *x = reinterpret_cast<uint64_t *>(slot(b, Slot::result0));
    auto *y = reinterpret_cast<uint64_t *>(slot(b, Slot::result1));
    sbn3_series_values block{};
    if (e.direct_root) block = tail.values;
    else {
        block.value[0].mantissa = {x, d.lane_words, 0, 0};
        block.value[1].mantissa = {x+d.lane_words, d.lane_words, 0, 0};
        block.value[2].mantissa = {y, d.lane_words, 0, 0};
    }
    series::finite_execute(b.finite, block);
    drop(*b.arena, b.finite_prepared); drop(*b.arena, b.finite_work);
    const size_t precision = e.block.precision + series::precision_guard_words;
    if (!e.block.next_precision) {
        for (unsigned j = 0; j < 2; ++j) series::copy_limited(tail.values.value[j], block.value[j], precision);
    } else {
        auto tables = acquire(b, Slot::workspace, 0, m.work_offset);
        auto work = acquire(b, Slot::workspace, m.work_offset, m.work_bytes);
        {
            Frame tf(*b.arena, tables), wf(*b.arena, work);
            const auto product = product_program_prepare(m.multiply, tf);
            ProductProgram contraction{};
            const bool single = m.contraction.prepared_bytes != 0;
            if (single) contraction = product_program_prepare(m.contraction, tf);
            const unsigned flags = product.contract & (single ? contraction.contract : ~0u);
            uint64_t *a = nullptr, *c = nullptr, *fallback = nullptr;
            if (!(flags & program_bounded_inputs)) {
                a = static_cast<uint64_t *>(tf.allocate(product.an*8, 128));
                c = static_cast<uint64_t *>(tf.allocate(product.bn*8, 128));
            }
            if ((flags & (program_bounded_inputs | program_consume_inputs)) !=
                         (program_bounded_inputs | program_consume_inputs))
                fallback = static_cast<uint64_t *>(tf.allocate((product.an+product.bn)*8, 128));
            series::PrecisionMergeCall call{&product, single ? &contraction : nullptr, &wf, &block, &tail.values,
                {{x, d.x_words, 0, 0}, 0}, {{y, d.y_words, 0, 0}, 0}, a, c, fallback, pair_words, precision, e.spec->formula.recipe};
            sbn3_team_run(b.team, series::precision_merge_action, &call);
        }
        drop(*b.arena, tables); drop(*b.arena, work);
    }
    require(tail.values.value[1].mantissa.size && !tail.values.value[1].mantissa.negative,
            SBN3_FATAL_MATH, "precision episode positive denominator");
    tail.begin = e.block.range.begin; tail.ready = true;
}
void release_precision_episode(PrecisionBinding &b) noexcept {
    require(b.episode && b.arena && !b.team->busy, SBN3_FATAL_LIFETIME, "precision episode release");
    drop(*b.arena, b.finite_prepared); drop(*b.arena, b.finite_work); drop(*b.arena, b.metadata);
    b = {};
}
} // namespace sbn::v3::planning
