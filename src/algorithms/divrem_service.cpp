// Exact integer division service: word/schoolbook for small shapes, block
// Barrett with a Newton block inverse and cached spectra otherwise.
// Derivations: experiments/docs/divrem-design-2026-09-18.md.
#include "sbn3/divrem.h"
#include "algorithms/divrem_tuning.hpp"
#include "algorithms/small_division.hpp"
#include "algorithms/local_divrem.hpp"
#include "product/difference.hpp"
#include "algorithms/divrem_prepared.hpp"
#include "algorithms/newton_tuning.hpp"
#include "algorithms/newton_limits.hpp"
#include "algorithms/local_inverse.hpp"
#include "algorithms/fused_divrem.hpp"
#include "algorithms/newton_planner.hpp"
#include "algorithms/newton_contract.hpp"
#include "algorithms/divide_terminal.hpp"
#include "product/stage.hpp"
#include "algorithms/inverse_tuning.hpp"
#include "common/checked.hpp"
#include "common/identity.hpp"
#include "product/native_capabilities.hpp"
#include "product/repeated_product.hpp"
#include "runtime/team.hpp"
#include "runtime/scratch.hpp"
#include "value/divrem_words.hpp"
#include "value/limbs.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <cmath>
#include <new>
#include <string.h>
#include <time.h>
static void execute_general(sbn3_divrem_binding *,sbn3_const_limbs,sbn3_limbs,sbn3_limbs,sbn3_divrem_result *);
using DivisionExecutor=sbn::v3::divrem_prepared::ExecuteFn;
static inline DivisionExecutor division_executor(const sbn3_divrem_binding *p) {
    DivisionExecutor fn=nullptr;
    if(p)std::memcpy(&fn,p,sizeof fn);
    return fn;
}
static inline bool compact_binding(const sbn3_divrem_binding *p) {
    const auto fn=division_executor(p);return fn&&fn!=execute_general;
}
namespace sbn::v3 {
namespace {
constexpr uint64_t magic = 0x53424e3344495652ULL; // "SBN3DIVR"
constexpr size_t page = 4096;
using ProductChoice=product::RepeatedProductChoice;
struct TerminalRecipe {
    newton_detail::Choice choice{};
    size_t n=0,bytes=0,bn=0,capacity=0;
    size_t y_at=0,cache_at=0,table_at=0,work_at=0,cache_bytes=0,table_bytes=0,work_bytes=0;
    size_t alignment=4096;
    size_t keep_bytes=0,compact_bytes=0;
    uint64_t local=0,keep_cache=1; // 1 ordinary, 2 retained compact, 3 compact
};
struct Plan {
    uint64_t marker = magic, seal = 0;
    sbn3_divrem_request request{};
    sbn3_divrem_options options{};
    sbn3_divrem_info info{};
    size_t in = 0, ring = 0, xlen = 0;
    size_t head = 0; // longest short head block served by word division
    ProductChoice u{}, t{};
    // Layout, byte offsets from the binding's range start.
    size_t persistent_offset = 0, d_offset = 0, dn_offset = 0, u_offset = 0, uspec_offset = 0, dspec_offset = 0;
    size_t utab_offset = 0, uwork_offset = 0, ttab_offset = 0, twork_offset = 0;
    size_t shared_offset = 0, newton_offset = 0, newton_bytes = 0, newton_alignment = 0;
    size_t xbuf_offset = 0, tbuf_offset = 0, ubuf_offset = 0, pbuf_offset = 0;
    size_t tcap = 0, pcap = 0, scratch_words = 0;
    // A full serialized word, so the opaque plan has no indeterminate tail
    // padding when a compact quote constructs the same execution plan.
    uint64_t shared_products = 0;
    uint64_t local_block = 0; // fixed local Barrett; no product-service bindings
    uint64_t local_fused=0;
    TerminalRecipe terminal{}; // n!=0 selects the fused short quotient
    size_t phase_offset=0;
};
using Queried=product::RepeatedProductPlans;
// The opaque plan carries the resolved recipes: bind neither searches nor
// re-queries; it only checks the seal and rebuilds the layout from them.
struct Stored {
    Plan plan{};
    Queried uq{}, tq{};
    sbn3_newton_plan nplan{};
    sbn3_newton_info ninfo{};
};
static_assert(sizeof(Stored) <= sizeof(sbn3_divrem_plan));
struct Binding {
    DivisionExecutor run = ::execute_general;
    Plan plan{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, persistent{}, shared{}, utab{}, uwork{}, ttab{}, twork{}, uspec_lease{}, dspec_lease{};
    uint64_t *D = nullptr, *Dn = nullptr, *U = nullptr;
    unsigned shift = 0;
    uint64_t dinv = 0; // 3/2 reciprocal of the top two normalized divisor limbs
    bool prepared = false, shared_leased = false;
    Queried uq{}, tq{};
    sbn3_mul_binding *uprod = nullptr, *tprod = nullptr;
    sbn3_spectrum *uspec = nullptr, *dspec = nullptr;
    sbn3_newton_plan nplan{};
    sbn3_newton_info ninfo{};
    newton_detail::Bundle *terminal_plans=nullptr;
    sbn3_lease phase_output{};
    uint64_t *xbuf = nullptr, *tbuf = nullptr, *ubuf = nullptr, *pbuf = nullptr;
    sbn3_divrem_metrics metrics{};
};
uint64_t feed(uint64_t h, uint64_t v) {
    return identity::word(h, v);
}
uint64_t seal(const Plan &p) {
    uint64_t h = feed(1469598103934665603ULL, p.marker);
    const uint64_t words[] = {p.request.numerator_limbs, p.request.denominator_limbs, p.options.workers,
                              p.options.prime_count, p.options.memory_budget, p.options.reuse_hint,
                              p.options.block_limbs, p.options.residual, p.options.algorithm, p.options.timing, p.info.algorithm, p.info.storage_bytes,
                              p.info.plan_id,p.in,p.ring,p.xlen,p.head,product::repeated_choice_identity(p.u),
                              product::repeated_choice_identity(p.t),p.persistent_offset,p.shared_offset,
                              p.newton_offset, p.newton_bytes, p.newton_alignment, p.tcap, p.pcap, p.scratch_words, uint64_t(p.shared_products),p.local_block,
                              p.terminal.n,p.terminal.bytes,product::window_choice_identity(p.terminal.choice),
                              p.phase_offset,p.terminal.bn,p.terminal.capacity,
                              p.terminal.y_at,p.terminal.cache_at,p.terminal.table_at,p.terminal.work_at,p.terminal.cache_bytes,
                              p.terminal.table_bytes,p.terminal.work_bytes,p.terminal.alignment,uint64_t(p.terminal.local),
                              p.terminal.keep_bytes,p.terminal.compact_bytes,uint64_t(p.terminal.keep_cache),uint64_t(p.local_fused)};
    for (uint64_t w : words)
        h = feed(h, w);
    return h;
}
void load(const sbn3_divrem_plan &p, Stored &s) {
    memcpy(&s, p.opaque, sizeof s);
    require(s.plan.marker == magic && s.plan.seal == seal(s.plan), SBN3_FATAL_ARGUMENT, "division plan identity");
}
size_t aligned(size_t n, size_t a) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "division layout alignment");
    return r;
}
uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
size_t inverse_local_limit(unsigned workers) {
    return workers==1?local_inverse_max_limbs:inverse_tuning::basecase_limbs;
}
bool inverse_basecase(size_t in,unsigned workers) {
    return in<=inverse_local_limit(workers);
}
sbn3_query_result inverse_query(size_t in, const sbn3_newton_options &o, sbn3_newton_plan &plan, sbn3_newton_info &info) {
    if (!inverse_basecase(in,o.workers))
        return sbn3_newton_query(SBN3_NEWTON_INVERSE, in, &o, &plan, &info);
    plan = {};
    info = {};
    info.kind = SBN3_NEWTON_INVERSE;
    info.precision_limbs = in;
    info.workers = 1;
    info.storage_bytes = local_inverse_approximate_bytes(in);
    info.storage_alignment = page;
    info.plan_id = feed(feed(1469598103934665603ULL, magic), in);
    return SBN3_SUPPORTED;
}
sbn3_newton_options newton_options(const Plan &p) {
    sbn3_newton_options o{};
    o.workers = p.options.workers;
    o.prime_count = p.options.prime_count;
    return o;
}
LocalTerminal local_terminal_recipe(uint64_t tag){
    return tag==2?LocalTerminal::CompactRetained:tag==3?LocalTerminal::Compact:LocalTerminal::Ordinary;
}
bool terminal_query(size_t n,const sbn3_newton_options&o,TerminalRecipe &r,newton_detail::Bundle &v,bool replay=false) {
    const size_t m=newton_contract::next_precision(n);
    if(!replay){
        r={};r.n=n;r.bn=std::max(m+1,newton_contract::residual_words(m,n));
        r.local=product::inline_window_preferred(n,o.prime_count,o.workers)&&n<=local_divide_max_limbs;
        if(!o.prime_count&&o.workers==1&&local_refinement_supported(n,true))r.local=2;
    }
    if(r.local){r.bytes=local_divide_terminal_bytes(n,local_terminal_recipe(r.local));return true;}
    newton_detail::Plan plan{};plan.options=o;plan.info.kind=SBN3_NEWTON_DIVIDE;plan.info.precision_limbs=n;
    if(replay){plan.choices[0]=r.choice;plan.choice_count=1;}
    if(!newton_detail::choose_cycle(plan,newton_detail::Cycle::Division,m,n,0,replay,v))return false;
    if(!replay){
        r.choice=plan.choices[0];r.capacity=std::max(r.choice.ring,n)+3;
        size_t work_align=128;
        for(unsigned k=0;k<v.count;++k){const auto&i=v.infos[k].mul;
            r.capacity=std::max(r.capacity,sbn3_mul_output_capacity(&i));r.table_bytes=std::max(r.table_bytes,i.table_bytes);
            r.work_bytes=std::max(r.work_bytes,i.workspace_bytes);work_align=std::max(work_align,i.workspace_alignment);}
        r.cache_bytes=v.future.storage_bytes;r.y_at=aligned(8*r.bn,128);
        r.cache_at=aligned(r.y_at+8*r.capacity,128);r.table_at=aligned(r.cache_at+r.cache_bytes,128);
        r.alignment=std::max(page,work_align);
        r.work_at=aligned(r.table_at+r.table_bytes,work_align);r.bytes=aligned(r.work_at+r.work_bytes,r.alignment);
        r.keep_bytes=r.bytes;
        for(unsigned k=0;k<v.count;++k){const auto &i=v.infos[k];
            const size_t table=aligned(r.cache_at+(i.cached_mask?r.cache_bytes:0),128);
            const size_t work=aligned(table+i.mul.table_bytes,i.mul.workspace_alignment);
            r.compact_bytes=std::max(r.compact_bytes,aligned(work+i.mul.workspace_bytes,r.alignment));}
    }
    return true;
}
bool fused_layout(Plan &p,const Queried&tq,const sbn3_newton_info&ni) {
    auto&i=p.info;const size_t dn=p.request.denominator_limbs,qn=i.quotient_limbs;
    size_t cursor=aligned(sizeof(Binding),128)+(p.terminal.local?0:aligned(sizeof(newton_detail::Bundle),128));
    i.control_bytes=cursor;p.persistent_offset=aligned(cursor,page);cursor=p.persistent_offset;
    auto place=[&](size_t bytes,size_t alignment,size_t&at){size_t end=0;at=aligned(cursor,alignment);
        if(!add_size(at,bytes,end))return false;cursor=end;return true;};
    if(!place(8*dn,128,p.dn_offset)||!place(8*(p.in+1),128,p.u_offset))return false;
    p.d_offset=p.dn_offset;p.newton_alignment=std::max(ni.storage_alignment,page);
    p.newton_offset=aligned(cursor,p.newton_alignment);p.newton_bytes=ni.storage_bytes;
    if(tq.cached&&!place(tq.future.storage_bytes,page,p.dspec_offset))return false;
    i.persistent_bytes=cursor-p.persistent_offset;i.spectrum_bytes=tq.cached?tq.future.storage_bytes:0;
    p.shared_offset=aligned(cursor,page);cursor=p.shared_offset;
    if(!place(8*(p.terminal.n+1),128,p.ubuf_offset))return false;
    const auto&tm=tq.consumer_info.mul;
    const size_t alignment=std::max({page,p.terminal.alignment,p.newton_alignment,tm.workspace_alignment});
    p.phase_offset=aligned(cursor,alignment);cursor=p.phase_offset;
    p.tcap=std::max(sbn3_mul_output_capacity(&tm),p.ring?p.ring+1:dn+qn+1);
    p.xlen=p.ring?p.ring:dn+qn+1;
    if(!place(p.tcap*8,128,p.tbuf_offset)||!place(tm.table_bytes,128,p.ttab_offset)||
       !place(tm.workspace_bytes,std::max(page,tm.workspace_alignment),p.twork_offset))return false;
    p.xbuf_offset=p.pbuf_offset=p.tbuf_offset;p.pcap=p.terminal.n+1;p.shared_products=0;
    const size_t other_end=std::max(cursor,p.newton_offset+p.newton_bytes);
    if(!p.terminal.local){
        p.terminal.keep_cache=p.phase_offset+p.terminal.keep_bytes<=other_end;
        p.terminal.bytes=p.terminal.keep_cache?p.terminal.keep_bytes:p.terminal.compact_bytes;
    }
    if(p.terminal.local==2){
        // Preserve the ordinary terminal's own existing allowance as well as
        // the other phases. Dropping its cache merely to shrink that allowance
        // can add a forward transform without making the complete call faster.
        const bool ordinary=p.terminal.n<=local_divide_max_limbs;
        const size_t old_bytes=ordinary?local_divide_terminal_bytes(p.terminal.n):0;
        const size_t existing_end=std::max(other_end,p.phase_offset+old_bytes);
        if(p.phase_offset+p.terminal.bytes>existing_end){
            p.terminal.local=ordinary?1:3;
            p.terminal.bytes=ordinary?old_bytes:local_divide_terminal_bytes(p.terminal.n,LocalTerminal::Compact);
        }
    }
    cursor=std::max(other_end,p.phase_offset+p.terminal.bytes);
    i.storage_alignment=alignment;i.storage_bytes=aligned(cursor,alignment);i.shared_bytes=i.storage_bytes-p.shared_offset;
    i.spectrum_bytes+=p.terminal.cache_bytes;
    i.scratch_bytes=8*(p.terminal.n+1)+std::max(8*p.tcap,p.terminal.cache_at);i.table_bytes=std::max(tm.table_bytes,p.terminal.table_bytes);
    i.product_workspace_bytes=std::max(tm.workspace_bytes,p.terminal.work_bytes);i.blocks=1;i.products=4;
    i.block_limbs=qn;i.inverse_limbs=p.in;i.ring_limbs=p.ring;i.head_limbs=0;
    i.workers=std::max({ni.workers,p.t.workers,p.terminal.local?1u:p.terminal.choice.workers});
    uint64_t key=feed(magic,p.terminal.n);
    for(uint64_t x:{i.storage_bytes,p.in,p.ring,p.terminal.bytes,product::window_choice_identity(p.terminal.choice),
                    tq.consumer_info.mul.arithmetic_id,ni.plan_id})key=feed(key,x);
    i.plan_id=key;return true;
}
// Complete storage layout from the queried recipes. Returns false on size overflow.
bool layout(Plan &p, const Queried *uq_data, const Queried *tq_data, const sbn3_newton_info *ninfo) {
    if(p.terminal.n)return tq_data&&ninfo&&fused_layout(p,*tq_data,*ninfo);
    auto &i = p.info;
    const size_t dn = p.request.denominator_limbs, nn = p.request.numerator_limbs, in = p.in;
    if(i.algorithm!=SBN3_DIVREM_BARRETT || p.local_block){
        divrem_prepared::Plan compact;
        divrem_prepared::quote(nn,dn,i.algorithm,p.local_block,p.options.timing,compact,i,true,p.options.reuse_hint,p.local_fused);
        p.in=i.inverse_limbs;p.ring=i.ring_limbs;
        return true;
    }
    size_t cursor = aligned(sizeof(Binding), 128);
    i.control_bytes = cursor;
    i.table_bytes = i.product_workspace_bytes = i.spectrum_bytes = i.scratch_bytes = 0;
    size_t max_align = page;
    auto place = [&](size_t bytes, size_t alignment, size_t &offset) {
        size_t at = 0, end = 0;
        if (!align_size(cursor, alignment, at) || !add_size(at, bytes, end))
            return false;
        offset = at;
        cursor = end;
        max_align = std::max(max_align, alignment);
        return true;
    };
    if (i.algorithm == SBN3_DIVREM_BARRETT && !p.local_block) {
        require(uq_data && tq_data && ninfo,SBN3_FATAL_MATH,"division product layout");
        const auto &uq=*uq_data,&tq=*tq_data;
        p.persistent_offset = aligned(cursor, page);
        cursor = p.persistent_offset;
        if (!place(bytes_for(dn, 8), 128, p.dn_offset) ||
            !place(bytes_for(in + 1, 8), 128, p.u_offset))
            return false;
        p.d_offset = p.dn_offset; // only the normalized divisor persists
        // No spectrum or product binding is live during the inverse. Its
        // workspace may occupy their future locations, after retained D/U.
        p.newton_bytes=ninfo->storage_bytes;
        p.newton_alignment=std::max(ninfo->storage_alignment,page);
        p.newton_offset=aligned(cursor,p.newton_alignment);
        if (uq.cached && !place(uq.future.storage_bytes, page, p.uspec_offset))
            return false;
        if (tq.cached && !place(tq.future.storage_bytes, page, p.dspec_offset))
            return false;
        i.spectrum_bytes = (uq.cached ? uq.future.storage_bytes : 0) + (tq.cached ? tq.future.storage_bytes : 0);
        i.persistent_bytes = cursor - p.persistent_offset;
        // The inverse completes before either product is bound. Its entire
        // binding can share storage with the later product tables, mutable
        // workspaces and block scratch; only D/U/spectra persist outside it.
        p.shared_offset = aligned(cursor, p.newton_alignment);
        cursor = p.shared_offset;
        const auto &um = uq.consumer_info.mul, &tm = tq.consumer_info.mul;
        // Cached consumers borrow their immutable roots from the spectra.
        // Rebinding only their small control/codec state lets the two serial
        // products share one mutable workspace. Use it when it saves at least
        // one of the existing large aligned allocations, keeping tiny products
        // bound throughout to avoid adding a binder to their inner loop.
        const size_t work_alignment = std::max({um.workspace_alignment,tm.workspace_alignment,page});
        const size_t planned_head=i.quotient_limbs%in<=p.head?i.quotient_limbs%in:0;
        const bool one_use=p.options.workers==1&&p.options.reuse_hint<=1&&i.quotient_limbs-planned_head<=in;
        p.shared_products = ((uq.cached && tq.cached)||one_use) &&
                            std::min(um.workspace_bytes,tm.workspace_bytes)>=(size_t(1)<<17);
        if (p.shared_products) {
            i.table_bytes=std::max(um.table_bytes,tm.table_bytes);
            i.product_workspace_bytes=std::max(um.workspace_bytes,tm.workspace_bytes);
            if(!place(i.table_bytes,page,p.utab_offset) ||
               !place(i.product_workspace_bytes,work_alignment,p.uwork_offset))return false;
            p.ttab_offset=p.utab_offset;p.twork_offset=p.uwork_offset;
        } else {
            if (!place(um.table_bytes, page, p.utab_offset) ||
                !place(um.workspace_bytes, std::max(um.workspace_alignment, page), p.uwork_offset) ||
                !place(tm.table_bytes, page, p.ttab_offset) ||
                !place(tm.workspace_bytes, std::max(tm.workspace_alignment, page), p.twork_offset))
                return false;
            i.table_bytes = um.table_bytes + tm.table_bytes;
            i.product_workspace_bytes = um.workspace_bytes + tm.workspace_bytes;
        }
        p.xlen = p.ring ? p.ring : dn + in;
        if (p.head >= in)
            return false;
        p.tcap = std::max(sbn3_mul_output_capacity(&tm), tm.output_limbs);
        p.pcap = std::max(sbn3_mul_output_capacity(&um), um.output_limbs);
        // One integer product buffer covers estimate -> residual -> lift,
        // and the short exact head even when the ring ends at dn itself.
        const size_t output_capacity=std::max({p.tcap,p.pcap,p.ring? p.ring+1:0,dn+p.head});
        if (!place(bytes_for(output_capacity, 8), page, p.tbuf_offset) ||
            !place(bytes_for(in, 8), page, p.ubuf_offset))
            return false;
        p.pbuf_offset=p.tbuf_offset;
        p.xbuf_offset = p.tbuf_offset; // product T becomes the signed residual, after its last read
        i.scratch_bytes = cursor - p.tbuf_offset;
        const size_t newton_end = aligned(p.newton_offset + p.newton_bytes, page);
        cursor = std::max(cursor, newton_end);
        max_align = std::max(max_align, p.newton_alignment);
        i.shared_bytes = cursor - p.shared_offset;
        const size_t head = i.quotient_limbs % in;
        i.head_limbs = head <= p.head ? head : 0;
        i.blocks = unsigned(i.quotient_limbs / in + (head > p.head));
        i.products = 2;
        i.workers = std::max({1u, p.u.workers, p.t.workers, ninfo->workers});
        i.block_limbs = in;
        i.inverse_limbs = in;
        i.ring_limbs = p.ring;
    }
    i.storage_alignment = max_align;
    i.storage_bytes = aligned(cursor, page);
    uint64_t key = feed(1469598103934665603ULL, i.algorithm);
    for (uint64_t v : {i.storage_bytes, i.storage_alignment, i.control_bytes, i.persistent_bytes, i.shared_bytes,
                       i.table_bytes, i.product_workspace_bytes, i.spectrum_bytes, i.scratch_bytes, p.xlen, p.head, p.tcap,
                       p.pcap, p.newton_offset, p.newton_bytes, uint64_t(p.shared_products),p.local_block,p.in,p.ring,
                       uq_data?uq_data->consumer_info.mul.arithmetic_id:0,tq_data?tq_data->consumer_info.mul.arithmetic_id:0,
                       uq_data?uq_data->consumer_info.mul.execution_id:0,tq_data?tq_data->consumer_info.mul.execution_id:0,
                       ninfo ? ninfo->plan_id : 0})
        key = feed(key, v);
    i.plan_id = key;
    return true;
}
Binding &binding(sbn3_divrem_binding *p) {
    auto *b = reinterpret_cast<Binding *>(p);
    require(b && b->run == ::execute_general, SBN3_FATAL_LIFETIME, "division binding");
    return *b;
}
void idle(const Binding &b) {
    require(pthread_equal(b.team->creator, pthread_self()) && !b.team->busy, SBN3_FATAL_TEAM,
            "division owner/idle");
}
uint8_t *at(const Binding &b, size_t offset) {
    return b.arena->base + b.offset + offset;
}
void check_span(const Binding &b, const void *data, size_t bytes, const char *where) {
    valid_span(data, bytes, where);
    require(!(reinterpret_cast<uintptr_t>(data) & 7), SBN3_FATAL_ARGUMENT, where);
    require(!overlaps(data, bytes, b.arena->base + b.offset, b.plan.info.storage_bytes), SBN3_FATAL_ARGUMENT,
            "division value/storage overlap");
    for (unsigned j = 0; j < b.team->width; ++j) {
        const auto &l = j ? b.team->stacks[j] : b.team->storage;
        require(!overlaps(data, bytes, l.data, l.bytes), SBN3_FATAL_ARGUMENT, "division value/team overlap");
    }
}
void unbind_products(Binding &b) {
    if (b.uprod) {
        sbn3_mul_unbind(b.uprod);
        b.uprod = nullptr;
    }
    if (b.tprod) {
        sbn3_mul_unbind(b.tprod);
        b.tprod = nullptr;
    }
    for (auto *l : {&b.utab, &b.uwork, &b.ttab, &b.twork})
        if (l->token) {
            b.arena->release(*l);
            *l = {};
        }
}
sbn3_mul_binding *product_binding(Binding &b,bool inverse) {
    auto *&slot=inverse?b.uprod:b.tprod;
    if(slot)return slot;
    if(b.plan.shared_products)unbind_products(b);
    const auto &q=inverse?b.uq:b.tq;
    const auto &i=q.consumer_info.mul;
    auto &table=inverse?b.utab:b.ttab;
    auto &work=inverse?b.uwork:b.twork;
    table=b.arena->acquire(b.offset+(inverse?b.plan.utab_offset:b.plan.ttab_offset),i.table_bytes);
    work=b.arena->acquire(b.offset+(inverse?b.plan.uwork_offset:b.plan.twork_offset),i.workspace_bytes);
    sbn3_product_bind(&q.consumer,b.arena,&table,&work,b.team,inverse?b.uspec:b.dspec,nullptr,&slot);
    return slot;
}
void release_products(Binding &b) {
    unbind_products(b);
    if(b.phase_output.token){b.arena->release(b.phase_output);b.phase_output={};}
    if (b.uspec) {
        sbn3_spectrum_release(b.uspec);
        b.uspec = nullptr;
        b.arena->release(b.uspec_lease);
    }
    if (b.dspec) {
        sbn3_spectrum_release(b.dspec);
        b.dspec = nullptr;
        b.arena->release(b.dspec_lease);
    }
    if (b.shared_leased) {
        b.arena->release(b.shared);
        b.shared = {};
        b.shared_leased = false;
    }
}
// N' = N << shift over [lo, lo+count) of its nn+1 limbs, from the unshifted source.
void shifted_span(sbn3_team *team, uint64_t *dst, const uint64_t *N, size_t nn, size_t lo, size_t count,
                  unsigned shift) {
    parallel_limbs::each(team, count, parallel_limbs::parts(team, count), [&](size_t b, size_t e, unsigned) {
        for (size_t k = b; k < e; ++k) {
            const size_t j = lo + k;
            uint64_t v = j < nn ? N[j] << shift : 0;
            if (shift && j && j-1<nn)
                v |= N[j - 1] >> (64 - shift);
            dst[k] = v;
        }
    });
}
void barrett_prepare(Binding &b, const uint64_t *D) {
    auto &p = b.plan;
    const size_t dn = p.request.denominator_limbs, in = p.in;
    release_products(b);
    b.prepared = false;
    b.shift = unsigned(__builtin_clzll(D[dn - 1]));
    if (b.shift)
        shifted_span(b.team, b.Dn, D, dn, 0, dn, b.shift);
    else
        parallel_limbs::copy(b.team, b.Dn, D, dn);
    // Block inverse of the top in limbs, in the shared union (unleased at this
    // point): bounded U by the local recurrence, or the spectral Newton binding.
    // Both satisfy |U - B^(2in)/Dtop| < 3 with B^in <= U < 2B^in.
    if (inverse_basecase(in,p.options.workers) && !p.newton_bytes) {
        auto scratch=Frame::external(nullptr,0);
        local_inverse_approximate(b.U,b.Dn+dn-in,in,scratch);
    } else if (inverse_basecase(in,p.options.workers)) {
        auto scratch = b.arena->acquire(b.offset + p.newton_offset, p.newton_bytes);
        {
            Frame frame(*b.arena,scratch);
            local_inverse_approximate(b.U,b.Dn+dn-in,in,frame);
        }
        b.arena->release(scratch);
    } else {
        sbn3_newton_binding *inverse = nullptr;
        sbn3_newton_bind(&b.nplan, b.arena, b.offset + p.newton_offset, b.team, &inverse);
        sbn3_newton_inputs inputs{};
        inputs.denominator = {b.Dn + dn - in, in};
        sbn3_newton_execute(inverse, &inputs, {b.U, in + 1});
        sbn3_newton_unbind(inverse);
    }
    require(b.U[in] == 1, SBN3_FATAL_MATH, "division block inverse framing");
    b.dinv = divrem_words::invert_pi1(b.Dn[dn - 1], b.Dn[dn - 2]);
    // Persistent products and divisor-side spectra.
    b.shared = b.arena->acquire(b.offset + (p.terminal.n?p.ubuf_offset:p.tbuf_offset),
                               p.terminal.n?8*(p.terminal.n+1):p.info.scratch_bytes);
    b.shared_leased = true;
    if (b.uq.cached) {
        b.uspec_lease = b.arena->acquire(b.offset + p.uspec_offset, b.uq.future.storage_bytes);
        sbn3_spectrum_reserve_plan(&b.uq.producer, SBN3_SPECTRUM_COLUMNS, b.uq.future.generation, b.arena,
                                   &b.uspec_lease, &b.uspec);
    }
    if (b.tq.cached) {
        b.dspec_lease = b.arena->acquire(b.offset + p.dspec_offset, b.tq.future.storage_bytes);
        sbn3_spectrum_reserve_plan(&b.tq.producer, SBN3_SPECTRUM_COLUMNS, b.tq.future.generation, b.arena,
                                   &b.dspec_lease, &b.dspec);
    }
    if (b.uspec)
        sbn3_spectrum_compute(product_binding(b,true), b.uspec, {b.U, in});
    if (b.dspec)
        sbn3_spectrum_compute(product_binding(b,false), b.dspec, {b.Dn, dn});
    if(p.terminal.n)unbind_products(b);
    b.prepared = true;
}
struct TerminalRun {
    const newton_detail::Bundle *plans;
    sbn3_arena *arena;sbn3_team *team;
    const TerminalRecipe *recipe;size_t base;
    sbn3_lease table{},work{},cache_lease{};
    sbn3_spectrum *cache=nullptr;
    sbn3_mul_binding *binding=nullptr;unsigned slot=UINT32_MAX;
};
void terminal_enter(void *,unsigned) {}
void terminal_leave(void *ptr){auto&s=*static_cast<TerminalRun *>(ptr);if(s.binding){sbn3_mul_unbind(s.binding);s.binding=nullptr;}
    if(s.work.token){s.arena->release(s.work);s.work={};}if(s.table.token){s.arena->release(s.table);s.table={};}s.slot=UINT32_MAX;}
sbn3_mul_binding *terminal_bind(void *ptr,unsigned slot){
    auto&s=*static_cast<TerminalRun *>(ptr);if(s.binding&&s.slot==slot)return s.binding;
    terminal_leave(ptr);const auto&i=s.plans->infos[slot];
    const auto&r=*s.recipe;
    if(!i.cached_mask&&!r.keep_cache&&s.cache){sbn3_spectrum_release(s.cache);s.cache=nullptr;s.arena->release(s.cache_lease);s.cache_lease={};}
    if(i.cached_mask&&!s.cache){s.cache_lease=s.arena->acquire(s.base+r.cache_at,r.cache_bytes);
        sbn3_spectrum_reserve_plan(&s.plans->producer,SBN3_SPECTRUM_COLUMNS,s.plans->future.generation,s.arena,&s.cache_lease,&s.cache);}
    const size_t table=r.keep_cache?r.table_at:aligned(r.cache_at+(i.cached_mask?r.cache_bytes:0),128);
    const size_t work=r.keep_cache?r.work_at:aligned(table+i.mul.table_bytes,i.mul.workspace_alignment);
    if(i.mul.table_bytes)s.table=s.arena->acquire(s.base+table,i.mul.table_bytes);
    if(i.mul.workspace_bytes)s.work=s.arena->acquire(s.base+work,i.mul.workspace_bytes);
    sbn3_product_bind(&s.plans->plans[slot],s.arena,&s.table,&s.work,s.team,i.cached_mask?s.cache:nullptr,nullptr,&s.binding);
    s.slot=slot;return s.binding;
}
sbn3_spectrum *terminal_cache(void *ptr){return static_cast<TerminalRun *>(ptr)->cache;}
const ProductStageOps terminal_ops{terminal_enter,terminal_bind,terminal_cache,terminal_leave};
void run_terminal(Binding&b) {
    const auto&p=b.plan;const auto&r=p.terminal;const size_t dn=p.request.denominator_limbs;
    if(r.local){auto frame=Frame::external(at(b,p.phase_offset),r.bytes);
        local_divide_terminal(b.ubuf,b.ubuf,b.Dn+dn-r.n,b.U,r.n,frame,local_terminal_recipe(r.local));return;}
    const size_t base=b.offset+p.phase_offset;
    auto values=b.arena->acquire(base,r.cache_at);
    TerminalRun run{b.terminal_plans,b.arena,b.team,&r,base};
    const ProductStage stage{&run,0,&terminal_ops};
    DivideTerminal terminal{p.in,r.n,r.choice.ring,nullptr,nullptr,nullptr,
        {static_cast<uint64_t *>(values.data),reinterpret_cast<uint64_t *>(at(b,p.phase_offset+r.y_at))},b.team,r.bn,&stage,r.capacity,bool(r.keep_cache)};
    terminal.fresh_residual_in_cached=b.terminal_plans->count==1;
    terminal.retain_u_across_residual|=terminal.fresh_residual_in_cached;
    terminal.repair_bytes=b.terminal_plans->repair_bytes;
    divide_terminal(terminal,b.ubuf,b.Dn+dn-r.n,b.U,b.ubuf);
    if(run.cache){sbn3_spectrum_release(run.cache);b.arena->release(run.cache_lease);}
    b.arena->release(values);
}
// One exact quotient limb by word division. x holds X = R*B + n0 on dn+1
// limbs with R < D', so q = floor(X/D') < B. The step follows GMP's sbpi1_div_qr (3/2
// estimate q or q+1, multiply-subtract, add-back; notice in value/divrem_basecase.cpp); each
// team part subtracts its own slice of qhat*D' and the slice carries are stitched exactly.
uint64_t word_step(sbn3_team *team, uint64_t *x, const uint64_t *Dn, size_t dn, uint64_t dinv, uint64_t &addbacks) {
    const uint64_t d1 = Dn[dn - 1], d0 = Dn[dn - 2];
    uint64_t q = UINT64_MAX; // <x[dn],x[dn-1]> == <d1,d0>: the quotient limb is B-1
    if (x[dn] != d1 || x[dn - 1] != d0) {
        uint64_t r1, r0;
        udiv_qr_3by2(q, r1, r0, x[dn], x[dn - 1], x[dn - 2], d1, d0, dinv);
        (void)r1;
        (void)r0;
    }
    const unsigned parts = parallel_limbs::parts(team, dn);
    uint64_t high[32]{};
    parallel_limbs::each(team, dn, parts, [&](size_t b, size_t e, unsigned k) {
        high[k] = e > b ? sbn3i_submul_1(x + b, Dn + b, long(e - b), q) : 0;
    });
    uint64_t borrow = 0;
    for (unsigned k = 0; k < parts; ++k) {
        const size_t e = parallel_limbs::cut(dn, parts, k + 1);
        borrow += parallel_limbs::sub_word(x + e, dn + 1 - e, high[k]);
    }
    // X - qhat*D' > -B^(dn+1): at most one wrap, undone by the add-back.
    require(borrow <= 1, SBN3_FATAL_MATH, "division head borrow");
    for (unsigned k = 0; borrow; ++k) {
        require(k < divrem_tuning::head_correction_limit, SBN3_FATAL_MATH, "division head correction bound");
        --q;
        ++addbacks;
        borrow -= parallel_limbs::add_to(team, x, dn + 1, Dn, dn);
    }
    require(!x[dn], SBN3_FATAL_MATH, "division head remainder");
    return q;
}
uint64_t add_estimate(sbn3_team *team,uint64_t *out,const uint64_t *a,const uint64_t *b,size_t n) {
    const unsigned parts=parallel_limbs::parts(team,n);uint64_t carries[32]{};
    parallel_limbs::each(team,n,parts,[&](size_t begin,size_t end,unsigned k){
        carries[k]=sbn3i_add_n(out+begin,a+begin,b+begin,long(end-begin));
    });
    uint64_t carry=0;
    for(unsigned k=0;k<parts;++k){
        const size_t begin=parallel_limbs::cut(n,parts,k),end=parallel_limbs::cut(n,parts,k+1);
        carry=carries[k]+(carry?parallel_limbs::add_word(out+begin,end-begin,1):0);
    }
    return carry;
}
void fused_execute(Binding&b,const uint64_t*N,size_t nn,uint64_t*Q,uint64_t*R,sbn3_divrem_result&result) {
    const auto&p=b.plan;const size_t dn=p.request.denominator_limbs,qn=nn-dn+1,planned=p.info.quotient_limbs;
    unbind_products(b);if(b.phase_output.token){b.arena->release(b.phase_output);b.phase_output={};}
    shifted_span(b.team,b.ubuf,N,nn,dn,p.terminal.n+1,b.shift);run_terminal(b);
    if(!parallel_limbs::zero(b.team,b.ubuf+qn,p.terminal.n+1-qn)){
        parallel_limbs::fill(b.team,b.ubuf,qn,UINT64_MAX);parallel_limbs::fill(b.team,b.ubuf+qn,p.terminal.n+1-qn);
    }
    b.phase_output=b.arena->acquire(b.offset+p.tbuf_offset,p.tcap*8);
    sbn3_product_inputs in{};if(!b.dspec)in.a={b.Dn,dn};in.b={b.ubuf,planned-1};
    sbn3_product_execute(product_binding(b,false),&in,{b.tbuf,p.tcap});
    if(!p.ring)parallel_limbs::fill(b.team,b.tbuf+dn+planned-1,2);
    const size_t L=p.ring?p.ring+1:p.xlen;
    product::add_shifted_word_product(b.tbuf,L,p.ring,{b.Dn,dn},planned-1,b.ubuf[planned-1],b.team);
    const product::DifferenceValue input{{N,nn},0,nn+1,b.shift,{}};
    const bool negative=product::product_difference(b.tbuf,L,p.ring,input,b.ubuf[0]*b.Dn[0],R,b.team).negative;
    uint64_t corrections=0;
    if(negative){
        while(b.tbuf[dn]||divrem_words::compare(b.tbuf,b.Dn,dn)>0){
            require(++corrections<=8,SBN3_FATAL_MATH,"fused negative correction");parallel_limbs::sub_from(b.team,b.tbuf,L,b.Dn,dn);}
        require(!product::subtract_value(b.team,b.tbuf,L,{{b.Dn,dn},0,dn,0,{}}),SBN3_FATAL_MATH,"fused absolute remainder");
        require(++corrections<=8&&!parallel_limbs::sub_word(b.ubuf,qn,corrections),SBN3_FATAL_MATH,"fused quotient subtract");
    }else{
        while(b.tbuf[dn]||divrem_words::compare(b.tbuf,b.Dn,dn)>=0){
            require(++corrections<=8,SBN3_FATAL_MATH,"fused positive correction");parallel_limbs::sub_from(b.team,b.tbuf,L,b.Dn,dn);}
        require(!parallel_limbs::add_word(b.ubuf,qn,corrections),SBN3_FATAL_MATH,"fused quotient add");
    }
    parallel_limbs::copy(b.team,Q,b.ubuf,qn);parallel_limbs::copy(b.team,R,b.tbuf,dn);
    if(b.shift)parallel_limbs::right_shift(b.team,R,dn,dn,b.shift);
    result.corrections=corrections;b.metrics.products_executed=4;
    unbind_products(b);b.arena->release(b.phase_output);b.phase_output={};
}
void barrett_execute(Binding &b, const uint64_t *N, size_t nn, uint64_t *Q, uint64_t *R, sbn3_divrem_result &out) {
    auto &p = b.plan;
    auto *team = b.team;
    const size_t dn = p.request.denominator_limbs, in = p.in, L = p.xlen;
    const bool cyclic = p.ring != 0;
    const unsigned s = b.shift;
    const uint64_t *Dn = b.Dn;
    uint64_t *xbuf = b.xbuf, *tbuf = b.tbuf, *ubuf = b.ubuf, *pbuf = b.pbuf, *qbuf = b.ubuf, *rbuf = R;
    // R = top dn limbs of N' (N'[nn] is the shifted-out top); Q < B^qn makes R < D'.
    const size_t qn = nn + 1 - dn;
    const size_t remainder=qn%in;
    const size_t head=remainder<=p.head?remainder:0;
    // A normalized dividend whose top word is below D's top word has a
    // provably zero leading quotient word. Read its remainder directly;
    // do not materialize/copy a dn+1-word X or multiply D by zero.
    const bool zero_head=head==1&&!s&&N[nn-1]<Dn[dn-1];
    shifted_span(team, rbuf, N, nn, qn-size_t(zero_head), dn, s);
    require(divrem_words::compare(rbuf, Dn, dn) < 0, SBN3_FATAL_MATH, "division normalized head");
    uint64_t corrections = 0;
    unsigned products = 0;
    size_t pos = qn;
    // A short first block (qn mod in limbs, e.g. the first quotient limb of
    // 2d/d under a full inverse) is exact word division over
    // X = R*B^head + N'[pos-head, pos); the blocks below are then complete.
    if (head) {
        pos -= head;
        if(zero_head)Q[pos]=0;
        else{
            shifted_span(team, xbuf, N, nn, pos, head, s);
            parallel_limbs::copy(team, xbuf + head, rbuf, dn);
            for (size_t j = head; j-- > 0;)
                Q[pos + j] = word_step(team, xbuf + j, Dn, dn, b.dinv, b.metrics.head_corrections);
            parallel_limbs::copy(team, rbuf, xbuf, dn);
        }
        b.metrics.head_limbs += head;
    }
    while (pos) {
        const size_t ic = std::min(in, pos);
        pos -= ic;
        // U=B^in+u. The known high word is an add, not a transform digit.
        parallel_limbs::copy(team, ubuf, rbuf + dn - ic, ic);
        parallel_limbs::fill(team, ubuf + ic, in - ic);
        sbn3_product_inputs pin{};
        if (!b.uspec)
            pin.a = {b.U, in};
        pin.b = {ubuf, in};
        sbn3_product_execute(product_binding(b,true), &pin, {pbuf, p.pcap});
        ++products;
        if (add_estimate(team,qbuf,pbuf+in,rbuf+dn-ic,ic))
            parallel_limbs::fill(team, qbuf, ic, UINT64_MAX);
        parallel_limbs::fill(team, qbuf + ic, in - ic);
        // T = qhat * D' (mod B^ring-1 on the cyclic recipe).
        sbn3_product_inputs tin{};
        if (!b.dspec)
            tin.a = {Dn, dn};
        tin.b = {qbuf, in};
        sbn3_product_execute(product_binding(b,false), &tin, {tbuf, p.tcap});
        ++products;
        auto *block=Q+pos;
        parallel_limbs::copy(team,block,qbuf,ic);
        // E = R*B^ic + N'block - T, |E| < 7 D' < B^(dn+1).
        // T's output storage becomes E. The U operand has already been read,
        // so reuse ubuf for the shifted low block instead of materializing X.
        shifted_span(team, ubuf, N, nn, pos, ic, s);
        const product::DifferenceValue input{{ubuf,ic},0,ic,0,{rbuf,dn}};
        const bool negative=product::product_difference(xbuf,L+size_t(cyclic),p.ring,input,block[0]*Dn[0],nullptr,team).negative;
        require(parallel_limbs::zero(team, xbuf + dn + 1, L + size_t(cyclic) - dn - 1), SBN3_FATAL_MATH, "division residual certificate");
        unsigned k = 0;
        auto above = [&](bool strict) {
            return xbuf[dn] != 0 || divrem_words::compare(xbuf, Dn, dn) > (strict ? 0 : -1);
        };
        if (negative) {
            // E = -m: add D' until nonnegative; the last step is R = D' - m.
            while (above(true)) {
                require(++k <= divrem_tuning::correction_limit, SBN3_FATAL_MATH, "division correction bound");
                parallel_limbs::sub_from(team, xbuf, dn + 1, Dn, dn);
            }
            require(!sbn3i_sub_n(xbuf, Dn, xbuf, long(dn)), SBN3_FATAL_MATH, "division negative residual");
            ++k;
            require(!parallel_limbs::sub_word(block, ic, k), SBN3_FATAL_MATH, "division quotient underflow");
        } else {
            while (above(false)) {
                require(++k <= divrem_tuning::correction_limit, SBN3_FATAL_MATH, "division correction bound");
                parallel_limbs::sub_from(team, xbuf, dn + 1, Dn, dn);
            }
            if (k)
            require(!parallel_limbs::add_word(block, ic, k), SBN3_FATAL_MATH, "division quotient overflow");
        }
        corrections += k;
        parallel_limbs::copy(team, rbuf, xbuf, dn);
    }
    if (s){
        const unsigned parts=parallel_limbs::parts(team,dn);uint64_t next[32]{};
        for(unsigned k=0;k<parts;++k){const size_t end=parallel_limbs::cut(dn,parts,k+1);next[k]=end<dn?rbuf[end]:0;}
        parallel_limbs::each(team, dn, parts, [&](size_t begin, size_t end, unsigned k) {
            for (size_t j = begin; j < end; ++j)
                R[j] = (rbuf[j] >> s) | ((j+1<end?rbuf[j+1]:next[k]) << (64-s));
        });
    }
    out.corrections = corrections;
    b.metrics.products_executed = products;
}
void save_simple_plan(Plan &,sbn3_divrem_plan *,sbn3_divrem_info *);
void save_plan(Plan &p,Stored &s,sbn3_divrem_plan *out,sbn3_divrem_info *info) {
    if(p.info.algorithm!=SBN3_DIVREM_BARRETT||p.local_block){save_simple_plan(p,out,info);return;}
    p.seal=seal(p);s.plan=p;*info=p.info;
    memset(out,0,sizeof *out);memcpy(out->opaque,&s,sizeof s);
}
void save_simple_plan(Plan &p,sbn3_divrem_plan *out,sbn3_divrem_info *info) {
    divrem_prepared::Plan compact;
    divrem_prepared::quote(p.request.numerator_limbs,p.request.denominator_limbs,p.info.algorithm,
                          p.local_block,p.options.timing,compact,*info,true,p.options.reuse_hint,p.local_fused);
    divrem_prepared::store(compact,out);
}
bool fixed_product(const Plan &p,size_t a,size_t b,bool residual,ProductChoice &c,Queried &q) {
    const size_t head=p.info.quotient_limbs%p.in<=p.head?p.info.quotient_limbs%p.in:0;
    const size_t uses=!p.terminal.n&&p.info.quotient_limbs-head>p.in?2:1;
    const product::RepeatedProductOptions options{p.options.workers,p.options.prime_count,p.options.reuse_hint,
                                                  p.options.residual,uses,bool(p.terminal.n)};
    return product::repeated_product_query(options,a,b,residual,c,q);
}
// One block for a short quotient. Otherwise add one block to the minimum
// needed under a full inverse and distribute the actual quotient capacity.
// This chooses the 3-block 5/2-M(n) balanced construction, keeps inverse
// storage below the full-divisor size, and avoids a redundant head pass for
// a short quotient. There are no candidate plans or fitted cost functions.
size_t general_block_limbs(size_t qn,size_t dn) {
    if(3*qn<=dn)return qn;
    const size_t blocks=(qn+dn-1)/dn+1;
    return (qn+blocks-1)/blocks;
}
bool query_fused_plan(Plan&p,Stored&s) {
    const size_t dn=p.request.denominator_limbs,nn=p.request.numerator_limbs,qn=p.info.quotient_limbs;
    if(nn<=dn||qn+2>dn||p.options.block_limbs||p.options.residual||p.options.algorithm||!product::repeated_product_available())return false;
    p.local_block=0;p.info.algorithm=SBN3_DIVREM_BARRETT;
    p.in=newton_contract::next_precision(qn+2);p.head=0;
    newton_detail::Bundle terminal{};
    if(!terminal_query(qn+2,newton_options(p),p.terminal,terminal))return false;
    if(!fixed_product(p,dn,qn-1,true,p.t,s.tq))return false;
    p.ring=p.t.ring;
    if(inverse_query(p.in,newton_options(p),s.nplan,s.ninfo)!=SBN3_SUPPORTED)return false;
    return layout(p,&s.uq,&s.tq,&s.ninfo);
}
// One arithmetic recipe, then a bounded resource ladder only when necessary.
// Rejected attempts never modify the caller's plan. Explicit block/residual
// options remain hard constraints; no measured cost ranks alternate plans.
sbn3_query_result query_blocks(Plan &p,sbn3_divrem_plan *out,sbn3_divrem_info *info,
                               const sbn3_divrem_info &known_need) {
    if(!product::repeated_product_available())return SBN3_UNSUPPORTED;
    const size_t dn=p.request.denominator_limbs,qn=p.info.quotient_limbs;
    size_t in=p.options.block_limbs?p.options.block_limbs:general_block_limbs(qn,dn);
    if(!in||in>dn)return SBN3_UNSUPPORTED;
    sbn3_divrem_info least=known_need;Stored s{};
    const unsigned attempts=p.options.memory_budget&&!p.options.block_limbs?1+divrem_tuning::budget_halvings:1;
    for(unsigned attempt=0;attempt<attempts;++attempt){
        Plan selected=p;selected.local_block=0;selected.in=in;selected.head=std::min<size_t>(4,in-1);
        if(!fixed_product(selected,in,in,false,selected.u,s.uq)||
           !fixed_product(selected,dn,in,true,selected.t,s.tq)){
            if(!least.storage_bytes)return SBN3_UNSUPPORTED;
            break;
        }
        selected.ring=selected.t.ring;
        auto no=newton_options(selected);
        const auto status=inverse_query(in,no,s.nplan,s.ninfo);
        if(status!=SBN3_SUPPORTED)return status;
        if(!layout(selected,&s.uq,&s.tq,&s.ninfo))return SBN3_QUERY_CAPACITY;
        if(!least.storage_bytes||selected.info.storage_bytes<least.storage_bytes)least=selected.info;
        if(!p.options.memory_budget||selected.info.storage_bytes<=p.options.memory_budget){
            p=selected;save_plan(p,s,out,info);return SBN3_SUPPORTED;
        }
        if(in==1)break;
        in=(in+1)/2;
    }
    // A bounded short quotient may use the linear-time word recipe under a
    // tight budget. Never silently replace an explicit algorithm or block.
    if(!p.options.algorithm&&!p.options.block_limbs&&!p.options.residual&&!p.options.prime_count&&
       (dn<=divrem_tuning::schoolbook_budget_divisor||qn<=divrem_tuning::schoolbook_budget_quotient)){
        Plan word=p;word.info.algorithm=SBN3_DIVREM_SCHOOLBOOK;word.local_block=0;
        if(layout(word,nullptr,nullptr,nullptr)){
            if(!least.storage_bytes||word.info.storage_bytes<least.storage_bytes)least=word.info;
            if(word.info.storage_bytes<=p.options.memory_budget){save_simple_plan(word,out,info);return SBN3_SUPPORTED;}
        }
    }
    *info=least;return SBN3_QUERY_CAPACITY;
}
} // namespace
} // namespace sbn::v3
using namespace sbn::v3;
sbn3_query_result sbn::v3::fused_divrem_query(const sbn3_divrem_request *request,const sbn3_divrem_options *options,
                                            sbn3_divrem_plan *out,sbn3_divrem_info *info) {
    require(request&&out&&info,SBN3_FATAL_ARGUMENT,"fused division query");*info={};
    Plan p{};p.request=*request;p.options=options?*options:sbn3_divrem_options{1,0,0,0,0,0,0,0};
    if(!p.request.denominator_limbs||!p.options.workers||p.options.workers>32||p.options.timing>1||
       p.options.residual||p.options.block_limbs||p.options.algorithm||
       (p.options.prime_count&&(p.options.prime_count<4||p.options.prime_count>10)))return SBN3_UNSUPPORTED;
    if(p.request.denominator_limbs>newton_limits::precision_words||p.request.numerator_limbs>newton_limits::precision_words)return SBN3_QUERY_CAPACITY;
    auto&i=p.info;i.numerator_limbs=p.request.numerator_limbs;i.denominator_limbs=i.remainder_limbs=p.request.denominator_limbs;
    i.quotient_limbs=i.numerator_limbs>=i.denominator_limbs?i.numerator_limbs-i.denominator_limbs+1:0;
    Stored stored{};if(!query_fused_plan(p,stored))return SBN3_UNSUPPORTED;*info=p.info;
    if(p.options.memory_budget&&p.info.storage_bytes>p.options.memory_budget)return SBN3_QUERY_CAPACITY;
    save_plan(p,stored,out,info);return SBN3_SUPPORTED;
}
static __attribute__((noinline)) sbn3_query_result query_general(const sbn3_divrem_request *request,
                                               const sbn3_divrem_options *options, sbn3_divrem_plan *out,
                                               sbn3_divrem_info *info) {
    require(request && out && info, SBN3_FATAL_ARGUMENT, "division query arguments");
    *info = {};
    Plan p{};
    p.request = *request;
    p.options = options ? *options : sbn3_divrem_options{1, 0, 0, 0, 0, 0, 0, 0};
    const size_t dn = request->denominator_limbs, nn = request->numerator_limbs;
    if (!dn || !p.options.workers || p.options.workers > 32 || p.options.timing > 1 || p.options.residual > 2 || p.options.algorithm > SBN3_DIVREM_DC ||
        (p.options.prime_count && (p.options.prime_count < newton_limits::first_ntt_prime_count ||
                                   p.options.prime_count > newton_limits::last_ntt_prime_count)))
        return SBN3_UNSUPPORTED;
    if (nn > newton_limits::precision_words || dn > newton_limits::precision_words)
        return SBN3_QUERY_CAPACITY;
    if(p.options.algorithm==SBN3_DIVREM_DC &&
       (dn<3 || dn>(size_t(1)<<20) || p.options.prime_count || p.options.block_limbs || p.options.residual || !native_available()))
        return SBN3_UNSUPPORTED;
    auto &i = p.info;
    i.numerator_limbs = nn;
    i.denominator_limbs = dn;
    i.quotient_limbs = nn >= dn ? nn - dn + 1 : 0;
    i.remainder_limbs = dn;
    sbn3_divrem_info fused_need{};
    if((dn>local_division_max_limbs||p.options.workers>1)&&
       fused_division_shape(nn,dn,p.options.reuse_hint)&&!p.options.algorithm&&!p.options.block_limbs&&!p.options.residual){
        Plan candidate=p;Stored stored{};
        if(query_fused_plan(candidate,stored)){
            if(!p.options.memory_budget||candidate.info.storage_bytes<=p.options.memory_budget){save_plan(candidate,stored,out,info);return SBN3_SUPPORTED;}
            fused_need=candidate.info;
        }
    }
    // The initial band has a fixed one-shot dispatch. A prepared-service
    // caller pays its binding contract, but never a competing-plan search.
    if((dn<=512 || (dn<=local_division_max_limbs && p.options.workers==1)) && !p.options.algorithm && !p.options.block_limbs && !p.options.residual &&
       !p.options.prime_count){
        i.algorithm=dn<=2?SBN3_DIVREM_WORD:small_division_u52(nn,dn)&&native_available()?SBN3_DIVREM_DC:SBN3_DIVREM_SCHOOLBOOK;
        if(i.algorithm==SBN3_DIVREM_DC){
            p.local_block=local_division_block(nn,dn,p.options.reuse_hint);
            if(p.local_block)i.algorithm=SBN3_DIVREM_BARRETT;
        }
        p.local_fused=p.local_block&&fused_division_shape(nn,dn,p.options.reuse_hint);
        if(!layout(p,nullptr,nullptr,nullptr))return SBN3_QUERY_CAPACITY;
        const size_t budget=p.options.memory_budget;
        if(budget && p.info.storage_bytes>budget){
            auto least=p.info;
            auto try_simple=[&](unsigned algorithm,size_t block){
                Plan q{};q.request=p.request;q.options=p.options;
                q.info.numerator_limbs=nn;q.info.denominator_limbs=dn;q.info.quotient_limbs=i.quotient_limbs;q.info.remainder_limbs=dn;
                q.info.algorithm=algorithm;q.local_block=block;
                if(!layout(q,nullptr,nullptr,nullptr))return false;
                if(q.info.storage_bytes<least.storage_bytes)least=q.info;
                if(q.info.storage_bytes<=budget){p=q;return true;}return false;
            };
            bool fits=false;
            if(p.local_fused)fits=try_simple(SBN3_DIVREM_BARRETT,p.local_block);
            if(!fits&&p.local_block>1)fits=try_simple(SBN3_DIVREM_BARRETT,(p.local_block+1)/2);
            if(!fits && p.local_block)fits=try_simple(SBN3_DIVREM_DC,0);
            const size_t excess=i.quotient_limbs>divrem_tuning::schoolbook_budget_quotient?
                                i.quotient_limbs-divrem_tuning::schoolbook_budget_quotient:0;
            if(!fits && (dn<=divrem_tuning::schoolbook_budget_divisor ||
                         double(dn)*double(excess)<=divrem_tuning::schoolbook_budget_work))
                fits=try_simple(SBN3_DIVREM_SCHOOLBOOK,0);
            if(!fits){
                // Resource-constrained callers retain access to the existing
                // smaller-block/residual-family ladder. Ordinary fresh calls
                // never enter this fallback or search its candidates.
                if(dn>=3&&i.quotient_limbs&&native_available()){
                    Plan constrained{};constrained.request=p.request;constrained.options=p.options;
                    constrained.info.numerator_limbs=nn;constrained.info.denominator_limbs=dn;
                    constrained.info.quotient_limbs=i.quotient_limbs;constrained.info.remainder_limbs=dn;
                    constrained.info.algorithm=SBN3_DIVREM_BARRETT;
                    return query_blocks(constrained,out,info,least);
                }
                *info=least;return SBN3_QUERY_CAPACITY;
            }
        }
        *info=p.info;
        if(budget && p.info.storage_bytes>budget)return SBN3_QUERY_CAPACITY;
        save_simple_plan(p,out,info);return SBN3_SUPPORTED;
    }
    if(p.options.algorithm==SBN3_DIVREM_DC)
        i.algorithm=SBN3_DIVREM_DC;
    else if (dn <= 2)
        i.algorithm = SBN3_DIVREM_WORD;
    else if (p.options.algorithm)
        i.algorithm = p.options.algorithm == SBN3_DIVREM_BARRETT && i.quotient_limbs ? SBN3_DIVREM_BARRETT : SBN3_DIVREM_SCHOOLBOOK;
    else if (!i.quotient_limbs || (!p.options.block_limbs&&!p.options.residual&&!p.options.prime_count&&!small_division_u52(nn,dn)))
        i.algorithm = SBN3_DIVREM_SCHOOLBOOK;
    else
        i.algorithm = SBN3_DIVREM_BARRETT;
    const size_t budget=p.options.memory_budget;
    if(i.algorithm==SBN3_DIVREM_BARRETT)
        return query_blocks(p,out,info,fused_need);
    if (!layout(p,nullptr,nullptr,nullptr))return SBN3_QUERY_CAPACITY;
    *info=p.info;
    if(budget && p.info.storage_bytes>budget)return SBN3_QUERY_CAPACITY;
    save_simple_plan(p,out,info);
    return SBN3_SUPPORTED;
}
// A scalar recipe has no transform geometry. Keep its constant-size quote
// out of the function that materializes native/FFT plans so register spills
// and stack frames for those plans cannot become a tiny-call tax.
template<unsigned Algorithm>
static __attribute__((noinline)) sbn3_query_result query_scalar(size_t nn,size_t dn,unsigned timing,
                                                               sbn3_divrem_plan *out,sbn3_divrem_info *info) {
    divrem_prepared::Plan p;
    divrem_prepared::quote(nn,dn,Algorithm,0,timing,p,*info);
    divrem_prepared::store(p,out);
    return SBN3_SUPPORTED;
}
[[gnu::always_inline]] static inline bool compact_algorithm(size_t nn,size_t dn,const sbn3_divrem_options &o,
                                                           unsigned &algorithm) {
    const uint64_t overrides=uint64_t(o.algorithm)|o.residual|o.prime_count|o.block_limbs;
    if(!dn||nn>newton_limits::precision_words||!o.workers||o.workers>32||o.timing>1||
       !(dn<=512||(dn<=local_division_max_limbs&&o.workers==1))||
       overrides)return false;
    algorithm=dn<=2?SBN3_DIVREM_WORD:
        small_division_u52(nn,dn)&&native_available()?SBN3_DIVREM_DC:SBN3_DIVREM_SCHOOLBOOK;
    return true;
}
[[gnu::always_inline]] static inline bool compact_recipe(size_t nn,size_t dn,const sbn3_divrem_options &o,
                                                        unsigned &algorithm,size_t &block) {
    if(!compact_algorithm(nn,dn,o,algorithm))return false;
    block=algorithm==SBN3_DIVREM_DC?local_division_block(nn,dn,o.reuse_hint):0;
    if(block)algorithm=SBN3_DIVREM_BARRETT;
    return true;
}
static __attribute__((noinline)) sbn3_query_result query_compact(
    const sbn3_divrem_request *request,const sbn3_divrem_options *options,sbn3_divrem_plan *out,
    sbn3_divrem_info *info,unsigned algorithm) {
    const auto o=options?*options:sbn3_divrem_options{1,0,0,0,0,0,0,0};
    const size_t nn=request->numerator_limbs,dn=request->denominator_limbs;
    const size_t block=algorithm==SBN3_DIVREM_DC?local_division_block(nn,dn,o.reuse_hint):0;
    if(block)algorithm=SBN3_DIVREM_BARRETT;
    divrem_prepared::Plan p;
    divrem_prepared::quote(nn,dn,algorithm,block,o.timing,p,*info,true,o.reuse_hint,true);
    if(!o.memory_budget||p.bytes<=o.memory_budget){divrem_prepared::store(p,out);return SBN3_SUPPORTED;}
    return query_general(request,options,out,info);
}
template<bool Defaults>
static __attribute__((noinline)) sbn3_query_result query_front(const sbn3_divrem_request *request,
                                               const sbn3_divrem_options *options,sbn3_divrem_plan *out,
                                               sbn3_divrem_info *info) {
    require(request&&out&&info,SBN3_FATAL_ARGUMENT,"division query arguments");
    const auto o=Defaults?sbn3_divrem_options{1,0,0,0,0,0,0,0}:*options;
    const size_t nn=request->numerator_limbs,dn=request->denominator_limbs;
    unsigned algorithm=0;
    if(compact_algorithm(nn,dn,o,algorithm)){
        if(!o.memory_budget){
            if(algorithm==SBN3_DIVREM_WORD)return query_scalar<SBN3_DIVREM_WORD>(nn,dn,o.timing,out,info);
            if(algorithm==SBN3_DIVREM_SCHOOLBOOK)return query_scalar<SBN3_DIVREM_SCHOOLBOOK>(nn,dn,o.timing,out,info);
        }
        return query_compact(request,options,out,info,algorithm);
    }
    return query_general(request,options,out,info);
}
extern "C" sbn3_query_result sbn3_divrem_query(const sbn3_divrem_request *request,
                                               const sbn3_divrem_options *options,sbn3_divrem_plan *out,
                                               sbn3_divrem_info *info) {
    if(!options)return query_front<true>(request,options,out,info);
    return query_front<false>(request,options,out,info);
}
static __attribute__((noinline)) void bind_general(const sbn3_divrem_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                                 sbn3_divrem_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "division bind arguments");
    Stored s{};
    load(*opaque, s);
    Plan p = s.plan;
    require(team->arena == arena && team->width >= p.info.workers && pthread_equal(team->creator, pthread_self()) &&
                !team->busy,
            SBN3_FATAL_TEAM, "division bind team");
    size_t end = 0;
    require(add_size(offset, p.info.storage_bytes, end) && end <= arena->virtual_bytes && arena->contains(offset, end) &&
                !(reinterpret_cast<uintptr_t>(arena->base + offset) & (p.info.storage_alignment - 1)) &&
                arena->unleased(offset, p.info.storage_bytes),
            SBN3_FATAL_WORKSPACE, "division prepared unleased range");
    // Layout replay from the stored recipes proves the plan's own consistency;
    // the product and Newton plans carry their own seals.
    const auto expected = p.info;
    require(layout(p, &s.uq, &s.tq, &s.ninfo) && p.info.plan_id == expected.plan_id &&
                p.info.storage_bytes == expected.storage_bytes,
            SBN3_FATAL_ARGUMENT, "division query/bind equivalence");
    const Queried &uq = s.uq, &tq = s.tq;
    const sbn3_newton_plan &nplan = s.nplan;
    const sbn3_newton_info &ninfo = s.ninfo;
    auto control = arena->acquire(offset, p.info.control_bytes);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->control = control;
    b->uq = uq;
    b->tq = tq;
    b->nplan = nplan;
    b->ninfo = ninfo;
    if(p.terminal.n&&!p.terminal.local){
        b->terminal_plans=::new(reinterpret_cast<unsigned char *>(b)+aligned(sizeof(Binding),128))newton_detail::Bundle{};
        auto recipe=p.terminal;
        require(terminal_query(recipe.n,newton_options(p),recipe,*b->terminal_plans,true),SBN3_FATAL_MATH,"fused terminal replay");
    }
    // Divisor-side values keep their own lease; product tables, workspaces
    // and spectra are claimed exclusively by their bindings at prepare.
    b->D = reinterpret_cast<uint64_t *>(at(*b, p.d_offset));
    b->persistent = arena->acquire(offset + p.d_offset, p.u_offset + bytes_for(p.in + 1, 8) - p.d_offset);
    b->Dn = reinterpret_cast<uint64_t *>(at(*b, p.dn_offset));
    b->U = reinterpret_cast<uint64_t *>(at(*b, p.u_offset));
    b->xbuf = reinterpret_cast<uint64_t *>(at(*b, p.xbuf_offset));
    b->tbuf = reinterpret_cast<uint64_t *>(at(*b, p.tbuf_offset));
    b->ubuf = reinterpret_cast<uint64_t *>(at(*b, p.ubuf_offset));
    b->pbuf = reinterpret_cast<uint64_t *>(at(*b, p.pbuf_offset));
    *out = reinterpret_cast<sbn3_divrem_binding *>(b);
}
extern "C" void sbn3_divrem_bind(const sbn3_divrem_plan *opaque,sbn3_arena *arena,size_t offset,sbn3_team *team,
                                  sbn3_divrem_binding **out) {
    require(opaque,SBN3_FATAL_ARGUMENT,"division plan");
    if(opaque->opaque[0]==divrem_prepared::magic)return divrem_prepared::bind(opaque,arena,offset,team,out);
    bind_general(opaque,arena,offset,team,out);
}
static __attribute__((noinline)) void prepare_general(sbn3_divrem_binding *opaque, sbn3_const_limbs denominator) {
    auto &b = binding(opaque);
    idle(b);
    const auto &p = b.plan;
    const size_t dn = p.request.denominator_limbs;
    require(denominator.count == dn && denominator.data, SBN3_FATAL_ARGUMENT, "division divisor length");
    check_span(b, denominator.data, bytes_for(dn, 8), "division divisor");
    require(denominator.data[dn - 1] != 0, SBN3_FATAL_ARGUMENT, "division divisor top limb / zero divisor");
    const auto start = p.options.timing ? now() : 0;
    barrett_prepare(b, denominator.data);
    ++b.metrics.prepares;
    if (p.options.timing)
        b.metrics.prepare_ns = now() - start;
}
template<size_t Dn>
static __attribute__((noinline)) void prepare_word(sbn3_divrem_binding *opaque,sbn3_const_limbs d) {
    auto &b=*reinterpret_cast<divrem_prepared::Binding *>(opaque);
    const size_t dn=Dn?Dn:b.dn;
    require(d.count==dn&&d.data&&d.data[dn-1],SBN3_FATAL_ARGUMENT,"prepared division divisor");
    auto *data=reinterpret_cast<unsigned char *>(&b);
    ::new(&divrem_prepared::word(b).state)divrem_words::PreparedDivisor(
        divrem_words::prepare(d.data,dn,reinterpret_cast<uint64_t *>(data+
            divrem_prepared::control_bytes(SBN3_DIVREM_WORD,0)),true));
    b.flags=divrem_prepared::ready;++b.prepares;
}
static __attribute__((noinline)) void prepare_compact(sbn3_divrem_binding *opaque,sbn3_const_limbs d) {
    divrem_prepared::prepare(divrem_prepared::get(opaque),d);
}
extern "C" void sbn3_divrem_prepare(sbn3_divrem_binding *opaque,sbn3_const_limbs d) {
    const auto run=division_executor(opaque);
    if constexpr(!SBN3_CHECK_SMALL){
        if(run==divrem_prepared::execute_word<1>)return prepare_word<1>(opaque,d);
        if(run==divrem_prepared::execute_word<2>)return prepare_word<2>(opaque,d);
        if(run==divrem_prepared::execute_word<0>)return prepare_word<0>(opaque,d);
    }
    if(run&&run!=execute_general)return prepare_compact(opaque,d);
    prepare_general(opaque,d);
}
static __attribute__((noinline)) void execute_general(sbn3_divrem_binding *opaque, sbn3_const_limbs numerator, sbn3_limbs quotient,
                                    sbn3_limbs remainder, sbn3_divrem_result *result) {
    auto &b = binding(opaque);
    idle(b);
    require(b.prepared, SBN3_FATAL_LIFETIME, "division divisor not prepared");
    const auto &p = b.plan;
    const size_t dn = p.request.denominator_limbs, nn = numerator.count;
    require(nn <= p.request.numerator_limbs && (!nn || numerator.data) && remainder.data &&
                remainder.capacity >= dn && result,
            SBN3_FATAL_ARGUMENT, "division execute spans");
    const size_t qcount = nn >= dn ? nn - dn + 1 : 0;
    require(!qcount || (quotient.data && quotient.capacity >= qcount), SBN3_FATAL_ARGUMENT, "division quotient capacity");
    const size_t nb = bytes_for(nn, 8), qb = bytes_for(qcount, 8), rb = bytes_for(dn, 8);
    if (nn)
        check_span(b, numerator.data, nb, "division numerator");
    if (qcount)
        check_span(b, quotient.data, qb, "division quotient");
    check_span(b, remainder.data, rb, "division remainder");
    require(!overlaps(numerator.data, nb, quotient.data, qb) && !overlaps(numerator.data, nb, remainder.data, rb) &&
                !overlaps(quotient.data, qb, remainder.data, rb),
            SBN3_FATAL_ARGUMENT, "division value overlap");
    const auto start = p.options.timing ? now() : 0;
    *result = {};
    size_t nn_eff = nn;
    while (nn_eff && !numerator.data[nn_eff - 1])
        --nn_eff;
    size_t written = 0;
    if (nn_eff < dn) {
        if(nn_eff)memcpy(remainder.data, numerator.data, nn_eff * 8);
        memset(remainder.data + nn_eff, 0, (dn - nn_eff) * 8);
        b.metrics.products_executed = 0;
    } else {
        written = nn_eff - dn + 1;
        if(p.terminal.n)fused_execute(b,numerator.data,nn_eff,quotient.data,remainder.data,*result);
        else barrett_execute(b, numerator.data, nn_eff, quotient.data, remainder.data, *result);
    }
    if (qcount > written)
        memset(quotient.data + written, 0, (qcount - written) * 8);
    size_t qlen = written;
    while (qlen && !quotient.data[qlen - 1])
        --qlen;
    size_t rlen = dn;
    while (rlen && !remainder.data[rlen - 1])
        --rlen;
    result->quotient_limbs = qlen;
    result->remainder_limbs = rlen;
    b.metrics.corrections += result->corrections;
    ++b.metrics.executes;
    if (p.options.timing)
        b.metrics.execute_ns = now() - start;
}
// Untimed scalar services must not inherit the registers/stack for transform
// execution or clock collection. DN=1/2 also removes unused generic branches.
template<size_t Dn>
__attribute__((noinline)) void sbn::v3::divrem_prepared::execute_word(sbn3_divrem_binding *opaque,sbn3_const_limbs n,
                                                  sbn3_limbs q,sbn3_limbs r,sbn3_divrem_result *result) {
    auto &b=*reinterpret_cast<divrem_prepared::Binding *>(opaque);
    auto &v=divrem_prepared::word(b);
    const size_t qs=divrem_words::schoolbook_prepared<Dn>(q.data,r.data,n.data,n.count,v.state,v.scratch);
    size_t rs=Dn?Dn:b.dn;
    while(rs&&!r.data[rs-1])--rs;
    *result={qs,rs,0};++b.executes;
}
__attribute__((noinline)) void sbn::v3::divrem_prepared::execute_compact(sbn3_divrem_binding *opaque,sbn3_const_limbs n,
                                                     sbn3_limbs q,sbn3_limbs r,sbn3_divrem_result *result) {
    divrem_prepared::execute(divrem_prepared::get(opaque),n,q,r,result);
}
extern "C" void sbn3_divrem_execute(sbn3_divrem_binding *opaque,sbn3_const_limbs n,sbn3_limbs q,
                                     sbn3_limbs r,sbn3_divrem_result *result) {
    const auto run=division_executor(opaque);
    if constexpr(SBN3_CHECK_SMALL)require(run,SBN3_FATAL_LIFETIME,"division executor");
    run(opaque,n,q,r,result);
}
extern "C" void sbn3_int_divrem_execute(sbn3_divrem_binding *opaque, sbn3_int *q, sbn3_int *r, sbn3_int_view n,
                                        sbn3_int_view d) {
    if(compact_binding(opaque)){
        auto &b=divrem_prepared::get(opaque);
        require(q&&r&&n.negative<=1&&d.negative<=1&&divrem_prepared::is_prepared(b),SBN3_FATAL_ARGUMENT,"integer division arguments");
        while(d.size&&!d.data[d.size-1])--d.size;
        while(n.size&&!n.data[n.size-1])--n.size;
        require(divrem_prepared::same_divisor(b,d),SBN3_FATAL_ARGUMENT,"integer prepared divisor");
        sbn3_divrem_result result;
        b.run(opaque,{n.data,n.size},{q->data,q->capacity},{r->data,r->capacity},&result);
        q->size=result.quotient_limbs;q->negative=q->size?(n.negative^d.negative):0;
        r->size=result.remainder_limbs;r->negative=r->size?n.negative:0;return;
    }
    auto &b = binding(opaque);
    require(q && r && n.negative <= 1 && d.negative <= 1 && b.prepared, SBN3_FATAL_ARGUMENT, "integer division arguments");
    const size_t dn = b.plan.request.denominator_limbs;
    while (d.size && !d.data[d.size - 1])
        --d.size;
    while (n.size && !n.data[n.size - 1])
        --n.size;
    bool same = d.size == dn;
    if (same && b.plan.info.algorithm == SBN3_DIVREM_BARRETT && b.shift) {
        uint64_t difference = 0, carry = 0;
        for (size_t j = 0; j < dn; ++j) {
            const uint64_t word = d.data[j];
            difference |= ((word << b.shift) | carry) ^ b.Dn[j];
            carry = word >> (64 - b.shift);
        }
        same = !difference && !carry;
    } else if (same) same = !memcmp(d.data, b.D, dn * 8);
    require(same, SBN3_FATAL_ARGUMENT, "integer division prepared divisor");
    const size_t qn = n.size >= dn ? n.size - dn + 1 : 0;
    require(q->capacity >= qn && r->capacity >= dn, SBN3_FATAL_SIZE, "integer division capacity");
    sbn3_divrem_result out{};
    sbn3_divrem_execute(opaque, {n.data, n.size}, {q->data, q->capacity}, {r->data, r->capacity}, &out);
    q->size = out.quotient_limbs;
    q->negative = out.quotient_limbs ? (n.negative ^ d.negative) : 0;
    r->size = out.remainder_limbs;
    r->negative = out.remainder_limbs ? n.negative : 0;
}
extern "C" void sbn3_divrem_get_metrics(const sbn3_divrem_binding *opaque, sbn3_divrem_metrics *out) {
    if(compact_binding(opaque)){
        require(out,SBN3_FATAL_ARGUMENT,"division metrics");
        *out=divrem_prepared::metrics(divrem_prepared::get(const_cast<sbn3_divrem_binding *>(opaque)));return;
    }
    const auto &b = binding(const_cast<sbn3_divrem_binding *>(opaque));
    require(out, SBN3_FATAL_ARGUMENT, "division metrics");
    *out = b.metrics;
}
static __attribute__((noinline)) void unbind_general(sbn3_divrem_binding *opaque) {
    auto &b = binding(opaque);
    idle(b);
    release_products(b);
    auto *arena = b.arena;
    const auto control = b.control;
    arena->release(b.persistent);
    b.run = nullptr;
    b.~Binding();
    arena->release(control);
}

extern "C" void sbn3_divrem_unbind(sbn3_divrem_binding *opaque) {
    if(compact_binding(opaque))
        return divrem_prepared::unbind(divrem_prepared::get(opaque));
    unbind_general(opaque);
}

static __attribute__((noinline)) sbn3_divrem_binding *prepare_into_general(
    const sbn3_divrem_request *request,const sbn3_divrem_options *options,sbn3_arena *arena,
    size_t offset,size_t bytes,sbn3_team *team,sbn3_const_limbs divisor) {
    sbn3_divrem_plan plan;
    sbn3_divrem_info info;
    require(sbn3_divrem_query(request,options,&plan,&info)==SBN3_SUPPORTED,
            SBN3_FATAL_ARGUMENT,"division constructor request");
    require(bytes>=info.storage_bytes,SBN3_FATAL_WORKSPACE,"division constructor storage",info.storage_bytes,bytes);
    sbn3_divrem_binding *result;
    sbn3_divrem_bind(&plan,arena,offset,team,&result);
    sbn3_divrem_prepare(result,divisor);
    return result;
}
template<bool Defaults>
static __attribute__((noinline)) sbn3_divrem_binding *prepare_into_impl(
    const sbn3_divrem_request *request,const sbn3_divrem_options *options,sbn3_arena *arena,
    size_t offset,size_t bytes,sbn3_team *team,sbn3_const_limbs divisor) {
    require(request,SBN3_FATAL_ARGUMENT,"division constructor request");
    const auto o=Defaults?sbn3_divrem_options{1,0,0,0,0,0,0,0}:*options;
    unsigned algorithm=0;size_t block=0;
    if(compact_recipe(request->numerator_limbs,request->denominator_limbs,o,algorithm,block)){
        divrem_prepared::Plan p;
        sbn3_divrem_info unused;
        divrem_prepared::quote(request->numerator_limbs,request->denominator_limbs,algorithm,block,o.timing,p,unused,false,o.reuse_hint,true);
        if(!o.memory_budget||p.bytes<=o.memory_budget){
            require(bytes>=p.bytes,SBN3_FATAL_WORKSPACE,"division constructor storage",p.bytes,bytes);
            sbn3_divrem_binding *result;
            if constexpr(!SBN3_CHECK_SMALL){
                if(algorithm<=SBN3_DIVREM_SCHOOLBOOK&&!o.timing){
                    require(arena&&team&&team->arena==arena,SBN3_FATAL_TEAM,"division constructor context");
                    require(divisor.count==p.dn&&divisor.data&&divisor.data[p.dn-1],
                            SBN3_FATAL_ARGUMENT,"division constructor divisor");
                    auto *data=arena->base+offset;
                    require(!(uintptr_t(data)&127),SBN3_FATAL_WORKSPACE,"division constructor alignment");
                    // Build D before publishing header stores: no store/load
                    // dependency through an empty binding, and no false alias
                    // dependency between those stores and the source D.
                    const auto state=divrem_words::prepare(divisor.data,p.dn,
                        reinterpret_cast<uint64_t *>(data+divrem_prepared::control_bytes(algorithm,o.timing)),true);
                    auto *b=::new(data)divrem_prepared::WordBinding;
                    ::new(&b->state)divrem_words::PreparedDivisor(state);
                    b->scratch=reinterpret_cast<uint64_t *>(data+p.work_at);
                    divrem_prepared::initialize(b->head,p.nn,p.dn,p.bytes,algorithm,divrem_prepared::ready,team,0);
                    return reinterpret_cast<sbn3_divrem_binding *>(b);
                }
            }
            divrem_prepared::bind_plan(p,arena,offset,team,&result);
            divrem_prepared::prepare(divrem_prepared::get(result),divisor);
            return result;
        }
    }
    return prepare_into_general(request,options,arena,offset,bytes,team,divisor);
}

extern "C" sbn3_divrem_binding *sbn3_divrem_prepare_into(
    const sbn3_divrem_request *request,const sbn3_divrem_options *options,sbn3_arena *arena,
    size_t offset,size_t bytes,sbn3_team *team,sbn3_const_limbs divisor) {
    if(!options)return prepare_into_impl<true>(request,options,arena,offset,bytes,team,divisor);
    return prepare_into_impl<false>(request,options,arena,offset,bytes,team,divisor);
}
