#include "product/spectrum_contract.hpp"
#include "common/identity.hpp"
/* Shared native p48 LIN instance. Mathematical kernels retain the cr implementation;
 * this TU supplies the pure query and the explicit resource-binding recipe. */
#include "product/backend.hpp"
#include "product/tuning_native.hpp"
#include "tuning/native_policy.hpp"
#include "runtime/team.hpp"
#include "backend/ntt_p48/engine.hpp"
#include "backend/ntt_p48/flat.hpp"
#include <new>
#include <type_traits>
#include <numeric>
#include <initializer_list>
using namespace sbn::v3;
namespace {
namespace p=sbn::v3::SBN3_P48_NS;
constexpr unsigned PN=p::NP;
constexpr unsigned LEAF=SBN3_P48_LEAF;
// Minimal rows owning complete 64-byte lines in each blocked plane.
constexpr unsigned stream_row_grain=64/std::gcd(64u,unsigned(TB)*unsigned(SLOT));
static_assert((stream_row_grain*unsigned(TB)*unsigned(SLOT))%64==0);
static_assert(LEAF>=5 && LEAF<=8);
constexpr uint64_t plan_magic=0x53424e33504c4e31ULL,binding_magic=0x53424e3342494e44ULL;
constexpr size_t work_alignment=2u<<20;
struct BasisKey {uint64_t id;};
/* Materialized row slots, not a claim that uncomputed TFT evaluations are zero.
 * [defined_slots,padded_slots) is physical zero padding only. */
struct StateDesc {size_t defined_slots,padded_slots;unsigned frontier;uint64_t scale[PN];};
struct ArithmeticPlan {
    p::Plan transform{};
    BasisKey basis{};
    StateDesc rows_a{},rows_b{},product{};
    uint64_t scale_a[PN]{};
};
struct ExecutionPlan {
    sbn3_mul_info info{};
    p::XposeCtx transpose_a{},transpose_b{}; // address fields remain NULL in a plan
    size_t pitch_alignment=64,tail_bytes=0,journal_bytes=0,spill_bytes=0;
    unsigned emit_tasks=0;
};
struct Recipe {
    unsigned kind=0,cached_mask=0;
    size_t lengths[4]{};
    sbn3_spectrum_desc cached[2]{};
    sbn3_window_certificate window{};
    size_t spectrum_bytes=0;
    uint64_t k[2][PN]{},kr[2][PN]{};
    bool scaled=false;
};
struct ProductPlan {
    uint64_t magic=plan_magic,backend_id=PN,seal=0;
    sbn3_product_spec product{};
    Recipe recipe{};
    sbn3_mul_options options{};
    ArithmeticPlan arithmetic{};
    ExecutionPlan execution{};
};
static_assert(sizeof(ProductPlan)<=sizeof(sbn3_mul_plan));
static_assert(std::is_trivially_copyable_v<ProductPlan>);
constexpr uint64_t spectrum_magic=0x53424e3353504543ULL;
struct Spectrum {
    sbn3_spectrum header{&SBN3_P48_BACKEND(),spectrum_magic};
    Arena *owner=nullptr;
    sbn3_lease storage{};
    uint64_t refs=1;
    uint32_t state=spectrum_contract::reserved; // 0 reserved, 1 computing, 2 immutable completed spectrum
    sbn3_spectrum_desc desc{};
    p::Primes primes{};
    uint8_t *planes[PN]{};
};
const Spectrum &spectrum(const sbn3_spectrum *s){return *reinterpret_cast<const Spectrum *>(s);}
void spectrum_retain(const sbn3_spectrum *);
void spectrum_release(const sbn3_spectrum *);
struct RunBindings {
    Arena *arena;
    sbn3_team *team;
    sbn3_lease table_memory,work_memory;
    p::Primes *primes=nullptr;
    p::FlatConstants *flat_constants=nullptr;
#ifdef CR_FLAT_SHARE_TABLES
    p::Primes *original_primes=nullptr;
    p::FlatConstants *original_flat_constants=nullptr;
#endif
    Frame *workers[32]{};
    uint8_t *a_planes[PN]{},*b_planes[PN]{},*spares[PN]{};
    uint8_t *a1_planes[PN]{},*b1_planes[PN]{},*pool=nullptr;
    const sbn3_spectrum *cached[2]{};
    uint8_t *transpose=nullptr;
    uint64_t *tail=nullptr;
    uint8_t *journal=nullptr;
    uint64_t (*spill)[8]=nullptr;
};
struct Binding {
    sbn3_mul_binding header;
    ProductPlan plan;
    RunBindings run;
    Frame root;
    p::Ctx context{};
    p::Garner garner{};
    p::EmitTabs emit_tables{};
    p::CcxGate gates[4]{};
    alignas(Frame) unsigned char worker_objects[32*sizeof(Frame)];
    size_t table_used=0,work_used=0;
    uint64_t executions=0;
    uint32_t active=0;
    uint64_t last_stage_ns[4]{};
    p::PassCounts counts[32]{};
#ifdef CR_FLAT_PROFILE
    uint64_t flat_phase[PN][8]{};
#endif
    Binding(const ProductPlan &q,Arena &a,const sbn3_lease &t,const sbn3_lease &w,sbn3_team &team)
        : header{&SBN3_P48_BACKEND(),binding_magic},plan(q),run{&a,&team,t,w},root(a,w) {}
    Binding(const ProductPlan &q,Frame &parent,void *data,sbn3_team &team)
        : header{&SBN3_P48_BACKEND(),binding_magic},plan(q),
          run{team.arena,&team,{}, {data,q.execution.info.workspace_bytes,0}},
          root(parent.borrowed_view(data,q.execution.info.workspace_bytes)) {}
};
static_assert(std::is_standard_layout_v<Binding>);

/* The same allocation order and alignment is used during bind. */
struct Sizer {
    size_t at=0;
    bool ok=true;
    void take(size_t bytes,size_t alignment=64) {
        size_t pos=0,end=0;
        if(!ok || !align_size(at,alignment,pos) || !add_size(pos,bytes,end)){ok=false;return;}
        at=end;
    }
    void words(size_t n,size_t width=8) {size_t bytes=0;if(!mul_size(n,width,bytes)){ok=false;return;}take(bytes);}
    void line_assembler(size_t n) {take(64*n+64);words(n);take(n+64);}
};
uint64_t feed(uint64_t h, uint64_t value) { return identity::word(h, value); }
uint64_t basis_id(const p::Plan &g) {
    uint64_t h=1469598103934665603ULL;
    for(unsigned k=0;k<PN;++k)h=feed(h,p::PR[k]);
    for(uint64_t x:{uint64_t(PN),uint64_t(g.T),g.N,g.C,g.M2,uint64_t(48),uint64_t(TB)})h=feed(h,x);
    return h;
}
uint64_t descriptor_seal(const sbn3_spectrum_desc &d) {
    uint64_t h=1469598103934665603ULL;
    for(uint64_t v:{d.basis_id,d.instance_id,d.generation,uint64_t(d.np),uint64_t(d.trunk_bits),uint64_t(d.frontier),
        uint64_t(d.format_version),d.C,d.M2,d.transform_trunks,d.live_slots,d.written_slots,d.source_limbs,
        d.source_trunks,d.block_stride,d.storage_bytes,d.table_bytes,d.plane_bytes})h=feed(h,v);
    for(auto v:d.scale)h=feed(h,v);h=feed(h,d.backend_id);h=feed(h,d.codec_mode);return h;
}
bool flat_packed_spectrum(const p::Plan &g,unsigned workers) {
    // Large serial cyclic spectra are read once per use; packing saves a
    // quarter of their plane footprint. Small/parallel spectra keep lazy64.
    return g.C==1&&g.ring_rn&&workers==1&&g.N>=native_policy::flat_spectrum_pack_trunks;
}
bool flat_spectrum_format(unsigned format) {
    return format==spectrum_contract::flat_lazy64_format||format==spectrum_contract::flat_packed48_format;
}
unsigned spectrum_format(const ProductPlan &q) {
    if(q.execution.info.algorithm!=SBN3_MUL_FLAT)return spectrum_contract::blocked48_format;
    // A consumer that is also a cache builder keeps its reserved input
    // representation, including when its team differs from the producer's.
    if(q.recipe.cached_mask&1)return q.recipe.cached[0].format_version;
    return flat_packed_spectrum(q.arithmetic.transform,q.execution.info.workers)?
        spectrum_contract::flat_packed48_format:spectrum_contract::flat_lazy64_format;
}
bool compatible(const sbn3_spectrum_desc &d,const p::Plan &g,size_t limbs) {
    if(d.seal!=descriptor_seal(d) || d.backend_id!=PN || d.codec_mode || d.np!=PN || !(g.C==1?flat_spectrum_format(d.format_version):d.format_version==spectrum_contract::blocked48_format) || d.frontier>1 ||
       d.basis_id!=basis_id(g) || d.trunk_bits!=unsigned(g.T) || d.C!=g.C || d.M2!=g.M2 || d.transform_trunks!=g.N ||
       d.source_limbs!=limbs || d.source_trunks!=(limbs*64+g.T-1)/g.T || d.block_stride!=(g.C==1?(d.format_version==spectrum_contract::flat_packed48_format?48:64):g.bstride) ||
       d.live_slots<g.lbv || d.written_slots<g.lbw || d.live_slots>d.M2 || d.written_slots>d.M2)return false;
    for(unsigned q=0;q<PN;++q)if(!d.scale[q] || d.scale[q]>=p::PR[q])return false;
    return true;
}
bool cache_matches(const sbn3_spectrum_desc &actual,const sbn3_spectrum_desc &expected,const p::Plan &g,size_t limbs){
    if(!actual.instance_id || !compatible(actual,g,limbs)||actual.format_version!=expected.format_version||actual.block_stride!=expected.block_stride)return false;
    if(expected.instance_id)return actual.seal==expected.seal;
    return actual.generation==expected.generation && actual.frontier==expected.frontier &&
        actual.live_slots>=expected.live_slots && actual.written_slots>=expected.written_slots &&
        actual.storage_bytes>=expected.storage_bytes && actual.table_bytes>=expected.table_bytes && actual.plane_bytes>=expected.plane_bytes &&
        !memcmp(actual.scale,expected.scale,sizeof actual.scale);
}
p::XposeCtx transpose_shape(const p::Plan &g,size_t limbs) {
    p::XposeCtx x{};x.an=limbs;x.T=g.T;x.C=g.C;
    x.natv=((limbs*64+g.T-1)/g.T+7)/8;
    // The wide codec selects only bytes [0,T). SIMD lookahead is readable
    // from the next slot or the >=64-byte row padding; it is masked away.
    // Narrow T=80/84/88 keeps its existing shifted-window representation.
    x.xs=g.T>88?size_t(g.T):(g.T+15)&~size_t(7);x.xtc=32;x.tr=g.C<64?g.C:64;
    x.ncv=(x.natv+g.C-1)/g.C;if(x.ncv>g.M2)x.ncv=g.M2;
    x.xrs=p::padded_row_stride(x.ncv*x.xs,2);return x;
}
size_t worker_bytes(const p::Plan &g,bool fused,const p::XposeCtx &x,unsigned codec_mode,unsigned kind=0,bool streamed=false) {
    size_t largest=0;
    auto finish=[&](const Sizer &s){require(s.ok,SBN3_FATAL_SIZE,"p48 scratch sizer");if(s.at>largest)largest=s.at;};
    if constexpr(LEAF!=8){
        {Sizer s;const size_t extra=g.T>192?16:g.T>88?8:0,gb=((32+8*size_t(g.T)*g.Lg+63)>>6)+2+extra;
            s.words(g.M2*gb+16);s.words(g.M2*g.Lg,64);for(unsigned q=0;q<PN;++q)s.line_assembler(g.nblk);finish(s);}
        {Sizer s;s.words(TB*g.Lg*(g.C+8),64);s.words(TB*g.Lg*(g.C+8),64);finish(s);}
        {Sizer s;s.words(g.M2*IROW_W,64);s.line_assembler(g.nblk);finish(s);}
        {Sizer s;s.words(PN*(p::OCH/8+3),64);finish(s);}
        size_t rounded=0;require(align_size(largest,64,rounded),SBN3_FATAL_SIZE,"leaf scratch round");return rounded;
    }
    {Sizer s;if(streamed)s.take(stream_row_grain*x.xrs+4096);size_t run=0,output=0;align_size((streamed?stream_row_grain:x.tr)*x.T+16,64,run);align_size(x.xtc*x.xs,64,output);s.take(run*x.xtc);s.take(output+64);finish(s);}
    {Sizer s;if(streamed)s.take(stream_row_grain*x.xrs+4096);const size_t gb=((32+8*size_t(g.T)+63)>>6)+2+(g.T>192?16:8);const size_t pcs=((size_t(g.T)+47)/48)*8;
        s.words(g.M2*(codec_mode==3 && pcs>gb?pcs:gb)+16);s.words(g.M2,64);for(unsigned q=0;q<PN;++q)s.line_assembler(g.nblk);finish(s);}
    {Sizer s;if(fused)s.line_assembler(g.nrows);s.words(2*(g.C+8),64);s.words(2*(g.C+8),64);s.take(g.nrows*TB*SLOT+128);s.words(4*g.C);finish(s);}
    {Sizer s;s.words(2*(g.C+8),64);s.words(4*g.C);s.words(4*g.C);finish(s);} // SPEC prepare
    if(kind==SBN3_PRODUCT_MAC2){Sizer s;if(fused)s.line_assembler(g.nrows);s.words(2*(g.C+8),64);s.words(2*(g.C+8),64);
        s.words(2*(g.C+8),64);s.words(2*(g.C+8),64);s.take(g.nrows*TB*SLOT+128);s.words(4*g.C);finish(s);}
    if(fused) {
        Sizer s;for(unsigned q=0;q<PN;++q)s.words(g.M2+16,64);
        for(int k=0;k<g.WD;++k)s.words(g.lbv+8);
        s.words(g.lbv+8);s.words(g.lbv+8);s.take(g.lbv+64);s.line_assembler(g.lbv);finish(s);
    } else {
        Sizer row;row.words(g.M2*IROW_W,64);row.line_assembler(g.nblk);finish(row);
        Sizer emit;emit.words(PN*(p::OCH/8+3),64);finish(emit);
    }
    size_t rounded=0;require(align_size(largest,64,rounded),SBN3_FATAL_SIZE,"scratch round");return rounded;
}
// Integrity of every semantic plan field. Padding is deliberately excluded;
// this is an accidental-corruption guard, not a security/content identity.
uint64_t plan_seal(const ProductPlan &q) {
    uint64_t h=feed(feed(1469598103934665603ULL,q.magic),q.backend_id);
    h=feed(h,q.options.algorithm);h=feed(h,uint64_t(q.options.fused_start_skew_us));h=feed(h,q.options.codec_mode);h=feed(h,q.options.prime_count);h=feed(h,q.options.crt_mode);
    for(uint64_t v:{q.product.a_limbs,q.product.b_limbs,uint64_t(q.options.workers),uint64_t(q.options.prime_batch),
                   uint64_t(q.options.trunk_bits),uint64_t(q.options.borrow_output),q.options.workspace_budget,
                   uint64_t(q.options.column_log2),uint64_t(q.options.row_log2)})h=feed(h,v);
    const auto &g=q.arithmetic.transform;
    h=feed(h,uint64_t(g.arm));
    h=feed(h,uint64_t(g.T));
    h=feed(h,uint64_t(g.LW));
    h=feed(h,uint64_t(g.BL));
    h=feed(h,uint64_t(g.BL16));
    h=feed(h,uint64_t(g.WD));
    h=feed(h,uint64_t(g.OT));
    h=feed(h,uint64_t(g.OL));
    h=feed(h,uint64_t(g.L));
    h=feed(h,uint64_t(g.Lg));
    h=feed(h,uint64_t(g.lgC));
    h=feed(h,uint64_t(g.C));
    h=feed(h,uint64_t(g.M2));
    h=feed(h,uint64_t(g.M));
    h=feed(h,uint64_t(g.N));
    h=feed(h,uint64_t(g.nrows));
    h=feed(h,uint64_t(g.lbv));
    h=feed(h,uint64_t(g.lbw));
    h=feed(h,uint64_t(g.plane_bytes));
    h=feed(h,uint64_t(g.nblk));
    h=feed(h,uint64_t(g.bstride));
    h=feed(h,uint64_t(g.bstride_o));
    h=feed(h,uint64_t(g.plane_bytes_o));
    h=feed(h,uint64_t(g.lbp));
    h=feed(h,uint64_t(g.full));
    h=feed(h,uint64_t(g.mrow_unnorm));
    h=feed(h,uint64_t(g.hspec));
    h=feed(h,uint64_t(g.an));
    h=feed(h,uint64_t(g.yn));
    h=feed(h,uint64_t(g.nat));
    h=feed(h,uint64_t(g.nyt));
    h=feed(h,uint64_t(g.natv_a));
    h=feed(h,uint64_t(g.natv_y));
    h=feed(h,uint64_t(g.Zv));
    h=feed(h,uint64_t(g.Zw));
    h=feed(h,uint64_t(g.nwt));
    h=feed(h,uint64_t(g.nwtp));
    h=feed(h,uint64_t(g.t0));
    h=feed(h,uint64_t(g.dlt));
    h=feed(h,uint64_t(g.pc));
    h=feed(h,uint64_t(g.need));
    h=feed(h,uint64_t(g.ntp));
    h=feed(h,uint64_t(g.nl));
    h=feed(h,uint64_t(g.outcap));
    h=feed(h,uint64_t(g.voff));
    h=feed(h,uint64_t(g.vtrunks));
    h=feed(h,uint64_t(g.ring_rn));
    h=feed(h,uint64_t(g.rowscale));
    h=feed(h,q.arithmetic.basis.id);
    for(auto *s:{&q.arithmetic.rows_a,&q.arithmetic.rows_b,&q.arithmetic.product}) {
        h=feed(h,s->defined_slots);h=feed(h,s->padded_slots);h=feed(h,s->frontier);
        for(auto v:s->scale)h=feed(h,v);
    }
    for(auto v:q.arithmetic.scale_a)h=feed(h,v);
    const auto &e=q.execution;const auto &i=e.info;
    h=feed(h,uint64_t(i.np));
    h=feed(h,i.algorithm);h=feed(h,i.fused_start_skew_us);h=feed(h,i.crt_mode);h=feed(h,i.codec_mode);
    h=feed(h,uint64_t(i.trunk_bits));
    h=feed(h,uint64_t(i.digit_words));
    h=feed(h,uint64_t(i.workers));
    h=feed(h,uint64_t(i.prime_batch));
    h=feed(h,uint64_t(i.fused));
    h=feed(h,uint64_t(i.row_major));
    h=feed(h,uint64_t(i.borrow_output));
    h=feed(h,uint64_t(i.full));
    h=feed(h,uint64_t(i.C));
    h=feed(h,uint64_t(i.M2));
    h=feed(h,uint64_t(i.lbv));
    h=feed(h,uint64_t(i.lbw));
    h=feed(h,uint64_t(i.nat));
    h=feed(h,uint64_t(i.nyt));
    h=feed(h,uint64_t(i.transform_trunks));
    h=feed(h,uint64_t(i.output_limbs));
    h=feed(h,uint64_t(i.output_alignment));
    h=feed(h,uint64_t(i.table_bytes));
    h=feed(h,uint64_t(i.workspace_bytes));
    h=feed(h,uint64_t(i.workspace_alignment));
    h=feed(h,uint64_t(i.per_worker_bytes));
    h=feed(h,uint64_t(i.plane_pitch));
    h=feed(h,uint64_t(i.pool_bytes));
    h=feed(h,uint64_t(i.transpose_bytes));
    h=feed(h,uint64_t(i.table_entries));
    h=feed(h,uint64_t(i.factor_levels));
    h=feed(h,uint64_t(i.root_order_log2));
    h=feed(h,uint64_t(i.block_stride));
    h=feed(h,uint64_t(i.product_row_stride));
    h=feed(h,uint64_t(i.emit_trunks));
    h=feed(h,uint64_t(i.emit_limbs));
    h=feed(h,uint64_t(i.emit_quantum_trunks));
    h=feed(h,uint64_t(i.emit_quantum_limbs));
    h=feed(h,uint64_t(i.rowscale));
    h=feed(h,uint64_t(i.row_task_grain));
    h=feed(h,uint64_t(i.fused_row_grain));
    h=feed(h,uint64_t(i.format_slot_bytes));
    h=feed(h,uint64_t(i.format_block_slots));
    h=feed(h,uint64_t(i.fused_items));
    h=feed(h,uint64_t(i.basis_id));
    h=feed(h,uint64_t(i.arithmetic_id));
    h=feed(h,uint64_t(i.execution_id));
    for(auto v:i.scale_a)h=feed(h,v);
    for(auto *x:{&e.transpose_a,&e.transpose_b}) {
        require(!x->a && !x->xp,SBN3_FATAL_ARGUMENT,"address in arithmetic plan");
        for(auto v:{x->an,x->natv,x->T,x->C,x->ncv,x->xrs,x->tr,x->xs,x->xtc,x->row0,uint64_t(x->cached_store)})h=feed(h,v);
    }
    for(uint64_t v:{e.pitch_alignment,e.tail_bytes,e.journal_bytes,e.spill_bytes,uint64_t(e.emit_tasks)})h=feed(h,v);
    const auto &r=q.recipe;
    for(auto v:r.lengths)h=feed(h,v);
    for(uint64_t v:{uint64_t(r.kind),uint64_t(r.cached_mask),r.window.offset_bits,r.window.width_bits,r.window.error_bits,r.spectrum_bytes,uint64_t(r.scaled)})h=feed(h,v);
    for(unsigned t=0;t<2;++t){h=feed(h,descriptor_seal(r.cached[t]));h=feed(h,r.cached[t].seal);
        for(unsigned k=0;k<PN;++k){h=feed(h,r.k[t][k]);h=feed(h,r.kr[t][k]);}}
    return h;
}
ProductPlan load_plan(const sbn3_mul_plan &opaque) {
    ProductPlan q{};memcpy(&q,opaque.opaque,sizeof q);
    require(q.magic==plan_magic && q.backend_id==PN && q.seal==plan_seal(q) && q.execution.info.np==PN && q.options.workers>=1 && q.options.workers<=32 &&
            q.arithmetic.transform.L==LEAF && q.arithmetic.basis.id==basis_id(q.arithmetic.transform),SBN3_FATAL_ARGUMENT,"p48 plan identity");
    return q;
}
// Owned root-table budget is independent of a binding borrowing a cache.
// Keep the historical spectrum upper bound, including its alignment slack.
size_t owned_table_bytes(size_t entries,size_t levels,size_t M2,bool flat) {
    Sizer tables;tables.take(sizeof(p::Primes));
    for(unsigned k=0;k<PN;++k){tables.take(p::tower_bytes(p::clog2(entries)));tables.take(p::tower_zero_slots*8);}
    for(unsigned k=0;k<PN;++k)for(int d=0;d<2;++d)tables.take(levels*M2*8+64);
    if(flat)tables.take(sizeof(p::FlatConstants));
    size_t bytes=0;return tables.ok && align_size(tables.at,64,bytes)?bytes:0;
}
template<bool materialize=true>
sbn3_query_result query_impl(const sbn3_product_request &request,const sbn3_mul_options &options,
                          sbn3_mul_plan &out,sbn3_mul_info &info) {
    if constexpr(LEAF!=8)if(request.cyclic_limbs || request.kind!=SBN3_PRODUCT_MUL || request.cached_a[0] || request.cached_a[1] ||
        !options.column_log2 || !options.row_log2 || options.algorithm==SBN3_MUL_FLAT || options.codec_mode>1)return SBN3_UNSUPPORTED;
    if(request.negacyclic)return SBN3_UNSUPPORTED;
    if(request.window_limbs&&(!request.cyclic_limbs||request.window_limbs>request.cyclic_limbs||
        request.a_limbs+(request.kind==SBN3_PRODUCT_SQR?request.a_limbs:request.b_limbs)>=request.cyclic_limbs))return SBN3_UNSUPPORTED;
    if(request.cyclic_limbs && (request.kind!=SBN3_PRODUCT_MUL && request.kind!=SBN3_PRODUCT_SQR))return SBN3_UNSUPPORTED;
    if(request.cyclic_limbs>(size_t(1)<<31))return SBN3_QUERY_CAPACITY;
    if(request.cyclic_limbs && (request.a_limbs>request.cyclic_limbs || request.b_limbs>request.cyclic_limbs))return SBN3_UNSUPPORTED;
    if(request.kind>SBN3_PRODUCT_MAC2 || (request.kind!=SBN3_PRODUCT_MAC2 && (request.a1_limbs || request.b1_limbs || request.cached_a[1])) ||
       (request.kind==SBN3_PRODUCT_SQR && request.b_limbs))return SBN3_UNSUPPORTED;
    for(auto n:{request.a_limbs,request.b_limbs,request.a1_limbs,request.b1_limbs})if(n>(size_t(1)<<31))return SBN3_QUERY_CAPACITY;
    ProductPlan q{};q.options=options;auto &r=q.recipe;
    if(options.algorithm>SBN3_MUL_FLAT || options.fused_start_skew_us < -1 || options.fused_start_skew_us>1000)return SBN3_UNSUPPORTED;r.kind=request.kind;
    r.lengths[0]=request.a_limbs;r.lengths[1]=request.kind==SBN3_PRODUCT_SQR?request.a_limbs:request.b_limbs;
    r.lengths[2]=request.a1_limbs;r.lengths[3]=request.b1_limbs;
    if(!r.lengths[0] || !r.lengths[1] || (r.kind==SBN3_PRODUCT_MAC2 && (!r.lengths[2] || !r.lengths[3])))return SBN3_UNSUPPORTED;
    for(unsigned t=0;t<2;++t)if(request.cached_a[t]){r.cached_mask|=1u<<t;r.cached[t]=*request.cached_a[t];}
    sbn3_product_spec spec{r.lengths[0],r.lengths[1]};
    if(r.kind==SBN3_PRODUCT_MAC2){if(r.lengths[2]>spec.a_limbs)spec.a_limbs=r.lengths[2];if(r.lengths[3]>spec.b_limbs)spec.b_limbs=r.lengths[3];}
    q.product=spec;
    if((options.prime_count && options.prime_count!=PN) || options.crt_mode>2 || (options.codec_mode!=0 && options.codec_mode!=1 && options.codec_mode!=3))return SBN3_UNSUPPORTED;
    q.options.prime_count=PN;q.options.crt_mode=options.crt_mode?options.crt_mode:default_crt(PN);
    if(!q.options.workers || q.options.workers>32 || q.options.prime_batch>PN || q.options.borrow_output>2)return SBN3_UNSUPPORTED;
    // A materialized cache fixes the mathematical geometry. Never silently rebuild it.
    const sbn3_spectrum_desc *basis=request.cached_a[0]?request.cached_a[0]:request.cached_a[1];
    if(basis){
        if(basis->seal!=descriptor_seal(*basis) || basis->np!=PN || !basis->C || !basis->M2 ||
           (basis->C&(basis->C-1)) || (basis->M2&(basis->M2-1)) || basis->C>8192 || basis->M2>(size_t(1)<<21))return SBN3_UNSUPPORTED;
        if((options.trunk_bits && unsigned(options.trunk_bits)!=basis->trunk_bits) ||
           (options.column_log2 && options.column_log2!=unsigned(p::clog2(basis->C))) ||
           (options.row_log2 && options.row_log2!=unsigned(p::clog2(basis->M2))))return SBN3_UNSUPPORTED;
        const unsigned layout=flat_spectrum_format(basis->format_version)?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
        if(options.algorithm && options.algorithm!=layout)return SBN3_UNSUPPORTED;
        q.options.algorithm=layout;q.options.trunk_bits=basis->trunk_bits;
        if(layout==SBN3_MUL_FLAT){if(basis->C!=1 || options.column_log2 || options.row_log2)return SBN3_UNSUPPORTED;q.options.column_log2=q.options.row_log2=0;}
        else{q.options.column_log2=p::clog2(basis->C);q.options.row_log2=p::clog2(basis->M2);}
    }
    if(request.cyclic_limbs && !q.options.algorithm && !q.options.column_log2 && !q.options.row_log2)
        q.options.algorithm=request.cyclic_limbs<=(size_t(1)<<16)?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
    const bool flat=q.options.algorithm==SBN3_MUL_FLAT;
    q.options.algorithm=flat?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
    if(flat && (options.column_log2 || options.row_log2 || (options.prime_batch && options.prime_batch!=PN)))return SBN3_UNSUPPORTED;
    const bool exact=q.options.column_log2 || q.options.row_log2;
    if(exact && (q.options.column_log2<3 || q.options.column_log2>13 || q.options.row_log2<4 ||
                 q.options.row_log2>21 || q.options.column_log2+q.options.row_log2>27))return SBN3_UNSUPPORTED;
    // The implemented cr tower chooser admits lgM<=27; this also keeps all
    // intermediate byte/bit arithmetic well inside size_t on the native ABI.
    if(spec.a_limbs>(size_t(1)<<31) || spec.b_limbs>(size_t(1)<<31))return SBN3_QUERY_CAPACITY;
    if(!spec.a_limbs || !spec.b_limbs)return SBN3_UNSUPPORTED; // zero uses the already implemented value/basecase entry
    auto &g=q.arithmetic.transform;
    q.options.codec_mode=options.codec_mode?options.codec_mode:1;
    int T=q.options.trunk_bits?q.options.trunk_bits:p::plan_T_auto(spec.a_limbs,spec.b_limbs);
    auto terms=[&](int t){size_t n[4];for(unsigned k=0;k<4;++k)n[k]=(r.lengths[k]*64+t-1)/t;return (n[0]<n[1]?n[0]:n[1])+(n[2]<n[3]?n[2]:n[3]);};
    if(r.kind==SBN3_PRODUCT_MAC2 && !q.options.trunk_bits)while(T>=p::plan_T_min() && !p::plan_T_ok(T,terms(T),terms(T)))T-=p::plan_T_step();
    if(T<=88 && q.options.codec_mode>1)return SBN3_UNSUPPORTED;
    if(T<p::plan_T_min() || T>p::plan_T_max() || T%p::plan_T_step())return SBN3_UNSUPPORTED;
    // The ring is an exact mathematical constraint. With automatic T, search
    // the finite codec set before refusing a period that the widest T misses.
    if(request.cyclic_limbs){
        auto fits=[&](int t){const size_t bits=request.cyclic_limbs*64,n=bits/size_t(t);
            return bits%size_t(t)==0 && n>=128 && !(n&(n-1)) && p::plan_T_ok(t,(spec.a_limbs*64+t-1)/t,(spec.b_limbs*64+t-1)/t);};
        if(!q.options.trunk_bits)while(T>=p::plan_T_min() && !fits(T))T-=p::plan_T_step();
        if(T<p::plan_T_min() || !fits(T))return SBN3_UNSUPPORTED;
    }
    const auto arm=request.cyclic_limbs?p::ARM_CYC:r.kind==SBN3_PRODUCT_TMP?p::ARM_TMP:p::ARM_LIN;
    if(flat){if(!p::plan_flat(g,arm,spec.a_limbs,spec.b_limbs,T,r.kind==SBN3_PRODUCT_MAC2?terms(T):0,basis?basis->M2:0,request.cyclic_limbs))return SBN3_QUERY_CAPACITY;}
    else if(!p::plan_make(g,arm,spec.a_limbs,spec.b_limbs,T,LEAF,
                    exact?int(q.options.column_log2+q.options.row_log2):0,exact?int(q.options.row_log2):0,8,request.cyclic_limbs,0,0,exact,r.kind==SBN3_PRODUCT_MAC2?terms(T):0))return SBN3_QUERY_CAPACITY;
    if(exact && (g.lgC!=q.options.column_log2 || g.M2!=(size_t(1)<<q.options.row_log2)))return SBN3_UNSUPPORTED;
    if(LEAF==8 && !flat && !exact && !basis && !request.cyclic_limbs && r.kind==SBN3_PRODUCT_MUL) {
        const unsigned target=p48_geometry::smt_column_log(q.options.workers,PN,unsigned(g.lgC),unsigned(p::clog2(g.M2)),g.lbv);
        if(target!=unsigned(g.lgC)) {
            const unsigned outer=unsigned(p::clog2(g.C*g.M2));p::Plan alternative{};
            if(p::plan_make(alternative,arm,spec.a_limbs,spec.b_limbs,T,LEAF,int(outer),int(outer-target),
                            8,0,0,0,1,0) && alternative.N==g.N && alternative.full==g.full &&
               alternative.C==(size_t(1)<<target)) {
                g=alternative;
                // Pin the resolved geometry for program replay and rebatching.
                q.options.column_log2=target;q.options.row_log2=outer-target;
            }
        }
    }
    if(r.kind==SBN3_PRODUCT_MAC2){g.outcap=(r.lengths[0]+r.lengths[1]>r.lengths[2]+r.lengths[3]?r.lengths[0]+r.lengths[1]:r.lengths[2]+r.lengths[3])+1;
        if(g.outcap>g.nl)return SBN3_QUERY_CAPACITY;}
    if(r.kind==SBN3_PRODUCT_TMP){r.window={uint64_t(T)*g.t0,uint64_t(T)*g.nwt,uint64_t(T+1+p::clog2(g.nat))};
        if(r.window.error_bits>=r.window.width_bits)return SBN3_UNSUPPORTED;}
    if(request.window_limbs)r.window={0,64*request.window_limbs,0};
    for(unsigned t=0;t<2;++t)if((r.cached_mask>>t)&1)if(!compatible(r.cached[t],g,r.lengths[2*t]))return SBN3_UNSUPPORTED;
    if(flat){if(options.codec_mode>1)return SBN3_UNSUPPORTED;q.options.codec_mode=1;q.options.prime_batch=PN;}
    else if(!options.codec_mode)q.options.codec_mode=default_codec(T,g.M2);
    if constexpr(LEAF!=8)q.options.codec_mode=1;
    auto &e=q.execution;auto &i=e.info;
    if(!q.options.prime_batch) {
        const size_t chunks=(g.nblk+7)/8;
        // Small columns complete a chunk quickly; sharing the queue across
        // primes amortizes the per-batch tail. Deep C=8192 keeps the compact pool.
        q.options.prime_batch=q.options.workers>1 && g.C<=2048 && g.M2<=8192 && chunks<=size_t(q.options.workers)*16?PN:1;
    }
    i.np=PN;i.crt_mode=q.options.crt_mode;i.codec_mode=q.options.codec_mode;i.trunk_bits=g.T;i.digit_words=g.WD;i.workers=q.options.workers;i.prime_batch=q.options.prime_batch;
    i.algorithm=q.options.algorithm;i.fused_start_skew_us=flat?0:options.fused_start_skew_us<0?0:options.fused_start_skew_us>0?unsigned(options.fused_start_skew_us):(spec.a_limbs<=(1u<<22) && spec.b_limbs<=(1u<<22)?0:40);
    i.fused=LEAF==8 && !flat && g.M2<=8192 && r.kind!=SBN3_PRODUCT_TMP && g.C>=p::fuse_row_grain(g.T);i.row_major=i.fused;i.full=g.full;
    i.C=g.C;i.M2=g.M2;i.lbv=g.lbv;i.lbw=g.lbw;i.nat=g.nat;i.nyt=g.nyt;i.transform_trunks=g.N;
    i.output_limbs=request.window_limbs?request.window_limbs:g.outcap;i.output_alignment=64;i.workspace_alignment=flat?64:work_alignment;
    i.table_entries=p::plan_tower_entries(g);i.factor_levels=flat?0:g.lgC+1;
    i.root_order_log2=g.lgC+p::clog2(g.M2)+(LEAF==8?1:3);
    i.block_stride=g.bstride;i.product_row_stride=p::padded_row_stride(g.lbw*SLOT,1);
    i.emit_trunks=request.window_limbs?((request.window_limbs+g.OL-1)/g.OL)*g.OT:g.ntp;i.emit_limbs=request.window_limbs?i.emit_trunks*g.T/64:g.nl;i.emit_quantum_trunks=g.OT;i.emit_quantum_limbs=g.OL;
    i.rowscale=g.rowscale;i.row_task_grain=LEAF==8?UR:2;i.fused_row_grain=p::fuse_row_grain(g.T);i.format_slot_bytes=SLOT;i.format_block_slots=TB;
    i.fused_items=i.fused?(i.workers<g.C/i.fused_row_grain?i.workers:unsigned(g.C/i.fused_row_grain)):0;
    if(i.root_order_log2>32)return SBN3_QUERY_CAPACITY;
    uint64_t h=0;
    if constexpr(materialize) {
    q.arithmetic.basis.id=i.basis_id=basis_id(g);
    q.arithmetic.rows_a={g.lbv,g.lbw,1,{}};q.arithmetic.rows_b={g.lbv,g.lbw,1,{}};
    q.arithmetic.product={g.lbv,g.lbw,2,{}};
    for(unsigned k=0;k<PN;++k) {
        q.arithmetic.scale_a[k]=i.scale_a[k]=q.arithmetic.rows_a.scale[k]=p::fold_scale_on(g)?p::mulm(p::plane_scale(g,p::PR[k]),i.crt_mode==2?p::cofactor_inverse(k):1,p::PR[k]):1;
        q.arithmetic.rows_b.scale[k]=1;
        q.arithmetic.product.scale[k]=p::mulm(p::mulm(g.C*(LEAF==8?1:8),p::fixed_setup.prime[k].r52_inverse,p::PR[k]),q.arithmetic.scale_a[k],p::PR[k]);
    }
    for(unsigned t=0;t<2;++t)for(unsigned k=0;k<PN;++k){
        const uint64_t alpha=(r.cached_mask&(1u<<t))?r.cached[t].scale[k]:q.arithmetic.scale_a[k];
        const uint64_t divisor=r.kind==SBN3_PRODUCT_SQR?p::mulm(alpha,alpha,p::PR[k]):alpha;
        if (divisor == q.arithmetic.scale_a[k]) r.k[t][k] = 1;
        else if (r.kind == SBN3_PRODUCT_SQR && alpha == q.arithmetic.scale_a[k]) {
            const auto &s = p::fixed_setup.prime[k];
            const uint64_t normalization = g.C * g.rowscale * (g.Lg > 1 ? 8 : 1);
            r.k[t][k] = p::fold_scale_on(g)
                ? p::mulm(p::mulm(normalization, s.r52_inverse, p::PR[k]), i.crt_mode == 2 ? s.cofactor : 1, p::PR[k])
                : 1;
        } else r.k[t][k] = p::mulm(q.arithmetic.scale_a[k], p::invm(divisor, p::PR[k]), p::PR[k]);
        r.kr[t][k]=uint64_t((p::u128(r.k[t][k])<<52)/p::PR[k]);r.scaled|=r.k[t][k]!=1;
    }
    if(r.cached_mask&1){q.arithmetic.rows_a.frontier=1+r.cached[0].frontier;
        for(unsigned k=0;k<PN;++k)q.arithmetic.rows_a.scale[k]=r.cached[0].scale[k];}
    h=feed(i.basis_id,i.crt_mode);
    for(auto v:q.arithmetic.scale_a)h=feed(h,v);
    for(uint64_t n:{g.an,g.yn,g.nat,g.nyt,g.lbv,g.lbw,uint64_t(g.full),g.rowscale})h=feed(h,n);
    h=feed(h,r.kind);h=feed(h,g.ring_rn);h=feed(h,request.window_limbs);for(auto v:r.lengths)h=feed(h,v);
    for(unsigned t=0;t<2;++t)for(auto v:r.k[t])h=feed(h,v);
    i.arithmetic_id=h;
    }
    const size_t owned_tables=owned_table_bytes(i.table_entries,i.factor_levels,g.M2,flat);
    if(!owned_tables)return SBN3_QUERY_CAPACITY;
    if(r.cached_mask) {
        // Keep a nonempty lease for the existing bind API. Only Flat has
        // local codec constants; Primes comes from a retained cache owner.
        Sizer local;local.take(64);if(flat)local.take(sizeof(p::FlatConstants));
        if(!local.ok || !align_size(local.at,64,i.table_bytes))return SBN3_QUERY_CAPACITY;
    } else i.table_bytes=owned_tables;
    if(!flat && LEAF==8){e.transpose_a=transpose_shape(g,g.an);e.transpose_b=transpose_shape(g,g.yn);}
    const size_t xrs=e.transpose_a.xrs>e.transpose_b.xrs?e.transpose_a.xrs:e.transpose_b.xrs;
    i.transpose_bytes=flat || LEAF!=8?0:g.C*xrs+4096;
    i.borrow_output=LEAF==8 && !flat && options.borrow_output
        ? (i.transpose_bytes<=i.output_limbs*8 ? 1u : options.borrow_output==2 ? 2u : 0u) : 0u;
    const size_t row_bytes=g.C*p::padded_row_stride(g.lbw*SLOT,1)+4096;
    const size_t need=i.fused && row_bytes>g.plane_bytes?row_bytes:g.plane_bytes;
    e.pitch_alignment=flat?64:need>=work_alignment?work_alignment:4096;
    unsigned plane_sets=(r.kind==SBN3_PRODUCT_SQR?1:2)-unsigned(bool(r.cached_mask&1));
    if(r.kind==SBN3_PRODUCT_MAC2)plane_sets+=2-unsigned(bool(r.cached_mask&2));
    unsigned slots=PN*plane_sets+(i.fused?i.prime_batch:0);
    if(r.kind==SBN3_PRODUCT_SQR && r.cached_mask)slots=PN; // readonly cache needs separate products
    if(flat)slots=PN;
    // Pool placement retains its original large-page alignment. Individual
    // plane views need only cache-line alignment and their readable padding;
    // do not round every slot to a separate 2 MiB extent. Odd line strides
    // also give successive prime planes different cache-set offsets.
    if(!align_size(need+128,64,i.plane_pitch))return SBN3_QUERY_CAPACITY;
    if(!flat && !(i.plane_pitch&64) && !add_size(i.plane_pitch,64,i.plane_pitch))return SBN3_QUERY_CAPACITY;
    if(!mul_size(i.plane_pitch,slots,i.pool_bytes))return SBN3_QUERY_CAPACITY;
    if(i.pool_bytes<i.transpose_bytes)i.pool_bytes=i.transpose_bytes; // prepare borrows idle pool
    if(flat){
        const unsigned buffers=(p::flat_packa_active(g) && r.kind==SBN3_PRODUCT_MUL && !r.cached_mask)?1:r.kind==SBN3_PRODUCT_SQR?1:(r.kind==SBN3_PRODUCT_MAC2?4:2)-unsigned(bool(r.cached_mask&1))-unsigned(bool(r.cached_mask&2));
        const size_t temp=buffers*(g.M2+16)*64,emit=PN*(p::OCH/8+3)*64;align_size(temp>emit?temp:emit,64,i.per_worker_bytes);
    }else {
        const auto &largest_x=e.transpose_a.xrs>=e.transpose_b.xrs?e.transpose_a:e.transpose_b;
        i.per_worker_bytes=worker_bytes(g,i.fused,largest_x,i.codec_mode,r.kind);
        if(LEAF==8 && !i.borrow_output) {
            const size_t local=worker_bytes(g,i.fused,largest_x,i.codec_mode,r.kind,true);
            // Exact storage inequality, not a size/performance guess.
            if(local*i.workers<i.per_worker_bytes*i.workers+i.transpose_bytes) {
                i.per_worker_bytes=local;i.transpose_bytes=0;i.row_task_grain=stream_row_grain;
            }
        }
    }
    Sizer cache;cache.take(sizeof(Spectrum));cache.take(owned_tables);for(unsigned k=0;k<PN;++k)cache.take(flat?g.lbw*(spectrum_format(q)==spectrum_contract::flat_packed48_format?48:64)+128:g.plane_bytes+128,64);
    if(!cache.ok || !align_size(cache.at,64,r.spectrum_bytes))return SBN3_QUERY_CAPACITY;
    e.tail_bytes=((request.window_limbs?std::max(g.nl-g.outcap,size_t(g.OL)):g.nl-g.outcap)+16)*8;
    e.journal_bytes=i.fused?i.workers*g.lbv+64:0;
    e.spill_bytes=i.fused?0:((g.ntp+p::OCH-1)/p::OCH)*64;
    Sizer work;work.take(sizeof(Binding));
    work.take(i.pool_bytes,e.pitch_alignment);
    if(!i.borrow_output)work.take(i.transpose_bytes);
    work.take(e.tail_bytes);if(e.journal_bytes)work.take(e.journal_bytes);if(e.spill_bytes)work.take(e.spill_bytes);
    for(unsigned k=0;k<i.workers;++k)work.take(i.per_worker_bytes);
    if(!work.ok || !align_size(work.at,64,i.workspace_bytes))return SBN3_QUERY_CAPACITY;
    if constexpr(materialize) {
    for(uint64_t n:{uint64_t(i.workers),uint64_t(i.prime_batch),uint64_t(i.fused),uint64_t(i.borrow_output),
                   i.table_bytes,i.plane_pitch,i.workspace_bytes,i.per_worker_bytes,i.transpose_bytes,i.block_stride,i.product_row_stride,
                   uint64_t(i.row_task_grain),uint64_t(i.fused_row_grain),uint64_t(i.fused_items)})h=feed(h,n);
    h=feed(h,i.algorithm);h=feed(h,i.fused_start_skew_us);h=feed(h,i.codec_mode);h=feed(h,r.cached_mask);for(const auto &d:r.cached)h=feed(h,d.frontier);i.execution_id=h;
    }
    info=i;
    if(options.workspace_budget && i.workspace_bytes>options.workspace_budget) {
        if(!flat && !options.prime_batch && q.options.prime_batch>1){auto smaller=options;smaller.prime_batch=1;return query_impl<materialize>(request,smaller,out,info);}
        return SBN3_QUERY_CAPACITY;
    }
    if constexpr(materialize){q.seal=plan_seal(q);memset(&out,0,sizeof out);memcpy(out.opaque,&q,sizeof q);}
    return SBN3_SUPPORTED;
}
sbn3_query_result geometry_query(const sbn3_product_request &r,const sbn3_mul_options &o,sbn3_mul_info &i) {
    i={};
    if(r.cached_a[0] || r.cached_a[1])return SBN3_UNSUPPORTED;
    sbn3_mul_plan unused;
    return query_impl<false>(r,o,unused,i);
}
sbn3_query_result query(const sbn3_product_spec &spec,const sbn3_mul_options &opt,sbn3_mul_plan &out,sbn3_mul_info &info){
    sbn3_product_request r{};r.a_limbs=spec.a_limbs;r.b_limbs=spec.b_limbs;return query_impl(r,opt,out,info);
}
sbn3_query_result product_query(const sbn3_product_request &req,const sbn3_mul_options &opt,sbn3_mul_plan &out,sbn3_product_info &info){
    if constexpr(LEAF!=8){info={};return SBN3_UNSUPPORTED;} // experiment exposes the one-shot MUL recipe only
    info={};auto result=query_impl(req,opt,out,info.mul);if(result!=SBN3_SUPPORTED)return result;
    // query_impl just constructed and sealed this plan in this call. Loading
    // it through the public-plan validation path would hash every field a
    // second time for every search candidate. Bind/spectrum entry points still
    // validate caller-supplied plans before using them.
    ProductPlan q;memcpy(&q,out.opaque,sizeof q);info.kind=static_cast<sbn3_product_kind>(q.recipe.kind);info.cached_mask=q.recipe.cached_mask;
    info.window=q.recipe.window;info.spectrum_bytes=q.recipe.spectrum_bytes;info.spectrum_alignment=64;info.cyclic_limbs=q.arithmetic.transform.ring_rn;
    for(unsigned t=0;t<2;++t)for(unsigned k=0;k<PN;++k)info.leaf_scale[t][k]=q.recipe.k[t][k];return result;
}
Binding &binding(sbn3_mul_binding *b) {return *reinterpret_cast<Binding *>(b);}
const Binding &binding(const sbn3_mul_binding *b) {return *reinterpret_cast<const Binding *>(b);}
void idle_owner(const Binding &b) {
    require(pthread_equal(b.run.team->creator,pthread_self()) && !b.run.team->busy,
            SBN3_FATAL_TEAM,"binding owner/active");
}
void execute_ptrs(sbn3_mul_binding *,const uint64_t *,const uint64_t *,uint64_t *);
void initialize_workspace(Binding *b,const sbn3_spectrum *const cached[2],
                          const p::Garner *garner=nullptr,const p::EmitTabs *emit=nullptr) {
    const auto &q=b->plan;const auto &i=q.execution.info;
    uint8_t *pool=static_cast<uint8_t *>(b->root.allocate(i.pool_bytes,q.execution.pitch_alignment));
    b->run.pool=pool;unsigned slot=0;
    for(unsigned t=0;t<2;++t){b->run.cached[t]=cached[t];if(cached[t])spectrum_retain(cached[t]);}
    const auto &r=q.recipe;
    for(unsigned k=0;k<PN;++k){
        if(i.algorithm==SBN3_MUL_FLAT){b->run.b_planes[k]=pool+(slot++)*i.plane_pitch;continue;}
        if(!(r.cached_mask&1))b->run.a_planes[k]=pool+(slot++)*i.plane_pitch;
        if(r.kind!=SBN3_PRODUCT_SQR)b->run.b_planes[k]=pool+(slot++)*i.plane_pitch;
        else b->run.b_planes[k]=(r.cached_mask&1)?pool+(slot++)*i.plane_pitch:b->run.a_planes[k];
        if(r.kind==SBN3_PRODUCT_MAC2){if(!(r.cached_mask&2))b->run.a1_planes[k]=pool+(slot++)*i.plane_pitch;
            b->run.b1_planes[k]=pool+(slot++)*i.plane_pitch;}
    }
    // Preserve the established one-shot LIN plane placement exactly. The generic
    // recipe allocator above interleaves operands; that is not a LIN optimization.
    if(i.algorithm!=SBN3_MUL_FLAT && r.kind==SBN3_PRODUCT_MUL && !r.cached_mask)
        for(unsigned k=0;k<PN;++k){b->run.a_planes[k]=pool+k*i.plane_pitch;b->run.b_planes[k]=pool+(PN+k)*i.plane_pitch;}
    if(i.fused && !(r.kind==SBN3_PRODUCT_SQR && r.cached_mask))
        for(unsigned k=0;k<i.prime_batch;++k)b->run.spares[k]=pool+(slot++)*i.plane_pitch;
    require(slot*i.plane_pitch<=i.pool_bytes,SBN3_FATAL_WORKSPACE,"plane pool sizing");
    if(!i.borrow_output)b->run.transpose=static_cast<uint8_t *>(b->root.allocate(i.transpose_bytes));
    b->run.tail=static_cast<uint64_t *>(b->root.allocate(q.execution.tail_bytes));
    if(q.execution.journal_bytes)b->run.journal=static_cast<uint8_t *>(b->root.allocate(q.execution.journal_bytes));
    if(q.execution.spill_bytes)b->run.spill=static_cast<uint64_t (*)[8]>(b->root.allocate(q.execution.spill_bytes));
    for(unsigned k=0;k<i.workers;++k)b->run.workers[k]=::new(b->worker_objects+k*sizeof(Frame))Frame(b->root.subframe(i.per_worker_bytes));
    b->work_used=b->root.used();
    require(b->table_used<=i.table_bytes && b->work_used<=i.workspace_bytes,SBN3_FATAL_WORKSPACE,"p48 sizing mismatch");
    auto &c=b->context;const auto &g=b->plan.arithmetic.transform;
    c.fused_start_skew_us=i.fused_start_skew_us;c.codec_mode=i.codec_mode;c.pl=&g;c.PS=b->run.primes;c.gates=b->gates;c.smalltw=1;c.hbs=g.bstride;c.obs=g.bstride_o;
    c.rpf=4;c.ipf=2;c.tpf=2;c.nts=g.plane_bytes>=(size_t(1)<<20);c.prm=i.row_major;
    c.rms=p::padded_row_stride(g.lbw*SLOT,1);
    if(garner){b->garner=*garner;b->emit_tables=*emit;}
    else {
    b->garner.init(*b->run.primes,g,i.crt_mode);
    p::EmitCtx ec{};ec.G=&b->garner;p::emit_tabs_init(b->emit_tables,ec,*b->run.primes,g);
    }
    for(unsigned t=0;t<2;++t){c.frontier[t]=cached[t]?spectrum(cached[t]).desc.frontier:0;c.leaf_k[t]=r.k[t];c.leaf_rec[t]=r.kr[t];}
    // Context pointers must refer to the persistent plan copy, never the bind stack.
    for(unsigned t=0;t<2;++t){c.leaf_k[t]=b->plan.recipe.k[t];c.leaf_rec[t]=b->plan.recipe.kr[t];}
}
void product_bind(const sbn3_mul_plan &opaque,sbn3_arena &arena,const sbn3_lease &tables,
            const sbn3_lease &workspace,sbn3_team &team,const sbn3_spectrum *cache0,const sbn3_spectrum *cache1,sbn3_mul_binding **out) {
    ProductPlan q=load_plan(opaque);const auto &i=q.execution.info;
    require(team.arena==&arena && team.width>=i.workers && pthread_equal(team.creator,pthread_self()) && !team.busy,
            SBN3_FATAL_TEAM,"p48 team binding");
    require(tables.bytes>=i.table_bytes && workspace.bytes>=i.workspace_bytes &&
            !(reinterpret_cast<uintptr_t>(tables.data)&63) &&
            !(reinterpret_cast<uintptr_t>(workspace.data)&(i.workspace_alignment-1)) &&
            !overlaps(tables.data,tables.bytes,workspace.data,workspace.bytes) &&
            !overlaps(team.storage.data,team.storage.bytes,workspace.data,workspace.bytes) &&
            !overlaps(team.storage.data,team.storage.bytes,tables.data,tables.bytes),
            SBN3_FATAL_WORKSPACE,"p48 resource spans",i.workspace_bytes,workspace.bytes);
    for(unsigned k=1;k<team.width;++k)
        require(!overlaps(team.stacks[k].data,team.stacks[k].bytes,workspace.data,workspace.bytes) &&
                !overlaps(team.stacks[k].data,team.stacks[k].bytes,tables.data,tables.bytes),SBN3_FATAL_WORKSPACE,"p48 stack alias");
    const sbn3_spectrum *cached[2]{cache0,cache1};
    for(unsigned t=0;t<2;++t){require(bool(cached[t])==bool(q.recipe.cached_mask&(1u<<t)),SBN3_FATAL_ARGUMENT,"missing/unexpected spectrum");
        if(cached[t]){const auto &sp=spectrum(cached[t]);require(cached[t]->backend==&SBN3_P48_BACKEND() && cached[t]->marker==spectrum_magic &&
            cache_matches(sp.desc,q.recipe.cached[t],q.arithmetic.transform,q.recipe.lengths[t*2]),SBN3_FATAL_ARGUMENT,"incompatible spectrum");}}
    arena.claim_unshared(tables);arena.claim_unshared(workspace);
    auto *b=::new(workspace.data) Binding(q,arena,tables,workspace,team);
    b->header.execute_fast=execute_ptrs;
    b->root.allocate(sizeof(Binding));
    ComputeLease setup(arena);
    {
        Frame table_frame(arena,tables);
        const auto *root_owner=cache0?cache0:cache1;
        if(root_owner) {
            const auto &roots=spectrum(root_owner).primes;
            require(roots.lg>=size_t(p::clog2(i.table_entries)),SBN3_FATAL_MATH,"cached table tower coverage");
            for(unsigned k=0;k<PN;++k) {
                const auto &pr=roots.P[k];
                require(pr.e && pr.lg>=size_t(p::clog2(i.table_entries)) && (i.algorithm==SBN3_MUL_FLAT ||
                    (pr.M2f>=i.M2 && pr.lgF>=q.arithmetic.transform.lgC)),SBN3_FATAL_MATH,"cached table factor coverage");
            }
            // initialize_workspace retains both cached handles. These tables
            // were built at reserve, and are read-only even before ready.
            b->run.primes=const_cast<p::Primes *>(&roots);
        } else {
            b->run.primes=::new(table_frame.allocate(sizeof(p::Primes)))p::Primes{};
            b->run.primes->init(table_frame,p::clog2(i.table_entries));
            if(i.algorithm!=SBN3_MUL_FLAT)b->run.primes->factors_for(table_frame,i.M2,q.arithmetic.transform.lgC);
        }
        if(i.algorithm==SBN3_MUL_FLAT){b->run.flat_constants=::new(table_frame.allocate(sizeof(p::FlatConstants)))p::FlatConstants{};
            p::dec_init(b->run.flat_constants->a,nullptr,q.arithmetic.transform,*b->run.primes,q.arithmetic.scale_a);
            p::dec_init(b->run.flat_constants->b,nullptr,q.arithmetic.transform,*b->run.primes);
        }
        b->table_used=table_frame.used();
    }
    initialize_workspace(b,cached);arena.release(workspace);
    *out=&b->header;
}
void bind(const sbn3_mul_plan &opaque,sbn3_arena &a,const sbn3_lease &t,const sbn3_lease &w,sbn3_team &team,sbn3_mul_binding **out){
    const auto q=load_plan(opaque);require(!q.arithmetic.transform.ring_rn && !q.recipe.cached_mask && q.recipe.kind==SBN3_PRODUCT_MUL,SBN3_FATAL_ARGUMENT,"mul binding recipe");
    product_bind(opaque,a,t,w,team,nullptr,nullptr,out);
}
struct Execute {Binding *b;sbn3_const_limbs a,y;sbn3_limbs output;bool prepared_a=false;};
uint64_t tick_ns() {timespec ts{};clock_gettime(CLOCK_MONOTONIC,&ts);return uint64_t(ts.tv_sec)*1000000000+ts.tv_nsec;}
// E1 is either one borrowed/global plane or a worker-local tile owning
// whole output lines. TB2/packed48 needs two rows (192 bytes) per tile.
// A local tile dies immediately after rows_fn; no output is borrowed, so
// product programs retain their consume-input contract.
struct StreamRows {p::Ctx context;p::XposeCtx shape;};
void stream_rows(void *ptr,uint64_t lo,uint64_t hi,int w,Frame *ws) {
    auto call=*static_cast<StreamRows*>(ptr);
    auto &c=call.context;auto &x=call.shape;
    x.tr=stream_row_grain;x.cached_store=true;
    x.xp=static_cast<uint8_t*>(ws->allocate(stream_row_grain*x.xrs+4096));
    c.xp=x.xp;c.xrs=x.xrs;c.xs=x.xs;c.a=x.a;c.an=x.an;c.natv=x.natv;
    require(lo%stream_row_grain==0 && hi%stream_row_grain==0,SBN3_FATAL_MATH,"streamed E1 whole-line row unit");
    for(size_t row=lo;row<hi;row+=stream_row_grain) {
        FrameMark mark(*ws);x.row0=row;c.xp_row0=row;
        {FrameMark transpose(*ws);p::xpose_fn(&x,row/stream_row_grain,row/stream_row_grain+1,w,ws);}
        p::rows_fn<8,0>(&c,row,row+stream_row_grain,w,ws);
    }
}
void forward_rows(Binding &b,sbn3_team_scope *scope,p::Ctx &c,p::XposeCtx x,
                  const uint64_t *src,uint64_t *out,size_t count=SIZE_MAX) {
    if(count!=SIZE_MAX)x.an=count;
    if(!b.plan.execution.info.transpose_bytes) {
        x.a=src;StreamRows call{c,x};
        kernel_for(scope,0,x.C,stream_row_grain,SBN3_STATIC,stream_rows,&call,b.run.workers);
    } else {
        x.a=src;x.xp=b.plan.execution.info.borrow_output?reinterpret_cast<uint8_t*>(out):b.run.transpose;
        kernel_for(scope,0,x.C/x.tr,1,SBN3_STATIC,p::xpose_fn,&x,b.run.workers);
        c.a=src;c.an=x.an;c.natv=x.natv;c.xp=x.xp;c.xrs=x.xrs;c.xs=x.xs;c.xp_row0=0;
        kernel_for(scope,0,x.C,UR,SBN3_STATIC,p::rows_fn<8,0>,&c,b.run.workers);c.xp=nullptr;
    }
}
void emit_range(void *ctx,uint64_t lo,uint64_t hi,int w,Frame *f) {
    for(uint64_t i=lo;i<hi;++i){FrameMark mark(*f);p::emit_fn<8>(ctx,static_cast<int>(i),w,f);}
}
#include "backend/ntt_p48/flat_execute.hpp"
template<unsigned L>
void execute_leaf(Execute &x,sbn3_team_scope *scope){
    auto &b=*x.b;auto &c=b.context;const auto &g=b.plan.arithmetic.transform;const auto &i=b.plan.execution.info;
    require_scope_leader(scope);require(scope->team==b.run.team && scope->width>=i.workers,SBN3_FATAL_TEAM,"leaf scope");
    sbn3_team_scope sub{scope->team,scope->first,i.workers,false,scope->epoch};scope=&sub;
    uint32_t expected=0;require(__atomic_compare_exchange_n(&b.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent leaf binding");
    c.counts=nullptr;c.xp=nullptr;c.reversed=0;memset(b.run.tail,0,b.plan.execution.tail_bytes);
    const auto t0=tick_ns();
    for(unsigned q=0;q<PN;++q){c.hp[q]=b.run.a_planes[q];c.fp[q]=b.run.a_planes[q];c.op[q]=nullptr;}
    c.a=x.a.data;c.an=x.a.count;c.natv=g.natv_a;c.dsc=b.plan.arithmetic.scale_a;
    kernel_for(scope,0,g.C,2,SBN3_STATIC,p::rows_fn<L,0>,&c,b.run.workers);const auto t1=tick_ns();
    for(unsigned q=0;q<PN;++q)c.fp[q]=b.run.b_planes[q];
    c.a=x.y.data;c.an=x.y.count;c.natv=g.natv_y;c.dsc=nullptr;
    kernel_for(scope,0,g.C,2,SBN3_STATIC,p::rows_fn<L,0>,&c,b.run.workers);const auto t2=tick_ns();
    kernel_for(scope,0,PN*g.nblk,8,SBN3_DYNAMIC,p::block_fn<L,0,0>,&c,b.run.workers);const auto t3=tick_ns();
    kernel_for(scope,0,PN*(g.nrows/IROW_G),1,SBN3_STATIC,p::irow_fn<p::ARM_LIN>,&c,b.run.workers);
    p::EmitCtx ec{};ec.pl=&g;ec.PS=b.run.primes;ec.G=&b.garner;ec.rp=x.output.data;ec.tail=b.run.tail;ec.spill=b.run.spill;
    ec.nch=(g.ntp+p::OCH-1)/p::OCH;ec.EK=&b.emit_tables.EK;ec.RG=&b.emit_tables.RG;for(unsigned q=0;q<PN;++q)ec.plane[q]=c.fp[q];
    p::EmitMap map;ec.map=&map;const int tasks=p::emit_tasks(map,g,ec.nch,0);
    auto emit=[](void *arg,uint64_t lo,uint64_t hi,int worker,Frame *frame){for(auto j=lo;j<hi;++j){FrameMark mark(*frame);p::emit_fn<L>(arg,int(j),worker,frame);}};
    kernel_for(scope,0,tasks,1,SBN3_DYNAMIC,emit,&ec,b.run.workers);p::emit_join(ec);const auto t4=tick_ns();
    b.last_stage_ns[0]=t1-t0;b.last_stage_ns[1]=t2-t1;b.last_stage_ns[2]=t3-t2;b.last_stage_ns[3]=t4-t3;++b.executions;__atomic_store_n(&b.active,0,__ATOMIC_RELEASE);
}
void execute_action(void *argument,sbn3_team_scope *scope) {
    auto &x=*static_cast<Execute *>(argument);auto &b=*x.b;auto &c=b.context;
    if constexpr(LEAF!=8){execute_leaf<LEAF>(x,scope);return;}
    const auto &e=b.plan.execution;const auto &i=e.info;const auto &g=b.plan.arithmetic.transform;
    if(i.algorithm==SBN3_MUL_FLAT){sbn3_product_inputs in{x.a,x.y,{},{}};execute_flat(b,scope,in,x.output,false,x.prepared_a?b.run.a_planes:nullptr);return;}
    c.counts=nullptr;for(auto &v:b.counts)v={};
    require_scope_leader(scope);
    require(scope->team==b.run.team && scope->width>=i.workers,SBN3_FATAL_TEAM,"p48 scope width");
    sbn3_team_scope subset{scope->team,scope->first,i.workers,false,scope->epoch};scope=&subset;
    uint32_t expected=0;
    require(__atomic_compare_exchange_n(&b.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
            SBN3_FATAL_LIFETIME,"concurrent binding execution");
    const uint64_t t0=tick_ns();
    for(auto &gate:b.gates){gate.next.store(0,std::memory_order_relaxed);gate.done.store(0,std::memory_order_relaxed);}
    // Reset only output tails/journals. Plane payload is redefined by the passes.
    memset(b.run.tail,0,e.tail_bytes);if(e.journal_bytes)memset(b.run.journal,0,e.journal_bytes);
    for(unsigned k=0;k<PN;++k){c.hp[k]=b.run.a_planes[k];c.fp[k]=b.run.a_planes[k];c.op[k]=nullptr;}
    c.dsc=b.plan.arithmetic.scale_a;
    if(!x.prepared_a){
        forward_rows(b,scope,c,e.transpose_a,x.a.data,x.output.data,x.a.count);
    }
    const uint64_t t1=tick_ns();
    for(unsigned k=0;k<PN;++k)c.fp[k]=b.run.b_planes[k];c.dsc=nullptr;
    forward_rows(b,scope,c,x.prepared_a?transpose_shape(g,x.y.count):e.transpose_b,x.y.data,x.output.data,x.y.count);c.xp=nullptr;
    const uint64_t t2=tick_ns();uint64_t t3=0;
    const KernelForFn tile=c.writeback_a?p::block_fn4<8,0,0,0,false,true>:p::block_fn4<8,0,0>;
    if(i.fused) {
        uint8_t *spare[PN];for(unsigned k=0;k<i.prime_batch;++k)spare[k]=b.run.spares[k];
        const size_t nbi=(g.nblk+7)&~size_t(7);
        for(unsigned first=0;first<PN;first+=i.prime_batch) {
            const unsigned count=first+i.prime_batch<=PN?i.prime_batch:PN-first;
            for(unsigned k=0;k<count;++k)c.op[first+k]=spare[k];
            kernel_for(scope,first*nbi,(first+count)*nbi,8,SBN3_DYNAMIC,tile,&c,b.run.workers);
            for(unsigned k=0;k<count;++k)spare[k]=b.run.b_planes[first+k];
        }
        t3=tick_ns();
        p::FuseCtx f{};f.c=&c;f.EK=&b.emit_tables.EK;f.rp=x.output.data;f.tail=b.run.tail;
        f.nts=g.outcap*8>=(size_t(1)<<20);f.W=i.fused_items;f.jrn=b.run.journal;
        kernel_for(scope,0,i.fused_items,1,SBN3_STATIC,p::irowemit_fn<8>,&f,b.run.workers);p::fuse_join(f,g);
    } else {
        kernel_for(scope,0,PN*g.nblk,8,SBN3_DYNAMIC,tile,&c,b.run.workers);
        t3=tick_ns();
        kernel_for(scope,0,PN*(g.nrows/IROW_G),1,SBN3_STATIC,p::irow_fn<p::ARM_LIN>,&c,b.run.workers);
        p::EmitCtx ec{};ec.pl=&g;ec.PS=b.run.primes;ec.G=&b.garner;ec.rp=x.output.data;ec.tail=b.run.tail;
        ec.spill=b.run.spill;ec.nch=(g.ntp+p::OCH-1)/p::OCH;ec.EK=&b.emit_tables.EK;ec.RG=&b.emit_tables.RG;
        for(unsigned k=0;k<PN;++k)ec.plane[k]=c.fp[k];
        p::EmitMap map;ec.map=&map;int tasks=p::emit_tasks(map,g,ec.nch,0);
        kernel_for(scope,0,tasks,1,SBN3_DYNAMIC,emit_range,&ec,b.run.workers);p::emit_join(ec);
    }
    const uint64_t t4=tick_ns();
    b.last_stage_ns[0]=t1-t0;b.last_stage_ns[1]=t2-t1;b.last_stage_ns[2]=t3-t2;b.last_stage_ns[3]=t4-t3;
    ++b.executions;
    __atomic_store_n(&b.active,0,__ATOMIC_RELEASE);
}
void validate_values(Binding &b,sbn3_const_limbs a,sbn3_const_limbs y,sbn3_limbs out) {
    const auto &s=b.plan.product;
    require(!b.plan.arithmetic.transform.ring_rn && !b.plan.recipe.kind && !b.plan.recipe.cached_mask,SBN3_FATAL_ARGUMENT,"mul execution recipe");
    const size_t capacity=sbn3_mul_output_capacity(&b.plan.execution.info);
    require(a.count==s.a_limbs && y.count==s.b_limbs && out.capacity>=capacity,
            SBN3_FATAL_ARGUMENT,"p48 value lengths");
    const size_t ab=bytes_for(a.count,8),yb=bytes_for(y.count,8),rb=bytes_for(capacity,8);
    valid_span(a.data,ab,"p48 a");valid_span(y.data,yb,"p48 b");valid_span(out.data,rb,"p48 output");
    require(!(reinterpret_cast<uintptr_t>(out.data)&63) && !(reinterpret_cast<uintptr_t>(a.data)&7) &&
            !(reinterpret_cast<uintptr_t>(y.data)&7) && !overlaps(out.data,rb,a.data,ab) && !overlaps(out.data,rb,y.data,yb),
            SBN3_FATAL_ARGUMENT,"p48 value alignment/alias");
    for(const auto &l:{b.run.table_memory,b.run.work_memory,b.run.team->storage})
        require(!overlaps(l.data,l.bytes,a.data,ab) && !overlaps(l.data,l.bytes,y.data,yb) && !overlaps(l.data,l.bytes,out.data,rb),
                SBN3_FATAL_ARGUMENT,"p48 resource/value alias");
    for(unsigned k=1;k<b.run.team->width;++k) {
        const auto &l=b.run.team->stacks[k];
        require(!overlaps(l.data,l.bytes,a.data,ab) && !overlaps(l.data,l.bytes,y.data,yb) && !overlaps(l.data,l.bytes,out.data,rb),
                SBN3_FATAL_ARGUMENT,"p48 stack/value alias");
    }
}
void execute(sbn3_mul_binding *opaque,sbn3_const_limbs a,sbn3_const_limbs y,sbn3_limbs out) {
    auto &b=binding(opaque);idle_owner(b);validate_values(b,a,y,out);
    Execute x{&b,a,y,out};sbn3_team_run(b.run.team,execute_action,&x);
}
void execute_ptrs(sbn3_mul_binding *opaque,const uint64_t *a,const uint64_t *y,uint64_t *out){
    const auto &q=binding(opaque).plan;
    execute(opaque,{a,q.recipe.lengths[0]},{y,q.recipe.lengths[1]},{out,sbn3_mul_output_capacity(&q.execution.info)});
}
void execute_scope(sbn3_mul_binding *opaque,sbn3_team_scope *scope,sbn3_const_limbs a,sbn3_const_limbs y,sbn3_limbs out) {
    auto &b=binding(opaque);require_scope_leader(scope);
    require(scope->team==b.run.team && scope->width==b.plan.options.workers,SBN3_FATAL_TEAM,"p48 borrowed scope");
    validate_values(b,a,y,out);Execute x{&b,a,y,out};execute_action(&x,scope);
}
void metrics(const sbn3_mul_binding *opaque,sbn3_mul_metrics &m) {
    const auto &b=binding(opaque);idle_owner(b);m={0,b.table_used,b.work_used,b.executions,{}};
    for(unsigned k=0;k<4;++k)m.last_stage_ns[k]=b.last_stage_ns[k];
    for(unsigned k=0;k<b.plan.options.workers;++k)if(b.run.workers[k]->peak()>m.worker_peak_bytes)m.worker_peak_bytes=b.run.workers[k]->peak();
}
void unbind(sbn3_mul_binding *opaque) {
    auto &b=binding(opaque);idle_owner(b);
    for(unsigned k=0;k<b.plan.options.workers;++k)b.run.workers[k]->~Frame();
    bool borrowed=false;
    for(auto *cache:b.run.cached)if(cache)borrowed|=b.run.primes==&spectrum(cache).primes;
    if(!borrowed)b.run.primes->~Primes();
    for(auto *cache:b.run.cached)if(cache)spectrum_release(cache);
    auto *a=b.run.arena;auto tables=b.run.table_memory;
    b.header.marker=0;b.~Binding();a->release(tables);
}
struct Program {
    ProductPlan plan{};
    p::Primes *primes=nullptr;
    p::FlatConstants *constants=nullptr;
    p::Garner garner{};
    p::EmitTabs emit{};
    size_t table_used=0;
};
sbn3_mul_options program_options(const sbn3_mul_plan &opaque) {
    const auto q=load_plan(opaque);auto o=q.options;
    o.trunk_bits=q.arithmetic.transform.T;
    return o;
}
unsigned program_contract(const sbn3_mul_plan &opaque) {
    const auto q=load_plan(opaque);
    // Ordinary LIN only. FLAT finishes all prime transforms before emission;
    // Bailey finishes both operand row passes before any inverse-row emission.
    // E1 must not borrow output storage while raw inputs are still live.
    return program_bounded_inputs | (!q.execution.info.borrow_output ? program_consume_inputs : 0u);
}
SharedPreparation program_tables(const sbn3_mul_plan &opaque) {
    const auto q=load_plan(opaque);const auto &i=q.execution.info;
    if(q.recipe.kind!=SBN3_PRODUCT_MUL || q.recipe.cached_mask || q.arithmetic.transform.ring_rn)return {};
    const bool flat=i.algorithm==SBN3_MUL_FLAT;
    Sizer z;z.take(sizeof(p::Primes));
    for(unsigned k=0;k<PN;++k){z.take(p::tower_bytes(p::clog2(i.table_entries)));z.take(p::tower_zero_slots*8);}
    if(!flat)for(unsigned k=0;k<PN;++k)for(unsigned d=0;d<2;++d)z.take(i.factor_levels*i.M2*8+64);
    size_t bytes=0;if(!z.ok || !align_size(z.at,128,bytes))return {};
    // Version/prime family/tower extent/factor geometry. No codec, scale,
    // input support, CRT mode, worker count or destination address is in this key.
    return {{0x5034385441423032ULL,PN,LEAF,i.table_entries,flat?0:i.M2,flat?0:i.factor_levels,0,0},bytes,128};
}
size_t program_local_bytes(const sbn3_mul_plan &opaque) {
    const auto q=load_plan(opaque);size_t header=0,constants=0;
    if(!align_size(sizeof(Program),128,header))return 0;
    if(q.execution.info.algorithm==SBN3_MUL_FLAT && !align_size(sizeof(p::FlatConstants),128,constants))return 0;
    return header+constants;
}
size_t program_bytes(const sbn3_mul_plan &opaque) {
    const auto t=program_tables(opaque);size_t total=0;
    return t.bytes && add_size(program_local_bytes(opaque),t.bytes,total)?total:0;
}
const void *program_tables_prepare(const sbn3_mul_plan &opaque,Frame &f) {
    const auto q=load_plan(opaque);const auto &i=q.execution.info;
    auto *primes=::new(f.allocate(sizeof(p::Primes),128))p::Primes{};
    primes->init(f,p::clog2(i.table_entries));
    if(i.algorithm!=SBN3_MUL_FLAT)primes->factors_for(f,i.M2,q.arithmetic.transform.lgC);
    return primes;
}
void program_finish(Program &v,const ProductPlan &q,Frame &f,const void *shared) {
    v.plan=q;v.primes=const_cast<p::Primes *>(static_cast<const p::Primes *>(shared));
    if(q.execution.info.algorithm==SBN3_MUL_FLAT){
        v.constants=::new(f.allocate(sizeof(p::FlatConstants),128))p::FlatConstants{};
        p::dec_init(v.constants->a,nullptr,q.arithmetic.transform,*v.primes,q.arithmetic.scale_a);
        p::dec_init(v.constants->b,nullptr,q.arithmetic.transform,*v.primes);
    }
    v.table_used=q.execution.info.table_bytes;
    v.garner.init(*v.primes,q.arithmetic.transform,q.execution.info.crt_mode);
    p::EmitCtx ec{};ec.G=&v.garner;p::emit_tabs_init(v.emit,ec,*v.primes,q.arithmetic.transform);
}
const void *program_prepare_shared(const sbn3_mul_plan &opaque,Frame &f,const void *shared) {
    const auto q=load_plan(opaque);
    auto *v=::new(f.allocate(sizeof(Program),128)) Program{};
    program_finish(*v,q,f,shared);return v;
}
const void *program_prepare(const sbn3_mul_plan &opaque,Frame &f) {
    const auto q=load_plan(opaque);const auto request=program_tables(opaque);
    require(request.bytes,SBN3_FATAL_ARGUMENT,"p48 program recipe");
    auto *v=::new(f.allocate(sizeof(Program),128)) Program{};
    auto tf=f.subframe(request.bytes,128);
    const void *shared=program_tables_prepare(opaque,tf);
    program_finish(*v,q,f,shared);return v;
}
void program_execute(const void *ptr,Frame &f,sbn3_team_scope *scope,sbn3_const_limbs a,sbn3_const_limbs y,sbn3_limbs out) {
    const auto &v=*static_cast<const Program *>(ptr);const auto &q=v.plan;const auto &i=q.execution.info;
    void *memory=f.allocate(i.workspace_bytes,i.workspace_alignment);
    auto *b=::new(memory) Binding(q,f,memory,*static_cast<sbn3_team *>(scope->team));
    b->root.allocate(sizeof(Binding));b->run.primes=v.primes;b->run.flat_constants=v.constants;
    b->table_used=v.table_used;const sbn3_spectrum *cached[2]{};
    initialize_workspace(b,cached,&v.garner,&v.emit);
    Execute x{b,a,y,out};execute_action(&x,scope);
    for(unsigned k=0;k<i.workers;++k)b->run.workers[k]->~Frame();
    b->~Binding();
}
#include "backend/ntt_p48/product_recipes.hpp"
size_t program_pair_bytes(const sbn3_mul_plan &opaque) {
    const auto q=load_plan(opaque);const auto &i=q.execution.info;
    if(LEAF!=8 || !program_bytes(opaque) || i.borrow_output)return 0;
    if(i.algorithm!=SBN3_MUL_FLAT)return i.workspace_bytes;
    return i.workspace_bytes+PN*(q.arithmetic.transform.lbw*48+128)+128;
}
void program_pair_execute(const void *ptr,Frame &f,sbn3_team_scope *scope,sbn3_const_limbs common,
                          sbn3_const_limbs x,sbn3_const_limbs y,sbn3_limbs out0,sbn3_limbs out1) {
    const auto &v=*static_cast<const Program *>(ptr);const auto &q=v.plan;const auto &i=q.execution.info;
    void *memory=f.allocate(i.workspace_bytes,i.workspace_alignment);
    auto *b=::new(memory) Binding(q,f,memory,*static_cast<sbn3_team *>(scope->team));
    b->root.allocate(sizeof(Binding));b->run.primes=v.primes;b->run.flat_constants=v.constants;
    b->table_used=v.table_used;const sbn3_spectrum *cached[2]{};
    initialize_workspace(b,cached,&v.garner,&v.emit);
    const auto &g=b->plan.arithmetic.transform;
    if(i.algorithm==SBN3_MUL_FLAT){
        p::FlatCtx c{};c.pl=&g;c.primes=v.primes;c.constants=v.constants;c.a[0]=common.data;c.an[0]=common.count;
        for(unsigned k=0;k<PN;++k){b->run.a_planes[k]=static_cast<uint8_t *>(f.allocate(g.lbw*48+128));c.output[k]=b->run.a_planes[k];}
        kernel_for(scope,0,PN,1,SBN3_STATIC,p::flat_program_prepare48,&c,b->run.workers);
    }else{
        product_rows(*b,scope,common,b->run.a_planes,q.arithmetic.scale_a,nullptr);
        // Private program-owned A planes begin at the row frontier. The
        // first tile completes and saves columns before the second apply.
        b->context.frontier[0]=0;b->context.writeback_a=1;
    }
    Execute first{b,{},x,out0,true};execute_action(&first,scope);
    b->context.writeback_a=0;b->context.frontier[0]=1;
    Execute second{b,{},y,out1,true};execute_action(&second,scope);
    for(unsigned k=0;k<i.workers;++k)b->run.workers[k]->~Frame();
    b->~Binding();
}
}
#ifdef CR_FLAT_PROFILE
extern "C" unsigned sbn3_bench_flat_profile(const sbn3_mul_binding *ptr,uint64_t *out){
    require(ptr && ptr->backend==&SBN3_P48_BACKEND(),SBN3_FATAL_ARGUMENT,"profile backend");
    const auto &b=binding(ptr);idle_owner(b);require(b.plan.recipe.kind==SBN3_PRODUCT_MUL && b.plan.execution.info.algorithm==SBN3_MUL_FLAT,SBN3_FATAL_ARGUMENT,"flat profiling recipe");
    memcpy(out,b.flat_phase,sizeof b.flat_phase);return PN;
}
#endif
#ifdef CR_FLAT_SHARE_TABLES
// Benchmark-only locality experiment. All bindings retain their own allocation;
// the harness restores these pointers before unbinding any member of a wave.
extern "C" void sbn3_bench_share_tables(sbn3_mul_binding *dst,const sbn3_mul_binding *src){
    auto &a=binding(dst);const auto &b=binding(src);idle_owner(a);idle_owner(b);
    require(a.plan.execution.info.arithmetic_id==b.plan.execution.info.arithmetic_id && a.plan.execution.info.algorithm==SBN3_MUL_FLAT &&
            a.plan.recipe.kind==SBN3_PRODUCT_MUL && !a.plan.recipe.cached_mask && !a.run.original_primes,SBN3_FATAL_ARGUMENT,"table locality experiment");
    a.run.original_primes=a.run.primes;a.run.original_flat_constants=a.run.flat_constants;
    a.run.primes=b.run.primes;a.run.flat_constants=b.run.flat_constants;a.context.PS=b.run.primes;
}
extern "C" void sbn3_bench_unshare_tables(sbn3_mul_binding *ptr){
    auto &a=binding(ptr);idle_owner(a);if(!a.run.original_primes)return;
    a.run.primes=a.run.original_primes;a.run.flat_constants=a.run.original_flat_constants;a.context.PS=a.run.primes;
    a.run.original_primes=nullptr;a.run.original_flat_constants=nullptr;
}
#endif
namespace sbn::v3 {
const Backend &SBN3_P48_BACKEND() noexcept {
    static const Backend instance{
        .id = PN,
        .name = SBN3_P48_NAME,
        .query = query,
        .bind = bind,
        .execute = execute,
        .execute_scope = execute_scope,
        .metrics = metrics,
        .unbind = unbind,
        .product_query = product_query,
        .product_bind = product_bind,
        .product_execute = product_execute,
        .product_metrics = product_metrics,
        .prepare = prepare,
        .describe = describe,
        .can_apply = can_apply,
        .spectrum_retain = spectrum_retain,
        .spectrum_release = spectrum_release,
        .spectrum_query = query_spectrum,
        .spectrum_reserve = reserve_spectrum,
        .spectrum_compute = compute_spectrum,
        .spectrum_reserve_plan = reserve_plan_spectrum,
        .spectrum_square = compute_square,
        .program_bytes = program_bytes,
        .program_prepare = program_prepare,
        .program_execute = program_execute,
        .program_contract = program_contract,
        .program_pair_bytes = program_pair_bytes,
        .program_pair_execute = program_pair_execute,
        .program_tables = program_tables,
        .program_local_bytes = program_local_bytes,
        .program_tables_prepare = program_tables_prepare,
        .program_prepare_shared = program_prepare_shared,
        .program_options = program_options,
        .spectrum_multiply = compute_multiply,
        .geometry_query = geometry_query,
        .product_execute_live = product_execute_live,
        .product_fresh_supported = product_fresh_supported,
        .product_execute_fresh = product_execute_fresh,
        .product_live_contract = product_live_contract,
        .product_scratch_bytes = product_scratch_bytes,
        .with_product_scratch = with_product_scratch,
    };
    return instance;
}
}
