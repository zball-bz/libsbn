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
#include "radix/parse_compact.hpp"
#include "radix/once.hpp"
#include "radix/decimal_small.hpp"
#include "algorithms/small_division.hpp"
#include "radix/bits.hpp"
#include "common/identity.hpp"
#include "radix/scale.hpp"
#include "value/divrem_words.hpp"
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
// Integer leaves have no tree, divisor or stage transcript. Their public
// query/bind sequence describes only the digit codec and caller-owned range.
constexpr uint64_t small_plan_magic=0x53424e3352505331ULL,small_binding_magic=0x53424e3352505332ULL;
struct SmallPlan {uint64_t marker,id,digits,limbs,base,workers,flags,work_bytes;uint8_t alphabet[64];};
constexpr size_t small_header=offsetof(SmallPlan,alphabet);
struct SmallBinding {
    struct Inputs {uint64_t digits,limbs,base,flags,work_bytes;};
    uint64_t marker;Inputs plan;Arena *arena;sbn3_team *team;sbn3_lease control;
    const DigitPlan *digits;
    union {DigitPlan custom;};
    const ParseLeafPlan *leaf;
    SmallBinding() noexcept {}
};
constexpr size_t small_storage=align_to(sizeof(SmallBinding),64);
uint64_t small_identity(const SmallPlan &p) noexcept {
    // Disjoint bit fields in the admitted domain: no dependent hash chain
    // for a numeric leaf. Workspace is a multiple of 64 bytes.
    uint64_t h=small_plan_magic^(p.flags&3)^((p.flags&8)<<57)^(p.workers<<2)^(p.base<<8)^(p.digits<<16)^
        (p.limbs<<32)^((p.work_bytes/64)<<45);
    if(p.flags&1)for(unsigned j=0;j<64;j+=8){uint64_t v;memcpy(&v,p.alphabet+j,8);h=identity::word(h,v);}
    return h;
}
sbn3_query_result small_query(const sbn3_parse_spec &s,const sbn3_radix_options *o,
                             sbn3_parse_plan &out,sbn3_parse_info &info,bool &handled) noexcept {
    handled=true;
    const unsigned workers=o?o->workers:1;
    if(s.base<2||s.base>64||!workers||workers>32||(o&&o->use_alphabet>1))return SBN3_UNSUPPORTED;
    SmallPlan p;p.marker=small_plan_magic;p.base=s.base;p.digits=s.integer_digits;p.workers=workers;
    p.flags=o?(uint64_t(o->use_alphabet)|(uint64_t(o->repeated!=0)<<1)):0;
    if(p.flags&1){
        if(s.base==10&&decimal_alphabet(o->alphabet))p.flags=(p.flags&~uint64_t(1))|8;
        else{
            for(unsigned j=0;j<s.base;++j)for(unsigned k=0;k<j;++k)
                if(o->alphabet[j]==o->alphabet[k])return SBN3_UNSUPPORTED;
            memcpy(p.alphabet,o->alphabet,64);
        }
    }
    if(!(s.base&(s.base-1)))p.limbs=(s.integer_digits*unsigned(__builtin_ctz(s.base))+63)/64;
    else p.limbs=uint64_t((__uint128_t(s.integer_digits)*(log2_floor_q56[s.base]+1)+((uint64_t(1)<<62)-1))>>62);
    if(p.limbs>256){handled=false;return SBN3_UNSUPPORTED;}
    p.work_bytes=(s.base&(s.base-1))&&p.digits>512?parse_local_bytes(p.limbs):0;
    p.id=small_identity(p);info.workers=workers;info.limbs=p.limbs;
    info.control_bytes=small_storage;info.workspace_bytes=p.work_bytes;
    info.storage_bytes=small_storage+p.work_bytes;info.storage_alignment=64;
    info.lease_peak=SBN3_CHECK_SMALL?1:0;info.plan_id=p.id;
    if(o&&o->memory_budget&&info.storage_bytes>o->memory_budget)return SBN3_QUERY_CAPACITY;
    memcpy(out.opaque,&p,small_header+(p.flags&1?64:0));return SBN3_SUPPORTED;
}
void small_bind(const sbn3_parse_plan &opaque,Arena &arena,size_t offset,sbn3_team &team,sbn3_parse_binding **out) noexcept {
    SmallPlan p;memcpy(&p,opaque.opaque,small_header);
    if(p.flags&1)memcpy(p.alphabet,reinterpret_cast<const uint8_t*>(opaque.opaque)+small_header,64);
    if constexpr(SBN3_CHECK_SMALL){
        require(p.id==small_identity(p)&&p.base>=2&&p.base<=64&&p.digits&&p.digits<=16384&&!(p.flags&~uint64_t(11))&&
            (!(p.flags&8)||(p.base==10&&!(p.flags&1)))&&
            p.workers&&p.workers<=32&&p.limbs<=256&&!(p.work_bytes%64)&&p.work_bytes/64<2048,
            SBN3_FATAL_ARGUMENT,"small parse plan");
        require(arena.accepts_team(team.arena)&&team.width>=p.workers&&pthread_equal(team.creator,pthread_self())&&!team.busy,SBN3_FATAL_TEAM,"small parse team");
        size_t end;require(add_size(offset,small_storage+p.work_bytes,end)&&end<=arena.virtual_bytes&&arena.contains(offset,end)&&
            !(uintptr_t(arena.base+offset)&63)&&arena.unleased(offset,small_storage+p.work_bytes),SBN3_FATAL_WORKSPACE,"small parse range");
    }
    sbn3_lease control{};if constexpr(SBN3_CHECK_SMALL)control=arena.acquire(offset,small_storage+p.work_bytes);
    auto *b=::new(arena.base+offset)SmallBinding;
    b->marker=small_binding_magic;b->plan={p.digits,p.limbs,p.base,p.flags,p.work_bytes};
    b->arena=&arena;b->team=&team;b->control=control;
    if(!(p.flags&9))b->digits=&identity_digit_plan(unsigned(p.base));
    else if(p.flags&8)b->digits=&decimal_ascii_plan();
    else{
        ::new(&b->custom)DigitPlan;
        if(p.base&(p.base-1))require(digit_plan_init(b->custom,unsigned(p.base),p.alphabet),SBN3_FATAL_ARGUMENT,"small parse alphabet");
        else{memset(b->custom.decode,0xff,256);for(unsigned j=0;j<p.base;++j)b->custom.decode[p.alphabet[j]]=uint8_t(j);}
        b->digits=&b->custom;
    }
    if((p.base&(p.base-1))&&p.digits>128)b->leaf=&parse_leaf_plan(unsigned(p.base));
    *out=reinterpret_cast<sbn3_parse_binding*>(b);
}
void small_arithmetic(SmallBinding &b,const uint8_t *digits,sbn3_limbs out,void*work,sbn3_parse_result *result) noexcept {
    const auto &p=b.plan;
    bool valid;
    if(!(p.base&(p.base-1))){
        const auto *decode=p.flags&1?b.digits->decode:nullptr;
        const auto invalid=validate_digits(digits,p.digits,unsigned(p.base),decode);
        *result={unsigned(invalid==p.digits),invalid==p.digits?0:invalid,0};
        if(invalid==p.digits){const unsigned shift=unsigned(__builtin_ctz(unsigned(p.base)));
            parse_bits_range(out.data,p.limbs,0,(p.limbs+shift-1)/shift,digits,p.digits,shift,0,decode);}
        return;
    }
    if(p.digits<=128)valid=parse_integer_blocks(out.data,p.limbs,digits,p.digits,128,*b.digits);
    else if(p.digits<=512){uint64_t value[parse_leaf_limbs];valid=parse_leaf(*b.leaf,*b.digits,digits,p.digits,value);memcpy(out.data,value,p.limbs*8);}
    else{auto scratch=Frame::external(work,p.work_bytes);
        valid=parse_local(*b.leaf,*b.digits,digits,p.digits,out.data,scratch);}
    *result={unsigned(valid),valid?0:validate_digits(digits,p.digits,unsigned(p.base),b.digits->identity?nullptr:b.digits->decode),0};
}
void small_execute(SmallBinding &b,const uint8_t *digits,sbn3_limbs out,sbn3_parse_result *result) noexcept {
    const auto &p=b.plan;
    if constexpr(SBN3_CHECK_SMALL){
        require(result&&digits&&out.data&&out.capacity>=p.limbs&&!(uintptr_t(out.data)&63),SBN3_FATAL_ARGUMENT,"small parse output");
        require(!overlaps(&b,small_storage+p.work_bytes,digits,p.digits)&&!overlaps(&b,small_storage+p.work_bytes,out.data,p.limbs*8)&&
            !overlaps(digits,p.digits,out.data,p.limbs*8),SBN3_FATAL_ARGUMENT,"small parse overlap");
        require(pthread_equal(b.team->creator,pthread_self())&&!b.team->busy,SBN3_FATAL_TEAM,"small parse owner");
    }
    small_arithmetic(b,digits,out,reinterpret_cast<uint8_t*>(&b)+small_storage,result);
}
// This leaf has no alphabet, powers or divisor state. Keep its descriptor
// separate so adding it does not enlarge the integer query/bind hot path.
constexpr uint64_t decimal_fraction_magic=0x53424e3352504431ULL;
struct DecimalFractionPlan {uint64_t marker,id,digits,limbs,workers;};
struct DecimalFractionBinding {
    DecimalFractionPlan plan;Arena *arena;sbn3_team *team;sbn3_lease lease;
};
constexpr size_t decimal_fraction_storage=align_to(sizeof(DecimalFractionBinding),64);
uint64_t fraction_identity(const DecimalFractionPlan &p) noexcept {
    return decimal_fraction_magic^(p.digits<<16)^(p.limbs<<32)^(p.workers<<2);
}
[[gnu::noinline]] void fraction_bind(const sbn3_parse_plan &opaque,Arena &arena,size_t offset,sbn3_team &team,sbn3_parse_binding **out) noexcept {
    DecimalFractionPlan p;memcpy(&p,opaque.opaque,sizeof p);
    if constexpr(SBN3_CHECK_SMALL){
        require(p.id==fraction_identity(p)&&p.limbs&&p.limbs<=decimal_fraction_read_limbs&&p.digits&&p.digits<=1920&&
            p.workers&&p.workers<=32,SBN3_FATAL_ARGUMENT,"decimal fraction plan");
        require(arena.accepts_team(team.arena)&&team.width>=p.workers&&pthread_equal(team.creator,pthread_self())&&!team.busy,SBN3_FATAL_TEAM,"decimal fraction team");
        size_t end;require(add_size(offset,decimal_fraction_storage,end)&&end<=arena.virtual_bytes&&arena.contains(offset,end)&&
            !(uintptr_t(arena.base+offset)&63)&&arena.unleased(offset,decimal_fraction_storage),SBN3_FATAL_WORKSPACE,"decimal fraction range");
    }
    sbn3_lease lease{};if constexpr(SBN3_CHECK_SMALL)lease=arena.acquire(offset,decimal_fraction_storage);
    auto *b=::new(arena.base+offset)DecimalFractionBinding{p,&arena,&team,lease};
    *out=reinterpret_cast<sbn3_parse_binding*>(b);
}
[[gnu::noinline]] void fraction_execute(DecimalFractionBinding &b,const uint8_t *digits,sbn3_limbs out,sbn3_parse_result *result) noexcept {
    if constexpr(SBN3_CHECK_SMALL){
        require(result&&digits&&out.data&&out.capacity>=b.plan.limbs&&!(uintptr_t(out.data)&63)&&
            !overlaps(&b,decimal_fraction_storage,digits,b.plan.digits)&&!overlaps(&b,decimal_fraction_storage,out.data,b.plan.limbs*8)&&
            !overlaps(digits,b.plan.digits,out.data,b.plan.limbs*8),SBN3_FATAL_ARGUMENT,"decimal fraction buffers");
        require(pthread_equal(b.team->creator,pthread_self())&&!b.team->busy,SBN3_FATAL_TEAM,"decimal fraction owner");
    }
    const bool valid=decimal_fraction_read(out.data,b.plan.limbs,digits,b.plan.digits);
    *result={unsigned(valid),valid?0:validate_digits(digits,b.plan.digits,10,decimal_ascii_plan().decode),0};
}
struct Plan {
    uint64_t marker = 0;
    sbn3_parse_spec spec{};
    sbn3_radix_options options{};
    sbn3_parse_info info{};
    unsigned shift = 0;
    uint64_t digits = 0, fragments = 0;
    uint64_t number_digits=0;
    bool split_parts=false;
    bool small_number=false;
    bool compact=false;
    bool fused_scale=false;
    size_t number_limbs = 0, divide_limbs = 0;
    size_t integer_limbs=0,quotient_limbs=0;
    size_t number_capacity=0;
    ProductChoice repair_choice=0;
    size_t repair_output_limbs=0;
    uint64_t power_fragments = 0;
    unsigned power_rest = 0;
    size_t prepared_offset = 0, prepared_bytes = 0, values_offset = 0, values_bytes = 0, divide_offset = 0,
           divide_bytes = 0, work_offset = 0, work_bytes = 0;
    size_t rail_at = 0, number_at = 0, denominator_at = 0, inverse_at = 0;
    int quotient_product = -1; // tree.extra index of N * reciprocal
    size_t divide_alignment = 0;   // of the Newton INVERSE storage (fraction digits)
    unsigned divide_lease_peak = 0;
    size_t tree_bytes=0;
    uint64_t tree_id = 0, seal = 0;
};
// The plan value: [sealed layout][Newton plan of the reciprocal][number of recorded searches][the transcript]
// (see format.cpp): bind assembles the tree plan again without searching again.
constexpr size_t value_divide_at = (sizeof(Plan) + 7) & ~size_t(7), value_count_at = value_divide_at + sizeof(sbn3_newton_plan),
                 value_entries_at = value_count_at + 8;
static_assert(value_entries_at + PlanTranscript::capacity * 8 <= sizeof(sbn3_parse_plan));
static_assert(value_divide_at+sizeof(CompactParsePlan)<=sizeof(sbn3_parse_plan));
uint64_t seal(const Plan &p, const sbn3_newton_plan &divide, const PlanTranscript &t) noexcept {
    uint64_t h = identity::fnv_seed;
    if constexpr(!SBN3_CHECK_SMALL)if(!p.tree_bytes){
        const uint64_t fields[]={p.spec.base,p.spec.integer_digits,p.spec.fraction_digits,p.spec.fraction_bits,
            p.options.workers,p.options.repeated,p.info.storage_bytes,p.tree_id};
        for(uint64_t v:fields)h=identity::word(h,v);
        if(p.options.use_alphabet)for(unsigned j=0;j<64;j+=8){uint64_t v;memcpy(&v,p.options.alphabet+j,8);h=identity::word(h,v);}
        return h;
    }
    const uint64_t fields[] = {p.marker, p.spec.base, p.spec.integer_digits, p.spec.fraction_digits, p.spec.fraction_bits,
                               p.options.workers, p.options.memory_budget, p.options.use_alphabet, p.options.repeated, p.shift, p.digits,
                               p.fragments, p.number_digits, uint64_t(p.split_parts),uint64_t(p.small_number),uint64_t(p.compact), p.integer_limbs,p.quotient_limbs,p.number_capacity,
                               p.repair_choice,p.repair_output_limbs,uint64_t(p.fused_scale),
                               p.number_limbs, p.divide_limbs, p.power_fragments, p.power_rest,
                               p.prepared_offset, p.prepared_bytes, p.values_offset, p.values_bytes, p.divide_offset,
                               p.divide_bytes, p.work_offset, p.work_bytes, p.rail_at, p.number_at, p.denominator_at, p.inverse_at,
                               uint64_t(p.quotient_product),
                               p.tree_id, p.tree_bytes, p.info.storage_bytes, p.info.limbs, p.info.control_bytes,
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
    if (p.divide_limbs)h=identity::words(h,divide.opaque,sizeof(divide.opaque)/sizeof(uint64_t));
    return h;
}
uint64_t tree_identity(const ParseTreeView &t) noexcept {
    uint64_t h = identity::fnv_seed;
    h = identity::word(h, t.class_count);
    h = identity::word(h,uint64_t(t.repeated));
    h = identity::word(h, t.rail.count);
    h = identity::word(h, t.rail.fixed_levels);
    h = identity::word(h, t.prepared_bytes());
    for(unsigned j=0;j<t.extra_count;++j)h=identity::word(h,t.extra[j].arithmetic_id);
    for (unsigned j = 0; j < t.class_count; ++j) {
        h = identity::word(h, t.classes[j].fragments * 64 + t.classes[j].workers);
        h = identity::word(h, t.classes[j].region_bytes + t.classes[j].slot_limbs + t.classes[j].episode_bytes);
        h = identity::word(h,t.classes[j].high_offset);
        h = identity::word(h,t.classes[j].rest_offset);
        h = identity::word(h, t.classes[j].product.arithmetic_id);
        h = identity::word(h,uint64_t(t.classes[j].direct)|(uint64_t(t.classes[j].direct_native)<<1));
    }
    return h;
}
struct Binding {
    uint64_t marker = 0;
    union {Plan plan;};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, prepared{}, values{}, work{};
    ParseTreeView *tree_view=nullptr;
    CompactParsePlan *compact_plan=nullptr;
    CompactParseState *compact_state=nullptr;
    const ParseTreeView &tree_plan() const noexcept { return *tree_view; }
    struct Direct {DigitPlan digits;ParseLeafPlan leaf;};
    Binding(const Plan &p) noexcept:plan(p){}
    union {ParseTree tree;Direct direct;};
    DigitPlan &digits() noexcept {return plan.tree_bytes?tree.digits:direct.digits;}
    ParseLeafPlan &leaf() noexcept {return plan.tree_bytes?tree.leaf:direct.leaf;}
    int root = -1;
    int integer_root=-1;
    union {sbn3_newton_plan divide;};
    uint64_t quotient_shift = 0; // g: M = quotient >> g
    int64_t compare_shift = 0;   // candidate * denominator <=> N << compare_shift
    divrem_words::PreparedDivisor small_divisor{};
    uint64_t small_denominator[parse_leaf_limbs]; // initialized only for a short divisor
    uint8_t decode[256];       // initialized only for base 2^s
    uint64_t *limbs(size_t at) const noexcept { return static_cast<uint64_t *>(values.data) + at; }
};
size_t program_offset(size_t tree_bytes) noexcept { return align_to(sizeof(Binding),64)+tree_bytes; }
struct Assembly {
    Plan &p;
    ParseTreePlan &tree;
    PlanTranscript &transcript; // records (query) or replays (bind) the product searches
    const Plan *sealed = nullptr; // bind: the sealed plan, whose Newton plan is in `divide` already
    Assembly(Plan &plan,ParseTreePlan &t,PlanTranscript &record,const Plan *source=nullptr) noexcept
        :p(plan),tree(t),transcript(record),sealed(source){}
    int root = -1;
    int integer_root=-1;
    union {sbn3_newton_plan divide;};
    sbn3_newton_info divide_info{};
    Chain power_chain{}; // odd^(64 power_fragments) with the winners of its products
    alignas(CompactParsePlan) unsigned char compact_storage[sizeof(CompactParsePlan)];
    CompactParsePlan &compact_plan() noexcept {return *reinterpret_cast<CompactParsePlan *>(compact_storage);}
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
                                                              : power_bits(base.base_bound, s.integer_digits))
                                                   : 0;
    info.limbs = std::max<size_t>(1, limbs_for_bits(integer_bits + s.fraction_bits));
    const bool fractional=s.fraction_digits&&s.fraction_bits;
    p.fused_scale=fractional&&!p.options.repeated;
    p.split_parts=fractional&&s.integer_digits;
    p.number_digits=fractional?s.fraction_digits:s.integer_digits;
    p.small_number=!p.shift&&((!fractional&&p.number_digits&&p.number_digits<=group_fragments*fragment_digits)||
                            (fractional&&p.digits<=group_fragments*fragment_digits&&s.fraction_bits<=64*parse_leaf_limbs));
    if(p.small_number&&fractional){
        // Keep the minimal word-only binding while word division is the
        // shared division policy. Larger quotient work uses its prepared
        // two-vector inverse through the compact recipe, even in one leaf.
        const size_t nn=limbs_for_bits(power_bits(base.base_bound,p.digits)+s.fraction_bits);
        const size_t dn=limbs_for_bits(power_bits(base.base_bound,s.fraction_digits));
        if(small_division_u52(nn,dn))p.small_number=false;
    }
    if(!p.shift&&!p.small_number&&(fractional?p.digits:p.number_digits)<=compact_parse_fragments*fragment_digits&&
       (!fractional||s.fraction_bits<=64*compact_parse_precision)){
        p.compact=compact_parse_query(s.base,size_t(fractional?p.digits:p.number_digits),
                                      size_t(fractional?s.fraction_digits:0),s.fraction_bits,compact_plan());
        if(p.compact&&p.options.repeated){
            // A retained spectrum is useful once either tree powers or the
            // scaling reciprocal reach the cached FFT band. Keep that
            // capability rather than replacing it with fresh local products.
            auto &q=compact_plan();RailProductPlan cached;
            bool keep_cache=fractional&&limbs_for_bits(s.fraction_bits)>=rail_product_min_limbs/2&&
                limbs_for_bits(power_bits(base.base_bound,s.fraction_digits))>=rail_product_min_limbs/2;
            for(unsigned j=0;!keep_cache&&j<q.node_count;++j){const auto &mul=q.products[q.nodes[j].product];
                keep_cache=rail_linear_plan(mul.an,mul.bn,cached);}
            if(keep_cache)p.compact=false;
        }
        if(p.compact){p.small_number=true;p.tree_id=compact_parse_identity(compact_plan());}
    }
    if(p.small_number&&fractional){p.number_digits=p.digits;p.split_parts=false;p.fused_scale=false;}
    p.quotient_limbs=p.split_parts?limbs_for_bits(s.fraction_bits+1):info.limbs;
    info.lease_peak = 1;
    size_t values = 0, work = 0;
    auto take = [&](size_t limbs) {
        const size_t at = values;
        values += align_to(limbs, 8);
        return at;
    };
    unsigned classes = 0;
    size_t prepared_alignment = 4096, divide_alignment = 4096;
    if (!p.shift&&p.number_digits&&!p.small_number) {
        p.fragments = (p.number_digits + fragment_digits - 1) / fragment_digits;
        const uint64_t integer_fragments=p.split_parts?(s.integer_digits+fragment_digits-1)/fragment_digits:0;
        if (p.fragments > max_fragments)
            return SBN3_QUERY_CAPACITY;
        auto rc = parse_tree_begin(s.base, p.options.workers, std::max(p.fragments,integer_fragments), tree);
        tree_begun = true;
        if (rc != SBN3_SUPPORTED)
            return rc;
        tree.transcript = &transcript;
        tree.repeated=p.options.repeated!=0;
        root = tree.add_tree(p.fragments);
        if(p.split_parts){integer_root=tree.add_tree(integer_fragments);
            if(integer_root<0)return tree.status;
            p.integer_limbs=tree.classes[integer_root].limbs;}
        p.power_fragments = fractional?s.fraction_digits / fragment_digits:0;
        p.power_rest = fractional?unsigned(s.fraction_digits % fragment_digits):0;
        if (p.power_fragments)
            tree.rail.count = std::max(tree.rail.count, top_bit(p.power_fragments) + 1);
        p.number_limbs = integer_limbs(tree.base, p.fragments);
        size_t n = 0, power_limbs = 0;
        if (fractional) {
            // Quotient precision: N 2^(a + |d|) / d with a = 64 (n - number_limbs); M takes all but g bits.
            const uint64_t power_low = uint64_t(floorl(base.log2_odd * (long double)s.fraction_digits * (1 - 1e-15L)));
            const uint64_t twos = uint64_t(base.twos) * s.fraction_digits;
            power_limbs = limbs_for_bits(power_bits(base.odd_bound, s.fraction_digits));
            n = std::max<size_t>({4, p.number_limbs, power_limbs});
            const uint64_t have = uint64_t(64) * (n - p.number_limbs) + power_low + twos;
            const uint64_t need = s.fraction_bits + guard_quotient_bits + 2;
            if (have < need)
                n += limbs_for_bits(need - have);
            p.divide_limbs = n;
            if(!sealed)memset(&divide,0,sizeof divide);
            if(!p.fused_scale)p.quotient_product = tree.add_product(p.number_limbs, n + 1);
        }
        rc = tree.finish();
        if (rc != SBN3_SUPPORTED)
            return rc;
        classes = tree.class_count;
        p.tree_bytes=tree.packed_bytes();
        p.tree_id = tree_identity(tree);
        require(p.number_limbs == tree.classes[root].limbs, SBN3_FATAL_MATH, "radix parse root size");
        p.rail_at = take(tree.rail.total_limbs);
        p.number_capacity=tree.classes[root].slot_limbs;
        if(p.split_parts)p.number_capacity=std::max(p.number_capacity,tree.classes[integer_root].slot_limbs);
        p.number_at = take(p.number_capacity);
        work = std::max(tree.rail.setup_bytes, tree.work_bytes(root));
        if(p.split_parts)work=std::max(work,tree.work_bytes(integer_root));
        prepared_alignment = std::max<size_t>(4096, tree.programs.alignment);
        if (fractional) {
            if (!p.fused_scale&&reciprocal_is_basecase(n)) {
                work = std::max(work, reciprocal_basecase_bytes(n)); // one schoolbook division at bind
            } else if (sealed) {
                // The Newton plan came with the plan value; its resources are sealed.
                divide_info.storage_bytes = sealed->divide_bytes;
                divide_info.storage_alignment = sealed->divide_alignment;
                divide_info.lease_peak = sealed->divide_lease_peak;
            } else {
                sbn3_newton_options o{};
                o.workers = p.options.workers;
                rc = sbn3_newton_query(p.fused_scale?SBN3_NEWTON_DIVIDE:SBN3_NEWTON_INVERSE, n, &o, &divide, &divide_info);
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
            size_t product = align_to((p.quotient_limbs + n) * 8, 64) + 256;
            if (std::max(p.quotient_limbs, n) > chain_basecase_limbs) {
                ProductShape shape{};
                rc = product_shape(p.quotient_limbs, n, p.options.workers, shape, nullptr, &transcript);
                if (rc != SBN3_SUPPORTED)
                    return rc;
                p.repair_choice=shape.choice;p.repair_output_limbs=shape.output_limbs;
                // temporary_bytes already includes the caller's result.
                product = shape.temporary_bytes() + 256;
            }
            work = std::max({work, chain, product, divide_info.storage_bytes,
                             p.fused_scale?size_t(0):tree.extra[p.quotient_product].episode_bytes()});
        }
        info.lease_peak += 3;
    }
    info.storage_alignment = std::max<size_t>({4096, prepared_alignment, divide_alignment});
    info.control_bytes = align_to(program_offset(p.tree_bytes) + size_t(classes) * (sizeof(ProductProgram) + sizeof(RailProduct)) + 64, 4096);
    if(p.compact)info.control_bytes=align_to(align_to(sizeof(Binding),64)+align_to(sizeof(CompactParsePlan),64)+sizeof(CompactParseState),4096);
    p.prepared_offset = align_to(info.control_bytes, prepared_alignment);
    p.prepared_bytes = !p.tree_bytes ? 0 : align_to(tree.prepared_bytes() + 128, 4096);
    p.values_offset = align_to(p.prepared_offset + p.prepared_bytes, 4096);
    p.values_bytes = align_to(p.compact?compact_plan().persistent_bytes+compact_plan().work_bytes:values*8,4096);
    p.divide_offset = 0;
    p.divide_bytes = p.divide_limbs ? divide_info.storage_bytes : 0; // bind-time use of the work range
    p.work_offset = align_to(p.values_offset + p.values_bytes, divide_alignment);
    p.work_bytes = !p.tree_bytes ? 0 : align_to(work + 64, 4096);
    info.storage_bytes = p.work_offset + p.work_bytes;
    info.table_bytes = p.prepared_bytes;
    info.value_bytes = p.values_bytes;
    info.workspace_bytes = p.work_bytes;
    info.divide_bytes = p.divide_bytes;
    if(p.compact){info.table_bytes=compact_plan().persistent_bytes;info.value_bytes=0;
        info.workspace_bytes=compact_plan().work_bytes;++info.lease_peak;}
    if constexpr(!SBN3_CHECK_SMALL)if(!p.tree_bytes)info.lease_peak=0;
    return SBN3_SUPPORTED;
}
// The sealed plan; the Newton plan and the transcript (ready to replay) of the plan value. `divide` and
// `transcript` come in value-initialised.
Plan load(const sbn3_parse_plan &opaque, sbn3_newton_plan &divide, PlanTranscript &transcript,CompactParsePlan &compact) noexcept {
    const auto *bytes = reinterpret_cast<const unsigned char *>(opaque.opaque);
    Plan p{};
    memcpy(static_cast<void *>(&p), bytes, sizeof p);
    require(p.marker==plan_magic,SBN3_FATAL_ARGUMENT,"parse plan marker");
    if(p.compact){memcpy(&compact,bytes+value_divide_at,sizeof compact);
        if constexpr(SBN3_CHECK_SMALL)require(p.tree_id==compact_parse_identity(compact)&&p.seal==seal(p,divide,transcript),
                                             SBN3_FATAL_ARGUMENT,"compact parse plan");
        transcript.replay=true;return p;
    }
    uint64_t count = 0;
    memcpy(&count, bytes + value_count_at, 8);
    require(p.marker == plan_magic && count <= PlanTranscript::capacity, SBN3_FATAL_ARGUMENT, "parse plan");
    if (p.divide_limbs)
        memcpy(&divide, bytes + value_divide_at, sizeof divide);
    transcript.count = uint32_t(count);
    memcpy(transcript.entry, bytes + value_entries_at, size_t(count) * 8);
    if(SBN3_CHECK_SMALL||p.tree_bytes)require(p.seal == seal(p, divide, transcript), SBN3_FATAL_ARGUMENT, "parse plan");
    transcript.replay = true;
    return p;
}
Binding &binding(sbn3_parse_binding *opaque) noexcept {
    auto *b = reinterpret_cast<Binding *>(opaque);
    require(b && b->marker == binding_magic, SBN3_FATAL_LIFETIME, "parse binding");
    if(SBN3_CHECK_SMALL||b->plan.tree_bytes)
        require(pthread_equal(b->team->creator, pthread_self()) && !b->team->busy, SBN3_FATAL_TEAM, "parse owner/idle");
    return *b;
}
void window(sbn3_team *team, uint64_t *out, size_t n, const uint64_t *m, size_t count, int64_t position) noexcept {
    parallel_limbs::each(team, n, parallel_limbs::parts(team, n), [&](size_t b, size_t e, unsigned) {
        dyadic_window(out + b, e - b, m, count, position + int64_t(64) * int64_t(b), 0, uint64_t(64) * count);
    });
}
// Base 2^s: digit j (0 = most significant) holds the bits [pos, pos + s), pos = s (integer - 1 - j) + bits,
// of the result; output limb i collects the digits that reach into [64 i, 64 i + 64). Bits below 0 are
// the truncation.
bool power_of_two(Binding &b, const unsigned char *digits, uint64_t *out) noexcept {
    const auto &p=b.plan;const unsigned s=p.shift;
    const auto *decode=p.options.use_alphabet?b.decode:nullptr;
    bool valid=true;
    const unsigned workers=std::min(p.options.workers,sbn3_team_workers(b.team));
    parallel_limbs::each(b.team,size_t(p.digits),p.digits>=(uint64_t(1)<<16)?workers:1,
        [&](size_t begin,size_t end,unsigned){
            if(validate_digits(digits+begin,end-begin,p.spec.base,decode)!=end-begin)
                __atomic_store_n(&valid,false,__ATOMIC_RELAXED);
        });
    if(!valid)return false;
    const size_t chunks=(p.info.limbs+s-1)/s;
    const int64_t origin=int64_t(p.spec.fraction_bits)-int64_t(s)*int64_t(p.spec.fraction_digits);
    parallel_limbs::each(b.team,chunks,p.info.limbs>=(size_t(1)<<13)?workers:1,
        [&](size_t begin,size_t end,unsigned){parse_bits_range(out,p.info.limbs,begin,end,digits,p.digits,s,origin,decode);});
    return true;
}

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
    return validate_digits(digits,size_t(count),64,decode);
}
} // namespace
bool parse_once_local_query(const sbn3_parse_spec&s,const sbn3_radix_options*o,sbn3_parse_info&info,sbn3_query_result&rc)noexcept{
    if(!s.integer_digits||s.integer_digits>16384||s.fraction_digits||s.fraction_bits)return false;
    sbn3_parse_plan plan;bool handled;auto options=o?*o:sbn3_radix_options{};if(!o)options.workers=1;options.memory_budget=0;
    info={};rc=small_query(s,&options,plan,info,handled);if(!handled||rc!=SBN3_SUPPORTED)return handled;
    info.storage_bytes=info.workspace_bytes;info.control_bytes=info.lease_peak=0;info.plan_id=0;
    if(o&&o->memory_budget&&info.storage_bytes>o->memory_budget)rc=SBN3_QUERY_CAPACITY;
    return true;
}
bool parse_once_local_raw(const sbn3_parse_spec&s,const sbn3_radix_options*o,const uint8_t*in,uint64_t*out,
                          void*work,size_t bytes,sbn3_parse_once_result*result)noexcept{
    if(!s.integer_digits||s.integer_digits>16384||s.fraction_digits||s.fraction_bits)return false;
    sbn3_parse_plan plan;sbn3_parse_info info{};bool handled;auto options=o?*o:sbn3_radix_options{};if(!o)options.workers=1;options.memory_budget=0;
    const auto rc=small_query(s,&options,plan,info,handled);if(!handled)return false;
    require(rc==SBN3_SUPPORTED&&bytes>=info.workspace_bytes&&(!info.workspace_bytes||(work&&!(uintptr_t(work)&63))),SBN3_FATAL_WORKSPACE,"raw local parse storage");
    SmallPlan p;memcpy(&p,plan.opaque,small_header);if(p.flags&1)memcpy(p.alphabet,reinterpret_cast<const uint8_t*>(plan.opaque)+small_header,64);
    SmallBinding b;b.plan={p.digits,p.limbs,p.base,p.flags,p.work_bytes};
    if(!(p.flags&9))b.digits=&identity_digit_plan(unsigned(p.base));
    else if(p.flags&8)b.digits=&decimal_ascii_plan();
    else{::new(&b.custom)DigitPlan;
        if(p.base&(p.base-1))require(digit_plan_init(b.custom,unsigned(p.base),p.alphabet),SBN3_FATAL_ARGUMENT,"raw parse alphabet");
        else{memset(b.custom.decode,0xff,256);for(unsigned j=0;j<p.base;++j)b.custom.decode[p.alphabet[j]]=uint8_t(j);}
        b.digits=&b.custom;}
    if((p.base&(p.base-1))&&p.digits>128)b.leaf=&parse_leaf_plan(unsigned(p.base));
    sbn3_parse_result r;small_arithmetic(b,in,{out,p.limbs},work,&r);
    *result={r.valid,r.invalid_index,r.valid?p.limbs:0,r.exact_fallback};return true;
}
} // namespace sbn::v3::radix
using namespace sbn::v3;
using namespace sbn::v3::radix;
[[gnu::noinline]] static sbn3_query_result parse_query_general(const sbn3_parse_spec *spec, const sbn3_radix_options *options,
                                              sbn3_parse_plan *out, sbn3_parse_info *info) {
    Plan p{};
    p.marker = plan_magic;
    p.spec = *spec;
    if (options)
        p.options = *options;
    else
        p.options.workers = 1;
    if (!p.options.use_alphabet)
        memset(p.options.alphabet, 0, sizeof p.options.alphabet);
    // Raw storage: parse_tree_begin constructs the tree plan, for plans that have a tree.
    alignas(ParseTreePlan) unsigned char tree_storage[sizeof(ParseTreePlan)];
    auto &tree = *reinterpret_cast<ParseTreePlan *>(tree_storage);
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
    auto *bytes = reinterpret_cast<unsigned char *>(out->opaque);
    memcpy(bytes, static_cast<const void *>(&p), sizeof p);
    if(p.compact){memcpy(bytes+value_divide_at,&a.compact_plan(),sizeof(CompactParsePlan));return SBN3_SUPPORTED;}
    if (p.divide_limbs)
        memcpy(bytes + value_divide_at, &a.divide, sizeof a.divide);
    const uint64_t count = transcript.count;
    memcpy(bytes + value_count_at, &count, 8);
    memcpy(bytes + value_entries_at, transcript.entry, size_t(count) * 8);
    return SBN3_SUPPORTED;
}
[[gnu::noinline]] static void parse_bind_general(const sbn3_parse_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                                sbn3_parse_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "parse bind arguments");
    sbn3_newton_plan divide;
    PlanTranscript transcript{};
    alignas(CompactParsePlan) unsigned char compact_storage[sizeof(CompactParsePlan)];
    auto &compact=*reinterpret_cast<CompactParsePlan *>(compact_storage);
    const Plan p = load(*opaque, divide, transcript,compact);
    const bool checked=SBN3_CHECK_SMALL||p.tree_bytes;
    require(arena->accepts_team(team->arena),SBN3_FATAL_TEAM,"parse arena/team");
    if(checked){
    require(arena->accepts_team(team->arena) && team->width >= p.options.workers && pthread_equal(team->creator, pthread_self()) &&
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
    }
    auto control = checked?arena->acquire(offset,p.info.control_bytes):sbn3_lease{arena->base+offset,p.info.control_bytes,0};
    auto *b = ::new (control.data) Binding(p);
    if(p.tree_bytes)::new(&b->tree)ParseTree{};
    else if(!p.shift)::new(&b->direct)Binding::Direct{};
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->control = control;
    // The assembly again, product by product from the transcript; its identity must be the sealed one.
    Plan replay = p;
    alignas(ParseTreePlan) unsigned char tree_storage[sizeof(ParseTreePlan)];
    auto &tree=*reinterpret_cast<ParseTreePlan *>(tree_storage);
    Assembly a{replay, tree, transcript, &p};
    if(p.divide_limbs)a.divide = divide;
    if(p.tree_bytes)require(a.run() == SBN3_SUPPORTED && replay.tree_id == p.tree_id && replay.info.storage_bytes == p.info.storage_bytes &&
                replay.work_bytes == p.work_bytes && replay.prepared_bytes == p.prepared_bytes &&
                replay.values_bytes == p.values_bytes && replay.tree_bytes==p.tree_bytes && replay.info.control_bytes == p.info.control_bytes,
            SBN3_FATAL_MATH, "parse bind plan replay");
    b->root = a.root;
    b->integer_root=a.integer_root;
    if(p.divide_limbs)b->divide = divide;
    if (!p.tree_bytes) {
        if(p.compact){
            auto *at=static_cast<uint8_t *>(control.data)+align_to(sizeof(Binding),64);
            b->compact_plan=::new(at)CompactParsePlan(compact);
            b->compact_state=::new(at+align_to(sizeof(CompactParsePlan),64))CompactParseState{};
            b->values=checked?arena->acquire(offset+p.values_offset,p.values_bytes):
                              sbn3_lease{arena->base+offset+p.values_offset,p.values_bytes,0};
            require(digit_plan_init(b->digits(),p.spec.base,p.options.use_alphabet?p.options.alphabet:nullptr),
                    SBN3_FATAL_ARGUMENT,"compact parse alphabet");
            compact_parse_prepare(*b->compact_plan,*b->compact_state,b->values.data);
            b->marker=binding_magic;*out=reinterpret_cast<sbn3_parse_binding *>(b);return;
        }
        if(p.small_number){
            require(digit_plan_init(b->digits(),p.spec.base,p.options.use_alphabet?p.options.alphabet:nullptr),
                    SBN3_FATAL_ARGUMENT,"small parse alphabet");
            b->leaf()=parse_leaf_plan(p.spec.base);
            if(p.spec.fraction_digits&&p.spec.fraction_bits){
                uint64_t power[parse_leaf_limbs]{1};size_t used=1,left=size_t(p.spec.fraction_digits);
                while(left){const unsigned count=unsigned(std::min<size_t>(8,left));
                    uint64_t factor=1;for(unsigned j=0;j<count;++j)factor*=p.spec.base;
                    uint64_t carry=0;
                    for(size_t j=0;j<used;++j){const __uint128_t x=__uint128_t(power[j])*factor+carry;
                        power[j]=uint64_t(x);carry=uint64_t(x>>64);}
                    if(carry)power[used++]=carry;
                    left-=count;
                }
                b->small_divisor=divrem_words::prepare(power,used,b->small_denominator,true);
            }
        }
        else{memset(b->decode, 0xff, sizeof b->decode);
            for (unsigned j = 0; j < p.spec.base; ++j)
                b->decode[p.options.use_alphabet ? p.options.alphabet[j] : uint8_t(j)] = uint8_t(j);}
        b->marker = binding_magic;
        *out = reinterpret_cast<sbn3_parse_binding *>(b);
        return;
    }
    b->tree_view=tree.pack(static_cast<uint8_t *>(control.data)+align_to(sizeof(Binding),64),p.tree_bytes);
    b->prepared = arena->acquire(offset + p.prepared_offset, p.prepared_bytes);
    b->values = arena->acquire(offset + p.values_offset, p.values_bytes);
    b->work = arena->acquire(offset + p.work_offset, p.work_bytes);
    auto *programs = reinterpret_cast<ProductProgram *>(static_cast<uint8_t *>(control.data) + program_offset(p.tree_bytes));
    auto *cached = reinterpret_cast<RailProduct *>(programs + b->tree_plan().class_count);
    parse_tree_bind(b->tree, b->tree_plan(), p.options.use_alphabet ? p.options.alphabet : nullptr, *arena, *team,
                    b->prepared, b->work, b->limbs(p.rail_at), programs, cached);
    if (p.divide_limbs) {
        // denominator = odd^fraction_digits, normalized to divide_limbs limbs
        const auto &base = b->tree_plan().base;
        const size_t n = p.divide_limbs;
        Frame scratch = Frame::borrow(*arena, b->work, b->work.data, b->work.bytes);
        const size_t capacity = limbs_for_bits(power_bits(base.odd_bound, p.spec.fraction_digits)) + 8;
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
            const uint64_t *big = chain_evaluate(chain, b->tree_plan().rail, b->tree.rail, team, p.options.workers, scratch);
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
    if (!p.fused_scale&&reciprocal_is_basecase(p.divide_limbs)) {
        Frame scratch = Frame::borrow(*arena, b->work, b->work.data, b->work.bytes);
        reciprocal_basecase(b->limbs(p.denominator_at), p.divide_limbs, b->limbs(p.inverse_at), scratch);
    } else if (p.divide_limbs&&!p.fused_scale) {
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
    if(p.divide_limbs&&!p.fused_scale)
        scale_product_complete(b->tree.extra[p.quotient_product],b->limbs(p.inverse_at));
    b->marker = binding_magic;
    *out = reinterpret_cast<sbn3_parse_binding *>(b);
}
[[gnu::noinline]] static void parse_execute_general(sbn3_parse_binding *opaque, const unsigned char *digits, sbn3_limbs out,
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
    if(!p.tree_bytes){
        if(p.compact){
            result.valid=compact_parse_execute(*b.compact_plan,*b.compact_state,b.digits(),digits,size_t(p.number_digits),
                                               p.spec.fraction_bits,out.data,p.info.limbs,b.values.data);
            if(!result.valid)result.invalid_index=first_invalid(digits,p.digits,b.digits().decode);
            else if(p.number_digits<p.digits){const auto bad=first_invalid(digits+p.number_digits,p.digits-p.number_digits,b.digits().decode);
                if(bad!=p.digits-p.number_digits){result.valid=0;result.invalid_index=p.number_digits+bad;}}
            *result_out=result;return;
        }
        if(p.small_number){
            const auto &d=b.digits();
            uint64_t value[parse_leaf_limbs];
            if(!parse_leaf(b.leaf(),d,digits,size_t(p.number_digits),value)){
                result.valid=0;result.invalid_index=first_invalid(digits,p.digits,d.decode);*result_out=result;return;}
            size_t used=b.leaf().limbs[(p.number_digits+63)/64];
            while(used&&!value[used-1])--used;
            if(b.small_divisor.limbs){
                // One leaf and at most one leaf's worth of fractional bits.
                // Exact word division needs no tree, Newton ladder,
                // guard product or boundary repair. Its divisor is retained.
                uint64_t numerator[2*parse_leaf_limbs],quotient[2*parse_leaf_limbs],
                         remainder[parse_leaf_limbs],scratch[2*parse_leaf_limbs+1];
                const size_t nn=used+limbs_for_bits(p.spec.fraction_bits);
                dyadic_window(numerator,nn,value,used,-int64_t(p.spec.fraction_bits),0,64*used);
                const size_t qn=divrem_words::schoolbook_prepared(quotient,remainder,numerator,nn,b.small_divisor,scratch);
                window(b.team,out.data,p.info.limbs,quotient,qn,0);
            }else window(b.team,out.data,p.info.limbs,value,used,-int64_t(p.spec.fraction_bits));
            if(p.number_digits<p.digits){const auto bad=first_invalid(digits+p.number_digits,p.digits-p.number_digits,d.decode);
                if(bad!=p.digits-p.number_digits){result.valid=0;result.invalid_index=p.number_digits+bad;}}
            *result_out=result;return;
        }
        const auto invalid=first_invalid(digits,p.digits,b.decode);
        result.valid=invalid==p.digits;result.invalid_index=invalid;
        std::fill_n(out.data,p.info.limbs,uint64_t(0));*result_out=result;return;
    }
    const size_t n = p.divide_limbs;
    uint64_t *number = b.limbs(p.number_at);
    uint64_t saved_boundary=0;
    if(p.split_parts){
        if(!parse_tree_run(b.tree,b.integer_root,digits,p.spec.integer_digits,number,p.number_capacity)){
            result.valid=0;result.invalid_index=first_invalid(digits,p.digits,b.digits().decode);*result_out=result;return;}
        window(b.team,out.data,p.info.limbs,number,p.integer_limbs,-int64_t(p.spec.fraction_bits));
        saved_boundary=out.data[p.spec.fraction_bits/64];
    }
    if (!parse_tree_run(b.tree, b.root, digits+(p.split_parts?p.spec.integer_digits:0), p.number_digits, number,p.number_capacity)) {
        result.valid = 0;
        result.invalid_index = first_invalid(digits, p.digits, b.digits().decode);
        *result_out = result;
        return;
    }
    if (!n) {
        // A zero fractional precision only needs integer arithmetic, while
        // every fractional input byte still participates in validation.
        if(p.number_digits<p.digits){
            const auto invalid=first_invalid(digits+p.number_digits,p.digits-p.number_digits,b.digits().decode);
            if(invalid!=p.digits-p.number_digits){result.valid=0;result.invalid_index=p.number_digits+invalid;*result_out=result;return;}
        }
        window(b.team, out.data, p.info.limbs, number, p.number_limbs, -int64_t(p.spec.fraction_bits));
        *result_out = result;
        return;
    }
    // q = floor(N * reciprocal / B^number_limbs): within four units of N 2^(bits + g) / b^Df.
    const uint64_t g = b.quotient_shift;
    int state = 0;
    bool overflow=false;
    auto extract=[&](const uint64_t *q){
        overflow=bits_at(q,n+1,int64_t(g)+int64_t(64)*int64_t(p.quotient_limbs))!=0;
        window(b.team,out.data,p.quotient_limbs,q,n+1,int64_t(g));state=boundary(q,g);
    };
    if(p.fused_scale){
        auto *q=b.limbs(p.inverse_at);
        std::fill_n(q,n+1,uint64_t(0));memcpy(q+n-p.number_limbs,number,p.number_limbs*8);
        divide_scale(b.divide,*b.arena,b.offset+p.work_offset,b.team,b.work,q,b.limbs(p.denominator_at),n);
        b.tree.work=b.work;extract(q);
    }else{
        const auto &shape = b.tree_plan().extra[p.quotient_product];
        Frame scratch = Frame::borrow(*b.arena, b.work, b.work.data, b.work.bytes);
        const auto *z=scale_product_execute(b.tree.extra[p.quotient_product],shape,scratch,b.team,number,b.limbs(p.inverse_at));
        const uint64_t *q = z + p.number_limbs;
        extract(q);
    }
    if (state||overflow) {
        // The true quotient is within a few units of candidate * 2^g: decide candidate or candidate - 1 exactly.
        result.exact_fallback = 1;
        if (state == 2&&!overflow)
            overflow=parallel_limbs::add_word(out.data, p.quotient_limbs, 1)!=0;
        if(overflow){
            // The candidate is at most one integer unit above the exact
            // answer, and the shape proves that the answer fits this span.
            // A candidate of 2^(64*limbs) therefore resolves to one less.
            // Do not lose that carry and multiply back the wrapped zero.
            require(!p.split_parts,SBN3_FATAL_MATH,"fraction candidate beyond guarded span");
            std::fill_n(out.data,p.quotient_limbs,UINT64_MAX);
            *result_out=result;return;
        }
        Frame scratch = Frame::borrow(*b.arena, b.work, b.work.data, b.work.bytes);
        const size_t an = p.quotient_limbs;
        uint64_t *product = nullptr;
        if (std::max(an, n) <= chain_basecase_limbs) {
            product = scratch.alloc<uint64_t>(an + n);
            sbn3_mul_basecase(product, an + n, out.data, an, b.limbs(p.denominator_at), n);
        } else {
            require(p.repair_choice&&p.repair_output_limbs>=an+n,SBN3_FATAL_MATH,"radix exact product plan");
            product = scratch.alloc<uint64_t>(p.repair_output_limbs);
            temporary_product(b.team, p.options.workers, scratch, out.data, an, b.limbs(p.denominator_at), n, product,
                              p.repair_output_limbs,p.repair_choice);
        }
        if (compare_shifted(product, an + n, number, p.number_limbs, b.compare_shift) > 0)
            parallel_limbs::sub_word(out.data, p.quotient_limbs, 1);
    }
    if(p.split_parts){
        const unsigned shift=unsigned(p.spec.fraction_bits%64);
        const size_t at=size_t(p.spec.fraction_bits/64);
        const uint64_t mask=shift?(uint64_t(1)<<shift)-1:0;
        require(!(out.data[at]&~mask),SBN3_FATAL_MATH,"fraction result below one");
        out.data[at]=(out.data[at]&mask)|saved_boundary;
    }
    *result_out = result;
}
[[gnu::noinline]] static sbn3_query_result parse_query_decimal_fraction(const sbn3_parse_spec *spec,const sbn3_radix_options *options,
                                                                      sbn3_parse_plan *out,sbn3_parse_info *info){
    if(!spec->integer_digits&&spec->base==10&&spec->fraction_digits&&spec->fraction_digits<=1920&&
       spec->fraction_bits&&spec->fraction_bits<=64*decimal_fraction_read_binding_limbs&&!(spec->fraction_bits%64)&&
       (spec->fraction_bits/64)*((spec->fraction_digits+18)/19)<=108&&options&&options->use_alphabet==1&&decimal_alphabet(options->alphabet)){
        if(!options->workers||options->workers>32)return SBN3_UNSUPPORTED;
        if(options->memory_budget&&options->memory_budget<decimal_fraction_storage)return SBN3_QUERY_CAPACITY;
        DecimalFractionPlan p{decimal_fraction_magic,0,spec->fraction_digits,spec->fraction_bits/64,options->workers};p.id=fraction_identity(p);
        info->limbs=p.limbs;info->exponent2=-int64_t(spec->fraction_bits);info->workers=unsigned(p.workers);
        info->storage_bytes=info->control_bytes=decimal_fraction_storage;info->storage_alignment=64;
        info->lease_peak=SBN3_CHECK_SMALL?1:0;info->plan_id=p.id;memcpy(out->opaque,&p,sizeof p);return SBN3_SUPPORTED;
    }
    return parse_query_general(spec,options,out,info);
}
extern "C" sbn3_query_result sbn3_parse_query(const sbn3_parse_spec *spec, const sbn3_radix_options *options,
                                              sbn3_parse_plan *out, sbn3_parse_info *info) {
    require(spec&&out&&info,SBN3_FATAL_ARGUMENT,"parse query arguments");
    *info={};
    if(spec->integer_digits&&spec->integer_digits<=16384&&!spec->fraction_digits&&!spec->fraction_bits){
        bool handled;const auto status=small_query(*spec,options,*out,*info,handled);if(handled)return status;}
    return parse_query_decimal_fraction(spec,options,out,info);
}
extern "C" void sbn3_parse_bind(const sbn3_parse_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                                sbn3_parse_binding **out) {
    require(opaque&&arena&&team&&out,SBN3_FATAL_ARGUMENT,"parse bind arguments");
    if(opaque->opaque[0]==small_plan_magic){small_bind(*opaque,*arena,offset,*team,out);return;}
    if(opaque->opaque[0]==decimal_fraction_magic){fraction_bind(*opaque,*arena,offset,*team,out);return;}
    parse_bind_general(opaque,arena,offset,team,out);
}
extern "C" void sbn3_parse_execute(sbn3_parse_binding *opaque, const unsigned char *digits, sbn3_limbs out,
                                   sbn3_parse_result *result_out) {
    if(opaque&&reinterpret_cast<const uint64_t*>(opaque)[0]==small_binding_magic){
        small_execute(*reinterpret_cast<SmallBinding*>(opaque),digits,out,result_out);return;}
    if(opaque&&reinterpret_cast<const uint64_t*>(opaque)[0]==decimal_fraction_magic){
        fraction_execute(*reinterpret_cast<DecimalFractionBinding*>(opaque),digits,out,result_out);return;}
    parse_execute_general(opaque,digits,out,result_out);
}
extern "C" void sbn3_parse_unbind(sbn3_parse_binding *opaque) {
    if(opaque&&reinterpret_cast<const uint64_t*>(opaque)[0]==small_binding_magic){
        auto &b=*reinterpret_cast<SmallBinding*>(opaque);auto *arena=b.arena;const auto lease=b.control;
        if constexpr(SBN3_CHECK_SMALL)require(pthread_equal(b.team->creator,pthread_self())&&!b.team->busy,SBN3_FATAL_TEAM,"small parse unbind");
        b.marker=0;b.~SmallBinding();if constexpr(SBN3_CHECK_SMALL)arena->release(lease);return;
    }
    if(opaque&&reinterpret_cast<const uint64_t*>(opaque)[0]==decimal_fraction_magic){
        auto &b=*reinterpret_cast<DecimalFractionBinding*>(opaque);auto *arena=b.arena;const auto lease=b.lease;
        if constexpr(SBN3_CHECK_SMALL)require(pthread_equal(b.team->creator,pthread_self())&&!b.team->busy,SBN3_FATAL_TEAM,"decimal fraction unbind");
        b.plan.marker=0;b.~DecimalFractionBinding();if constexpr(SBN3_CHECK_SMALL)arena->release(lease);return;
    }
    auto &b = binding(opaque);
    auto *arena = b.arena;
    const auto control = b.control;
    if (b.plan.work_bytes)
        arena->release(b.work);
    if (b.plan.values_bytes&&(SBN3_CHECK_SMALL||b.plan.tree_bytes))
        arena->release(b.values);
    if (b.plan.prepared_bytes)
        arena->release(b.prepared);
    const bool checked=SBN3_CHECK_SMALL||b.plan.tree_bytes;
    b.marker = 0;
    b.~Binding();
    if(checked)arena->release(control);
}
