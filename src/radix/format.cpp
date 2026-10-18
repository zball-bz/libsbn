// Format service: dyadic number -> digits (include/sbn3/radix.h).
//
//   integer part   a = floor(X):  root of the tree is (a + 3/4) / b^(64 F): the numerator times the
//                  reciprocal of b^(64 F), a Newton INVERSE computed once at bind (error below four
//                  units of the root). The last fragment is exact because the true value sits 3/4
//                  above a digit boundary.
//   fraction part  f = frac(X):   root is the top limbs of f, no division. The tree returns floor or
//                  floor - 1 of f b^(64 F + 8); only a guard of all (b-1) digits leaves the requested
//                  prefix open, and then it is settled exactly (see resolve()).
//   base 2^s       linear bit extraction.
#include "sbn3/radix.h"
#include "sbn3/newton.h"
#include "radix/format_tree.hpp"
#include "radix/format_frontier.hpp"
#include "radix/bits.hpp"
#include "radix/word_base.hpp"
#include "radix/short_fraction_tuning.hpp"
#include "radix/decimal_small.hpp"
#include "radix/once.hpp"
#include "radix/digit_bounds.hpp"
#include "radix/power_bank.hpp"
#include "backend/u52/kernels.hpp"
#include "common/identity.hpp"
#include "radix/scale.hpp"
#include "radix/word_base.hpp"
#include "value/divrem_words.hpp"
#include "product/word_window.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <new>
#include <string.h>
namespace sbn::v3::radix {
namespace {
constexpr uint64_t plan_magic = 0x53424e3352465031ULL, binding_magic = 0x53424e3352464231ULL;
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
enum IntegerPath : unsigned { integer_none = 0, integer_schoolbook = 1, integer_tree = 2,integer_scaled = 3 };
constexpr uint64_t short_plan_magic=0x53424e3352465331ULL,short_binding_magic=0x53424e3352465332ULL;
struct ShortPlan {uint64_t marker,id,words,digits,base,workers,flags,method,root,work_bytes;uint8_t alphabet[64];};
constexpr size_t short_header=offsetof(ShortPlan,alphabet);
struct ShortBinding {
    struct Inputs {uint64_t words,digits,base,flags,method,root,work_bytes;};
    uint64_t marker;Inputs plan;Arena *arena;sbn3_team *team;sbn3_lease control;
    const DigitPlan *digits;const uint64_t *inverse;unsigned shift;
    union {DigitPlan custom;};
    ShortBinding() noexcept {}
};
constexpr size_t short_control=align_to(sizeof(ShortBinding),64);
uint64_t short_identity(const ShortPlan &p) noexcept {
    uint64_t h=identity::fnv_seed;const uint64_t f[]={p.words,p.digits,p.base,p.workers,p.flags,p.method,p.root,p.work_bytes};
    for(auto v:f)h=identity::word(h,v);
    if(p.flags&1)for(unsigned j=0;j<64;j+=8){uint64_t v;memcpy(&v,p.alphabet+j,8);h=identity::word(h,v);}
    return h;
}
bool short_recipe(const sbn3_format_spec &s,const sbn3_radix_options *o,ShortPlan &p,sbn3_query_result &status) noexcept {
    const bool fraction=s.exponent2&&s.base==10&&s.limbs&&s.limbs<=256&&
        s.exponent2==-int64_t(64*s.limbs)&&s.mode==SBN3_RADIX_EXACT&&
        s.fraction_digits<=(uint64_t(1)<<40)&&o&&o->use_alphabet==1&&decimal_alphabet(o->alphabet)&&
        !tree_policy().fraction_recipe;
    if((!fraction&&(s.exponent2||s.fraction_digits))||!s.limbs||s.limbs>256)return false;
    status=SBN3_UNSUPPORTED;
    const unsigned workers=o?o->workers:1;
    if(s.base<2||s.base>64||!workers||workers>32||unsigned(s.mode)>1||(o&&o->use_alphabet>1))return true;
    p.marker=short_plan_magic;p.words=s.limbs;p.base=s.base;p.workers=workers;
    p.flags=o?(uint64_t(o->use_alphabet)|(uint64_t(o->repeated!=0)<<1)):0;p.root=0;p.work_bytes=0;
    const bool bits=!(s.base&(s.base-1));
    const uint64_t digits=bits?(64*s.limbs+unsigned(__builtin_ctz(s.base))-1)/unsigned(__builtin_ctz(s.base)):
                                 integer_digit_capacity(64*s.limbs,s.base);
    // The existing one-word kernel emits the service's whole padded block
    // directly. The unpadded u64 primitive belongs in decimal.h instead.
    const bool decimal=!fraction&&s.base==10&&s.limbs>1&&s.limbs<=decimal_write_limbs&&(p.flags&1)&&decimal_alphabet(o->alphabet);
    p.digits=fraction?s.fraction_digits:(digits+63)&~uint64_t(63);p.method=fraction?4:decimal?3:bits?0:1;
    if(fraction&&s.limbs>decimal_fraction_write_limbs){
        BaseInfo base;base_info(10,base);p.root=std::min(p.digits,64*s.limbs);
        // Keep the compact descriptor when handing off to the existing
        // carry-deferred IFMA leaf. A tree still uses the general service.
        if(!short_fraction_kind(base,s.limbs,p.root,false,o->repeated))return false;
        p.method=5;p.work_bytes=serial_fraction_words(s.limbs,p.root,10)*8;
    }
    if(!bits&&!fraction&&!decimal){
        const auto &policy=tree_policy();const size_t chunks=(p.digits+word_bases[s.base].digits-1)/word_bases[s.base].digits;
        const size_t work=s.limbs*chunks;
        if(policy.integer_scaled_work&&work>=policy.integer_scaled_work&&
           (!(p.flags&2)||s.limbs<policy.integer_tree_limbs_repeated)){
            const size_t n=limbs_for_bits(power_bits({log2_floor_q56[s.base]},p.digits+8)+guard_bits);
            if(fixed_integer_inverse(s.base,unsigned(p.digits/64),n).data){
                p.method=2;p.root=n;if(n*(n+1)>=policy.group_u52_work)p.flags|=4;
                p.work_bytes=8*(align_to(n+1,8)+align_to(2*n+1,8))+
                    std::max(p.flags&4?u52::scratch_bytes(n,n+1):size_t(0),serial_fraction_words(n,p.digits,s.base)*8);
            }
        }
        if(p.method==1){
            if((p.flags&2)?s.limbs>=policy.integer_tree_limbs_repeated:work>=policy.integer_tree_work)return false;
            p.work_bytes=s.limbs==1?0:8*(s.limbs+chunks);
        }
    }
    if(p.flags&1){
        if(s.base==10&&decimal_alphabet(o->alphabet))p.flags=(p.flags&~uint64_t(1))|8;
        else{
            for(unsigned j=0;j<s.base;++j)for(unsigned k=0;k<j;++k)if(o->alphabet[j]==o->alphabet[k])return true;
            memcpy(p.alphabet,o->alphabet,64);
        }
    }
    p.work_bytes=align_to(p.work_bytes,64);status=SBN3_SUPPORTED;return true;
}
void short_info(const ShortPlan &p,sbn3_format_info &info,bool binding) noexcept {
    const bool fraction=p.method==4||p.method==5;
    info.workers=binding?unsigned(p.workers):1;
    if(fraction){info.integer_digits=info.fraction_offset=0;info.fraction_digits=p.digits;info.digit_bytes=align_to(p.digits,64);}
    else{info.integer_digits=p.digits;info.fraction_offset=info.digit_bytes=p.digits;}
    info.control_bytes=binding?short_control:0;info.workspace_bytes=p.work_bytes;
    info.storage_bytes=info.control_bytes+p.work_bytes;info.storage_alignment=64;
    info.lease_peak=SBN3_CHECK_SMALL&&(binding||p.work_bytes)?1:0;info.plan_id=binding?p.id:0;
}
bool short_query(const sbn3_format_spec &s,const sbn3_radix_options *o,sbn3_format_plan &out,
                 sbn3_format_info &info,sbn3_query_result &status) noexcept {
    ShortPlan p;if(!short_recipe(s,o,p,status))return false;
    if(status!=SBN3_SUPPORTED)return true;
    p.id=short_identity(p);short_info(p,info,true);
    if(o&&o->memory_budget&&info.storage_bytes>o->memory_budget){status=SBN3_QUERY_CAPACITY;return true;}
    memcpy(out.opaque,&p,short_header+(p.flags&1?64:0));return true;
}
void short_bind(const sbn3_format_plan &opaque,Arena &arena,size_t offset,sbn3_team &team,sbn3_format_binding **out) noexcept {
    ShortPlan p;memcpy(&p,opaque.opaque,short_header);
    if(p.flags&1)memcpy(p.alphabet,reinterpret_cast<const uint8_t*>(opaque.opaque)+short_header,64);
    const size_t bytes=short_control+p.work_bytes;
    if constexpr(SBN3_CHECK_SMALL){
        require(p.id==short_identity(p)&&p.base>=2&&p.base<=64&&p.words&&p.words<=256&&p.method<=5&&
            !(p.flags&~uint64_t(15))&&(!(p.flags&8)||(p.base==10&&!(p.flags&1))),SBN3_FATAL_ARGUMENT,"short format plan");
        require(arena.accepts_team(team.arena)&&team.width>=p.workers&&pthread_equal(team.creator,pthread_self())&&!team.busy,SBN3_FATAL_TEAM,"short format team");
        size_t end;require(add_size(offset,bytes,end)&&end<=arena.virtual_bytes&&arena.contains(offset,end)&&
            !(uintptr_t(arena.base+offset)&63)&&arena.unleased(offset,bytes),SBN3_FATAL_WORKSPACE,"short format range");
    }
    sbn3_lease control{};if constexpr(SBN3_CHECK_SMALL)control=arena.acquire(offset,bytes);
    auto *b=::new(arena.base+offset)ShortBinding;b->marker=short_binding_magic;
    b->plan={p.words,p.digits,p.base,p.flags,p.method,p.root,p.work_bytes};
    b->arena=&arena;b->team=&team;b->control=control;
    b->inverse=nullptr;b->shift=0;
    if(!(p.flags&9))b->digits=&identity_digit_plan(unsigned(p.base));
    else if(p.flags&8)b->digits=&decimal_ascii_plan();
    else{::new(&b->custom)DigitPlan;
        if(p.method)require(digit_plan_init(b->custom,unsigned(p.base),p.alphabet),SBN3_FATAL_ARGUMENT,"short format alphabet");
        else memcpy(b->custom.encode,p.alphabet,64);
        b->digits=&b->custom;}
    if(p.method==2){const auto inv=fixed_integer_inverse(unsigned(p.base),unsigned(p.digits/64),p.root);
        require(inv.data,SBN3_FATAL_MATH,"short format reciprocal");b->inverse=inv.data;b->shift=inv.shift;}
    *out=reinterpret_cast<sbn3_format_binding*>(b);
}
void short_execute(ShortBinding &,sbn3_const_limbs,uint8_t *,sbn3_format_result *) noexcept;
struct Plan {
    uint64_t marker = 0;
    sbn3_format_spec spec{};
    sbn3_radix_options options{};
    sbn3_format_info info{};
    unsigned shift = 0; // base == 2^shift, else 0
    int64_t point = 0;  // bit index of the binary point inside the mantissa
    uint64_t integer_bits = 0, fraction_bits = 0;
    unsigned integer_path = integer_none;
    size_t integer_limbs = 0;
    uint64_t integer_fragments = 0, fraction_fragments = 0;
    uint64_t tree_digits = 0; // fraction digits the tree decides (the rest of an EXACT request are zeros)
    size_t divide_limbs = 0;
    bool fused_scale=false;
    bool published_scale=false;
    size_t scaled_product_at=0,scaled_work_at=0,scaled_work_bytes=0;
    bool scaled_native=false;
    size_t pool_offset = 0, pool_bytes = 0;
    size_t prepared_offset = 0, prepared_bytes = 0, values_offset = 0, values_bytes = 0, divide_offset = 0,
           divide_bytes = 0, work_offset = 0, work_bytes = 0;
    size_t tree_work_bytes = 0; // live with the ring pool; setup/repair may use a larger overlapping lease
    size_t tree_bytes = 0; // packed live plan directory in the control reservation
    ProductChoice witness_choice=0;
    size_t rail_at = 0, side_at = 0, numerator_at = 0, inverse_at = 0, small_at = 0; // limbs in values
    size_t fraction_small_at = 0, fraction_small_limbs = 0;
    unsigned fraction_small_codec=0; // 0: exact word, 1: IFMA, 2: guarded word
    int root_product = -1; // tree.extra index of numerator * reciprocal
    size_t divide_alignment = 0;   // of the Newton INVERSE storage (integer tree)
    unsigned divide_lease_peak = 0;
    uint64_t tree_id = 0, seal = 0;
};
// The plan value: [sealed layout][Newton plan of the reciprocal][number of recorded searches][the transcript].
// Bind assembles the tree plan again (it is far larger than a plan value) from these: no search runs twice for
// one conversion. Only the parts a plan uses are written and read (a plan without a tree is a few hundred bytes
// of a zeroed value).
constexpr size_t value_divide_at = (sizeof(Plan) + 7) & ~size_t(7), value_count_at = value_divide_at + sizeof(sbn3_newton_plan),
                 value_entries_at = value_count_at + 8;
static_assert(value_entries_at + PlanTranscript::capacity * 8 <= sizeof(sbn3_format_plan));
uint64_t seal(const Plan &p, const sbn3_newton_plan &divide, const PlanTranscript &t) noexcept {
    uint64_t h = identity::fnv_seed;
    if constexpr(!SBN3_CHECK_SMALL)if(!p.tree_bytes){
        const uint64_t fields[]={p.spec.base,p.spec.limbs,uint64_t(p.spec.exponent2),p.spec.fraction_digits,
            uint64_t(p.spec.mode),p.options.workers,p.options.repeated,p.fraction_small_codec,p.integer_path,p.info.storage_bytes};
        for(uint64_t v:fields)h=identity::word(h,v);
        if(p.options.use_alphabet)for(unsigned j=0;j<64;j+=8){uint64_t v;memcpy(&v,p.options.alphabet+j,8);h=identity::word(h,v);}
        return h;
    }
    const uint64_t fields[] = {p.marker, p.spec.base, p.spec.limbs, uint64_t(p.spec.exponent2), p.spec.fraction_digits,
                               uint64_t(p.spec.mode), p.options.workers, p.options.memory_budget, p.options.use_alphabet, p.options.repeated,
                               p.shift, uint64_t(p.point), p.integer_bits, p.fraction_bits, p.integer_path, p.integer_limbs,
                               p.integer_fragments, p.fraction_fragments, p.tree_digits, p.divide_limbs,uint64_t(p.fused_scale),uint64_t(p.published_scale),
                               p.scaled_product_at,p.scaled_work_at,p.scaled_work_bytes,uint64_t(p.scaled_native),p.prepared_offset,
                               p.prepared_bytes, p.values_offset, p.values_bytes, p.divide_offset, p.divide_bytes,
                               p.work_offset, p.work_bytes, p.tree_work_bytes, p.tree_bytes, p.witness_choice, p.pool_offset, p.pool_bytes, p.rail_at, p.side_at, p.numerator_at, p.inverse_at, uint64_t(p.root_product),
                               p.small_at, p.fraction_small_at, p.fraction_small_limbs,p.fraction_small_codec,
                               p.tree_id, p.info.storage_bytes, p.info.digit_bytes,
                               p.info.fraction_digits, p.info.integer_digits, p.info.control_bytes,
                               p.divide_alignment, p.divide_lease_peak, t.count,
                               p.info.storage_alignment, p.info.lease_peak, p.info.workers, p.info.fraction_offset};
    for (uint64_t f : fields)
        h = identity::word(h, f);
    for (unsigned j = 0; j < 64; j += 8) {
        uint64_t w = 0;
        memcpy(&w, p.options.alphabet + j, 8);
        h = identity::word(h, w);
    }
    for (uint32_t j = 0; j < t.count; ++j)
        h = identity::word(h, t.entry[j]);
    if (p.divide_limbs&&!p.published_scale)h=identity::words(h,divide.opaque,sizeof(divide.opaque)/sizeof(uint64_t));
    return h;
}
uint64_t tree_identity(const FormatTreeView &t) noexcept {
    uint64_t h = identity::fnv_seed;
    auto cache_identity=[&](const SplitPlan &p){if(!p.cyclic.enabled)return;
        const auto &q=p.cyclic;const auto &s=q.shape;
        h=identity::word(h,uint64_t(s.nfull)<<32|s.branch);
        h=identity::word(h,uint64_t(s.radix)<<32|uint64_t(s.bits)<<16|unsigned(s.recipe)|unsigned(s.balanced)<<8|unsigned(s.centered)<<9);
        h=identity::word(h,q.common_limbs);h=identity::word(h,q.fresh_limbs);
    };
    h = identity::word(h, t.class_count);
    h = identity::word(h, t.rail.count);
    h = identity::word(h, t.rail.fixed_levels);
    h = identity::word(h,t.repeated);
    h = identity::word(h,t.balanced);
    h = identity::word(h,t.block_leaves);
    if(!t.repeated)for(unsigned j=0;j<t.powers.count;++j){h=identity::word(h,t.powers.entries[j].exponent);h=identity::word(h,t.powers.entries[j].retained);h=identity::word(h,t.powers.entries[j].cached_square);}
    h = identity::word(h, t.group_u52);
    h = identity::word(h, t.prepared_bytes());
    for(unsigned j=0;j<t.extra_count;++j)h=identity::word(h,t.extra[j].arithmetic_id);
    for (unsigned j = 0; j < t.class_count; ++j) {
        h = identity::word(h, t.classes[j].fragments);
        h = identity::word(h, t.classes[j].seams);
        h = identity::word(h,t.classes[j].left_fragments);
        h = identity::word(h,t.classes[j].slot_limbs);
        h = identity::word(h, t.classes[j].region_bytes + t.classes[j].persist_bytes);
        h = identity::word(h, t.classes[j].split.product.arithmetic_id + t.classes[j].split.cyclic.ring);
        h = identity::word(h, t.classes[j].split.middle_words);
        h = identity::word(h, uint64_t(t.classes[j].split.cache_slot));
        cache_identity(t.classes[j].split);
        h = identity::word(h, t.classes[j].split.middle_shift);
        if(t.classes[j].split.middle_words)h=identity::word(h,t.classes[j].split.product.work_bytes);
    }
    for (unsigned k = 0; k < t.tree_count; ++k)
        for (unsigned s = 0; s < t.trees[k].stage_count; ++s) {
            const auto &stage = t.trees[k].stages[s];
            h = identity::word(h, uint64_t(stage.node_class) << 32 | stage.groups << 8 | stage.workers);
            h = identity::word(h,stage.borrow_output);
            h = identity::word(h,stage.borrow_result);
            h = identity::word(h,uint64_t(stage.partner_class));h=identity::word(h,stage.main_count);
            h = identity::word(h, stage.split.product.arithmetic_id + stage.split.cyclic.ring);
            cache_identity(stage.split);
            h = identity::word(h, stage.split.middle_words);
            h = identity::word(h, stage.split.middle_shift);
            if(stage.split.middle_words)h=identity::word(h,stage.split.product.work_bytes);
        }
    return h;
}
unsigned top_bit(uint64_t v) noexcept { return 63 - unsigned(__builtin_clzll(v)); }
// ---- binding -----------------------------------------------------------------------------------------
struct Binding {
    uint64_t marker = 0;
    union {Plan plan;};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, prepared{}, values{}, work{};
    // Tree state is present only for tree recipes; direct/bit paths keep a
    // small control object instead of reserving a maximum-size tree plan.
    FormatTreeView *tree_plan_data=nullptr;
    FormatTreeView &tree_plan() noexcept {return *tree_plan_data;}
    const FormatTreeView &tree_plan()const noexcept {return *tree_plan_data;}
    union {FormatTree tree;DigitPlan direct_digits;};
    const DigitPlan &digits() const noexcept {return plan.tree_bytes?tree.digits:
        plan.options.use_alphabet?direct_digits:identity_digit_plan(plan.spec.base);}
    int integer_root = -1, fraction_root = -1;
    union {sbn3_newton_plan divide;}; // constructed only when a scaling ladder exists
    unsigned denominator_shift = 0; // numerator = (4a + 3) << denominator_shift
    const uint64_t *published_inverse=nullptr;
    const uint64_t *inverse()const noexcept {return published_inverse?published_inverse:limbs(plan.inverse_at);}
    uint8_t encode[64]{};           // base 2^s digits
    Binding() noexcept {}
    uint64_t *limbs(size_t at) const noexcept { return static_cast<uint64_t *>(values.data) + at; }
};
size_t tree_plan_offset() noexcept {return align_to(sizeof(Binding),64);}
size_t program_offset(size_t tree_bytes) noexcept {return align_to(tree_bytes?tree_plan_offset()+tree_bytes:sizeof(Binding),64);}
// ---- plan --------------------------------------------------------------------------------------------
struct Assembly {
    Plan &p;
    FormatTreePlan *tree;
    PlanTranscript &transcript; // records (query) or replays (bind) the product searches
    const Plan *sealed = nullptr; // bind: the sealed plan, whose Newton plan is in `divide` already
    Assembly(Plan &plan,FormatTreePlan *t,PlanTranscript &record,const Plan *source=nullptr) noexcept
        :p(plan),tree(t),transcript(record),sealed(source){}
    int integer_root = -1, fraction_root = -1;
    sbn3_newton_plan divide;
    sbn3_newton_info divide_info{};
    PlanSlots<Chain,1> integer_chain{}; // live only for the integer-tree recipe
    bool tree_begun = false; // `tree` is raw storage until *_tree_begin constructed it
    sbn3_query_result run() noexcept;
    sbn3_query_result assemble() noexcept;
};
sbn3_query_result Assembly::run() noexcept {
    const auto rc = assemble();
    if (tree_begun)
        tree->transcript = nullptr; // the tree plan outlives the transcript
    return rc;
}
sbn3_query_result Assembly::assemble() noexcept {
    const auto &s = p.spec;
    BaseInfo base{};
    if (!base_info(s.base, base) || !s.limbs || unsigned(s.mode) > SBN3_RADIX_ENCLOSED || !p.options.workers ||
        p.options.workers > 32 || p.options.use_alphabet > 1)
        return SBN3_UNSUPPORTED;
    if (s.limbs > (size_t(1) << 40) || s.exponent2 > (int64_t(1) << 46) || s.exponent2 < -(int64_t(1) << 46) ||
        s.fraction_digits > (uint64_t(1) << 40))
        return SBN3_QUERY_CAPACITY;
    if (s.mode == SBN3_RADIX_ENCLOSED && s.exponent2 > 0)
        return SBN3_UNSUPPORTED;
    if (p.options.use_alphabet)
        for (unsigned i = 0; i < s.base; ++i)
            for (unsigned j = 0; j < i; ++j)
                if (p.options.alphabet[i] == p.options.alphabet[j])
                    return SBN3_UNSUPPORTED;
    p.shift = base.odd == 1 ? base.twos : 0;
    p.point = -s.exponent2;
    const int64_t width = int64_t(64) * int64_t(s.limbs);
    p.integer_bits = p.point >= width ? 0 : uint64_t(width - p.point);
    p.fraction_bits = p.point > 0 ? uint64_t(p.point) : 0;
    p.integer_limbs = limbs_for_bits(p.integer_bits);
    auto &info = p.info;
    info = {};
    info.workers = p.options.workers;
    uint64_t integer_digits = 0;
    if (p.integer_bits)
        integer_digits = p.shift ? (p.integer_bits + p.shift - 1) / p.shift
                                 : integer_digit_capacity(p.integer_bits,s.base);
    p.integer_fragments = (integer_digits + fragment_digits - 1) / fragment_digits;
    info.integer_digits = p.integer_fragments * fragment_digits;
    uint64_t digits = s.fraction_digits;
    if (s.mode == SBN3_RADIX_ENCLOSED) {
        // Digits that an input known to 2^-fraction_bits can determine, keeping a guard of one word.
        uint64_t cap = 0;
        if (p.shift)
            cap = p.fraction_bits / p.shift;
        else if (p.fraction_bits > 2) {
            const uint64_t d = fraction_digit_capacity(p.fraction_bits-2,s.base);
            cap = d > word_digits ? d - word_digits : 0;
        }
        digits = std::min(digits, cap);
    }
    info.fraction_digits = digits;
    p.tree_digits = digits;
    if (!p.fraction_bits)
        p.tree_digits = 0;
    else if (base.twos) // a dyadic fraction ends after ceil(bits / twos) digits in an even base
        p.tree_digits = std::min(digits, (p.fraction_bits + base.twos - 1) / base.twos);
    p.fraction_fragments = p.shift ? 0 : (p.tree_digits + fragment_digits - 1) / fragment_digits;
    // Short fractions extract maximal radix powers with word/IFMA kernels.
    // Price the actual rectangular work, not just the mantissa length. A
    // repeated binding retains the tree's lower execution-only crossover.
    const size_t fraction_limbs=limbs_for_bits(p.fraction_bits);
    const auto fraction_recipe=tree_policy().fraction_recipe;
    const unsigned short_kind=!p.shift&&p.fraction_fragments&&!fraction_recipe?
        short_fraction_kind(base,fraction_limbs,p.tree_digits,s.mode==SBN3_RADIX_ENCLOSED,p.options.repeated):0;
    if(!p.shift && p.fraction_fragments &&
       (fraction_recipe==1 || fraction_recipe==3 || short_kind)){
        p.fraction_small_limbs=fraction_limbs;
        p.fraction_small_codec=fraction_recipe==3||short_kind==2;
        if(!p.fraction_small_codec&&s.mode==SBN3_RADIX_EXACT&&fraction_limbs>=16&&
           power_bits(base.base_bound,p.tree_digits)<=64*fraction_limbs+64)p.fraction_small_codec=2;
        p.fraction_fragments=0;
    }
    info.fraction_offset = size_t(info.integer_digits);
    info.digit_bytes = info.fraction_offset + size_t(align_to(digits, fragment_digits));
    if (p.fraction_fragments > max_fragments || p.integer_fragments > max_fragments)
        return SBN3_QUERY_CAPACITY;
    // Schoolbook or tree (TreePolicy): by execute time for repeated conversions, by the whole call for one.
    const auto &policy = tree_policy();
    const bool integer_by_tree = p.options.repeated ? p.integer_limbs >= policy.integer_tree_limbs_repeated
                                 : p.integer_limbs >= (size_t(1) << 20) ||
                                       p.integer_limbs * size_t((info.integer_digits+word_bases[s.base].digits-1)/word_bases[s.base].digits) >= policy.integer_tree_work;
    p.integer_path = !p.integer_bits || p.shift ? integer_none : integer_by_tree ? integer_tree : integer_schoolbook;
    if(p.integer_path!=integer_none&&policy.integer_scaled_work&&
       (!p.options.repeated||p.integer_limbs<policy.integer_tree_limbs_repeated)&&
       p.integer_limbs*size_t((info.integer_digits+word_bases[s.base].digits-1)/word_bases[s.base].digits)>=policy.integer_scaled_work&&
       fixed_integer_inverse(s.base,unsigned(p.integer_fragments),node_limbs(base,p.integer_fragments)).data)
        p.integer_path=integer_scaled;
    p.published_scale=p.integer_path==integer_tree&&fixed_integer_inverse(s.base,unsigned(p.integer_fragments),node_limbs(base,p.integer_fragments)).data;
    p.fused_scale=p.integer_path==integer_tree&&!p.options.repeated&&!p.published_scale;
    size_t values = 0, work = 0;
    auto take = [&](size_t limbs) {
        const size_t at = values;
        values += align_to(limbs, 8);
        return at;
    };
    if(p.integer_path==integer_scaled){
        p.divide_limbs=node_limbs(base,p.integer_fragments);p.published_scale=true;
        const size_t n=p.divide_limbs;p.numerator_at=take(n+1);p.scaled_product_at=take(2*n+1);
        p.scaled_native=n*(n+1)>=policy.group_u52_work;
        p.scaled_work_bytes=std::max(p.scaled_native?u52::scratch_bytes(n,n+1):size_t(0),
                                   serial_fraction_words(n,info.integer_digits,s.base)*8);
        p.scaled_work_at=take((p.scaled_work_bytes+7)/8);
    }
    if(p.fraction_small_limbs){const size_t extra=p.fraction_small_codec==1?serial_fraction_words(p.fraction_small_limbs,
            p.tree_digits+(s.mode==SBN3_RADIX_ENCLOSED?word_digits:0),s.base):0;
        p.fraction_small_at=take(align_to(p.fraction_small_limbs,8)+extra);}
    info.lease_peak = 1;
    const bool trees = p.fraction_fragments || p.integer_path == integer_tree;
    unsigned classes = 0;
    if (trees) {
        auto rc = format_tree_begin(s.base, p.options.workers,
                                    std::max(p.integer_path == integer_tree ? p.integer_fragments : 0, p.fraction_fragments), *tree);
        tree_begun = true;
        if (rc != SBN3_SUPPORTED)
            return rc;
        tree->transcript = &transcript;
        tree->select_reuse_recipe(p.options.repeated!=0);
        if (p.integer_path == integer_tree) {
            integer_root = tree->add_tree(p.integer_fragments);
            if(tree->repeated)tree->rail.count = std::max(tree->rail.count, top_bit(p.integer_fragments) + 1);
        }
        if (p.fraction_fragments) {
            fraction_root = tree->add_tree(p.fraction_fragments);
            if(tree->repeated&&s.mode==SBN3_RADIX_EXACT)
                tree->rail.count = std::max(tree->rail.count, top_bit(p.fraction_fragments) + 1);
        }
        if (p.integer_path == integer_tree&&!p.fused_scale) {
            const size_t n = tree->root_limbs(integer_root);
            p.root_product = tree->add_product(n, n + 1,p.options.repeated!=0);
        }
        rc = tree->finish();
        if (rc != SBN3_SUPPORTED)
            return rc;
        classes = tree->class_count;
        p.tree_bytes=tree->packed_bytes();
        p.tree_id = tree_identity(*tree);
        p.rail_at = take(tree->rail.total_limbs);
        p.side_at = take(2 * std::max(p.integer_path == integer_tree ? tree->seam_count(integer_root) : 0,
                                     p.fraction_fragments ? tree->seam_count(fraction_root) : 0));
        work = tree->rail.setup_bytes;
        // A balanced squaring may keep an auxiliary-power prefix leased
        // beside its temporary product table/work leases.
        if(!tree->repeated&&tree->powers.setup_bytes)info.lease_peak+=2;
    }
    if (p.integer_path == integer_tree) {
        if(!sealed&&!p.published_scale)memset(&divide,0,sizeof divide);
        p.divide_limbs = tree->root_limbs(integer_root);
        if(p.published_scale){
            // The exact reciprocal is immutable; only its product is planned.
            divide_info={};divide_info.storage_alignment=64;
        }else if (!p.fused_scale&&reciprocal_is_basecase(p.divide_limbs)) {
            work = std::max(work, reciprocal_basecase_bytes(p.divide_limbs)); // one schoolbook division at bind
        } else if (sealed) {
            // The Newton plan came with the plan value; its resources are sealed.
            divide_info.storage_bytes = sealed->divide_bytes;
            divide_info.storage_alignment = sealed->divide_alignment;
            divide_info.lease_peak = sealed->divide_lease_peak;
        } else {
            sbn3_newton_options o{};
            o.workers = p.options.workers;
            const auto rc = sbn3_newton_query(p.fused_scale?SBN3_NEWTON_DIVIDE:SBN3_NEWTON_INVERSE, p.divide_limbs, &o, &divide, &divide_info);
            if (rc != SBN3_SUPPORTED)
                return rc;
        }
        p.divide_alignment = divide_info.storage_alignment;
        p.divide_lease_peak = divide_info.lease_peak;
        p.numerator_at = take(p.divide_limbs + 1);
        if(!p.published_scale||p.options.repeated)p.inverse_at = take(p.divide_limbs + 1);
        size_t chain = 0;
        if(!p.published_scale){integer_chain.emplace(0,tree->chain(integer_root));
            const auto crc = chain_bytes(tree->base, integer_chain[0], p.options.workers, chain, &transcript);
            if (crc != SBN3_SUPPORTED)return crc;}
        // The reciprocal is computed inside the work range at bind; execute multiplies there, then runs the tree->
        work = std::max({work, chain, tree->work_bytes(integer_root), divide_info.storage_bytes,
                         p.fused_scale?size_t(0):tree->extra[p.root_product].episode_bytes()});
        info.lease_peak += divide_info.lease_peak;
    } else if (p.integer_path == integer_schoolbook) {
        p.small_at = take(p.integer_limbs + (info.integer_digits+word_bases[s.base].digits-1)/word_bases[s.base].digits);
    }
    if (p.fraction_fragments) {
        work = std::max(work, tree->work_bytes(fraction_root));
        if(s.mode==SBN3_RADIX_EXACT){
            // Exact tie resolution: odd^(64 F), a copy of the low fraction limbs, and their product.
            auto chain = tree->chain(fraction_root);
            size_t bytes = 0;
            auto rc = chain_bytes(tree->base, chain, p.options.workers, bytes, &transcript);
            if (rc != SBN3_SUPPORTED)
                return rc;
            const uint64_t scale = uint64_t(tree->base.twos) * fragment_digits * p.fraction_fragments;
            if (p.fraction_bits > scale) {
                const size_t low = limbs_for_bits(p.fraction_bits - scale + 64);
                const size_t an = std::min(low, s.limbs), bn = std::min(low, chain.limbs);
                product::WordWindowPlan witness;
                rc=product::word_window_query(an,bn,p.fraction_bits-scale,p.options.workers,witness,
                                               sealed?sealed->witness_choice:0);
                if(rc!=SBN3_SUPPORTED)return rc;
                p.witness_choice=witness.choice;
                bytes=std::max(bytes,chain.resident_bytes()+witness.bytes+256);
            }
            work = std::max(work, bytes);
        }
    }
    // The ring pool is live only while the conversion tree runs. Rail setup,
    // integer-root construction and exact fraction repair run before/after it.
    // Reserve the maximum of those phases, not the sum of their peaks.
    size_t tree_work = 0;
    if(integer_root >= 0)tree_work=std::max(tree_work,tree->work_bytes(integer_root));
    if(fraction_root >= 0)tree_work=std::max(tree_work,tree->work_bytes(fraction_root));
    p.tree_work_bytes=trees?align_to(tree_work+64,4096):0;
    const size_t prepared_alignment = trees ? std::max<size_t>(4096, tree->prepared_alignment()) : 4096;
    const size_t divide_alignment = p.divide_limbs ? std::max<size_t>(4096, divide_info.storage_alignment) : 4096;
    const size_t pool_alignment = trees ? tree->pool_alignment() : 4096;
    info.storage_alignment = std::max({pool_alignment, prepared_alignment, divide_alignment});
    info.control_bytes = align_to(program_offset(p.tree_bytes) + size_t(trees ? tree->program_slots() : 0) * sizeof(ProductProgram) +
                                      size_t(classes) * sizeof(RailProduct) + size_t(trees ? tree->ring_stages : 0) * sizeof(RingStage) + 192,
                                  4096);
    p.prepared_offset = align_to(info.control_bytes, prepared_alignment);
    p.prepared_bytes = trees ? align_to(tree->prepared_bytes() + 128, 4096) : 0;
    p.values_offset = align_to(p.prepared_offset + p.prepared_bytes, 4096);
    p.values_bytes = align_to(values * 8, 4096);
    p.divide_offset = 0;
    p.divide_bytes = p.divide_limbs ? divide_info.storage_bytes : 0; // bind-time use of the work range
    // The pool of the ring stages stays unleased: the product service leases inside it stage by stage.
    p.pool_bytes = trees ? align_to(tree->pool_bytes(), 4096) : 0;
    p.work_offset = align_to(p.values_offset + p.values_bytes, divide_alignment);
    // Each stage places its pool after its own live episode. A large linear
    // episode and another stage's large ring pool never coexist.
    p.pool_offset = p.work_offset;
    if(integer_root>=0)work=std::max(work,tree->runtime_bytes(integer_root,p.work_offset));
    if(fraction_root>=0)work=std::max(work,tree->runtime_bytes(fraction_root,p.work_offset));
    if (p.pool_bytes)
        info.lease_peak += 1 + 2 * ring_max_groups;
    p.work_bytes = trees ? align_to(work + 64, 4096) : 0;
    info.storage_bytes = p.work_offset+p.work_bytes;
    info.table_bytes = p.prepared_bytes;
    info.value_bytes = p.values_bytes;
    info.workspace_bytes = p.work_bytes;
    info.divide_bytes = p.divide_bytes;
    info.lease_peak += (p.prepared_bytes != 0) + (p.values_bytes != 0) + (p.work_bytes != 0);
    if constexpr(!SBN3_CHECK_SMALL)if(!trees)info.lease_peak=0;
    return SBN3_SUPPORTED;
}
// The sealed plan; the Newton plan and the transcript (ready to replay) of the plan value. `divide` and
// `transcript` come in value-initialised.
Plan load(const sbn3_format_plan &opaque, sbn3_newton_plan &divide, PlanTranscript &transcript) noexcept {
    const auto *bytes = reinterpret_cast<const unsigned char *>(opaque.opaque);
    Plan p{};
    memcpy(static_cast<void *>(&p), bytes, sizeof p);
    uint64_t count = 0;
    memcpy(&count, bytes + value_count_at, 8);
    require(p.marker == plan_magic && count <= PlanTranscript::capacity, SBN3_FATAL_ARGUMENT, "format plan");
    if (p.divide_limbs&&!p.published_scale)
        memcpy(&divide, bytes + value_divide_at, sizeof divide);
    transcript.count = uint32_t(count);
    memcpy(transcript.entry, bytes + value_entries_at, size_t(count) * 8);
    if(SBN3_CHECK_SMALL||p.tree_bytes)require(p.seal == seal(p, divide, transcript), SBN3_FATAL_ARGUMENT, "format plan");
    transcript.replay = true;
    return p;
}
Binding &binding(sbn3_format_binding *opaque) noexcept {
    auto *b = reinterpret_cast<Binding *>(opaque);
    require(b && b->marker == binding_magic, SBN3_FATAL_LIFETIME, "format binding");
    if(SBN3_CHECK_SMALL||b->plan.tree_bytes)
        require(pthread_equal(b->team->creator, pthread_self()) && !b->team->busy, SBN3_FATAL_TEAM, "format owner/idle");
    return *b;
}
// ---- execute pieces ----------------------------------------------------------------------------------
template <class F> void for_digits(sbn3_team *team, uint64_t count,unsigned workers, F &&fn) noexcept {
    const unsigned parts = count >= (uint64_t(1) << 16) ? std::min(workers,sbn3_team_workers(team)) : 1;
    parallel_limbs::each(team, size_t(count), parts, [&](size_t b, size_t e, unsigned) { fn(b, e); });
}
void window(sbn3_team *team, uint64_t *out, size_t n, const uint64_t *m, size_t count, int64_t position,
            uint64_t low, uint64_t high) noexcept {
    parallel_limbs::each(team, n, parallel_limbs::parts(team, n), [&](size_t b, size_t e, unsigned) {
        dyadic_window(out + b, e - b, m, count, position + int64_t(64) * int64_t(b), low, high);
    });
}
void power_of_two(Binding &b, const uint64_t *m, unsigned char *digits) noexcept {
    const auto &p = b.plan;
    const unsigned s = p.shift;
    const size_t count = p.spec.limbs;
    const int64_t point = p.point;
    const uint8_t *encode = b.encode;
    const uint64_t area = p.info.integer_digits;
    // integer digit i (weight base^i) sits at byte area - 1 - i; bits outside the mantissa are zero
    for_digits(b.team, area,p.options.workers, [&](size_t begin, size_t end) {
        if(begin<end)format_bits(digits+begin,end-begin,m,count,s,point+int64_t((area-1-begin)*s),encode);
    });
    unsigned char *fraction = digits + p.info.fraction_offset;
    for_digits(b.team, p.info.fraction_digits,p.options.workers, [&](size_t begin, size_t end) {
        format_bits(fraction+begin,end-begin,m,count,s,point-int64_t((begin+1)*s),encode);
    });
}
size_t integer_words(uint64_t *a,size_t count,uint64_t *words,size_t area,const DigitPlan &dp,uint8_t *digits) noexcept {
    if(count==1&&area==64)return format_limb(digits,a[0],dp);
    const auto radix=word_bases[dp.base];
    const size_t total=(area+radix.digits-1)/radix.digits;
    size_t top = count, produced = 0;
    while (top && !a[top - 1])
        --top;
    while (top) {
        const uint64_t r=divrem_words::divrem_1_prepared(a,a,top,radix.power<<radix.shift,radix.shift,radix.inverse);
        while (top && !a[top - 1])
            --top;
        require(produced < total, SBN3_FATAL_MATH, "radix integer digit capacity");
        words[total - 1 - produced++] = r;
    }
    const size_t emitted=produced*radix.digits;
    size_t skip=emitted>area?emitted-area:0,at=emitted<area?area-emitted:0;
    const size_t first=at;
    memset(digits,dp.encode[0],at);
    for(size_t j=0;j<produced;j+=8){const unsigned count=unsigned(std::min<size_t>(8,produced-j));
        if(skip){uint8_t head[8*64];emit_word_chunks(head,words+total-produced+j,count,dp);
            const size_t n=count*radix.digits-skip;memcpy(digits+at,head+skip,n);at+=n;skip=0;
        }else{emit_word_chunks(digits+at,words+total-produced+j,count,dp);at+=count*radix.digits;}
    }
    size_t begin=first;while(begin<area&&digits[begin]==dp.encode[0])++begin;return begin;
}
void schoolbook(Binding &b,const uint64_t *m,uint8_t *digits) noexcept {
    const auto &p=b.plan;auto *a=b.limbs(p.small_at);
    dyadic_window(a,p.integer_limbs,m,p.spec.limbs,p.point,p.point>0?uint64_t(p.point):0,uint64_t(64)*p.spec.limbs);
    integer_words(a,p.integer_limbs,a+p.integer_limbs,p.info.integer_digits,b.digits(),digits);
}
[[gnu::always_inline]] inline void short_arithmetic(const ShortBinding::Inputs &p,const DigitPlan &dp,
        const uint64_t *inverse,unsigned inv_shift,uint64_t *a,sbn3_const_limbs input,uint8_t *digits,sbn3_format_result *result) noexcept {
    if(p.method==3){*result={decimal_write_padded(digits,p.digits,input.data,p.words),0,0};return;}
    if(p.method==4){decimal_fraction_write(digits,p.digits,input.data,p.words);*result={0,p.digits,0};return;}
    if(p.method==5){auto *work=a;
        serial_fraction_emit(digits,nullptr,input.data,p.words,p.root,0,dp,work);
        if(p.digits>p.root)memset(digits+p.root,'0',size_t(p.digits-p.root));
        *result={0,p.digits,0};return;}
    size_t first=0;
    if(p.method==0){const unsigned shift=unsigned(__builtin_ctz(unsigned(p.base)));
        format_bits(digits,p.digits,input.data,p.words,shift,int64_t((p.digits-1)*shift),dp.encode);
        if(digits[0]==dp.encode[0]){
            size_t used=input.count;while(used&&!input.data[used-1])--used;
            const unsigned bits=used?unsigned(64*used)-unsigned(__builtin_clzll(input.data[used-1])):0;
            first=p.digits-(bits+shift-1)/shift;
        }
    }else if(p.method==1){
        if(p.words==1)first=format_limb(digits,input.data[0],dp);
        else{memcpy(a,input.data,p.words*8);first=integer_words(a,p.words,a+p.words,p.digits,dp,digits);}
    }else{
        const size_t n=p.root;const unsigned shift=inv_shift;
        dyadic_window(a,n+1,input.data,p.words,-int64_t(shift+2),0,p.words*64);
        a[shift/64]|=uint64_t(3)<<(shift%64);if(shift%64==63)a[shift/64+1]|=1;
        auto *z=a+align_to(n+1,8),*work=z+align_to(2*n+1,8);
        auto scratch=Frame::external(work,p.work_bytes-size_t(work-a)*8);
        if(p.flags&4)u52::multiply(z,a,n,inverse,n+1,scratch);
        else sbn3_mul_basecase(z,2*n+1,a,n,inverse,n+1);
        require(!z[2*n],SBN3_FATAL_MATH,"short integer root");
        auto *serial=scratch.alloc<uint64_t>(serial_fraction_words(n,p.digits,unsigned(p.base)));
        serial_fraction_emit(digits,nullptr,z+n,n,p.digits,0,dp,serial);
    }
    if(p.method==2)first=first_nonzero_digit(digits,p.digits,dp.encode[0]);
    *result={first,0,0};
}
void short_execute(ShortBinding &b,sbn3_const_limbs input,uint8_t *digits,sbn3_format_result *result) noexcept {
    const auto &p=b.plan;
    if constexpr(SBN3_CHECK_SMALL){
        require(result&&input.data&&input.count==p.words&&digits&&!(uintptr_t(input.data)&7),SBN3_FATAL_ARGUMENT,"short format input");
        require(!overlaps(&b,short_control+p.work_bytes,input.data,p.words*8)&&
            !overlaps(&b,short_control+p.work_bytes,digits,p.digits)&&!overlaps(input.data,p.words*8,digits,p.digits),
            SBN3_FATAL_ARGUMENT,"short format overlap");
        require(pthread_equal(b.team->creator,pthread_self())&&!b.team->busy,SBN3_FATAL_TEAM,"short format owner");
    }
    auto *work=reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(&b)+short_control);
    short_arithmetic(p,*b.digits,b.inverse,b.shift,work,input,digits,result);
}

uint64_t run_tree(Binding &b,int root,const uint64_t *y,uint8_t *out,uint64_t *first,uint64_t *overlap) noexcept {
    const auto result=format_tree_run(b.tree,root,y,out,first,overlap);
    // The tree may have reissued this lease while placing stage-local pools.
    // All Frames and products have joined; the original full extent is live.
    b.work=b.tree.work;
    return result;
}
void integer_numerator(Binding &b,const uint64_t *m) noexcept {
    const auto &p = b.plan;
    const size_t n = p.divide_limbs;
    uint64_t *numerator = b.limbs(p.numerator_at);
    const unsigned sh = b.denominator_shift;
    // (4 a + 3) << sh, a = the mantissa bits from the point upward
    window(b.team, numerator, n + 1, m, p.spec.limbs, p.point - int64_t(sh + 2), p.point > 0 ? uint64_t(p.point) : 0,
           uint64_t(64) * p.spec.limbs);
    numerator[sh / 64] |= uint64_t(3) << (sh % 64);
    if (sh % 64 == 63)
        numerator[sh / 64 + 1] |= 1;
}
void integer_scaled_run(Binding &b,const uint64_t *m,unsigned char *digits) noexcept {
    const auto &p=b.plan;const size_t n=p.divide_limbs;integer_numerator(b,m);
    auto *z=b.limbs(p.scaled_product_at);
    auto scratch=Frame::external(b.limbs(p.scaled_work_at),p.scaled_work_bytes);
    if(p.scaled_native)u52::multiply(z,b.limbs(p.numerator_at),n,b.inverse(),n+1,scratch);
    else sbn3_mul_basecase(z,2*n+1,b.limbs(p.numerator_at),n,b.inverse(),n+1);
    require(!z[2*n],SBN3_FATAL_MATH,"integer leaf root");
    auto *work=scratch.alloc<uint64_t>(serial_fraction_words(n,p.info.integer_digits,p.spec.base));
    // The existing +3/4 bias separates the integer from both digit boundaries.
    // The exact shared fractional leaf therefore needs no tie-repair pass.
    serial_fraction_emit(digits,nullptr,z+n,n,p.info.integer_digits,0,b.digits(),work);
}
void integer_tree_run(Binding &b, const uint64_t *m, unsigned char *digits) noexcept {
    const auto &p=b.plan;const size_t n=p.divide_limbs;
    auto *numerator=b.limbs(p.numerator_at);integer_numerator(b,m);
    // root = floor(numerator * reciprocal / B^n): within four units of B^n (a + 3/4) / b^(64 F)
    if(p.fused_scale){
        divide_scale(b.divide,*b.arena,b.offset+p.work_offset,b.team,b.work,numerator,b.limbs(p.inverse_at),n);
        b.tree.work=b.work;
        require(!numerator[n],SBN3_FATAL_MATH,"radix integer division root");
    }else{
        const auto &shape = b.tree_plan().extra[p.root_product];
        Frame scratch = Frame::borrow(*b.arena, b.work, b.work.data, b.work.bytes);
        const auto *z=scale_product_execute(b.tree.extra[p.root_product],shape,scratch,b.team,numerator,b.inverse());
        require(!z[2 * n], SBN3_FATAL_MATH, "radix integer root");
        parallel_limbs::copy(b.team, numerator, z + n, n);
    }
    uint64_t *side = b.limbs(p.side_at);
    run_tree(b, b.integer_root, numerator, digits, side, side + b.tree_plan().seam_count(b.integer_root));
}
// The tree left the requested prefix open: every computed digit after it is b - 1. True when the
// prefix must be incremented.
bool resolve(Binding &b, const uint64_t *m, const unsigned char *digits, sbn3_format_result &result) noexcept {
    const auto &p = b.plan;
    const auto &base = b.tree_plan().base;
    const size_t count = p.spec.limbs;
    const uint64_t bits = p.fraction_bits;
    // frac(X) * base^tree_digits is an integer: the computed string is that integer minus one.
    const uint64_t zeros = trailing_zero_bits(m, count, std::min<uint64_t>(bits, uint64_t(64) * count));
    if (zeros + uint64_t(base.twos) * p.tree_digits >= bits)
        return true;
    // Otherwise compare, modulo 2^64, the computed 64 F digit string S with J = floor(f b^(64F)):
    // S is J or J - 1.
    result.exact_fallback = 1;
    const uint64_t fragments = p.fraction_fragments;
    const auto &dp = b.digits();
    uint64_t computed = 0;
    for (uint64_t f = 0; f < fragments; ++f) {
        uint64_t w[8];
        require(parse_words(w, digits + f * fragment_digits, 8, dp), SBN3_FATAL_MATH, "radix digit readback");
        for (unsigned j = 0; j < 8; ++j)
            computed = computed * dp.b8 + w[j];
    }
    const uint64_t scale = uint64_t(base.twos) * fragment_digits * fragments;
    Frame scratch = Frame::borrow(*b.arena, b.work, b.work.data, b.work.bytes);
    const auto chain = b.tree_plan().chain(b.fraction_root);
    const uint64_t *power = chain_evaluate(chain, b.tree_plan().rail, b.tree.rail, b.team, p.options.workers, scratch,b.tree.cyclic);
    uint64_t exact = 0;
    if (bits <= scale) {
        uint64_t low = 0;
        dyadic_window(&low, 1, m, count, 0, 0, bits);
        exact = scale - bits >= 64 ? 0 : (low * power[0]) << (scale - bits);
    } else {
        const uint64_t t = bits - scale;
        const size_t low = limbs_for_bits(t + 64);
        const size_t an = std::min(low, count), bn = std::min(low, chain.limbs);
        product::WordWindowPlan witness;
        require(product::word_window_query(an,bn,t,p.options.workers,witness,p.witness_choice)==SBN3_SUPPORTED,
                SBN3_FATAL_MATH,"radix exact word plan");
        exact=product::word_window_execute(witness,{m,an},{power,bn},t,scratch,b.team,bits);
    }
    require(exact == computed || exact == computed + 1, SBN3_FATAL_MATH, "radix exact resolution", exact, computed);
    return exact != computed;
}
void fraction_tree_run(Binding &b, const uint64_t *m, unsigned char *out, sbn3_format_result &result) noexcept {
    const auto &p = b.plan;
    const size_t n = b.tree_plan().root_limbs(b.fraction_root), count = p.spec.limbs;
    const int64_t position = p.point - int64_t(64) * int64_t(n);
    const uint64_t *y = nullptr;
    if (position >= 0 && !(p.point & 63) && uint64_t(p.point) <= uint64_t(64) * count) {
        y = m + size_t(position >> 6); // the fraction's top limbs are the root as they stand
    } else {
        uint64_t *copy = format_tree_root_buffer(b.tree,b.fraction_root);
        window(b.team, copy, n, m, count, position, 0, std::min<uint64_t>(p.fraction_bits, uint64_t(64) * count));
        y = copy;
    }
    uint64_t *side = b.limbs(p.side_at);
    const uint64_t fragments = p.fraction_fragments;
    uint64_t overlap = run_tree(b, b.fraction_root, y, out, side, side + b.tree_plan().seam_count(b.fraction_root));
    const auto &dp = b.digits();
    const uint8_t top = dp.encode[dp.base - 1];
    const uint64_t computed = fragments * fragment_digits;
    // The computed 64 F + 8 digits are the exact ones or one below (in the last place). Only a tail of
    // all b - 1 after the requested prefix leaves the prefix open; settle it exactly.
    bool open = overlap == dp.b8 - 1;
    for (uint64_t j = computed; open && j-- > p.tree_digits;)
        open = out[j] == top;
    // A terminating dyadic is cheap to settle in either mode. ENCLOSED does
    // not need a full exact tie product: its output may conservatively stop
    // before an uncertain digit. EXACT retains the deterministic repair.
    const auto &base=b.tree_plan().base;
    const bool terminating=open&&trailing_zero_bits(m,count,std::min<uint64_t>(p.fraction_bits,64*count))+
        uint64_t(base.twos)*p.tree_digits>=p.fraction_bits;
    if (open && (terminating || (p.spec.mode==SBN3_RADIX_EXACT&&resolve(b, m, out, result)))) {
        increment_digits(out, size_t(computed), dp);
        overlap = 0;
    }
    if (p.spec.mode == SBN3_RADIX_EXACT) {
        if (p.tree_digits < p.info.fraction_digits)
            memset(out + p.tree_digits, dp.encode[0], size_t(p.info.fraction_digits - p.tree_digits));
        result.fraction_digits = p.info.fraction_digits;
        return;
    }
    // S is the computed integer at K=computed+8 places, floor(X*b^K)
    // is S or S+1. Put O=computed-digits and U=floor(S/b^O).
    // The input interval adds <1/4 at digits+8 places. Thus the possible
    // integer there is U..U+1 for O>0, and U..U+2 for O=0. A guard that
    // cannot overflow certifies the whole prefix; otherwise return the
    // common prefix of the produced digits and those digits plus one.
    const uint64_t digits = p.info.fraction_digits;
    unsigned char tail[word_digits];
    uint64_t w = overlap;
    for (unsigned j = word_digits; j-- > 0; w /= dp.base)
        tail[j] = dp.encode[w % dp.base];
    auto at = [&](uint64_t j) { return j < computed ? out[j] : tail[j - computed]; };
    uint64_t guard=0;
    for(uint64_t j=digits;j<digits+word_digits;++j)
        guard=guard*dp.base+(dp.identity?at(j):dp.decode[at(j)]);
    const uint64_t uncertainty=computed==digits?2:1;
    const bool shared=guard<=dp.b8-1-uncertainty;
    uint64_t certified = digits;
    if (!shared) {
        while (certified && out[certified - 1] == top)
            --certified;
        certified = certified ? certified - 1 : 0;
    }
    result.fraction_digits = certified;
}
template<bool Trim>void fraction_schoolbook(Binding &b,const uint64_t *m,unsigned char *out,sbn3_format_result &result) noexcept {
    const auto &p=b.plan;const auto &dp=b.digits();
    const size_t n=p.fraction_small_limbs;
    auto *a=b.limbs(p.fraction_small_at);
    if(p.point==int64_t(64*n)&&p.spec.limbs>=n)memcpy(a,m,n*8);
    else dyadic_window(a,n,m,p.spec.limbs,p.point-int64_t(64*n),0,
                       std::min<uint64_t>(p.fraction_bits,uint64_t(64)*p.spec.limbs));
    size_t low=0;while(low<n && !a[low])++low;
    uint64_t zero_bits=64*low+(low<n?unsigned(__builtin_ctzll(a[low])):0);
    const uint64_t digits=p.tree_digits;
    const BinaryLogBound log{log2_floor_q56[dp.base]};
    constexpr bool trim=Trim;
    const uint64_t guard_count=p.spec.mode==SBN3_RADIX_ENCLOSED||trim?word_digits:0;
    const auto wb=word_bases[dp.base];
    const unsigned word_zeros=unsigned(__builtin_ctzll(wb.power));
    const uint64_t total=digits+guard_count;
    uint8_t guard[word_digits]{};
    bool discarded=false;
    const uint64_t rounds=Trim?(total+wb.digits-1)/wb.digits:1;
    const unsigned error_bits=rounds>1?65-unsigned(__builtin_clzll(rounds-1)):1;
    uint64_t remaining_bits=Trim?power_bits(log,total)+error_bits:0;
    const unsigned word_bits=63-wb.shift; // floor(log2(base^K))
    if(p.fraction_small_codec==1)serial_fraction_emit(out,guard,a,n,digits,unsigned(guard_count),dp,a+align_to(n,8));
    else for(uint64_t at=0;at<total;at+=8*wb.digits){
        uint64_t words[8]{};uint8_t block[512];
        const unsigned count=unsigned(std::min<uint64_t>(8,(total-at+wb.digits-1)/wb.digits));
        for(unsigned j=0;j<count;++j){
            if(trim){
                // Each discarded tail changes the final guard integer by
                // <2^-error_bits. Choosing ceil(log2(rounds))+1 keeps their
                // sum below one. Resolve an all-(base-1) guard exactly below.
                // Subtracting floor(log2(base^K)) preserves an upper bound
                // without a logarithm/product in the per-word loop.
                const size_t keep=limbs_for_bits(remaining_bits);
                const size_t drop=keep<n?n-keep:0;
                if(drop>low){if(!discarded)for(size_t k=low;k<drop;++k)discarded|=a[k]!=0;low=drop;zero_bits=64*drop;}
                remaining_bits-=word_bits;
            }
            if(low<n)words[j]=sbn3i_mul_1(a+low,a+low,long(n-low),wb.power);
            // The radix contributes this many known low zero bits. After
            // a precision cut this remains a lower bound; no data-dependent
            // scan of freshly written words is needed in the recurrence.
            zero_bits+=word_zeros;low=std::min(n,size_t(zero_bits/64));
        }
        emit_word_chunks(block,words,count,dp);
        const uint64_t produced=count*wb.digits;
        if(at<digits)memcpy(out+at,block,size_t(std::min<uint64_t>(produced,digits-at)));
        const uint64_t begin=std::max(at,digits),end=std::min(at+produced,total);
        if(end>begin)memcpy(guard+begin-digits,block+begin-at,size_t(end-begin));
    }
    if(p.spec.mode==SBN3_RADIX_EXACT){
        if(trim&&discarded){bool ambiguous=true;for(auto c:guard)ambiguous&=c==dp.encode[dp.base-1];
            if(ambiguous){fraction_schoolbook<false>(b,m,out,result);result.exact_fallback=1;return;}}
        if(digits<p.info.fraction_digits)memset(out+digits,dp.encode[0],size_t(p.info.fraction_digits-digits));
        result.fraction_digits=p.info.fraction_digits;return;
    }
    // Same conservative enclosure rule as the tree, now with exact guard
    // digits. The admitted input interval moves less than one guard unit.
    bool shared=false;const auto top=dp.encode[dp.base-1];
    for(auto c:guard)shared|=c!=top;
    uint64_t certified=p.info.fraction_digits;
    if(!shared){while(certified && out[certified-1]==top)--certified;certified=certified?certified-1:0;}
    result.fraction_digits=certified;
}
} // namespace
bool format_once_local_query(const sbn3_format_spec &s,const sbn3_radix_options *o,sbn3_format_info &info,sbn3_query_result &rc) noexcept {
    ShortPlan p;if(!short_recipe(s,o,p,rc))return false;
    if(rc!=SBN3_SUPPORTED)return true;
    short_info(p,info,false);
    if(o&&o->memory_budget&&info.storage_bytes>o->memory_budget)rc=SBN3_QUERY_CAPACITY;
    return true;
}
static void format_local_arithmetic(const ShortPlan&p,const uint64_t*input,uint8_t*out,void*work,
                                    sbn3_format_once_result*result)noexcept{
    const DigitPlan *dp;
    alignas(DigitPlan) unsigned char custom[sizeof(DigitPlan)];
    if(!(p.flags&9))dp=&identity_digit_plan(unsigned(p.base));
    else if(p.flags&8)dp=&decimal_ascii_plan();
    else{auto *d=::new(custom)DigitPlan;
        if(p.method)require(digit_plan_init(*d,unsigned(p.base),p.alphabet),SBN3_FATAL_ARGUMENT,"format once alphabet");
        else memcpy(d->encode,p.alphabet,64);dp=d;}
    FixedIntegerInverse inv{};
    if(p.method==2){inv=fixed_integer_inverse(unsigned(p.base),unsigned(p.digits/64),p.root);
        require(inv.data,SBN3_FATAL_MATH,"format once inverse");}
    const ShortBinding::Inputs inputs{p.words,p.digits,p.base,p.flags,p.method,p.root,p.work_bytes};
    sbn3_format_result r;
    short_arithmetic(inputs,*dp,inv.data,inv.shift,static_cast<uint64_t*>(work),{input,p.words},out,&r);
    *result={r.integer_first,p.method==4||p.method==5?0:p.digits,r.fraction_digits,r.exact_fallback};
}
bool format_once_local(const sbn3_format_spec &s,const sbn3_radix_options *o,const uint64_t *input,uint8_t *out,
                       const sbn3_radix_workspace *w,sbn3_format_once_result *result) noexcept {
    ShortPlan p;sbn3_query_result rc;if(!short_recipe(s,o,p,rc))return false;
    require(rc==SBN3_SUPPORTED&&(!o||!o->memory_budget||p.work_bytes<=o->memory_budget),SBN3_FATAL_ARGUMENT,"format once local request");
    OnceSpace work(w,p.work_bytes,64);format_local_arithmetic(p,input,out,work.data,result);return true;
}
bool format_once_local_raw(const sbn3_format_spec&s,const sbn3_radix_options*o,const uint64_t*input,uint8_t*out,
                           void*work,size_t bytes,sbn3_format_once_result*result)noexcept{
    ShortPlan p;sbn3_query_result rc;if(!short_recipe(s,o,p,rc))return false;
    require(rc==SBN3_SUPPORTED&&bytes>=p.work_bytes&&(!p.work_bytes||(work&&!(uintptr_t(work)&63))),SBN3_FATAL_WORKSPACE,"format raw local storage");
    format_local_arithmetic(p,input,out,work,result);
    return true;
}
} // namespace sbn::v3::radix
using namespace sbn::v3;
using namespace sbn::v3::radix;
extern "C" void sbn3_radix_alphabet(unsigned base, unsigned char out[64]) {
    require(out, SBN3_FATAL_ARGUMENT, "radix alphabet output");
    static const char lower[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    static const char wide[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/";
    memset(out, 0, 64);
    if (base <= 36)
        memcpy(out, lower, 36);
    else
        memcpy(out, wide, 64);
}
[[gnu::noinline]] static sbn3_query_result format_query_general(const sbn3_format_spec *spec, const sbn3_radix_options *options,
                                               sbn3_format_plan *out, sbn3_format_info *info) {
    Plan p{};
    p.marker = plan_magic;
    p.spec = *spec;
    if (options)
        p.options = *options;
    else
        p.options.workers = 1;
    if (!p.options.use_alphabet)
        memset(p.options.alphabet, 0, sizeof p.options.alphabet);
    // Raw storage: format_tree_begin constructs the tree plan, for plans that have a tree.
    alignas(FormatTreePlan) unsigned char tree_storage[sizeof(FormatTreePlan)];
    auto *tree = reinterpret_cast<FormatTreePlan *>(tree_storage);
    PlanTranscript transcript{};
    Assembly a{p, tree, transcript};
    const auto rc = a.run();
    *info = p.info;
    if (rc != SBN3_SUPPORTED)
        return rc;
    if (p.options.memory_budget && p.info.storage_bytes > p.options.memory_budget)
        return SBN3_QUERY_CAPACITY;
    // The identity covers the transcript and the Newton plan; info.plan_id is the seal, so it is not part of it.
    p.seal = seal(p, a.divide, transcript);
    p.info.plan_id = info->plan_id = p.seal;
    auto *bytes = reinterpret_cast<unsigned char *>(out->opaque);
    memcpy(bytes, static_cast<const void *>(&p), sizeof p);
    if (p.divide_limbs&&!p.published_scale)
        memcpy(bytes + value_divide_at, &a.divide, sizeof a.divide);
    const uint64_t count = transcript.count;
    memcpy(bytes + value_count_at, &count, 8);
    memcpy(bytes + value_entries_at, transcript.entry, size_t(count) * 8);
    return SBN3_SUPPORTED;
}
[[gnu::noinline]] static void format_bind_general(const sbn3_format_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                                 sbn3_format_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "format bind arguments");
    sbn3_newton_plan divide;
    PlanTranscript transcript{};
    const Plan p = load(*opaque, divide, transcript);
    const bool checked=SBN3_CHECK_SMALL||p.tree_bytes;
    require(arena->accepts_team(team->arena),SBN3_FATAL_TEAM,"format arena/team");
    if(checked){
    require(arena->accepts_team(team->arena) && team->width >= p.options.workers && pthread_equal(team->creator, pthread_self()) &&
                !team->busy,
            SBN3_FATAL_TEAM, "format bind team");
    size_t end = 0;
    require(add_size(offset, p.info.storage_bytes, end) && end <= arena->virtual_bytes && arena->contains(offset, end) &&
                !(reinterpret_cast<uintptr_t>(arena->base + offset) & (p.info.storage_alignment - 1)) &&
                arena->unleased(offset, p.info.storage_bytes),
            SBN3_FATAL_WORKSPACE, "format prepared unleased range");
    sbn3_arena_stats stats{};
    sbn3_arena_get_stats(arena, &stats);
    require(stats.active_leases + p.info.lease_peak <= Arena::max_leases, SBN3_FATAL_WORKSPACE, "format lease capacity");
    }
    auto control = checked?arena->acquire(offset,p.info.control_bytes):sbn3_lease{arena->base+offset,p.info.control_bytes,0};
    auto *b = ::new (control.data) Binding{};
    ::new(&b->plan)Plan(p);
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->control = control;
    // The assembly again, product by product from the transcript; its identity must be the sealed one.
    Plan replay = p;
    const bool has_tree=p.fraction_fragments||p.integer_path==integer_tree;
    if(has_tree)::new(&b->tree)FormatTree{};
    else if(!p.shift&&p.options.use_alphabet)::new(&b->direct_digits)DigitPlan{};
    alignas(FormatTreePlan) unsigned char tree_storage[sizeof(FormatTreePlan)];
    auto *tree=reinterpret_cast<FormatTreePlan *>(tree_storage);
    Assembly a{replay, tree, transcript, &p};
    if(p.divide_limbs&&!p.published_scale)a.divide = divide;
    if(has_tree)require(a.run() == SBN3_SUPPORTED && replay.tree_id == p.tree_id && replay.info.storage_bytes == p.info.storage_bytes &&
                replay.work_bytes == p.work_bytes && replay.tree_work_bytes == p.tree_work_bytes &&
                replay.work_offset == p.work_offset && replay.pool_offset == p.pool_offset &&
                replay.prepared_bytes == p.prepared_bytes && replay.pool_bytes == p.pool_bytes &&
                replay.values_bytes == p.values_bytes && replay.info.control_bytes == p.info.control_bytes &&
                replay.tree_bytes==p.tree_bytes,
            SBN3_FATAL_MATH, "format bind plan replay");
    if(has_tree)b->tree_plan_data=tree->pack(static_cast<uint8_t *>(control.data)+tree_plan_offset(),p.tree_bytes);
    b->integer_root = a.integer_root;
    b->fraction_root = a.fraction_root;
    if(p.divide_limbs&&!p.published_scale)b->divide = divide;
    if (p.prepared_bytes)
        b->prepared = arena->acquire(offset + p.prepared_offset, p.prepared_bytes);
    if (p.values_bytes)
        b->values = checked?arena->acquire(offset+p.values_offset,p.values_bytes):
                           sbn3_lease{arena->base+offset+p.values_offset,p.values_bytes,0};
    if (p.work_bytes)
        b->work = arena->acquire(offset + p.work_offset, p.work_bytes);
    if (p.shift) {
        for (unsigned j = 0; j < 64; ++j)
            b->encode[j] = p.options.use_alphabet ? p.options.alphabet[j] : uint8_t(j);
    } else if (!p.prepared_bytes) {
        // schoolbook only: the digit plan alone
        if(p.options.use_alphabet)require(digit_plan_init(b->direct_digits,p.spec.base,p.options.alphabet),
                                         SBN3_FATAL_ARGUMENT,"radix alphabet");
    } else {
        auto *programs = reinterpret_cast<ProductProgram *>(static_cast<uint8_t *>(control.data) +
                                                            program_offset(p.tree_bytes));
        auto *cyclic = reinterpret_cast<RailProduct *>(programs + b->tree_plan().program_slots());
        auto *rings = reinterpret_cast<RingStage *>(
            static_cast<uint8_t *>(control.data) +
            align_to(size_t(reinterpret_cast<uint8_t *>(cyclic + b->tree_plan().class_count) - static_cast<uint8_t *>(control.data)), 64));
        format_tree_bind(b->tree, b->tree_plan(), p.options.use_alphabet ? p.options.alphabet : nullptr, *arena, *team,
                         b->prepared, b->work, b->limbs(p.rail_at), programs, cyclic, rings, offset + p.pool_offset,true);
    }
    if(p.published_scale){
        const auto inv=fixed_integer_inverse(p.spec.base,unsigned(p.integer_fragments),p.divide_limbs);
        require(inv.data,SBN3_FATAL_MATH,"published integer reciprocal");b->denominator_shift=inv.shift;
        // A retained tree repeatedly streams the integer reciprocal. Keep
        // its aligned, binding-local placement while skipping its inverse
        // computation. One-shot leaves borrow the immutable source directly.
        if(p.integer_path==integer_tree&&p.options.repeated)
            memcpy(b->limbs(p.inverse_at),inv.data,(p.divide_limbs+1)*8);
        else b->published_inverse=inv.data;
    }else if (p.integer_path == integer_tree) {
        // Normalized divisor: odd^(64 F) shifted to the top of divide_limbs limbs; the numerator shift
        // absorbs the power of two of the base.
        const auto &base = b->tree_plan().base;
        const size_t n = p.divide_limbs;
        Frame scratch = Frame::borrow(*arena, b->work, b->work.data, b->work.bytes);
        const auto &chain = a.integer_chain[0];
        const uint64_t *power = chain_evaluate(chain, b->tree_plan().rail, b->tree.rail, team, p.options.workers, scratch,b->tree.cyclic);
        size_t top = chain.limbs;
        while (top && !power[top - 1])
            --top;
        require(top && top <= n, SBN3_FATAL_MATH, "radix divisor size");
        const uint64_t length = uint64_t(64) * (top - 1) + (64 - uint64_t(__builtin_clzll(power[top - 1])));
        const uint64_t z = uint64_t(64) * n - length;
        const uint64_t twos = uint64_t(base.twos) * fragment_digits * p.integer_fragments;
        require(z >= twos + 2 && z - twos - 2 < (uint64_t(1) << 31), SBN3_FATAL_MATH, "radix divisor normalization");
        b->denominator_shift = unsigned(z - twos - 2);
        // The normalized divisor goes through the numerator buffer; its reciprocal stays.
        uint64_t *divisor = b->limbs(p.fused_scale?p.inverse_at:p.numerator_at);
        window(team, divisor, n, power, top, -int64_t(z), 0, uint64_t(64) * top);
    }
    if (p.integer_path == integer_tree && !p.fused_scale && !p.published_scale && reciprocal_is_basecase(p.divide_limbs)) {
        Frame scratch = Frame::borrow(*arena, b->work, b->work.data, b->work.bytes);
        reciprocal_basecase(b->limbs(p.numerator_at), p.divide_limbs, b->limbs(p.inverse_at), scratch);
    } else if (p.integer_path == integer_tree&&!p.fused_scale&&!p.published_scale) {
        // Newton INVERSE inside the (still unused) work range: release it for the episode.
        const size_t n = p.divide_limbs;
        arena->release(b->work);
        sbn3_newton_binding *inverse = nullptr;
        sbn3_newton_bind(&b->divide, arena, offset + p.work_offset, team, &inverse);
        const sbn3_newton_inputs in{{}, {b->limbs(p.numerator_at), n}, 0};
        sbn3_newton_execute(inverse, &in, {b->limbs(p.inverse_at), n + 1});
        sbn3_newton_unbind(inverse);
        b->work = arena->acquire(offset + p.work_offset, p.work_bytes);
        b->tree.work = b->work;
    }
    if(p.integer_path==integer_tree&&!p.fused_scale)
        scale_product_complete(b->tree.extra[p.root_product],b->inverse());
    b->marker = binding_magic;
    *out = reinterpret_cast<sbn3_format_binding *>(b);
}
[[gnu::noinline]] static void format_execute_general(sbn3_format_binding *opaque, sbn3_const_limbs mantissa, unsigned char *digits,
                                    sbn3_format_result *out) {
    auto &b = binding(opaque);
    const auto &p = b.plan;
    if(SBN3_CHECK_SMALL||p.tree_bytes){
    require(out && mantissa.data && mantissa.count == p.spec.limbs && (digits || !p.info.digit_bytes) &&
                !(reinterpret_cast<uintptr_t>(mantissa.data) & 7),
            SBN3_FATAL_ARGUMENT, "format execute arguments");
    const auto *storage = b.arena->base + b.offset;
    require(!overlaps(mantissa.data, mantissa.count * 8, storage, p.info.storage_bytes) &&
                !overlaps(digits, p.info.digit_bytes, storage, p.info.storage_bytes) &&
                !overlaps(digits, p.info.digit_bytes, mantissa.data, mantissa.count * 8),
            SBN3_FATAL_ARGUMENT, "format value overlap");
    }
    sbn3_format_result result{};
    const uint64_t *m = mantissa.data;
    unsigned char *fraction = digits + p.info.fraction_offset;
    if (p.shift) {
        power_of_two(b, m, digits);
        result.fraction_digits = p.info.fraction_digits;
    } else {
        if (p.integer_path == integer_schoolbook)
            schoolbook(b, m, digits);
        else if(p.integer_path==integer_scaled)
            integer_scaled_run(b,m,digits);
        else if (p.integer_path == integer_tree)
            integer_tree_run(b, m, digits);
        if(p.fraction_small_limbs){
            if(p.fraction_small_codec==2)
                fraction_schoolbook<true>(b,m,fraction,result);
            else fraction_schoolbook<false>(b,m,fraction,result);
        }else if (p.fraction_fragments) {
            fraction_tree_run(b, m, fraction, result);
        } else {
            // no fraction bits (EXACT: zeros), or nothing certifiable
            const uint8_t zero = p.options.use_alphabet ? p.options.alphabet[0] : 0;
            if (p.info.fraction_digits)
                memset(fraction, zero, size_t(p.info.fraction_digits));
            result.fraction_digits = p.info.fraction_digits;
        }
    }
    const uint8_t zero = p.options.use_alphabet ? p.options.alphabet[0] : 0;
    result.integer_first = p.info.integer_digits?first_nonzero_digit(digits,p.info.integer_digits,zero):0;
    *out = result;
}
extern "C" sbn3_query_result sbn3_format_query(const sbn3_format_spec *spec, const sbn3_radix_options *options,
                                               sbn3_format_plan *out, sbn3_format_info *info) {
    require(spec&&out&&info,SBN3_FATAL_ARGUMENT,"format query arguments");
    *info={};sbn3_query_result status;
    if(short_query(*spec,options,*out,*info,status))return status;
    return format_query_general(spec,options,out,info);
}
extern "C" void sbn3_format_bind(const sbn3_format_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                                 sbn3_format_binding **out) {
    require(opaque&&arena&&team&&out,SBN3_FATAL_ARGUMENT,"format bind arguments");
    if(opaque->opaque[0]==short_plan_magic){short_bind(*opaque,*arena,offset,*team,out);return;}
    format_bind_general(opaque,arena,offset,team,out);
}
extern "C" void sbn3_format_execute(sbn3_format_binding *opaque, sbn3_const_limbs mantissa, unsigned char *digits,
                                    sbn3_format_result *out) {
    if(opaque&&reinterpret_cast<const uint64_t*>(opaque)[0]==short_binding_magic){
        short_execute(*reinterpret_cast<ShortBinding*>(opaque),mantissa,digits,out);return;}
    format_execute_general(opaque,mantissa,digits,out);
}
extern "C" void sbn3_format_unbind(sbn3_format_binding *opaque) {
    if(opaque&&reinterpret_cast<const uint64_t*>(opaque)[0]==short_binding_magic){
        auto &b=*reinterpret_cast<ShortBinding*>(opaque);auto *arena=b.arena;const auto lease=b.control;
        if constexpr(SBN3_CHECK_SMALL)require(pthread_equal(b.team->creator,pthread_self())&&!b.team->busy,SBN3_FATAL_TEAM,"short format unbind");
        b.marker=0;b.~ShortBinding();if constexpr(SBN3_CHECK_SMALL)arena->release(lease);return;
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
namespace sbn::v3::radix {
sbn3_query_result format_frontier_query(unsigned base,size_t limbs,uint64_t fragments,
    const sbn3_radix_options&options,sbn3_format_plan&out,sbn3_format_info&info) noexcept {
    if(!fragments||fragments>max_fragments||!limbs||limbs>(size_t(1)<<40)||options.use_alphabet||(base&(base-1))==0)return SBN3_UNSUPPORTED;
    const sbn3_format_spec spec{base,limbs,-int64_t(64*limbs),fragment_digits*fragments,SBN3_RADIX_EXACT};
    const auto rc=format_query_general(&spec,&options,&out,&info);if(rc!=SBN3_SUPPORTED)return rc;
    sbn3_newton_plan divide{};PlanTranscript transcript{};const auto p=load(out,divide,transcript);
    if(!p.tree_bytes||p.integer_path!=integer_none||p.fraction_fragments!=fragments||p.tree_digits!=spec.fraction_digits)
        return SBN3_UNSUPPORTED;
    return SBN3_SUPPORTED;
}
uint64_t format_frontier_execute(sbn3_format_binding*opaque,sbn3_const_limbs input,uint8_t*out) noexcept {
    auto&b=binding(opaque);const auto&p=b.plan;
    require(p.tree_bytes&&b.fraction_root>=0&&p.integer_path==integer_none&&p.spec.mode==SBN3_RADIX_EXACT&&
        !p.options.use_alphabet&&p.fraction_bits==input.count*64&&input.count==p.spec.limbs&&
        b.tree_plan().root_limbs(b.fraction_root)==input.count&&p.tree_digits==fragment_digits*p.fraction_fragments,
        SBN3_FATAL_ARGUMENT,"native SRT frontier shape");
    valid_span(input.data,input.count*8,"native SRT frontier input");valid_span(out,p.info.digit_bytes,"native SRT frontier output");
    const auto*storage=b.arena->base+b.offset;
    require(!overlaps(input.data,input.count*8,storage,p.info.storage_bytes)&&
        !overlaps(out,p.info.digit_bytes,storage,p.info.storage_bytes)&&!overlaps(input.data,input.count*8,out,p.info.digit_bytes),
        SBN3_FATAL_ARGUMENT,"native SRT frontier alias");
    auto*side=b.limbs(p.side_at);auto overlap=run_tree(b,b.fraction_root,input.data,out,side,
        side+b.tree_plan().seam_count(b.fraction_root));
    if(overlap==b.digits().b8-1){sbn3_format_result result{};
        if(resolve(b,input.data,out,result)){increment_digits(out,size_t(p.tree_digits),b.digits());overlap=0;}}
    return overlap;
}
}
