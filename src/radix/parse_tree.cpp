#include "radix/parse_tree.hpp"
#include "radix/power_bank.hpp"
#include "backend/u52/kernels.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <new>
#include <string.h>
namespace sbn::v3::radix {
namespace {
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
unsigned pow2_floor(unsigned v) noexcept { return 1u << (31 - unsigned(__builtin_clz(v))); }
} // namespace
sbn3_query_result parse_tree_begin(unsigned base, unsigned workers, uint64_t largest_fragments, ParseTreePlan &p) noexcept {
    static_assert(__is_trivially_destructible(ParseTreePlan)); // may be raw storage
    ::new (&p) ParseTreePlan{};
    if (!base_info(base, p.base) || p.base.odd == 1 || !workers || workers > 32)
        return p.status = SBN3_UNSUPPORTED;
    p.workers = workers;
    p.top_workers = pow2_floor(workers);
    for (uint64_t n = 1; n <= group_fragments; ++n)
        p.group_limbs[n] = integer_limbs(p.base, n);
    if (p.group_limbs[group_fragments] > parse_leaf_limbs ||
        p.group_limbs[group_fragments / 2] + rail_limbs(p.base, 2) > parse_leaf_limbs)
        return p.status = SBN3_UNSUPPORTED;
    const auto &policy = tree_policy();
    const size_t share = integer_limbs(p.base, std::max<uint64_t>(largest_fragments, 1)) /
                         std::max<size_t>(1, policy.frontier_tasks_per_worker * workers);
    p.frontier_limbs = std::min(policy.team_node_limbs, std::max(policy.frontier_floor_limbs, share));
    p.rail.count = 3;
    return p.status = SBN3_SUPPORTED;
}
int ParseTreePlan::classify(uint64_t n, unsigned w) noexcept {
    const auto &policy = tree_policy();
    const size_t limbs = integer_limbs(base, n);
    if (w == 1 || n <= group_fragments || limbs < frontier_limbs)
        w = 1;
    for (unsigned j = 0; j < class_count; ++j)
        if (classes[j].fragments == n && classes[j].workers == w)
            return int(j);
    if (status != SBN3_SUPPORTED)
        return -1;
    ParseClass c{};
    c.fragments = n;
    c.workers = w;
    c.limbs = limbs;
    c.slot_limbs=limbs;
    c.frontier = w == 1;
    if (n <= group_fragments) {
        c.group = true;
    } else {
        c.level = split_level(n);
        const uint64_t low = uint64_t(1) << c.level, high = n - low;
        c.parallel = !c.frontier && low == high &&
                     ((limbs >= policy.fork_node_limbs && w >= 2 * policy.fork_min_workers) ||
                      limbs < policy.team_node_limbs);
        const unsigned child = c.parallel ? w / 2 : w;
        c.low = classify(low, child);
        c.high = classify(high, child);
        if (c.low < 0 || c.high < 0)
            return -1;
        c.low_limbs = classes[c.low].limbs;
        c.high_limbs = classes[c.high].limbs;
        c.shift_limbs = size_t(base.twos) << c.level;
        rail.count = std::max(rail.count, c.level + 1);
        require(c.shift_limbs <= c.low_limbs && c.low_limbs <= c.limbs, SBN3_FATAL_MATH, "radix merge geometry");
        const auto &l = classes[c.low], &h = classes[c.high];
        if(!c.frontier){c.high_offset=align_to(l.slot_limbs,8);
            c.slot_limbs=std::max(c.limbs,c.high_offset+align_to(h.slot_limbs,8));}
        if(!c.frontier)c.tasks=l.tasks+h.tasks;
    }
    if (class_count == max_parse_classes) {
        status = SBN3_QUERY_CAPACITY;
        return -1;
    }
    classes.emplace(class_count,c);
    return int(class_count++);
}
int ParseTreePlan::add_tree(uint64_t fragments) noexcept {
    if (status != SBN3_SUPPORTED)
        return -1;
    if (!fragments || fragments > max_fragments) {
        status = SBN3_QUERY_CAPACITY;
        return -1;
    }
    if(root_count==2){status=SBN3_QUERY_CAPACITY;return -1;}
    const int root=classify(fragments,top_workers);
    if(root>=0)roots[root_count++]=root;
    return root;
}
int ParseTreePlan::add_product(size_t an, size_t bn) noexcept {
    if (status != SBN3_SUPPORTED || extra_count == 2)
        return -1;
    const auto rc = scale_product_plan(an,bn,workers,repeated,extra[extra_count],transcript);
    if (rc != SBN3_SUPPORTED) {
        status = rc;
        return -1;
    }
    return int(extra_count++);
}
sbn3_query_result ParseTreePlan::finish() noexcept {
    if (status != SBN3_SUPPORTED)
        return status;
    uint64_t uses[max_parse_classes]{};
    for(unsigned j=0;j<root_count;++j)++uses[roots[j]];
    for(unsigned j=class_count;j-- >0;)if(!classes[j].group){
        uses[classes[j].low]+=uses[j];uses[classes[j].high]+=uses[j];}
    const auto &policy=tree_policy();frontier_region_bytes=0;
    for(unsigned j=0;j<class_count;++j){auto &c=classes[j];
        if(!c.group){
            const size_t rail_size=rail_limbs(base,c.level);
            // Below the shared FFT power threshold, a program/transform
            // binding adds no arithmetic capability. Use the same exact
            // word/U52 product and work threshold as format-tree leaves.
            c.direct=c.frontier&&rail_size<rail_cached_min_common;
            // A one-use power spectrum has no saved transform unless its
            // class occurs elsewhere in this binding. Reused bindings price
            // apply cost; fresh products price their actual finite use.
            if(c.direct){
                c.cached={};c.product={};c.product.an=c.high_limbs;c.product.bn=rail_size;
                c.product.output_limbs=c.high_limbs+rail_size;
                c.direct_native=c.high_limbs*rail_size>=policy.group_u52_work;
                c.product.work_bytes=c.direct_native?u52::scratch_bytes(c.high_limbs,rail_size):0;
            }else if(!(c.frontier&&policy.cyclic_products&&(repeated||uses[j]>1)&&
                 rail_linear_plan(c.high_limbs,rail_size,c.cached))){
                c.cached={};
                const unsigned w=c.limbs<policy.wide_product_limbs?std::min(c.workers,8u):c.workers;
                status=product_shape(c.high_limbs,rail_size,w,c.product,nullptr,transcript,repeated?0:uses[j]);
                if(status!=SBN3_SUPPORTED)return status;
            }else{c.product={};c.product.an=c.high_limbs;c.product.bn=rail_size;}
            const auto &l=classes[c.low],&h=classes[c.high];
            const size_t episode=c.cached.enabled?c.cached.episode_bytes():c.product.episode_bytes();
            if(c.frontier){c.rest_offset=align_to(c.high_limbs*8,64);
                c.region_bytes=c.rest_offset+align_to(std::max({episode,l.region_bytes,h.region_bytes}),64);}
            else{const size_t le=l.frontier?0:l.episode_bytes,he=h.frontier?0:h.episode_bytes;
                c.high_episode_offset=c.parallel?le:0;
                c.episode_bytes=align_to(std::max(episode,c.parallel?le+he:std::max(le,he)),64);}
        }
        if(c.frontier)frontier_region_bytes=std::max(frontier_region_bytes,c.region_bytes);
    }
    status = rail_finish(base, workers, rail, transcript);
    if (status != SBN3_SUPPORTED)
        return status;
    programs = {};
    for (unsigned j = 0; j < extra_count; ++j)
        if(!extra[j].cached.enabled)programs.account(extra[j]);
    cached_bytes = 0;
    for(unsigned j=0;j<extra_count;++j)if(extra[j].cached.enabled)
        cached_bytes += align_to(extra[j].cached.prepared_bytes(),128);
    for (unsigned j = 0; j < class_count; ++j) {
        if (classes[j].cached.enabled)
            cached_bytes += align_to(classes[j].cached.prepared_bytes(), 128);
        else if (!classes[j].group&&!classes[j].direct)
            programs.account(classes[j].product);
    }
    return status;
}
size_t ParseTreeView::packed_bytes() const noexcept {
    return align_to(sizeof(ParseTreeView),64)+align_to(sizeof(ParseClass)*class_count,64);
}
ParseTreeView *ParseTreeView::pack(void *storage,size_t bytes) const noexcept {
    require(bytes>=packed_bytes()&&!(uintptr_t(storage)&63),SBN3_FATAL_WORKSPACE,"compact parse plan");
    auto *p=::new(storage)ParseTreeView(*this);
    p->classes.reset(static_cast<uint8_t *>(storage)+align_to(sizeof(ParseTreeView),64));
    for(unsigned j=0;j<class_count;++j)p->classes.emplace(j,classes[j]);
    p->transcript=nullptr;return p;
}
size_t ParseTreeView::task_bytes(int root) const noexcept {
    return align_to(size_t(classes[root].tasks) * sizeof(ParseTask), 64);
}
size_t ParseTreeView::work_bytes(int root) const noexcept {
    const auto &c = classes[root];
    return task_bytes(root)+std::max(c.frontier?size_t(0):c.episode_bytes,
           size_t(workers)*align_to(frontier_region_bytes,64));
}
void parse_tree_bind(ParseTree &t, const ParseTreeView &plan, const uint8_t *alphabet, Arena &arena, sbn3_team &team,
                     const sbn3_lease &prepared, sbn3_lease &work, uint64_t *rail_storage,
                     ProductProgram *programs, RailProduct *cached) noexcept {
    t = {};
    t.plan = &plan;
    t.arena = &arena;
    t.team = &team;
    t.work = work;
    t.programs = programs;
    t.cached = cached;
    require(digit_plan_init(t.digits, plan.base.base, alphabet), SBN3_FATAL_ARGUMENT, "radix alphabet");
    t.leaf=parse_leaf_plan(plan.base.base);
    require(work.bytes >= plan.rail.setup_bytes && !(uintptr_t(work.data) & 63) && prepared.bytes >= plan.prepared_bytes(),
            SBN3_FATAL_WORKSPACE, "radix tree storage");
    rail_build(plan.base, plan.rail, plan.workers, arena, team, work, rail_storage, t.rail);
    t.work = work; // (re-acquired by the rail build)
    ProgramSetBuilder builder(plan.programs, arena, prepared);
    const size_t cached_at = align_to(plan.programs.bytes(), 128);
    Frame spectra = Frame::borrow(arena, prepared, static_cast<uint8_t *>(prepared.data) + cached_at, prepared.bytes - cached_at);
    for(unsigned j=0;j<plan.extra_count;++j)scale_product_prepare(t.extra[j],plan.extra[j],builder,spectra);
    for (unsigned j = 0; j < plan.class_count; ++j) {
        const auto &c = plan.classes[j];
        ::new (&programs[j]) ProductProgram{};
        ::new (&cached[j]) RailProduct{};
        if (c.cached.enabled)
            rail_product_prepare(cached[j], c.cached, spectra, t.rail[c.level]);
        else if (!c.group&&!c.direct)
            programs[j] = builder.prepare(c.product);
    }
}
namespace {
struct Run {
    ParseTree *tree;
    const uint8_t *digits;
    uint64_t count, fragments; // digits present; fragments of the root (the first may be partial)
    bool valid;
};
void combine(const ParseLeafPlan &p,const uint64_t *values,size_t l1,unsigned blocks,uint64_t *out) noexcept {
    // At most four 128-digit blocks. Horner uses the same one-block power
    // at every merge, including a partial leading block. For decimal the
    // leading scalar-product count equals the balanced tree's count, while
    // avoiding its recursive frames and alternating multiplier widths.
    memcpy(out,values,l1*8);size_t used=l1;
    uint64_t z[2*parse_leaf_limbs];
    const size_t pn=p.power_limbs[1],shift=size_t(p.twos)<<1;
    for(unsigned b=1;b<blocks;++b){
        while(used&&!out[used-1])--used;
        const size_t product=used?used+pn:0,words=p.limbs[(b+1)*2];
        if(used)sbn3_mul_basecase(z,product,out,used,p.powers[1],pn);
        memcpy(out,values+size_t(b)*l1,l1*8);std::fill(out+l1,out+words,uint64_t(0));
        const size_t count=std::min(product,words-shift);uint64_t carry=0;
        for(size_t j=0;j<count;++j){const __uint128_t sum=__uint128_t(out[shift+j])+z[j]+carry;
            out[shift+j]=uint64_t(sum);carry=uint64_t(sum>>64);}
        for(size_t j=shift+count;carry&&j<words;++j)carry=++out[j]==0;
        used=words;
    }
}
} // namespace
bool parse_leaf(const ParseLeafPlan &p,const DigitPlan &d,const uint8_t *digits,size_t count,uint64_t *out) noexcept {
    require(count&&count<=group_fragments*fragment_digits,SBN3_FATAL_ARGUMENT,"parse leaf count");
    if(count<=64)return parse_integer_fragments(out,p.limbs[1],digits,count,d);
    const unsigned n=unsigned((count+63)/64),blocks=unsigned((count+127)/128);const size_t l1=p.limbs[2];
    uint64_t values[group_fragments*8];
    const bool valid=parse_integer_blocks(values,l1,digits,count,128,d);
    uint64_t result[parse_leaf_limbs];
    combine(p,values,l1,blocks,result);memcpy(out,result,p.limbs[n]*8);return valid;
}
[[gnu::noinline]] bool parse_leaf_exact(const ParseLeafPlan &p,const DigitPlan &d,const uint8_t *digits,size_t count,
                                      uint64_t *out,size_t words) noexcept {
    alignas(64) uint64_t value[parse_leaf_limbs];
    const bool valid=parse_leaf(p,d,digits,count,value);memcpy(out,value,words*8);return valid;
}
size_t parse_local_bytes(size_t words) noexcept {
    // If a child has C words, its reserved high sibling has <=N-C+1.
    // Telescoping down a path leaves <=N-C+depth live ancestor words.
    // The current merge adds <=ceil(C/2) high words and <=C+2 product
    // words. Thus 1.5N+128 words cover values and alignment at all eight
    // possible levels in the 16384-digit domain. Balanced
    // operands maximize the local product's recursive scratch envelope;
    // eight extra limbs per side cover conversion/vector rounding.
    const size_t half=(words+1)/2+8;
    return align_to(((3*words+1)/2+128)*8,64)+u52::scratch_bytes(half,half);
}
namespace {
size_t local_words(unsigned base,size_t digits) noexcept {
    return size_t((__uint128_t(digits)*(log2_floor_q56[base]+1)+((uint64_t(1)<<62)-1))>>62);
}
bool local_node(const ParseLeafPlan &leaf,const DigitPlan &d,const uint8_t *digits,size_t count,
                uint64_t *out,Frame &scratch) noexcept {
    const size_t n=local_words(d.base,count);
    if(count<=128)return parse_integer_blocks(out,n,digits,count,128,d);
    if(count<=group_fragments*fragment_digits)return parse_leaf_exact(leaf,d,digits,count,out,n);
    FrameMark mark(scratch);
    const unsigned level=split_level((count+63)/64);
    const size_t low_digits=size_t(64)<<level,high_digits=count-low_digits;
    size_t hn=local_words(d.base,high_digits);
    auto *high=scratch.alloc<uint64_t>(hn);
    if(!local_node(leaf,d,digits+high_digits,low_digits,out,scratch)||
       !local_node(leaf,d,digits,high_digits,high,scratch))return false;
    const size_t ln=local_words(d.base,low_digits);
    std::fill(out+ln,out+n,uint64_t(0));
    while(hn&&!high[hn-1])--hn;
    if(!hn)return true;
    const unsigned odd=d.base>>leaf.twos;
    const auto power=fixed_power(odd,level);
    size_t pn=size_t((__uint128_t(low_digits)*(log2_floor_q56[odd]+1)+((uint64_t(1)<<62)-1))>>62);
    while(pn&&!power.data[pn-1])--pn;
    auto *z=scratch.alloc<uint64_t>(hn+pn);
    if(std::min(hn,pn)<=2||hn*pn<tree_policy().group_u52_work)sbn3_mul_basecase(z,hn+pn,high,hn,power.data,pn);
    else u52::multiply(z,high,hn,power.data,pn,scratch);
    const size_t shift=size_t(leaf.twos)<<level,used=std::min(hn+pn,n-shift);
    uint64_t carry=0;
    for(size_t j=0;j<used;++j){const __uint128_t v=__uint128_t(out[shift+j])+z[j]+carry;
        out[shift+j]=uint64_t(v);carry=uint64_t(v>>64);}
    for(size_t j=shift+used;carry&&j<n;++j)carry=++out[j]==0;
    require(!carry,SBN3_FATAL_MATH,"local parse capacity");return true;
}
}
bool parse_local(const ParseLeafPlan &leaf,const DigitPlan &d,const uint8_t *digits,size_t count,
                 uint64_t *out,Frame &scratch) noexcept {
    require(count&&count<=16384,SBN3_FATAL_ARGUMENT,"local parse count");
    return local_node(leaf,d,digits,count,out,scratch);
}
namespace {
void leaf_group(Run &run,const ParseClass &c,uint64_t fragment,uint64_t *out) noexcept {
    auto &t=*run.tree;const size_t missing=size_t(run.fragments*fragment_digits-run.count);
    const size_t skip=fragment?0:missing,at=fragment?size_t(fragment*fragment_digits)-missing:0;
    if(!parse_leaf(t.leaf,t.digits,run.digits+at,size_t(c.fragments*fragment_digits)-skip,out))
        __atomic_store_n(&run.valid,false,__ATOMIC_RELAXED);
}
// out = low value (already in out[0..low_limbs)) + (high * rail[level]) << (64 * shift_limbs)
void merge(Run &run, int index, const uint64_t *high, uint64_t *out, uint8_t *episode, sbn3_team_scope *scope) noexcept {
    auto &t = *run.tree;
    const auto &c = t.plan->classes[index];
    auto *z = reinterpret_cast<uint64_t *>(episode);
    const size_t product_limbs = c.high_limbs + c.product.bn;
    if(c.direct){
        if(c.direct_native){
            auto *work=episode+align_to(product_limbs*8,64);
            auto frame=t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work,c.product.work_bytes);
            u52::multiply(z,high,c.high_limbs,t.rail[c.level],c.product.bn,frame);
        }else sbn3_mul_basecase(z,product_limbs,high,c.high_limbs,t.rail[c.level],c.product.bn);
    }else if (c.cached.enabled) {
        const auto &rp = t.cached[index];
        auto *work = episode + align_to(rp.plan.output_limbs() * 8, 64) + 64;
        auto frame = t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work, rp.plan.scratch_bytes);
        rail_product_execute(rp, frame, scope, high, z);
    } else {
        const auto &s = c.product;
        const uintptr_t after = reinterpret_cast<uintptr_t>(episode) + align_to(s.output_limbs * 8, 64);
        auto *work = reinterpret_cast<uint8_t *>((after + s.work_alignment - 1) & ~uintptr_t(s.work_alignment - 1));
        auto frame = t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work, s.work_bytes);
        product_program_execute(t.programs[index], frame, scope, {high, s.an}, {t.rail[c.level], s.bn}, {z, s.output_limbs});
    }
    const size_t count = std::min(product_limbs, c.limbs - c.shift_limbs);
    const size_t overlap = c.low_limbs - c.shift_limbs; // limbs of z that meet the low value
    const uint64_t carry = parallel_limbs::add_to_scope(scope, out + c.shift_limbs, overlap, z, std::min(overlap, count));
    if (count > overlap)
        parallel_limbs::copy_scope(scope, out + c.low_limbs, z + overlap, count - overlap);
    for (size_t j = std::max(c.shift_limbs + count, c.low_limbs); j < c.limbs; ++j)
        out[j] = 0;
    if (carry)
        parallel_limbs::add_word(out + c.low_limbs, c.limbs - c.low_limbs, carry);
}
// A frontier subtree, serially inside `region`.
void serial_node(Run &run, int index, uint64_t fragment, uint64_t *out, uint8_t *region, sbn3_team_scope *scope) noexcept {
    const auto &c = run.tree->plan->classes[index];
    if (c.group) {
        leaf_group(run, c, fragment, out);
        return;
    }
    auto *high = reinterpret_cast<uint64_t *>(region);
    auto *rest=region+c.rest_offset;
    const uint64_t high_fragments = c.fragments - (uint64_t(1) << c.level);
    serial_node(run, c.low, fragment + high_fragments, out, rest, scope);
    serial_node(run, c.high, fragment, high, rest, scope);
    merge(run, index, high, out, rest, scope);
}
// The frontier tasks below a top node, in the order of the top recursion.
void collect(const ParseTreeView &p, int index, uint64_t fragment, uint64_t *out, ParseTask *tasks) noexcept {
    const auto &c = p.classes[index];
    if (c.frontier) {
        *tasks = {index, fragment, out};
        return;
    }
    const auto &l = p.classes[c.low];
    const uint64_t high_fragments = c.fragments - (uint64_t(1) << c.level);
    collect(p, c.low, fragment + high_fragments, out, tasks);
    collect(p, c.high, fragment, out+c.high_offset, tasks + l.tasks);
}
struct TopTask {
    Run *run;
    int node;
    uint64_t *out;
    uint8_t *episode;
};
void top_node(Run &run, int index, uint64_t *out, uint8_t *episode, sbn3_team_scope *scope) noexcept;
void top_action(void *v, sbn3_team_scope *scope) {
    auto &k = *static_cast<TopTask *>(v);
    top_node(*k.run, k.node, k.out, k.episode, scope);
}
void top_node(Run &run, int index, uint64_t *out, uint8_t *episode, sbn3_team_scope *scope) noexcept {
    const auto &p = *run.tree->plan;
    const auto &c = p.classes[index];
    if (c.frontier)
        return; // evaluated by its frontier task
    auto *high = out+c.high_offset;
    TopTask lt{&run, c.low, out, episode};
    TopTask ht{&run, c.high, high, episode + c.high_episode_offset};
    if (c.parallel) {
        sbn3_team_invoke2(scope, SBN3_PARALLEL_CHILDREN, sbn3_team_width(scope) / 2, top_action, &lt, top_action, &ht);
    } else {
        top_action(&lt, scope);
        top_action(&ht, scope);
    }
    merge(run, index, high, out, episode, scope);
}
struct RootTask {
    Run *run;
    int root;
    uint64_t *out;
    ParseTask *tasks;
    uint8_t *episode, *regions;
    size_t region_bytes;
    sbn3_team_scope *scope;
};
void frontier_tasks(void *v, uint64_t begin, uint64_t end, unsigned rank) {
    auto &k = *static_cast<RootTask *>(v);
    sbn3_team_scope self{k.scope->team, k.scope->first + rank, 1, false, k.scope->epoch};
    uint8_t *region = k.regions + size_t(rank) * k.region_bytes;
    for (uint64_t j = begin; j < end; ++j)
        serial_node(*k.run, k.tasks[j].node, k.tasks[j].fragment, k.tasks[j].out, region, &self);
}
} // namespace
bool parse_tree_run(ParseTree &t, int root, const uint8_t *digits, uint64_t count, uint64_t *out,size_t capacity) noexcept {
    const auto &p = *t.plan;
    const auto &c = p.classes[root];
    require(capacity>=c.slot_limbs,SBN3_FATAL_WORKSPACE,"parse value slab",c.slot_limbs,capacity);
    require(count && count <= c.fragments * fragment_digits && count > (c.fragments - 1) * fragment_digits,
            SBN3_FATAL_ARGUMENT, "radix parse digit count");
    require(t.work.bytes >= p.work_bytes(root), SBN3_FATAL_WORKSPACE, "radix tree storage", p.work_bytes(root), t.work.bytes);
    Run run{&t, digits, count, c.fragments, true};
    if (c.group) {
        leaf_group(run, c, 0, out);
        return run.valid;
    }
    struct alignas(64) RootStorage {
        unsigned char bytes[(sizeof(Frame) + 63) & ~size_t(63)];
    };
    RootStorage storage[32];
    const unsigned workers = sbn3_team_workers(t.team);
    for (unsigned j = 0; j < workers; ++j) {
        auto *frame = ::new (storage[j].bytes) Frame(*t.arena, t.work);
        frame->allocate(t.work.bytes);
        t.roots[j] = frame;
    }
    auto *base = static_cast<uint8_t *>(t.work.data);
    RootTask task{&run, root, out, reinterpret_cast<ParseTask *>(base), base + p.task_bytes(root), base+p.task_bytes(root),
                  align_to(p.frontier_region_bytes, 64), nullptr};
    collect(p, root, 0, out, task.tasks);
    sbn3_team_run(t.team, [](void *v, sbn3_team_scope *scope) {
        auto &k = *static_cast<RootTask *>(v);
        const auto &plan = *k.run->tree->plan;
        sbn3_team_scope active{scope->team,scope->first,std::min(plan.workers,sbn3_team_width(scope)),false,scope->epoch};
        k.scope = &active;
        sbn3_team_for(&active, 0, plan.classes[k.root].tasks, 1, SBN3_DYNAMIC, frontier_tasks, &k);
        sbn3_team_scope top{scope->team, scope->first, std::min(plan.top_workers, active.width), false, scope->epoch};
        top_node(*k.run, k.root, k.out, k.episode, &top);
    }, &task);
    for (unsigned j = 0; j < workers; ++j) {
        t.roots[j]->~Frame();
        t.roots[j] = nullptr;
    }
    return run.valid;
}
} // namespace sbn::v3::radix
