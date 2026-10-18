#include "radix/format_tree.hpp"
#include "radix/power_bank.hpp"
#include "backend/u52/kernels.hpp"
#include "value/parallel_limbs.hpp"
#include "product/cost_model.hpp"
#include "product/word_window.hpp"
#include <algorithm>
#include <new>
#include <string.h>
// -DSBN3_RADIX_TRACE: per class wall time of products, windows and leaf groups on stderr (probe builds only).
#ifdef SBN3_RADIX_TRACE
#include <stdio.h>
#include <time.h>
namespace {
struct TraceRow { uint64_t product_ns, window_ns, leaf_ns, count; };
TraceRow trace_rows[512];
uint64_t trace_now() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec; }
void trace_add(uint64_t &slot, uint64_t v) { __atomic_add_fetch(&slot, v, __ATOMIC_RELAXED); }
struct TraceSpan {
    uint64_t *slot, start;
    explicit TraceSpan(uint64_t &s) : slot(&s), start(trace_now()) {}
    ~TraceSpan() { trace_add(*slot, trace_now() - start); }
};
}
#define SBN3_RADIX_SPAN(name, slot) TraceSpan name(slot)
#else
#define SBN3_RADIX_SPAN(name, slot) ((void)0)
#endif
namespace sbn::v3::radix {
namespace {
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
unsigned pow2_floor(unsigned v) noexcept { return 1u << (31 - unsigned(__builtin_clz(v))); }
uint64_t request_seed(uint64_t fragments,unsigned workers) noexcept {
    const auto configured=tree_policy().format_seed_fragments;
    const unsigned limit=configured?configured:workers>1?31:16;
    while(fragments>limit)fragments=(fragments+1)/2;
    return fragments;
}
bool coupled_ladder(const BaseInfo &base,uint64_t fragments) noexcept {
    if(fragments<=2*format_group_fragments)return false;
    if(tree_policy().short_tail_prefix){
        const uint64_t prefix=uint64_t(1)<<split_level(fragments),tail=fragments-prefix;
        const size_t common=rail_limbs(base,split_level(fragments));
        const size_t fresh=node_limbs(base,fragments)-base.twos*prefix;
        if(split_level(fragments)<fixed_power_levels&&node_limbs(base,tail)<=fraction_block_limbs&&
           u52::middle_guard_supported(common,fresh))return false;
    }
    uint64_t left=request_seed(fragments,1);while(2*left<fragments)left*=2;
    const uint64_t parent=left/2;
    return limbs_for_bits(power_bits(base.odd_bound,fragment_digits*parent))>=rail_cached_min_common&&
        node_limbs(base,left)-size_t(base.twos)*parent>=tree_policy().cyclic_min_fresh;
}
sbn3_query_result witness_query(size_t a,size_t b,uint64_t bit,unsigned workers,
                                product::WordWindowPlan &p,PlanTranscript *transcript) noexcept {
    const size_t block=std::min({(size_t(bit/64)+1+unsigned(bit%64!=0)+3)/4,a,b});
    if(!a||!b||bit/64>=a+b||block<=8||2*block<=native_policy::window_inline_words)
        return product::word_window_query(a,b,bit,workers,p);
    const uint64_t saved=transcript&&transcript->replay?transcript->next():0;
    const ProductChoice choice=saved&((uint64_t(1)<<product_choice_bits)-1);
    auto rc=SBN3_UNSUPPORTED;
    if(!transcript||!transcript->replay||choice)rc=product::word_window_query(a,b,bit,workers,p,choice);
    auto check=[&]{return PlanTranscript::check(a,b,bit^workers,p.bytes);};
    if(transcript&&transcript->replay&&(!choice||rc!=SBN3_SUPPORTED||check()!=saved>>product_choice_bits)){
        ++transcript->searched;rc=product::word_window_query(a,b,bit,workers,p);
    }
    if(transcript&&!transcript->replay)transcript->record(rc==SBN3_SUPPORTED?p.choice:0,check());
    return rc;
}
} // namespace
sbn3_query_result format_tree_begin(unsigned base, unsigned workers, uint64_t largest_fragments, FormatTreePlan &p) noexcept {
    // The caller may supply raw storage. The plan owns no resources; placement
    // construction also safely replaces an existing trivially destructible plan.
    static_assert(__is_trivially_destructible(FormatTreePlan));
    ::new (&p) FormatTreePlan{};
    if (!base_info(base, p.base) || p.base.odd == 1 || !workers || workers > 32)
        return p.status = SBN3_UNSUPPORTED;
    p.workers = workers;
    p.top_workers = pow2_floor(workers);
    const auto &policy = tree_policy();
    p.block_leaves=policy.block_leaves;
    p.block_max_limbs=policy.block_tree_limbs?policy.block_tree_limbs:
        largest_fragments>=2*8*format_group_fragments*workers?fraction_block_tree_limbs:fraction_block_limbs;
    if(policy.format_seed_fragments&&(policy.format_seed_fragments<format_group_fragments||policy.format_seed_fragments>2*format_group_fragments))
        return p.status=SBN3_UNSUPPORTED;
    p.balanced=policy.format_split_policy==2 ||
        (!policy.format_split_policy&&((workers==1&&coupled_ladder(p.base,largest_fragments))||
         (workers>1&&node_limbs(p.base,largest_fragments)>=policy.team_node_limbs)));
    const size_t share = node_limbs(p.base, std::max<uint64_t>(largest_fragments, 1)) /
                         std::max<size_t>(1, policy.frontier_tasks_per_worker * workers);
    p.frontier_limbs = std::min(policy.team_node_limbs, std::max(policy.frontier_floor_limbs, share));
    // A single worker still needs owner-side stages for product-service
    // bindings. Keep only in-cache subtrees on its recursive frontier.
    if(workers==1)p.frontier_limbs=std::max(p.frontier_limbs,policy.ring_node_limbs);
    for (uint64_t n = 1; n <= format_group_fragments; ++n)
        p.group_limbs[n] = node_limbs(p.base, n);
    if(p.block_leaves&&(p.block_max_limbs<p.group_limbs[1]||p.block_max_limbs>fraction_block_tree_limbs))
        return p.status=SBN3_UNSUPPORTED;
    if(!p.block_leaves)for(unsigned n=2;n<=format_group_fragments;++n){
        const unsigned k=split_level(n),left=1u<<k,right=n-left;
        const size_t a=p.group_limbs[n]-(size_t(p.base.twos)<<k),b=rail_limbs(p.base,k);
        if(a*b>=policy.group_u52_work)p.group_u52|=1u<<n;
        p.group_work[n]=std::max({p.group_work[left],p.group_work[right],
                                  p.group_u52>>n&1?u52::scratch_bytes(a,b):size_t(0)});
    }
    const uint64_t leaf_bits = power_bits(p.base.base_bound, fragment_digits + word_digits) + guard_bits;
    p.fragment_u52 = unsigned((leaf_bits + 51) / 52);
    // The leaf kernel reads whole u52 digits from the top of the fragment fraction.
    if (p.group_limbs[1] > max_fragment_limbs || p.fragment_u52 > max_fragment_u52 ||
        uint64_t(52) * p.fragment_u52 >= uint64_t(64) * p.group_limbs[1] + 52 ||
        p.group_limbs[format_group_fragments] + rail_limbs(p.base, 3) > group_product_limbs ||
        p.group_limbs[format_group_fragments / 2] > group_right_limbs)
        return p.status = SBN3_UNSUPPORTED;
    p.rail.count = 4; // local groups use rail[0..3]
    return p.status = SBN3_SUPPORTED;
}
// The split of one node size on `workers` workers: the wrap-around product with the cached rail spectrum
// where the FFT kernels cover it (one worker), the exact product program otherwise.
sbn3_query_result FormatTreePlan::split_plan(const NodeClass &c, unsigned w, uint64_t count, SplitPlan &out) const noexcept {
    const auto &policy = tree_policy();
    out = {};
    const size_t rail_size = limbs_for_bits(power_bits(base.odd_bound,fragment_digits*c.left_fragments));
    // Everything above the window wraps at least one limb below it when the ring holds the rest of the product.
    const size_t wrapped_ring = c.split_limbs + rail_size - c.window_limbs + 1;
    const unsigned product_workers = c.limbs < policy.wide_product_limbs ? std::min(w, 8u) : w;
    if(rail_size<32){
        out.direct=true;out.direct_native=c.split_limbs*rail_size>=policy.group_u52_work;
        out.product.an=c.split_limbs;out.product.bn=rail_size;out.product.output_limbs=c.split_limbs+rail_size;
        out.product.work_bytes=out.direct_native?u52::scratch_bytes(c.split_limbs,rail_size):0;
        return SBN3_SUPPORTED;
    }
    auto middle = [&]() {
        // A serial frontier has exact per-class dimensions and exclusive
        // scratch, including when it runs inside a wider outer team.
        // Staged envelopes may contain shorter live inputs and use their
        // bounded product provider instead.
        if(!c.frontier || w!=1 || rail_size<32 ||
           !u52::middle_guard_supported(rail_size,c.split_limbs))return false;
        const size_t a=(64*rail_size+51)/52,b=(64*c.split_limbs+51)/52;
        // Keep the four-diagonal kernel where its guard reaches the window;
        // adjacent balanced shapes occasionally need two more diagonals.
        const unsigned guards=64*c.window_limbs>=52*(a-5)+128?4:6;
        const uint64_t start=52*(a-1-guards),window=64*c.window_limbs;
        if(window<start+128 || 64*(c.window_limbs+c.right_limbs)>52*b)return false;
        // The omitted low contribution can change this window by at most
        // one. Only its first exact word is needed to resolve a guard tie.
        out.product.an=c.split_limbs;out.product.bn=rail_size;
        out.product.output_limbs=c.split_limbs+rail_size;
        out.middle_guards=guards;
        out.middle_words=(52*(b-a+guards+3)+63)/64;
        out.middle_shift=window-start;
        out.middle_bytes=u52::middle_guard_scratch_bytes(rail_size,c.split_limbs);
        product::WordWindowPlan witness;
        if(witness_query(c.split_limbs,rail_size,window,1,witness,transcript)!=SBN3_SUPPORTED)return false;
        out.low_bytes=witness.bytes;out.low_choice=witness.choice;
        return true;
    };
    // Below the cached FFT's fresh-input domain, a supported native middle
    // band needs no full-product plan merely to rediscover that same kernel.
    if(c.split_limbs<policy.cyclic_min_fresh&&middle())return SBN3_SUPPORTED;
    const bool level_fft=balanced&&!repeated&&w==1&&
        (rail_size>=native_policy::window_fft_min_words||(count>1&&rail_size>=rail_cached_min_common));
    const bool one_use_window=!level_fft&&!repeated && count<=2 &&
        (c.right_limbs<=rail_size || rail_size<native_policy::window_fft_min_words);
    if(one_use_window && middle())return SBN3_SUPPORTED;
    ProductShape linear{};
    ProductProgramPlan linear_plan{};
    bool have_linear = false;
    if (!level_fft&&(w == 1 || (!c.frontier && count==1))) {
        const auto rc = product_shape(c.split_limbs, rail_size, product_workers, linear, &linear_plan, transcript, repeated?0:count);
        if (rc != SBN3_SUPPORTED)
            return rc;
        have_linear = true;
    }
    const bool short_middle=linear_plan.info.algorithm==SBN3_MUL_U52;
    if(have_linear && short_middle && middle())return SBN3_SUPPORTED;
    bool cyclic = w == 1 && policy.cyclic_products && (c.frontier||c.limbs<policy.ring_node_limbs) &&
                  rail_product_plan(c.split_limbs, rail_size, std::max(c.split_limbs, wrapped_ring), out.cyclic,
                                    repeated?0:count,policy.cyclic_min_fresh);
    if (cyclic && have_linear) {
        const double ordinary = pq16::native_cost(out.cyclic.shape, rail_size, c.split_limbs);
        const double cached = (repeated?0:.27 * double(out.cyclic.table_bytes) + cost_model::prepare_share(ordinary)) +
                              cost_model::cached_share(ordinary) * double(count);
        cyclic = cached < (repeated?0:linear.prepare_ns) + linear.apply_ns * double(count);
    }
    if (cyclic) {
        if(out.cyclic.ring>=c.split_limbs+rail_size){
            out.product.an=c.split_limbs;out.product.bn=rail_size;
            return SBN3_SUPPORTED;
        }
        // The exact tie-break: limb h of the full product from the low h + 1 limbs of the operands.
        out.gap_limbs = out.cyclic.ring - wrapped_ring + 1;
        out.low_limbs = c.split_limbs + rail_size - out.cyclic.ring + 1;
        out.low_rail_limbs = std::min(out.low_limbs, rail_size);
        product::WordWindowPlan witness;
        cyclic=witness_query(out.low_limbs,out.low_rail_limbs,64*(out.low_limbs-1),1,witness,transcript)==SBN3_SUPPORTED;
        if(cyclic){out.low_bytes=witness.bytes;out.low_choice=witness.choice;}
    }
    if (cyclic) {
        out.product.an = c.split_limbs;
        out.product.bn = rail_size;
        return SBN3_SUPPORTED;
    }
    out = {};
    if (!c.frontier && policy.ring_products && c.limbs >= policy.ring_node_limbs &&
        (!repeated || w==1 || count==1 || count>=policy.ring_min_count) &&
        ring_plan(c.split_limbs, rail_size, std::max(c.split_limbs, wrapped_ring), product_workers, out.ring, transcript,
                  count) &&
        (!have_linear || (policy.prefer_short_ring&&out.ring.ring<c.split_limbs+rail_size) || out.ring.prepare_ns + count*out.ring.predicted_ns <
                            (repeated?0:linear.prepare_ns) + count*linear.apply_ns)) {
        if(out.ring.ring>=c.split_limbs+rail_size){
            out.product.an=c.split_limbs;out.product.bn=rail_size;out.product.workers=product_workers;
            return SBN3_SUPPORTED;
        }
        out.gap_limbs = out.ring.ring - wrapped_ring + 1;
        out.low_limbs = c.split_limbs + rail_size - out.ring.ring + 1;
        out.low_rail_limbs = std::min(out.low_limbs, rail_size);
        out.low_workers = product_workers;
        product::WordWindowPlan witness;
        const bool ok=witness_query(out.low_limbs,out.low_rail_limbs,64*(out.low_limbs-1),product_workers,witness,transcript)==SBN3_SUPPORTED;
        if(ok){out.low_bytes=witness.bytes;out.low_choice=witness.choice;}
        if (ok) {
            out.product.an = c.split_limbs;
            out.product.bn = rail_size;
            out.product.workers = product_workers;
            return SBN3_SUPPORTED;
        }
        out = {};
    }
    if(have_linear){out={};out.product=linear;return SBN3_SUPPORTED;}
    if(middle())return SBN3_SUPPORTED;
    return product_shape(c.split_limbs, rail_size, product_workers, out.product, nullptr, transcript);
}
int FormatTreePlan::classify(uint64_t n,uint64_t seed) noexcept {
    // One request-scaled dyadic ladder. All full subtrees of a level share
    // the same power; the ragged right spine only skips unused upper levels.
    // The immutable power bank is a source of constants, not a split grid.
    uint64_t left=0;
    const size_t words=node_limbs(base,n);
    const bool group=n<=format_group_fragments&&(!block_leaves||words<=
        block_max_limbs);
    if(!group){
        if(n<=seed)left=uint64_t(1)<<split_level(n);
        else{left=seed;while(2*left<n)left*=2;}
    }
    for (unsigned j = 0; j < class_count; ++j)
        if (classes[j].fragments == n && classes[j].left_fragments==left)
            return int(j);
    if (status != SBN3_SUPPORTED)
        return -1;
    NodeClass c{};
    c.fragments = n;
    c.left_fragments=left;
    c.limbs = words;
    c.frontier = group || c.limbs < frontier_limbs;
    if (group) {
        c.group = true;
        c.seams=block_leaves?1:n;
        c.region_bytes=group_work[n];
        c.slot_limbs=c.limbs;
    } else {
        require(left&&left<n,SBN3_FATAL_MATH,"balanced radix split");
        const int power=repeated?int(split_level(n)):powers.request(base,left);
        if(power<0){status=SBN3_QUERY_CAPACITY;return -1;}
        c.level=unsigned(power);const uint64_t right=n-left;
        c.left = classify(left,seed);
        c.right = classify(right,seed);
        if (c.left < 0 || c.right < 0)
            return -1;
        c.right_limbs = classes[c.right].limbs;
        const size_t shift = size_t(base.twos)*left;
        require(c.right_limbs+shift<=c.limbs && shift<c.limbs,SBN3_FATAL_MATH,
                "radix window geometry");
        c.window_limbs = c.limbs-c.right_limbs-shift;
        c.split_limbs = c.limbs - shift;
        if(repeated)rail.count = std::max(rail.count, c.level + 1);
        const auto &l = classes[c.left], &r = classes[c.right];
        c.seams=l.seams+r.seams;
        c.slot_limbs=l.slot_limbs+r.slot_limbs;
        require(c.slot_limbs>=c.limbs,SBN3_FATAL_MATH,"radix child capacities");
        if (!c.frontier) {
            // Descendants reuse this one slot-partitioned slab. Each parent
            // product finishes reading before its right child overwrites the
            // low slot; its left child's high view is disjoint and survives.
            c.persist_bytes = align_to(c.slot_limbs*8,64);
            c.tasks = l.tasks + r.tasks;
            c.top_nodes = 1 + l.top_nodes + r.top_nodes;
        }
    }
    if (class_count == max_classes) {
        status = SBN3_QUERY_CAPACITY;
        return -1;
    }
    classes.emplace(class_count,c);
    return int(class_count++);
}
int FormatTreePlan::add_tree(uint64_t fragments) noexcept {
    if (status != SBN3_SUPPORTED)
        return -1;
    if (!fragments || fragments > max_fragments || tree_count == max_trees) {
        status = SBN3_QUERY_CAPACITY;
        return -1;
    }
    if(!repeated&&!powers.count){for(uint64_t e=1;e<=format_group_fragments;e<<=1)if(powers.request(base,e)<0){status=SBN3_QUERY_CAPACITY;return -1;}}
    uint64_t seed=1;
    if(!repeated&&balanced)seed=request_seed(fragments,workers);
    const int root = classify(fragments,seed);
    if (root < 0)
        return -1;
    const auto stages=trees[tree_count].stages;
    trees[tree_count]=TreeShape{};trees[tree_count].stages=stages;
    trees[tree_count].root = root;
    return int(tree_count++);
}
Chain FormatTreeView::chain(int tree)const noexcept {
    const int root=trees[tree].root;
    auto attach=[&](Chain c){
        for(unsigned k=0;k<c.steps;++k){auto &step=c.step[k];
            auto use=[&](unsigned index,const SplitPlan &s){
                if(step.cache_slot<0&&s.cyclic.common_limbs==step.power_limbs&&
                   rail_product_prefix_supported(s.cyclic,step.acc_limbs)){
                    step.cache_slot=s.cache_slot<0?int(index):s.cache_slot;step.cache_bytes=s.cyclic.scratch_bytes;}
            };
            for(unsigned j=0;j<class_count;++j)if(!classes[j].group&&classes[j].frontier&&classes[j].level==step.level)use(j,classes[j].split);
            for(unsigned t=0;t<tree_count;++t)for(unsigned j=0;j<trees[t].stage_count;++j){const auto &s=trees[t].stages[j];
                if(classes[s.node_class].level==step.level)use(unsigned(s.node_class),s.split);}
        }
        return c;
    };
    if(repeated)return attach(chain_of(base,classes[root].fragments));
    int path[max_rail];unsigned depth=0;int tail=root;
    while(!classes[tail].group){require(depth<max_rail,SBN3_FATAL_SIZE,"balanced chain depth");path[depth++]=tail;tail=classes[tail].right;}
    const uint64_t tail_fragments=classes[tail].fragments;
    uint64_t exponent=uint64_t(1)<<unsigned(__builtin_ctzll(tail_fragments));
    const int first=powers.find(exponent);require(first>=0,SBN3_FATAL_MATH,"balanced chain seed");
    Chain c{};c.first=unsigned(first);c.first_limbs=c.limbs=c.widest=powers.entries[first].limbs;
    auto append=[&](unsigned level){const auto &power=powers.entries[level];
        exponent+=power.exponent;const size_t next=limbs_for_bits(power_bits(base.odd_bound,fragment_digits*exponent));
        c.step[c.steps++]={level,c.limbs,next,power.limbs,0};c.widest=std::max(c.widest,c.limbs+power.limbs);c.limbs=next;};
    for(uint64_t bit=exponent<<1;bit<=tail_fragments;bit<<=1)if(tail_fragments&bit){
        const int i=powers.find(bit);require(i>=0,SBN3_FATAL_MATH,"balanced chain leaf power");append(unsigned(i));}
    while(depth)append(classes[path[--depth]].level);
    require(exponent==classes[root].fragments,SBN3_FATAL_MATH,"balanced chain exponent");return attach(c);
}
int FormatTreePlan::add_product(size_t an, size_t bn,bool reuse) noexcept {
    if (status != SBN3_SUPPORTED || extra_count == 2)
        return -1;
    const auto rc = scale_product_plan(an,bn,workers,reuse,extra[extra_count],transcript);
    if (rc != SBN3_SUPPORTED) {
        status = rc;
        return -1;
    }
    return int(extra_count++);
}
sbn3_query_result FormatTreePlan::finish() noexcept {
    if (status != SBN3_SUPPORTED)
        return status;
    status = rail_finish(base, workers, rail, transcript);
    if (status != SBN3_SUPPORTED)
        return status;
    const auto &policy = tree_policy();
    uint64_t uses[max_classes]{};
    for (unsigned t = 0; t < tree_count; ++t)
        ++uses[trees[t].root];
    for (unsigned j = class_count; j-- > 0;)
        if (!classes[j].group) {
            uses[classes[j].left] += uses[j];
            uses[classes[j].right] += uses[j];
        }
    if(block_leaves&&class_count){unsigned largest=0,words=0;
        for(unsigned j=0;j<class_count;++j)if(classes[j].group){largest=std::max(largest,unsigned(classes[j].fragments));words=std::max(words,unsigned(classes[j].limbs));}
        block_work_bytes=fraction_block_tree_scratch_bytes(words,largest);
        for(unsigned j=0;j<class_count;++j)if(classes[j].group){auto &c=classes[j];c.region_bytes=block_work_bytes;
            c.block_batch_offset=block_batch_bytes;
            block_batch_bytes+=align_to(sizeof(BlockBatch),64)+8*align_to(c.limbs*8,64);}
    }
    // A full node and the ragged tail at one level consume the same power.
    // Compile one cached envelope for their actual live supports and share
    // it. Small direct/middle kernels retain their exact local dimensions.
    if(balanced&&!repeated)for(unsigned j=0;j<class_count;++j){
        auto &c=classes[j];
        if(c.group||!c.frontier||c.split.cache_slot>=0||
           powers.entries[c.level].limbs<rail_cached_min_common)continue;
        NodeClass envelope=c;uint64_t applications=0;
        for(unsigned k=j;k<class_count;++k){const auto &q=classes[k];
            if(q.group||!q.frontier||q.level!=c.level)continue;
            applications+=uses[k];envelope.limbs=std::max(envelope.limbs,q.limbs);
            envelope.split_limbs=std::max(envelope.split_limbs,q.split_limbs);
            envelope.window_limbs=std::min(envelope.window_limbs,q.window_limbs);
            envelope.right_limbs=std::max(envelope.right_limbs,q.right_limbs);
        }
        SplitPlan shared{};const auto rc=split_plan(envelope,1,applications,shared);
        if(rc!=SBN3_SUPPORTED)return status=rc;
        if(!shared.cyclic.enabled)continue;
        shared.cache_slot=int(j);
        for(unsigned k=j;k<class_count;++k){auto &q=classes[k];
            if(!q.group&&q.frontier&&q.level==c.level)q.split=shared;}
    }
    frontier_region_bytes = 0;
    for (unsigned j = 0; j < class_count; ++j) {
        auto &c = classes[j];
        if (!c.frontier)
            continue;
        if(!c.group){
            const auto rc = c.split.cache_slot>=0?SBN3_SUPPORTED:split_plan(c, 1, uses[j], c.split);
            if (rc != SBN3_SUPPORTED)return status = rc;
            c.region_bytes = align_to(std::max({c.split.episode_bytes(), classes[c.left].region_bytes,
                                                           classes[c.right].region_bytes}), 64);
        }
        frontier_region_bytes = std::max(frontier_region_bytes, c.region_bytes);
    }
    programs = {};
    ring_stages = 0;
    for (unsigned j = 0; j < extra_count; ++j)
        if(!extra[j].cached.enabled)programs.account(extra[j]);
    // One prepared wrap-around product per class: the first split plan that asks for it (bind prepares the same one).
    bool wrapped[max_classes]{};
    cyclic_bytes = 0;
    for(unsigned j=0;j<extra_count;++j)if(extra[j].cached.enabled)
        cyclic_bytes += align_to(extra[j].cached.prepared_bytes(),128);
    auto wrap = [&](unsigned j, const RailProductPlan &plan) {
        if (!wrapped[j])
            cyclic_bytes += align_to(plan.prepared_bytes(), 128);
        wrapped[j] = true;
    };
    for (unsigned j = 0; j < class_count; ++j) {
        const auto &c = classes[j];
        if (c.group || !c.frontier)
            continue;
        if (c.split.cyclic.enabled)
            wrap(c.split.cache_slot<0?j:unsigned(c.split.cache_slot), c.split.cyclic);
        else if(!c.split.middle_words&&!c.split.direct)
            programs.account(c.split.product);
    }
    // Stages of every tree: node counts flow from each top class to its children; classes were created
    // children first, so descending index order visits parents before children... but not by size.
    for (unsigned t = 0; t < tree_count; ++t) {
        auto &tree = trees[t];
        const auto &root = classes[tree.root];
        tree.tasks = root.tasks;
        tree.top_nodes = root.top_nodes;
        uint64_t count[max_classes]{};
        count[tree.root] = 1;
        // order the top classes by descending size
        int order[max_classes];
        unsigned tops = 0;
        for (unsigned j = 0; j < class_count; ++j)
            if (!classes[j].frontier)
                order[tops++] = int(j);
        std::sort(order, order + tops, [&](int a, int b) { return classes[a].fragments > classes[b].fragments; });
        uint32_t first = 0;
        for (unsigned k = 0; k < tops; ++k) {
            const int j = order[k];
            if (!count[j])
                continue;
            const auto &c = classes[j];
            int partner=-1;
            if(!repeated)for(unsigned h=k+1;h<tops;++h){const int other=order[h];
                // Equal powers do not imply independent nodes: a coarse
                // 3-block node can produce a 2-block child using that same
                // power. Only merge classes larger than either child.
                if(count[other]&&classes[other].fragments>std::max(classes[c.left].fragments,classes[c.right].fragments)&&
                   classes[other].level==c.level&&classes[other].left_fragments==c.left_fragments){partner=other;break;}}
            const uint64_t primary=count[j],secondary=partner>=0?count[partner]:0;
            count[c.left] += count[j];
            count[c.right] += count[j];
            if(partner>=0){const auto&q=classes[partner];count[q.left]+=secondary;count[q.right]+=secondary;count[partner]=0;}
            if (tree.stage_count == max_stages)
                return status = SBN3_QUERY_CAPACITY;
            auto &s = tree.stages.emplace(tree.stage_count++);
            s.node_class = j;
            s.partner_class=partner;s.main_count=uint32_t(primary);
            s.count = uint32_t(primary+secondary);
            s.first = first;
            first += s.count;
            // Concurrent groups: as many as there are nodes, but groups keep fork_min_workers while nodes are large.
            const unsigned floor_workers = c.limbs < policy.team_node_limbs ? 1u : std::min(policy.fork_min_workers, top_workers);
            s.groups = std::min<uint32_t>({pow2_floor(unsigned(std::min<uint64_t>(s.count, 32))), top_workers / floor_workers,
                                           c.limbs < policy.team_node_limbs ? 32u : ring_max_groups});
            s.workers = top_workers / s.groups;
            NodeClass envelope=c;
            if(partner>=0)envelope.window_limbs=std::min(c.window_limbs,classes[partner].window_limbs);
            if(policy.staged_serial_fft&&balanced&&!repeated&&s.workers>1&&c.limbs<policy.ring_node_limbs){
                const size_t common=limbs_for_bits(power_bits(base.odd_bound,fragment_digits*c.left_fragments));
                const size_t period=std::max(c.split_limbs,c.split_limbs+common-envelope.window_limbs+1);
                RailProductPlan cache;
                // This is a capability check. The ordinary split lowering
                // below pins the same FFT geometry and its exact witness.
                if(common>=native_policy::window_fft_min_words&&
                   rail_product_plan(c.split_limbs,common,period,cache,s.count,policy.cyclic_min_fresh)){
                    s.workers=1;s.groups=std::min<uint32_t>(s.count,top_workers);
                }
            }
            const auto rc = split_plan(envelope, s.workers, s.count, s.split);
            if (rc != SBN3_SUPPORTED)
                return status = rc;
            if (s.split.cyclic.enabled)
                wrap(unsigned(j), s.split.cyclic);
            else if (s.split.ring.enabled) {
                s.ring_stage = int(ring_stages++);
                // A rare carry witness runs after unbinding the cyclic stage.
                // Its temporary product reuses the same pool, one node at a
                // time, instead of reserving one extra workspace per group.
                tree.pool_bytes = std::max({tree.pool_bytes, s.split.ring.pool_bytes(s.groups),s.split.low_bytes});
            } else if(!s.split.middle_words&&!s.split.direct)
                programs.account(s.split.product);
            const bool ordinary=!s.split.wraps()&&!s.split.direct&&!s.split.middle_words;
            s.borrow_result=ordinary&&size_t(s.groups)*align_to(s.split.product.output_limbs*8,64)+63<=root.fragments*fragment_digits;
            const size_t episode=size_t(s.groups)*align_to(s.episode_bytes(),64);
            s.borrow_output=s.split.ring.enabled&&episode+63<=root.fragments*fragment_digits;
            if(!s.borrow_output)tree.episode_bytes=std::max(tree.episode_bytes,episode);
        }
        require(first == tree.top_nodes, SBN3_FATAL_MATH, "radix stage instances", first, tree.top_nodes);
    }
    if(!repeated){
        const RailProductPlan *caches[max_balanced_powers]{};
        auto keep=[&](const NodeClass &c,const SplitPlan &s){
            if(s.cyclic.enabled&&!caches[c.level])caches[c.level]=&s.cyclic;};
        for(unsigned j=0;j<class_count;++j)if(!classes[j].group&&classes[j].frontier)keep(classes[j],classes[j].split);
        for(unsigned t=0;t<tree_count;++t)for(unsigned j=0;j<trees[t].stage_count;++j){const auto &s=trees[t].stages[j];keep(classes[s.node_class],s.split);}
        status=powers.finish(base,workers,transcript,caches);if(status!=SBN3_SUPPORTED)return status;
        rail.total_limbs=powers.total_limbs;rail.setup_bytes=powers.setup_bytes;
    }
    return status;
}
size_t FormatTreeView::packed_bytes()const noexcept {
    size_t bytes=align_to(sizeof(FormatTreeView),64)+align_to(sizeof(NodeClass)*class_count,64)+
        align_to(sizeof(BalancedPowers::Entry)*powers.count,64);
    for(unsigned t=0;t<tree_count;++t)bytes+=align_to(sizeof(Stage)*trees[t].stage_count,64);
    return bytes;
}
FormatTreeView *FormatTreeView::pack(void *storage,size_t bytes)const noexcept {
    require(bytes>=packed_bytes()&&!(uintptr_t(storage)&63),SBN3_FATAL_WORKSPACE,"compact radix plan");
    auto *p=::new(storage)FormatTreeView(*this);
    auto *at=static_cast<unsigned char *>(storage)+align_to(sizeof(FormatTreeView),64);
    p->classes.reset(at);for(unsigned j=0;j<class_count;++j)p->classes.emplace(j,classes[j]);
    at+=align_to(sizeof(NodeClass)*class_count,64);
    p->powers.entries.reset(at);for(unsigned j=0;j<powers.count;++j)p->powers.entries.emplace(j,powers.entries[j]);
    at+=align_to(sizeof(BalancedPowers::Entry)*powers.count,64);
    for(unsigned t=0;t<max_trees;++t){p->trees[t].stages.reset(at);
        for(unsigned j=0;j<trees[t].stage_count;++j)p->trees[t].stages.emplace(j,trees[t].stages[j]);
        at+=align_to(sizeof(Stage)*trees[t].stage_count,64);}
    p->transcript=nullptr;return p;
}
size_t FormatTreeView::work_prefix_bytes(int t) const noexcept {
    const auto &tree = trees[t];
    return align_to(size_t(tree.top_nodes) * sizeof(FormatInstance), 64) + align_to(size_t(tree.tasks) * sizeof(FormatTask), 64) +
           batch_bytes(t)+seam_bytes(t)+
           align_to(root_storage_words(t)*8,64);
}
size_t FormatTreeView::work_bytes(int t) const noexcept {
    return work_prefix_bytes(t)+std::max(trees[t].episode_bytes,
        size_t(classes[trees[t].root].group?1:workers)*align_to(frontier_region_bytes,64));
}
size_t FormatTreeView::stage_work_bytes(int t,unsigned s)const noexcept {
    const auto &tree=trees[t];const auto &stage=tree.stages[s];
    size_t bytes=stage.borrow_output?0:size_t(stage.groups)*align_to(stage.episode_bytes(),64);
    if(s+1==tree.stage_count)bytes=std::max(bytes,size_t(workers)*align_to(frontier_region_bytes,64));
    return work_prefix_bytes(t)+bytes;
}
size_t FormatTreeView::stage_pool_offset(int t,unsigned s,size_t at)const noexcept {
    const auto &r=trees[t].stages[s].split.ring;
    const size_t alignment=std::max({size_t(4096),r.work_alignment,r.spectrum_alignment});
    return align_to(at+stage_work_bytes(t,s),alignment);
}
size_t FormatTreeView::runtime_bytes(int t,size_t at)const noexcept {
    size_t bytes=work_bytes(t);const auto &tree=trees[t];
    for(unsigned s=0;s<tree.stage_count;++s){const auto &stage=tree.stages[s];
        if(stage.split.ring.enabled)bytes=std::max(bytes,stage_pool_offset(t,s,at)-at+
            std::max(stage.split.ring.pool_bytes(stage.groups),stage.split.low_bytes));}
    return bytes;
}
uint64_t *format_tree_root_buffer(FormatTree &t,int index) noexcept {
    const auto &p=*t.plan;const auto &tree=p.trees[index];const auto &c=p.classes[tree.root];
    require(t.work.bytes>=p.work_bytes(index),SBN3_FATAL_WORKSPACE,"radix root slab");
    const size_t metadata=align_to(size_t(tree.top_nodes)*sizeof(FormatInstance),64)+
                          align_to(size_t(tree.tasks)*sizeof(FormatTask),64)+p.batch_bytes(index)+p.seam_bytes(index);
    auto *slab=reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(t.work.data)+metadata);
    return slab+p.root_storage_words(index)-c.limbs;
}
void format_tree_bind(FormatTree &t, const FormatTreeView &plan, const uint8_t *alphabet, sbn3_arena &arena,
                      sbn3_team &team, const sbn3_lease &prepared, sbn3_lease &work,
                      uint64_t *rail_storage, ProductProgram *programs, RailProduct *cyclic, RingStage *rings,
                      size_t pool_offset,bool shared_pool) noexcept {
    t = {};
    t.rings = rings;
    t.ring_arena = &arena;
    t.pool_offset = pool_offset;
    t.shared_pool=shared_pool;
    t.plan = &plan;
    t.arena = &arena;
    t.team = &team;
    t.work = work;
    t.programs = programs;
    t.cyclic = cyclic;
    require(digit_plan_init(t.digits, plan.base.base, alphabet), SBN3_FATAL_ARGUMENT, "radix alphabet");
    if(plan.block_leaves)t.block_plan=&fraction_block_plan(plan.base.base);
    require(work.bytes >= plan.rail.setup_bytes && !(uintptr_t(work.data) & 63), SBN3_FATAL_WORKSPACE,
            "radix tree storage");
#ifdef SBN3_RADIX_TRACE
    const uint64_t bind_start = trace_now();
#endif
    require(prepared.bytes >= plan.prepared_bytes(), SBN3_FATAL_WORKSPACE, "radix prepared storage", plan.prepared_bytes(),
            prepared.bytes);
    const size_t cyclic_at = align_to(plan.programs.bytes(), 128);
    Frame spectra = Frame::borrow(arena, prepared, static_cast<uint8_t *>(prepared.data) + cyclic_at, prepared.bytes - cyclic_at);
    for(unsigned j=0;j<plan.class_count;++j)::new (&cyclic[j]) RailProduct{};
    RailProduct *power_caches[max_balanced_powers]{};
    auto reserve=[&](const NodeClass &c,const SplitPlan &split){
        if(!split.cyclic.enabled)return;
        const unsigned index=split.cache_slot<0?unsigned(plan.classes.index_of(&c)):unsigned(split.cache_slot);
        if(!cyclic[index].tables)rail_product_reserve(cyclic[index],split.cyclic,spectra);
        if(!power_caches[c.level])power_caches[c.level]=&cyclic[index];
    };
    for(unsigned j=0;j<plan.class_count;++j)if(!plan.classes[j].group&&plan.classes[j].frontier)reserve(plan.classes[j],plan.classes[j].split);
    for(unsigned k=0;k<plan.tree_count;++k)for(unsigned s=0;s<plan.trees[k].stage_count;++s){const auto &stage=plan.trees[k].stages[s];reserve(plan.classes[stage.node_class],stage.split);}
    if(plan.repeated)rail_build(plan.base, plan.rail, plan.workers, arena, team, work, rail_storage, t.rail);
    else balanced_powers_build(plan.base,plan.powers,plan.workers,arena,team,work,rail_storage,t.rail,power_caches);
    t.work = work; // (re-acquired by the rail build)
#ifdef SBN3_RADIX_TRACE
    const uint64_t rail_done = trace_now();
#endif
    ProgramSetBuilder builder(plan.programs, arena, prepared);
    for(unsigned j=0;j<plan.extra_count;++j)scale_product_prepare(t.extra[j],plan.extra[j],builder,spectra);
    for (unsigned j = 0; j < plan.program_slots(); ++j)
        ::new (&programs[j]) ProductProgram{};
    auto prepare = [&](const NodeClass &c, const SplitPlan &split, ProductProgram &slot) {
        const unsigned index = split.cache_slot<0?unsigned(plan.classes.index_of(&c)):unsigned(split.cache_slot);
        if (split.ring.enabled || split.middle_words || split.direct)
            return; // replayed below, bound stage by stage at run time
        if (!split.cyclic.enabled)
            slot = builder.prepare(split.product);
        else if (!cyclic[index].common)
            rail_product_forward(cyclic[index],t.rail[c.level]);
    };
    for (unsigned j = 0; j < plan.class_count; ++j)
        if (!plan.classes[j].group && plan.classes[j].frontier)
            prepare(plan.classes[j], plan.classes[j].split, programs[j]);
    for (unsigned k = 0; k < plan.tree_count; ++k)
        for (unsigned s = 0; s < plan.trees[k].stage_count; ++s) {
            const auto &stage = plan.trees[k].stages[s];
            prepare(plan.classes[stage.node_class], stage.split, programs[plan.stage_program(k,s)]);
            if (stage.ring_stage >= 0) {
                ::new (&rings[stage.ring_stage]) RingStage{};
                ring_replay(stage.split.ring, uint64_t(stage.ring_stage) + 1, rings[stage.ring_stage]);
            }
        }
#ifdef SBN3_RADIX_TRACE
    fprintf(stderr, "bind: rail %.3f ms, programs and spectra %.3f ms (%u classes)\n", (rail_done - bind_start) / 1e6,
            (trace_now() - rail_done) / 1e6, plan.class_count);
#endif
}
namespace {
// ---- execute ----------------------------------------------------------------------------------------
struct Run {
    FormatTree *tree;
    uint8_t *out;
    uint64_t *first, *overlap;
    FragmentBatch *batches=nullptr;
    uint8_t *blocks=nullptr;
    uint64_t *seam_ends=nullptr;
};
void flush_blocks(Run &run,const NodeClass &c,BlockBatch &batch,Frame &scratch) noexcept {
    if(!batch.count)return;
    auto *values=reinterpret_cast<uint64_t *>(reinterpret_cast<uint8_t *>(&batch)+align_to(sizeof(BlockBatch),64));
    const size_t stride=align_to(c.limbs,8);
    FractionBlock blocks[8];
    for(unsigned j=0;j<batch.count;++j){const auto at=batch.position[j],seam=batch.seam[j];
        blocks[j]={values+j*stride,unsigned(c.limbs),unsigned(c.fragments),run.out+64*at,run.first+seam,run.overlap+seam};}
    emit_fraction_block_tree(blocks,batch.count,run.tree->digits,*run.tree->block_plan,scratch);
    batch.count=0;
}
void flush_fragments(Run&run,FragmentBatch&batch) noexcept {
    if(!batch.count)return;
    const auto&p=*run.tree->plan;const uint64_t*input[8]{};
    for(unsigned j=0;j<batch.count;++j)input[j]=batch.value[j];
    alignas(64) uint8_t out[512];uint64_t first[8],overlap[8];
    emit_fragments(out,first,overlap,input,unsigned(p.group_limbs[1]),p.fragment_u52,batch.count,run.tree->digits);
    for(unsigned j=0;j<batch.count;++j){const auto at=batch.position[j];
        memcpy(run.out+64*at,out+64*j,64);run.first[at]=first[j];run.overlap[at]=overlap[j];}
    batch.count=0;
}
void leaf_group(Run &run, const NodeClass &c, const uint64_t *y, uint64_t fragment,Frame *scratch,unsigned worker=0,uint64_t seam=0) noexcept {
    auto &t = *run.tree;
#ifdef SBN3_RADIX_TRACE
    trace_add(trace_rows[t.plan->classes.index_of(&c)].count, 1);
#endif
    SBN3_RADIX_SPAN(span, trace_rows[t.plan->classes.index_of(&c)].leaf_ns);
    const auto &p = *t.plan;
    if(p.block_leaves){
        require(scratch,SBN3_FATAL_WORKSPACE,"block leaf workspace");
        run.seam_ends[seam]=fragment+c.fragments;
        if(!run.blocks){const FractionBlock block{y,unsigned(c.limbs),unsigned(c.fragments),run.out+64*fragment,
            run.first+seam,run.overlap+seam};
            emit_fraction_block_tree(&block,1,t.digits,*t.block_plan,*scratch);return;}
        auto &batch=*reinterpret_cast<BlockBatch *>(run.blocks+size_t(worker)*p.block_batch_bytes+c.block_batch_offset);
        auto *values=reinterpret_cast<uint64_t *>(reinterpret_cast<uint8_t *>(&batch)+align_to(sizeof(BlockBatch),64));
        const unsigned j=batch.count++;memcpy(values+j*align_to(c.limbs,8),y,c.limbs*8);
        batch.position[j]=fragment;batch.seam[j]=seam;
        if(batch.count==8)flush_blocks(run,c,batch,*scratch);return;
    }
    const uint64_t *lane[format_group_fragments]{};
    uint64_t pool[format_group_fragments - 1][group_right_limbs];
    uint64_t z[group_product_limbs];
    unsigned used = 0;
    struct Item {
        const uint64_t *y;
        size_t limbs;
        unsigned n, slot;
    } stack[4];
    unsigned depth = 0;
    Item it{y, c.limbs, unsigned(c.fragments), 0};
    for (;;) {
        while (it.n > 1) {
            const unsigned k = split_level(it.n), left = 1u << k, right = it.n - left;
            const size_t right_limbs = p.group_limbs[right], left_limbs = p.group_limbs[left];
            const size_t qn = p.rail.limbs[k];
            // the top twos * 2^k limbs only reach limbs above the window
            const size_t used_limbs = it.limbs - (size_t(p.base.twos) << k);
            if(p.group_u52>>it.n&1){
                require(scratch,SBN3_FATAL_WORKSPACE,"radix leaf scratch");
                u52::multiply(z,it.y,used_limbs,t.rail[k],qn,*scratch);
            }else sbn3_mul_basecase(z, used_limbs + qn, it.y, used_limbs, t.rail[k], qn);
            uint64_t *r = pool[used++];
            take_window(r, right_limbs, z, used_limbs + qn, window_bit(p.base, it.limbs, right_limbs, k));
            stack[depth++] = {r, right_limbs, right, it.slot + left};
            it = {it.y + (it.limbs - left_limbs), left_limbs, left, it.slot};
        }
        lane[it.slot] = it.y;
        if (!depth)
            break;
        it = stack[--depth];
    }
    for(unsigned j=0;j<c.fragments;){
        const unsigned count=unsigned(std::min<uint64_t>(8,c.fragments-j));
        // Full groups already occupy all lanes. Pending fragments have
        // disjoint output positions and can wait while this group emits
        // directly; they do not require repacking the whole stream.
        if(!run.batches || count==8){
            const uint64_t*inputs[8]{};for(unsigned u=0;u<count;++u)inputs[u]=lane[j+u];
            emit_fragments(run.out+(fragment+j)*fragment_digits,run.first+fragment+j,run.overlap+fragment+j,inputs,
                           unsigned(p.group_limbs[1]),p.fragment_u52,count,t.digits);j+=count;
        }else{
            auto&batch=run.batches[worker];const unsigned at=batch.count++;
            memcpy(batch.value[at],lane[j],p.group_limbs[1]*8);batch.position[at]=fragment+j;++j;
            if(batch.count==8)flush_fragments(run,batch);
        }
    }
}
// The right child of a node: the window of y * rail[level].
void split(Run &run, int index, const SplitPlan &plan, const ProductProgram &program, sbn3_mul_binding *consumer,
           const uint64_t *y, uint64_t *right, uint8_t *episode, sbn3_team_scope *scope,int recipe_class=-1,
           unsigned *pending=nullptr,uint64_t *external_result=nullptr) noexcept {
    auto &t = *run.tree;
    const auto &c = t.plan->classes[index];
    auto *z = external_result?external_result:reinterpret_cast<uint64_t *>(episode);
    const size_t window = c.window_limbs;
#ifdef SBN3_RADIX_TRACE
    trace_add(trace_rows[index].count,1);
#endif
    SBN3_RADIX_SPAN(split_span,trace_rows[index].product_ns);
    if(plan.direct){
        if(plan.direct_native){auto*at=episode+align_to(plan.product.output_limbs*8,64);
            auto frame=t.roots[sbn3_team_first_worker(scope)]->borrowed_view(at,plan.product.work_bytes);
            u52::multiply(z,y,c.split_limbs,t.rail[c.level],plan.product.bn,frame);
        }else sbn3_mul_basecase(z,c.split_limbs+plan.product.bn,y,c.split_limbs,t.rail[c.level],plan.product.bn);
        memcpy(right,z+window,c.right_limbs*8);return;
    }
    if(plan.middle_words){
        auto *work=episode+align_to(plan.middle_words*8,64);
        {
            auto frame=t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work,plan.middle_bytes);
            if(plan.middle_guards==6)u52::middle_guard6(z,t.rail[c.level],plan.product.bn,y,plan.product.an,frame);
            else u52::middle_guard(z,t.rail[c.level],plan.product.bn,y,plan.product.an,frame);
        } // retire/unpoison the middle workspace before reusing its bytes
        // Omitted carry is <2*na*2^52. Its carry into word 1 is at most
        // ceil(na/2048); the requested window begins at bit 128 or above.
        const size_t na=(64*plan.product.bn+51)/52;
        const uint64_t carry_bound=(na+2047)/2048;
        if(z[1]<=UINT64_MAX-carry_bound){
            take_window(right,c.right_limbs,z,plan.middle_words,plan.middle_shift);
            return;
        }
        // Finish all reads before publishing into a possibly aliased child.
        auto repair=t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work,plan.low_bytes);
        product::WordWindowPlan witness;
        require(product::word_window_query(plan.product.an,plan.product.bn,64*window,1,witness,plan.low_choice)==SBN3_SUPPORTED,
                SBN3_FATAL_MATH,"middle carry word plan");
        const uint64_t exact=product::word_window_execute(witness,{y,plan.product.an},{t.rail[c.level],plan.product.bn},
                                                         64*window,repair,t.team,UINT64_MAX,scope);
        take_window(right,c.right_limbs,z,plan.middle_words,plan.middle_shift);
        const uint64_t delta=exact-right[0];require(delta<=1,SBN3_FATAL_MATH,"middle word correction bound");
        if(delta)for(size_t j=0;j<c.right_limbs&&++right[j]==0;++j){}
        return;
    }
    if (plan.wraps()) {
        // Wrap-around product with the cached rail spectrum (rail_product.hpp, ring_product.hpp): exact window.
        const size_t residue = plan.cyclic.enabled ? plan.cyclic.output_limbs() : plan.ring.output_limbs;
        auto *work = episode + align_to(residue * 8, 64) + 64;
        auto frame = plan.cyclic.enabled?t.roots[sbn3_team_first_worker(scope)]->borrowed_view(
            work,std::max(plan.cyclic.scratch_bytes,plan.low_bytes)):Frame::external(nullptr,0);
        {
            if (plan.cyclic.enabled) {
                const int owner=plan.cache_slot<0?(recipe_class<0?index:recipe_class):plan.cache_slot;
                if(rail_product_try_window(t.cyclic[owner],frame,y,c.split_limbs,window,c.right_limbs,z,right))return;
                rail_product_execute(t.cyclic[owner], frame, scope, y, z,c.split_limbs);
            } else {
                sbn3_product_inputs in{};
                if(!plan.ring.cached)in.a={t.rail[c.level],plan.ring.common_limbs};
                in.b = {y, c.split_limbs};
                if(plan.ring.cached && c.split_limbs<plan.ring.fresh_limbs){
                    require(consumer->backend->product_execute_live,SBN3_FATAL_MATH,"radix bounded cyclic capability");
                    consumer->backend->product_execute_live(consumer,scope,in,{z,plan.ring.output_limbs});
                }else sbn3_product_execute_on_scope(consumer, scope, &in, {z, plan.ring.output_limbs});
            }
        }
        SBN3_RADIX_SPAN(window_span, trace_rows[index].window_ns);
        if(pending&&*pending>=2){
            memcpy(right,z+window,c.right_limbs*8);
            if(*pending==3)for(size_t j=0;j<c.right_limbs&&right[j]--==0;++j){}
            *pending=0;return;
        }
        // If the complete product fits in the ring, no high part wraps.
        if(!plan.low_limbs){memcpy(right,z+window,c.right_limbs*8);return;}
        require(plan.low_limbs<=c.split_limbs&&plan.low_limbs<=window,SBN3_FATAL_MATH,"radix shared wrap gap");
        uint64_t clean = 0;
        for (size_t j = plan.low_limbs-1; j < window; ++j)
            clean |= z[j];
        bool carried=false;
        if (!clean) {
            // Gap all zero: either nothing carried, or a carry ran through a gap of ones. Limb h of the
            // exact product of the low limbs tells which.
            const size_t h = plan.low_limbs - 1;
            uint64_t any = 0;
            for (size_t j = 0; j < plan.low_limbs && !any; ++j)
                any = y[j];
            if (any) { // (zero low limbs of the fraction make limb h zero)
                if(plan.ring.enabled){
                    require(pending,SBN3_FATAL_MATH,"radix staged carry witness");
                    *pending=1;return; // leave the aliased source intact
                }
                product::WordWindowPlan witness;
                require(product::word_window_query(plan.low_limbs,plan.low_rail_limbs,64*h,1,witness,plan.low_choice)==SBN3_SUPPORTED,
                        SBN3_FATAL_MATH,"radix carry word plan");
                const auto exact=product::word_window_execute(witness,{y,plan.low_limbs},{t.rail[c.level],plan.low_rail_limbs},
                                                     64*h,frame,t.team,UINT64_MAX,scope);
                carried=exact!=0;
            }
        }
        // The right slot may overlap the low source limbs. Finish the exact
        // witness read before publishing the child into that slot.
        memcpy(right,z+window,c.right_limbs*8);
        if(carried)for(size_t j=0;j<c.right_limbs&&right[j]--==0;++j){}
        return;
    }
    const auto &s = plan.product;
    const uintptr_t after = reinterpret_cast<uintptr_t>(episode) + (external_result?0:align_to(s.output_limbs * 8, 64));
    auto *work = reinterpret_cast<uint8_t *>((after + s.work_alignment - 1) & ~uintptr_t(s.work_alignment - 1));
    auto frame = t.roots[sbn3_team_first_worker(scope)]->borrowed_view(work, s.work_bytes);
    {
        if(c.split_limbs<s.an)product_program_bounded(program,frame,scope,{y,c.split_limbs},{t.rail[c.level],s.bn},{z,s.output_limbs});
        else product_program_execute(program, frame, scope, {y, s.an}, {t.rail[c.level], s.bn}, {z, s.output_limbs});
    }
    SBN3_RADIX_SPAN(window_span, trace_rows[index].window_ns);
    parallel_limbs::each_scope(scope, c.right_limbs, parallel_limbs::scope_parts(scope, c.right_limbs),
                               [&](size_t b, size_t e, unsigned) { memcpy(right + b, z + window + b, (e - b) * 8); });
}
// A frontier subtree, serially inside `region`.
void serial_node(Run &run, int index, const uint64_t *y, uint64_t *storage,uint8_t *region, uint64_t fragment, uint64_t seam, sbn3_team_scope *scope) noexcept {
    auto &t = *run.tree;
    const auto &p = *t.plan;
    const auto &c = p.classes[index];
    if (c.group) {
        if(c.region_bytes){
            auto scratch=t.roots[sbn3_team_first_worker(scope)]->borrowed_view(region,c.region_bytes);
            leaf_group(run,c,y,fragment,&scratch,sbn3_team_first_worker(scope),seam);
        }else leaf_group(run,c,y,fragment,nullptr,sbn3_team_first_worker(scope),seam);
        return;
    }
    const auto &left=p.classes[c.left],&r=p.classes[c.right];
    auto *right=storage+r.slot_limbs-r.limbs;
    split(run,index,c.split,t.programs[index],nullptr,y,right,region,scope);
    serial_node(run,c.right,right,storage,region,fragment+c.left_fragments,seam+left.seams,scope);
    serial_node(run,c.left,y+(c.limbs-left.limbs),storage+r.slot_limbs,region,fragment,seam,scope);
}
// Instances of every stage and the frontier tasks, in depth-first order.
struct Collector {
    const FormatTreeView *plan;
    FormatInstance *instances;
    FormatTask *tasks;
    uint32_t next[max_classes];
    uint32_t task = 0;
    void visit(int index, const uint64_t *y, uint64_t *storage, uint64_t fragment, uint64_t seam) noexcept {
        const auto &c = plan->classes[index];
        if (c.frontier) {
            tasks[task++] = {index, y, storage, fragment, seam};
            return;
        }
        const size_t right_capacity=plan->classes[c.right].slot_limbs;
        auto *right=storage+right_capacity-c.right_limbs;
        instances[next[index]++] = {y, right,index};
        const auto &l = plan->classes[c.left];
        visit(c.left,y+(c.limbs-l.limbs),storage+right_capacity,fragment,seam);
        visit(c.right,right,storage,fragment+c.left_fragments,seam+l.seams);
    }
};
struct RootTask {
    Run *run;
    int tree;
    FormatInstance *instances;
    FormatTask *tasks;
    uint8_t *episodes, *regions;
    size_t region_bytes;
    sbn3_team_scope *scope;
    // the stage being fanned out
    const Stage *stage;
    const ProductProgram *program;
    const RingBound *bound;
    unsigned first_stage, last_stage; // stages [first, last) of one team episode
    bool frontier;
    bool repair_pass=false;
};
struct GroupTask {
    RootTask *root;
    uint32_t low, high; // groups [low, high) of the current stage
};
void fan(void *v, sbn3_team_scope *scope) {
    auto &g = *static_cast<GroupTask *>(v);
    auto &k = *g.root;
    const auto &stage = *k.stage;
    if (g.high - g.low > 1) {
        const uint32_t mid = g.low + (g.high - g.low) / 2;
        GroupTask a{g.root, g.low, mid}, b{g.root, mid, g.high};
        sbn3_team_invoke2(scope, SBN3_PARALLEL_CHILDREN, sbn3_team_width(scope) / 2, fan, &a, fan, &b);
        return;
    }
    // one group: a contiguous share of the stage's nodes, its own episode memory
    const uint32_t q = stage.count / stage.groups, r = stage.count % stage.groups;
    const uint32_t begin = g.low * q + std::min(g.low, r), end = begin + q + (g.low < r);
    uint8_t *episode = k.episodes + size_t(g.low) * align_to(stage.episode_bytes(), 64);
    auto *result=stage.borrow_result?reinterpret_cast<uint64_t *>(
        ((reinterpret_cast<uintptr_t>(k.run->out)+63)&~uintptr_t(63))+
        size_t(g.low)*align_to(stage.split.product.output_limbs*8,64)):nullptr;
    sbn3_team_scope active{scope->team,scope->first,std::min(stage.workers,scope->width),false,scope->epoch};
    for (uint32_t j = begin; j < end; ++j) {
        auto &node = k.instances[stage.first + j];
        if(k.repair_pass&&node.repair<2)continue;
        split(*k.run, node.node_class, stage.split, *k.program, k.bound ? k.bound->consumers[g.low] : nullptr, node.y,
              node.right, episode, &active,stage.node_class,&node.repair,result);
    }
}
void frontier_tasks(void *v, uint64_t begin, uint64_t end, unsigned rank) {
    auto &k = *static_cast<RootTask *>(v);
    // One worker of the running loop: a scope of its own for the serial products.
    sbn3_team_scope self{k.scope->team, k.scope->first + rank, 1, false, k.scope->epoch};
    uint8_t *region = k.regions + size_t(rank) * k.region_bytes;
    for (uint64_t j = begin; j < end; ++j)
        serial_node(*k.run, k.tasks[j].node, k.tasks[j].y, k.tasks[j].storage,region, k.tasks[j].fragment, k.tasks[j].seam, &self);
}
} // namespace
size_t increment_digits(uint8_t *out, size_t count, const DigitPlan &p) noexcept {
    const uint8_t top = uint8_t(p.base - 1);
    size_t j = count;
    while (j-- > 0) {
        const uint8_t v = p.identity ? out[j] : p.decode[out[j]];
        if (v != top) {
            out[j] = p.encode[v + 1];
            return j;
        }
        out[j] = p.encode[0];
    }
    return 0;
}
uint64_t format_tree_run(FormatTree &t, int tree_index, const uint64_t *y, uint8_t *out, uint64_t *first,
                         uint64_t *overlap) noexcept {
    const auto &p = *t.plan;
    const auto &tree = p.trees[tree_index];
    const auto &c = p.classes[tree.root];
    const size_t entry_work_bytes=t.work.bytes;
    const size_t work_offset=size_t(static_cast<uint8_t *>(t.work.data)-t.arena->base);
    require(t.work.bytes >= p.work_bytes(tree_index), SBN3_FATAL_WORKSPACE, "radix tree storage", p.work_bytes(tree_index),
            t.work.bytes);
    Run run{&t, out, first, overlap};
    if (c.group) {
        auto *metadata=static_cast<uint8_t *>(t.work.data)+align_to(size_t(tree.tasks)*sizeof(FormatTask),64);
        if(p.block_leaves)run.seam_ends=reinterpret_cast<uint64_t *>(metadata);
        if(c.region_bytes){
            auto *region=metadata+p.seam_bytes(tree_index)+
                         align_to(p.root_storage_words(tree_index)*8,64);
            auto scratch=Frame::borrow(*t.arena,t.work,region,c.region_bytes);
            leaf_group(run,c,y,0,&scratch);
        }else leaf_group(run,c,y,0,nullptr);
    } else {
        struct alignas(64) RootStorage {
            unsigned char bytes[(sizeof(Frame) + 63) & ~size_t(63)];
        };
        RootStorage storage[32];
        const unsigned workers = sbn3_team_workers(t.team);
        auto close_frames=[&]{for(unsigned j=0;j<workers;++j)if(t.roots[j]){t.roots[j]->~Frame();t.roots[j]=nullptr;}};
        auto open_frames=[&]{for(unsigned j=0;j<workers;++j){
            auto *frame=::new(storage[j].bytes)Frame(*t.arena,t.work);frame->allocate(t.work.bytes);t.roots[j]=frame;}};
        auto reframe=[&](size_t bytes){if(!t.shared_pool||bytes==t.work.bytes)return;
            // Every prior team episode and product binding has joined. Only
            // the prefix values survive; acquiring the same bytes preserves them.
            close_frames();t.arena->release(t.work);t.work=t.arena->acquire(work_offset,bytes);open_frames();};
        open_frames();
        auto *base = static_cast<uint8_t *>(t.work.data);
        RootTask task{};
        task.run = &run;
        task.tree = tree_index;
        task.instances = reinterpret_cast<FormatInstance *>(base);
        base += align_to(size_t(tree.top_nodes) * sizeof(FormatInstance), 64);
        task.tasks = reinterpret_cast<FormatTask *>(base);
        base += align_to(size_t(tree.tasks) * sizeof(FormatTask), 64);
        if(p.batch_bytes(tree_index)){
            if(p.block_leaves){run.blocks=base;
                for(unsigned j=0;j<p.workers;++j)for(unsigned k=0;k<p.class_count;++k)if(p.classes[k].group)
                    ::new(base+size_t(j)*p.block_batch_bytes+p.classes[k].block_batch_offset)BlockBatch{};
            }else{run.batches=reinterpret_cast<FragmentBatch*>(base);
                for(unsigned j=0;j<p.workers;++j)::new(run.batches+j)FragmentBatch{};}
            base+=p.batch_bytes(tree_index);
        }
        if(p.block_leaves){run.seam_ends=reinterpret_cast<uint64_t *>(base);base+=p.seam_bytes(tree_index);}
        auto *slab=reinterpret_cast<uint64_t *>(base);
        base += align_to(p.root_storage_words(tree_index)*8,64);
        auto *root=slab+p.root_storage_words(tree_index)-c.limbs;
        if(y!=root)parallel_limbs::copy(t.team,root,y,c.limbs);
        y=root;
        task.episodes = base;
        // Every staged split joins and copies its right child into the slab
        // before the frontier loop starts. Their temporary regions are disjoint
        // in time, including when the last stage and frontier share a team run.
        task.regions = base;
        task.region_bytes = align_to(p.frontier_region_bytes, 64);
        Collector collect{&p, task.instances, task.tasks, {}, 0};
        for (unsigned s = 0; s < tree.stage_count; ++s){
            collect.next[tree.stages[s].node_class] = tree.stages[s].first;
            if(tree.stages[s].partner_class>=0)collect.next[tree.stages[s].partner_class]=tree.stages[s].first+tree.stages[s].main_count;
        }
        collect.visit(tree.root,y,slab,0,0);
        require(collect.task == tree.tasks, SBN3_FATAL_MATH, "radix frontier tasks");
        // Episodes: runs of stages that need no binding share one team run; a ring stage is bound by this
        // (owner) thread around its own run. The frontier loop joins the last episode.
        auto episode = [](void *v, sbn3_team_scope *scope) {
            auto &k = *static_cast<RootTask *>(v);
            const auto &plan = *k.run->tree->plan;
            const auto &shape = plan.trees[k.tree];
            sbn3_team_scope active{scope->team,scope->first,std::min(plan.workers,sbn3_team_width(scope)),false,scope->epoch};
            k.scope = &active;
            // The staged part runs on the power-of-two prefix of the team.
            sbn3_team_scope top{scope->team, scope->first, std::min(plan.top_workers, sbn3_team_width(scope)), false,
                                scope->epoch};
            for (unsigned s = k.first_stage; s < k.last_stage; ++s) {
                k.stage = &shape.stages[s];
                k.program = &k.run->tree->programs[plan.stage_program(unsigned(k.tree),s)];
                GroupTask all{&k, 0, k.stage->groups};
                fan(&all, &top);
            }
            bool pending=false;
            if(k.bound)for(uint32_t j=0;j<k.stage->count;++j)pending|=k.instances[k.stage->first+j].repair!=0;
            if (k.frontier&&!pending)
                sbn3_team_for(&active, 0, shape.tasks, 1, SBN3_DYNAMIC, frontier_tasks, &k);
        };
#ifdef SBN3_RADIX_TRACE
        fprintf(stderr, "work breakdown: instances+tasks %.1f MB, persist %.1f MB, stage episodes %.1f MB, frontier regions %u x %.1f MB, pool %.1f MB\n",
                (size_t(tree.top_nodes) * sizeof(FormatInstance) + size_t(tree.tasks) * sizeof(FormatTask)) / 1e6,
                (c.frontier ? 0 : c.persist_bytes) / 1e6, tree.episode_bytes / 1e6, p.workers, p.frontier_region_bytes / 1e6,
                p.pool_bytes() / 1e6);
        for (unsigned s = 0; s < tree.stage_count; ++s)
            fprintf(stderr, "  stage %2u class %3d limbs %9zu count %5u groups %2u x w %2u episode %.1f MB each (%s)\n", s,
                    tree.stages[s].node_class, p.classes[tree.stages[s].node_class].limbs, tree.stages[s].count,
                    tree.stages[s].groups, tree.stages[s].workers, tree.stages[s].split.episode_bytes() / 1e6,
                    tree.stages[s].split.ring.enabled ? "ring" : tree.stages[s].split.cyclic.enabled ? "fft wrap" : "program");
#endif
        RingBound bound{};
        for (unsigned s = 0; s <= tree.stage_count;) {
            unsigned e = s;
            size_t pool_offset=t.pool_offset,pool_bytes=p.pool_bytes();
            if (s < tree.stage_count && tree.stages[s].ring_stage >= 0) {
                const auto &stage = tree.stages[s];
                // All products in this episode finish reading their numeric
                // outputs before the frontier starts writing digit bytes.
                task.episodes=stage.borrow_output?reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(out)+63)&~uintptr_t(63)):base;
                if(t.shared_pool){reframe(p.stage_work_bytes(tree_index,s));
                    pool_offset=p.stage_pool_offset(tree_index,s,work_offset);
                    pool_bytes=std::max(stage.split.ring.pool_bytes(stage.groups),stage.split.low_bytes);}
#ifdef SBN3_RADIX_TRACE
                const uint64_t bind_begin = trace_now(); // ring stage trace
#endif
                ring_bind(bound, stage.split.ring, t.rings[stage.ring_stage], stage.groups, t.ring_arena, pool_offset,
                          pool_bytes, t.team, t.rail[p.classes[stage.node_class].level]);
#ifdef SBN3_RADIX_TRACE
                fprintf(stderr, "ring stage class %d: np %u T %d alg %u ring %zu groups %u x w %u, bind+spectrum %.3f ms\n", stage.node_class,
                        stage.split.ring.np, stage.split.ring.trunk_bits, stage.split.ring.algorithm, stage.split.ring.ring,
                        stage.groups, stage.workers, (trace_now() - bind_begin) / 1e6);
#endif
                task.bound = &bound;
                e = s + 1;
            } else {
                task.episodes=base;
                reframe(p.work_bytes(tree_index));
                task.bound = nullptr;
                while (e < tree.stage_count && tree.stages[e].ring_stage < 0)
                    ++e;
            }
            task.first_stage = s;
            task.last_stage = e;
            task.frontier = e == tree.stage_count;
#ifdef SBN3_RADIX_TRACE
            const uint64_t episode_begin = trace_now();
#endif
            sbn3_team_run(t.team, episode, &task);
#ifdef SBN3_RADIX_TRACE
            const uint64_t episode_end = trace_now();
#endif
            if (task.bound)
                ring_unbind(bound, t.ring_arena);
            if(task.bound){
                const auto &stage=tree.stages[s];const auto &sp=stage.split;
                bool pending=false;
                for(uint32_t j=0;j<stage.count;++j)pending|=task.instances[stage.first+j].repair!=0;
                if(pending){
                    auto lease=t.arena->acquire(pool_offset,sp.low_bytes);
                    {
                        Frame scratch(*t.arena,lease);
                        for(uint32_t j=0;j<stage.count;++j){
                            auto &node=task.instances[stage.first+j];if(node.repair!=1)continue;
                            const auto *power=t.rail[p.classes[node.node_class].level];
                            product::WordWindowPlan witness;
                            const uint64_t bit=64*(sp.low_limbs-1);
                            require(product::word_window_query(sp.low_limbs,sp.low_rail_limbs,bit,sp.low_workers,witness,sp.low_choice)==SBN3_SUPPORTED,
                                    SBN3_FATAL_MATH,"radix staged witness plan");
                            node.repair=2+unsigned(product::word_window_execute(witness,{node.y,sp.low_limbs},
                                {power,sp.low_rail_limbs},bit,scratch,t.team)!=0);
                        }
                    }
                    t.arena->release(lease);
                    // Regenerate only the ambiguous residues after retiring
                    // witness scratch. No fraction copy survives in the pool.
                    ring_bind(bound,sp.ring,t.rings[stage.ring_stage],stage.groups,t.ring_arena,pool_offset,
                              pool_bytes,t.team,t.rail[p.classes[stage.node_class].level]);
                    task.repair_pass=true;sbn3_team_run(t.team,episode,&task);task.repair_pass=false;
                    ring_unbind(bound,t.ring_arena);
                }
            }
#ifdef SBN3_RADIX_TRACE
            fprintf(stderr, "episode stages [%u,%u)%s: run %.3f ms, unbind %.3f ms\n", s, e, task.frontier ? " + frontier" : "",
                    (episode_end - episode_begin) / 1e6, (trace_now() - episode_end) / 1e6);
#endif
            if (task.frontier)
                break;
            s = e;
        }
        if(run.batches)for(unsigned j=0;j<p.workers;++j)flush_fragments(run,run.batches[j]);
        if(run.blocks)for(unsigned j=0;j<p.workers;++j){
            auto scratch=t.roots[j]->borrowed_view(task.regions+size_t(j)*task.region_bytes,p.block_work_bytes);
            for(unsigned k=0;k<p.class_count;++k)if(p.classes[k].group){const auto &leaf=p.classes[k];
                auto &batch=*reinterpret_cast<BlockBatch *>(run.blocks+size_t(j)*p.block_batch_bytes+leaf.block_batch_offset);
                flush_blocks(run,leaf,batch,scratch);}
        }
        close_frames();
        if(t.shared_pool&&t.work.bytes!=entry_work_bytes){
            t.arena->release(t.work);t.work=t.arena->acquire(work_offset,entry_work_bytes);}
    }
#ifdef SBN3_RADIX_TRACE
    for (unsigned j = 0; j < p.class_count; ++j) {
        const auto &k = p.classes[j];
        if (!trace_rows[j].count)
            continue;
        unsigned groups = 1, width = 1, product_workers = 1;
        bool wrapped = k.split.cyclic.enabled;
        for (unsigned s = 0; s < tree.stage_count; ++s)
            if (tree.stages[s].node_class == int(j)) {
                groups = tree.stages[s].groups;
                width = tree.stages[s].workers;
                product_workers = tree.stages[s].split.product.workers;
                wrapped = tree.stages[s].split.cyclic.enabled;
            }
        fprintf(stderr, "class %3u frag %8llu %s groups %2u x w %2u (product w %2u%s) limbs %8zu count %6llu product %9.3f window %8.3f leaf %9.3f ms\n",
                j, (unsigned long long)k.fragments, k.frontier ? "frontier" : "stage   ", groups, width, product_workers,
                wrapped ? ", wrap" : "", k.limbs, (unsigned long long)trace_rows[j].count, trace_rows[j].product_ns / 1e6,
                trace_rows[j].window_ns / 1e6, trace_rows[j].leaf_ns / 1e6);
        trace_rows[j] = {};
    }
    const uint64_t seam_start = trace_now();
#endif
    // Whole-block leaves are exact internally; only their outer seams remain.
    // Legacy leaves retain one seam per 64-digit fragment.
    const uint64_t b8 = t.digits.b8;
    const uint64_t seams=p.seam_count(tree_index);
    for (uint64_t i = seams - 1; i-- > 0;) {
        const uint64_t f = first[i + 1], o = overlap[i];
        if (o == f)
            continue;
        require((o + 1 == b8 ? 0 : o + 1) == f, SBN3_FATAL_MATH, "radix seam", o, f);
        if (f)
            continue;
        const uint64_t start=p.block_leaves?(i?run.seam_ends[i-1]:0):i;
        const uint64_t count=p.block_leaves?run.seam_ends[i]-start:1;
        const size_t changed = increment_digits(out + start * fragment_digits, count*fragment_digits, t.digits);
        if (changed < word_digits)
            first[i] = first[i] + 1 == b8 ? 0 : first[i] + 1;
    }
#ifdef SBN3_RADIX_TRACE
    fprintf(stderr, "seam pass %.3f ms\n", (trace_now() - seam_start) / 1e6);
#endif
    return overlap[seams - 1];
}
} // namespace sbn::v3::radix
