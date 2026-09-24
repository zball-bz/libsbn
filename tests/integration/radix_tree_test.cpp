// Format tree gate: the division-free scaled remainder tree against exact integer arithmetic.
// For a node fraction y / 2^(64 limbs) the tree must return J or J - 1, J = floor(y b^(64 F) / 2^(64 limbs)),
// as a 64 F digit string (seams reconciled), for random and adversarial fractions.
#include "product_support.hpp"
#include "radix/format_tree.hpp"
#include "radix/power_bank.hpp"
#include "backend/u52/kernels.hpp"
#include <initializer_list>
#include <memory>
#include <deque>
using namespace sbn::v3;
using namespace sbn::v3::radix;
namespace {
struct Reference {
    ref_int value, power, scaled, t;
    Reference() { ref_inits(value, power, scaled, t, nullptr); }
    ~Reference() { ref_clears(value, power, scaled, t, nullptr); }
    // digits of floor(y * base^count / 2^(64 limbs)), MSD first
    std::vector<uint8_t> digits(const uint64_t *y, size_t limbs, unsigned base, uint64_t count) {
        ref_import(value, limbs, -1, 8, 0, 0, y);
        ref_set_ui(power, 1);
        ref_set_ui(t, base);
        // power = base^count by binary exponentiation
        ref_int acc;
        ref_init(acc);
        ref_set_ui(acc, base);
        for (uint64_t e = count; e; e >>= 1) {
            if (e & 1)
                ref_mul(power, power, acc);
            ref_mul(acc, acc, acc);
        }
        ref_clear(acc);
        ref_mul(scaled, value, power);
        ref_fdiv_q_2exp(scaled, scaled, 64 * limbs);
        std::vector<uint8_t> out(count);
        uint64_t b8 = 1;
        for (unsigned j = 0; j < 8; ++j)
            b8 *= base;
        ref_set_ui(t, b8);
        ref_int q;
        ref_init(q);
        for (uint64_t pos = count; pos;) {
            uint64_t w = ref_fdiv_ui(scaled, b8);
            ref_fdiv_q(q, scaled, t);
            ref_set(scaled, q);
            for (unsigned j = 0; j < 8 && pos; ++j) {
                out[--pos] = uint8_t(w % base);
                w /= base;
            }
        }
        assert(ref_sgn(scaled) == 0);
        ref_clear(q);
        return out;
    }
    // y = ceil(k 2^(64 limbs) / base^count) for the digit string k (MSD first): the smallest fraction whose
    // first `count` digits are k. The tree sees a value a hair above a digit boundary.
    void boundary(uint64_t *y, size_t limbs, unsigned base, const std::vector<uint8_t> &k) {
        ref_set_ui(value, 0);
        ref_set_ui(power, 1);
        for (uint8_t d : k) {
            ref_mul_ui(value, value, base);
            ref_add_ui(value, value, d);
            ref_mul_ui(power, power, base);
        }
        ref_mul_2exp(value, value, 64 * limbs);
        ref_add(value, value, power);
        ref_sub_ui(value, value, 1);
        ref_fdiv_q(scaled, value, power);
        memset(y, 0, limbs * 8);
        size_t count = 0;
        if (ref_sgn(scaled))
            ref_export(y, &count, -1, 8, 0, 0, scaled);
        assert(count <= limbs);
    }
};
struct Case {
    unsigned base;
    uint64_t fragments;
};
uint64_t checked = 0, equal = 0, below = 0;
uint64_t forced_repairs = 0;
// One prepared block per fixture; every run leases aligned pieces of it and releases them again.
struct Pool {
    Fixture &f;
    size_t offset = 0, bytes = 0, cursor = 0;
    std::deque<sbn3_lease> live; // stable references: the tree re-acquires its work lease in place
    Pool(Fixture &fixture, size_t size) : f(fixture), bytes(size) {
        auto block = f.allocate(size, size_t(1) << 21);
        offset = size_t(reinterpret_cast<uintptr_t>(block.data) - reinterpret_cast<uintptr_t>(f.arena->base));
        sbn3_arena_release(f.arena, &f.leases.back());
        f.leases.pop_back();
    }
    sbn3_lease &take(size_t size, size_t alignment) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(f.arena->base) + offset;
        cursor = up(base + cursor, alignment) - base;
        assert(cursor + size <= bytes);
        sbn3_lease l{};
        sbn3_arena_acquire(f.arena, offset + cursor, size, &l);
        cursor += size;
        live.push_back(l);
        return live.back();
    }
    void reset() {
        for (auto &l : live)
            sbn3_arena_release(f.arena, &l);
        live.clear();
        cursor = 0;
    }
};
void force_middle_carry(Pool &pool,const FormatTree &tree,int root,std::vector<uint64_t> &y) {
    const auto &c=tree.plan->classes[tree.plan->trees[root].root];
    const auto &s=c.split;
    assert(s.middle_words && !s.product.choice && s.product.work_bytes);
    const size_t na=(64*s.product.bn+51)/52;
    const uint64_t origin=52*(na-5),window=64*c.window_limbs,bits=window+64;
    ref_int a,x,t,two,expected,observed;ref_inits(a,x,t,two,expected,observed,nullptr);
    ref_import(a,s.product.bn,-1,8,0,0,tree.rail[c.level]);
    assert(ref_get_ui(a)&1);
    // Odd rail inverse modulo 2^bits, using independent exact reference
    // arithmetic. Set product residue to 2^window + (2^origin - 1): the
    // omitted low carry must cross the guard and increment the window.
    ref_set_ui(x,1);ref_set_ui(two,2);
    for(uint64_t known=1;known<bits;){
        known=std::min(2*known,bits);
        ref_mul(t,a,x);ref_sub(t,two,t);ref_mul(x,x,t);ref_fdiv_r_2exp(x,x,known);
    }
    ref_set_ui(t,1);ref_mul_2exp(t,t,origin);ref_sub_ui(t,t,1);
    ref_set_ui(expected,1);ref_mul_2exp(expected,expected,window);ref_add(t,t,expected);
    ref_mul(x,x,t);ref_fdiv_r_2exp(x,x,bits);
    std::fill(y.begin(),y.end(),0);size_t written=0;ref_export(y.data(),&written,-1,8,0,0,x);
    assert(written<=s.product.an);
    std::vector<uint64_t> middle(s.middle_words),got(c.right_limbs);
    auto &lease=pool.take(s.middle_bytes,64);Frame scratch(*pool.f.arena,lease);
    allocation_watch_start();
    {ComputeLease computing(*pool.f.arena);u52::middle_guard(middle.data(),tree.rail[c.level],s.product.bn,y.data(),s.product.an,scratch);}
    assert(!allocation_watch_stop() && middle[1]==UINT64_MAX);
    take_window(got.data(),got.size(),middle.data(),middle.size(),s.middle_shift);
    ref_mul(expected,a,x);ref_fdiv_q_2exp(expected,expected,window);
    ref_fdiv_r_2exp(expected,expected,64*c.right_limbs);
    ref_import(observed,got.size(),-1,8,0,0,got.data());
    ref_sub(t,expected,observed);assert(ref_cmp_ui(t,1)==0);
    ref_clears(a,x,t,two,expected,observed,nullptr);++forced_repairs;
}
void run(Pool &pool, unsigned base, uint64_t fragments, unsigned workers) {
    Fixture &f = pool.f;
    auto plan = std::make_unique<FormatTreePlan>();
    assert(format_tree_begin(base, workers, fragments, *plan) == SBN3_SUPPORTED);
    const int root = plan->add_tree(fragments);
    assert(root >= 0 && plan->finish() == SBN3_SUPPORTED);
    const size_t limbs = plan->root_limbs(root);
    const size_t work_bytes = std::max(plan->work_bytes(root), plan->rail.setup_bytes) + 64;
    auto &prepared = pool.take(plan->prepared_bytes() + 128, std::max<size_t>(128, plan->prepared_alignment()));
    auto &work = pool.take(work_bytes, 1u << 21);
    auto &rail = pool.take(plan->rail.total_limbs * 8 + 64, 64);
    std::vector<ProductProgram> programs(plan->program_slots());
    std::vector<RailProduct> cyclic(plan->class_count);
    std::vector<RingStage> rings(plan->ring_stages + 1);
    // the ring pool: a prepared range nobody leases (the product service leases inside it)
    size_t ring_pool = 0;
    if (plan->pool_bytes()) {
        auto &reserve = pool.take(plan->pool_bytes(), size_t(1) << 21);
        ring_pool = size_t(reinterpret_cast<uintptr_t>(reserve.data) - reinterpret_cast<uintptr_t>(f.arena->base));
        sbn3_arena_release(f.arena, &pool.live.back());
        pool.live.pop_back();
    }
    FormatTree tree;
    allocation_watch_start();
    format_tree_bind(tree, *plan, nullptr, *f.arena, *f.team, prepared, work, static_cast<uint64_t *>(rail.data),
                     programs.data(), cyclic.data(), rings.data(), ring_pool);
    assert(allocation_watch_stop() == 0);
    std::vector<uint64_t> y(limbs), side(2 * fragments);
    std::vector<uint8_t> out(fragments * 64 + 64, 0xee);
    Reference ref;
    const unsigned ordinary_patterns = fragments > 300 ? 3 : 14;
    const unsigned patterns=ordinary_patterns+(plan->classes[plan->trees[root].root].split.middle_words!=0);
    for (unsigned pattern = 0; pattern < patterns; ++pattern) {
        for (auto &w : y)
            w = random_word();
        if(pattern==ordinary_patterns)force_middle_carry(pool,tree,root,y);
        else switch (pattern) {
        case 0: break;
        case 1: std::fill(y.begin(), y.end(), ~uint64_t(0)); break;
        case 2: std::fill(y.begin(), y.end(), 0); break;
        case 3: std::fill(y.begin(), y.end(), 0); y[0] = 1; break;
        case 4: std::fill(y.begin(), y.end(), 0); y[limbs - 1] = uint64_t(1) << 63; break; // 1/2
        case 5: { // just below 1/base^t: digits 0..0 (b-1)(b-1)...
            std::fill(y.begin(), y.end(), 0);
            y[limbs - 1] = ~uint64_t(0) / base;
            for (size_t j = 0; j + 1 < limbs; ++j) y[j] = y[limbs - 1];
            break;
        }
        case 6: for (size_t j = 0; j < limbs / 2; ++j) y[j] = 0; break;               // exact short fraction
        case 7: for (size_t j = limbs / 2; j < limbs; ++j) y[j] = ~uint64_t(0); break; // long run of b-1 digits
        case 8: for (size_t j = limbs / 3; j < limbs; ++j) y[j] = 0; break;           // long run of zero digits
        case 9: y[limbs - 1] = random_word() >> (random_word() % 64); break;
        default: { // digit strings ending in zeros from a fragment boundary on: every later fragment borrows
            std::vector<uint8_t> k(fragments * 64 + 8, 0);
            const uint64_t keep = pattern == 10 ? 1 : pattern == 11 ? fragments * 64 : 1 + random_word() % (fragments * 64);
            const uint64_t live = pattern == 13 ? keep : (keep + 63) / 64 * 64 - (pattern == 12 ? 0 : 0);
            for (uint64_t j = 0; j < std::min<uint64_t>(live, k.size()); ++j)
                k[j] = uint8_t(random_word() % base);
            if (live && !k[live - 1])
                k[live - 1] = 1;
            ref.boundary(y.data(), limbs, base, k);
            break;
        }
        }
        std::fill(out.begin(), out.end(), 0xee);
        allocation_watch_start();
        const uint64_t overlap = format_tree_run(tree, root, y.data(), out.data(), side.data(), side.data() + fragments);
        assert(allocation_watch_stop() == 0);
        for (size_t j = fragments * 64; j < out.size(); ++j)
            assert(out[j] == 0xee);
        auto expect = ref.digits(y.data(), limbs, base, fragments * 64 + 8);
        // expected overlap word and the 64 F digit prefix
        uint64_t expect_overlap = 0;
        for (unsigned j = 0; j < 8; ++j)
            expect_overlap = expect_overlap * base + expect[fragments * 64 + j];
        const bool same = !memcmp(out.data(), expect.data(), fragments * 64);
        if (same) {
            // The 64F+8 digit value is J or J - 1: equal prefix means the overlap word is equal or one below.
            assert(overlap == expect_overlap || overlap + 1 == expect_overlap);
            ++equal;
        } else {
            // prefix one below: only possible when the true overlap word is zero and ours is b^8 - 1
            DigitPlan dp{};
            assert(digit_plan_init(dp, base, nullptr));
            increment_digits(out.data(), fragments * 64, dp);
            assert(!memcmp(out.data(), expect.data(), fragments * 64));
            assert(expect_overlap == 0 && overlap == dp.b8 - 1);
            ++below;
        }
        ++checked;
    }
    pool.reset();
}
// The wrap-around paths (FFT kernels in the frontier, product service rings in the stages) must return the
// very digits of the exact product programs: sizes beyond the quadratic oracle are checked policy against policy.
std::vector<uint8_t> convert(Pool &pool, unsigned base, uint64_t fragments, unsigned workers, const std::vector<uint64_t> &y,
                             uint64_t &overlap) {
    Fixture &f = pool.f;
    auto plan = std::make_unique<FormatTreePlan>();
    assert(format_tree_begin(base, workers, fragments, *plan) == SBN3_SUPPORTED);
    const int root = plan->add_tree(fragments);
    assert(root >= 0 && plan->finish() == SBN3_SUPPORTED);
    assert(plan->root_limbs(root) == y.size());
    auto &prepared = pool.take(plan->prepared_bytes() + 128, std::max<size_t>(128, plan->prepared_alignment()));
    auto &work = pool.take(std::max(plan->work_bytes(root), plan->rail.setup_bytes) + 64, 1u << 21);
    auto &rail = pool.take(plan->rail.total_limbs * 8 + 64, 64);
    size_t ring_pool = 0;
    if (plan->pool_bytes()) {
        auto &reserve = pool.take(plan->pool_bytes(), size_t(1) << 21);
        ring_pool = size_t(reinterpret_cast<uintptr_t>(reserve.data) - reinterpret_cast<uintptr_t>(f.arena->base));
        sbn3_arena_release(f.arena, &pool.live.back());
        pool.live.pop_back();
    }
    std::vector<ProductProgram> programs(plan->program_slots());
    std::vector<RailProduct> cyclic(plan->class_count);
    std::vector<RingStage> rings(plan->ring_stages + 1);
    FormatTree tree;
    format_tree_bind(tree, *plan, nullptr, *f.arena, *f.team, prepared, work, static_cast<uint64_t *>(rail.data),
                     programs.data(), cyclic.data(), rings.data(), ring_pool);
    std::vector<uint64_t> side(2 * fragments);
    std::vector<uint8_t> out(fragments * 64);
    allocation_watch_start();
    overlap = format_tree_run(tree, root, y.data(), out.data(), side.data(), side.data() + fragments);
    assert(allocation_watch_stop() == 0);
    pool.reset();
    return out;
}
uint64_t cross_checked = 0, ring_stage_runs = 0;
void cross_check(Pool &pool, unsigned base, uint64_t fragments, unsigned workers) {
    BaseInfo info{};
    assert(base_info(base, info));
    const size_t limbs = node_limbs(info, fragments);
    auto &policy = tree_policy();
    const TreePolicy saved = policy;
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
        std::vector<uint64_t> y(limbs);
        for (auto &w : y)
            w = pattern == 0 ? random_word() : pattern == 1 ? ~uint64_t(0) : 0;
        if (pattern == 3)
            y[limbs - 1] = uint64_t(1) << 63;
        uint64_t reference_overlap = 0, overlap = 0;
        policy.cyclic_products = false;
        policy.ring_products = false;
        const auto reference = convert(pool, base, fragments, workers, y, reference_overlap);
        for (unsigned variant = 0; variant < 3; ++variant) {
            policy = saved;
            policy.cyclic_products = variant != 1;
            policy.ring_products = variant != 0;
            policy.ring_min_count = 2;
            auto plan = std::make_unique<FormatTreePlan>();
            assert(format_tree_begin(base, workers, fragments, *plan) == SBN3_SUPPORTED && plan->add_tree(fragments) == 0 &&
                   plan->finish() == SBN3_SUPPORTED);
            ring_stage_runs += plan->ring_stages;
            assert(variant == 0 || plan->ring_stages);
            const auto out = convert(pool, base, fragments, workers, y, overlap);
            assert(out == reference && overlap == reference_overlap);
            ++cross_checked;
        }
        policy = saved;
    }
}
} // namespace
int main(int argc,char **argv) {
    if(argc>1){
        assert(argc==2);
        if(!strcmp(argv[1],"--leaf-groups-only")){
            for(unsigned workers:{1u,4u}){
                Fixture f(workers,false);Pool pool(f,size_t(160)<<20);
                for(unsigned base=3;base<64;++base)if(base&(base-1))
                    for(uint64_t fragments=1;fragments<=8;++fragments)run(pool,base,fragments,workers);
            }
            printf("radix leaf roots: %llu conversions, all bases and widths PASS\n",(unsigned long long)checked);
            return 0;
        }
        assert(!strcmp(argv[1],"--middle-repair-only"));
        Fixture f(1,false);Pool pool(f,size_t(160)<<20);
        for(unsigned base:{3u,10u,63u})for(uint64_t fragments:{33ull,65ull,129ull})run(pool,base,fragments,1);
        assert(forced_repairs==9);
        printf("radix middle repair: %llu conversions, %llu constructed carries through full tree PASS\n",
               (unsigned long long)checked,(unsigned long long)forced_repairs);
        return 0;
    }
    {
        ref_int expected,actual;ref_inits(expected,actual,nullptr);
        unsigned constants=0;
        for(unsigned odd=3;odd<=63;odd+=2){
            ref_set_ui(expected,odd);
            for(unsigned j=0;j<6;++j)ref_mul(expected,expected,expected);
            BaseInfo base{};assert(base_info(odd,base));
            for(unsigned k=0;k<fixed_power_levels;++k){
                const auto p=fixed_power(odd,k);
                assert(p.data && !(uintptr_t(p.data)&63) && p.capacity>=rail_limbs(base,k)+7);
                ref_import(actual,p.capacity,-1,8,0,0,p.data);assert(ref_cmp(actual,expected)==0);
                ++constants;ref_mul(expected,expected,expected);
            }
            assert(!fixed_power(odd,fixed_power_levels).data);
        }
        assert(!fixed_power(2,0).data && !fixed_power(65,0).data);
        printf("radix fixed powers: %u exact constants, %zu shared bytes PASS\n",constants,fixed_power_bytes());
        ref_clears(expected,actual,nullptr);
    }
    rlimit stack{};
    getrlimit(RLIMIT_STACK, &stack);
    for (unsigned workers : {1u, 4u}) {
        Fixture f(workers, false);
        Pool pool(f, size_t(160) << 20);
        for (unsigned base : {3u, 5u, 6u, 7u, 10u, 12u, 36u, 62u, 63u})
            for (uint64_t fragments : {1ull, 2ull, 3ull, 5ull, 8ull, 9ull, 15ull, 16ull, 17ull, 24ull, 31ull, 33ull, 64ull, 100ull,
                                       257ull, 600ull})
                run(pool, base, fragments, workers);
        run(pool, 10, 5000, workers);
        run(pool, 10, 1 << 14, workers);
    }
    for (unsigned workers : {4u, 8u}) {
        Fixture f(workers, false);
        Pool pool(f, size_t(208) << 20);
        cross_check(pool, 10, uint64_t(1) << 17, workers);
        cross_check(pool, 7, (uint64_t(1) << 16) + 12345, workers);
    }
    printf("radix tree: %llu conversions (%llu exact prefix, %llu one below) against exact arithmetic, %llu wrap-around runs "
           "(%llu ring stages) identical to the exact programs PASS\n",
           (unsigned long long)checked, (unsigned long long)equal, (unsigned long long)below, (unsigned long long)cross_checked,
           (unsigned long long)ring_stage_runs);
    assert(forced_repairs);
    printf("radix middle repair: %llu constructed carries exercised the full tree path PASS\n",(unsigned long long)forced_repairs);
}
