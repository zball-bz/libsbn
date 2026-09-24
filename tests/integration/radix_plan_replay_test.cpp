// Plan replay gate of the radix services.
//
// The tree plans do not fit a plan value, so bind assembles them again. The public query records the winner of
// every product search (PlanTranscript) and bind repeats the winner's one backend query. This test checks the
// statements that make that safe:
//   1. a chosen product query returns the plan of the search, bit for bit (plain products and the rail's squares);
//   2. a recorded assembly replays into the same tree plan without a single search, for every node class, stage,
//      rail squaring and ring plan; a transcript without room, a transcript of another request and damaged
//      entries fall back to the search and still give the same plan;
//   3. the plan value is sealed: a changed transcript word, Newton plan word or layout word is refused at bind;
//   4. the schoolbook reciprocal of small scaling powers is inside the Newton INVERSE contract.
#include "product_support.hpp"
#include "radix/format_tree.hpp"
#include "radix/parse_tree.hpp"
#include "sbn3/radix.h"
#include <initializer_list>
#include <memory>
using namespace sbn::v3;
using namespace sbn::v3::radix;
namespace {
unsigned searches_replayed = 0, fallbacks_forced = 0;
bool same_shape(const ProductShape &a, const ProductShape &b) {
    return a.an == b.an && a.bn == b.bn && a.workers == b.workers && a.requested == b.requested && a.choice == b.choice &&
           a.prepared_bytes == b.prepared_bytes && a.local_bytes == b.local_bytes && a.tables == b.tables &&
           a.work_bytes == b.work_bytes && a.work_alignment == b.work_alignment && a.output_limbs == b.output_limbs &&
           a.arithmetic_id == b.arithmetic_id;
}
// 1. chosen == searched, as complete plan values. A plan value carries bytes no query determines: padding of the
// backends' records, copied from whatever the stack held. They are found by running the winner's query over a
// stack filled with zeros and again over one filled with ones: every bit that follows the stack is padding,
// every other bit must be the bit of the public search.
[[gnu::noinline]] void dirty_stack(unsigned char byte) {
    volatile unsigned char pad[size_t(1) << 17];
    for (size_t j = 0; j < sizeof pad; ++j)
        pad[j] = byte;
}
bool same_info(const sbn3_mul_info &a, const sbn3_mul_info &b) {
    return a.np == b.np && a.trunk_bits == b.trunk_bits && a.workers == b.workers && a.algorithm == b.algorithm && a.C == b.C &&
           a.M2 == b.M2 && a.full == b.full && a.rowscale == b.rowscale && a.output_limbs == b.output_limbs &&
           a.table_bytes == b.table_bytes && a.workspace_bytes == b.workspace_bytes && a.workspace_alignment == b.workspace_alignment &&
           a.transpose_bytes == b.transpose_bytes && a.borrow_output == b.borrow_output && a.prime_batch == b.prime_batch &&
           a.codec_mode == b.codec_mode && a.crt_mode == b.crt_mode && a.basis_id == b.basis_id && a.arithmetic_id == b.arithmetic_id &&
           a.execution_id == b.execution_id;
}
uint64_t padding_bits = 0, compared_bits = 0;
// AddressSanitizer keeps locals on a fake stack of its own (detect_stack_use_after_return), which dirty_stack
// cannot reach; this gate turns that one option off so that the calibration works there too (every other ASan
// check stays on).
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
extern "C" const char *__asan_default_options() { return "detect_stack_use_after_return=0"; }
#endif
#endif
void same_plan(const sbn3_mul_plan &searched, const sbn3_mul_plan &zeros, const sbn3_mul_plan &ones) {
    for (unsigned j = 0; j < 384; ++j) {
        const uint64_t padding = zeros.opaque[j] ^ ones.opaque[j];
        assert(!((searched.opaque[j] ^ zeros.opaque[j]) & ~padding));
        padding_bits += uint64_t(__builtin_popcountll(padding));
        compared_bits += uint64_t(64 - __builtin_popcountll(padding));
    }
}
void products() {
    unsigned count = 0;
    for (unsigned workers : {1u, 2u, 8u, 16u})
        for (size_t an : {9ul, 25ul, 97ul, 300ul, 1000ul, 1025ul, 4097ul, 20000ul, 33000ul, 70001ul, 250000ul, 1100000ul})
            for (size_t bn : {an, an / 2 + 1, an / 7 + 3}) {
                sbn3_mul_options o{};
                o.workers = workers;
                auto plain = std::make_unique<ProductProgramPlan>(), searched = std::make_unique<ProductProgramPlan>(),
                     zeros = std::make_unique<ProductProgramPlan>(), ones = std::make_unique<ProductProgramPlan>();
                ProductChoice choice = 0;
                const auto rc = product_program_query(an, bn, o, *plain);
                assert(rc == product_program_choose(an, bn, o, *searched, choice));
                if (rc != SBN3_SUPPORTED)
                    continue;
                assert(choice && !(choice >> product_choice_bits));
                // the winner's one backend query, over two stacks
                dirty_stack(0);
                assert(product_program_chosen(an, bn, o, choice, *zeros) == SBN3_SUPPORTED);
                dirty_stack(0xff);
                assert(product_program_chosen(an, bn, o, choice, *ones) == SBN3_SUPPORTED);
                same_plan(plain->plan, zeros->plan, ones->plan);    // the public search
                same_plan(searched->plan, zeros->plan, ones->plan); // the recording search
                assert(same_info(plain->info, zeros->info) && same_info(searched->info, zeros->info) &&
                       plain->prepared_bytes == zeros->prepared_bytes && plain->contract == zeros->contract &&
                       plain->pair_workspace_bytes == zeros->pair_workspace_bytes &&
                       product_program_tables(*plain) == product_program_tables(*zeros));
                // not a choice: refused, never a different plan
                assert(product_program_chosen(an, bn, o, 0, *zeros) == SBN3_UNSUPPORTED);
                assert(product_program_chosen(an, bn, o, choice | uint64_t(1) << product_choice_bits, *zeros) == SBN3_UNSUPPORTED);
                ++count;
            }
    for (unsigned workers : {1u, 8u, 16u})
        for (size_t n : {4096ul, 4700ul, 19000ul, 76000ul, 304000ul}) {
            sbn3_mul_options o{};
            o.workers = workers;
            sbn3_product_request r{};
            r.kind = SBN3_PRODUCT_SQR;
            r.a_limbs = n;
            auto plain = std::make_unique<sbn3_mul_plan>(), searched = std::make_unique<sbn3_mul_plan>(), zeros = std::make_unique<sbn3_mul_plan>(),
                 ones = std::make_unique<sbn3_mul_plan>();
            sbn3_product_info ip{}, is{}, iz{}, io{};
            ProductChoice choice = 0;
            const auto rc = sbn3_product_query(&r, &o, plain.get(), &ip);
            assert(rc == square_query_choose(n, o, *searched, is, choice));
            if (rc != SBN3_SUPPORTED)
                continue;
            dirty_stack(0);
            assert(choice && square_query_chosen(n, o, choice, *zeros, iz) == SBN3_SUPPORTED);
            dirty_stack(0xff);
            assert(square_query_chosen(n, o, choice, *ones, io) == SBN3_SUPPORTED);
            same_plan(*plain, *zeros, *ones);
            same_plan(*searched, *zeros, *ones);
            assert(same_info(ip.mul, iz.mul) && same_info(is.mul, iz.mul) && ip.kind == iz.kind && ip.cyclic_limbs == iz.cyclic_limbs);
            ++count;
        }
    // almost every bit is compared: the padding is a few words of a 384-word value
    assert(count > 150 && padding_bits * 20 < compared_bits);
}
// 2. format and parse tree plans.
bool same_split(const SplitPlan &a, const SplitPlan &b) {
    return a.middle_words==b.middle_words && a.middle_bytes==b.middle_bytes && a.middle_shift==b.middle_shift &&
           same_shape(a.product, b.product) && a.cyclic.enabled == b.cyclic.enabled && a.cyclic.ring == b.cyclic.ring &&
           a.ring.enabled == b.ring.enabled && a.ring.np == b.ring.np && a.ring.algorithm == b.ring.algorithm &&
           a.ring.trunk_bits == b.ring.trunk_bits && a.ring.ring == b.ring.ring && a.ring.table_bytes == b.ring.table_bytes &&
           a.ring.work_bytes == b.ring.work_bytes && a.ring.spectrum_bytes == b.ring.spectrum_bytes &&
           a.ring.output_limbs == b.ring.output_limbs && a.ring.predicted_ns == b.ring.predicted_ns && a.gap_limbs == b.gap_limbs &&
           a.low_limbs == b.low_limbs && a.low_bytes == b.low_bytes && a.episode_bytes() == b.episode_bytes();
}
bool same_rail(const RailPlan &a, const RailPlan &b) {
    bool same = a.count == b.count && a.fixed_levels == b.fixed_levels && a.total_limbs == b.total_limbs && a.setup_bytes == b.setup_bytes;
    for (unsigned k = 0; same && k < a.count; ++k)
        same = a.limbs[k] == b.limbs[k] && a.offset[k] == b.offset[k] && a.service[k] == b.service[k] &&
               a.square_choice[k] == b.square_choice[k];
    return same;
}
void same_format(const FormatTreePlan &a, const FormatTreePlan &b) {
    assert(a.class_count == b.class_count && a.tree_count == b.tree_count && a.extra_count == b.extra_count &&
           a.prepared_bytes() == b.prepared_bytes() && a.pool_bytes() == b.pool_bytes() && a.ring_stages == b.ring_stages &&
           a.frontier_region_bytes == b.frontier_region_bytes && same_rail(a.rail, b.rail));
    assert(a.group_u52==b.group_u52);
    for(unsigned n=0;n<=group_fragments;++n)assert(a.group_work[n]==b.group_work[n]);
    for (unsigned j = 0; j < a.extra_count; ++j)
        assert(same_shape(a.extra[j], b.extra[j]));
    for (unsigned j = 0; j < a.class_count; ++j)
        assert(a.classes[j].fragments == b.classes[j].fragments && a.classes[j].region_bytes == b.classes[j].region_bytes &&
               a.classes[j].persist_bytes == b.classes[j].persist_bytes && same_split(a.classes[j].split, b.classes[j].split));
    for (unsigned t = 0; t < a.tree_count; ++t) {
        assert(a.trees[t].stage_count == b.trees[t].stage_count && a.work_bytes(int(t)) == b.work_bytes(int(t)));
        for (unsigned s = 0; s < a.trees[t].stage_count; ++s)
            assert(a.trees[t].stages[s].groups == b.trees[t].stages[s].groups && a.trees[t].stages[s].workers == b.trees[t].stages[s].workers &&
                   same_split(a.trees[t].stages[s].split, b.trees[t].stages[s].split));
    }
}
sbn3_query_result assemble(FormatTreePlan &plan, unsigned base, unsigned workers, uint64_t fragments, uint64_t second,
                           PlanTranscript *transcript) {
    auto rc = format_tree_begin(base, workers, std::max(fragments, second), plan);
    if (rc != SBN3_SUPPORTED)
        return rc;
    plan.transcript = transcript;
    const int first = plan.add_tree(fragments);
    if (second)
        plan.add_tree(second);
    if (first >= 0)
        plan.add_product(plan.root_limbs(first), plan.root_limbs(first) + 1);
    rc = plan.finish();
    plan.transcript = nullptr;
    return rc;
}
sbn3_query_result assemble(ParseTreePlan &plan, unsigned base, unsigned workers, uint64_t fragments, PlanTranscript *transcript) {
    auto rc = parse_tree_begin(base, workers, fragments, plan);
    if (rc != SBN3_SUPPORTED)
        return rc;
    plan.transcript = transcript;
    const int root = plan.add_tree(fragments);
    if (root >= 0)
        plan.add_product(plan.classes[root].limbs, plan.classes[root].limbs + 5);
    rc = plan.finish();
    plan.transcript = nullptr;
    return rc;
}
void trees() {
    auto searched = std::make_unique<FormatTreePlan>(), recorded = std::make_unique<FormatTreePlan>(), replayed = std::make_unique<FormatTreePlan>();
    auto transcript = std::make_unique<PlanTranscript>();
    unsigned plans = 0;
    for (unsigned base : {10u, 3u, 12u, 63u})
        for (unsigned workers : {1u, 5u, 16u})
            for (uint64_t fragments : {9ul, 40ul, 129ul, 1233ul, 4099ul, 19731ul, 78913ul, 315653ul, 1262611ul}) {
                if (base != 10 && fragments > 80000)
                    continue;
                const uint64_t second = fragments % 3 == 0 ? fragments / 2 + 1 : 0;
                // The planner must construct every live field itself; unused
                // fixed-capacity slots may contain arbitrary prior bytes.
                memset(static_cast<void *>(recorded.get()),0xa5,sizeof(*recorded));
                memset(static_cast<void *>(searched.get()),0x5a,sizeof(*searched));
                memset(static_cast<void *>(replayed.get()),0xcc,sizeof(*replayed));
                *transcript = {};
                const auto rc = assemble(*recorded, base, workers, fragments, second, transcript.get());
                assert(rc == assemble(*searched, base, workers, fragments, second, nullptr));
                if (rc != SBN3_SUPPORTED)
                    continue;
                same_format(*searched, *recorded); // recording changes nothing
                assert(transcript->count && transcript->count < PlanTranscript::capacity);
                // replay: every entry consumed, every product the recorded winner
                auto replay = std::make_unique<PlanTranscript>(*transcript);
                replay->replay = true;
                assert(assemble(*replayed, base, workers, fragments, second, replay.get()) == SBN3_SUPPORTED);
                same_format(*searched, *replayed);
                assert(replay->cursor == replay->count && !replay->searched);
                searches_replayed += replay->count;
                // no room: a prefix replays, the rest is searched
                replay = std::make_unique<PlanTranscript>(*transcript);
                replay->replay = true;
                replay->count = replay->count / 2;
                assert(assemble(*replayed, base, workers, fragments, second, replay.get()) == SBN3_SUPPORTED);
                same_format(*searched, *replayed);
                assert(replay->searched == transcript->count - replay->count);
                // damaged entries: another legal candidate, a check that does not match, no choice at all
                replay = std::make_unique<PlanTranscript>(*transcript);
                replay->replay = true;
                unsigned damaged = 0;
                for (uint32_t j = 0; j < replay->count; ++j) {
                    const unsigned kind = unsigned(random_word() % 4);
                    damaged += kind == 0 || kind == 2;
                    if (kind == 0)
                        replay->entry[j] ^= uint64_t(1) << (product_choice_bits + random_word() % 20); // the check
                    else if (kind == 1)
                        replay->entry[j] = (replay->entry[j] & ~uint64_t(0xfe)) | uint64_t(4 + random_word() % 5) << 1; // another backend entry
                    else if (kind == 2)
                        replay->entry[j] &= ~uint64_t(1);
                    fallbacks_forced += kind != 3;
                }
                assert(assemble(*replayed, base, workers, fragments, second, replay.get()) == SBN3_SUPPORTED);
                same_format(*searched, *replayed);
                assert(replay->searched >= damaged); // (another backend entry may be refused or give another identity)
                // the transcript of another request
                auto other = std::make_unique<PlanTranscript>();
                assert(assemble(*recorded, base, workers, fragments + 7, 0, other.get()) == SBN3_SUPPORTED);
                other->replay = true;
                assert(assemble(*replayed, base, workers, fragments, second, other.get()) == SBN3_SUPPORTED);
                same_format(*searched, *replayed);
                ++plans;
            }
    auto ps = std::make_unique<ParseTreePlan>(), pr = std::make_unique<ParseTreePlan>();
    for (unsigned base : {10u, 7u, 36u})
        for (unsigned workers : {1u, 16u})
            for (uint64_t fragments : {9ul, 129ul, 1233ul, 19731ul, 315653ul}) {
                if (base != 10 && fragments > 80000)
                    continue;
                memset(static_cast<void *>(ps.get()),0xa5,sizeof(*ps));
                memset(static_cast<void *>(pr.get()),0x5a,sizeof(*pr));
                *transcript = {};
                const auto rc = assemble(*ps, base, workers, fragments, transcript.get());
                if (rc != SBN3_SUPPORTED)
                    continue;
                auto replay = std::make_unique<PlanTranscript>(*transcript);
                replay->replay = true;
                assert(assemble(*pr, base, workers, fragments, replay.get()) == SBN3_SUPPORTED && replay->cursor == replay->count &&
                       !replay->searched);
                assert(ps->class_count == pr->class_count && ps->prepared_bytes() == pr->prepared_bytes() && same_rail(ps->rail, pr->rail) &&
                       ps->work_bytes(int(ps->class_count) - 1) == pr->work_bytes(int(pr->class_count) - 1));
                for (unsigned j = 0; j < ps->class_count; ++j)
                    assert(same_shape(ps->classes[j].product, pr->classes[j].product) && ps->classes[j].cached.enabled == pr->classes[j].cached.enabled);
                for (unsigned j = 0; j < ps->extra_count; ++j)
                    assert(same_shape(ps->extra[j], pr->extra[j]));
                searches_replayed += replay->count;
                ++plans;
            }
    assert(plans > 60 && searches_replayed > 1000 && fallbacks_forced > 500);
}
// 3. sealed plan values: a changed word of the layout, of the Newton plan or of the transcript is refused by bind
// (fatal before anything is leased). One forked child per word; one worker, so the child owns every thread it needs.
unsigned refusals = 0;
template <class Bind> unsigned refused_words(const uint64_t *value, Bind bind, unsigned &longest_run) {
    unsigned refused = 0, run = 0;
    longest_run = 0;
    for (unsigned word = 0; word < 512; ++word) {
        uint64_t damaged[512];
        memcpy(damaged, value, sizeof damaged);
        damaged[word] ^= 1; // the lowest bit of a word always belongs to a field
        const auto child = fork();
        assert(child >= 0);
        if (!child) {
            rlimit no_core{0, 0};
            setrlimit(RLIMIT_CORE, &no_core);
            close(STDERR_FILENO);
            alarm(20);
            bind(damaged);
            _exit(0);
        }
        int status = 0;
        assert(waitpid(child, &status, 0) == child);
        // refused (the fatal diagnostic aborts), or a word that is not part of the plan: bound and unbound normally
        assert((WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) || (WIFEXITED(status) && !WEXITSTATUS(status)));
        const bool fatal = WIFSIGNALED(status);
        run = fatal ? run + 1 : 0;
        longest_run = std::max(longest_run, run);
        refused += fatal;
    }
    return refused;
}
void sealed_values() {
    Fixture f(1, false, size_t(512) << 20);
    sbn3_radix_options o{};
    o.workers = 1;
    // integer format and fraction parse above the schoolbook reciprocal: layout + Newton plan + transcript
    sbn3_format_spec fs{};
    fs.base = 10;
    fs.limbs = 4097;
    fs.exponent2 = 0;
    fs.mode = SBN3_RADIX_EXACT;
    sbn3_format_plan fplan{};
    sbn3_format_info finfo{};
    assert(sbn3_format_query(&fs, &o, &fplan, &finfo) == SBN3_SUPPORTED && finfo.divide_bytes);
    sbn3_parse_spec ps{};
    ps.base = 10;
    ps.fraction_digits = 80000;
    ps.fraction_bits = 64 * 4200;
    sbn3_parse_plan pplan{};
    sbn3_parse_info pinfo{};
    assert(sbn3_parse_query(&ps, &o, &pplan, &pinfo) == SBN3_SUPPORTED && pinfo.divide_bytes);
    static_assert(sizeof fplan == 4096 && sizeof pplan == 4096);
    // an unleased prepared range for the bindings
    const size_t bytes = std::max(finfo.storage_bytes, pinfo.storage_bytes), alignment = std::max(finfo.storage_alignment, pinfo.storage_alignment);
    auto block = f.allocate(bytes, alignment);
    const size_t at = size_t(reinterpret_cast<uintptr_t>(block.data) - reinterpret_cast<uintptr_t>(f.arena->base));
    sbn3_arena_release(f.arena, &f.leases.back());
    f.leases.pop_back();
    // the untouched values bind (twice: a plan value is reusable) and unbind
    for (unsigned round = 0; round < 2; ++round) {
        sbn3_format_binding *fb = nullptr;
        sbn3_format_bind(&fplan, f.arena, at, f.team, &fb);
        sbn3_format_unbind(fb);
        sbn3_parse_binding *pb = nullptr;
        sbn3_parse_bind(&pplan, f.arena, at, f.team, &pb);
        sbn3_parse_unbind(pb);
    }
    unsigned run = 0;
    unsigned refused = refused_words(fplan.opaque, [&](const uint64_t *value) {
        sbn3_format_plan damaged;
        memcpy(damaged.opaque, value, sizeof damaged);
        sbn3_format_binding *fb = nullptr;
        sbn3_format_bind(&damaged, f.arena, at, f.team, &fb);
        sbn3_format_unbind(fb);
    }, run);
    // the whole Newton plan (256 words) and the recorded searches behind it are one run of sealed words
    assert(run >= 256 + 10 && refused >= run + 10);
    refusals += refused;
    refused = refused_words(pplan.opaque, [&](const uint64_t *value) {
        sbn3_parse_plan damaged;
        memcpy(damaged.opaque, value, sizeof damaged);
        sbn3_parse_binding *pb = nullptr;
        sbn3_parse_bind(&damaged, f.arena, at, f.team, &pb);
        sbn3_parse_unbind(pb);
    }, run);
    assert(run >= 256 + 10 && refused >= run + 10);
    refusals += refused;
}
// 4. schoolbook reciprocal: U = floor((B^(2n) - 1) / D), so U D <= B^(2n) - 1 < (U + 1) D.
void reciprocals() {
    Fixture f(1, false);
    auto lease = f.allocate(size_t(4) << 20, 64);
    Frame frame = Frame::borrow(*static_cast<Arena *>(f.arena), lease, lease.data, lease.bytes);
    for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(17), size_t(64), size_t(255), reciprocal_basecase_limbs})
        for (unsigned kind = 0; kind < 4; ++kind) {
            FrameMark mark(frame);
            auto *d = frame.alloc<uint64_t>(n), *u = frame.alloc<uint64_t>(n + 1), *z = frame.alloc<uint64_t>(2 * n + 2);
            for (size_t j = 0; j < n; ++j)
                d[j] = kind == 0 ? random_word() : kind == 1 ? ~uint64_t(0) : kind == 2 ? 0 : random_word() & random_word() & random_word();
            d[n - 1] |= uint64_t(1) << 63;
            if (kind == 2)
                d[0] |= 1; // 2^(64n-1) + 1: the largest reciprocal
            assert(frame.capacity() - frame.used() >= reciprocal_basecase_bytes(n));
            reciprocal_basecase(d, n, u, frame);
            assert(u[n] == 1);
            // z = U D <= B^(2n) - 1 and r = B^(2n) - 1 - z (the complement of z) < D
            sbn3_mul_basecase(z, 2 * n + 1, u, n + 1, d, n);
            assert(!z[2 * n]);
            bool below = false;
            for (size_t j = 2 * n; j-- > 0 && !below;) {
                const uint64_t r = ~z[j], dj = j < n ? d[j] : 0;
                if (r != dj) {
                    assert(r < dj);
                    below = true;
                }
            }
            assert(below);
        }
}
} // namespace
int main() {
    products();
    trees();
    sealed_values();
    reciprocals();
    printf("radix plan replay: chosen product plans identical to searched ones, %u recorded searches replayed without a search, "
           "%u damaged entries planned by the search again, %u damaged plan values refused, schoolbook reciprocals exact PASS\n",
           searches_replayed, fallbacks_forced, refusals);
    return 0;
}
