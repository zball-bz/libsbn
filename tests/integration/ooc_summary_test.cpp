// SPDX-License-Identifier: LGPL-3.0-or-later
#include "product_support.hpp"
#include "planning/subtree.hpp"
#include <array>
#include <limits>
using namespace sbn::v3;
using namespace sbn::v3::planning;
using namespace sbn::v3::series;

static SlotEnvelope shape(size_t value, size_t work) {
    SlotEnvelope e{};
    e[Slot::result0] = {value * 64, 64};
    e[Slot::workspace] = {work * 64, 64};
    return e;
}
static void resource_checks() {
    SlotEnvelope e{};
    e[Slot::result0] = {65, 64};
    e[Slot::metadata] = {17, 256};
    e[Slot::workspace] = {123, 128};
    SlotLayout l{};
    assert(layout(e, l) == SummaryStatus::ok);
    assert(l.offsets[0] == 0 && l.offsets[unsigned(Slot::metadata)] == 256);
    assert(l.offsets[unsigned(Slot::workspace)] == 384 && l.bytes == 512 && l.alignment == 256);
    size_t total = 0;
    assert(admit(e, {1536, 1024}, l, total) == SummaryStatus::ok && total == 1536);
    assert(admit(e, {1535, 1024}, l, total) == SummaryStatus::capacity && total == 1536);
    const auto old = l;
    e[Slot::workspace].alignment = 3;
    assert(layout(e, l) == SummaryStatus::invalid && l.bytes == old.bytes);
    e[Slot::workspace] = {std::numeric_limits<size_t>::max(), 1};
    assert(layout(e, l) == SummaryStatus::overflow && l.bytes == old.bytes);
    e = shape(1, 1);
    assert(admit(e, {SIZE_MAX, SIZE_MAX - 63}, l, total) == SummaryStatus::overflow);
    SlotEnvelope joined{};
    assert(join(shape(3, 2), shape(1, 7), joined) == SummaryStatus::ok);
    assert(contains(joined, shape(3, 7)) && !contains(joined, shape(4, 7)));
}
static void frontier_checks() {
    // Budget-dependent early realization is wrong: at 60 units it would take
    // compact and admit BOTH blocks, at 100 it would take ordinary and admit ONE.
    // Sizing first, independently of the budget, removes this regression.
    BlockSummary blocks[2]{};
    blocks[0] = {{0, 10}, {{true, shape(40, 20), 1}, {true, shape(40, 20), 2}}};
    blocks[1] = {{10, 20}, {{true, shape(20, 80), 3}, {true, shape(40, 20), 4}}};
    Decomposition decisions[2]{};
    ResidentFrontier f{};
    assert(select_resident_suffix(blocks, {60*64, 0}, decisions, f) == SummaryStatus::ok);
    assert(f.first == 0 && decisions[1] == Decomposition::compact);
    assert(select_resident_suffix(blocks, {100*64, 0}, decisions, f) == SummaryStatus::ok);
    assert(f.first == 0 && decisions[1] == Decomposition::compact);
    assert(select_resident_suffix(blocks, {120*64, 0}, decisions, f) == SummaryStatus::ok);
    assert(f.first == 0 && decisions[1] == Decomposition::ordinary && f.total_bytes == 120*64);
    size_t previous = 2;
    for (size_t bytes = 0; bytes <= 130*64; bytes += 13) {
        assert(select_resident_suffix(blocks, {bytes, 0}, decisions, f) == SummaryStatus::ok);
        assert(f.first <= previous && f.total_bytes <= bytes); previous = f.first;
        for (size_t i = 0; i < 2; ++i) {
            if (i < f.first) assert(decisions[i] == Decomposition::external);
            else assert(contains(f.envelope, blocks[i].candidate[unsigned(decisions[i])].slots));
        }
    }
    assert(select_resident_suffix(blocks, {0, 1}, decisions, f) == SummaryStatus::capacity);
    assert(select_resident_suffix(blocks, {8192, 0}, std::span(decisions, 1), f) == SummaryStatus::capacity);
    blocks[1].range.begin = 11;
    assert(select_resident_suffix(blocks, {8192, 0}, decisions, f) == SummaryStatus::invalid);
    blocks[1].range.begin = 10;
    blocks[1].candidate[0].supported = blocks[1].candidate[1].supported = false;
    assert(select_resident_suffix(blocks, {8192, 0}, decisions, f) == SummaryStatus::ok && f.first == 2);
}
static constexpr uint64_t prime = (uint64_t(1) << 61) - 1;
static uint64_t mulmod(uint64_t a, uint64_t b) { return __uint128_t(a) * b % prime; }
static uint64_t mod(const sbn3_series_value &v) {
    uint64_t r = 0;
    for (size_t i = v.mantissa.size; i-- > 0;) r = ((__uint128_t(r) << 64) + v.mantissa.data[i]) % prime;
    return v.mantissa.negative && r ? prime - r : r;
}
static void verify(const Formula &f, sbn3_series_range range, const sbn3_series_values &v) {
    uint64_t t = 0, d = 1, u = 1;
    for (uint64_t k = range.begin; k < range.end; ++k) {
        uint64_t lt = 1, ld = k, lu = 1;
        if (f.kind == FormulaKind::Chudnovsky) {
            if (!k) { lt = 13591409; ld = 1; }
            else {
                lu = mulmod(mulmod(6*k-5, 2*k-1), 6*k-1);
                ld = mulmod(mulmod(mulmod(k, k), k), 10939058860032000ULL);
                lt = mulmod(lu, 545140134*k + 13591409);
                if (k & 1) lt = lt ? prime-lt : 0;
            }
        }
        t = mulmod(t, ld);
        if (f.kind == FormulaKind::BinaryLog) {
            if (k > range.begin) t = mulmod(t, uint64_t(1) << f.radix_bits);
            lt = mulmod(lt, d);
        } else if (f.kind == FormulaKind::Chudnovsky) lt = mulmod(lt, u);
        t = (t + lt) % prime; d = mulmod(d, ld); u = mulmod(u, lu);
    }
    assert(mod(v.value[0]) == t && mod(v.value[1]) == d);
    if (f.kind == FormulaKind::Chudnovsky) assert(mod(v.value[2]) == u);
    assert(v.value[0].exponent2 == (f.kind == FormulaKind::BinaryLog ? -int64_t(f.radix_bits*(range.end-1)) : 0));
}
static void native_replay(FormulaKind kind, unsigned workers) {
    Formula formula{kind, 3};
    const auto finite = finite_formula(formula);
    const unsigned need = kind == FormulaKind::Chudnovsky ? 7 : 3;
    const uint64_t start = kind == FormulaKind::Chudnovsky ? 0 : 1;
    const sbn3_series_range ranges[]{{start, start+17}, {start+17, start+131}, {start+131, start+517}};
    // Deliberately filtered old options: the new total-region adapter owns admission.
    sbn3_series_options options{workers, 8, 4, 0, 1};
    NativeSubtree native[3]{};
    BlockSummary blocks[3]{};
    SlotEnvelope full{};
    for (unsigned i = 0; i < 3; ++i) {
        allocation_watch_start();
        assert(summarize_native_subtree(finite, ranges[i], need, options, native[i]) == SummaryStatus::ok);
        assert(allocation_watch_stop() == 0);
        blocks[i] = native[i].summary;
        for (unsigned d = 0; d < 2; ++d) {
            assert(blocks[i].candidate[d].supported && native[i].plans[d].info.serial_prefix == d);
            assert(join(full, blocks[i].candidate[d].slots, full) == SummaryStatus::ok);
        }
        auto ordinary_options = options; ordinary_options.workspace_budget = 0; ordinary_options.max_serial_prefix = 0;
        FinitePlan prior{};
        assert(finite_query(finite, ranges[i], need, ordinary_options, prior) == SBN3_SUPPORTED);
        assert(prior.info.schedule_id == native[i].plans[0].info.schedule_id);
    }
    SlotLayout physical{};
    assert(layout(full, physical) == SummaryStatus::ok);
    assert(physical.bytes < (size_t(8) << 20));
    Decomposition decisions[3]{};
    ResidentFrontier frontier{};
    constexpr size_t fixed_bound = size_t(32) << 20; // conservative test context/stack/shared-data grant
    assert(select_resident_suffix(blocks, {fixed_bound+physical.bytes, fixed_bound}, decisions, frontier) == SummaryStatus::ok);
    assert(frontier.first == 0);
    Fixture fixture(workers, false, size_t(64) << 20);
    const auto region = fixture.allocate(physical.bytes, physical.alignment);
    sbn3_lease roles[slot_count]{};
    for (unsigned i = 0; i < slot_count; ++i) if (full.slots[i].bytes)
        sbn3_arena_acquire(fixture.arena, uintptr_t(region.data)-fixture.base+physical.offsets[i],
                          full.slots[i].bytes, &roles[i]);
    for (unsigned i = 0; i < 3; ++i) {
        std::array<std::vector<uint64_t>, 3> previous;
        std::array<unsigned, 3> signs{};
        for (unsigned d = 0; d < 2; ++d) {
            const auto &plan = native[i].plans[d];
            sbn3_series_values output{};
            for (unsigned j = 0; j < 3; ++j) if (need & (1u << j))
                output.value[j].mantissa = {static_cast<uint64_t *>(roles[j].data), roles[j].bytes/8, 0, 0};
            FiniteBinding binding{};
            allocation_watch_start();
            finite_prepare(plan, *fixture.arena, *fixture.team, roles[unsigned(Slot::metadata)],
                           roles[unsigned(Slot::prepared)], roles[unsigned(Slot::workspace)], binding);
            finite_execute(binding, output);
            assert(allocation_watch_stop() == 0);
            verify(formula, ranges[i], output);
            for (unsigned j = 0; j < 3; ++j) if (need & (1u << j)) {
                const auto &v = output.value[j].mantissa;
                if (!d) { previous[j].assign(v.data, v.data+v.size); signs[j] = v.negative; }
                else assert(previous[j].size() == v.size && signs[j] == v.negative &&
                            !memcmp(previous[j].data(), v.data, v.size*8));
            }
        }
    }
    for (auto &l : roles) if (l.token) sbn3_arena_release(fixture.arena, &l);
    printf("OOC native summary/replay: family=%u workers=%u region=%zu bytes, 3 ranges x 2 decompositions PASS\n",
           unsigned(kind), workers, physical.bytes);
}
int main() {
    resource_checks(); frontier_checks();
    for (auto kind : {FormulaKind::Euler, FormulaKind::Chudnovsky, FormulaKind::BinaryLog})
        for (unsigned workers : {1u, 4u}) native_replay(kind, workers);
    puts("OOC summaries: checked admission, monotone resident frontier, native replay and allocation gates PASS");
}
