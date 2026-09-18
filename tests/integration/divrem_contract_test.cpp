// Spectrum build contract of the exact-division service on shapes whose
// divisor-side products change representation between the policy producer
// (full transform under the lattice rule) and the geometry-pinned cached
// consumer (truncated support, different scale). Such shapes start at about
// 2^21 transform points, so this gate states its memory budget explicitly:
// every case is admitted by query against the budget before any page is
// prepared, and the arena's resident budget is that same number.
#include "divrem_support.hpp"
namespace {
constexpr size_t budget = size_t(1) << 30; // extended tier: tools/run_extended.py grants 1 GiB of locked memory
struct Case {
    size_t dn, nn;
    unsigned workers, prime_count;
    size_t block;
    size_t head; // expected word-division head
};
size_t mem_available() {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return SIZE_MAX;
    char line[256];
    size_t kib = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %zu kB", &kib) == 1)
            break;
    fclose(f);
    return kib ? kib << 10 : SIZE_MAX;
}
// Public-API fact: the policy producer of the linear product a x b and its
// cached consumer disagree on support or scale, i.e. the consumer binding
// could not build the producer-described spectrum.
bool representation_varies(size_t a, size_t b, unsigned workers, unsigned prime_count) {
    sbn3_product_request r{};
    r.a_limbs = a;
    r.b_limbs = b;
    sbn3_mul_options o{};
    o.workers = workers;
    o.prime_count = prime_count;
    o.borrow_output = 1;
    sbn3_mul_plan producer{}, consumer{};
    sbn3_product_info pi{}, ci{};
    sbn3_spectrum_desc future{}, own{};
    if (sbn3_product_query(&r, &o, &producer, &pi) != SBN3_SUPPORTED ||
        sbn3_spectrum_query(&producer, SBN3_SPECTRUM_COLUMNS, 1, &future) != SBN3_SUPPORTED)
        return false;
    r.cached_a[0] = &future;
    if (sbn3_product_query(&r, &o, &consumer, &ci) != SBN3_SUPPORTED ||
        sbn3_spectrum_query(&consumer, SBN3_SPECTRUM_COLUMNS, 1, &own) != SBN3_SUPPORTED)
        return false;
    return pi.mul.full != ci.mul.full || own.live_slots != future.live_slots ||
           memcmp(own.scale, future.scale, sizeof own.scale) != 0;
}
std::vector<uint64_t> random_limbs(size_t n) {
    std::vector<uint64_t> v(n);
    for (auto &w : v)
        w = random_word();
    return v;
}
bool run(const Case &c) {
    sbn3_divrem_request request{c.nn, c.dn};
    sbn3_divrem_options options{c.workers, c.prime_count, 0, 0, c.block, 0, 0, 0};
    sbn3_divrem_plan plan{};
    sbn3_divrem_info info{};
    assert(sbn3_divrem_query(&request, &options, &plan, &info) == SBN3_SUPPORTED);
    assert(info.algorithm == SBN3_DIVREM_BARRETT && info.spectrum_bytes && info.head_limbs == c.head);
    const size_t in = info.block_limbs;
    const bool u_varies = representation_varies(in + 1, in, c.workers, c.prime_count),
               t_varies = !info.ring_limbs && representation_varies(c.dn, in, c.workers, c.prime_count);
    // The case must exercise the variation, or it no longer guards the contract.
    assert(u_varies || t_varies);
    const size_t team_bytes = sbn3_team_storage_bytes() + sbn3_team_stack_resident_bytes(c.workers);
    const size_t admission = info.storage_bytes + team_bytes + (size_t(8) << 20);
    assert(admission <= budget);
    if (mem_available() < 2 * budget) {
        printf("divrem contract dn=%zu: SKIPPED, host MemAvailable below twice the %zu MiB budget\n", c.dn, budget >> 20);
        return false;
    }
    Fixture f(c.workers, false, budget);
    Service s(f, c.nn, c.dn, c.block, 0, 0, 0, c.prime_count);
    assert(s.info.plan_id == info.plan_id && s.info.storage_bytes == info.storage_bytes);
    // Two divisors with different normalization shifts, re-prepared on one binding.
    auto d0 = random_limbs(c.dn), d1 = random_limbs(c.dn);
    d0.back() |= uint64_t(1) << 63;
    d1.back() = (d1.back() >> 37) | 1;
    auto n0 = random_limbs(c.nn), n1 = random_limbs(c.nn - 1);
    // Top limbs <d1,d0> over zeros: the first word-division limb takes its add-back.
    auto n2 = random_limbs(c.nn);
    std::fill(n2.begin() + (c.nn - c.dn), n2.end(), 0);
    n2[c.nn - 1] = d0[c.dn - 1];
    n2[c.nn - 2] = d0[c.dn - 2];
    std::vector<uint64_t> first_q, first_r;
    unsigned executes = 0;
    auto check = [&](const std::vector<uint64_t> &d, const std::vector<uint64_t> &n, bool keep) {
        std::vector<uint64_t> q(n.size() - c.dn + 1, 0xdead), r(c.dn, 0xdead);
        const auto out = s.execute(n, q, r);
        assert(out.quotient_limbs == trim(q.data(), q.size()) && out.remainder_limbs == trim(r.data(), c.dn));
        certify(n.data(), n.size(), d.data(), c.dn, q.data(), out.quotient_limbs, r.data(), out.remainder_limbs, false);
        ++executes;
        if (keep) {
            first_q = q;
            first_r = r;
        }
    };
    s.prepare(d0);
    check(d0, n0, true);
    check(d0, n1, false);
    check(d0, n2, false);
    if (c.head) { // team-wide word division: the crafted numerator takes the add-back through the stitched parts
        sbn3_divrem_metrics head{};
        sbn3_divrem_get_metrics(s.bound, &head);
        assert(head.head_limbs >= 2 * c.head && head.head_corrections >= 1);
    }
    s.prepare(d1);
    check(d1, n0, false);
    s.prepare(d0);
    sbn3_divrem_metrics m{};
    sbn3_divrem_get_metrics(s.bound, &m);
    assert(m.prepares == 3 && m.executes == executes);
    // Rebind the same prepared range: identical result, no arena growth.
    sbn3_arena_stats before{}, after{};
    sbn3_arena_get_stats(f.arena, &before);
    s.close();
    s.bind();
    s.prepare(d0);
    std::vector<uint64_t> q(first_q.size()), r(c.dn);
    s.execute(n0, q, r);
    assert(q == first_q && r == first_r);
    sbn3_arena_get_stats(f.arena, &after);
    assert(after.payload_resident_bytes == before.payload_resident_bytes && after.prepare_calls == before.prepare_calls);
    printf("divrem contract dn=%zu nn=%zu W%u np=%u in=%zu ring=%zu blocks=%u head=%zu: producer/consumer variation U=%d T=%d, "
           "storage %zu MiB within the %zu MiB budget, 4 prepares, %u executes, rebind identical PASS\n",
           c.dn, c.nn, c.workers, c.prime_count, in, info.ring_limbs, info.blocks, info.head_limbs, int(u_varies), int(t_varies),
           info.storage_bytes >> 20, budget >> 20, executes + 1);
    fflush(stdout);
    return true;
}
} // namespace
int main() {
    // Default block policy with the prime family pinned: linear U and D spectra both vary.
    // Forced full inverse under a one-limb head: cyclic residual, varying U spectrum, parallel word division.
    const Case cases[] = {{3859823, 7719646, 16, 4, 0, 0}, {2546617, 5093234, 16, 0, 2546617, 1}};
    unsigned ran = 0;
    for (const auto &c : cases)
        ran += run(c);
    printf("division spectrum contract gates %s (%u of %zu cases ran)\n", ran ? "PASS" : "SKIPPED", ran,
           sizeof cases / sizeof cases[0]);
}
