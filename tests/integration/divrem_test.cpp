// Exact division gates: schoolbook entry, signed wrappers and the prepared
// service (word / schoolbook / block Barrett) against an independent oracle.
// Large shapes are certified by R<D plus N==Q*D+R modulo two 61-bit primes,
// computed here without the production divider; small shapes additionally
// compare exactly with the reference integers.
#include "divrem_support.hpp"
struct Numbers {
    std::vector<uint64_t> n, d;
};
// Numerator patterns for a given divisor: random, all ones, zero, shorter
// than D, D itself, D-1, and Q*D+R with R in {0, 1, D-1}.
static std::vector<std::vector<uint64_t>> numerators(const std::vector<uint64_t> &d, size_t nn) {
    std::vector<std::vector<uint64_t>> out;
    const size_t dn = d.size();
    std::vector<uint64_t> v(nn);
    for (auto &w : v)
        w = random_word();
    out.push_back(v);
    std::fill(v.begin(), v.end(), UINT64_MAX);
    out.push_back(v);
    std::fill(v.begin(), v.end(), 0);
    out.push_back(v);
    if (nn > dn) {
        v.assign(nn, 0);
        for (size_t k = 0; k < dn - 1; ++k)
            v[k] = random_word();
        out.push_back(v);
    }
    if (nn >= dn) {
        v.assign(nn, 0);
        std::copy(d.begin(), d.end(), v.begin());
        out.push_back(v);
        if (!(dn == 1 && d[0] == 1)) {
            ref_int x;
            ref_init(x);
            import(x, d.data(), dn);
            ref_sub_ui(x, x, 1);
            size_t count = nn;
            std::fill(v.begin(), v.end(), 0);
            ref_export(v.data(), &count, -1, 8, 0, 0, x);
            out.push_back(v);
            ref_clear(x);
        }
        if (nn > dn && !(dn == 1 && d[0] == 1)) {
            // Top dn limbs equal D-1 over a random low part: the quotient is all
            // ones below a zero top limb, and a word-division head meets
            // <n2,n1> == <d1,d0> (quotient limb B-1) from its second limb on.
            ref_int x;
            ref_init(x);
            import(x, d.data(), dn);
            ref_sub_ui(x, x, 1);
            for (auto &w : v)
                w = random_word();
            std::vector<uint64_t> top(dn, 0);
            size_t count = dn;
            ref_export(top.data(), &count, -1, 8, 0, 0, x);
            std::copy(top.begin(), top.end(), v.begin() + (nn - dn));
            out.push_back(v);
            ref_clear(x);
        }
        if (nn > dn && dn >= 3) {
            // Top limbs <d1,d0> over zeros, random below the divisor's span: for a
            // normalized divisor with a nonzero low part the 3/2 estimate of the
            // first quotient limb is 1 while the limb is 0 (word-division add-back).
            for (auto &w : v)
                w = random_word();
            std::fill(v.begin() + (nn - dn), v.end(), 0);
            v[nn - 1] = d[dn - 1];
            v[nn - 2] = d[dn - 2];
            out.push_back(v);
        }
        for (unsigned which = 0; which < 3; ++which) {
            ref_int q, dd, r;
            ref_inits(q, dd, r, nullptr);
            import(dd, d.data(), dn);
            std::vector<uint64_t> qv(nn - dn + 1);
            for (auto &w : qv)
                w = random_word();
            qv.back() = which == 2 ? UINT64_MAX >> 1 : which; // short/long quotient variants
            if (which == 1)
                qv.back() = 0;
            import(q, qv.data(), qv.size());
            ref_mul(q, q, dd);
            if (which == 1)
                ref_add_ui(q, q, 1);
            if (which == 2) {
                ref_sub_ui(r, dd, 1);
                ref_add(q, q, r);
            }
            size_t count = nn + 1;
            v.assign(nn + 1, 0);
            if (ref_sizeinbase(q, 2) <= 64 * nn) {
                ref_export(v.data(), &count, -1, 8, 0, 0, q);
                v.resize(nn);
                out.push_back(v);
            }
            ref_clears(q, dd, r, nullptr);
        }
    }
    return out;
}
static std::vector<std::vector<uint64_t>> divisors(size_t dn) {
    std::vector<std::vector<uint64_t>> out;
    std::vector<uint64_t> v(dn);
    for (auto &w : v)
        w = random_word();
    v.back() |= 1;
    out.push_back(v);              // random
    v.back() = uint64_t(1) << 63;  // shift 0
    out.push_back(v);
    v.back() = 1;                  // shift 63
    out.push_back(v);
    std::fill(v.begin(), v.end(), UINT64_MAX); // all ones
    out.push_back(v);
    std::fill(v.begin(), v.end(), 0);
    v.back() = uint64_t(1) << 17;  // power of two
    out.push_back(v);
    if (dn == 1) {
        out.push_back({1});
        out.push_back({2});
    }
    return out;
}
static void basecase_gates() {
    unsigned cases = 0;
    for (size_t dn : {1u, 2u, 3u, 4u, 7u, 16u, 33u, 64u})
        for (auto &d : divisors(dn))
            for (size_t nn : {size_t(0), dn - 1, dn, dn + 1, dn + 3, 2 * dn, 5 * dn})
                for (auto &n : numerators(d, nn)) {
                    std::vector<uint64_t> q(nn >= dn ? nn - dn + 1 : 0, 0xdead), r(dn, 0xdead), s(nn + dn + 1);
                    allocation_watch_start();
                    const size_t qn = sbn3_divrem_basecase(q.data(), r.data(), n.data(), nn, d.data(), dn, s.data());
                    assert(!allocation_watch_stop());
                    assert(qn == trim(q.data(), q.size()));
                    certify(n.data(), nn, d.data(), dn, q.data(), qn, r.data(), trim(r.data(), dn), true);
                    // Signed wrapper, all sign combinations including negative zero.
                    for (unsigned sn = 0; sn < 2; ++sn)
                        for (unsigned sd = 0; sd < 2; ++sd) {
                            std::vector<uint64_t> qq(q.size() + 1), rr(dn + 1);
                            sbn3_int Q{qq.data(), qq.size(), 0, 0}, R{rr.data(), rr.size(), 0, 0};
                            sbn3_int_divrem_basecase(&Q, &R, {n.data(), nn, sn}, {d.data(), dn, sd}, {s.data(), s.size()});
                            assert(Q.size == qn && (Q.size ? Q.negative == (sn ^ sd) : !Q.negative));
                            assert(R.size == trim(r.data(), dn) && (R.size ? R.negative == sn : !R.negative));
                            assert(!memcmp(qq.data(), q.data(), qn * 8) && !memcmp(rr.data(), r.data(), R.size * 8));
                        }
                    ++cases;
                }
    printf("divrem basecase: %u shapes/patterns, exact oracle, signed wrapper PASS\n", cases);
}
static unsigned total_short_heads = 0; // executions under policy plans (no requested block size) that used the word-division head
static void service_gates(size_t dn, size_t nn, unsigned workers, size_t block = 0, unsigned reuse = 0, unsigned residual = 0, unsigned algorithm = 0,
                          const std::vector<size_t> &lengths = {}, int expected_head = -1) {
    Fixture f(workers);
    Service s(f, nn, dn, block, reuse, residual, algorithm);
    if (expected_head >= 0)
        assert(s.info.head_limbs == size_t(expected_head));
    if (s.info.algorithm == SBN3_DIVREM_BARRETT)
        assert(s.info.head_limbs < s.info.block_limbs &&
               size_t(s.info.blocks) * s.info.block_limbs + s.info.head_limbs >= s.info.quotient_limbs &&
               size_t(s.info.blocks) * s.info.block_limbs + s.info.head_limbs < s.info.quotient_limbs + s.info.block_limbs);
    const bool exact = nn * dn <= (size_t(1) << 20);
    unsigned executes = 0, short_heads = 0;
    uint64_t corrections = 0;
    std::vector<uint64_t> first_q, first_r, first_n;
    auto divs = divisors(dn);
    if (dn > 64)
        divs.resize(3);
    for (size_t j = 0; j < divs.size(); ++j) {
        const auto &d = divs[j];
        s.prepare(d);
        sbn3_divrem_metrics before{};
        sbn3_divrem_get_metrics(s.bound, &before);
        // D-1 as limbs, to recognize the pattern whose second head limb is B-1.
        std::vector<uint64_t> dm1(d);
        for (size_t k = 0; k < dn && !dm1[k]--; ++k) {
        }
        const bool normalized = d.back() >> 63;
        auto nums = numerators(d, nn);
        if (nn > 2 * dn && j == 0)
            for (auto &n : numerators(d, nn - dn / 2))
                nums.push_back(n);
        if (j == 0)
            for (size_t length : lengths)
                for (auto &n : numerators(d, length))
                    nums.push_back(n);
        // Shorter numerators that leave one to three quotient limbs above complete blocks of the plan, and quotients
        // of one to three limbs: the use every plan, the policy's included, makes of the word-division head.
        if (j == 0 && s.info.algorithm == SBN3_DIVREM_BARRETT)
            for (size_t complete : {size_t(s.info.blocks) - 1, size_t(0)})
                for (size_t left = 1; left <= 3; ++left)
                    if (const size_t length = dn + complete * s.info.block_limbs + left - 1; length < nn)
                        for (auto &n : numerators(d, length))
                            nums.push_back(n);
        for (auto &n : nums) {
            const size_t count = n.size();
            std::vector<uint64_t> q(count >= dn ? count - dn + 1 : 0, 0xdead), r(dn, 0xdead);
            sbn3_divrem_metrics entry{}, exit{};
            sbn3_divrem_get_metrics(s.bound, &entry);
            const auto out = s.execute(n, q, r);
            sbn3_divrem_get_metrics(s.bound, &exit);
            assert(out.quotient_limbs == trim(q.data(), q.size()) && out.remainder_limbs == trim(r.data(), dn));
            // Every execution serves the quotient limbs left over by the block size either by word division, without a
            // product pair, or as one padded block; a full-length numerator does exactly what the plan states.
            if (const size_t live = trim(n.data(), count); s.info.algorithm == SBN3_DIVREM_BARRETT && live >= dn) {
                const size_t in = s.info.block_limbs, quotient = live - dn + 1, left = quotient % in;
                const uint64_t by_words = exit.head_limbs - entry.head_limbs;
                assert((by_words == left && exit.products_executed == 2 * (quotient / in)) ||
                       (!by_words && exit.products_executed == 2 * ((quotient + in - 1) / in)));
                if (live == nn)
                    assert(by_words == s.info.head_limbs && exit.products_executed == 2 * s.info.blocks);
                short_heads += by_words && live < nn;
            }
            assert(out.corrections <= 8 * (s.info.block_limbs ? (q.size() + s.info.block_limbs - 1) / s.info.block_limbs : 1));
            certify(n.data(), count, d.data(), dn, q.data(), out.quotient_limbs, r.data(), out.remainder_limbs, exact);
            // Word-division head, <n2,n1> == <d1,d0>: below a zero first limb the second head limb is exactly B-1.
            if (normalized && count == nn && s.info.head_limbs >= 2 && std::equal(dm1.begin(), dm1.end(), n.begin() + (nn - dn)))
                assert(q[q.size() - 1] == 0 && q[q.size() - 2] == UINT64_MAX);
            corrections += out.corrections;
            ++executes;
            if (first_n.empty() && count == nn) {
                first_n = n;
                first_q = q;
                first_r = r;
            }
        }
        // The planned head is served product-free, and on a normalized divisor with a nonzero low part the
        // crafted <d1,d0>-over-zeros numerator forces the 3/2 estimate's add-back.
        sbn3_divrem_metrics after{};
        sbn3_divrem_get_metrics(s.bound, &after);
        if (s.info.head_limbs) {
            assert(after.head_limbs >= before.head_limbs + s.info.head_limbs);
            if (normalized && dn >= 3 && nn > dn)
                assert(after.head_corrections > before.head_corrections);
        } else if (s.info.algorithm != SBN3_DIVREM_BARRETT)
            assert(!after.head_limbs && !after.head_corrections);
    }
    // Signed wrapper on the last prepared divisor.
    {
        const auto &d = divs.back();
        std::vector<uint64_t> n(nn);
        for (auto &w : n)
            w = random_word();
        std::vector<uint64_t> q(nn >= dn ? nn - dn + 1 : 0), r(dn);
        const auto out = s.execute(n, q, r);
        for (unsigned sn = 0; sn < 2; ++sn)
            for (unsigned sd = 0; sd < 2; ++sd) {
                std::vector<uint64_t> qq(q.size() + 1), rr(dn + 1);
                sbn3_int Q{qq.data(), qq.size(), 0, 0}, R{rr.data(), rr.size(), 0, 0};
                allocation_watch_start();
                sbn3_int_divrem_execute(s.bound, &Q, &R, {n.data(), nn, sn}, {d.data(), dn, sd});
                assert(!allocation_watch_stop());
                assert(Q.size == out.quotient_limbs && (Q.size ? Q.negative == (sn ^ sd) : !Q.negative));
                assert(R.size == out.remainder_limbs && (R.size ? R.negative == sn : !R.negative));
                assert(!memcmp(qq.data(), q.data(), Q.size * 8) && !memcmp(rr.data(), r.data(), R.size * 8));
            }
    }
    sbn3_divrem_metrics m{};
    sbn3_divrem_get_metrics(s.bound, &m);
    assert(m.prepares == divs.size() && m.executes == executes + 4 + 1);
    // The longest numerator takes exactly the planned product pairs; the head is product-free.
    if (s.info.algorithm == SBN3_DIVREM_BARRETT)
        assert(m.products_executed == 2 * s.info.blocks);
    // Rebind the same prepared range; the first divisor must reproduce its result exactly.
    s.close();
    s.bind();
    s.prepare(divs[0]);
    std::vector<uint64_t> q(first_q.size()), r(dn);
    s.execute(first_n, q, r);
    assert(q == first_q && r == first_r);
    printf("divrem service dn=%zu nn=%zu W%u alg=%u in=%zu ring=%zu blocks=%u head=%zu: %u executes (%u shorter numerators with a "
           "word-division head), %zu divisors, %llu corrections, storage %zu KiB (spectra %zu KiB, shared %zu KiB), rebind identical PASS\n",
           dn, nn, workers, s.info.algorithm, s.info.block_limbs, s.info.ring_limbs, s.info.blocks, s.info.head_limbs, executes, short_heads,
           divs.size(), (unsigned long long)corrections, s.info.storage_bytes >> 10, s.info.spectrum_bytes >> 10,
           s.info.shared_bytes >> 10);
    fflush(stdout);
    total_short_heads += block ? 0 : short_heads;
}
// Every normalization shift against one binding (64 re-prepares), on a shape
// whose quotient has a two-limb word-division head above one complete block
// and, with the policy block size, on plain blocks.
static void shift_gates(size_t dn, size_t nn, size_t block, unsigned workers) {
    Fixture f(workers);
    // The block algorithm is the subject: requested by its block size, or by name with the policy's block size (a
    // one-shot request of this size is the policy's schoolbook case, which has no normalization of its own to gate).
    Service s(f, nn, dn, block, 0, 0, block ? 0 : SBN3_DIVREM_BARRETT);
    assert(s.info.algorithm == SBN3_DIVREM_BARRETT);
    unsigned executes = 0;
    for (unsigned shift = 0; shift < 64; ++shift) {
        std::vector<uint64_t> d(dn);
        for (auto &w : d)
            w = random_word();
        const uint64_t top = uint64_t(1) << (63 - shift);
        d.back() = top | (d.back() & (top - 1));
        s.prepare(d);
        for (auto &n : numerators(d, nn)) {
            std::vector<uint64_t> q(nn - dn + 1, 0xdead), r(dn, 0xdead);
            const auto out = s.execute(n, q, r);
            certify(n.data(), nn, d.data(), dn, q.data(), out.quotient_limbs, r.data(), out.remainder_limbs, true);
            ++executes;
        }
    }
    sbn3_divrem_metrics m{};
    sbn3_divrem_get_metrics(s.bound, &m);
    assert(m.prepares == 64 && m.executes == executes);
    printf("divrem shifts dn=%zu nn=%zu W%u in=%zu blocks=%u head=%zu: 64 shifts, %u executes, exact oracle PASS\n", dn, nn,
           workers, s.info.block_limbs, s.info.blocks, s.info.head_limbs, executes);
}
// Plan facts of one public query (no arena).
static sbn3_divrem_info planned(size_t dn, size_t nn, unsigned workers, unsigned reuse, size_t block = 0, size_t budget = 0,
                                sbn3_query_result expected = SBN3_SUPPORTED) {
    sbn3_divrem_request request{nn, dn};
    sbn3_divrem_options options{workers, 0, budget, reuse, block, 0, 0, 0};
    sbn3_divrem_plan plan{};
    sbn3_divrem_info info{};
    const auto rc = sbn3_divrem_query(&request, &options, &plan, &info);
    assert(rc == expected);
    return info;
}
// Quotient limbs a plan pays products for without needing them: blocks times block size plus the head, less the quotient.
static size_t padding(const sbn3_divrem_info &i) {
    return size_t(i.blocks) * i.block_limbs + i.head_limbs - i.quotient_limbs;
}
// Complete divisor-length blocks on request (block_limbs = dn). The quotient limbs above them are a word-division
// head while the final recipe serves it and one padded block otherwise: the cyclic ring may leave no room above the
// divisor (ring - dn < head), or the product pair may be cheaper than that many word steps. The longest served head
// is found from the public query; both sides of it are executed (values, exact head limbs and product pairs per
// execution, no allocation), and the ring-room and cost-limit denials must both occur. Whatever block size the policy
// takes on the same requests, it never pays products for a block that is mostly padding: near-equal blocks leave
// fewer padded limbs than blocks, a word-division head none.
static void head_recipe_gates() {
    unsigned kept = 0, no_room = 0, over_limit = 0;
    auto request = [](size_t dn, size_t complete, size_t head) { return (complete + 1) * dn + head - 1; }; // qn = complete * dn + head
    auto classify = [&](const sbn3_divrem_info &requested, size_t dn, size_t complete, size_t head) {
        assert(requested.block_limbs == dn && (!requested.ring_limbs || dn + requested.head_limbs <= requested.ring_limbs));
        if (requested.head_limbs) {
            assert(requested.head_limbs == head && requested.blocks == complete && !padding(requested));
            return &kept;
        }
        assert(requested.blocks == complete + 1 && padding(requested) == dn - head);
        return requested.ring_limbs && requested.ring_limbs - dn < head ? &no_room : &over_limit;
    };
    struct Shape {
        size_t dn;
        unsigned workers, reuse;
    };
    // 12285 sits three limbs under the ring 12288, so ring room ends its head; 3001 at one worker and 20000 at sixteen
    // have ample room above the divisor, so the cost limit does. The counters below hold the shapes to that.
    for (const Shape &shape : {Shape{12285, 16, 8}, Shape{3001, 1, 0}, Shape{20000, 16, 8}}) {
        size_t limit = 0;
        while (limit + 1 < shape.dn &&
               planned(shape.dn, request(shape.dn, 1, limit + 1), shape.workers, shape.reuse, shape.dn).head_limbs == limit + 1)
            ++limit;
        assert(limit);
        for (size_t head : {limit, limit + 1}) {
            const size_t nn = request(shape.dn, 1, head);
            ++*classify(planned(shape.dn, nn, shape.workers, shape.reuse, shape.dn), shape.dn, 1, head);
            service_gates(shape.dn, nn, shape.workers, shape.dn, shape.reuse, 0, 0, {nn - 1, nn - head}, head == limit ? int(head) : 0);
            service_gates(shape.dn, nn, shape.workers, 0, shape.reuse, 0, 0, {nn - 1, nn - head}); // the policy's plan on the same request
        }
    }
    assert(kept && no_room && over_limit);
    const unsigned executed = kept + no_room + over_limit, small_no_room = no_room, small_over_limit = over_limit;
    // Query level, where both denials were found on large divisors: a few limbs under the rings 393216 and 1310720,
    // and divisors whose ring has ample room while five head limbs exceed the cyclic pair's limit.
    unsigned policy = 0;
    for (size_t dn : {393214u, 1310717u, 325545u, 387141u, 710019u})
        for (size_t complete : {1u, 2u})
            for (size_t head = 1; head <= 8; ++head) {
                const size_t nn = request(dn, complete, head);
                ++*classify(planned(dn, nn, 16, 8, dn), dn, complete, head);
                const auto chosen = planned(dn, nn, 16, 8);
                assert(chosen.algorithm == SBN3_DIVREM_BARRETT && padding(chosen) < chosen.blocks);
                ++policy;
            }
    assert(no_room > small_no_room && over_limit > small_over_limit); // both denials occur on the large divisors too
    printf("divrem head recipe: %u requested complete-block plans keep their word head, %u have no ring room, %u are over the "
           "cost limit (%u executed on both sides of the limit); %u policy plans pad fewer limbs than blocks PASS\n",
           kept, no_room, over_limit, executed, policy);
}
// The policy's algorithm is a cost decision between exact algorithms. What it owes, without naming a threshold: more
// expected executions per divisor never move a request from the block algorithm back to the schoolbook (the planning
// the block algorithm pays once is what the schoolbook avoids); a request served by the schoolbook for one use is
// served in blocks once enough executions are expected, unless the shape is one the schoolbook always serves
// (it is then the schoolbook at every hint); a requested block size is not overruled by the one-use rule.
static void algorithm_policy_gates() {
    unsigned shapes = 0, moved = 0;
    for (size_t dn : {3u, 40u, 65u, 97u, 200u, 511u, 1000u, 4099u, 70001u})
        for (size_t qn : {1u, 8u, 9u, 33u, 200u, 1001u, 5000u}) {
            const size_t nn = dn + qn - 1;
            unsigned last = SBN3_DIVREM_WORD;
            bool blocks = false;
            for (unsigned reuse : {0u, 1u, 2u, 8u, 64u, 4096u, 1u << 20, 1u << 30}) {
                const auto info = planned(dn, nn, 1, reuse);
                assert(!blocks || info.algorithm == SBN3_DIVREM_BARRETT);
                blocks = info.algorithm == SBN3_DIVREM_BARRETT;
                last = info.algorithm;
            }
            const auto once = planned(dn, nn, 1, 0);
            moved += once.algorithm == SBN3_DIVREM_SCHOOLBOOK && last == SBN3_DIVREM_BARRETT;
            if (once.algorithm == SBN3_DIVREM_SCHOOLBOOK && last != SBN3_DIVREM_BARRETT) // always the schoolbook's shape
                assert(planned(dn, nn, 1, 1u << 30).algorithm == once.algorithm);
            if (dn > 2 && qn > 1 && once.algorithm == SBN3_DIVREM_SCHOOLBOOK && last == SBN3_DIVREM_BARRETT)
                assert(planned(dn, nn, 1, 0, std::min(dn, qn)).algorithm == SBN3_DIVREM_BARRETT);
            ++shapes;
        }
    assert(moved); // the one-use rule is exercised: some request changes algorithm with the expected executions
    printf("divrem algorithm policy: %u shapes, the block algorithm is kept under every larger reuse hint; %u move from the "
           "schoolbook to blocks with the hint PASS\n", shapes, moved);
}
// memory_budget passes over block sizes that do not fit instead of rejecting the request. The least requirement among
// the policy's block sizes is public (a query that nothing fits reports it), so the property needs no block size by
// name: that requirement is supported exactly and nothing below it is; a budget one byte below the unconstrained plan
// still gets a plan that fits whenever a smaller block size exists; the plan taken under a budget divides correctly.
// Returns whether the request had a smaller block size to pass to.
static bool budget_gates(size_t dn, size_t nn, unsigned workers, unsigned reuse) {
    const auto free = planned(dn, nn, workers, reuse);
    const auto need = planned(dn, nn, workers, reuse, 0, 4096, SBN3_QUERY_CAPACITY);
    assert(need.storage_bytes > 4096 && need.storage_bytes <= free.storage_bytes);
    const auto least = planned(dn, nn, workers, reuse, 0, need.storage_bytes);
    assert(least.storage_bytes == need.storage_bytes);
    assert(planned(dn, nn, workers, reuse, 0, need.storage_bytes - 1, SBN3_QUERY_CAPACITY).storage_bytes == need.storage_bytes);
    assert(planned(dn, nn, workers, reuse, 0, free.storage_bytes).plan_id == free.plan_id); // a budget that fits changes nothing
    const bool passes = need.storage_bytes < free.storage_bytes;
    if (passes) {
        const auto fitted = planned(dn, nn, workers, reuse, 0, free.storage_bytes - 1);
        assert(fitted.storage_bytes < free.storage_bytes && fitted.plan_id != free.plan_id && padding(fitted) < fitted.blocks);
    }
    // The plan taken under the least budget is a working plan.
    Fixture f(workers);
    Service s(f, nn, dn, 0, reuse, 0, 0, 0, need.storage_bytes);
    assert(s.info.plan_id == least.plan_id && s.info.storage_bytes == need.storage_bytes);
    auto d = divisors(dn)[0];
    s.prepare(d);
    unsigned executes = 0;
    for (auto &n : numerators(d, nn)) {
        std::vector<uint64_t> q(nn - dn + 1, 0xdead), r(dn, 0xdead);
        const auto out = s.execute(n, q, r);
        certify(n.data(), nn, d.data(), dn, q.data(), out.quotient_limbs, r.data(), out.remainder_limbs, false);
        ++executes;
    }
    printf("divrem budget dn=%zu nn=%zu W%u reuse=%u: unconstrained %zux%u+%zu %zu KiB; least requirement %zux%u+%zu %zu KiB "
           "(%s), %u executes under it PASS\n", dn, nn, workers, reuse, free.block_limbs, free.blocks, free.head_limbs,
           free.storage_bytes >> 10, least.block_limbs, least.blocks, least.head_limbs, need.storage_bytes >> 10,
           passes ? "a smaller block size than the unconstrained plan" : "the unconstrained plan", executes);
    return passes;
}
int main() {
    basecase_gates();
    service_gates(3, 40, 1);
    service_gates(64, 128, 1);
    service_gates(65, 130, 1);
    service_gates(100, 200, 1);
    service_gates(200, 201, 1);
    service_gates(200, 230, 1);
    service_gates(1000, 1017, 1);
    service_gates(1000, 1200, 1);
    service_gates(1000, 2000, 1);
    for (size_t block : {1000u, 500u, 334u, 17u})
        service_gates(1000, 2000, 1, block);
    // Requested block sizes through both block-inverse routes (one schoolbook division for short blocks, the Newton
    // ladder above) and next to each other at the switch; every divisor pattern, including the all-ones and the
    // power-of-two divisors that frame the inverse at its two ends.
    for (size_t block : {1u, 2u, 15u, 16u, 863u, 864u, 865u, 866u, 1500u, 3001u})
        service_gates(3001, 6002, 1, block);
    service_gates(300, 4096, 1);
    service_gates(1000, 2000, 1, 0, 0, 1);
    service_gates(32, 64, 1, 0, 0, 0, 2);
    service_gates(4096, 4100, 1, 0, 0, 0, 2);
    // The block algorithm by name on the small shapes above, whatever the policy takes for one use of them, and the
    // schoolbook by name on a shape the policy divides in blocks.
    service_gates(65, 130, 1, 0, 0, 0, 2);
    service_gates(100, 200, 1, 0, 0, 0, 2);
    service_gates(200, 230, 1, 0, 0, 0, 2);
    service_gates(1000, 1017, 1, 0, 0, 0, 2);
    service_gates(300, 4096, 1, 0, 0, 0, 2);
    service_gates(1000, 2000, 1, 0, 0, 0, 1);
    algorithm_policy_gates();
    service_gates(1000, 2000, 1, 0, 0, 0, 1);
    service_gates(1000, 2000, 1, 0, 0, 2);
    service_gates(65536, 131072, 16, 0, 0, 2);
    service_gates(4096, 8192, 16, 0, 8);
    service_gates(65536, 131072, 16);
    service_gates(65536, 69632, 16);
    service_gates(262144, 524288, 16);
    // Short head block by word division: forced complete blocks under heads of
    // 1, 2, 3 and 7 limbs, the 2dn-1 / 2dn / 2dn+1 neighbourhood of a full
    // inverse, and shorter numerators crossing every block boundary of the plan.
    service_gates(1000, 2000, 1, 1000, 0, 0, 0, {1000, 1001, 1002, 1499, 1999}, 1);
    service_gates(1000, 1999, 1, 1000, 0, 0, 0, {}, 0);
    service_gates(1000, 2001, 1, 1000, 0, 0, 0, {}, 2);
    service_gates(1000, 2001, 1, 333, 0, 0, 0, {1332, 1333, 1334, 1665, 1666, 1667}, 3);
    service_gates(1000, 2006, 1, 500, 0, 0, 0, {}, 7);
    service_gates(1000, 2001, 1, 1000, 0, 1, 0, {}, 2); // linear residual under the head
    service_gates(1000, 2001, 1, 1000, 0, 2, 0, {}, 2); // cyclic residual under the head
    service_gates(1000, 2400, 1, 1000); // 401-limb remainder: padded block, as planned
    for (size_t dn : {777u, 3001u, 4097u})
        for (size_t nn : {2 * dn - 1, 2 * dn, 2 * dn + 1, 3 * dn, 3 * dn + 1})
            service_gates(dn, nn, 1, 0, nn == 2 * dn ? 8 : 0, 0, 0, {dn, dn + 1, nn - 1});
    service_gates(3001, 6002, 1, 3001, 0, 0, 0, {}, 1);
    service_gates(70001, 140002, 16, 70001, 0, 0, 0, {70001, 70002, 140001}, 1);
    service_gates(70001, 140003, 16, 0, 8, 0, 0, {140002});
    service_gates(70001, 210004, 16, 0, 0, 0, 0, {140002, 140003});
    head_recipe_gates();
    // Budgets: non-power-of-two 2d/d and 3d/d shapes, one and sixteen workers; at least one must have a smaller block
    // size than its unconstrained plan, or the pass-over path is no longer exercised.
    unsigned passed_over = 0;
    passed_over += budget_gates(70001, 140003, 16, 8);
    passed_over += budget_gates(3001, 6002, 1, 0);
    passed_over += budget_gates(20011, 60034, 16, 0);
    assert(passed_over);
    shift_gates(70, 141, 70, 1);
    shift_gates(70, 141, 0, 1);
    shift_gates(300, 601, 300, 1);
    // The word-division head is live under the policy's own plans (shorter numerators), not only on requested block sizes.
    // A head the policy plans for its longest numerator exists only above 2^20 divisor limbs: divrem_contract_test.
    assert(total_short_heads);
    printf("divrem policy plans: %u executions of shorter numerators served their leftover limbs by word division\n", total_short_heads);
    puts("exact division gates PASS");
}
