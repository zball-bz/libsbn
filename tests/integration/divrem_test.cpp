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
    unsigned executes = 0;
    uint64_t corrections = 0;
    std::vector<uint64_t> first_q, first_r, first_n;
    auto divs = divisors(dn);
    if (dn > 64)
        divs.resize(3);
    for (size_t j = 0; j < divs.size(); ++j) {
        const auto &d = divs[j];
        s.prepare(d);
        auto nums = numerators(d, nn);
        if (nn > 2 * dn && j == 0)
            for (auto &n : numerators(d, nn - dn / 2))
                nums.push_back(n);
        if (j == 0)
            for (size_t length : lengths)
                for (auto &n : numerators(d, length))
                    nums.push_back(n);
        for (auto &n : nums) {
            const size_t count = n.size();
            std::vector<uint64_t> q(count >= dn ? count - dn + 1 : 0, 0xdead), r(dn, 0xdead);
            const auto out = s.execute(n, q, r);
            assert(out.quotient_limbs == trim(q.data(), q.size()) && out.remainder_limbs == trim(r.data(), dn));
            assert(out.corrections <= 8 * (s.info.block_limbs ? (q.size() + s.info.block_limbs - 1) / s.info.block_limbs : 1));
            certify(n.data(), count, d.data(), dn, q.data(), out.quotient_limbs, r.data(), out.remainder_limbs, exact);
            corrections += out.corrections;
            ++executes;
            if (first_n.empty() && count == nn) {
                first_n = n;
                first_q = q;
                first_r = r;
            }
        }
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
    printf("divrem service dn=%zu nn=%zu W%u alg=%u in=%zu ring=%zu blocks=%u head=%zu: %u executes, %zu divisors, "
           "%llu corrections, storage %zu KiB (spectra %zu KiB, shared %zu KiB), rebind identical PASS\n",
           dn, nn, workers, s.info.algorithm, s.info.block_limbs, s.info.ring_limbs, s.info.blocks, s.info.head_limbs, executes,
           divs.size(), (unsigned long long)corrections, s.info.storage_bytes >> 10, s.info.spectrum_bytes >> 10,
           s.info.shared_bytes >> 10);
    fflush(stdout);
}
// Every normalization shift against one binding (64 re-prepares), on a shape
// whose quotient has a two-limb word-division head above one complete block
// and, with the policy block size, on plain blocks.
static void shift_gates(size_t dn, size_t nn, size_t block, unsigned workers) {
    Fixture f(workers);
    Service s(f, nn, dn, block);
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
    service_gates(300, 4096, 1);
    service_gates(1000, 2000, 1, 0, 0, 1);
    service_gates(32, 64, 1, 0, 0, 0, 2);
    service_gates(4096, 4100, 1, 0, 0, 0, 2);
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
    shift_gates(70, 141, 70, 1);
    shift_gates(70, 141, 0, 1);
    shift_gates(300, 601, 300, 1);
    puts("exact division gates PASS");
}
