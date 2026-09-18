// Exact division gates: schoolbook entry, signed wrappers and the prepared
// service (word / schoolbook / block Barrett) against an independent oracle.
// Large shapes are certified by R<D plus N==Q*D+R modulo two 61-bit primes,
// computed here without the production divider; small shapes additionally
// compare exactly with the reference integers.
#include "product_support.hpp"
#include "sbn3/divrem.h"
#include <algorithm>
static uint64_t prime_after(uint64_t p) {
    p |= 1;
    while (!ref_prime61(p))
        p += 2;
    return p;
}
static const uint64_t primes[2] = {prime_after(0x1fffffffffff0001ULL), prime_after(0x1ffffffff0000123ULL)};
static void import(ref_int x, const uint64_t *a, size_t n) {
    if (n)
        ref_import(x, n, -1, 8, 0, 0, a);
    else
        ref_set_ui(x, 0);
}
// Independent certificate: N == Q*D + R and 0 <= R < D.
static void certify(const uint64_t *N, size_t nn, const uint64_t *D, size_t dn, const uint64_t *Q, size_t qn,
                    const uint64_t *R, size_t rn, bool exact) {
    assert(rn <= dn);
    // R < D by limb comparison with zero extension.
    int cmp = 0;
    for (size_t k = dn; k-- > 0 && !cmp;) {
        const uint64_t r = k < rn ? R[k] : 0;
        if (r != D[k])
            cmp = r > D[k] ? 1 : -1;
    }
    assert(cmp < 0);
    for (uint64_t p : primes) {
        const unsigned __int128 q = ref_mod_words(Q, qn, p), d = ref_mod_words(D, dn, p), r = ref_mod_words(R, rn, p),
                                n = ref_mod_words(N, nn, p);
        assert((q * d + r) % p == n);
    }
    if (exact) {
        ref_int a, b, x, y;
        ref_inits(a, b, x, y, nullptr);
        import(a, N, nn);
        import(b, D, dn);
        ref_fdiv_q(x, a, b);
        import(y, Q, qn);
        assert(ref_cmp(x, y) == 0);
        ref_mul(x, x, b);
        ref_sub(x, a, x);
        import(y, R, rn);
        assert(ref_cmp(x, y) == 0);
        ref_clears(a, b, x, y, nullptr);
    }
}
static size_t trim(const uint64_t *a, size_t n) {
    while (n && !a[n - 1])
        --n;
    return n;
}
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
struct Service {
    Fixture &f;
    sbn3_divrem_plan plan{};
    sbn3_divrem_info info{};
    sbn3_divrem_binding *bound = nullptr;
    size_t offset = 0;
    Service(Fixture &fixture, size_t nn, size_t dn, size_t block = 0, unsigned reuse = 0, unsigned residual = 0) : f(fixture) {
        sbn3_divrem_request request{nn, dn};
        sbn3_divrem_options options{sbn3_team_workers(f.team), 0, 0, reuse, block, residual, 1};
        allocation_watch_start();
        const auto rc = sbn3_divrem_query(&request, &options, &plan, &info);
        assert(!allocation_watch_stop());
        if (rc != SBN3_SUPPORTED)
            fprintf(stderr, "divrem query nn=%zu dn=%zu: %u\n", nn, dn, rc);
        assert(rc == SBN3_SUPPORTED);
        const auto saved = plan;
        options.memory_budget = info.storage_bytes - 1;
        sbn3_divrem_info need{};
        assert(sbn3_divrem_query(&request, &options, &plan, &need) == SBN3_QUERY_CAPACITY &&
               need.storage_bytes == info.storage_bytes && !memcmp(&saved, &plan, sizeof plan));
        offset = up(f.base + f.cursor, info.storage_alignment) - f.base;
        f.cursor = up(offset + info.storage_bytes, 4096) + 4096;
        sbn3_error e{};
        assert(sbn3_arena_prepare(f.arena, offset, info.storage_bytes, &e) == SBN3_OK);
        bind();
    }
    void bind() {
        allocation_watch_start();
        sbn3_divrem_bind(&plan, f.arena, offset, f.team, &bound);
        assert(!allocation_watch_stop());
    }
    void prepare(const std::vector<uint64_t> &d) {
        allocation_watch_start();
        sbn3_divrem_prepare(bound, {d.data(), d.size()});
        assert(!allocation_watch_stop());
    }
    sbn3_divrem_result execute(const std::vector<uint64_t> &n, std::vector<uint64_t> &q, std::vector<uint64_t> &r) {
        sbn3_divrem_result out{};
        allocation_watch_start();
        sbn3_divrem_execute(bound, {n.data(), n.size()}, {q.data(), q.size()}, {r.data(), r.size()}, &out);
        assert(!allocation_watch_stop());
        return out;
    }
    void close() {
        if (bound) {
            allocation_watch_start();
            sbn3_divrem_unbind(bound);
            assert(!allocation_watch_stop());
            bound = nullptr;
        }
    }
    ~Service() { close(); }
};
static void service_gates(size_t dn, size_t nn, unsigned workers, size_t block = 0, unsigned reuse = 0, unsigned residual = 0) {
    Fixture f(workers);
    Service s(f, nn, dn, block, reuse, residual);
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
    // Rebind the same prepared range; the first divisor must reproduce its result exactly.
    s.close();
    s.bind();
    s.prepare(divs[0]);
    std::vector<uint64_t> q(first_q.size()), r(dn);
    s.execute(first_n, q, r);
    assert(q == first_q && r == first_r);
    printf("divrem service dn=%zu nn=%zu W%u alg=%u in=%zu ring=%zu blocks=%u: %u executes, %zu divisors, "
           "%llu corrections, storage %zu KiB (spectra %zu KiB, shared %zu KiB), rebind identical PASS\n",
           dn, nn, workers, s.info.algorithm, s.info.block_limbs, s.info.ring_limbs, s.info.blocks, executes,
           divs.size(), (unsigned long long)corrections, s.info.storage_bytes >> 10, s.info.spectrum_bytes >> 10,
           s.info.shared_bytes >> 10);
    fflush(stdout);
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
    service_gates(1000, 2000, 1, 0, 0, 2);
    service_gates(65536, 131072, 16, 0, 0, 2);
    service_gates(4096, 8192, 16, 0, 8);
    service_gates(65536, 131072, 16);
    service_gates(65536, 69632, 16);
    service_gates(262144, 524288, 16);
    puts("exact division gates PASS");
}
