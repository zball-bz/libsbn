// Contract test of the fp reciprocal and reciprocal square root kernels sbn3f_inv / sbn3f_rsqrt (src/fp/kernel.hpp),
// k = 1..512 result limbs, by exact integer arithmetic in the test (no reference library, no research code).
//
// The contract: k = e_top - L2 result limbs at grid L2 = floor((leps - 2) / 64), unit u = B^L2 (B = 2^64), and
//   inv:   |Z - 1/b| < 4u,         checked as  (Z - 4u) |b| < 1 < (Z + 4u) |b|,
//   rsqrt: |Z - 1/sqrt(x)| < 4u,   checked as  (Z - 4u <= 0 or (Z - 4u)^2 x < 1) and 1 < (Z + 4u)^2 x,
// with Z the kernel's output value (sign: b's for inv, + for rsqrt); every product is formed exactly over the limbs and
// compared with the power of B it must straddle. Also checked: the output is normalized (top limb nonzero, at most
// inv_need / rsqrt_need limbs; zero exactly when k <= 0), the kernel writes nothing outside z[0 .. need) (canaries on
// both sides), leaves the operand unchanged, and reads nothing outside it (operands placed against PROT_NONE pages on
// either side, and the output's capacity ending at one).
// Operands: every k 1..512 at several leading-zero counts (every one for small k: the scalar routes switch on it) and
// several leps in the same grid, with these families: random limbs (k + 1 limbs, the kernels' window), short operands,
// long operands (truncated by the kernels), all ones, powers of two (B^(e-1) among them), a top limb with one bit and
// the rest ones; rsqrt also every s = c/2 + 32 [e odd] (c = lzcnt & ~1) for small k and the s 61..63 band; both signs
// for inv; k <= 0 (zero results).
#include "fp/kernel.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <immintrin.h>
#include <unistd.h>
#include <vector>

using sbn::v3::fp::fx;
using sbn::v3::fp::fxr;
using sbn::v3::fp::limb;
using u128 = unsigned __int128;
using Nat = std::vector<uint64_t>;                     // little-endian limbs, no requirement on the top limb

namespace {

uint64_t rng_state = 0x5eed2026101700ull;
uint64_t rnd() {                                        // splitmix64
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

void trim(Nat &a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}
Nat mul_school(const Nat &a, const Nat &b) {
    if (a.empty() || b.empty()) return {};
    Nat r(a.size() + b.size(), 0);
    for (size_t j = 0; j < b.size(); ++j) {
        const uint64_t bj = b[j];
        if (!bj) continue;
        uint64_t c = 0;
        uint64_t *rj = r.data() + j;
        for (size_t i = 0; i < a.size(); ++i) {
            const u128 t = u128(a[i]) * bj + rj[i] + c;
            rj[i] = uint64_t(t);
            c = uint64_t(t >> 64);
        }
        rj[a.size()] = c;
    }
    trim(r);
    return r;
}
Nat shl_limbs(const Nat &a, size_t s) {                 // a B^s
    if (a.empty()) return {};
    Nat r(s, 0);
    r.insert(r.end(), a.begin(), a.end());
    return r;
}
Nat mul_small(const Nat &a, uint64_t m) {
    Nat r(a.size() + 1, 0);
    uint64_t c = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const u128 t = u128(a[i]) * m + c;
        r[i] = uint64_t(t);
        c = uint64_t(t >> 64);
    }
    r[a.size()] = c;
    trim(r);
    return r;
}
Nat add(const Nat &a, const Nat &b) {
    const Nat &x = a.size() >= b.size() ? a : b, &y = a.size() >= b.size() ? b : a;
    Nat r(x.size() + 1, 0);
    unsigned char c = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        unsigned long long t;
        c = _addcarry_u64(c, x[i], i < y.size() ? y[i] : 0, &t);
        r[i] = t;
    }
    r[x.size()] = c;
    trim(r);
    return r;
}
int cmp(const Nat &a, const Nat &b) {                   // both trimmed
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (size_t i = a.size(); i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
Nat sub(const Nat &a, const Nat &b) {                   // a >= b
    Nat r(a.size(), 0);
    unsigned char c = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        unsigned long long t;
        c = _subborrow_u64(c, a[i], i < b.size() ? b[i] : 0, &t);
        r[i] = t;
    }
    trim(r);
    return r;
}
Nat slice(const Nat &a, size_t lo, size_t hi) {        // limbs [lo, hi) of a, trimmed
    hi = hi < a.size() ? hi : a.size();
    Nat r(lo < hi ? a.begin() + long(lo) : a.end(), lo < hi ? a.begin() + long(hi) : a.end());
    trim(r);
    return r;
}
// exact product: schoolbook below 40 limbs, else Karatsuba (unbalanced operands: the longer one in pieces of the
// shorter one's length); checked against the schoolbook product at start-up (main)
Nat mul(const Nat &a, const Nat &b) {
    const size_t s = a.size() < b.size() ? a.size() : b.size(), l = a.size() < b.size() ? b.size() : a.size();
    if (s < 40) return mul_school(a, b);
    const Nat &lo = a.size() < b.size() ? a : b, &hi = a.size() < b.size() ? b : a;
    if (2 * s <= l) {
        Nat r;
        for (size_t off = 0; off < l; off += s) r = add(r, shl_limbs(mul(lo, slice(hi, off, off + s)), off));
        return r;
    }
    const size_t h = l / 2;
    const Nat a0 = slice(a, 0, h), a1 = slice(a, h, a.size()), b0 = slice(b, 0, h), b1 = slice(b, h, b.size());
    const Nat z0 = mul(a0, b0), z2 = mul(a1, b1), z1 = sub(sub(mul(add(a0, a1), add(b0, b1)), z0), z2);
    return add(add(z0, shl_limbs(z1, h)), shl_limbs(z2, 2 * h));
}
// sign of A * B^p - 1 for A >= 0 (p any integer): A B^p vs 1
int cmp_one(const Nat &A, int64_t p) {
    if (A.empty()) return -1;
    if (p > 0) return 1;                                // A >= 1: A B^p >= B > 1
    // A vs B^(-p)
    const size_t q = size_t(-p);
    if (A.size() != q + 1) return A.size() < q + 1 ? -1 : 1;
    if (A[q] != 1) return A[q] < 1 ? -1 : 1;
    for (size_t i = 0; i < q; ++i)
        if (A[i]) return 1;
    return 0;
}

struct Op {
    Nat v;                                              // the limbs (top nonzero)
    int64_t e;                                          // exponent: value v B^(e - n)
    bool neg;
};

// the operand's top limb with exactly ld leading zeros: one bit (pow2), all ones below it, or random
uint64_t top_limb(unsigned ld, int kind) {
    const uint64_t hb = uint64_t(1) << (63 - ld);
    if (kind == 1) return hb;
    if (kind == 2) return hb | (hb - 1);
    return hb | (rnd() & (hb - 1));
}
// families: 0 random, 1 power of two (lower limbs zero), 2 all ones, 3 one top bit and the rest ones
Op make_op(unsigned ld, size_t n, int fam, int64_t e, bool neg) {
    Op o{Nat(n), e, neg};
    for (size_t i = 0; i + 1 < n; ++i) o.v[i] = fam == 0 ? rnd() : fam == 1 ? 0 : ~uint64_t(0);
    o.v[n - 1] = top_limb(ld, fam == 0 ? 0 : fam == 1 || fam == 3 ? 1 : 2);
    return o;
}

// operand pages: the limbs end at a PROT_NONE page (place 1) or start right after one (place 2); place 0: heap
struct Pages {
    uint8_t *base = nullptr;
    size_t page = 0, span = 0;
    void init(size_t bytes) {
        page = size_t(sysconf(_SC_PAGESIZE));
        span = (bytes + page - 1) / page * page;
        base = static_cast<uint8_t *>(mmap(nullptr, span + 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (base == MAP_FAILED) { perror("mmap"); exit(2); }
        mprotect(base, page, PROT_NONE);
        mprotect(base + page + span, page, PROT_NONE);
    }
    uint64_t *at_end(size_t limbs) { return reinterpret_cast<uint64_t *>(base + page + span) - limbs; }
    uint64_t *at_start() { return reinterpret_cast<uint64_t *>(base + page); }
};
Pages opnd_pages, out_pages;

long n_checks = 0, n_fail = 0;
constexpr int G = 16;                                   // canary limbs on each side of the output
constexpr uint64_t CAN = 0xa5c3e1f00f1e3c5aull;

void fail(const char *op, const char *what, int64_t k, unsigned ld, int fam, int64_t leps, const Op &o) {
    ++n_fail;
    if (n_fail > 20) return;
    fprintf(stderr, "FAIL %s: %s (k %lld, ld %u, family %d, leps %lld, e %lld, n %zu, sign %d, top %016llx)\n", op, what,
            (long long)k, ld, fam, (long long)leps, (long long)o.e, o.v.size(), int(o.neg), (unsigned long long)o.v.back());
}

// one call: the operand at its place, the output with canaries (place 1: the output capacity ends at a PROT_NONE page)
template <bool RSQRT>
void check(const Op &o, int64_t leps, unsigned ld, int fam, int place) {
    const char *op = RSQRT ? "rsqrt" : "inv";
    const size_t n = o.v.size();
    uint64_t *d;
    std::vector<uint64_t> heap;
    if (place == 1) d = opnd_pages.at_end(n);
    else if (place == 2) d = opnd_pages.at_start();
    else { heap.assign(n + 2, 0); d = heap.data() + 1; }
    memcpy(d, o.v.data(), 8 * n);
    const fx x{d, o.e, uint64_t(n) | (o.neg ? sbn::v3::fp::sign_bit : 0)};
    const int64_t need = RSQRT ? sbn::v3::fp::rsqrt_need(x, leps) : sbn::v3::fp::inv_need(x, leps);
    const int64_t L2 = (leps - 2) >> 6, k = RSQRT ? ((2 - o.e) >> 1) - L2 : 1 - o.e - L2;
    std::vector<uint64_t> buf;
    uint64_t *z;
    if (place == 1 && need > 0) {
        z = out_pages.at_end(size_t(need));
        for (int i = 1; i <= G; ++i) z[-i] = CAN;
    } else {
        buf.assign(size_t(need) + 2 * G, CAN);
        z = buf.data() + G;
    }
    const fxr r = RSQRT ? sbn3f_rsqrt(z, &x, leps) : sbn3f_inv(z, &x, leps);
    ++n_checks;
    if (memcmp(d, o.v.data(), 8 * n) != 0) fail(op, "the operand changed", k, ld, fam, leps, o);
    for (int i = 1; i <= G; ++i)
        if (z[-i] != CAN) { fail(op, "write below z", k, ld, fam, leps, o); break; }
    if (place != 1)
        for (int i = 0; i < G; ++i)
            if (z[need + i] != CAN) { fail(op, "write past need", k, ld, fam, leps, o); break; }
    const int64_t rn = sbn::v3::fp::len(r.ns);
    if (need != (k > 0 ? k : 0)) fail(op, "need differs from k", k, ld, fam, leps, o);
    if (k <= 0) {
        if (r.ns != 0 || r.e != 0) fail(op, "k <= 0 must give zero", k, ld, fam, leps, o);
    } else {
        if (rn < 1 || rn > need) { fail(op, "output length outside 1 .. need", k, ld, fam, leps, o); return; }
        if (z[rn - 1] == 0) fail(op, "output not normalized", k, ld, fam, leps, o);
        if ((r.ns >> 63) != (RSQRT ? 0 : uint64_t(o.neg))) fail(op, "sign", k, ld, fam, leps, o);
    }
    // the value: Z = Zv B^gz; m = min(gz, L2); Zs = Zv B^(gz - m), D = 4 B^(L2 - m)
    Nat Zv(z, z + (k > 0 ? rn : 0));
    trim(Zv);
    const int64_t gz = Zv.empty() ? L2 : r.e - rn;
    const int64_t m = gz < L2 ? gz : L2;
    const Nat Zs = shl_limbs(Zv, size_t(gz - m));
    const size_t j = size_t(L2 - m);                     // D = 4 B^j
    Nat X = o.v;
    trim(X);
    const int64_t gx = o.e - int64_t(n);
    bool lo_ok, hi_ok;
    if (!RSQRT) {
        // (Zs -+ D) X B^(m + gx) vs 1
        const Nat P = mul(Zs, X), Q = shl_limbs(mul_small(X, 4), j);
        const Nat up = add(P, Q);
        hi_ok = cmp_one(up, m + gx) > 0;
        lo_ok = cmp(P, Q) <= 0 || cmp_one(sub(P, Q), m + gx) < 0;
    } else {
        // (Zs -+ D)^2 X B^(2m + gx) vs 1: Zs^2 X -+ 8 B^j Zs X + 16 B^2j X
        const Nat T1 = mul(Zs, X), T2 = mul(Zs, T1);
        const Nat c1 = shl_limbs(mul_small(T1, 8), j), c2 = shl_limbs(mul_small(X, 16), 2 * j);
        const Nat base = add(T2, c2);
        hi_ok = cmp_one(add(base, c1), 2 * m + gx) > 0;
        const Nat four = shl_limbs(Nat{4}, j);
        lo_ok = cmp(Zs, four) <= 0 || cmp_one(sub(base, c1), 2 * m + gx) < 0;
    }
    if (!lo_ok) fail(op, "Z - 4u is not below the exact value", k, ld, fam, leps, o);
    if (!hi_ok) fail(op, "Z + 4u is not above the exact value", k, ld, fam, leps, o);
}

}  // namespace

int main() {
    const timespec t0 = [] { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t; }();
    for (size_t an : {1, 39, 40, 41, 97, 512, 1030})                 // the test's own multiply against the schoolbook one
        for (size_t bn : {1, 40, 63, 300, 513, 1100}) {
            Nat a(an), b(bn);
            for (auto &x : a) x = rnd();
            for (auto &x : b) x = rnd();
            if (an > 2) a[an / 2] = 0, a[an - 2] = ~uint64_t(0);
            if (cmp(mul(a, b), mul_school(a, b)) != 0) { fprintf(stderr, "test multiply broken (%zu x %zu)\n", an, bn); return 2; }
        }
    opnd_pages.init(8 * 600);
    out_pages.init(8 * 600);
    const unsigned edge_ld[] = {0, 1, 2, 31, 32, 54, 55, 58, 59, 60, 61, 62, 63};
    long inv_n = 0, rs_n = 0;
    for (int64_t k = -2; k <= 512; ++k) {
        // inv: b = v B^(e - n), k = 1 - e - L2
        const bool small = k <= 24;
        std::vector<unsigned> lds;
        if (small) for (unsigned l = 0; l < 64; ++l) lds.push_back(l);
        else {
            for (unsigned l : edge_ld) lds.push_back(l);
            lds.push_back(unsigned(k * 13 % 64));
            lds.push_back(unsigned((k * 29 + 7) % 64));
        }
        for (unsigned ld : lds) {
            const int64_t e = int64_t(rnd() % 7) - 3, L2 = 1 - e - k, leps = 64 * L2 + 2 + int64_t(rnd() % 64);
            const size_t nk = size_t(k > 0 ? k : 1) + 1;
            check<false>(make_op(ld, nk, 0, e, rnd() & 1), leps, ld, 0, int(ld % 3));
            ++inv_n;
            if (small || ld == 0 || ld == 63 || ld == 61) {
                for (int fam = 1; fam <= 3; ++fam) {
                    check<false>(make_op(ld, nk, fam, e, false), leps, ld, fam, fam % 3);
                    ++inv_n;
                }
            }
        }
        {
            const unsigned ld = unsigned(rnd() % 64);
            const int64_t e = int64_t(rnd() % 7) - 3, L2 = 1 - e - k, leps = 64 * L2 + 2 + int64_t(rnd() % 64);
            const size_t nk = size_t(k > 0 ? k : 1);
            check<false>(make_op(ld, 1 + rnd() % 3, 0, e, false), leps, ld, 0, 1);                   // short
            check<false>(make_op(ld, nk + 2 + rnd() % 6, 0, e, true), leps, ld, 0, 2);              // long
            check<false>(make_op(63, 1 + rnd() % 3, 1, e, false), leps, 63, 1, 0);                  // B^(e-1)
            inv_n += 3;
        }
        // rsqrt: x = v B^(e - n), k = floor((2 - e) / 2) - L2; c = lzcnt & ~1, s = c/2 + 32 [e odd]
        std::vector<unsigned> ss;
        if (k <= 12) for (unsigned s = 0; s < 64; ++s) ss.push_back(s);
        else {
            for (unsigned s : {0u, 1u, 31u, 32u, 60u, 61u, 62u, 63u}) ss.push_back(s);
            ss.push_back(unsigned(rnd() % 64));
        }
        for (unsigned s : ss) {
            const bool odd = s >= 32;
            const unsigned c = 2 * (s - (odd ? 32 : 0)), ld = c + unsigned(rnd() & 1);
            int64_t e = 2 * (int64_t(rnd() % 7) - 3) + (odd ? 1 : 0);
            const int64_t L2 = ((2 - e) >> 1) - k, leps = 64 * L2 + 2 + int64_t(rnd() % 64);
            const size_t nk = size_t(k > 0 ? k : 1) + 1;
            check<true>(make_op(ld, nk, 0, e, false), leps, ld, 0, int(s % 3));
            ++rs_n;
            if (k <= 12 || s == 0 || s == 63) {
                for (int fam = 1; fam <= 3; ++fam) {
                    check<true>(make_op(c, nk, fam, e, false), leps, c, fam, fam % 3);
                    ++rs_n;
                }
            }
        }
        {
            const unsigned s = unsigned(rnd() % 64), c = 2 * (s % 32), ld = c + unsigned(rnd() & 1);
            const int64_t e = 2 * (int64_t(rnd() % 7) - 3) + (s >= 32 ? 1 : 0), L2 = ((2 - e) >> 1) - k;
            const int64_t leps = 64 * L2 + 2 + int64_t(rnd() % 64);
            const size_t nk = size_t(k > 0 ? k : 1);
            check<true>(make_op(ld, 1 + rnd() % 3, 0, e, false), leps, ld, 0, 1);                    // short
            check<true>(make_op(ld, nk + 2 + rnd() % 6, 0, e, false), leps, ld, 0, 2);               // long
            const int64_t eo = 2 * (int64_t(rnd() % 7) - 3) + 1, L2o = ((2 - eo) >> 1) - k;  // x = B^(e-1), e odd: 1/sqrt x = B^e_top
            check<true>(make_op(63, 1 + rnd() % 3, 1, eo, false), 64 * L2o + 2 + int64_t(rnd() % 64), 63, 1, 0);
            rs_n += 3;
        }
    }
    timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double sec = double(t1.tv_sec - t0.tv_sec) + 1e-9 * double(t1.tv_nsec - t0.tv_nsec);
    if (n_fail) {
        fprintf(stderr, "fp_inv_rsqrt_test: %ld of %ld checks failed\n", n_fail, n_checks);
        return 1;
    }
    printf("fp_inv_rsqrt_test: OK, %ld checks (inv %ld, rsqrt %ld; k -2..512), %.2f s\n", n_checks, inv_n, rs_n, sec);
    return 0;
}
