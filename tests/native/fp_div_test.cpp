// Contract test of the fp division kernel sbn3f_div (src/fp/kernel.hpp), k = 1..512 quotient limbs, by exact integer
// arithmetic in the test (no reference library, no research code).
//
// The contract: k = e_top - L2 quotient limbs (e_top = ea - eb + 1) at grid L2 = floor((leps - 2) / 64), unit u = B^L2
// (B = 2^64): |Z - a/b| < 4u, checked as (Z - 4u) |b| < |a| < (Z + 4u) |b| (or Z - 4u <= 0), every product formed
// exactly over the limbs. Also checked: the output is normalized (top limb nonzero, at most div_need limbs; zero exactly
// when k <= 0), its sign, the kernel writes nothing outside z[0 .. need) (canaries), leaves the operands unchanged and
// reads nothing outside them (operands against PROT_NONE pages on either side, the output's capacity ending at one).
// Operands: every k 1..512; every divisor leading-zero count for k <= 40 (the scalar cores and the Karp-Markstein path's
// short reciprocal chains, k 22..40), the edge counts and two more above; random dividend leading-zero counts; families
// random limbs (k + 1 limbs), short operands, long operands (truncated by the kernel), all ones, powers of two (B^(e-1)
// among them), a top limb with one bit and the rest ones, both signs; the quotient's overflow edge (b = B^(eb-1), a all
// ones: the result B^k - 1); a dividend whose top limb is 1; k <= 0 (zero results).
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

uint64_t rng_state = 0x5eed2026101800ull;
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
Pages opnd_pages2;

void fail(const char *what, int64_t k, const Op &a, const Op &b, int64_t leps) {
    ++n_fail;
    if (n_fail > 20) return;
    fprintf(stderr, "FAIL div: %s (k %lld, leps %lld; a: e %lld n %zu top %016llx; b: e %lld n %zu top %016llx)\n", what,
            (long long)k, (long long)leps, (long long)a.e, a.v.size(), (unsigned long long)a.v.back(), (long long)b.e, b.v.size(),
            (unsigned long long)b.v.back());
}
// A B^p vs C (p any integer): sign of A B^p - C
int cmp_shift(const Nat &A, int64_t p, const Nat &C) {
    return p >= 0 ? cmp(shl_limbs(A, size_t(p)), C) : cmp(A, shl_limbs(C, size_t(-p)));
}
uint64_t *place_op(const Op &o, int place, Pages &pg, std::vector<uint64_t> &heap) {
    const size_t n = o.v.size();
    uint64_t *d;
    if (place == 1) d = pg.at_end(n);
    else if (place == 2) d = pg.at_start();
    else { heap.assign(n + 2, 0); d = heap.data() + 1; }
    memcpy(d, o.v.data(), 8 * n);
    return d;
}
void check(const Op &a, const Op &b, int64_t leps, int place) {
    std::vector<uint64_t> ha, hb;
    uint64_t *da = place_op(a, place, opnd_pages, ha), *db = place_op(b, 3 - place == 3 ? 0 : 3 - place, opnd_pages2, hb);
    const fx x{da, a.e, uint64_t(a.v.size()) | (a.neg ? sbn::v3::fp::sign_bit : 0)};
    const fx y{db, b.e, uint64_t(b.v.size()) | (b.neg ? sbn::v3::fp::sign_bit : 0)};
    const int64_t need = sbn::v3::fp::div_need(x, y, leps);
    const int64_t L2 = (leps - 2) >> 6, k = a.e - b.e + 1 - L2;
    std::vector<uint64_t> buf;
    uint64_t *z;
    if (place == 1 && need > 0) {
        z = out_pages.at_end(size_t(need));
        for (int i = 1; i <= G; ++i) z[-i] = CAN;
    } else {
        buf.assign(size_t(need) + 2 * G, CAN);
        z = buf.data() + G;
    }
    const fxr r = sbn3f_div(z, &x, &y, leps);
    ++n_checks;
    if (memcmp(da, a.v.data(), 8 * a.v.size()) != 0 || memcmp(db, b.v.data(), 8 * b.v.size()) != 0) fail("an operand changed", k, a, b, leps);
    for (int i = 1; i <= G; ++i)
        if (z[-i] != CAN) { fail("write below z", k, a, b, leps); break; }
    if (place != 1)
        for (int i = 0; i < G; ++i)
            if (z[need + i] != CAN) { fail("write past need", k, a, b, leps); break; }
    const int64_t rn = sbn::v3::fp::len(r.ns);
    if (need != (k > 0 ? k : 0)) fail("need differs from k", k, a, b, leps);
    if (k <= 0) {
        if (r.ns != 0 || r.e != 0) fail("k <= 0 must give zero", k, a, b, leps);
    } else if (rn > 0) {
        if (rn > need) { fail("output length above need", k, a, b, leps); return; }
        if (z[rn - 1] == 0) fail("output not normalized", k, a, b, leps);
        if ((r.ns >> 63) != uint64_t(a.neg != b.neg)) fail("sign", k, a, b, leps);
    }
    // Z = Zv B^gz; m = min(gz, L2); Zs = Zv B^(gz - m), D = 4 B^(L2 - m): (Zs -+ D) |b| B^(m + gb - ga) vs |a|
    Nat Zv(z, z + (k > 0 ? rn : 0));
    trim(Zv);
    const int64_t gz = Zv.empty() ? L2 : r.e - rn;
    const int64_t m = gz < L2 ? gz : L2;
    const Nat Zs = shl_limbs(Zv, size_t(gz - m)), D = shl_limbs(Nat{4}, size_t(L2 - m));
    Nat A = a.v, Bv = b.v;
    trim(A);
    trim(Bv);
    const int64_t p = m + (b.e - int64_t(b.v.size())) - (a.e - int64_t(a.v.size()));
    const bool hi_ok = cmp_shift(mul(add(Zs, D), Bv), p, A) > 0;
    const bool lo_ok = cmp(Zs, D) <= 0 || cmp_shift(mul(sub(Zs, D), Bv), p, A) < 0;
    if (!lo_ok) fail("Z - 4u is not below the exact value", k, a, b, leps);
    if (!hi_ok) fail("Z + 4u is not above the exact value", k, a, b, leps);
}
// a top limb with ld leading zeros for the dividend, then the operand
Op opnd(unsigned ld, size_t n, int fam, int64_t e, bool neg) { return make_op(ld, n, fam, e, neg); }

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
    opnd_pages2.init(8 * 600);
    out_pages.init(8 * 600);
    const unsigned edge_ld[] = {0, 1, 2, 31, 32, 54, 55, 58, 59, 60, 61, 62, 63};
    for (int64_t k = -2; k <= 512; ++k) {
        const bool small = k <= 40;
        std::vector<unsigned> lds;
        if (small) for (unsigned l = 0; l < 64; ++l) lds.push_back(l);
        else {
            for (unsigned l : edge_ld) lds.push_back(l);
            lds.push_back(unsigned(k * 13 % 64));
            lds.push_back(unsigned((k * 29 + 7) % 64));
        }
        const size_t nk = size_t(k > 0 ? k : 1) + 1;
        for (unsigned ld : lds) {
            const int64_t ea = int64_t(rnd() % 7) - 3, eb = int64_t(rnd() % 7) - 3, L2 = ea - eb + 1 - k;
            const int64_t leps = 64 * L2 + 2 + int64_t(rnd() % 64);
            check(opnd(unsigned(rnd() % 64), nk, 0, ea, rnd() & 1), opnd(ld, nk, 0, eb, rnd() & 1), leps, int(ld % 3));
            if (small || ld == 0 || ld == 63 || ld == 61)
                for (int fam = 1; fam <= 3; ++fam) {
                    check(opnd(unsigned(rnd() % 64), nk, fam, ea, false), opnd(ld, nk, fam, eb, false), leps, fam % 3);
                    check(opnd(0, nk, 2, ea, false), opnd(ld, nk, fam, eb, true), leps, (fam + 1) % 3);   // a all ones
                }
        }
        {
            const int64_t ea = int64_t(rnd() % 7) - 3, eb = int64_t(rnd() % 7) - 3, L2 = ea - eb + 1 - k;
            const int64_t leps = 64 * L2 + 2 + int64_t(rnd() % 64);
            const unsigned lb = unsigned(rnd() % 64), la = unsigned(rnd() % 64);
            check(opnd(la, 1 + rnd() % 3, 0, ea, false), opnd(lb, nk, 0, eb, false), leps, 1);               // short a
            check(opnd(la, nk, 0, ea, false), opnd(lb, 1 + rnd() % 3, 0, eb, false), leps, 2);               // short b
            check(opnd(la, nk + 1 + rnd() % 7, 0, ea, true), opnd(lb, nk + 1 + rnd() % 7, 0, eb, false), leps, 0);   // long
            check(opnd(0, nk + rnd() % 4, 2, ea, false), opnd(63, 1 + rnd() % 3, 1, eb, false), leps, 1);   // the overflow edge
            check(opnd(0, nk, 2, ea, false), opnd(63, nk, 1, eb, false), leps, 2);
            check(opnd(63, nk, 0, ea, false), opnd(lb, nk, 0, eb, false), leps, 0);                       // a's top limb 1
            check(opnd(63, nk, 1, ea, false), opnd(0, nk, 2, eb, false), leps, 1);                        // tiny / large
        }
    }
    timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double sec = double(t1.tv_sec - t0.tv_sec) + 1e-9 * double(t1.tv_nsec - t0.tv_nsec);
    if (n_fail) {
        fprintf(stderr, "fp_div_test: %ld of %ld checks failed\n", n_fail, n_checks);
        return 1;
    }
    printf("fp_div_test: OK, %ld checks (k -2..512), %.2f s\n", n_checks, sec);
    return 0;
}
