#include "oracle.h"
#include "certificates.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
using U = uint64_t;
using W = __uint128_t;
static U state = 0x4143438792916349ULL;
static U rnd() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}
static void read(ref_number *z, std::string s) {
    bool neg = !s.empty() && s[0] == '-';
    if (neg)
        s.erase(0, 1);
    std::vector<unsigned char> b((s.size() + 1) / 2);
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[s.size() - 1 - i];
        unsigned v = c >= '0' && c <= '9' ? c - '0' : c - 'a' + 10;
        assert(v < 16);
        b[i / 2] |= v << ((i % 2) * 4);
    }
    ref_import(z, b.size(), -1, 1, 0, 0, b.data());
    if (neg)
        ref_neg(z, z);
}
static void print(const ref_number *z) {
    size_t n = (ref_sizeinbase(z, 2) + 63) / 64;
    std::vector<U> w(n);
    ref_export(w.data(), &n, -1, 8, 0, 0, z);
    if (ref_sgn(z) < 0)
        putchar('-');
    if (!n)
        putchar('0');
    else {
        printf("%llx", (unsigned long long)w[n - 1]);
        for (size_t j = n - 1; j--;)
            printf("%016llx", (unsigned long long)w[j]);
    }
    putchar('\n');
}
static void selftest() {
    unsigned checks = 0;
    const U p = (U(1) << 61) - 1;
    assert(ref_prime61(p));
    assert(!ref_prime61(p - 2));
    for (size_t n : {0u, 1u, 2u, 3u, 8u, 63u, 64u, 65u, 66u, 67u, 255u, 256u, 257u, 258u, 511u, 1024u})
        for (unsigned k = 0; k < 12; ++k) {
            std::vector<U> a(n);
            for (auto &x : a)
                x = rnd();
            U q = k & 1 ? p : ((U(1) << 60) + 39);
            U expected = 0;
            for (size_t j = n; j--;)
                expected = U(((W(expected) << 64) | a[j]) % q);
            assert(ref_mod_words(a.data(), n, q) == expected);
            ++checks;
        }
    const size_t n = 128;
    std::vector<U> a(n), b(n), out(2 * n);
    for (auto &v : a)
        v = rnd();
    for (auto &v : b)
        v = rnd();
    for (size_t i = 0; i < n; ++i) {
        U carry = 0;
        for (size_t j = 0; j < n; ++j) {
            W x = W(a[i]) * b[j] + out[i + j] + carry;
            out[i + j] = U(x);
            carry = U(x >> 64);
        }
        out[i + n] = carry;
    }
    assert(ref_product_equal(a.data(), n, b.data(), n, out.data(), out.size()));
    for (size_t j : {size_t(0), size_t(1), size_t(63), size_t(127), size_t(255)})
        for (unsigned bit : {0u, 31u, 63u}) {
            out[j] ^= U(1) << bit;
            assert(!ref_product_equal(a.data(), n, b.data(), n, out.data(), out.size()));
            out[j] ^= U(1) << bit;
            ++checks;
        }
    std::vector<U> twice = out;
    U overflow = ref_add_n(twice.data(), twice.data(), out.data(), out.size());
    twice.push_back(overflow);
    assert(ref_mac2_equal(a.data(), n, b.data(), n, a.data(), n, b.data(), n, twice.data(), twice.size()));
    twice[37] ^= 1;
    assert(!ref_mac2_equal(a.data(), n, b.data(), n, a.data(), n, b.data(), n, twice.data(), twice.size()));
    // An error invisible to two fixed moduli must not be accepted by fresh challenges.
    auto wrong = out;
    W error = W(p) * (p - 30), v = W(wrong[0]) + U(error);
    wrong[0] = U(v);
    v = W(wrong[1]) + U(error >> 64) + U(v >> 64);
    wrong[1] = U(v);
    U carry = U(v >> 64);
    for (size_t j = 2; carry && j < wrong.size(); ++j) {
        v = W(wrong[j]) + carry;
        wrong[j] = U(v);
        carry = U(v >> 64);
    }
    assert(!ref_product_equal(a.data(), n, b.data(), n, wrong.data(), wrong.size()));
    printf("test oracle: %u independent reducer/mutation gates, fresh-modulus blind-spot gate PASS\n",
           checks + 2);
}
int main(int argc, char **argv) {
    if (argc == 1) {
        selftest();
        return 0;
    }
    assert(argc == 2 && !strcmp(argv[1], "--server"));
    ref_int a, b, r;
    ref_inits(a, b, r, nullptr);
    std::string op, x, y;
    while (std::cin >> op >> x >> y) {
        read(a, x);
        read(b, y);
        if (op == "mul")
            ref_mul(r, a, b);
        else if (op == "add")
            ref_add(r, a, b);
        else if (op == "sub")
            ref_sub(r, a, b);
        else if (op == "div")
            ref_fdiv_q(r, a, b);
        else if (op == "mod")
            ref_mod(r, a, b);
        else if (op == "shl")
            ref_mul_2exp(r, a, ref_get_ui(b));
        else if (op == "shr")
            ref_tdiv_q_2exp(r, a, ref_get_ui(b));
        else if (op == "floor_shr")
            ref_fdiv_q_2exp(r, a, ref_get_ui(b));
        else if (op == "low")
            ref_fdiv_r_2exp(r, a, ref_get_ui(b));
        else if (op == "sqrt")
            ref_sqrt(r, a);
        else
            assert(false);
        print(r);
    }
    ref_clears(a, b, r, nullptr);
}
