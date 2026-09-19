#pragma once
#include "divrem_support.hpp"
// Numerator patterns for a given divisor: random, all ones, zero, shorter
// than D, D itself, D-1, and Q*D+R with R in {0, 1, D-1}.
inline std::vector<std::vector<uint64_t>> numerators(const std::vector<uint64_t> &d, size_t nn) {
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
inline std::vector<std::vector<uint64_t>> divisors(size_t dn) {
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
