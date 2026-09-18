#include "oracle.h"
#include "certificates.h"
#include "sbn3/mul.h"
#include "sbn3/newton.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/random.h>
#include <errno.h>

namespace {
using U = uint64_t;
using W = __uint128_t;
struct Integer {
    std::vector<U> d;
    bool negative = false;
    Integer(U x = 0) {
        if (x)
            d.push_back(x);
    }
    void trim() {
        while (!d.empty() && !d.back())
            d.pop_back();
        if (d.empty())
            negative = false;
    }
    size_t bits() const { return d.empty() ? 0 : 64 * (d.size() - 1) + 64 - std::countl_zero(d.back()); }
};
Integer &get(ref_number *p) {
    assert(p && p->value);
    return *static_cast<Integer *>(p->value);
}
const Integer &get(const ref_number *p) {
    assert(p && p->value);
    return *static_cast<const Integer *>(p->value);
}
int magnitude(const Integer &a, const Integer &b) {
    if (a.d.size() != b.d.size())
        return a.d.size() < b.d.size() ? -1 : 1;
    for (size_t i = a.d.size(); i--;)
        if (a.d[i] != b.d[i])
            return a.d[i] < b.d[i] ? -1 : 1;
    return 0;
}
int compare(const Integer &a, const Integer &b) {
    if (a.negative != b.negative)
        return a.negative ? -1 : 1;
    return (a.negative ? -1 : 1) * magnitude(a, b);
}
Integer add_abs(const Integer &a, const Integer &b) {
    Integer r;
    r.d.resize(std::max(a.d.size(), b.d.size()) + 1);
    U carry = 0;
    for (size_t i = 0; i + 1 < r.d.size(); ++i) {
        W v = W(i < a.d.size() ? a.d[i] : 0) + (i < b.d.size() ? b.d[i] : 0) + carry;
        r.d[i] = U(v);
        carry = U(v >> 64);
    }
    r.d.back() = carry;
    r.trim();
    return r;
}
Integer sub_abs(const Integer &a, const Integer &b) {
    assert(magnitude(a, b) >= 0);
    Integer r;
    r.d.resize(a.d.size());
    U borrow = 0;
    for (size_t i = 0; i < a.d.size(); ++i) {
        W v = W(i < b.d.size() ? b.d[i] : 0) + borrow;
        r.d[i] = a.d[i] - U(v);
        borrow = (W(a.d[i]) < v);
    }
    assert(!borrow);
    r.trim();
    return r;
}
Integer add(const Integer &a, const Integer &b) {
    if (a.negative == b.negative) {
        auto r = add_abs(a, b);
        r.negative = a.negative && !r.d.empty();
        return r;
    }
    const bool larger = magnitude(a, b) >= 0;
    auto r = larger ? sub_abs(a, b) : sub_abs(b, a);
    r.negative = (larger ? a.negative : b.negative) && !r.d.empty();
    return r;
}
Integer sub(const Integer &a, Integer b) {
    if (!b.d.empty())
        b.negative = !b.negative;
    return add(a, b);
}
Integer shl(const Integer &a, size_t bits) {
    if (a.d.empty())
        return {};
    Integer r;
    size_t words = bits / 64;
    unsigned shift = bits % 64;
    r.d.resize(a.d.size() + words + bool(shift));
    U carry = 0;
    for (size_t i = 0; i < a.d.size(); ++i) {
        r.d[i + words] = (a.d[i] << shift) | carry;
        carry = shift ? a.d[i] >> (64 - shift) : 0;
    }
    if (shift)
        r.d.back() = carry;
    r.negative = a.negative;
    r.trim();
    return r;
}
Integer shr(const Integer &a, size_t bits) {
    size_t words = bits / 64;
    unsigned shift = bits % 64;
    if (words >= a.d.size())
        return {};
    Integer r;
    r.d.resize(a.d.size() - words);
    for (size_t i = 0; i < r.d.size(); ++i) {
        r.d[i] = a.d[i + words] >> shift;
        if (shift && i + words + 1 < a.d.size())
            r.d[i] |= a.d[i + words + 1] << (64 - shift);
    }
    r.negative = a.negative;
    r.trim();
    return r;
}
Integer lower(const Integer &a, size_t bits) {
    Integer r = a;
    r.negative = false;
    if (bits / 64 < r.d.size()) {
        r.d.resize(bits / 64 + bool(bits % 64));
        if (bits % 64)
            r.d.back() &= (U(1) << (bits % 64)) - 1;
    }
    r.trim();
    return r;
}
Integer exact_mul(const Integer &a, const Integer &b) {
    Integer r;
    if (a.d.empty() || b.d.empty())
        return r;
    r.d.resize(a.d.size() + b.d.size());
    for (size_t i = 0; i < a.d.size(); ++i) {
        U carry = 0;
        for (size_t j = 0; j < b.d.size(); ++j) {
            W v = W(a.d[i]) * b.d[j] + r.d[i + j] + carry;
            r.d[i + j] = U(v);
            carry = U(v >> 64);
        }
        r.d[i + b.d.size()] = carry;
    }
    r.negative = a.negative != b.negative;
    r.trim();
    return r;
}
U modmul(U a, U b, U p) {
    return U(W(a) * b % p);
}
U power(U a, size_t n, U p) {
    U r = 1;
    for (; n; n >>= 1, a = modmul(a, a, p))
        if (n & 1)
            r = modmul(r, a, p);
    return r;
}
U entropy() {
    U x;
    size_t done = 0;
    while (done < sizeof x) {
        auto n = getrandom(reinterpret_cast<char *>(&x) + done, sizeof x - done, 0);
        if (n < 0 && errno == EINTR)
            continue;
        assert(n > 0);
        done += size_t(n);
    }
    return x;
}
void challenges(U (&p)[2]) {
    static FILE *replay = []() {
        const char *s = getenv("SBN3_ORACLE_REPLAY");
        if (!s)
            return static_cast<FILE *>(nullptr);
        FILE *f = fopen(s, "r");
        assert(f);
        return f;
    }();
    if (replay) {
        char line[1024];
        assert(fgets(line, sizeof line, replay));
        char *s = strstr(line, "\"moduli\":[");
        assert(s);
        s += 10;
        for (auto &q : p) {
            q = strtoull(s, &s, 10);
            assert(ref_prime61(q));
            if (*s == ',')
                ++s;
        }
        return;
    }
    for (auto &q : p) {
        do {
            q = (entropy() & ((U(1) << 60) - 1)) | (U(1) << 60) | 1;
        } while (!ref_prime61(q));
    }
}
void trace(const U (&p)[2], size_t an, size_t bn, size_t rn, bool ok) {
    const char *name = getenv("SBN3_ORACLE_TRACE");
    if (!name)
        return;
    FILE *f = fopen(name, "a");
    assert(f);
    fprintf(f, "{\"moduli\":[%llu,%llu],\"an\":%zu,\"bn\":%zu,\"rn\":%zu,\"accepted\":%s}\n",
            (unsigned long long)p[0], (unsigned long long)p[1], an, bn, rn, ok ? "true" : "false");
    assert(!fclose(f));
}
size_t up(size_t n, size_t a) {
    return (n + a - 1) & ~(a - 1);
}
struct Runtime {
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    sbn3_lease control{}, tables{}, work{}, values{};
    explicit Runtime(size_t bytes, unsigned workers) {
        sbn3_error e{};
        const size_t stacks = sbn3_team_stack_virtual_bytes(workers),
                     resident = sbn3_team_stack_resident_bytes(workers);
        sbn3_arena_config c{up(bytes + stacks + (4u << 20), 4096), up(bytes + resident + (4u << 20), 4096)};
        assert(sbn3_arena_create(&c, &arena, &e) == SBN3_OK);
        assert(sbn3_arena_prepare(arena, 0, sbn3_team_storage_bytes(), &e) == SBN3_OK);
        sbn3_arena_acquire(arena, 0, sbn3_team_storage_bytes(), &control);
        sbn3_team_config t{};
        t.workers = workers;
        t.stack_offset = 65536;
        for (int &v : t.cpu_ids)
            v = -1;
        assert(sbn3_team_create(arena, &control, &t, &team, &e) == SBN3_OK);
    }
    sbn3_lease region(size_t at, size_t bytes) {
        sbn3_error e{};
        assert(sbn3_arena_prepare(arena, at, bytes, &e) == SBN3_OK);
        sbn3_lease l{};
        sbn3_arena_acquire(arena, at, bytes, &l);
        return l;
    }
    size_t aligned(size_t at, size_t a) const {
        const size_t base = reinterpret_cast<size_t>(control.data);
        return up(base + at, a) - base;
    }
    size_t begin() const {
        return aligned(65536 + sbn3_team_stack_virtual_bytes(sbn3_team_workers(team)) + 4096, 65536);
    }
    ~Runtime() {
        for (auto *l : {&values, &work, &tables})
            if (l->data)
                sbn3_arena_release(arena, l);
        sbn3_team_destroy(team);
        sbn3_arena_release(arena, &control);
        sbn3_arena_destroy(arena);
    }
};
Integer witness_mul(const Integer &a, const Integer &b) {
    sbn3_mul_options o{};
    o.workers = 1; // The caller may already be pinned by a live test fixture.
    o.borrow_output = 1;
    sbn3_product_spec spec{a.d.size(), b.d.size()};
    sbn3_mul_plan p{};
    sbn3_mul_info i{};
    assert(sbn3_mul_query(&spec, &o, &p, &i) == SBN3_SUPPORTED);
    const size_t sa = up(a.d.size() + 8, 8), sb = up(b.d.size() + 8, 8),
                 sr = up(a.d.size() + b.d.size() + 8, 8);
    Runtime rt(i.table_bytes + i.workspace_bytes + 8 * (sa + sb + sr) + 2 * i.workspace_alignment +
                   (2u << 20),
               i.workers);
    size_t at = rt.begin();
    rt.tables = rt.region(at, i.table_bytes);
    at = rt.aligned(at + i.table_bytes, i.workspace_alignment);
    rt.work = rt.region(at, i.workspace_bytes);
    at = rt.aligned(at + i.workspace_bytes, 64);
    rt.values = rt.region(at, 8 * (sa + sb + sr));
    auto *A = static_cast<U *>(rt.values.data), *B = A + sa, *R = B + sb;
    memcpy(A, a.d.data(), a.d.size() * 8);
    memcpy(B, b.d.data(), b.d.size() * 8);
    sbn3_mul_binding *bound = nullptr;
    sbn3_mul_bind(&p, rt.arena, &rt.tables, &rt.work, rt.team, &bound);
    sbn3_mul_execute(bound, {A, a.d.size()}, {B, b.d.size()}, {R, a.d.size() + b.d.size()});
    sbn3_mul_unbind(bound);
    assert(ref_product_equal(a.d.data(), a.d.size(), b.d.data(), b.d.size(), R, a.d.size() + b.d.size()));
    Integer r;
    r.d.assign(R, R + a.d.size() + b.d.size());
    r.negative = a.negative != b.negative;
    r.trim();
    return r;
}
Integer multiply(const Integer &a, const Integer &b) {
    if (std::min(a.d.size(), b.d.size()) <= 4 || a.d.size() * b.d.size() <= 4096)
        return exact_mul(a, b);
    return witness_mul(a, b);
}
Integer newton(sbn3_newton_kind kind, size_t n, const Integer &a, const Integer &d) {
    sbn3_newton_options o{1, 0, 0, 0};
    sbn3_newton_plan p{};
    sbn3_newton_info i{};
    assert(sbn3_newton_query(kind, n, &o, &p, &i) == SBN3_SUPPORTED);
    const size_t stride = up(n + 9, 8);
    Runtime rt(i.storage_bytes + 3 * stride * 8 + 2 * i.storage_alignment + (2u << 20), i.workers);
    size_t at = rt.aligned(rt.begin(), i.storage_alignment);
    sbn3_error e{};
    assert(sbn3_arena_prepare(rt.arena, at, i.storage_bytes, &e) == SBN3_OK);
    rt.values = rt.region(rt.aligned(at + i.storage_bytes, 64), 3 * stride * 8);
    auto *A = static_cast<U *>(rt.values.data), *D = A + stride, *R = D + stride;
    assert(a.d.size() <= n + 1 && d.d.size() <= n);
    if (!a.d.empty())
        memcpy(A, a.d.data(), a.d.size() * 8);
    if (!d.d.empty())
        memcpy(D, d.d.data(), d.d.size() * 8);
    sbn3_newton_binding *b = nullptr;
    sbn3_newton_bind(&p, rt.arena, at, rt.team, &b);
    sbn3_newton_inputs in{{A, n + 1}, {D, n}, 2};
    sbn3_newton_execute(b, &in, {R, n + 1});
    sbn3_newton_unbind(b);
    Integer r;
    r.d.assign(R, R + n + 1);
    r.trim();
    return r;
}
Integer divide(const Integer &, const Integer &);
Integer certified_divide(const Integer &a, const Integer &b) {
    unsigned shift = std::countl_zero(b.d.back());
    size_t n = std::max(b.d.size(), a.d.size() - b.d.size() + 1);
    Integer denominator = shl(shl(b, shift), 64 * (n - b.d.size())),
            numerator = shr(shl(a, shift), 64 * b.d.size());
    Integer q = newton(SBN3_NEWTON_DIVIDE, n, numerator, denominator);
    Integer rem = sub(a, multiply(q, b));
    unsigned corrections = 0;
    while (rem.negative) {
        assert(++corrections <= 8);
        q = sub(q, Integer(1));
        rem = add(rem, b);
    }
    while (magnitude(rem, b) >= 0) {
        assert(++corrections <= 8);
        q = add(q, Integer(1));
        rem = sub(rem, b);
    }
    assert(!rem.negative && magnitude(rem, b) < 0);
    return q;
}
Integer divide(const Integer &a, const Integer &b) {
    assert(!a.negative && !b.negative && !b.d.empty());
    if (magnitude(a, b) < 0)
        return {};
    if (b.d.size() == 1) {
        Integer q;
        q.d.resize(a.d.size());
        U rem = 0;
        for (size_t i = a.d.size(); i--;) {
            W z = (W(rem) << 64) | a.d[i];
            q.d[i] = U(z / b.d[0]);
            rem = U(z % b.d[0]);
        }
        q.trim();
        return q;
    }
    if (a.d.size() * b.d.size() > 1048576)
        return certified_divide(a, b);
    unsigned s = std::countl_zero(b.d.back());
    Integer v = shl(b, s), u = shl(a, s);
    const size_t n = b.d.size(), m = a.d.size() - n;
    u.d.resize(a.d.size() + 1);
    Integer q;
    q.d.resize(m + 1);
    for (size_t j = m + 1; j--;) {
        W top = (W(u.d[j + n]) << 64) | u.d[j + n - 1];
        W guess = top / v.d[n - 1], rem = top % v.d[n - 1];
        if (guess > UINT64_MAX) {
            guess = UINT64_MAX;
            rem = top - guess * v.d[n - 1];
        }
        while (rem < (W(1) << 64) && guess * v.d[n - 2] > (rem << 64) + u.d[j + n - 2]) {
            --guess;
            rem += v.d[n - 1];
        }
        U borrow = 0;
        for (size_t k = 0; k < n; ++k) {
            W t = guess * v.d[k] + borrow;
            U old = u.d[j + k];
            u.d[j + k] = old - U(t);
            borrow = U(t >> 64) + (old < U(t));
        }
        U old = u.d[j + n];
        u.d[j + n] -= borrow;
        if (old < borrow) {
            --guess;
            U carry = 0;
            for (size_t k = 0; k < n; ++k) {
                W t = W(u.d[j + k]) + v.d[k] + carry;
                u.d[j + k] = U(t);
                carry = U(t >> 64);
            }
            u.d[j + n] += carry;
        }
        q.d[j] = U(guess);
    }
    q.trim();
    return q;
}
Integer sqrt_floor(const Integer &a) {
    assert(!a.negative);
    if (a.d.empty())
        return {};
    if (a.bits() > 4096) {
        const size_t bits = a.bits() - 1;
        assert(bits % 128 == 1 && magnitude(a, shl(Integer(1), bits)) == 0);
        Integer q = newton(SBN3_SQRT2_RSQRT, bits / 128, {}, {});
        Integer square = multiply(q, q);
        assert(magnitude(square, a) <= 0);
        Integer high = add(square, add(shl(q, 1), Integer(1)));
        assert(magnitude(a, high) < 0);
        return q;
    }
    Integer x = shl(Integer(1), (a.bits() + 1) / 2);
    for (;;) {
        Integer y = shr(add(x, divide(a, x)), 1);
        if (magnitude(y, x) >= 0)
            return x;
        x = std::move(y);
    }
}
} // namespace
extern "C" int ref_prime61(U p) {
    if (p < (U(1) << 60) || p >= (U(1) << 61) || !(p & 1))
        return 0;
    constexpr U bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    U d = p - 1;
    unsigned s = std::countr_zero(d);
    d >>= s;
    for (U a : bases) {
        if (p % a == 0)
            return 0;
        U x = power(a, d, p);
        if (x == 1 || x == p - 1)
            continue;
        bool pass = false;
        for (unsigned i = 1; i < s; ++i) {
            x = modmul(x, x, p);
            if (x == p - 1) {
                pass = true;
                break;
            }
        }
        if (!pass)
            return 0;
    }
    return 1;
}
extern "C" U ref_mod_words(const U *a, size_t n, U p) {
    assert(p && (p & 1) && p < (U(1) << 61));
    U inv = 1;
    for (unsigned j = 0; j < 6; ++j)
        inv *= 2 - p * inv;
    inv = 0 - inv;
    auto step = [&](U r, U word) {
        W v = W(r) + word;
        U m = U(v) * inv;
        U next = U((v + W(m) * p) >> 64);
        return next >= p ? next - p : next;
    };
    const U base = U((W(1) << 64) % p);
    if (n < 64) {
        U r = 0;
        for (size_t j = 0; j < n; ++j)
            r = step(r, a[j]);
        return modmul(r, power(base, n, p), p);
    }
    // Four contiguous chunks have independent Montgomery recurrences.
    // Restoring each chunk's global radix exponent combines them exactly.
    const size_t chunk = n / 4;
    U r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    for (size_t j = 0; j < chunk; ++j) {
        r0 = step(r0, a[j]);
        r1 = step(r1, a[chunk + j]);
        r2 = step(r2, a[2 * chunk + j]);
        r3 = step(r3, a[3 * chunk + j]);
    }
    for (size_t j = 4 * chunk; j < n; ++j)
        r3 = step(r3, a[j]);
    U b1 = power(base, chunk, p), b2 = modmul(b1, b1, p), b3 = modmul(b2, b1, p);
    return (modmul(r0, b1, p) + modmul(r1, b2, p) + modmul(r2, b3, p) + modmul(r3, power(base, n, p), p)) % p;
}
extern "C" int ref_product_equal(const U *a, size_t an, const U *b, size_t bn, const U *r, size_t rn) {
    if (!an || !bn) {
        for (size_t j = 0; j < rn; ++j)
            if (r[j])
                return 0;
        return 1;
    }
    if (std::min(an, bn) <= 4 || an * bn <= 4096) {
        Integer x, y, z;
        x.d.assign(a, a + an);
        y.d.assign(b, b + bn);
        if (rn)
            z.d.assign(r, r + rn);
        x.trim();
        y.trim();
        z.trim();
        return magnitude(exact_mul(x, y), z) == 0;
    }
    U p[2];
    challenges(p);
    bool ok = true;
    for (U q : p)
        ok &= modmul(ref_mod_words(a, an, q), ref_mod_words(b, bn, q), q) == ref_mod_words(r, rn, q);
    trace(p, an, bn, rn, ok);
    return ok;
}
extern "C" void ref_init(ref_number *p) {
    p->value = new Integer;
}
extern "C" void ref_clear(ref_number *p) {
    delete static_cast<Integer *>(p->value);
    p->value = nullptr;
}
extern "C" void ref_inits(ref_number *p, ...) {
    va_list a;
    va_start(a, p);
    while (p) {
        ref_init(p);
        p = va_arg(a, ref_number *);
    }
    va_end(a);
}
extern "C" void ref_clears(ref_number *p, ...) {
    va_list a;
    va_start(a, p);
    while (p) {
        ref_clear(p);
        p = va_arg(a, ref_number *);
    }
    va_end(a);
}
extern "C" void ref_import(ref_number *p, size_t count, int order, size_t size, int endian, size_t nails,
                           const void *data) {
    assert(order == -1 && (size == 1 || size == 8) && endian == 0 && !nails);
    Integer r;
    r.d.resize((count * size + 7) / 8);
    if (count)
        memcpy(r.d.data(), data, count * size);
    r.trim();
    get(p) = std::move(r);
}
extern "C" void ref_export(void *out, size_t *count, int order, size_t size, int endian, size_t nails,
                           const ref_number *p) {
    assert(out && order == -1 && (size == 1 || size == 8) && endian == 0 && !nails);
    const auto &a = get(p);
    size_t n = (a.bits() + 8 * size - 1) / (8 * size);
    if (n)
        memcpy(out, a.d.data(), n * size);
    if (count)
        *count = n;
}
extern "C" void ref_set(ref_number *r, const ref_number *a) {
    get(r) = get(a);
}
extern "C" void ref_set_ui(ref_number *r, U a) {
    get(r) = Integer(a);
}
extern "C" U ref_get_ui(const ref_number *a) {
    return get(a).d.empty() ? 0 : get(a).d[0];
}
extern "C" int ref_cmp(const ref_number *a, const ref_number *b) {
    return compare(get(a), get(b));
}
extern "C" int ref_cmp_ui(const ref_number *a, U b) {
    return compare(get(a), Integer(b));
}
extern "C" int ref_sgn(const ref_number *a) {
    return get(a).d.empty() ? 0 : get(a).negative ? -1 : 1;
}
extern "C" int ref_fits_ulong_p(const ref_number *a) {
    return !get(a).negative && get(a).d.size() <= 1;
}
extern "C" size_t ref_sizeinbase(const ref_number *a, unsigned base) {
    assert(base == 2);
    return std::max(size_t(1), get(a).bits());
}
extern "C" void ref_neg(ref_number *r, const ref_number *a) {
    auto x = get(a);
    if (!x.d.empty())
        x.negative = !x.negative;
    get(r) = std::move(x);
}
extern "C" void ref_abs(ref_number *r, const ref_number *a) {
    auto x = get(a);
    x.negative = false;
    get(r) = std::move(x);
}
extern "C" void ref_add(ref_number *r, const ref_number *a, const ref_number *b) {
    get(r) = add(get(a), get(b));
}
extern "C" void ref_sub(ref_number *r, const ref_number *a, const ref_number *b) {
    get(r) = sub(get(a), get(b));
}
extern "C" void ref_mul(ref_number *r, const ref_number *a, const ref_number *b) {
    get(r) = multiply(get(a), get(b));
}
extern "C" void ref_addmul(ref_number *r, const ref_number *a, const ref_number *b) {
    get(r) = add(get(r), multiply(get(a), get(b)));
}
extern "C" void ref_add_ui(ref_number *r, const ref_number *a, U b) {
    get(r) = add(get(a), Integer(b));
}
extern "C" void ref_sub_ui(ref_number *r, const ref_number *a, U b) {
    get(r) = sub(get(a), Integer(b));
}
extern "C" void ref_mul_ui(ref_number *r, const ref_number *a, U b) {
    get(r) = exact_mul(get(a), Integer(b));
}
extern "C" void ref_mul_2exp(ref_number *r, const ref_number *a, size_t n) {
    get(r) = shl(get(a), n);
}
extern "C" void ref_tdiv_q_2exp(ref_number *r, const ref_number *a, size_t n) {
    get(r) = shr(get(a), n);
}
extern "C" void ref_fdiv_q_2exp(ref_number *r, const ref_number *a, size_t n) {
    auto q = shr(get(a), n);
    if (get(a).negative && !lower(get(a), n).d.empty())
        q = sub(q, Integer(1));
    get(r) = std::move(q);
}
extern "C" void ref_fdiv_r_2exp(ref_number *r, const ref_number *a, size_t n) {
    auto q = lower(get(a), n);
    if (get(a).negative && !q.d.empty())
        q = sub_abs(shl(Integer(1), n), q);
    get(r) = std::move(q);
}
extern "C" void ref_fdiv_q(ref_number *r, const ref_number *a, const ref_number *b) {
    assert(!get(a).negative && !get(b).negative);
    get(r) = divide(get(a), get(b));
}
extern "C" void ref_mod(ref_number *r, const ref_number *a, const ref_number *b) {
    assert(!get(b).negative && !get(b).d.empty());
    // For word-aligned 2^k +/- 1, exact reduction needs only signed folds.
    // This also avoids asking the production divider to produce a witness.
    const auto &modulus = get(b);
    const auto &input = get(a);
    bool minus = std::all_of(modulus.d.begin(), modulus.d.end(), [](U w) { return w == UINT64_MAX; });
    bool plus = modulus.d.size() > 1 && modulus.d.front() == 1 && modulus.d.back() == 1 &&
                std::all_of(modulus.d.begin() + 1, modulus.d.end() - 1, [](U w) { return w == 0; });
    if (minus || plus) {
        const size_t width = modulus.d.size() - unsigned(plus);
        Integer rem;
        for (size_t off = 0, block = 0; off < input.d.size(); off += width, ++block) {
            Integer part;
            part.d.assign(input.d.begin() + off, input.d.begin() + std::min(input.d.size(), off + width));
            part.trim();
            rem = plus && (block & 1) ? sub(rem, part) : add(rem, part);
            if (rem.negative)
                rem = add(rem, modulus);
            else if (magnitude(rem, modulus) >= 0)
                rem = sub_abs(rem, modulus);
        }
        if (input.negative && !rem.d.empty())
            rem = sub_abs(modulus, rem);
        get(r) = std::move(rem);
        return;
    }
    auto x = get(a);
    x.negative = false;
    auto q = divide(x, get(b));
    auto rem = sub_abs(x, multiply(q, get(b)));
    if (get(a).negative && !rem.d.empty())
        rem = sub_abs(get(b), rem);
    get(r) = std::move(rem);
}
extern "C" U ref_fdiv_ui(const ref_number *a, U p) {
    assert(p);
    U r = 0;
    for (size_t i = get(a).d.size(); i--;)
        r = U(((W(r) << 64) | get(a).d[i]) % p);
    return get(a).negative && r ? p - r : r;
}
extern "C" void ref_divexact_ui(ref_number *r, const ref_number *a, U b) {
    assert(!ref_fdiv_ui(a, b));
    auto x = get(a);
    bool neg = x.negative;
    x.negative = false;
    auto q = divide(x, Integer(b));
    q.negative = neg && !q.d.empty();
    get(r) = std::move(q);
}
extern "C" void ref_sqrt(ref_number *r, const ref_number *a) {
    get(r) = sqrt_floor(get(a));
}
extern "C" U ref_add_n(U *r, const U *a, const U *b, size_t n) {
    U c = 0;
    for (size_t i = 0; i < n; ++i) {
        W x = W(a[i]) + b[i] + c;
        r[i] = U(x);
        c = U(x >> 64);
    }
    return c;
}
extern "C" U ref_sub_n(U *r, const U *a, const U *b, size_t n) {
    U c = 0;
    for (size_t i = 0; i < n; ++i) {
        W x = W(b[i]) + c;
        U old = a[i];
        r[i] = old - U(x);
        c = W(old) < x;
    }
    return c;
}
extern "C" U ref_mul_1(U *r, const U *a, size_t n, U b) {
    U c = 0;
    for (size_t i = 0; i < n; ++i) {
        W x = W(a[i]) * b + c;
        r[i] = U(x);
        c = U(x >> 64);
    }
    return c;
}

extern "C" int ref_mac2_equal(const U *a, size_t an, const U *b, size_t bn, const U *c, size_t cn, const U *d,
                              size_t dn, const U *r, size_t rn) {
    U p[2];
    challenges(p);
    bool ok = true;
    for (U q : p) {
        U first = modmul(ref_mod_words(a, an, q), ref_mod_words(b, bn, q), q);
        U second = modmul(ref_mod_words(c, cn, q), ref_mod_words(d, dn, q), q);
        ok &= (first + second) % q == ref_mod_words(r, rn, q);
    }
    trace(p, an + cn, bn + dn, rn, ok);
    return ok;
}
