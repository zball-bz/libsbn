#pragma once

namespace sbn::v3::bbp {
// Word-wise CIOS shared by the IFMA lane island and test-only scalar radices.
// O supplies unsigned digits in a wider accumulator, and independent lane masks.
// Inputs and outputs are canonical [0,m), m odd, 1 < m < B^L. No allocation.
template <class O, unsigned L> struct WordMontgomery {
    using V = typename O::V;
    using M = typename O::M;
    V m[L], inverse;

    explicit WordMontgomery(const V (&modulus)[L]) {
        for (unsigned j = 0; j < L; ++j)
            m[j] = modulus[j];
        // For odd m, (3*m) XOR 2 is already an inverse modulo 32.
        // Four precision doublings suffice for u52; three for u32.
        V y = O::bit_xor(O::add(modulus[0], O::add(modulus[0], modulus[0])), O::splat(2));
        for (unsigned precision = 5; precision < O::bits; precision *= 2)
            y = O::lo(y, O::sub(O::splat(2), O::lo(m[0], y)));
        inverse = O::digit(O::sub(O::zero(), y));
    }

    // Input has L digits plus a carry and lies in [0,2m). Returns x >= m.
    M reduce(V (&x)[L], V top) const {
        V difference[L], borrow = O::zero();
        for (unsigned j = 0; j < L; ++j) {
            const V subtrahend = O::add(m[j], borrow);
            difference[j] = O::digit(O::sub(x[j], subtrahend));
            borrow = O::select(O::lt(x[j], subtrahend), O::one(), O::zero());
        }
        const M take = M(O::nonzero(top) | O::mask_not(O::nonzero(borrow)));
        for (unsigned j = 0; j < L; ++j)
            x[j] = O::select(take, difference[j], x[j]);
        return take;
    }

    // Each scan normalizes immediately: low + digit + carry < 3B,
    // high + carry(low sum) <= B. This fits u64 for both B=2^32 and 2^52.
    void multiply(V (&out)[L], const V (&a)[L], const V (&b)[L]) const {
        if constexpr (L == 1) {
            // Subtractive one-word REDC. With u=lo(a*b)*m^-1 (mod B),
            // the low digits cancel exactly, so there is no low carry:
            // (a*b-u*m)/B = hi(a*b)-hi(u*m), in (-m,m).
            const V low = O::lo(a[0], b[0]);
            const V high = O::hi(a[0], b[0]);
            const V u = O::lo(low, O::sub(O::zero(), inverse));
            const V correction = O::hi(u, m[0]);
            const V difference = O::sub(high, correction);
            out[0] = O::select(O::lt(high, correction), O::add(difference, m[0]), difference);
            return;
        }
        V t[L + 2];
        for (auto &v : t)
            v = O::zero();
        for (unsigned i = 0; i < L; ++i) {
            V carry = O::zero();
            for (unsigned j = 0; j < L; ++j) {
                const V v = O::add(O::add(t[j], carry), O::lo(a[j], b[i]));
                t[j] = O::digit(v);
                carry = O::add(O::hi(a[j], b[i]), O::carry(v));
            }
            V top = O::add(t[L], carry);
            t[L] = O::digit(top);
            t[L + 1] = O::carry(top);
            const V u = O::lo(t[0], inverse);
            carry = O::zero();
            for (unsigned j = 0; j < L; ++j) {
                const V v = O::add(O::add(t[j], carry), O::lo(u, m[j]));
                if (j)
                    t[j - 1] = O::digit(v);
                carry = O::add(O::hi(u, m[j]), O::carry(v));
            }
            top = O::add(t[L], carry);
            t[L - 1] = O::digit(top);
            t[L] = O::add(t[L + 1], O::carry(top));
        }
        for (unsigned j = 0; j < L; ++j)
            out[j] = t[j];
        reduce(out, t[L]);
    }

    // Returns the next binary fraction digit. Also used for masked pow2 steps.
    M double_mod(V (&x)[L]) const {
        if constexpr (L == 1) {
            const V doubled = O::add(x[0], x[0]); // <2B, fits the wider accumulator.
            const M take = O::mask_not(O::lt(doubled, m[0]));
            x[0] = O::select(take, O::sub(doubled, m[0]), doubled);
            return take;
        }
        V carry = O::zero();
        for (unsigned j = 0; j < L; ++j) {
            const V v = O::add(O::add(x[j], x[j]), carry);
            x[j] = O::digit(v);
            carry = O::carry(v);
        }
        return reduce(x, carry);
    }

    void masked_double(V (&x)[L], M active) const {
        if constexpr (L == 1) {
            const V v = O::add(x[0], O::select(active, x[0], O::zero()));
            x[0] = O::select(O::lt(v, m[0]), v, O::sub(v, m[0]));
            return;
        }
        V doubled[L];
        for (unsigned j = 0; j < L; ++j)
            doubled[j] = x[j];
        double_mod(doubled);
        for (unsigned j = 0; j < L; ++j)
            x[j] = O::select(active, doubled[j], x[j]);
    }

    void power_tail(V (&x)[L], V exponent, unsigned remaining) const {
        for (unsigned bit = remaining; bit--;) {
            multiply(x, x, x);
            masked_double(x, O::bit(exponent, bit));
        }
    }
};
} // namespace sbn::v3::bbp
