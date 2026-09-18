# Radix conversion

include/sbn3/radix.h supports bases 2..64. The input is a nonnegative dyadic
value X = M * 2^exponent2, with little-endian uint64 mantissa limbs; the caller
carries the sign. This covers integers and finite binary fractions.

Formatting returns most-significant-first digit bytes, with separate integer
and fractional regions. The decimal point is not stored. By default bytes are
digit values; a caller-provided alphabet maps them to characters.

- EXACT treats X as exact and returns certified truncated fractional digits.
- ENCLOSED treats X as the lower endpoint of [X, X + 2^exponent2) and returns
  only digits determined by the whole interval. It requires exponent2 <= 0.

Parsing returns M = floor(value * 2^fraction_bits), exactly. Invalid input reports
the first invalid position; it is not a successful arithmetic value. The header
specifies alignment, padding and output capacities.

The lifecycle is query, prepare arena, bind, execute, unbind. Bind prepares powers
and product programs. A radix binding can execute repeatedly for its declared
shape. The repeated option lets planning distinguish one-shot and repeated use.

See examples/radix.c for an executable caller. The conventional alphabet is
0-9 a-z through base 36, and 0-9 A-Z a-z + / for larger bases.
