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

One conversion plans once. The tree plan is far larger than a plan value, so bind
assembles it again; the plan value (4096 bytes) carries the sealed layout, the
Newton plan of the scaling reciprocal and a transcript with the winning candidate
of every product search the query ran. Bind repeats each winner's single backend
query instead of the search, checks every entry against its request and the
identity it stands for, and compares the identity of the whole assembly with the
sealed one; an entry that does not reproduce, or a transcript without room, is
planned by the search again. Program preparation, the rail squarings and the power
chain use the same recorded winners. Nothing is kept between calls: the transcript
is part of the caller's plan value. Scaling reciprocals of at most 864 limbs are
one schoolbook division at bind, cheaper than the complete Newton route there.
For one conversion (repeated = 0) the integer part goes to the tree from about
40 000 limb-words of schoolbook work (limbs x eight-digit words), in any base.
tests/integration/radix_plan_replay_test.cpp is the gate of these statements.

See examples/radix.c for an executable caller. The conventional alphabet is
0-9 a-z through base 36, and 0-9 A-Z a-z + / for larger bases.
