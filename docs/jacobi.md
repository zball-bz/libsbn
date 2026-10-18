# Jacobi symbol

`sbn3/jacobi.h` evaluates `(a/n)` in `{-1,0,1}` for a signed integer numerator
and a positive odd denominator. `(a/1)=1`, including `(0/1)`; a nontrivial
common factor gives zero. Kronecker's extension to even/negative denominators
is not part of this API. Calls are variable-time.

The public entries are `sbn3_jacobi_u64`, unsigned-array `sbn3_jacobi`, and
signed-view `sbn3_int_jacobi`. Each declared input may contain up to 2^20
little-endian 64-bit limbs; leading zeros are allowed. Values belong to the
caller and remain unchanged. The two input views may overlap. Scratch must be
disjoint from both and obey the query's alignment. There is no output array.

`*_scratch_bytes` and `*_query` are pure length/worker queries. Storage equals
the corresponding ordinary-GCD quote. A shorter operand of at most two limbs
needs no scratch; ordinary small calls need only two private operand buffers.
Large calls borrow a caller's existing team (or a stack serial scope), Frame,
and the existing multiplication backends. No allocation, reservation, page
operation, input spectrum, or operation-specific prepared object is hidden in
an arithmetic call. The extra immutable symbol table is 512 bytes and is
included in `published_constants_bytes`.

## Reduction and symbol state

The word terminal encodes odd values as `(x-1)/2`; factors of two and
reciprocity update one sign parity. Two-word inputs use directional binary
reduction with shifts bounded below 64 bits on the common path, then the
word terminal. These terminal schedules adapt GNU MP `jacobi_2.c`; its
original notice and source identity are retained.
A two-limb denominator uses canonical remainder reduction. An odd word
denominator uses GCD's four-stream inverse reduction: its residue is the
negative of the input times a power of 2^64 modulo the denominator. That power
is a square, so only the `(-1/n)` factor needs correcting. This avoids a serial
full-remainder pass. A longer denominator first
strips numerator twos and applies reciprocity, then uses the existing streaming
remainder. It never retains an original-size quotient.

The small kernel shares GCD's scalar/IFMA matrix application, including the
persistent radix-2^52 representation. HGCD shares its arithmetic, split rules,
parallel product batches, middle products, and terminal storage. It has a
separate compile-time control instantiation: ordinary GCD contains no symbol
callback or per-quotient symbol branch.

Following [Möller's quotient-direction algorithm](https://arxiv.org/abs/1907.07795),
the observer retains sign parity, the two complete operands modulo four, and
the last reduction direction. It consumes only committed quotient residues
modulo four. Low bits are never read from truncated HGCD prefixes. A final
matrix modulo four or eight cannot replace this state: quotient history can
carry an additional sign.

The head observer attaches to accepted matrix updates, excluding trial
subtractions discarded at a stop. Exact fallback division records the final
quotient after stop correction. A rejected middle-product reduction restores
the symbol state with its matrix; accepted first children retain their state.
The streamed leaf producer owns this state. Its matrix consumer only composes
matrices and never reads or writes it. Matrix arithmetic and symbol state thus
have the same commit boundaries without inter-worker sharing of the state.

A matrix-only child is safe to lift when its existing integer residual
certificate accepts: the composed matrix is a product of nonnegative
Euclidean elementary matrices and its final residual pair is positive, so the
intermediate full-value reductions are positive too. The quotient rule admits
partial quotients; it does not require every quotient to be maximal.

## Verification

Production tests use an independent binary-Jacobi reference, word/state
exhaustion, signed and unbalanced operands, leading zeros, unit denominators,
non-coprime cases, exact scratch bounds, preserved inputs, and a caller team.
The allocation watcher covers whole calls. The C header also has a plain C
caller test. Experimental GMP comparisons and fresh scaling evidence live in
the separate research checkout; the library and tests do not depend on it or
on GMP.
