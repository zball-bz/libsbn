# GCD and extended GCD

`sbn3/gcd.h` provides ordinary variable-time GCD for unsigned little-endian
magnitudes and extended GCD for signed values. The general domain is currently
at most 2^20 declared limbs per operand; the direct small entry supports 1,024.
Empty values and leading zero words are allowed; gcd(0,0)=0. These are not
constant-time cryptographic APIs.

## General and extended calls

```c
sbn3_gcd_options options = {1};
sbn3_gcd_info info;
if (sbn3_gcd_query(an, bn, &options, &info) == SBN3_SUPPORTED) {
    /* Caller provides g[info.gcd_capacity] and aligned scratch. */
    size_t gn = sbn3_gcd(g, a, an, b, bn, scratch, info.scratch_bytes, NULL);
}
```

`sbn3_gcd_scratch_bytes(an,bn,workers)` is the thin size-only query; SIZE_MAX
means unsupported. Ordinary calls in the small domain dispatch directly to the
small kernel. Larger calls use recursive half-GCD, stop-aware Lehmer leaves,
native multiplication and exact bounded quotient corrections. Highly unequal
inputs first use a streaming remainder. The local division backend can discard
quotient blocks and prepare the divisor once for the whole numerator, avoiding
an extra whole-input quotient and repeated chunk preparation. Optional faster
recipes are admitted against their complete workspace quote before execution;
the generic fallback fits the GCD's existing bound. No input-size grid is
searched by GCD.
Common low zero limbs are cancelled through read-only input views. Their
radix power is restored only in the GCD output; Bezout coefficients are
unchanged. A high prefix that is equal or already below the reduction stop is
rejected before copying it or preparing a middle product. Single-word reduction
matrices reuse the exact IFMA/scalar Lehmer application kernel when their
coefficients fit its u63 contract.

Cyclic and middle matrix dot products request their two-term coefficient bound
during basis selection. A one-product certificate is insufficient: a prime
set that is valid for MUL can miss the accumulation bit required by MAC2.
Middle geometry also enforces its permitted period range before choosing a
basis, so an inadmissible candidate does not hide another supported basis.
The final value-only reduction targets the remaining Lehmer size directly;
being a few limbs above that size does not require another full half reduction.
The outer ordinary-GCD loop processes the upper half; serial extended GCD
processes the upper two thirds, and multiworker extended GCD uses half to
reduce live workspace at similar speed. These are fixed choices per output
and worker class, not an input-size search. The recursive HGCD split is
separate from this outer chop.
At eligible sizes, ordinary and extended GCD compute only the child matrix and a
certified residual lower bound. A coefficient middle window selects the next
reduction; its start follows the matrix's actual support, including carry and
alignment guards. The unchanged first matrix's spectra can serve both
that window and composition. The cache is local to one node and admitted with
the second child's full storage bound. Smaller or unsupported episodes retain
the conventional reducer. These choices do not enlarge the storage query.
The same recurrence is used by serial extended GCD. Bounded exact
residuals are recovered with cyclic products where supported. Dead working
prefixes are released before that application. A fused residual pair shares
its two source spectra across both rows. The Flat NTT implementation streams
coefficient rows through two further buffers and uses the existing convolution
kernel. A bounded coefficient bias, divisible by `2^T-1`, keeps CRT unsigned;
its integer contribution vanishes in the minus ring. Unsupported geometries
retain the established pair of products. All variants preserve the public
scratch quote and perform only query-admitted arithmetic.

The recursive children remain dependent. Parallelism is inside matrix episodes
and independent products. Eligible matrices use the native FFT's fused DOT2
and middle-window capabilities, with a separate accumulated-product numerical
envelope. The 16–20-bit codecs share the existing published root bank. Their
cached first-matrix arrays are directly executable, so no second unpacked copy
is reserved. A full matrix uses at most four FFT workers; one row or a residual
pair uses at most two. These counts follow the independent inverse/output
streams. The same matrix batch grain used by independent products decides
when a small composition can recruit workers.

Other medium compositions retain the seven-product algorithm, with immutable
operand combinations until the jobs join. Residual
updates may similarly batch four products before either input tail is changed.
At most four workers execute these serial backend recipes concurrently; this
preserves the fast serial FFT choices. A fixed 256-limb combined-length grain
and complete scratch admission decide whether to batch. Smaller episodes retain
the in-place schedule. Large fused NTT matrix episodes retain shared transforms;
after all primes finish reading inputs, their workspace is reused by a full-team
CRT pass over output entries and chunks, with separate carry journals. These
schedules use the existing public scratch bound and have no persistent job queue.
Extended GCD defers the outer Bezout reconstruction. It retains normalized
stage matrices, emits the GCD, then evaluates only the second column of their
product in reverse order. The two dead operand arrays hold that column. This
replaces repeated updates of an original-size four-entry accumulator with
matrix-vector products. Eligible positive linear columns use six forwards
and two inverses; other geometries use ordinary multiplication pairs. FFT
forward scheduling counts live, uncached inputs rather than eight nominal
slots, including in cached middle products.

When both reconstruction components are retained and the normalized operands
have equal limb lengths, one exact seed step precedes HGCD. Its quotient fits
one unsigned word, or a certified Lehmer block supplies word coefficients.
The seed matrix uses fixed stack storage and is applied after the history.
It does not consume the eager-suffix allowance when that first step removes no
whole limb. Single-row roots retain their smaller admission contract and skip
this seed. Equal residuals finish directly.

History has at most 64 descriptors in fixed stack storage. Before retaining
each matrix, the Frame must admit both an eager suffix's complete storage and
the later column reconstruction. Otherwise that suffix finishes eagerly in
the existing space. No geometric shrinkage assumption is needed for storage
safety, and no failed allocation or retry selects the fallback. The public
scratch quote is unchanged; reduced live usage does not imply a smaller
caller reservation.

For Bezout coefficients, query `sbn3_exgcd_query(an,bn,mask,&options,&info)` and
call `sbn3_int_exgcd(&g,&x,&y,a,b,scratch,bytes,team)`. The signed inputs use
`sbn3_int_view`; outputs use caller-owned `sbn3_int` buffers. The equation is
`x*a+y*b=g`, with nonnegative g and normalized coefficient signs. Either x or y
may be NULL, selected independently with SBN3_EXGCD_X / SBN3_EXGCD_Y in the query.
An optional-output root can use a one-row eager fallback and a half-prefix
reduction to fit its smaller quote. Retained descendant matrices remain
complete because either final coefficient generally depends on both of their
rows. An initial streamed quotient is retained only if it is needed for a
requested coefficient. Any valid Bezout pair may be returned; minimal
or unique coefficients, or agreement with GMP's choice, are not promised.

The unconditional capacities are max(an,bn) for g, max(bn,1) for x and max(an,1)
for y. Query also reports the decomposition of scratch into mutable values,
cofactor rows, recursive temporaries and arithmetic workspace. The bound includes
all representation conversion and arithmetic preparation. Published root banks
are reported separately from private scratch. Fixed stack temporaries and caller
values must be counted separately when budgeting an application.

With both inputs zero, all outputs are zero. With only one nonzero input, its
coefficient is its sign and the other coefficient is zero. Inputs are preserved;
all writable spans are disjoint from each other and from inputs. Read-only
inputs may overlap. Scratch has the query's alignment and full byte count.
Arithmetic does not allocate, reserve pages, expand scratch, or create workers.
A one-worker call permits NULL team; parallel calls borrow an already-created,
idle caller team. Product phases use bounded scopes; their scratch dies after
the phase joins. The caller must keep its values and scratch live throughout.

Matrix-bearing leaves retain u64 operands. Their word-matrix application uses
four vector 32x32 products for each u64-by-u63 product; scalar updates handle
short rows and the final residual group. There is no u52 conversion in these
leaves. Signed residual carries and unsigned reconstruction carries remain
separate, exact operations. This private kernel is distinct from the ordinary
small-GCD implementation described below.

With spare workers, a bounded leaf can pipeline residual reduction with matrix
accumulation. The next certified head depends on updated residuals, not on the
accumulated matrix. Equal prefixes exit without dispatch. One initial reduction
runs synchronously, so one-step cases do not recruit a helper. One helper
consumes the remaining heads in order through a 64-entry
stream. Exact fallback drains pending heads; final drain and join precede the
parent's use of the matrix. This adds about 2.3 KiB of fixed stack storage and
does not enlarge caller scratch or parallelize dependent recursive children.

Internally, a reduction matrix satisfies `original = M * reduced`, has
nonnegative entries and determinant one. Exact residual application therefore
uses signed differences, while composing reconstruction matrices uses positive
sums. These have distinct carry contracts, including distinct IFMA carry
stitches. No truncated product is used for a quotient decision without a bound
covering its omitted carry. Matrix-only internal calls explicitly do not return
residual values; their scratch contents must not be reused as canonical values.
The middle-window route certifies positivity for every allowed omitted carry.
A declined window returns to a conventional subtree; it does not pretend that
a partial reduction met the parent's target. Carried matrix values are returned
to parents. No raw product spectrum is relabelled as canonical, and no
cross-node FFT-doubling capability or polynomial complexity constant is claimed.

## Direct small calls

```c
size_t bytes = sbn3_gcd_small_scratch_bytes(an, bn);
/* SIZE_MAX means this bounded implementation does not support the shape. */
size_t gn = sbn3_gcd_small(g, a, an, b, bn, caller_scratch);
```

Inputs remain unchanged. For two nonzero values, `min(an,bn)` output limbs
suffice; if one value is zero, the output must fit the other. `max(an,bn)` is
the unconditional bound. Only the returned `gn` limbs are written. No quotient
storage is needed, and a zero GCD permits a null output pointer.
Output and scratch are disjoint from each other and from both inputs; the
read-only inputs may overlap. Scratch is caller-owned and 8-byte aligned.
No plan, arena, team, allocation, page operation or cross-call cache is needed.
Normal builds trust these span/capacity preconditions; checked builds audit them.
Numerical carry/coefficient invariants remain fatal diagnostics.

The capacity query describes all variable workspace, including conversion and
fallback. Fixed register/stack temporaries are separate. Either operand having
at most two declared limbs needs zero caller scratch. Otherwise, with
`N=max(an,bn)`:

- Below 160 limbs, workspace is `16 * round_down(N+9,8)` bytes: two u64 work
  arrays, each with a guard and padding.
- From 160 limbs, workspace is
  `16 * round_up(ceil(64*(N+1)/52),8)` bytes: two persistent u52 work arrays.
  Conversion and exceptional division reuse those arrays in place.

For example, balanced 128-limb inputs need 2,176 bytes and balanced 1,024-limb
inputs need 20,224 bytes. Exceptional remainder steps discard quotient digits
directly, without output temporaries or a quotient array. All sizes up to
the supported limit fit within `(2.5*N+32)` u64 words of caller scratch.

`sbn3_gcd_u64(a,b)` is the direct scalar word entry. The two-word terminal tries
one Lehmer batch when both odd operands retain at least 97 significant bits and
their high words differ by at least two, then finishes with binary GCD. Accepted
matrix rows fit 128 bits, so unsigned low products recover the exact differences.
This needs no caller scratch. A long operand against one word uses GCD-equivalent
odd-modulus cancellation, with four independent chunks for sufficiently long
inputs and length-aware modular recombination. This scaled residue is private;
it is not exposed as a canonical remainder. A long operand against two words
uses a normalized streaming remainder and then the two-word GCD terminal.
The same short-divisor finish is used when an internal reduction reaches one or
two words; a general quotient/remainder step is not repeated at that point.

The multiword algorithm batches leading-word reductions into a small Lehmer
matrix. Exact hardware quotient estimates use a checked algebraic correction,
with a bitwise division fallback for large quotients. The ordinary implementation
uses fused scalar matrix updates below 48 active limbs and IFMA updates above
that. Inputs of at least 160 limbs first reduce in u64 until a productive wide
head is available, then convert once to persistent u52 and apply that already
computed matrix. A near-equal pair can finish with a short difference before
conversion; values that shrink below the crossover continue in u64.
These are two fixed measured crossovers; there is no per-input timing search.
Matrix coefficients too wide for IFMA take an exact scalar update. Vector carry
state stays in vector registers; endpoint carry chains use a bounded exact stitch.
The persistent u52 update fuses each signed row into IFMA accumulations:
with `B=2^52` and `e=(-d) mod B`, `c*x-d*y = c*x+e*y-[d!=0]*B*y`.
The low sum stays redundant until carry normalization. This removes two vector
subtractions per pair of rows and avoids many borrow chains in the intermediate
representation. The fast carry path accepts the corrected digits only when all
are in `[0,B)`; otherwise it uses the exact scalar stitch. Zero coefficients and
long carry/borrow chains do not rely on random-input assumptions. The private
helper is tested directly against scalar signed 128-bit products, including
cross-vector carry state and endpoint chains.

Common low zero words are scanned with SIMD when needed. Only the remaining
shifted values are copied, avoiding writes of regions immediately discarded.
All input normalization and representation conversions belong to the call.

`sbn3_gcd_u64_batch(g,a,b,n)` computes independent word pairs. It processes eight
pairs per SIMD group, handles arbitrary tails without overread, and needs no
scratch. Exact `g==a` or `g==b` is supported; other partial output overlap is not.
It is a throughput API, not a claim that SIMD accelerates one word GCD's
dependent iteration chain.

The implementation has no GMP library dependency. Its leading-word reducer and
odd-cancellation recurrence retain their GNU MP source notices; see the private
headers and `config/provenance/donors.json`. Numerical tests use a separate binary
reference, constructed common factors, exact-size buffers, allocation checks,
batch alias/tail tests and guard pages. The public header is tested from C with
baseline compiler flags. Experimental comparisons live in the ignored research
checkout and are not required to build or use the library.

Jacobi symbol evaluation reuses these arithmetic kernels and storage bounds
with a quotient-direction observer; see [jacobi.md](jacobi.md).
