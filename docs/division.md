# Division

`include/sbn3/divrem.h` provides exact integer division, B=2^64:

- unsigned: N = Q·D + R with 0 <= R < D;
- signed (`sbn3_int_*`): truncation toward zero, |R| < |D|, R carries the sign
  of N, outputs normalized without negative zero.

A zero divisor is a fatal `SBN3_FATAL_ARGUMENT` on the value boundary
(`prepare` or the basecase entry). Inputs may carry leading zero limbs; the
quotient span is always written completely (nn-dn+1 limbs, high zeros
included) and the remainder span has exactly dn limbs. The result structure
reports the normalized lengths.

## Entries

| Entry | Use | Work |
|---|---|---|
| `sbn3_divrem_basecase`, `sbn3_int_divrem_basecase` | one-shot, no plan/arena/team; caller scratch of nn+dn+1 limbs | O(nn·dn) words: GMP-derived divrem_1/divrem_2/3-by-2 schoolbook |
| `sbn3_divrem_small_scratch_bytes`, `sbn3_divrem_small` | one-shot, 1..32768 divisor limbs, caller-owned scratch | word/schoolbook, two-vector/DC, local Barrett or fused short quotient; all preparation belongs to the call |
| `sbn3_divrem_query/bind/prepare/execute/unbind` | prepared divisor, repeated numerators, large shapes | threshold-selected local arithmetic, block Barrett or fused short quotient |
| `sbn3_divrem_prepare_into` then execute/unbind | construct the same reusable divisor directly in caller storage; selection and preparation are included | same policy and kernels, without a public intermediate plan |
| service with `algorithm=SBN3_DIVREM_DC` | explicit native D&C candidate, 3..2^20 divisor limbs | u52 division over 416-bit blocks; exact Q/R |

The D&C recipe uses caller-planned `Frame` storage and a single top-divisor
reciprocal for its leaves. Its quotient estimate computes only a guarded
upper product band; the additional under-estimate is below 2^-150 quotient
ulps and the remainder correction remains exact. It stores the highest
quotient block explicitly, avoiding an artificial zero dividend block.
Unlike block Barrett, this recipe converts the whole numerator into its
temporary u52 representation, so its scratch also depends on numerator
length. Query reports that complete requirement and never substitutes a
different recipe for an explicitly requested D&C plan.

## Small approximate quotient

`sbn3_newton_query(SBN3_NEWTON_DIVIDE, n, ...)` supports n>=1. For normalized
n-word D and n+1-word A with A[n]<=1, it returns n+1 words Q satisfying
`|Q-B^n*A/D|<3`. This is a quotient-only contract, not exact Q/R.

Below the existing local FFT product threshold, the default shares exact
division's word and two-vector u52/DC kernels. The scalar path first computes the high
quotient and remainder of `B^(n-1)*A / D`; it estimates the final word from
the existing top-two-word reciprocal, omitting that word's full divisor
update. Let R<D be the intermediate remainder, with leading pairs
`r=r1*B+r0` and `d=d1*B+d0`, and let `V=B+inverse=floor((B^3-1)/d)`.
The estimate is `p=floor(E)`, where `E=(r1*V+r0)/B`. Write
`delta=B^3/d-V`, so `0<delta<=1`. For `y=R*B/D`, truncation gives
`y-E < r1*delta/B + r0*(B^2/d-1)/B + B/d <= B^2/d <= 2`;
also `E<=B*r/d` and `B*r/d-y<2/B`. Hence `-3<p-y<1`, so `|p-y|<3`.
Equal top pairs saturate to B-1, whose distance is below
2 because D is normalized. For n=1 the one-word reciprocal gives an
underestimate less than 2 away. Existing exact primitives retain their
full corrections and remainder semantics.

The native path converts the virtual numerator `B^n*A` directly to u52,
without materializing its zero u64 half, and omits final remainder
denormalization/emission. It normally returns the exact floor. When a
one-bit prefix would precede complete 832-bit quotient steps, it instead
divides the exactly integral `B^n*A/2` and returns `2*floor(B^n*A/(2D))+1`.
This has error at most 1 and removes that extra prefix update. Selection
depends on quotient-bit geometry, not a fitted limb threshold. At the local
FFT product threshold, requests retain the half-precision inverse and bounded
product windows: forcing exact division's later DC/Barrett crossover onto
this quotient-only operation would do unnecessary work. Both paths use the
compact wrapper until the existing repeated-operand FFT capability requests
stored product records.

All variants read A before writing Q and permit Q==A; D remains unchanged.
Only queried scratch and caller output are used. The compact service skips
the general plan seal/lease bookkeeping in ordinary builds; checked builds
audit its caller-granted exclusive range. Full fresh timing still includes
query, bind, divisor preparation, execution and unbind.

## Local one-shot path

The thin entry takes the actual numerator length, a nonzero divisor top limb,
disjoint caller-owned values, and the queried scratch. It requires no arena
or team. Small cases use the word/IFMA threshold directly. Larger local cases
use three work thresholds on divisor length times quotient length to select
DC or Barrett. For quotient increment q=nn-dn and divisor length n, Barrett
uses one block when q<=n/3, otherwise at least two nearly equal blocks of at
most ceil(n/2) limbs. The integer ceiling uses ceil(n/2) explicitly, so odd
divisors do not add a spurious block to a 3n/2-limb quotient. There is no
cost table, candidate scoring or transform grid at this boundary. Long
quotients (at least four divisor lengths) and repeated divisors can fill up
to half the already-selected residual ring; this amortizes the larger
reciprocal without searching alternative plans. Scratch
query and execution independently make the same choice; the thin entry
retains no state.

Fresh short quotients use a fused recipe when dn>=8192 and
512<=nn-dn<=ceil(dn/2), unless an explicit recipe or reuse hint selects otherwise.
This is one size/ratio threshold, not a candidate search. The thin and compact
W1 entries use the same local fused kernels. Other shapes keep the block rules
above.

The automatic service uses these same preparation/execution kernels for native requests with
dn<=512, or dn<=32768 and one worker, including long numerators and repeated
divisors. A reuse hint of at least two allows one half-divisor-sized quotient
block where repeated applications amortize its reciprocal. Explicit recipe
options retain the general path described below. A bound
local plan also accepts shorter numerators with its original block and
transform geometry; it does not reselect a recipe during execution. The retained form prepares D once; the thin call constructs and consumes that state within its scratch lifetime.

Long quotients stream fixed-capacity blocks plus a tail, so their inverse
and workspace bounds do not grow with numerator length.

Local arithmetic uses one transient integer buffer: the quotient estimate
is retired into Q, then the same buffer receives D*qhat and is overwritten
with the signed residual. The dividend window is subtracted from its input
spans without materializing a second X buffer. Inverse preparation scratch
dies before the spectra are constructed. Longer numerators add blocks, not
a whole-numerator conversion or a larger reciprocal.

Local Barrett computes a bounded inverse of the divisor's top m limbs. With
U=B^m+u, the quotient estimate is `rtop + high(rtop*u)`, so the known leading
one does not enlarge a multiplication. Each call prepares the U and D
spectra once. When their transform shapes agree, they share one immutable
root-table allocation; their value spectra remain distinct. All tables and
spectra are released with the caller's scratch frame.

Up to four leading quotient words are handled directly when that removes a
tiny residual FFT block. The word reciprocal is shared across that prefix,
and a zero quotient word needs no divisor update. Other short tails may use
an IFMA high-prefix product: retaining two guard words of the inverse and
128 guard bits of convolution makes the combined underestimate at most one.
The existing exact correction budget covers it. Full blocks retain the cached
FFT, and the high-prefix temporary storage fits the already-planned workspace.

A cyclic residual may use a period r>=dn, without extra guard limbs. Let
M=B^r-1, E=X-qhat*D, and z be its computed residue in [0,M] (including the
redundant zero representation M). The exact low word
`ell=(X[0]-qhat[0]*D[0]) mod B` costs one word multiplication. Since M=-1 mod B,
`E=z+t*M` implies `t=sign_extend(z[0]-ell)`; the bounded quotient error proves
|t|<2^63 and |E|<B^(r+1)/2. Forming `z-t+t*B^r` in r+1 words therefore recovers
the exact signed residual, even when |E| exceeds M/2. Ordinary bounded
quotient/remainder correction then gives exact Q/R. The scratch bound includes
the extra word used for the lift. This is an exact recovery rule, not a
probabilistic checksum.

## Prepared and repeated-divisor path

For the fused short-quotient recipe, prepare normalizes and copies D and
computes a half-precision reciprocal. Execute forms a bounded quotient from a
short divisor prefix using the same terminal as approximate division, then
performs one full-divisor residual and at most eight exact corrections.
The extra quotient-capacity word is handled by a shifted word product so it
does not unnecessarily enlarge the residual transform.

The quotient terminal and final residual reuse one phase region. The inverse
scratch can occupy the future spectra and phase region during prepare. U's
spectrum is kept across the terminal's two uses only when it fits the peak
already needed by another phase; otherwise the terminal rebuilds it. D and
the half reciprocal remain valid for repeated execute calls. A reuse hint of
at least two keeps the amortized block-Barrett policy.

The remaining general path chooses a block length before materializing any product
plans. For qn quotient-capacity words, a short quotient (3*qn<=dn) uses one
block. Otherwise it uses ceil(qn/dn)+1 nearly equal blocks, keeping the
reciprocal smaller than a full divisor. Balanced 2n/n therefore uses the
three-block construction with a smooth-transform work target of 5/2 M(n).
That is a work model, not a finite-size timing guarantee.

Short products use u52. W1 linear products use the engine's fixed FFT recipe
in its admitted band. Parallel products and larger shapes use Flat NTT,
then Bailey. Cyclic residuals are enabled by a fixed width/size threshold,
or explicitly requested. A cyclic FFT is used only within its numeric
contract; otherwise a native ring is rounded up to cover the divisor.
A ring that cannot shorten the product falls back to a linear recipe.

Each NTT prime family contributes one legal geometry. Its maximum digit
width is determined by the CRT bound; a cyclic period can then tighten the
digit width without changing transform order. The choice uses active
trunk-prime volume, including TFT fill. Flat's N/8 metadata is not charged
as a Bailey column tower. Only the chosen producer/consumer pair is
materialized. There is no block-size, T, row/column, or cached/uncached
performance grid in division. Newton inverse rungs and quotient terminals
use the same fixed-geometry approach; rsqrt retains its separate policy.
Multiworker inverse/quotient cycles use Flat through a 2^18-word period and
packed Bailey above it. This smaller layout cutoff accounts for the live
cached-cycle workspace. The three integer reconstruction guard words do not
enlarge the transform period. W1 selects Flat through 2^21 transform coefficients per prime. The cutoff
uses physical transform size, so changing NP or digit width does not silently
change the per-prime size limit. Integer reconstruction guard words do not
participate in either layout cutoff. These are machine-profile limits, not a
runtime geometry search.

Bounded inverse/quotient windows may additionally use a smaller serial NTT ring
with an independently exact low-product witness. This does not change the
outer exact residual's modulus or its quotient/remainder contract. Both Newton
and the exact fused terminal size their integer result buffer by logical
precision as well as physical period, and propagate the same repair recipe.
The low products borrow idle planes rather than adding a separate scratch peak.

An exact fused terminal can use the same admitted compact FFT provider as
approximate division, consuming its already prepared half reciprocal. Spectrum
retention is selected from the existing phase-union and ordinary-terminal capacity, so avoiding a
forward transform does not enlarge that union merely to retain a cache. The
final exact residual and bounded Q/R corrections remain unchanged.

The reciprocal satisfies |U-B^(2m)/Dtop|<3. Division uses U=B^m+u to estimate
qhat=rtop+high(rtop*u), excluding the known leading one from the transform.
The residual is exact: the cyclic path uses the low-word lift described
above, so its ring only needs to cover dn. At most eight quotient
corrections are allowed; a violated bound is a fatal mathematical error.
An exact reciprocal multiply-back is unnecessary for this contract.

For one W1 quotient block with reuse_hint<=1, ordinary products avoid
retaining two operand spectra. Their temporary workspace is reused serially;
uncached general product bindings are created on first use, after inverse
preparation. Local plans retain only D/U and their root-table descriptions.
Multiple blocks, a reuse hint, and wider teams keep cached spectra. This is
a use-count/worker rule, not a comparison of candidate plans. All required
work remains inside the fresh call, and repeated execution remains legal.

Up to four leftover leading quotient words may use exact word division.
Their integer workspace is sized independently of spare ring limbs. A
provably zero normalized leading quotient word is emitted directly; the
remainder is read from its numerator view without multiplying D by zero or
copying a full X window.

## Resources and lifetime

`sbn3_divrem_info` reports the complete storage of a binding and its
components. Every service prepare performs the denominator-only work:
word division retains normalized D and its 3/2 reciprocal (a single-word
reciprocal for dn=1); IFMA/DC retains shifted u52 D and its two-vector
quotient reciprocal; local Barrett retains normalized D, its block inverse
and spectra. Thin calls and retained bindings share the private prepare/apply
kernels. Execute does not repeat these preparations. In the general Barrett
path the normalized divisor, block inverse and cached spectra persist
until the next `prepare` or `unbind`. Inverse scratch may overlap the
future spectra as well as product tables/workspaces: spectra are reserved
only after the inverse's leases have ended. Cached products with at least
128 KiB of mutable workspace share that workspace serially. Their immutable roots remain in their
spectra; switching rebinds their already compiled plans and codec constants.
Small products remain bound throughout. Estimate and residual outputs share
one integer buffer. The quotient is retained in caller Q before its product
buffer is overwritten, and caller R holds the normalized running remainder.
Parallel in-place denormalization snapshots chunk boundary words before
launching workers. Execution performs no allocation,
page operation, plan search
or algorithm switch. Extra workspace grows with the divisor, block and ring
lengths, not with the total quotient length: numerator blocks are read
through an in-flight shift.

Compact bindings contain at most 1 KiB of control state. They borrow the
caller's resident, exclusive range until unbind; release builds do not
register a second arena lease. Checked builds additionally audit leases
and aliases. The caller must not trim the pages, reuse the bytes or execute
concurrently through one binding. The public opaque plan retains its ABI
envelope, but unused tail bytes are neither initialized nor read and are
not a serialization format.

Binding resolves an execution function once, including the scalar divisor
width and instrumentation mode. Normal execute calls dispatch through that
function directly; they do not reinterpret the recipe or carry stack frames
for the other algorithms. Word divisors use this compact route regardless
of reuse_hint because that hint does not change their arithmetic recipe.

`prepare_into` accepts the same request/options (null options mean W1
defaults) and an explicit available byte count. It selects a recipe, binds
and prepares D without an intermediate public plan. Query remains available
for support/budget admission and storage provisioning; its result is not an
implicit input to this constructor. Execute, re-prepare and unbind have the
same contract as the phased interface.

Under a `memory_budget`, compact recipes try a fixed smaller-block/DC
fallback sequence. The general recipe halves the initial block at most
twice; it does not rank alternative plans. Bounded short-divisor/short-quotient
schoolbook fallbacks remain available. Rejection leaves the caller's plan
untouched and reports the smallest complete requirement encountered.
Explicit algorithm, block and residual constraints are preserved.

`prepare` may be repeated with another divisor of the same length; the plan
is reusable and a range may be rebound after `unbind`. `reuse_hint` may increase the retained block precision to amortize preparation;
`block_limbs` pins the block size for experiments. Numerator, quotient, remainder, binding storage and team
storage must be mutually disjoint.
