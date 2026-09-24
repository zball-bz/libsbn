# Product windows and dependent groups

Arithmetic recipes describe word-aligned products, signed cancellation,
requested windows and magnitude/error bounds in `product/window.hpp`.
A `WindowGroupShape` is an ordered group of at most three dependent products;
value IDs describe repeated mathematical operands. It is not an interpreter.

`algorithms/reciprocal.hpp` and `refinement.hpp` implement the precision ladder
and correction equations without inspecting FFT, NTT, prime or codec fields.
The product provider chooses physical geometry, numerical domain and reuse.
Small providers are inlined. `local_program.hpp` preserves their selected
shared, cancellation and peeling layouts as packed, pointer-free records for
public-plan replay. Ordinary leaf geometry remains a kernel responsibility.
`FullWindowFactory` supplies a correctness fallback
for complete multiplication, and `ProgramMultiply` adapts the existing ordinary
ProductProgram interface. Thus optimized windows are an optional backend
capability rather than a prerequisite for the arithmetic recipe.

Repeated NTT products carry an existing single-use/multiple-use distinction.
The native provider admits NP9/NP10 automatically for repeated Flat episodes
when the worker budget can execute every prime in one wave. It still derives
one capacity-certified geometry per basis, before computing any inverse;
there is no additional codec/layout search or length-specific winner table.
Single-use W1/W16 windows keep NP4..8, and the existing W32 and explicit-prime
capabilities are unchanged. This is a measured native policy, not a theorem
that wider prime sets always run faster: CRT and preparation can outweigh the
smaller transform. Complete storage queries include the selected basis.

A window denotes the magnitude slice after its declared word origin, with a
separate sign. Exact requests have zero error. A bounded request may choose a
different sign near zero, but its signed error remains within the requested
budget. When the declared magnitude bound covers all high words, the error is
an ordinary signed truncation error; low windows otherwise have modular
semantics. This deliberate approximation budget is distinct from FFT numerical
admission and does not authorize changing a caller's exact cyclic modulus.

The local provider supports ordinary IFMA/FFT products, guarded middle bands,
shared FFT spectra and bounded boundary peeling. Peeling is admitted only near
a smaller legal transform, with a bounded high tail and no workspace increase.
The body is multiplied cyclically, peeled high terms are added exactly, and an
independent low prefix resolves the wraps. The selected lift has one through
eight words and obeys the documented signed-wrap bound. Division guards are
preserved; physical period and requested precision remain separate.

Cyclic FFT admission has its own measured coefficient envelope. Wider codecs
use that envelope rather than half of a linear-transform size cap. Automatic
fresh window groups only add wider choices whose root arrays are already
published; explicit product queries retain the full numerical capability.
The precompiled cyclic provider can use balanced 16-bit at the same period
when unsigned input support is insufficient. A full-period balanced operand
wraps its final carry digit into the first digit. Its output bias counts the
actual shorter digit support; coefficient-error checks alone do not validate
this integer carry rule.
For 32-bit pair slots, the multi-carry check includes a pre-carry sum of
`2*radix-1`: an incoming one can create a second overflow. Such a vector uses
the exact carry path, including vectors entered with a multi-bit carry.

Wide CT cancellation can use the residual's existing 64-bit error allowance
to omit low coefficient packing. For word origin o>0, the emitter starts at
the preceding complete 64-digit block. It joins all retained fronts and omits
the final cyclic carry into word zero. The existing coefficient bound limits
the resulting modular high-window error to less than 2^40. Clipping the
subtraction operand loses at most two low borrows; centering/sign conversion
costs at most one further truncation unit. This remains inside the declared
2^64 residual allowance. Newton's two guard words preserve its strict <3-ulp
result; exact division keeps its final integer remainder and correction.

This optional capability applies to 17–20-bit CT residuals whose magnitude
bound fits strictly inside half the ring. Exact product/window requests use
the existing emitters. The transform, input codec, roots and numerical caps
are unchanged: the saving is in integer packing, stores and subtraction, not
in omitted FFT stages. The full buffer is still reserved, but only its high
view is valid. The primitive leaves words below
`bits*floor(o/bits)` untouched. No extra scratch, plan candidate or public API
is introduced.

Local window groups may instead use the private bounded plus-ring capability,
modulo `2^(64*r)+1`. This is a representation choice for a bounded product or
signed cancellation, not a change to a public cyclic product's modulus.
Both operands are shorter than the period; the short support is below 9/16
of it. A cancellation's declared magnitude fits strictly inside the period,
so the centered modular representative recovers the signed result uniquely.
The canonical representation has `r+1` words and includes the endpoint
`2^(64*r)`. Its extra word is included in the complete workspace query.

The native implementation uses RightAngle FFT kernels. For digit radix
`b=2^bits`, adding coefficient bias `m*(b-1)` produces a nonnegative carry
stream. The added integer is `m*(2^(64*r)-1)`, or `-2*m` in the plus ring.
If the carry stream ends with high word C, the final low-word adjustment is
therefore `2*m-C`. The unsigned 16-bit emitter packs biased coefficient pairs
and then four coefficients into each limb; the proven pair-capacity bound
precedes its vector carry chain. Endpoint normalization and centered sign
recovery are integer operations shared by the product provider.

Default selection only substitutes this representation at an already selected
unsigned 16-bit odd-radix geometry with published roots and lower group cost.
It adds no codec/order search. Wider private variants have separate measured
caps but are not selected automatically here. In particular, the balanced
18-bit radix-3 case at 12288 complex points reaches coefficient error 0.25
and remains excluded; the gate requires strictly less than 0.25. Existing
public plus/minus and linear numerical domains are unchanged.

`compact_windows.hpp` supplies a W1 provider for the admitted cyclic FFT band.
It selects an enclosing ring or the immediately smaller legal ring, charging
the exact low products needed by the latter. The quarter-period witness bound
comes from the largest gap in the supported radix family. Arbitrary input-length
exceptions and a Cartesian plan search are unnecessary.
The smaller ring can remain legal beyond the largest enclosing FFT. In that
case, selection also compares its full group cost, including low products and
cache rebuilding, against the ordinary NTT group. Neither this check nor the
signed lift enlarges the FFT's numerical envelope.

For each product, E=z+t*(B^r-1), and g low words identify t modulo B^g. The
magnitude bound supplies a unique signed lift. The provider retains one integer
buffer and at most two full FFT arrays, reuses the residual's dead storage,
and releases/rebuilds the quotient cache across its uncached residual when
necessary. All witnesses, low-product work and cache rebuilds are included in
the fresh operation and its byte query. Arithmetic still calls the unchanged
reciprocal/quotient refinement equations.

The compact FFT provider also supports retaining its common spectrum through
the uncached residual. This removes its rebuild but requires a third FFT array
during that product. The exact fused terminal selects this mode only when the
existing inverse/residual phase union or ordinary terminal allowance covers
the full quote. Otherwise it preserves the ordinary local recipe, or uses the
two-array compact recipe outside that local domain. Standalone compact
services keep their prior policy.

The compiled serial NTT provider can also choose a smaller period. Each prime
family contributes only its widest certified width at the immediately smaller
order; a choice must reduce weighted trunk-prime volume and pay for its exact
low products. The maximum scratch requirement is checked over every group
member, since the scratch of different multiplication recipes need not be
monotone in length. The entire low-product calculation must fit the selected
binding's existing result-plane pool. No separate repair allocation is made.

Before a product, the provider borrows those dead planes and computes the low
witness. A quotient retains it in the integer output's unwritten high tail.
Reciprocal correction consumes rho in place, so it instead borrows the unused
high words of the caller's reciprocal output, beyond the old approximation.
Witness/input disjointness is checked. The loan ends before the NTT starts;
after reconstruction the witness is dead before final value assembly.
The tail-lift variant resolves low carries before adjusting the signed high
wrap count, so it needs no second witness copy.

Native cached cyclic products also expose a private bounded-input execution
capability: fresh B may be shorter than planned and is zero-extended logically.
The basis, cached A, workspace and output capacity remain unchanged. Public
product execution still requires its declared input lengths. Reciprocal
correction can therefore reuse its cancellation binding without materializing
or transforming a zero tail. Providers lacking this capability retain the
explicit padding path.

Shared local FFT products consume their complete input before integer emission.
Reciprocal correction reuses the dead residual buffer under this guarantee;
IFMA leaves retain disjoint buffers. The native FFT/window crossover is one
generated backend parameter (256 short-operand words), shared by reciprocal
and quotient recipes rather than separate per-length rules.

`DifferenceValue` describes either a normalized value or a low window followed
by a high span. `product_difference` forms the exact signed value-minus-product,
including cyclic folding and low-word recovery, without materializing the
concatenated value. Exact division consumes this signed result and retains its
bounded Q/R correction. Prepared-product selection and spectrum construction
compatibility live in `local_product.hpp` and `repeated_product.hpp`.

The local exact-division product provider also admits the existing wide FFT
codecs. It visits adjacent smaller geometries in the `{1,3,5,7}*2^k` family
and derives the minimum digit width from capacity and the backend's strict
short-input support bound. Unsigned/balanced admission, published roots,
estimated product work and complete per-product storage are checked before
selection. It constructs no competing division/block plans and uses no
per-length winner table. The guarded short-tail kernel is explicitly included
in the scratch quote. Sharing/alignment of the enclosing service can differ
slightly even when individual product allocations do not grow.

Immutable engine roots, prepared operand representations and transient work
have separate lifetimes. Shared FFT groups prepare U once. The FFT backend's
build/apply callback performs the first forward and multiplication in one team
episode and publishes a ready spectrum only on return. Dependent stages retain
real synchronization boundaries. A cache is not kept by increasing the admitted
peak merely to avoid a rebuild.

Published FFT constants are counted once per process. Operation queries report
all private storage. In-RAM execution performs no heap/page allocation or plan
search; candidate measurements include the full fresh query/preparation/use/
cleanup lifecycle. Generated machine profiles own crossovers, while numerical
admission remains a separate backend contract.

The native FFT implementation TU uses Clang `-O3`; other production TUs retain
their configured flags. Native FFT coefficient gates use the same optimization
level. The 64-digit packer is forced inline so each block need not transfer its
vectors and carry state through an out-of-line call. No fast-math flags or
numerical-cap changes accompany this code-generation choice.

Unpinned ordinary FFT products include root construction in their use. If the
prepared-cost choice is RightAngle with unpublished roots, the product provider
uses CT at the same point count, radix and digit width when that CT shape is
legal, has compact/published roots, and fits the caller's workspace budget.
It adds neither a geometry search nor a length exception. Pinned and cached
recipes retain their basis. CT's signed tail can need more workspace despite
smaller total storage, so a tight workspace-only budget keeps the original
fitting recipe. A prepared RightAngle spectrum remains a supported capability.

Large W1 Flat cyclic spectra use format 4: canonical residues in packed
48-bit slots, with the same mathematical basis and scale as the lazy64 format
2. The selected native threshold is 2^18 coefficients per prime; smaller and
parallel producers keep lazy64. Consumers honor the actual descriptor format,
including mixed-format MACs and a wider team consuming a packed spectrum.
Format and slot stride are checked even for a planned descriptor with no
instance ID. A larger lazy64 allocation is not interchangeable with packed48.
Each plane reserves 128 readable padding bytes; the 48-bit load may read up to
16 bytes beyond its slot, while exposing only its eight valid residues.

A cached serial Flat cyclic binding also exposes an internal fresh-MUL
capability. Its query checks operand support, CRT capacity and the existing
worker workspace. Fresh A uses the output planes as temporary packed spectrum
storage; the cached operand and its immutable tables remain live. Fresh decode
uses the plan's own scale and skips compensation for the cached operand.
The quotient group can therefore use one binding for coarse product, residual
and correction, preserving U instead of rebuilding it. Public cached execute
still requires its original declared inputs; the alternate entry is private.
