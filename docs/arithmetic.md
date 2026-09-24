# Arithmetic contracts

The value layer uses little-endian uint64 limbs. Signed values use a separate
sign and magnitude. Public operations specify normalization, capacities,
alignment and legal overlap; these are requirements, not hints.

Product requests distinguish linear multiplication, squares, cached applications
and supported cyclic/window operations. A plan is valid only for its declared
input bounds and recipe. Use query results for the actual supported domain and
storage; do not infer one backend's capabilities from another's.

Newton and constant services carry explicit precision and output contracts.
Finite binary-splitting values may be exact integers; bounded series execution
may retain scaled, limited mantissas with an error certificate. These represent
different contracts even when their final rounded values agree.

Final certification can fail fatally when an error interval straddles an output
boundary. Do not bypass the guard or relabel an approximate intermediate as an
exact result. The series sum service combines working-precision components and
certifies the final value, including signed coefficients and cancellation.

Exact integer division (divrem.h) returns the true quotient and remainder;
its contracts are summarized in [division](division.md).

The headers newton.h, series.h, formula.h, formula_sum.h and constants.h define
precise public domains. Tests include independent integer references, modular
checks, range/capacity checks and numerical-boundary cases.

## Shared reciprocal and quotient kernels

The reciprocal ladder and quotient terminal share the private correction
recipe in `algorithms/refinement.hpp` and the provider-parameterized drivers
in `algorithms/reciprocal.hpp`. They describe word windows, signed cancellation
bounds and repeated operands. The product layer owns local FFT/IFMA selection,
cyclic-window lowering and the inverse/quotient group's physical NTT geometry.
An ordinary full-product provider can run the same drivers through the window
fallback, including the existing ProductProgram interface. Specialized leaves
remain direct; the public C ABI and exact Q/R finishing contract are unchanged.

Window error is signed. A bounded approximation may choose the other sign
near zero, provided the returned signed value stays inside its error budget.
With all high support present, this is an ordinary truncation-error bound;
exact low/middle windows otherwise retain their declared modular semantics.

Reciprocal and approximate division use bounded local u52/FFT kernels below
their fixed crossover, including a guarded middle residual. The middle
kernel reads its divisor with two logical zero words; it does not require a
copied padded input. The quotient terminal subtracts the shifted numerator
in this band and retains the same published error bound of less than three
units in its output scale. Larger ladders start from a local reciprocal
prefix and use the spectral kernels above it.

An exact division consumer may use a bounded quotient estimate, but must
finish with an exact residual and correction. Its prefix truncation and the
terminal's error together fit the eight-correction bound. Approximate
products and exact high/window products are not interchangeable without
propagating their error through the recipe.

## Exact cyclic reconstruction

Let B=2^64, M=B^r-1, and let z be the computed residue of an integer E.
If E=z+tM, then t is determined modulo B^g by the independently computed
low g words of E, since M=-1 modulo B^g for r>=g. A proved bound
|t|<B^g/2 selects its signed representative. Reconstruct
E=z-t+t*B^r, with enough high words, then take its sign and magnitude.
The redundant zero representation z=M is also supported.

Newton residuals need one low word. The first quotient product and Newton
correction products may need three: their bounds are below
64*B^(n+2), so an n-word ring can be used with three reconstruction words.
Wider rings retain the cheaper centered/exact-integer path. This is an exact
integer identity, not a probabilistic checksum. Query accounts for these
output words when laying out scratch and selecting its storage class.
