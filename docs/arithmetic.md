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

The headers newton.h, series.h, formula.h, formula_sum.h and constants.h define
precise public domains. Tests include independent integer references, modular
checks, range/capacity checks and numerical-boundary cases.
