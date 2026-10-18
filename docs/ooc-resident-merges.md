# Native products and scratch loans inside stored merges

A stored series merge may promote an individual product to the existing native
engine when its complete inputs, result capacity, prepared tables, arithmetic
workspace and IO buffers fit the admitted RAM grant. Inputs and final values
can remain stored even when the arithmetic operation is resident. The initial
provider uses the native p48 Bailey engine, whose shared bank is already covered
by the external product budget.

Selection is bounded: one native query per eligible stage, with one compact
prime-batch replay if required. Arithmetic geometry is not searched again to
fit RAM. The complete native proposal is compared with the existing spilled/
cached proposal using write costs and the current analytic profile. No choice
depends on operand values discovered during execution.

`ResidentProductPlan` now carries bounded source windows. Before execution,
rebinding can change the actual source slice, shift or virtual padding within
the quoted capacity. Input arrays have explicit readable guards; missing high
words are zero-filled. High-output projection preserves full-product m61
validation before a window is exposed. Existing exact-low-prefix output still
requires omitted high words to be zero.

A single-use numerator product can remain in its result array until the sum,
including across another product or cache build. Producers are placed backward
through the fixed stage list. Each retained array lies beyond the complete
workspace of every intervening stage and its consumer, including any later
loan and cache storage. The complete grant is checked before enabling the loan;
an inadmissible loan retains ordinary stored output. No arithmetic search or
extra multiplication is introduced.

The native layout may put that output after its own inputs, tables and transform
scratch. Those bytes die on return and can then serve as the consumer's prefix,
avoiding a duplicated scratch-sized hole before the output. The native call
ends all arithmetic/table users before returning the span and magnitude summary.
Each span is cleared at its last use, after the sum joins. This preserves the
existing product truncation and signed-sum rounding barriers.

`LinearInput` accepts an immutable resident span alongside stored views and
virtual scalars. A resident `ValueReader` performs checked range copies without
transport or checksum-index IO. The native product has already checked m61;
the linear consumer performs its normal arithmetic check. The caller owns the
span and keeps it stable throughout the call. Source shifts and precision
truncation use the same rules as stored magnitudes.

Borrowed products have no Store reservation or backing charge. Final numerator
and denominator values are still published as ordinary immutable stored values.
Cancellation and IO errors release all taken values and return no output; there
is no destructor IO, arithmetic allocation or automatic arena growth. Durable
checkpoint formats and their algorithm-phase boundaries are unchanged.

Complete merge queries include native alignment, the maximum of sequential
phase storage, any live shared-row cache and the borrowed result's lifetime.
The resulting plan alignment propagates to the enclosing series grant. More
RAM can be used within the caller's existing limit to reduce external passes;
MEMLOCK permission remains separate from that limit.

Exact finite/tail results, signed inputs, projected products, allocation watches,
write/backing budgets, failure cleanup and checkpoint recovery are tested. The
explicit native/borrowed-path counters ensure these tests exercise promotion.

## Native subtree input handoff

The last native child of a stored finite merge can return private resident
magnitudes. The compiler first tries the complete requested child tuple, then
the single value with the largest `(reader_count+1)*bytes` transfer saving.
These are two bounded lifetime choices; arithmetic plans, tree cuts and child
visit order remain unchanged. The current lowering requires an all-native
parent merge. A spilled/cached parent retains ordinary stored inputs.

Input slots are packed at the end of the existing series workspace, ordered by
last read. At stage j, its private workspace stops before the lowest still-live
input slot. Once that stage joins, dead input space becomes part of the prefix
available to later stages. For a CommonP2B3 right child, T_R dies after the
secondary numerator product, D_R after the denominator product, and U_R after
the contraction product. Their storage never requires a general allocator or
an additional arena reservation.

After the producer joins, it publishes any unselected values and compacts the
selected magnitudes into those slots. A static alias check protects every other
return still awaiting its copy. Native products read the selected immutable
windows directly, omitting their former copied input arrays. The backend must
support bounded inputs; shifted/complemented/virtually padded resident windows
are not admitted by this lowering. No extra readable input padding is required.
Any unwritten high part of a bounded product is explicitly zeroed before its
full-product m61 check or publication.

Incoming loans and intermediate result loans share the same last-use analysis.
Every operation's complete layout is replayed before accepting the handoff.
The compiler then recomputes the stored frontier's write and backing bounds;
it does not subtract hypothetical transfers from runtime counters. Private
input loans are consuming: they remain immutable while live, then may be
overwritten by later work. Final parent results still own ordinary immutable
Store values, so job/checkpoint phase boundaries are unchanged.

## Shared transform across numerator and denominator

After native-child handoffs have been fixed, an admitted resident merge may
combine `T_L * D_R`, its original signed/rounded numerator sum, and
`D_L * D_R` into one episode. A single fixed-geometry replay of the larger
product supplies a common basis. The complete pair must fit the existing grant
without evicting a child loan, increasing stored writes or peak backing. The
existing multiplication cost model rejects padding that costs more than the
saved forward transform; there is no new geometry/threshold search.

The private product continuation runs only after the first product's workers
join. It checks the complete product with m61, applies the original precision
limit, and immediately consumes that view in the original numerator sum. The
same result array then stages the next fresh operand and receives the
denominator product. The common completed spectrum stays live throughout.
The sum's IO buffers are separately quoted outside transform storage and the
result array. Final numerator and denominator retain normal Store ownership.

This keeps one full result array. Product rounding and signed-sum rounding
remain distinct; a spectral MAC would require a separate rounding contract.
Both products are checked independently. Zero operands use the same compiled
ordinary program and storage. IO failure at the joined continuation prevents
the second product from starting, joins the episode and releases all
unpublished taken state. No arithmetic allocator, helper thread, extra public
API or mutable cached handle is introduced.
