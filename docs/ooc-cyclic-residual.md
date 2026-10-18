# Certified cyclic residuals

The private spilled-product provider supports the bounded signed residual
`E = scalar*X*Y - C*B^e`, with `B=2^64`, through a complete cyclic transform.
This is used by reciprocal refinement and the final scaled square in fused
multiply-rsqrt with C=1, and by quotient cancellation with a stored numerator
C. It is not a public arbitrary-ring API. General middle windows still require
their own carry/enclosure contract.

`CyclicResidualSpec` gives a minimum period, the unit's word exponent and a
strict signed magnitude bound. The selected period is K words and the modulus
is H=B^K-1. Admission requires K>=bound.words+2 and a nonzero high-word bound,
so the admitted error interval lies strictly inside (-H/2,H/2). Inputs fit the
period. The ordinary CRT coefficient bound remains valid: each wrapped
coefficient contains at most min(input_trunks) products. Scalar multiplication
is applied after CRT and has its own carry-width bound.

The stored-subtractand variant also names an immutable SourceWindow with at
most K logical words. Its original physical backing may be wider. The owner
binds a third value/Region, preserved through return. Input shifts, complement
and virtual zero words use the same source contract as ordinary products.

The ring has complete outer support, using the same polynomial rows and root
conventions as LIN. The private arithmetic request distinguishes it explicitly
from LIN. The optional result descriptor is mandatory for cyclic execution;
passing a cyclic plan through an ordinary-product call is rejected before
dispatch.

## Carry and subtraction

The integer emitter adds the positive representative H-B^(e mod K), represented
by all-ones words with one cleared bit. It constructs no constant file. The
existing scalar multiplier, when needed, precedes this addition.

For stored C, reducing C*B^e modulo H rotates its at-most-K source words by
e mod K; no two source words overlap. Before each CRT worker batch, the reader
fills the corresponding subtractand segments. Each worker adds the local
ones-complement words to its CRT output and outgoing carry. Together these
segments add H-(C*B^e mod H). There is no product-temporary file or separate
subtraction/absolute-value pass.

Query charges the original source's pinned index, transfer buffers and one
subtractand segment per CRT worker. Its exact reads comprise covered chunks
plus repeated chunks at segment boundaries. Rotated segment cuts form an
arithmetic progression; alignment and clipping counts are computed directly,
without enumerating transforms, segments or candidate plans. Bounded rebinding
can shrink the backing/physical reads but cannot change the arithmetic period,
segment layout or workspace grant.

Each independent segment retains a 512-bit guard, its following all-ones run,
the first other byte, partial fragments and its outgoing tail. Each segment's
carry transfer is its tail plus at most one. The complete transfer F is
monotone and its image for nonnegative inputs is contained in {F(0),F(0)+1}.
Iteration from zero therefore reaches a fixed point in at most three traversals
of these descriptors. Only then are the guards and fragments published once.
No first-chunk repair or full-result correction pass is needed. Current T,
transform-order and scalar bounds put the incoming carry well below 512 bits;
overflow remains a fatal invariant violation.

The redundant all-ones representative H is allowed and lifts to zero. Positive
and complemented leading-word summaries are collected from finalized chunks,
including output outside the stored projection. The sign and strict magnitude
bound are checked before publication.

The usual full-product m61 equation is not valid on an arbitrary ring residue.
This provider checks the recovered signed E directly against

```
m61(scalar) * m61(X) * m61(Y) - m61(C)*2^(64e)  modulo 2^61-1.
```

Input residues describe the actual decoded source windows, including shifts or
complementing. C's residue is collected during its existing segment reads,
weighted by original source positions and e, rather than the rotated positions.
For a negative lift, subtract m61(H) from the carried R residue.
The descriptor also reports the absolute magnitude's m61. The writer seals and
the owner publishes only after carry, magnitude and m61 checks complete.

## Views, reuse and planning

The stored bytes are a projected window of R. `CyclicResidualResult` tells the
consumer whether its magnitude view is complemented. `SourceWindow` applies
ones-complement to bounded stored words before shifting; virtual guards remain
zero. H-R is exactly this ones-complement, so no absolute-value file or extra
carry scan is needed. Consumers must preserve that representation descriptor;
the raw stored bytes alone are not the signed error's magnitude.

Reciprocal refinement caches U's Rows or completed Columns once for residual and correction.
The correction retains ordinary LIN/TFT support and can consume a prefix of the
wider cached rows. Its smaller actual source window and retained-plane grant
are precompiled. The final multiply-rsqrt square already has its own geometry,
so it uses the same residual provider without introducing another cache.

Ring basis selection uses one certified period per available prime basis,
NP4 through NP10 in the current bundle. The widest legal byte-aligned codec
fixes the transform order; T then tightens arithmetically within that order.
Ranking uses complete residue volume and the existing prime-cost factors,
without the unrelated native Bailey tower penalty. The selected basis receives
at most four complete execution summaries from the existing resource planner.
There is no T/layout grid or runtime retuning after the inverse is known.

The complete LIN and cyclic recipes are compared using their byte costs and the
existing analytic profile. LIN remains available when the cyclic geometry is
not advantageous. Quoted storage includes cache lifetimes, projected output,
per-phase retention, coefficient/carry storage and all cleanup lifetimes.
IO failures release unpublished work; the existing algorithm-phase restart and
checkpoint contracts are unchanged.

Quotient refinement compares two complete fixed recipes. Its coarse quotient
and final correction retain shared U Rows; cancellation can instead use the
stored-numerator cyclic provider. The checked residual window feeds correction
through an explicit complemented magnitude view. The existing <3-ulp rung and
<7-ulp original-ratio contracts are unchanged. No plan choice depends on the
computed reciprocal or operand digits.

Tests compare wrapped NP6/9/10 residual windows against independent exact
integer references, with both signs and output projection. Carry gates include
all-ones wraps requiring the third fixed-point traversal and the redundant
zero representation. Shift/complement guards, full refinement ladders,
generic file jobs and checkpoint recovery also pass.
