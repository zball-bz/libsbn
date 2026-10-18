# Generic stored constant jobs

`external/job` compiles and executes the existing series, Newton, fixed-output
and radix providers as one generic operation. The caller supplies a formula,
term interval, positive-sum bounds and tail certificate. There is no pi/e
branch in the runner. It currently materializes stored values at algorithm
boundaries; direct resident loans remain an optimization.

The private API has three operations:

- `query_job_precision`: derive binary, quotient and series precision from the
  requested radix digits, terminal bounds and maximum block count.
- `compile_job`: choose all backends and phase resources before operands exist.
  The plan, formula contexts and supplied series metadata have stable addresses.
- `advance_job` / `run_job`: execute one phase or finish the remaining phases in
  one exclusively assigned, prepared arena interval. Arithmetic allocates no
  heap storage and does not change page residency.

These private C++ types are not a new public ABI. The
[checkpoint layer](ooc-checkpoint.md) has a separate versioned manifest format.

## Mathematical input

Let the true series sum be S, with
`2^lower_bits <= S <= 2^upper_bits`. The terminal produces

`integer_addend + scalar * (reciprocal ? 1/S : S) / sqrt(radicand)`.

The scalar and radicand are positive u64 values. The complete value must fit
the formatter's u64 integer part; the compiler uses conservative dyadic bounds.
The existing u64 finishing-error descriptor and scaled-product support also
bound the admitted scalar. A query can reject unsupported bounds; execution
does not search for a different numerical recipe.

The selected finite sum's error from S must be at most
`tail_error_units * B^-P`, where `B=2^64` and P is the supplied series precision.
Together with the series compiler's `2*blocks+1` rounding certificate, write
`E = 2*blocks+1+tail_error_units`. For direct finishing, the input error is at
most `E B^-P`. For reciprocal finishing it is at most
`2 E B^-P 2^(-2 lower_bits)`, provided `E B^-P <= 2^(lower_bits-1)`.
This follows by keeping the finite sum at least half the certified lower bound.

The quotient uses normalized prefixes at precision n. Its result's exponent e
is known only after the series values exist. The compiler nevertheless bounds
it using the supplied sum interval. It chooses P so the propagated input error
is at most one raw quotient unit for every admitted e. Execution checks that
inequality against the actual normalization exponent before computing the
inverse. The quotient's strict 7-unit error then becomes a strict 8-unit bound
relative to the true target ratio.

Without a final root/scalar multiplication, fixed output uses 8 error units.
Otherwise it uses `8*scalar + 2^38`, incorporating the existing fused
multiply-rsqrt contract and `1/sqrt(radicand)<=1`. Four or more binary guard
words separate the requested output; the final guard test is still mandatory.
An ambiguous floor is a fatal mathematical certificate failure, as in the
resident constant services. Bounds and the tail certificate are formula
preconditions; this runner does not prove arbitrary user-supplied formulas.

For e, the Euler Hyperdescent definition returns e-1 and the terminal addend is
1. Pi uses the reciprocal Chudnovsky ratio, scalar 4,270,934,400 and radicand
10,005. BinaryBBP ratios use the same direct terminal. Formula selection and
term-count certification stay outside this generic executor.

## Plan before data, bind views afterward

The series compiler now exposes result backing bounds, including padding behind
retained windows. Shape-only queries compile inverse/quotient, fused root, fixed
output and format plans against those bounds. They require neither hypothetical
Store handles nor an already computed inverse.

After a series phase succeeds, `rebind_stored_quotient` installs the actual
normalized source views and exponent. Roots, NP/T, geometry, execution batches,
retention, workspace and write caps remain fixed. Only bounded physical reads
and scale metadata change. A short stored magnitude receives virtual low zero
words; it is not copied into a padded normalization file. Equivalent prefix
views have a canonical identity. Physical stripe accounting excludes wholly
virtual zero stripes and charges only actual shared integrity chunks.

The fixed-output phase instantiates only scalar window/guard metadata from the
actual exponent and verifies it against its compiled storage/write envelope.
No backend selection occurs there. Native bind operations retain their normal
replay checks.

## One resource grant

The common fixed cost includes the complete job plan, series metadata, engine
and radix constants, a controller allowance and caller-declared other live
memory. The working arena is the aligned maximum of the sequential phases,
including the series' separate IO prefix. Phases cannot independently spend
the same fixed overhead. Available working bytes are rounded down before
retained-plane admission.

The write bound sums all phases, including final binary and radix output.
Peak backing is the maximum of their full live-input/output/temporary bounds.
The query includes both outputs where they coexist. It also reports the
required free Store descriptor count. The [object-file Store](ooc-files.md)
makes the aggregate allocation bound sufficient for object placement, using
disjoint virtual slots rather than contiguous-bank packing. Actual OS/file
resource failures remain fallible. The old contiguous mode does not gain this
fragmentation guarantee.

## Safe points and failure

`JobState` owns the current immutable values and identifies the next phase:
series, quotient, optional fused root, fixed binary output, radix, complete.
An IO error or cancellation retains the previous phase's state. Local new
outputs disappear on failure, work leases are released, and failed submitted
writes remain charged. A retry is explicit and needs enough remaining write
allowance. Native arithmetic observes cancellation at its existing safe/IO
boundaries. Math/invariant violations remain fatal.

Success installs the new value state and retires the previous one. The caller
may retain the current checkpoint root while the next phase runs, publish its
replacement after success, then release the old root before advancing again.
That old/new overlap already occurs inside the phase bound. Arbitrary older
history requires additional declared backing. `run_job` simply advances until
complete; `advance_job` permits checkpoint/progress integration between phases.

The in-process plan cookie prevents accidental use with a different compiled
job. It is not a persistent StepKey or content hash. Checkpoints pair it with an
explicit build/schema key, verify immutable files on import, and restore these
phase states. Internal pass/subtree recovery remains pending.
The final state owns both the binary value and the certified-prefix digit
result, so a verifier can inspect the binary tail independently of formatting.
`complete` means all phases finished; callers must still inspect the certified
digit count. A shortened enclosed prefix does not establish a requested-digit
acceptance run.

## Evidence and remaining work

Production-runner gates cover Hyperdescent e, CommonP2B3 pi and BinaryBBP.
They compare binary and decimal output with independent exact/Machin references.
An injected quotient write failure preserves the series value identities;
checkpoint pins remain valid, retry succeeds, and cumulative accounting includes
the failed write. Safe-point cancellation, preflight cap refusal, allocation
watches, lease cleanup and completed-state idempotence pass.

The complete 5B-digit pi query at 4 GiB, with a 64 MiB other-live allowance,
uses F=259,525,636 binary fraction words, n=259,525,640 quotient words and
P=259,525,644 series words. It leaves 6 series blocks before the resident suffix
and selects 7 radix frontiers. Its total resident quote is 4,293,475,808 bytes,
peak backing 26,841,251,840 bytes, cumulative writes 297,881,231,360 bytes, and
free Store descriptor requirement 110. This is a whole-job metadata query,
not a 5B arithmetic run, IO trace or YC comparison.

Linux object-file transport and slot admission now pass the same job gates.
Durable phase publication/recovery now passes tmpfs crash gates. Full LIN windows, unshared radix spectra, native
frontier repair reservations and mandatory stored phase boundaries remain
efficiency work before verified 5B/fresh YC qualification. All current operand
IO uses simulation or RAM-backed files and real SSD operand/swap writes remain zero.
