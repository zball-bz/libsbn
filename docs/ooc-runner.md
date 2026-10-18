# Bounded constant application

`examples/ooc_constant.cpp` is a reference application over the private generic
JobPlan, FileStore and Checkpoint providers. It is not a public C ABI. The
library and this example have no dependency on the research checkout.

Build only this example with the pinned toolchain:

```sh
python3 tools/build.py examples --only-example ooc_constant
```

The default is a query, with no operand computation or data files:

```sh
build/native/ooc_constant --constant pi --digits 5000000000 \
  --ram-bytes 4294967296 --workers 16
```

The complete resident grant includes the arithmetic arena, plans, metadata,
published tables, team/stack costs and an extra live-process allowance (64 MiB
by default). The query reports peak backing, a cumulative data/index write
bound, required locked pages and admission against current host/tmpfs capacity.
MEMLOCK grants permission to pin pages; it is not the memory budget. The
process must already have the reported allowance before executing. No system
limit is changed by the application.

The application prepares io_uring QD16 by default, with one bounded kernel
fallback worker per ring. `--io-depth 0` explicitly selects scalar IO; other
supported power-of-two depths are diagnostic choices, not a planner search.
Ring mapping bounds are charged to the context before job compilation, and
creation/destruction are included in the cold fresh lifecycle. Failure to
prepare the requested provider is a setup error, with no implicit fallback.
The library's FileStore default is unchanged.

Execution requires all of `--run`, a new `--directory`, and `--write-limit`.
The directory's filesystem must be tmpfs unless `--allow-disk` is explicit.
A normal run requires a resident budget of at least 4 GiB. Real-device runs
also require a finite cumulative write limit; selecting another pathname does
not enable them. File writes, including failed submissions, consume that quota.
Host admission includes computation RAM, tmpfs backing when applicable, and a
4 GiB host reserve. An external supervisor should additionally bound the
process group and account for tmpfs pages in its host/cgroup allowance. SSD
experiments should use one new owned directory, a predicted traffic bound and
a declared wear budget, after RAM-backed qualification.

The application takes `build/benchmark.lock` and checks Tctl before arithmetic.
`--externally-locked` is only for a supervisor already holding that same lock.
It does not waive memory, filesystem, write-quota or thermal admission. A long run
may warm beyond its starting gate; both temperatures are reported.

`pi`, `e` and `log8` select formula/terminal descriptions for one generic job.
Here `log8` means the BinaryBBP series `sum(k>=1, 8^-k/k) = log(8/7)`, not
`log(8)`. The provider receives coefficients, tail bounds and terminal scales;
it has no constant-specific execution branch. Precision safety margins and the
term count are derived before operands are produced.

The JSONL output records the complete query, selected plan identity, each phase's
time/process-CPU/read/write ledger, queue usage, and the final result. Optional
`--describe-series` adds coarse node/stage shapes for query-time investigation.
`cold_fresh_to_output_ns` starts
before planning and includes setup, arithmetic, decimal formatting, plan
retirement and intermediate cleanup. The result remains live for verification.
`verification_ns` measures the independent 128-bit tail check for pi and
log(8/7), widening its guard window from 192 to 384 bits if necessary. Euler
does not claim a BBP check. Output/context disposal is reported separately;
`whole_session_ns` includes it and verification. Resume has a distinct timing
field and never reports itself as a cold fresh execution.

Default output is a certified stored decimal payload with an integer part in
metadata. Successful ephemeral runs verify it and remove their private data
directory. They do not export a standalone text file. The `certified_digits`
count is checked against the request; an uncertified suffix is not success.

`--checkpoint-key` accepts a nonzero 64-hex-digit application compatibility
key. It enables durable algorithm-phase publication and preserves the final
root on close. Recreate the same formula, build-compatible plan and key, then
use `--resume --directory ...`. Restore validates indexed payloads and numeric
summaries before continuing; see [checkpoint semantics](ooc-checkpoint.md).
This key must reflect the application/build/formula schema; it is not inferred
from a pathname. The manifest write allowance is separate from data writes.

SIGINT/SIGTERM latch cancellation. The IO/algorithm safe points drain owned
work and retain the latest committed root; publication defers cancellation
through its commit critical section. Resident arithmetic is not interrupted
inside kernels. Cleanup failures are explicit and never disguised as success.

Small forced external paths are available only through `--diagnostic-small`
and the explicit stored-depth/native-seed/radix-frontier controls. They support
focused correctness tests and are recorded in the selection JSON. Production
planning remains a bounded heuristic, without a parameter search.
