# Durable job-phase checkpoints

`external/checkpoint` persists the [generic job runner](ooc-job.md) at its
algorithm boundaries. It uses the [object-file Store](ooc-files.md), a fixed
binary manifest and an exclusive directory lock. It restores values and
continues the remaining phases; it does not restore private spectra or resume
inside a multiplication pass or series subtree.

The caller recreates the formula and compiled job, supplies a nonzero explicit
build/schema compatibility key, and keeps the plan and file directory stable.
The existing job cookie is also checked, but is not by itself a persistent
compatibility promise. No pointers or opaque native plan bytes are serialized.

## Manifest and verification

The 4 KiB, versioned little-endian manifest contains the generation, job cookie,
build key, next phase, charged writes, any reserved attempt, certified digit
count, integer part and up to five value records. Numeric records retain the
Region identity/index instance, logical length, revision, sign, exponent,
physical limb window and complete stored magnitude summary. Digit records name
the requested-width file and its certified prefix. The manifest has XXH3-64
integrity; each value retains the existing indexed chunk checksums.

Numeric `WordExtent` observation now also accumulates m61 during the existing
write stream. This adds no operand read or file pass at commit. The complete
logical carry observer avoids duplicating its already existing m61 work.
The persisted numeric summary describes the physical stored payload, including
any retained guard words; the separate view/exponent describes its value.
It is a byte/value fingerprint, not a replacement for the executor's arithmetic
certificates.

Import assigns a fresh virtual slot while preserving file identity and checksum
domain. Imported files open read-only and remain protected from deletion until
the complete restore validates. The reader verifies every payload/index chunk
and recomputes numeric extents/m61. A bad build key, malformed/torn manifest,
wrong length, corrupt data or inconsistent summary refuses restore. Failed
import cleanup closes descriptors without deleting the preexisting root files.
Use a fresh creation namespace on restart; imported identities remain unchanged.

## Publication protocol

Commit retains pins for the candidate root and follows this order:

1. Reclaim already retired private files and synchronize each root value file.
2. Synchronize the data directory so newly created filenames are durable.
3. Create a unique manifest draft, write it and synchronize it.
4. Attempt atomic rename over the current root manifest.
5. Synchronize the publication directory, then replace the old root pins.

Step 2 matters: file fsync alone does not make a new filename durable. It keeps
either possible root self-contained if a crash interrupts the final directory
sync. Old root files stay pinned until publication is confirmed.

Before rename, a failure preserves the old root and disposes of the owned
draft. An exclusive-create failure never deletes an existing file. Once rename
is attempted, any error is conservatively `commit_unknown`, even if the syscall
itself reported an error. Both root sets stay pinned and new object creation
is suspended. `resolve()` reads the visible manifest, accepts only the known
old or candidate image (or the original absence), and synchronizes the namespace
before releasing either set of pins.

Cancellation is checked before the publication critical section. Once entered,
it is deferred through publication; interrupted sync syscalls retry EINTR.
Cancellation alone therefore leaves a known old or committed new root.
Independent IO errors can still produce UNKNOWN and must be resolved. No
arithmetic runs until that uncertainty is resolved.

## Restart, GC and shutdown

Restore verifies the visible root and its files, then synchronizes the namespace
before orphan collection. Otherwise a later crash could roll back a visible
rename after GC had deleted the old root. Creation stays suspended until GC
completes. The collector first snapshots a bounded list of generated object and
draft names, then removes that list; it does not mutate a directory while
enumerating it. Live mappings and the valid current root are retained.

`close_preserving()` marks committed/uncertain root files for retention, releases
checkpoint pins and closes metadata descriptors. The caller then releases its
JobState and explicitly closes FileStore. That preservation transition forbids
new creation. Public owner destructors perform no IO. This differs from closing
an ephemeral file Store, which deletes its retired objects.

## Crash write allowance

`Checkpoint::advance` publishes an intent before invoking a numerical phase.
The intent conservatively charges that phase's entire worst-case write bound
while retaining the previous value root. Normal return publishes the actual
submitted-write charge, whether computation succeeded or returned an IO error.
After a crash, the reserved charge remains consumed; recomputation needs enough
remaining allowance. The runner cannot hide a crashed attempt by restoring an
older, smaller write counter.

If numerical work finished but its metadata publication failed, the caller
retains the new in-memory state. A later advance first publishes that state;
it does not redo its arithmetic. The active file Store blocks creation during
UNKNOWN/restore-GC states even if a caller bypasses the checkpoint runner.

Manifest IO has separate per-writer submitted/completed counters and an explicit
write allowance. It is not included in the job's operand/index charge. A fresh
checkpointed phase normally writes two 4 KiB manifests; at most an old head and
one current draft coexist on the normal path. Recovery workspace, manifest
files, orphan allowance and filesystem metadata must be budgeted beside the
arithmetic query. Per-writer metadata counters are not a global physical/NAND
write meter across crashes.

## Current gates and limits

Gates restore e and restart pi after every phase using freshly compiled plans.
Independent binary/decimal references pass. Commit performs no operand read.
Injected failures before rename preserve the old root; uncertain publication
resolves to either known root, including modeled rollback to initial absence.
Cancellation at rename completes publication, and a preexisting cancellation
leaves the generation unchanged. Wrong-build, torn-manifest, m61-summary and
payload-corruption cases refuse restore without deleting source files.

Separate processes receive SIGKILL at seven replacement-publication boundaries
and after a partial data write. Restart recovers the appropriate root, keeps the
crashed attempt charged, collects orphans and finishes with the exact reference
result. All test files are on verified tmpfs and are cleaned up. These are
process-crash and storage-model gates, not VM/hardware power-loss qualification.
No real SSD operand/swap writes or performance benchmark ran.

This first implementation is strict same-build/job recovery with one latest
root, synchronous file IO and whole algorithm-phase restart. Extra checkpoint
generations, subtree/pass recovery and configurable cost-based checkpoint
policies remain extensions. Large execution, actual memory/time/IO measurement,
window/fusion improvements and the verified 5B/fresh YC comparison are still
required by the full implementation goal.
