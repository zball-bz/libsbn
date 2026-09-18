# libsbn development

This is the library repository. src/, include/, tests/, examples/, production
config/ and necessary build/audit tools belong here. Experimental reports,
probes, benchmarks, results and frozen references belong to the separate private
libsbn_experiments checkout in ignored experiments/. Never stage that checkout
or generated build output in this repository.

Use the pinned Clang toolchain and explicit manifests. Preserve caller-owned
values, planned storage and allocation-free arithmetic; read docs/runtime.md and
the service header before changing contracts. Keep ISA intrinsics private.

Compilation and timing use build/benchmark.lock and must never overlap. Query
full storage and check host memory before performance runs: MEMLOCK is not a RAM
budget. Start timing at Tctl <= 62 C and retain raw repeats and identities in the
research repository. Keep core functionality independent of that repository.

The old GIMP copy is a migration backup, not the active checkout. Historical
receipts retain old paths and hashes; do not reinterpret them as current builds.
