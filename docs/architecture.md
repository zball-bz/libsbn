# Architecture

The public C layer exposes values, plans, explicit storage and operation
lifecycles. Private implementation layers are separated by their contracts:

| Layer | Responsibility |
|---|---|
| runtime | pinned arena, scratch leases and recursive teams |
| value/core | limb operations and signed values |
| float | exact-dyadic views, bit precision, local error reports and bounded-storage real arithmetic |
| backends | native small/u52, floating FFT and p48 IFMA NTT kernels |
| product | algorithm selection, cached spectra, windows and workspace planning |
| algorithms | Newton services and composed arithmetic |
| series | formula facts, binary splitting, precision planning and terminals |
| radix | base 2..64 formatting/parsing and power/product plans |

ISA intrinsics remain private. Performance-critical templates and fused kernels
need not be split across ABI boundaries. Plans describe actual representation
and resource requirements; execution consumes caller-prepared storage.

Production policy comes from reviewed config/tuning profiles and explicit
code-generation templates. Experiments may propose policies, but do not inject
hidden environment switches into production hot paths.

The current private `external` implementation supplies checked storage/IO,
p48 external arithmetic, stored numerical recipes and constant-job recovery.
Its generic boundaries are not yet a public OOC ABI. The proposed
[OOC cleanup](ooc-architecture-cleanup-2026-10-05.md) separates byte storage,
arithmetic providers and job-state recovery while preserving the existing
pipeline. The [bigfloat roadmap](bigfloat-roadmap-2026-10-05.md) identifies the
missing numerical layer above the existing integer/Newton/radix services.
The OOC cleanup remains a proposal. Bigfloat implementation now follows the
[implementation schedule](bigfloat-implementation-plan.md): the first C++ value,
rounding and linear-arithmetic slice is in `sbn3/float.hpp`, while the later
root/elementary stages retain explicit completion gates. The
[API design](bigfloat-api-design.md), [implemented float API](float.md),
[root design](bigfloat-roots-design.md) and
[elementary design](elementary-implementation-design.md) record numerical and
algorithmic requirements; a design entry does not by itself advertise support.
