# Architecture

The public C layer exposes values, plans, explicit storage and operation
lifecycles. Private implementation layers are separated by their contracts:

| Layer | Responsibility |
|---|---|
| runtime | pinned arena, scratch leases and recursive teams |
| value/core | limb operations and signed values |
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
