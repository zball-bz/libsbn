# libsbn

A SIMD multiprecision arithmetic library. The current native target is AMD Zen 5
with AVX-512 IFMA; the build uses the pinned Clang toolchain. The public C API
contains no SIMD vector types. The API is still under development.

## Capabilities

- Signed integer values and scalar/u52 base arithmetic.
- Small, FFT and NTT multiplication; squares, cached operands and supported product windows/rings.
- Inverse, approximate division and reciprocal square root services.
- Exact finite binary splitting and precision-bounded series; e, pi, ArcCoth and integer logarithms.
- Formatting and parsing in bases 2 through 64 for integers and dyadic values.

Individual query functions define supported domains and required storage.
Out-of-core execution/checkpointing and general-purpose arbitrary-precision floating-point arithmetic are not yet complete.

## Build and test

    make lib
    make examples
    make check-plan
    make check
    make check-extended
    make audit

See [building](docs/building.md), [runtime contracts](docs/runtime.md),
[architecture](docs/architecture.md), [arithmetic](docs/arithmetic.md),
[radix conversion](docs/radix.md) and [testing](docs/testing.md).
Public declarations are in [include/sbn3](include/sbn3); runnable examples are in [examples](examples).

## Research checkout

The library builds and tests without the research repository. Experimental
reports, probes, benchmarks, measurements, historical plans and old reference
implementations are maintained separately in the private repository
[zball-bz/libsbn_experiments](https://github.com/zball-bz/libsbn_experiments).
Developers can clone it into the ignored experiments/ directory:

    git clone https://github.com/zball-bz/libsbn_experiments.git experiments

Keep generated build output in ignored build/. Do not add raw experiments or
large artifacts to this repository. Production tuning values and their provenance
identities remain versioned under config/; optional full evidence verification
uses the research checkout.

Existing license notices and their scope are described in [LICENSE.md](LICENSE.md).
