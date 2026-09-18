# libsbn

A SIMD multiprecision arithmetic library. The current native target is AMD Zen 5
with AVX-512 IFMA; the build uses the pinned Clang toolchain. The public C API
contains no SIMD vector types. The API is still under development.

## Acknowledgements

libsbn owes a great deal to [y-cruncher](https://www.numberworld.org/y-cruncher/)
and its creator, Alexander J. Yee. Its published technical articles and its
performance as a reference implementation have guided our work on multiplication,
binary splitting, memory management, and radix conversion. Without y-cruncher,
libsbn would not have reached its current level of optimization. We are grateful
for both the software and the knowledge shared with the community.

## Implementation and experimental provenance

Our research has included behavioral probing and disassembly-based analysis of
y-cruncher to understand its algorithms and performance. No y-cruncher kernels
or code recovered through reverse engineering have been copied, translated, or
incorporated into the libsbn library. Our corresponding implementations are
written independently and validated through our own tests and experiments.

We design, carry out, and evaluate our own experiments. Our performance results
come from our own measurements, with raw data, build identities, and validation
records retained in the [research repository](#research-checkout). Components
imported from other sources retain their separate provenance and license notices;
see [LICENSE.md](LICENSE.md).

## Capabilities

- Signed integer values and scalar/u52 base arithmetic.
- Small, FFT and NTT multiplication; squares, cached operands and supported product windows/rings.
- Inverse, approximate division and reciprocal square root services.
- Exact finite binary splitting and precision-bounded series; e, pi, ArcCoth and integer logarithms.
- Exact integer division with remainder (unsigned and signed truncating), one-shot or with a reusable prepared divisor.
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
[radix conversion](docs/radix.md), [exact division](docs/division.md) and [testing](docs/testing.md).
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

## License

libsbn is licensed under **LGPL-3.0-or-later**, except components that retain
their own license notices. We chose LGPL to keep redistributed improvements to
the library available as source to recipients, while allowing applications under
other licenses to use it in accordance with the LGPL.

See [LICENSE.md](LICENSE.md) for scope and third-party provenance, and
[LICENSE](LICENSE) for the LGPL text.
