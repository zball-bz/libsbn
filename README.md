# libsbn

A SIMD multiprecision arithmetic library. The current native target is AMD Zen 5
with AVX-512 IFMA; the build uses the pinned Clang toolchain. The public C and
C++ APIs contain no SIMD vector types. The API is still under development.

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
- C++20 owning integers, eager arithmetic, allocation-free destination calls and explicit prepared expressions.
- Resident C++ bigfloats with bounded-error arithmetic, roots, elementary functions, AGM/K/E and explicitly shared constants.
- Infinite signed bit operations, division rounding modes, ordinary powers, GCD/extended GCD/Jacobi, LCM and modular inverse.
- Small, FFT and NTT multiplication; squares, cached operands and supported product windows/rings.
- Inverse, approximate division and reciprocal square root services.
- Exact finite binary splitting and precision-bounded series; e, pi, ArcCoth and integer logarithms.
- Exact integer division with remainder (unsigned and signed truncating), one-shot or with a reusable prepared divisor.
- Formatting and parsing in bases 2 through 64 for integers and dyadic values.

Individual query functions define supported domains and required storage.
General integer roots and modular powering remain to be implemented.
Stored multiplication/constant execution is documented in the
[OOC runner guide](docs/ooc-runner.md). Resident float calls do not spill
implicitly; public OOC floats and correctly rounded transcendental functions
remain separate work.

## C++ integers

```cpp
#include <sbn3/sbn.hpp>

sbn::integer a = 123, b = 456;
auto value = sbn::pow(a, 3) + b;  // eager owning result
auto [quotient, remainder] = sbn::divrem(value, 97);
auto text = sbn::to_string(quotient);
```

For controlled memory, use caller-provided output and `*_into` operations with
queried workspace. `sbn::context` additionally compiles explicit expressions
with bounded inputs and reusable intermediate storage. None of these resident
interfaces hides disk IO. See the [C++ guide](docs/cpp-integer.md) and
[runnable example](examples/cpp_integer.cpp) for ownership, preparation,
conversion and current domain limits.

## C++ bigfloats

```cpp
auto x = sbn::bigfloat::from(2, sbn::bits{256});
auto y = sbn::log(x) + sbn::sqrt(x);
```

Stored inputs are exact dyadic points. General function results are bounded
approximations; use `*_into` for an explicit local error report and control of
output storage, scratch and Team. An optional context-owned constant bank
shares pi/log2 preparation across functions. See the [float guide](docs/float.md)
and [runnable example](examples/cpp_float.cpp). Performance tuning and
correct-rounding guarantees are distinct from this implemented numerical scope.

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
