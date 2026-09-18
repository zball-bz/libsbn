# Building

The production source list is explicit in config/sources.json; platform flags
and the Clang version pin are in config/native.json. Do not replace them with a
wildcard build or arbitrary global ISA flags. The current implementation requires
the declared native CPU features and Linux runtime facilities.

    make lib
    make examples
    make check-plan
    make check-generated

The static library is written to build/native/libsbn_v3.a. C headers are in
include/sbn3. Link with pthread and the system math library as the examples do.
Consumers do not need AVX-512 flags merely to include public headers.

Build receipts record compiler commands, dependency hashes and artifact identity.
Compilation and measurements take the same build/benchmark.lock.

Arena-backed programs require sufficient locked-memory permission from the host.
Resource reservation happens before arithmetic. tools/run_extended.py prepares
only the test process's MEMLOCK limit; a larger MEMLOCK limit is not a host RAM
budget. Examples allocate their own input/output buffers and print query sizes.
