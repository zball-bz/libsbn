# Runtime and ownership

Values and output arrays are caller-owned. Query functions report support,
alignment and complete workspace requirements before execution. Prepare arena
storage explicitly, then bind an operation to its plan, storage and team.

The arithmetic path does not perform ordinary heap allocation or automatic
scratch expansion. Scratch comes from reserved arena regions and is divided
among recursive tasks. Exhausting a correctly planned region or violating a
mathematical invariant is a fatal diagnostic; it is not a request to silently
switch algorithms. Resource preparation reports failure before execution.

A binding holds its leases until unbind. In-place result views remain valid only
for the lifetime declared by that service. Inputs, output and workspace must obey
the overlap rules in the public header. Cache compatibility includes arithmetic
basis, completion state, scaling and support, not merely transform size.

A team has one controlling thread. Binding and top-level operations must obey
its idle/owner rules; internal worker scopes may split into subteams. Never share
an exclusive workspace between concurrent bindings. Read-only prepared tables
may be shared only through their explicit lifetime contract.

Consult include/sbn3/arena.h, team.h, product.h and the specific service header.
Some bindings are single-use; radix bindings explicitly support repeated execute.
