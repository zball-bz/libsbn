#pragma once
#include "sbn3/formula.h"
#include "series/terminal.hpp"
namespace sbn::v3::series {
// The terminal plan is a pure function of the precision and the width, not of
// the formula: components of one composition planned at the same precision
// share it instead of planning the same division once each. Caller-owned
// scratch, value-initialized before the first component; it lives only
// through the composition's query.
struct FormulaTerminalCache {
    bool valid = false;
    size_t fractional_limbs = 0;
    unsigned workers = 0;
    RatioTerminalPlan plan{};
};
// sbn3_formula_query with an optional shared terminal plan (nullptr: none).
sbn3_query_result formula_query_shared(const sbn3_formula_def *, size_t n, const sbn3_formula_options *, void *object,
                                       sbn3_formula_plan *, sbn3_formula_info *, FormulaTerminalCache *) noexcept;
// Private composition entry: nonnegative dyadic value at the query's
// fractional precision, with absolute error <2 units of its last limb.
// No component-level output-boundary guard; the combination propagates this
// error and certifies its own final output. Lifetime is until unbind.
sbn3_const_limbs formula_execute_approximate(sbn3_formula_binding *) noexcept;
// The same value as a writable slot the caller may consume until unbind:
// data[0 .. output_limbs) is the value, capacity leaves at least seven spare
// limbs above it (for example a coefficient multiply carry).
sbn3_limbs formula_execute_consumable(sbn3_formula_binding *) noexcept;
} // namespace sbn::v3::series
