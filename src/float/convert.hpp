// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"

// The public integer-output entries are declared in sbn3/float.hpp. Their
// implementation uses no prepared object, workspace or allocation: exact
// dyadic shifts and a captured guard/sticky/parity rounding decision suffice.
