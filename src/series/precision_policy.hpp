// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "series/psr.hpp"
namespace sbn::v3::series {
inline constexpr size_t precision_guard_words = 4;
struct PrecisionBlock {
    sbn3_series_range range{};
    size_t precision = 0, next_precision = 0, exact_words = 0;
    unsigned need = 0;
};
bool precision_spec_supported(const PsrSpec &) noexcept;
size_t precision_words(const PsrSpec &, uint64_t at) noexcept;
// Same cuts and exact-suffix certificate as resident PSR. No block-count limit
// or allocation: callers may retain descriptors or generate them incrementally.
sbn3_query_result precision_block_query(const PsrSpec &, uint64_t begin, PrecisionBlock &) noexcept;
// The spec and optional exact_words storage must remain at stable addresses
// through query/prepare/execute of every derived finite plan.
FiniteFormula precision_formula(const PsrSpec &, const size_t *exact_words = nullptr) noexcept;
} // namespace sbn::v3::series
