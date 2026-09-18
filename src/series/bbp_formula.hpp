#pragma once
#include "series/formula_def.hpp"
#include "sbn3/bitwindow.h"
// One immutable BBP definition, three evaluation requests.
//   C = sum_j sum_{k>=0} numerator_j * (-1)^(alternating_j*k) * 2^(shift_j - stride_j*k)
//                       / (scale_j * (a_j*k + b_j)^power_j)
// The definition is the public window stream list, so the bit-window backend
// consumes it unchanged. The exact finite backend maps it onto the BinaryBBP
// recipe over an interleaved virtual index i = k*count + j (term k of stream
// j); the bounded prefix reuses the PSR with an additive attenuation
// certificate. Switching backends re-queries the same definition; nothing
// converts a running tree or cache.
namespace sbn::v3::series {
constexpr unsigned max_bbp_streams = 32;
struct BbpFormula {
    sbn3_bbp_stream stream[max_bbp_streams]{};
    unsigned count = 0;
};
struct DataBbp {
    BbpFormula def{};
    uint64_t terms = 0;        // per stream: k in [0, terms); virtual domain [0, terms*count)
    const char *rejection = nullptr;
    unsigned leaf_words = 0;   // widest single leaf value
    int64_t shift_max = 0, shift_min = 0;
    uint32_t stride_min = 0;
    unsigned amplitude_bits = 0; // sum_j |numerator_j| 2^(shift_j-shift_min) / Q_j(k) < 2^amplitude_bits for all k
    uint64_t formula_id = 0, parameter_id = 0;
    ProductAnalysis Q[max_bbp_streams]; // per stream denominator envelope on k in [0, terms)
    // --- capacity ---
    sbn3_query_result bounds(sbn3_series_range, uint64_t max_terms, unsigned need, sbn3_series_shape &) const noexcept;
    // --- contribution ---
    // Additive certificate on the virtual index: the sum of all terms at
    // virtual index >= i has magnitude < 2^(amplitude_bits + shift_min + 1 - attenuation_bits(i)).
    // Zero at i == 0, monotone nondecreasing (stride_min per complete k row).
    uint64_t attenuation_bits(uint64_t i) const noexcept;
    // --- performance ---
    double work(sbn3_series_range) const noexcept;
    // --- leaf ---
    void leaf(uint64_t i, unsigned need, sbn3_series_values &) const noexcept;
};
// Per-stream term count needed so that every omitted term row is attenuated
// by at least `bits` beyond the amplitude; 0 when no stream row can supply it.
uint64_t bbp_terms_for(const BbpFormula &, uint64_t bits) noexcept;
bool bbp_prepare(const BbpFormula &, uint64_t terms_per_stream, DataBbp &out) noexcept;
FiniteFormula bbp_finite_formula(const DataBbp &) noexcept;
// Built-in definitions shared with the window backend.
BbpFormula bbp_log2() noexcept;          // ln 2 = sum_{k>=0} 2^(-k-1)/(k+1)
BbpFormula bbp_pi_standard() noexcept;   // Bailey-Borwein-Plouffe four streams, stride 4
BbpFormula bbp_pi_bellard() noexcept;    // Bellard seven streams, stride 10, alternating
BbpFormula bbp_catalan_huvent() noexcept; // Huvent nine streams (same rows as sbn3_catalan_bbp)
} // namespace sbn::v3::series
