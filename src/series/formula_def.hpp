#pragma once
#include "series/arithmetic.hpp"
// Structured formula definitions for the finite binary-splitting service.
//
// Mathematics (half-open index intervals, fixed recursion order):
//   leaf k:  T = P(k), D = Q(k), U = R(k)          (CommonP2B3)
//            T = P(k), D = Q(k), U = 1 implied      (Hyperdescent: R == 1, Q > 0)
//            T = P(k) * 2^(shift - stride*k), D = Q(k)   (BinaryBBP, exact exponent)
//   F[a,b)(x) = (T + U x) / D,   F[a,b) = F[a,m) o F[m,b)
//   value     = T/D = sum_{k in [a,b)} P(k)/Q(k) * prod_{j in [a,k)} R(j)/Q(j)
// Each of P, Q, R is  c * (-1)^(alternating*k) * prod_i (a_i*k + b_i)^power_i.
// A definition only adds data and analysis facts; the merge kernels, the
// scheduler and the resource planning in arithmetic.cpp are untouched.
namespace sbn::v3::series {
struct LinearFactor {
    uint64_t a = 0; // a >= 0
    int64_t b = 0;  // signed offset; a*k+b evaluated exactly in 128 bits
    unsigned power = 1;
};
constexpr unsigned max_formula_factors = 6, max_polynomial_degree = 4;
struct FactorProduct {
    uint64_t constant_low = 1, constant_high = 0; // |c| < 2^128, c != 0
    bool negative = false;                        // sign of c
    bool alternating = false;                     // extra (-1)^k
    unsigned count = 0;
    LinearFactor factor[max_formula_factors]{};
    // Optional integer polynomial multiplier sum_i coefficient[i] k^i
    // (degree 0 means none). Numerators only; evaluated by Horner in 128 bits,
    // so |value| must stay below 2^127 on the domain (checked at preparation).
    unsigned degree = 0;
    int64_t coefficient[max_polynomial_degree + 1]{};
};
// Optional explicit leaf at k == begin (for example Chudnovsky's k = 0 term
// where Q(0) = 0). Magnitudes are single words; U must be 1 for Hyperdescent.
struct FirstTerm {
    uint64_t t = 0, d = 1, u = 1;
    bool t_negative = false;
};
struct FormulaDef {
    sbn3_series_recipe recipe = SBN3_SERIES_HYPERDESCENT;
    uint64_t begin = 1; // first index of the domain
    FactorProduct P, Q, R;
    int64_t shift = 0;   // BinaryBBP only
    uint32_t stride = 0; // BinaryBBP only, >= 1
    bool explicit_first = false;
    FirstTerm first{};
    // Preparation-time known power of two (Common/BinaryBBP): the 2-adic
    // valuation of Q's constant is moved out of D into the binary exponents of
    // T and U of every factored leaf, so T/D and U/D are unchanged while D's
    // mantissa loses that many bits per term. Exact values only; the raw
    // T/D/U words differ from the unextracted form (ratio identity, not bytes).
    bool extract_twos = false;
};
// Cumulative bit lengths of one monotone linear form a*k+c (a >= 0, c >= 0,
// a*k+c >= 1) on [begin, end). Built once; sums are O(log) lookups.
struct FactorBitTable {
    uint64_t a = 0, c = 0, begin = 0, end = 0;
    unsigned first_width = 0, count = 0;
    uint64_t first[128]{};      // first index whose value has width first_width + j
    uint64_t cumulative[128]{}; // sum of widths on [begin, first[j])
    void build(uint64_t a, uint64_t c, uint64_t begin, uint64_t end) noexcept;
    uint64_t prefix(uint64_t k) const noexcept; // sum on [begin, k), begin <= k <= end
    uint64_t sum(uint64_t lo, uint64_t hi) const noexcept { return prefix(hi) - prefix(lo); }
    unsigned width(uint64_t k) const noexcept;
};
// Per-product analysis: upper envelope |X(k)| < 2^up(k) and, for sign-definite
// products, lower envelope |X(k)| >= 2^(lo(k)-1). Both are integer bounds
// from factor bit lengths; the upper sum also bounds the product's bit length.
struct ProductAnalysis {
    unsigned constant_up = 0, constant_lo = 0; // per-term constant contributions
    unsigned factors = 0, degree = 0;          // degree = sum of powers
    unsigned power[max_formula_factors]{};     // multiplicity of each factor
    bool sign_definite = false;                // value never zero/changes sign on the domain
    bool has_lower = false;
    FactorBitTable up[max_formula_factors];    // a*k+|b|
    FactorBitTable lo[max_formula_factors];    // a*k+b (only when sign_definite)
    unsigned twos = 0;                         // 2-adic valuation of the constant
    uint64_t up_sum(uint64_t lo_k, uint64_t hi_k) const noexcept;
    uint64_t lo_sum(uint64_t lo_k, uint64_t hi_k) const noexcept;
    unsigned up_at(uint64_t k) const noexcept;
    unsigned lo_at(uint64_t k) const noexcept;
};
enum class LeafKind : unsigned { Factored = 0, HyperWordBatch = 1, CommonWordBatch = 2, CommonWideBatch = 3 };
// Contribution certificate table: per-term attenuation floors on sub-octave
// index buckets, in units of 2^-fraction_bits bit. Each bucket's floor is a
// pointwise lower bound of log2|Q(k)/R(k)| for every k in the bucket: the
// value at the bucket's first or last index when the ratio is proven monotone
// on the domain (see ratio_direction), otherwise Q at the bucket start
// against R at its end. prefix() is therefore a lower bound of the cumulative
// attenuation from the first factored index, monotone in k. Built from long
// double logarithms with an explicit rounding allowance.
struct AttenuationTable {
    static constexpr unsigned fraction_bits = 8;
    unsigned count = 0;
    uint64_t begin = 0, end = 0;
    uint64_t first[2048]{};
    uint64_t cumulative[2048]{}; // whole bits on [begin, first[j])
    uint32_t per_term[2048]{};   // bucket floor in 2^-fraction_bits bit
    uint8_t fraction[2048]{};    // remainder of cumulative[j] in 2^-fraction_bits bit
    uint64_t prefix(uint64_t k) const noexcept; // whole bits on [begin, k)
};
// Immutable prepared formula. Caller-owned storage, no allocation, alive
// through query/prepare/execute of every FiniteFormula derived from it.
struct DataFormula {
    FormulaDef def{};
    uint64_t index_end = 0; // domain [def.begin, index_end)
    const char *rejection = nullptr;
    // Capacity facts (sound bounds used for allocation).
    unsigned leaf_words = 0;  // max words of any single leaf value
    bool contraction = false; // |R(k)| <= |Q(k)| on the whole domain (bucketed or integer certificate)
    bool integer_contraction = false; // the integer bit-length certificate alone holds (envelope use)
    unsigned ratio_bits = 0;  // |P(k)/Q(k)| < 2^ratio_bits on the whole domain
    // Contribution facts (used by bounded/limited consumers, never for allocation).
    uint64_t attenuation_minimum = 0; // min per-term attenuation bits (integer bit-length certificate, envelope use)
    uint64_t first_attenuation = 0;   // exact attenuation bits of the explicit first term
    AttenuationTable attenuation{};   // bucketed real-log certificate used by attenuation_bits()
    // +1: log2|Q(k)/R(k)| is nondecreasing on the factored domain, -1: nonincreasing, 0: not established.
    int ratio_direction = 0;
    bool probe_only = false; // prepared without the attenuation certificate (stopping-index search only)
    unsigned twos_shift = 0;   // bits removed from every factored D (extract_twos)
    // Performance facts (proxies only; never weaken a bound).
    bool smooth_split = false; // equal-mass serial policy and envelope available
    LeafKind leaf_kind = LeafKind::Factored;
    unsigned batch_words = 1; // exact batch growth bound per term, in words (batch leaf kinds)
    uint64_t formula_id = 0, parameter_id = 0;
    ProductAnalysis P, Q, R;
    uint64_t first_positive = 0; // first factored index (begin + explicit_first)
    unsigned first_t_bits = 0, first_d_bits = 0, first_u_bits = 0;

    // --- capacity ---
    // span_only: the exact-span fact, with the factor products measured by the
    // logarithmic estimator instead of the integer bit-length sums that size
    // capacities (position-independent, shared by equal subtrees).
    sbn3_query_result bounds(sbn3_series_range, uint64_t max_terms, unsigned need, sbn3_series_shape &,
                             bool normalized = false, bool span_only = false) const noexcept;
    sbn3_query_result serial_envelope(sbn3_series_range, unsigned depth, unsigned need, uint64_t &max_terms,
                                      sbn3_series_shape &, bool normalized = false) const noexcept;
    // --- contribution / tail ---
    // Root-anchored certificate, valid when contraction holds: for every
    // begin <= a <= index_end, |U(begin,a)/D(begin,a)| <= 2^-attenuation_bits(a)
    // (Common/Hyper). This is the weight of an error injected at a node that
    // starts at a (series-precision-planner-math (17)/(22)). Zero at def.begin,
    // monotone nondecreasing, O(log). Differences of two values are not a
    // certificate for the interval between them (they may exceed the true
    // interval attenuation by less than one bit).
    uint64_t attenuation_bits(uint64_t a) const noexcept;
    // Smallest index N in the domain whose total attenuation lower bound
    // (a real-valued integral bound, no increment property) reaches `bits`,
    // or 0 when the domain cannot supply it. The consumer adds amplitude and guard.
    uint64_t tail_terms(uint64_t bits) const noexcept;
    // Real-valued lower bound of sum_{k in [begin,a)} log2|Q(k)/R(k)| with a
    // native-libm rounding allowance subtracted; monotone in a.
    long double attenuation_total(uint64_t a) const noexcept;
    // --- infinite tail (structural, independent of the prepared domain) ---
    // Convergence is decided from the factored structure of P, Q, R alone:
    // for all j >= N, |R(j)/Q(j)| <= rho_N and |P(j)/Q(j)| <= (A_P/A_Q) j^(deg P - deg Q),
    // with rho_N < 1 required. Then |sum_{k>=N} term_k| < 2^-infinite_tail_bits(N).
    // A definition whose term ratio does not tend to a limit below 1 is not
    // certifiable here (tail_rejection names the reason). Explicit first terms
    // and signs do not matter; the bound uses magnitudes only.
    // Real-valued log2 upper bound of |T/D| over any subrange of the prepared
    // domain: sum_k |P(k)/Q(k)| prod_{j<k}|R(j)/Q(j)| summed over the
    // attenuation buckets (requires contraction), plus the explicit first term.
    long double value_log2_bound() const noexcept;
    const char *tail_rejection = nullptr; // nullptr: the infinite tail can be certified
    long double infinite_tail_bits(uint64_t N) const noexcept; // requires N in [first_positive, index_end]
    // Smallest N in the domain with infinite_tail_bits(N) >= bits, or 0 when
    // the tail cannot be certified within the domain (or at all).
    uint64_t infinite_tail_terms(uint64_t bits) const noexcept;
    // --- performance ---
    double work(sbn3_series_range) const noexcept;
    uint64_t split_point(sbn3_series_range, double fraction) const noexcept;
    uint64_t mass(sbn3_series_range) const noexcept; // exact integer upper bit mass of Q
    // --- leaf programs ---
    void leaf(uint64_t k, unsigned need, sbn3_series_values &) const noexcept;
    void batch(sbn3_series_range, unsigned need, sbn3_series_values &) const noexcept;
};
// Analyze a definition for the domain [def.begin, index_end). Returns false
// with out.rejection set when the definition is outside the supported class.
// probe_only: structure, capacity facts and the structural tail verdict on a
// (typically huge) domain that is only searched for the stopping index. The
// bucketed attenuation certificate, which costs a table over the whole domain,
// is not built and contraction is not decided; such an object answers
// infinite_tail_terms() from the cumulative integral bound and must not be
// executed or asked for attenuation_bits(). The caller prepares the chosen
// prefix again in full.
bool data_formula_prepare(const FormulaDef &, uint64_t index_end, DataFormula &out, bool probe_only = false) noexcept;
// Largest index_end this definition can be prepared on: 2^48, lowered when a
// numerator polynomial would exceed the 127-bit Horner evaluation earlier.
uint64_t formula_domain_limit(const FormulaDef &) noexcept;
// The FiniteFormula borrows the prepared object; no per-value vtable.
FiniteFormula data_finite_formula(const DataFormula &) noexcept;
// Built-in definitions expressed as data (same conventions as the typed
// Formula kinds, so exact T/D/U agree bit for bit).
FormulaDef euler_definition() noexcept;                         // e - 1 = sum_{k>=1} 1/k!
FormulaDef chudnovsky_definition() noexcept;                    // k = 0 explicit (A,1,1); k >= 1 factored
FormulaDef binary_log_definition(unsigned radix_bits) noexcept; // sum_{k>=1} 2^(-r k)/k
FormulaDef exp_reciprocal_definition(uint64_t m) noexcept;      // exp(1/m) = sum_{k>=0} 1/(m^k k!), k=0 explicit
FormulaDef exp_minus_one_definition() noexcept;                 // exp(-1) = sum_{k>=0} (-1)^k/k!, k=0 explicit
} // namespace sbn::v3::series
