// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/integer.hpp"
namespace sbn::v3::external {
struct SignedLiftPlan {
    size_t modulus_words=0,storage_bytes=0,transfer_bytes=0;
    size_t io[3]{},index[3]{},data=0,subtract=0;
    size_t subtract_words=0,subtract_shift=0;
    product::Window window{};
    product::MagnitudeBound bound{};
    uint64_t output_bytes=0,output_allocation=0,write_bytes=0;
};
// Unique signed lift of x mod B^k, then an exact magnitude window. The caller
// proves |x|<bound<B^k/2. All lifted high words are checked during the scan.
bool query_signed_lift(size_t modulus_words,product::Window,product::MagnitudeBound,SignedLiftPlan &,
                       size_t subtract_words=0,size_t subtract_shift=0) noexcept;
// Optional nonnegative subtrahend is reduced modulo B^modulus_words. Its
// exponent is the planned nonnegative whole-word shift. Sign is resolved
// before streaming absolute words; no full cancellation value is materialized.
IoError run_signed_lift(const SignedLiftPlan &,Transport &,ValueStore &,const StoredInteger &,
                       StoredInteger &,std::span<uint8_t>,const LinearInput *subtract=nullptr) noexcept;
// Planned constant-pattern output used for the reciprocal endpoint clamp.
// low is 0 or UINT64_MAX; the final limb is supplied separately. No reads.
IoError write_word_pattern(Transport &,ValueStore &,size_t words,uint64_t low,uint64_t high,
                          StoredInteger &,std::span<uint8_t>) noexcept;
struct FixedOutputPlan {
    SourceWindow input{};
    int64_t exponent2=0;
    size_t fractional=0,storage_bytes=0,transfer_bytes=0,io[2]{},index[2]{},data=0;
    uint64_t discarded_bits=0,error_units=0,output_bytes=0,output_allocation=0,write_bytes=0,integer_addend=0;
};
// Given nonnegative true value V and |V-X*2^e| < error_units*2^e,
// certify floor(V*B^fractional) by guard separation. Ambiguous mathematical
// output is a fatal certificate failure, not a retry or silently rounded result.
// Returned stored value is the certified floor times B^-fractional, plus an
// optional exact nonnegative integer addend fused into the final output word.
// The resulting integer part must fit u64; overflow is a fatal math diagnostic.
bool query_fixed_output(const StoredInteger &,size_t fractional,uint64_t error_units,FixedOutputPlan &,uint64_t integer_addend=0) noexcept;
// Shape-only form. Output magnitude must fit the stated u64 integer part;
// run checks actual magnitude before IO publication.
bool query_fixed_output_source(SourceWindow,int64_t exponent2,size_t fractional,uint64_t error_units,
                               FixedOutputPlan &,uint64_t integer_addend=0) noexcept;
IoError run_fixed_output(const FixedOutputPlan &,Transport &,ValueStore &,const StoredInteger &,
                         StoredInteger &,std::span<uint8_t>) noexcept;
} // namespace sbn::v3::external
