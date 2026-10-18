// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/inverse_planes.hpp"
#include "external/carry_output.hpp"
#include "product/window.hpp"
#include "external/source_window.hpp"
namespace sbn::v3::external {
struct CyclicResidualSpec {
    size_t minimum_words=0,subtract_word=0; // product - B^subtract_word modulo B^K-1
    product::MagnitudeBound bound{}; // uniquely recovered signed residual
    SourceWindow subtract_source{};
    bool stored_subtract=false; // subtract source * B^subtract_word instead of the unit
    explicit operator bool() const noexcept {return minimum_words!=0;}
};
struct CyclicResidualResult {
    bool complemented=false; // stored R lifts to -(ones-complement R); all-ones lifts to zero
    WordExtent magnitude{}; // full absolute residual, independent of output projection
};
// Exact linear-product window recovered from a shorter minus ring. The
// planner leaves a nonempty gap above wrapped high words. A zero gap needs
// the preplanned linear witness; it is never silently treated as exact.
struct MiddleWindowSpec {
    size_t first=0,end=0;
    explicit operator bool() const noexcept {return end!=0;}
};
struct MiddleWindowResult {bool exact=false;};
struct IntegerEmitPlan {
    const product::CrtBackend *crt=nullptr;
    unsigned trunk_bits=0,workers=0,team_workers=0;
    size_t rows=0,inner_cells=0,stripe_cells=0;
    size_t row_slots=0;
    CarryOutputPlan carry{};
    size_t storage_bytes=0,prepared=0,descriptors=0,fragments=0,assembly=0,tails=0,rows_buffer=0,row_buffer_stride=0;
    product::MagnitudeBound bound{};
    uint64_t scalar=1; // post-CRT integer scaling; caller proves output support
    CyclicResidualSpec cyclic{};
    size_t subtract_io=0,subtract_index=0,subtract_rows=0,subtract_index_bytes=0,subtract_capacity_words=0;
    uint64_t subtract_read_bytes=0;
    bool pipeline=false; // stream existing row slots to the single carry/IO owner
    bool preconditioned=false; // residues include crt->input_scale(q)
    MiddleWindowSpec middle{};
    size_t middle_guard_first=0;
};
bool query_integer_emit(unsigned primes,unsigned T,size_t rows,size_t inner_cells,size_t stripe_cells,
                        uint64_t output_bytes,unsigned workers,IntegerEmitPlan &,CyclicResidualSpec={},size_t row_slots=0,
                        bool ordered_rows=false,MiddleWindowSpec middle={},size_t guard_first=0) noexcept;
bool rebind_emit_subtract(IntegerEmitPlan &,SourceWindow) noexcept;
struct IntegerEmitBinding {
    const IntegerEmitPlan *plan=nullptr;
    std::span<uint8_t> workspace{};
    ChunkWriter *writer=nullptr;
    sbn3_team *team=nullptr;
    CarryOutput carry{};
    ValueReader subtract_reader{};
    uint64_t subtract_residue=0; // shifted ORIGINAL subtrahend, not its ring representative
};
IoError prepare_integer_emit(const IntegerEmitPlan &,std::span<uint8_t>,ChunkWriter &,sbn3_team &,IntegerEmitBinding &,
                            const Region *subtract=nullptr) noexcept;
// Internal stage primitives. Slots remain exclusive until submission completes;
// one IO owner mutates subtraction/carry/writer state, compute owns each row slot.
IoError load_integer_subtract_row(IntegerEmitBinding &,const CoefficientStripe &,size_t row,unsigned slot) noexcept;
void compute_integer_row(IntegerEmitBinding &,const CoefficientStripe &,size_t row,unsigned slot) noexcept;
// Apply the same scalar/cyclic postprocessing to a row already reconstructed
// by the fused inverse+CRT kernel in its assigned output/tail slot.
void finish_integer_row(IntegerEmitBinding &,const CoefficientStripe &,size_t row,unsigned slot) noexcept;
IoError submit_integer_row(IntegerEmitBinding &,const CoefficientStripe &,size_t row,unsigned slot) noexcept;
IoError emit_integer_stripe(IntegerEmitBinding &,const CoefficientStripe &) noexcept;
IoError finish_integer_emit(IntegerEmitBinding &,uint64_t expected_m61,CyclicResidualResult * = nullptr,
                           MiddleWindowResult * = nullptr) noexcept;
} // namespace sbn::v3::external
