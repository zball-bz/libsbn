#pragma once
#include "sbn3/product.h"
#include "common/shared_preparation.hpp"
namespace sbn::v3 {
class Frame;
enum ProgramContract : unsigned {
    program_bounded_inputs = 1, // runtime spans may be shorter than planned, zero-extended
    program_consume_inputs = 2  // every input read completes before any output write
};
struct Backend {
    uint64_t id;
    const char *name;
    sbn3_query_result (*query)(const sbn3_product_spec &, const sbn3_mul_options &, sbn3_mul_plan &,
                               sbn3_mul_info &);
    void (*bind)(const sbn3_mul_plan &, sbn3_arena &, const sbn3_lease &, const sbn3_lease &, sbn3_team &,
                 sbn3_mul_binding **);
    void (*execute)(sbn3_mul_binding *, sbn3_const_limbs, sbn3_const_limbs, sbn3_limbs);
    void (*execute_scope)(sbn3_mul_binding *, sbn3_team_scope *, sbn3_const_limbs, sbn3_const_limbs,
                          sbn3_limbs);
    void (*metrics)(const sbn3_mul_binding *, sbn3_mul_metrics &);
    void (*unbind)(sbn3_mul_binding *);
    sbn3_query_result (*product_query)(const sbn3_product_request &, const sbn3_mul_options &,
                                       sbn3_mul_plan &, sbn3_product_info &);
    void (*product_bind)(const sbn3_mul_plan &, sbn3_arena &, const sbn3_lease &, const sbn3_lease &,
                         sbn3_team &, const sbn3_spectrum *, const sbn3_spectrum *, sbn3_mul_binding **);
    void (*product_execute)(sbn3_mul_binding *, sbn3_team_scope *, const sbn3_product_inputs &, sbn3_limbs);
    void (*product_metrics)(const sbn3_mul_binding *, sbn3_product_metrics &);
    void (*prepare)(sbn3_mul_binding *, sbn3_const_limbs, unsigned, uint64_t, sbn3_arena &,
                    const sbn3_lease &, sbn3_spectrum **);
    void (*describe)(const sbn3_spectrum *, sbn3_spectrum_desc &);
    bool (*can_apply)(const sbn3_mul_plan &, const sbn3_spectrum *, unsigned);
    void (*spectrum_retain)(const sbn3_spectrum *);
    void (*spectrum_release)(const sbn3_spectrum *);
    sbn3_query_result (*spectrum_query)(const sbn3_mul_plan &, unsigned, uint64_t,
                                        sbn3_spectrum_desc &) = nullptr;
    void (*spectrum_reserve)(sbn3_mul_binding *, unsigned, uint64_t, sbn3_arena &, const sbn3_lease &,
                             sbn3_spectrum **) = nullptr;
    void (*spectrum_compute)(sbn3_mul_binding *, sbn3_spectrum *, sbn3_const_limbs) = nullptr;
    void (*spectrum_reserve_plan)(const sbn3_mul_plan &, unsigned, uint64_t, sbn3_arena &, const sbn3_lease &,
                                  sbn3_spectrum **) = nullptr;
    void (*spectrum_square)(sbn3_mul_binding *, sbn3_spectrum *, sbn3_const_limbs, sbn3_limbs) = nullptr;
    // Internal plain-MUL program: immutable preparation, caller-owned stage
    // scratch. Original binding/entry points retain their existing behavior.
    size_t (*program_bytes)(const sbn3_mul_plan &) = nullptr;
    const void *(*program_prepare)(const sbn3_mul_plan &, Frame &) = nullptr;
    void (*program_execute)(const void *, Frame &, sbn3_team_scope *, sbn3_const_limbs, sbn3_const_limbs,
                            sbn3_limbs) = nullptr;
    unsigned (*program_contract)(const sbn3_mul_plan &) = nullptr;
    // Internal two-product episode. A common operand is transformed once;
    // output0 may consume common/fresh0, but must preserve fresh1 until use.
    size_t (*program_pair_bytes)(const sbn3_mul_plan &) = nullptr;
    void (*program_pair_execute)(const void *, Frame &, sbn3_team_scope *, sbn3_const_limbs,
                                 sbn3_const_limbs, sbn3_const_limbs, sbn3_limbs, sbn3_limbs) = nullptr;
    SharedPreparation (*program_tables)(const sbn3_mul_plan &) = nullptr;
    size_t (*program_local_bytes)(const sbn3_mul_plan &) = nullptr;
    const void *(*program_tables_prepare)(const sbn3_mul_plan &, Frame &) = nullptr;
    const void *(*program_prepare_shared)(const sbn3_mul_plan &, Frame &, const void *) = nullptr;
    // Canonical replay parameters, including the geometry policy used to
    // produce full/rowscale. Consumers must not infer these from C/M2 alone.
    sbn3_mul_options (*program_options)(const sbn3_mul_plan &) = nullptr;
    // Build a reserved term-0 spectrum and perform its first cached MUL.
    // The spectrum is immutable/ready only after the whole episode returns.
    void (*spectrum_multiply)(sbn3_mul_binding *, sbn3_spectrum *, sbn3_const_limbs,
                              sbn3_const_limbs, sbn3_limbs) = nullptr;
};
const Backend *backend_lookup(uint64_t) noexcept;
const Backend &short_backend() noexcept;
const Backend &fft_backend() noexcept;
const Backend &np4_backend() noexcept;
const Backend &np5_backend() noexcept;
const Backend &np6_backend() noexcept;
const Backend &np7_backend() noexcept;
const Backend &np8_backend() noexcept;
const Backend &np9_backend() noexcept;
const Backend &np10_backend() noexcept;
} // namespace sbn::v3
struct sbn3_mul_binding {
    const sbn::v3::Backend *backend;
    uint64_t marker;
    // Bound once; normal execute is one indirect jump, with no algorithm
    // selection. Diagnostic dispatch uses the registered backend instead.
    void (*execute_fast)(sbn3_mul_binding *, const uint64_t *, const uint64_t *, uint64_t *) = nullptr;
};

struct sbn3_spectrum {
    const sbn::v3::Backend *backend;
    uint64_t marker;
};
