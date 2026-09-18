/* Imported from libsbn/include/sbn/vec.h; arithmetic preserved, private namespace and prepared Frame adapter. */
#pragma once
namespace sbn::v3::u52 {

/* sbn/vec.h - the AVX-512 vector DSL (port of the old tree's types.h with
 * every name prefixed: sb_ for ops/types, SB_ for utility macros).
 *
 * Dual-language: compiles as C11 (gnu) and C++20. The type-dispatching
 * entries (sb_add/sb_sub/sb_mul over int/double vectors, the k-mask ops)
 * use _Generic in C and inline overloads in C++ with IDENTICAL spellings.
 * Everything else stays object-like/variadic macros expanding straight to
 * intrinsics -- zero abstraction cost by construction.
 *
 * PORT MAP (old types.h name -> new name), the sed table for migrating
 * kernel headers out of the old tree:
 *   _limb/_hlimb/plimb/phlimb   -> sb_limb / sb_hlimb / sb_plimb / sb_phlimb
 *   _vec/pvec/cpvec/_dvec/pdvec -> sb_vec / sb_pvec / sb_cpvec / sb_dvec / sb_pdvec
 *   load_vec / store_vec        -> sb_load / sb_store
 *   as_ivec / as_dvec           -> sb_as_ivec / sb_as_dvec
 *   add/sub/mul                 -> sb_add / sb_sub / sb_mul   (dual C/C++)
 *   andnot/and_v/or_v/xor_v     -> sb_andn / sb_and / sb_or / sb_xor
 *   srlv/sllv/srli/slli/srai    -> sb_srlv / sb_sllv / sb_srli / sb_slli / sb_srai
 *   minu/maxu                   -> sb_minu / sb_maxu
 *   maskz/blend/blend8..64      -> sb_maskz / sb_blend / sb_blend8..64
 *   ternary/or3/xor3/maj/mux    -> sb_ternary / sb_or3 / sb_xor3 / sb_maj / sb_mux
 *   bitselect                   -> sb_bitselect
 *   zero/dzero/ones             -> sb_zero / sb_dzero / sb_ones
 *   splat_load(_m)/set1_64/32/d -> sb_splat_load(_m) / sb_set1_64 / sb_set1_32 / sb_set1_d
 *   setr_64/setr_32             -> sb_setr_64 / sb_setr_32
 *   eq/neq/ltu/leu/gtu/geu      -> sb_eq / sb_neq / sb_ltu / sb_leu / sb_gtu / sb_geu
 *   is_zero                     -> sb_is_zero
 *   k8/k16/k32/k64              -> sb_k8 / sb_k16 / sb_k32 / sb_k64
 *   kadd/kand/kandn/kor/kxor/kxnor/knot -> sb_kadd/... (dual C/C++)
 *   klsh/krsh                   -> sb_klsh / sb_krsh (C++: plain shifts, same value)
 *   alignr64/alignr8            -> sb_alignr64 / sb_alignr8
 *   lane_lsh/lane_rsh/lane_rotl -> sb_lane_lsh / sb_lane_rsh / sb_lane_rotl
 *   shuffle8/perm64/perm64x2(z)/perm32/permb -> sb_shuffle8 / sb_perm64 / ...
 *   unpacklo64/unpackhi64/shufi64x2 -> sb_unpacklo64 / sb_unpackhi64 / sb_shufi64x2
 *   fshl64i/fshr64i/fshl64v/fshr64v -> sb_fshl64i / sb_fshr64i / sb_fshl64v / sb_fshr64v
 *   madd52lo/madd52hi           -> sb_madd52lo / sb_madd52hi
 *   adc/sbb                     -> sb_adc / sb_sbb
 *   SWAP                        -> SB_SWAP
 *   MASK52                      -> SB_MASK52()
 *   CAT/CAT3/NARGS/EVAL/EMPTY/de_args/list_cat -> SB_CAT/... (internal)
 */




#ifndef __cplusplus


#endif

typedef uint64_t  sb_limb;
typedef uint32_t  sb_hlimb;
typedef uint64_t *sb_plimb;
typedef uint32_t *sb_phlimb;

#define SB_SWAP(T, x, y) { T sb__s = (x); (x) = (y); (y) = sb__s; }

/* 64-byte (AVX-512 line) alignment specifier, dual-language: keyword in
 * C++20, _Alignas keyword in C11 (no header). Used as a declaration prefix,
 * e.g. `SB_ALIGN64 static const uint8_t perm[64] = {...};` (migration C3.4:
 * replaces bare alignas(64) so both the C and C++ front-ends accept it). */
#ifdef __cplusplus
#define SB_ALIGN64 alignas(64)
#else
#define SB_ALIGN64 _Alignas(64)
#endif

#define SB_EVAL(x) x
#define SB_EMPTY
#define SB_CAT_(x, y) x##y
#define SB_CAT(x, y) SB_CAT_(x, y)
#define SB_CAT3_(x, y, z) x##y##z
#define SB_CAT3(x, y, z) SB_CAT3_(x, y, z)
#define SB_DE_ARGS(...)        __VA_ARGS__
#define SB_LIST_CAT_0(a, ...)  (SB_DE_ARGS a __VA_OPT__(,) __VA_ARGS__)
#define SB_LIST_CAT(a, b)      SB_LIST_CAT_0(a, SB_DE_ARGS b)
#define SB_NARGS_(_1, _2, _3, _4, _5, _6, N, ...) N
#define SB_NARGS(...) SB_NARGS_(__VA_ARGS__, 6, 5, 4, 3, 2, 1)

#define SB_VEC_BITS 512
#define sb__vec_base    SB_CAT(__m, SB_VEC_BITS)
#define sb__vec_tok     SB_CAT(si, SB_VEC_BITS)
#define sb__vec_fn_typ  SB_CAT3(_mm, SB_VEC_BITS, _)
#define sb__fn(x)       SB_CAT(sb__vec_fn_typ, x)

typedef SB_CAT(sb__vec_base, i)  sb_vec;
typedef SB_CAT(sb__vec_base, i) *sb_pvec;
typedef SB_CAT(sb__vec_base, d)  sb_dvec;
typedef SB_CAT(sb__vec_base, d) *sb_pdvec;
typedef const SB_CAT(sb__vec_base, i) *sb_cpvec;

#define sb_as_ivec(x) sb__fn(SB_CAT(castpd_, sb__vec_tok))(x)
#define sb_as_dvec(x) sb__fn(SB_CAT3(cast, sb__vec_tok, _pd))(x)

#define sb_load(...) SB_CAT(sb__load_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__load_1(p) sb__fn(SB_CAT(loadu_, sb__vec_tok))((const void *)(p))
#define sb__load_2(p, mask) \
  sb__fn(maskz_loadu_epi64)((__mmask8)(mask), (const void *)(p))

#define sb_store(...) SB_CAT(sb__store_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__store_2(p, v) sb__fn(SB_CAT(storeu_, sb__vec_tok))((void *)(p), (v))
#define sb__store_3(p, v, mask) \
  sb__fn(mask_storeu_epi64)((void *)(p), (__mmask8)(mask), (v))

/* arity dispatch: (a,b) raw | (a,b,mask) maskz | (a,b,mask,src) merge */
#define sb__mask_fn(len, x) SB_CAT(sb__mask_fn_, len)(x)
#define sb__mask_fn_1(x) sb__fn(x)
#define sb__mask_fn_2(x) sb__fn(SB_CAT(maskz_, x))
#define sb__mask_fn_3(x) sb__fn(SB_CAT(mask_, x))
#define sb__mask_call(...) SB_CAT(sb__mask_call_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__mask_call_2(a, b) a, b
#define sb__mask_call_3(a, b, mask) mask, a, b
#define sb__mask_call_4(a, b, mask, src) src, mask, a, b
#define sb__call_mask_fn(x, a, b, ...) \
  sb__mask_fn(SB_NARGS(0, ##__VA_ARGS__), x)(sb__mask_call(a, b, ##__VA_ARGS__))

/* ---- type-dispatching arithmetic: dual C/C++ --------------------------- */
#ifdef __cplusplus
static inline sb_vec  sb_add(sb_vec a, sb_vec b){ return _mm512_add_epi64(a, b); }
static inline sb_vec  sb_add(sb_vec a, sb_vec b, __mmask8 m){ return _mm512_maskz_add_epi64(m, a, b); }
static inline sb_vec  sb_add(sb_vec a, sb_vec b, __mmask8 m, sb_vec src){ return _mm512_mask_add_epi64(src, m, a, b); }
static inline sb_dvec sb_add(sb_dvec a, sb_dvec b){ return _mm512_add_pd(a, b); }
static inline sb_dvec sb_add(sb_dvec a, sb_dvec b, __mmask8 m){ return _mm512_maskz_add_pd(m, a, b); }
static inline sb_dvec sb_add(sb_dvec a, sb_dvec b, __mmask8 m, sb_dvec src){ return _mm512_mask_add_pd(src, m, a, b); }
static inline sb_vec  sb_sub(sb_vec a, sb_vec b){ return _mm512_sub_epi64(a, b); }
static inline sb_vec  sb_sub(sb_vec a, sb_vec b, __mmask8 m){ return _mm512_maskz_sub_epi64(m, a, b); }
static inline sb_vec  sb_sub(sb_vec a, sb_vec b, __mmask8 m, sb_vec src){ return _mm512_mask_sub_epi64(src, m, a, b); }
static inline sb_dvec sb_sub(sb_dvec a, sb_dvec b){ return _mm512_sub_pd(a, b); }
static inline sb_dvec sb_sub(sb_dvec a, sb_dvec b, __mmask8 m){ return _mm512_maskz_sub_pd(m, a, b); }
static inline sb_dvec sb_sub(sb_dvec a, sb_dvec b, __mmask8 m, sb_dvec src){ return _mm512_mask_sub_pd(src, m, a, b); }
static inline sb_vec  sb_mul(sb_vec a, sb_vec b){ return _mm512_mul_epi32(a, b); }
static inline sb_vec  sb_mul(sb_vec a, sb_vec b, __mmask8 m){ return _mm512_maskz_mul_epi32(m, a, b); }
static inline sb_vec  sb_mul(sb_vec a, sb_vec b, __mmask8 m, sb_vec src){ return _mm512_mask_mul_epi32(src, m, a, b); }
static inline sb_dvec sb_mul(sb_dvec a, sb_dvec b){ return _mm512_mul_pd(a, b); }
static inline sb_dvec sb_mul(sb_dvec a, sb_dvec b, __mmask8 m){ return _mm512_maskz_mul_pd(m, a, b); }
static inline sb_dvec sb_mul(sb_dvec a, sb_dvec b, __mmask8 m, sb_dvec src){ return _mm512_mask_mul_pd(src, m, a, b); }
#else
#define sb__generic_mask_fn(ivec_fn, dvec_fn, a, b, ...)                       \
  _Generic((a),                                                                \
      sb_vec: sb__mask_fn(SB_NARGS(0, ##__VA_ARGS__), ivec_fn),                \
      sb_dvec: sb__mask_fn(SB_NARGS(0, ##__VA_ARGS__), dvec_fn))(              \
      sb__mask_call(a, b, ##__VA_ARGS__))
#define sb_add(a, b, ...) sb__generic_mask_fn(add_epi64, add_pd, (a), (b), ##__VA_ARGS__)
#define sb_sub(a, b, ...) sb__generic_mask_fn(sub_epi64, sub_pd, (a), (b), ##__VA_ARGS__)
#define sb_mul(a, b, ...) sb__generic_mask_fn(mul_epi32, mul_pd, (a), (b), ##__VA_ARGS__)
#endif

/* ---- single-type integer ops (plain macros, both languages) ------------ */
#define sb_andn(a, b, ...) sb__call_mask_fn(andnot_epi64, (a), (b), ##__VA_ARGS__)
#define sb_and(a, b, ...)  sb__call_mask_fn(and_epi64, (a), (b), ##__VA_ARGS__)
#define sb_or(a, b, ...)   sb__call_mask_fn(or_epi64, (a), (b), ##__VA_ARGS__)
#define sb_xor(a, b, ...)  sb__call_mask_fn(xor_epi64, (a), (b), ##__VA_ARGS__)
#define sb_srlv(a, b, ...) sb__call_mask_fn(srlv_epi64, (a), (b), ##__VA_ARGS__)
#define sb_sllv(a, b, ...) sb__call_mask_fn(sllv_epi64, (a), (b), ##__VA_ARGS__)
#define sb_srli(a, b, ...) sb__call_mask_fn(srli_epi64, (a), (b), ##__VA_ARGS__)
#define sb_slli(a, b, ...) sb__call_mask_fn(slli_epi64, (a), (b), ##__VA_ARGS__)
#define sb_srai(a, b, ...) sb__call_mask_fn(srai_epi64, (a), (b), ##__VA_ARGS__)
#define sb_minu(a, b, ...) sb__call_mask_fn(min_epu64, (a), (b), ##__VA_ARGS__)
#define sb_maxu(a, b, ...) sb__call_mask_fn(max_epu64, (a), (b), ##__VA_ARGS__)

#define sb_maskz(mask, a)      sb__fn(maskz_mov_epi64)((__mmask8)(mask), (a))
#define sb_blend(src, mask, a) sb__fn(mask_mov_epi64)((src), (__mmask8)(mask), (a))
#define sb_blend8(mask, off, on)  sb__fn(mask_blend_epi8)((__mmask64)(mask), (off), (on))
#define sb_blend16(mask, off, on) sb__fn(mask_blend_epi16)((__mmask32)(mask), (off), (on))
#define sb_blend32(mask, off, on) sb__fn(mask_blend_epi32)((__mmask16)(mask), (off), (on))
#define sb_blend64(mask, off, on) sb__fn(mask_blend_epi64)((__mmask8)(mask), (off), (on))

/* sb_ternary(a,b,c,imm8[,mask[,zero]]) - see old types.h ternary() */
#define sb_ternary(...) SB_CAT(sb__ternary_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__ternary_4(a, b, c, imm8) sb__fn(ternarylogic_epi64)((a), (b), (c), (imm8))
#define sb__ternary_5(a, b, c, imm8, mask) \
  sb__fn(mask_ternarylogic_epi64)((a), (__mmask8)(mask), (b), (c), (imm8))
#define sb__ternary_6(a, b, c, imm8, mask, zero) \
  sb__fn(maskz_ternarylogic_epi64)((__mmask8)(mask), (a), (b), (c), (imm8))

#define sb_zero()  sb__fn(setzero_si512)()
#define sb_dzero() sb__fn(setzero_pd)()
#define sb_ones()  sb__fn(set1_epi64)(-1LL)
#define sb_splat_load(p, ind) sb__fn(set1_epi64)(((const uint64_t *)(p))[ind])
#define sb_splat_load_m(src, mask, p, ind)                                     \
  sb__fn(mask_set1_epi64)((src), (__mmask8)(mask),                             \
                          (long long)((const uint64_t *)(p))[ind])
#define sb_set1_64(x) sb__fn(set1_epi64)((long long)(x))
#define sb_set1_32(x) sb__fn(set1_epi32)((int)(x))
#define sb_set1_d(x)  sb__fn(set1_pd)((double)(x))
#define sb_setr_64(e0, e1, e2, e3, e4, e5, e6, e7)                             \
  sb__fn(setr_epi64)((long long)(e0), (long long)(e1), (long long)(e2),        \
                     (long long)(e3), (long long)(e4), (long long)(e5),        \
                     (long long)(e6), (long long)(e7))
#define sb_setr_32(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, \
                   e14, e15)                                                   \
  sb__fn(setr_epi32)((int)(e0), (int)(e1), (int)(e2), (int)(e3), (int)(e4),    \
                     (int)(e5), (int)(e6), (int)(e7), (int)(e8), (int)(e9),    \
                     (int)(e10), (int)(e11), (int)(e12), (int)(e13),           \
                     (int)(e14), (int)(e15))

/* unsigned epi64 comparisons -> mask8 */
#define sb_eq(a, b)  sb__fn(cmpeq_epu64_mask)((a), (b))
#define sb_neq(a, b) sb__fn(cmpneq_epu64_mask)((a), (b))
#define sb_ltu(a, b) sb__fn(cmplt_epu64_mask)((a), (b))
#define sb_leu(a, b) sb__fn(cmple_epu64_mask)((a), (b))
#define sb_gtu(a, b) sb__fn(cmpgt_epu64_mask)((a), (b))
#define sb_geu(a, b) sb__fn(cmpge_epu64_mask)((a), (b))
#define sb_is_zero(a) sb_eq((a), sb_zero())

/* common ternlog names; operand order matters for mux */
#define sb_or3(a, b, c)  sb_ternary((a), (b), (c), 0xfe)
#define sb_xor3(a, b, c) sb_ternary((a), (b), (c), 0x96)
#define sb_maj(a, b, c)  sb_ternary((a), (b), (c), 0xe8)
#define sb_mux(s, a, b)  sb_ternary((s), (a), (b), 0xd8)
#define sb_bitselect(s, a, b) sb_mux((s), (a), (b))

/* ---- k-mask casts and ops: dual C/C++ ---------------------------------- */
#define sb_k8(x)  ((__mmask8)(x))
#define sb_k16(x) ((__mmask16)(x))
#define sb_k32(x) ((__mmask32)(x))
#define sb_k64(x) ((__mmask64)(x))

#ifdef __cplusplus
static inline __mmask8  sb_kadd(__mmask8 a, __mmask8 b){ return _kadd_mask8(a, b); }
static inline __mmask16 sb_kadd(__mmask16 a, __mmask16 b){ return _kadd_mask16(a, b); }
static inline __mmask32 sb_kadd(__mmask32 a, __mmask32 b){ return _kadd_mask32(a, b); }
static inline __mmask64 sb_kadd(__mmask64 a, __mmask64 b){ return _kadd_mask64(a, b); }
static inline __mmask8  sb_kand(__mmask8 a, __mmask8 b){ return _kand_mask8(a, b); }
static inline __mmask16 sb_kand(__mmask16 a, __mmask16 b){ return _kand_mask16(a, b); }
static inline __mmask32 sb_kand(__mmask32 a, __mmask32 b){ return _kand_mask32(a, b); }
static inline __mmask64 sb_kand(__mmask64 a, __mmask64 b){ return _kand_mask64(a, b); }
static inline __mmask8  sb_kandn(__mmask8 a, __mmask8 b){ return _kandn_mask8(a, b); }
static inline __mmask16 sb_kandn(__mmask16 a, __mmask16 b){ return _kandn_mask16(a, b); }
static inline __mmask32 sb_kandn(__mmask32 a, __mmask32 b){ return _kandn_mask32(a, b); }
static inline __mmask64 sb_kandn(__mmask64 a, __mmask64 b){ return _kandn_mask64(a, b); }
static inline __mmask8  sb_kor(__mmask8 a, __mmask8 b){ return _kor_mask8(a, b); }
static inline __mmask16 sb_kor(__mmask16 a, __mmask16 b){ return _kor_mask16(a, b); }
static inline __mmask32 sb_kor(__mmask32 a, __mmask32 b){ return _kor_mask32(a, b); }
static inline __mmask64 sb_kor(__mmask64 a, __mmask64 b){ return _kor_mask64(a, b); }
static inline __mmask8  sb_kxor(__mmask8 a, __mmask8 b){ return _kxor_mask8(a, b); }
static inline __mmask16 sb_kxor(__mmask16 a, __mmask16 b){ return _kxor_mask16(a, b); }
static inline __mmask32 sb_kxor(__mmask32 a, __mmask32 b){ return _kxor_mask32(a, b); }
static inline __mmask64 sb_kxor(__mmask64 a, __mmask64 b){ return _kxor_mask64(a, b); }
static inline __mmask8  sb_kxnor(__mmask8 a, __mmask8 b){ return _kxnor_mask8(a, b); }
static inline __mmask16 sb_kxnor(__mmask16 a, __mmask16 b){ return _kxnor_mask16(a, b); }
static inline __mmask32 sb_kxnor(__mmask32 a, __mmask32 b){ return _kxnor_mask32(a, b); }
static inline __mmask64 sb_kxnor(__mmask64 a, __mmask64 b){ return _kxnor_mask64(a, b); }
static inline __mmask8  sb_knot(__mmask8 a){ return _knot_mask8(a); }
static inline __mmask16 sb_knot(__mmask16 a){ return _knot_mask16(a); }
static inline __mmask32 sb_knot(__mmask32 a){ return _knot_mask32(a); }
static inline __mmask64 sb_knot(__mmask64 a){ return _knot_mask64(a); }
/* value-identical to kshift: masks are integer typedefs, in-width shift */
#define sb_klsh(a, imm) ((decltype(a))((a) << (imm)))
#define sb_krsh(a, imm) ((decltype(a))((a) >> (imm)))
#else
#define sb_kadd(a, b) _Generic((a), __mmask8: _kadd_mask8, __mmask16: _kadd_mask16, \
    __mmask32: _kadd_mask32, __mmask64: _kadd_mask64)((a), (b))
#define sb_kand(a, b) _Generic((a), __mmask8: _kand_mask8, __mmask16: _kand_mask16, \
    __mmask32: _kand_mask32, __mmask64: _kand_mask64)((a), (b))
#define sb_kandn(a, b) _Generic((a), __mmask8: _kandn_mask8, __mmask16: _kandn_mask16, \
    __mmask32: _kandn_mask32, __mmask64: _kandn_mask64)((a), (b))
#define sb_kor(a, b) _Generic((a), __mmask8: _kor_mask8, __mmask16: _kor_mask16, \
    __mmask32: _kor_mask32, __mmask64: _kor_mask64)((a), (b))
#define sb_kxor(a, b) _Generic((a), __mmask8: _kxor_mask8, __mmask16: _kxor_mask16, \
    __mmask32: _kxor_mask32, __mmask64: _kxor_mask64)((a), (b))
#define sb_kxnor(a, b) _Generic((a), __mmask8: _kxnor_mask8, __mmask16: _kxnor_mask16, \
    __mmask32: _kxnor_mask32, __mmask64: _kxnor_mask64)((a), (b))
#define sb_knot(a) _Generic((a), __mmask8: _knot_mask8, __mmask16: _knot_mask16, \
    __mmask32: _knot_mask32, __mmask64: _knot_mask64)(a)
#define sb_klsh(a, imm) _Generic((a),                                          \
    __mmask8: _kshiftli_mask8(sb_k8(a), (imm)),                                \
    __mmask16: _kshiftli_mask16(sb_k16(a), (imm)),                             \
    __mmask32: _kshiftli_mask32(sb_k32(a), (imm)),                             \
    __mmask64: _kshiftli_mask64(sb_k64(a), (imm)))
#define sb_krsh(a, imm) _Generic((a),                                          \
    __mmask8: _kshiftri_mask8(sb_k8(a), (imm)),                                \
    __mmask16: _kshiftri_mask16(sb_k16(a), (imm)),                             \
    __mmask32: _kshiftri_mask32(sb_k32(a), (imm)),                             \
    __mmask64: _kshiftri_mask64(sb_k64(a), (imm)))
#endif

/* ---- lane movement / permutes ------------------------------------------ */
#define sb_alignr64(a, b, imm) sb__fn(alignr_epi64)((a), (b), (imm))
#define sb_alignr8(...) SB_CAT(sb__alignr8_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__alignr8_3(a, b, imm) sb__fn(alignr_epi8)((a), (b), (imm))
#define sb__alignr8_4(a, b, imm, mask) \
  sb__fn(maskz_alignr_epi8)((__mmask64)(mask), (a), (b), (imm))
#define sb__alignr8_5(a, b, imm, mask, src) \
  sb__fn(mask_alignr_epi8)((src), (__mmask64)(mask), (a), (b), (imm))
#define sb_lane_lsh(a, n)  sb_alignr64((a), sb_zero(), 8 - (n))
#define sb_lane_rsh(a, n)  sb_alignr64(sb_zero(), (a), (n))
#define sb_lane_rotl(a, n) sb_alignr64((a), (a), 8 - (n))
#define sb_shuffle8(a, idx, ...) sb__call_mask_fn(shuffle_epi8, (a), (idx), ##__VA_ARGS__)
#define sb_perm64(idx, a)      sb__fn(permutexvar_epi64)((idx), (a))
#define sb_perm64x2(a, idx, b) sb__fn(permutex2var_epi64)((a), (idx), (b))
#define sb_perm64x2z(mask, a, idx, b) \
  sb__fn(maskz_permutex2var_epi64)((__mmask8)(mask), (a), (idx), (b))
#define sb_perm32(idx, a) sb__fn(permutexvar_epi32)((idx), (a))
#define sb_permb(idx, a)  sb__fn(permutexvar_epi8)((idx), (a))
#define sb_unpacklo64(a, b) sb__fn(unpacklo_epi64)((a), (b))
#define sb_unpackhi64(a, b) sb__fn(unpackhi_epi64)((a), (b))
#define sb_shufi64x2(a, b, imm) sb__fn(shuffle_i64x2)((a), (b), (imm))

/* ---- VBMI2 double shifts ------------------------------------------------ */
#define sb_fshl64i(...) SB_CAT(sb__fshl64i_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__fshl64i_3(a, b, imm) sb__fn(shldi_epi64)((a), (b), (imm))
#define sb__fshl64i_4(a, b, imm, mask) \
  sb__fn(maskz_shldi_epi64)((__mmask8)(mask), (a), (b), (imm))
#define sb__fshl64i_5(a, b, imm, mask, src) \
  sb__fn(mask_shldi_epi64)((src), (__mmask8)(mask), (a), (b), (imm))
#define sb_fshr64i(...) SB_CAT(sb__fshr64i_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__fshr64i_3(a, b, imm) sb__fn(shrdi_epi64)((a), (b), (imm))
#define sb__fshr64i_4(a, b, imm, mask) \
  sb__fn(maskz_shrdi_epi64)((__mmask8)(mask), (a), (b), (imm))
#define sb__fshr64i_5(a, b, imm, mask, src) \
  sb__fn(mask_shrdi_epi64)((src), (__mmask8)(mask), (a), (b), (imm))
#define sb_fshl64v(...) SB_CAT(sb__fshl64v_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__fshl64v_3(a, b, counts) sb__fn(shldv_epi64)((a), (b), (counts))
#define sb__fshl64v_4(a, b, counts, mask) \
  sb__fn(maskz_shldv_epi64)((__mmask8)(mask), (a), (b), (counts))
#define sb__fshl64v_5(a, b, counts, mask, src) \
  sb_blend((src), (mask), sb__fshl64v_3((a), (b), (counts)))
#define sb_fshr64v(...) SB_CAT(sb__fshr64v_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__fshr64v_3(a, b, counts) sb__fn(shrdv_epi64)((a), (b), (counts))
#define sb__fshr64v_4(a, b, counts, mask) \
  sb__fn(maskz_shrdv_epi64)((__mmask8)(mask), (a), (b), (counts))
#define sb__fshr64v_5(a, b, counts, mask, src) \
  sb_blend((src), (mask), sb__fshr64v_3((a), (b), (counts)))

/* ---- IFMA --------------------------------------------------------------- */
#define sb_madd52lo(...) SB_CAT(sb__madd52lo_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__madd52lo_3(acc, a, b) sb__fn(madd52lo_epu64)((acc), (a), (b))
#define sb__madd52lo_4(acc, a, b, mask) \
  sb__fn(mask_madd52lo_epu64)((acc), (__mmask8)(mask), (a), (b))
#define sb_madd52hi(...) SB_CAT(sb__madd52hi_, SB_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define sb__madd52hi_3(acc, a, b) sb__fn(madd52hi_epu64)((acc), (a), (b))
#define sb__madd52hi_4(acc, a, b, mask) \
  sb__fn(mask_madd52hi_epu64)((acc), (__mmask8)(mask), (a), (b))

/* ---- scalar carry chains ------------------------------------------------ */
#define sb_adc(carry, a, b, r) _addcarry_u64((carry), (a), (b), (unsigned long long *)(r))
#define sb_sbb(carry, a, b, r) _subborrow_u64((carry), (a), (b), (unsigned long long *)(r))

#define SB_MASK52() sb_set1_64((1ull << 52) - 1)

} // namespace
