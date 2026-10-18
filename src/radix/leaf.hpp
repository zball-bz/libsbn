#pragma once
// Radix conversion leaf layer: words of eight digits <-> digit bytes, and the
// fragment kernels that connect them to binary fractions and integers.
//
// Every non-power-of-two base uses words of K = 8 digits (b^8 <= 63^8 < 2^48,
// inside a u52 lane) and fragments of F = 8 words = 64 digits, the unit of the
// conversion trees (docs/radix-conversion-design-2026-09-18.md). Digit bytes
// are MSD first. The byte written for a digit value is table[value]: the
// identity table gives digit values, any other table is an alphabet.
#include <stddef.h>
#include <stdint.h>
#include <string.h>
namespace sbn::v3 {class Frame;}
namespace sbn::v3::radix {
inline constexpr unsigned word_digits = 8, fragment_words = 8, fragment_digits = 64;
// Largest fragment fraction: 72 digits of base 63 plus guard, in u52 digits.
inline constexpr unsigned max_fragment_u52 = 10, max_fragment_limbs = 8;
struct DigitPlan {
    unsigned base = 0;
    uint64_t b2 = 0, b4 = 0, b8 = 0;
    uint64_t m4 = 0, m2 = 0; // exact division magics: x/b^4 for x < b^8, x/b^2 for x < b^4 (52-bit high product)
    unsigned s4 = 0, s2 = 0;
    uint16_t m1 = 0;         // x/b for x < b^2 (16-bit high product)
    unsigned s1 = 0;
    uint8_t encode[64]{};    // digit value -> output byte
    uint8_t decode[256]{};   // input byte -> digit value, 0xff when not a digit of this base
    bool identity = true;    // bytes are digit values (no alphabet)
    bool decimal_ascii = false; // standard decimal bytes; subtract/add '0'
};
// table == nullptr: digit values (identity). Otherwise the first `base` bytes
// of table are the alphabet; they must be distinct. False when the base is
// outside 3..63 or is a power of two (those never use word kernels), or the
// alphabet repeats a byte.
bool digit_plan_init(DigitPlan &, unsigned base, const uint8_t *table) noexcept;
// Read-only identity-alphabet plans, with no per-operation construction.
const DigitPlan &identity_digit_plan(unsigned base) noexcept;
const DigitPlan &decimal_ascii_plan() noexcept;
inline bool decimal_alphabet(const uint8_t *p) noexcept {
    return memcmp(p,"0123456789",10)==0;
}
// Published read-only numeric constants, counted once per process.
size_t digit_constant_bytes() noexcept;
// Power-of-two codecs. A parse chunk covers 64 digits / `shift` output
// limbs. Ranges partition output chunks, so no worker shares output words.
size_t validate_digits(const uint8_t *,size_t,unsigned base,const uint8_t *decode=nullptr) noexcept;
void format_bits(uint8_t *,size_t,const uint64_t *,size_t,unsigned shift,int64_t first_low_bit,const uint8_t *) noexcept;
void parse_bits_range(uint64_t *,size_t words,size_t begin,size_t end,const uint8_t *,size_t,
                      unsigned shift,int64_t origin,const uint8_t *decode=nullptr) noexcept;
// count words (each < b^8), count a multiple of 8 -> 8*count bytes, MSD first.
void emit_words(uint8_t *out, const uint64_t *words, size_t count, const DigitPlan &) noexcept;
// Up to eight words in the maximal u64 radix b^K. Emit count*K digit bytes;
// each word is below b^K. Internally all eight SIMD lanes remain occupied.
void emit_word_chunks(uint8_t *out,const uint64_t *words,unsigned count,const DigitPlan &) noexcept;
// One u64 -> a zero-padded 64-digit block; returns its first nonzero digit.
unsigned format_limb(uint8_t *out,uint64_t,const DigitPlan &) noexcept;
// First nonzero digit in a fully initialized padded integer area. count is
// a multiple of 64; count itself denotes zero. No readable padding is needed.
size_t first_nonzero_digit(const uint8_t *,size_t count,uint8_t zero) noexcept;
// 8*count digit bytes (count a multiple of 8) -> count words. Returns false if
// any byte is not a digit of the base (the words are then unspecified).
bool parse_words(uint64_t *words, const uint8_t *digits, size_t count, const DigitPlan &) noexcept;
// Up to eight 64-digit integers, with only the leading fragment possibly
// partial. Each output occupies `stride` zero-padded u64 words; stride must
// hold a complete base^64 fragment. Input reads stay within `count` bytes.
bool parse_integer_fragments(uint64_t *out,size_t stride,const uint8_t *digits,size_t count,
                             const DigitPlan &) noexcept;
bool parse_integer_blocks(uint64_t *out,size_t stride,const uint8_t *digits,size_t count,
                          unsigned block_digits,const DigitPlan &) noexcept;
// Eight fragment fractions -> words. fraction[u] points at `limbs` little-endian
// u64 limbs of lane u with the binary point above the top limb (nullptr: a zero
// lane). Each of `rounds` (8 or 9) steps multiplies the fraction by b^8 and
// moves the integer part out: words[u * 9 + r] for r < rounds. Only the top
// 52 * u52_digits bits of each fraction are used (truncation toward zero).
void extract_words(uint64_t *words, const uint64_t *const fraction[8], unsigned limbs, unsigned u52_digits,
                   unsigned rounds, const DigitPlan &) noexcept;
// Up to eight fragments: write 64 digit bytes plus first/overlap words per
// active lane. All input reads finish before output, which is disjoint from
// the two side arrays. Inactive input lanes are nullptr, as for extract_words.
void emit_fragments(uint8_t *out,uint64_t *first,uint64_t *overlap,const uint64_t *const fraction[8],
                    unsigned limbs,unsigned u52_digits,unsigned count,const DigitPlan &) noexcept;
// Scalar references (the authority of the leaf gate).
void emit_words_reference(uint8_t *out, const uint64_t *words, size_t count, const DigitPlan &) noexcept;
bool parse_words_reference(uint64_t *words, const uint8_t *digits, size_t count, const DigitPlan &) noexcept;
inline constexpr unsigned fraction_block_limbs=32;
inline constexpr unsigned fraction_block_tree_limbs=64;
struct FractionBlock {
    const uint64_t *value;
    unsigned limbs,fragments;
    uint8_t *output;
    uint64_t *first,*overlap; // one eight-digit head and tail for the whole block
};
// Eight homogeneous subtrees in transposed u52 arithmetic. Internal splits
// truncate downward; the complete digit string plus overlap is exact or one
// below, like the surrounding SRT. Internal seams are reconciled here.
struct FractionBlockPlan {
    uint64_t power[8][64]{};
    unsigned begin[8]{},end[8]{},precision[17]{};
};
const FractionBlockPlan &fraction_block_plan(unsigned base) noexcept;
size_t fraction_block_tree_scratch_bytes(unsigned max_limbs,unsigned fragments) noexcept;
void emit_fraction_block_tree(const FractionBlock *,unsigned count,const DigitPlan &,
                              const FractionBlockPlan &,::sbn::v3::Frame &) noexcept;
// Exact one-fraction basecase with redundant radix-2^51 state and one final
// carry pass. Work is caller-owned; digits and the optional guard are disjoint.
size_t serial_fraction_words(size_t limbs,uint64_t digits,unsigned base) noexcept;
void serial_fraction_emit(uint8_t *out,uint8_t *guard,const uint64_t *input,size_t limbs,uint64_t digits,
                          unsigned guards,const DigitPlan &,uint64_t *work) noexcept;
} // namespace sbn::v3::radix
