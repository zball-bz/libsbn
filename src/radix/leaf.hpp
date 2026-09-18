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
};
// table == nullptr: digit values (identity). Otherwise the first `base` bytes
// of table are the alphabet; they must be distinct. False when the base is
// outside 3..63 or is a power of two (those never use word kernels), or the
// alphabet repeats a byte.
bool digit_plan_init(DigitPlan &, unsigned base, const uint8_t *table) noexcept;
// count words (each < b^8), count a multiple of 8 -> 8*count bytes, MSD first.
void emit_words(uint8_t *out, const uint64_t *words, size_t count, const DigitPlan &) noexcept;
// 8*count digit bytes (count a multiple of 8) -> count words. Returns false if
// any byte is not a digit of the base (the words are then unspecified).
bool parse_words(uint64_t *words, const uint8_t *digits, size_t count, const DigitPlan &) noexcept;
// Eight fragment fractions -> words. fraction[u] points at `limbs` little-endian
// u64 limbs of lane u with the binary point above the top limb (nullptr: a zero
// lane). Each of `rounds` (8 or 9) steps multiplies the fraction by b^8 and
// moves the integer part out: words[u * 9 + r] for r < rounds. Only the top
// 52 * u52_digits bits of each fraction are used (truncation toward zero).
void extract_words(uint64_t *words, const uint64_t *const fraction[8], unsigned limbs, unsigned u52_digits,
                   unsigned rounds, const DigitPlan &) noexcept;
// Scalar references (the authority of the leaf gate).
void emit_words_reference(uint8_t *out, const uint64_t *words, size_t count, const DigitPlan &) noexcept;
bool parse_words_reference(uint64_t *words, const uint8_t *digits, size_t count, const DigitPlan &) noexcept;
} // namespace sbn::v3::radix
