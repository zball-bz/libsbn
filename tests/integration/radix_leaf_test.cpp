// Radix leaf gate: the vector word kernels against their scalar references for every base that uses
// them, alphabets and invalid bytes, and the fragment extraction against exact integer arithmetic.
#include "radix/leaf.hpp"
#include "radix/word_base.hpp"
#include "../oracle/oracle.h"
#include <assert.h>
#include <initializer_list>
#include <stdio.h>
#include <string.h>
using namespace sbn::v3::radix;
namespace {
uint64_t state = 0x243f6a8885a308d3ULL;
uint64_t rnd() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
const char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+";
}
int main() {
    unsigned bases = 0;
    size_t words_checked = 0, fragments_checked = 0;
    for (unsigned base = 2; base <= 64; ++base) {
        const auto wb=word_bases[base];uint64_t b8=1,power=1;
        for(unsigned j=0;j<8;++j)b8*=base;
        for(unsigned j=0;j<wb.digits;++j)power*=base;
        assert(power==wb.power && (__uint128_t)power*base>UINT64_MAX);
        for(unsigned j=0;j<256;++j){
            const uint64_t x=j==0?0:j==1?1:j==2?b8-1:j==3?b8:j==4?b8+1:
                             j==5?wb.power-1:j==6?UINT64_MAX:j==7?UINT64_MAX-1:rnd();
            uint64_t q=x;const auto r=split_radix_word(q,b8,wb.reciprocal8);
            assert(q==x/b8 && r==x%b8);
        }
        DigitPlan values{}, text{};
        const bool expected = base >= 3 && base <= 63 && (base & (base - 1));
        assert(digit_plan_init(values, base, nullptr) == expected);
        if (!expected)
            continue;
        assert(digit_plan_init(text, base, reinterpret_cast<const uint8_t *>(alphabet)));
        ++bases;
        for(unsigned count=1;count<=8;++count)for(unsigned trial=0;trial<8;++trial){
            uint64_t wide[8];uint8_t actual[512],expected_digits[512];
            for(unsigned j=0;j<count;++j)wide[j]=trial==0?0:trial==1?wb.power-1:rnd()%wb.power;
            for(const auto *p:{&values,&text}){
                memset(actual,0xcc,sizeof(actual));emit_word_chunks(actual,wide,count,*p);
                for(unsigned j=0;j<count;++j){uint64_t q=wide[j];
                    for(unsigned k=wb.digits;k-->0;){expected_digits[j*wb.digits+k]=p->encode[q%base];q/=base;}
                }
                assert(!memcmp(actual,expected_digits,count*wb.digits));
                for(size_t j=count*wb.digits;j<sizeof(actual);++j)assert(actual[j]==0xcc);
            }
        }
        uint64_t words[64], back[64];
        uint8_t fast[512], slow[512];
        for (unsigned trial = 0; trial < 40; ++trial) {
            for (unsigned i = 0; i < 64; ++i) {
                const unsigned kind = (trial + i) % 7;
                words[i] = kind == 0   ? 0
                           : kind == 1 ? values.b8 - 1
                           : kind == 2 ? values.b4
                           : kind == 3 ? values.b4 - 1
                                       : rnd() % values.b8;
            }
            for (const DigitPlan *p : {&values, &text}) {
                emit_words(fast, words, 64, *p);
                emit_words_reference(slow, words, 64, *p);
                assert(!memcmp(fast, slow, 512));
                assert(parse_words(back, fast, 64, *p) && !memcmp(back, words, sizeof words));
                assert(parse_words_reference(back, fast, 64, *p) && !memcmp(back, words, sizeof words));
                words_checked += 64;
            }
        }
        // Every invalid byte position is reported, the smallest invalid value included.
        emit_words(fast, words, 8, values);
        for (unsigned position : {0u, 7u, 31u, 63u}) {
            uint8_t broken[64];
            memcpy(broken, fast, 64);
            broken[position] = uint8_t(base);
            assert(!parse_words(back, broken, 8, values) && !parse_words_reference(back, broken, 8, values));
            broken[position] = 0xff;
            assert(!parse_words(back, broken, 8, values));
        }
        emit_words(fast, words, 8, text);
        fast[5] = '/'; // not in the alphabet
        assert(!parse_words(back, fast, 8, text));
        // A repeated alphabet byte is rejected.
        uint8_t repeated[64];
        memcpy(repeated, alphabet, 64);
        repeated[base - 1] = repeated[0];
        DigitPlan rejected{};
        assert(!digit_plan_init(rejected, base, repeated));
        // Fragment extraction: every round is floor(frac * b^8) of the fraction truncated to the u52 digits used.
        ref_int fraction, scaled, word, modulus;
        ref_inits(fraction, scaled, word, modulus, nullptr);
        for (unsigned limbs : {1u, 3u, 4u, 5u, 8u})
            for (unsigned trial = 0; trial < 6; ++trial) {
                const unsigned most = (64 * limbs + 51) / 52;
                const unsigned digits52 = trial % 2 ? most : (most > 1 ? most - 1 : 1);
                if (digits52 > max_fragment_u52)
                    continue;
                uint64_t lanes[8][max_fragment_limbs]{};
                const uint64_t *pointers[8]{};
                for (unsigned u = 0; u < 8; ++u) {
                    for (unsigned j = 0; j < limbs; ++j)
                        lanes[u][j] = trial == 0 ? ~uint64_t(0) : trial == 1 && u % 2 ? 0 : rnd();
                    pointers[u] = u == 5 && trial > 1 ? nullptr : lanes[u];
                }
                const unsigned rounds = 8 + trial % 2;
                uint64_t got[8 * 9]{};
                extract_words(got, pointers, limbs, digits52, rounds, values);
                for (unsigned u = 0; u < 8; ++u) {
                    // truncate to the top 52*digits52 bits: y' = floor(y / 2^drop) with n' = 64 limbs - drop bits
                    const int64_t keep = int64_t(52) * digits52, total = int64_t(64) * limbs;
                    ref_set_ui(fraction, 0);
                    if (pointers[u])
                        ref_import(fraction, limbs, -1, 8, 0, 0, lanes[u]);
                    size_t bits = size_t(total);
                    if (keep < total) {
                        ref_fdiv_q_2exp(fraction, fraction, size_t(total - keep));
                        bits = size_t(keep);
                    }
                    for (unsigned r = 0; r < rounds; ++r) {
                        ref_mul_ui(scaled, fraction, values.b8);
                        ref_fdiv_q_2exp(word, scaled, bits);
                        ref_fdiv_r_2exp(fraction, scaled, bits);
                        assert(ref_get_ui(word) == got[u * 9 + r] && got[u * 9 + r] < values.b8);
                    }
                    ++fragments_checked;
                }
            }
        ref_clears(fraction, scaled, word, modulus, nullptr);
    }
    assert(bases == 57); // 3..63 without 4, 8, 16, 32
    printf("radix leaf: %u bases, %zu words emit/parse against the scalar reference (values and alphabet), "
           "invalid bytes, %zu fragment extractions against exact arithmetic PASS\n",
           bases, words_checked, fragments_checked);
}
