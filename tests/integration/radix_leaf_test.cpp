// Radix leaf gate: the vector word kernels against their scalar references for every base that uses
// them, alphabets and invalid bytes, and the fragment extraction against exact integer arithmetic.
#include "radix/leaf.hpp"
#include "radix/word_base.hpp"
#include "radix/digit_bounds.hpp"
#include "radix/geometry.hpp"
#include "radix/power_bank.hpp"
#include "../oracle/oracle.h"
#include <assert.h>
#include <initializer_list>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
using namespace sbn::v3::radix;
namespace {
uint64_t state = 0x243f6a8885a308d3ULL;
uint64_t rnd() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
const char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+";
void decimal_boundaries() {
    DigitPlan numeric,text;
    assert(digit_plan_init(numeric,10,nullptr));
    assert(digit_plan_init(text,10,reinterpret_cast<const uint8_t*>(alphabet)));
    const size_t page=size_t(sysconf(_SC_PAGESIZE));
    auto *memory=static_cast<uint8_t*>(mmap(nullptr,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    assert(memory!=MAP_FAILED&&mprotect(memory+page,page,PROT_NONE)==0);
    for(const auto *p:{&numeric,&text})for(unsigned n=1;n<=128;++n){
        auto *input=memory+page-n;uint64_t expected[16]{},out[18];
        for(unsigned j=0;j<n;++j){const uint64_t digit=rnd()%10;input[j]=p->encode[digit];uint64_t carry=digit;
            for(auto &v:expected){const __uint128_t x=__uint128_t(v)*10+carry;v=uint64_t(x);carry=uint64_t(x>>64);}
            assert(!carry);}
        memset(out,0xa5,sizeof out);
        assert(parse_integer_blocks(out+1,16,input,n,128,*p));
        assert(!memcmp(out+1,expected,sizeof expected));
        assert(out[0]==0xa5a5a5a5a5a5a5a5ULL&&out[17]==out[0]);
        if(n<=38)for(unsigned j=0;j<n;++j){const auto saved=input[j];
            for(unsigned byte=0;byte<256;++byte){input[j]=uint8_t(byte);
                const bool valid=p->decode[byte]!=0xff;
                assert(parse_integer_blocks(out+1,16,input,n,128,*p)==valid);}
            input[j]=saved;}
    }
    assert(munmap(memory,2*page)==0);
    // Every 4-digit quotient boundary and both encodings exercise the
    // range-specific reciprocal constants in the shared emitter.
    for(unsigned q=0;q<10000;++q)for(const auto *p:{&numeric,&text}){
        uint64_t words[8];uint8_t got[64],want[64];
        for(unsigned j=0;j<8;++j)words[j]=q*10000+((j&1)?9999:j);
        emit_words(got,words,8,*p);emit_words_reference(want,words,8,*p);
        assert(!memcmp(got,want,64));
    }
}
}
int main() {
    decimal_boundaries();
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
        DigitPlan values{}, text{},scrambled{};
        const bool expected = base >= 3 && base <= 63 && (base & (base - 1));
        assert(digit_plan_init(values, base, nullptr) == expected);
        if (!expected)
            continue;
        for(unsigned bits=1;bits<=120;++bits){
            const __uint128_t target=__uint128_t(1)<<bits;__uint128_t power=1;unsigned digits=0;
            while(power<target){power*=base;++digits;}
            const unsigned lower=digits-(power!=target);
            assert(integer_digit_capacity(bits,base)>=digits&&integer_digit_capacity(bits,base)<=digits+1);
            assert(fraction_digit_capacity(bits,base)<=lower&&fraction_digit_capacity(bits,base)+1>=lower);
        }
        assert(digit_plan_init(text, base, reinterpret_cast<const uint8_t *>(alphabet)));
        uint8_t table[64];for(unsigned j=0;j<64;++j)table[j]=uint8_t(73*j+151);
        assert(digit_plan_init(scrambled,base,table));
        ++bases;
        for(unsigned trial=0;trial<64;++trial)for(const auto *p:{&values,&text,&scrambled}){
            const uint64_t value=trial==0?0:trial==1?UINT64_MAX:trial==2?values.b8:trial==3?values.b8-1:rnd();
            uint8_t actual[64],expected_digits[64];uint64_t q=value;
            for(unsigned j=64;j-->0;){expected_digits[j]=p->encode[q%base];q/=base;}
            const auto first=format_limb(actual,value,*p);unsigned lead=0;
            while(lead<64&&expected_digits[lead]==p->encode[0])++lead;
            assert(first==lead&&!memcmp(actual,expected_digits,64));
        }
        for(unsigned count:{1u,7u,8u,9u,15u,16u,17u,19u,31u,32u,33u,63u,64u,65u,77u,79u,80u,81u,127u,128u,129u,255u,256u,257u,511u,512u})
          for(unsigned block:{64u,128u})for(unsigned trial=0;trial<3;++trial)for(const auto *p:{&values,&text,&scrambled}){
            uint8_t input[512];uint64_t actual[66],expected[64]{};
            for(unsigned j=0;j<count;++j)input[j]=p->encode[trial==0?0:trial==1?base-1:rnd()%base];
            const unsigned stride=block/8;unsigned length=(count-1)%block+1,at=0,fragment=0;
            while(at<count){
                for(unsigned j=0;j<length;++j){uint64_t carry=p->decode[input[at+j]];
                    for(unsigned k=0;k<stride;++k){const __uint128_t t=__uint128_t(expected[fragment*stride+k])*base+carry;
                        expected[fragment*stride+k]=uint64_t(t);carry=uint64_t(t>>64);}
                    assert(!carry);}
                at+=length;length=block;++fragment;
            }
            memset(actual,0xa5,sizeof(actual));
            assert(parse_integer_blocks(actual+1,stride,input,count,block,*p));
            assert(!memcmp(actual+1,expected,fragment*stride*sizeof(uint64_t)));
            assert(actual[0]==0xa5a5a5a5a5a5a5a5ULL&&actual[fragment*stride+1]==0xa5a5a5a5a5a5a5a5ULL);
            input[count-1]=p->identity?uint8_t(base):uint8_t(0);
            if(p->decode[input[count-1]]==0xff||(input[count-1]>=base&&p->identity))
                assert(!parse_integer_blocks(actual+1,stride,input,count,block,*p));
        }
        for(unsigned count=1;count<=8;++count)for(unsigned trial=0;trial<8;++trial){
            uint64_t wide[8];uint8_t actual[512],expected_digits[512];
            for(unsigned j=0;j<count;++j)wide[j]=trial==0?0:trial==1?wb.power-1:rnd()%wb.power;
            for(const auto *p:{&values,&text,&scrambled}){
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
          for(unsigned digits52=1;digits52<=max_fragment_u52 && 52*digits52<64*limbs+52;++digits52)
            for (unsigned trial = 0; trial < 6; ++trial) {
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
                if(rounds==9)for(unsigned count=1;count<=8;++count)for(const DigitPlan *p:{&values,&text}){
                    uint8_t emitted[512],reference[512];uint64_t first[8],overlap[8];
                    memset(emitted,0xa5,sizeof emitted);memset(first,0xa5,sizeof first);memset(overlap,0xa5,sizeof overlap);
                    emit_fragments(emitted,first,overlap,pointers,limbs,digits52,count,*p);
                    for(unsigned u=0;u<count;++u){
                        emit_words_reference(reference+64*u,got+9*u,8,*p);
                        assert(first[u]==got[9*u] && overlap[u]==got[9*u+8]);
                    }
                    assert(!memcmp(emitted,reference,64*count));
                    for(unsigned j=64*count;j<512;++j)assert(emitted[j]==0xa5);
                    for(unsigned u=count;u<8;++u)assert(first[u]==0xa5a5a5a5a5a5a5a5ULL && overlap[u]==0xa5a5a5a5a5a5a5a5ULL);
                }
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
    // Every published reciprocal, including prefixes shared by different
    // powers-of-two factors of a radix, is the exact all-ones quotient.
    ref_int power,denominator,normalized,inverse,product,limit,next;
    ref_inits(power,denominator,normalized,inverse,product,limit,next,nullptr);
    unsigned inverse_cases=0;
    for(unsigned base=3;base<=63;++base)if(base&(base-1)){
        BaseInfo geometry;assert(base_info(base,geometry));ref_set_ui(power,1);
        for(unsigned j=0;j<64;++j)ref_mul_ui(power,power,base);
        ref_set_ui(denominator,1);
        for(unsigned f=1;f<=fixed_integer_inverse_fragments;++f){
            const size_t n=node_limbs(geometry,f);const auto inv=fixed_integer_inverse(base,f,n);
            if(n>fixed_integer_inverse_words){assert(!inv.data);break;}
            assert(inv.data);ref_mul(denominator,denominator,power);
            ref_mul_2exp(normalized,denominator,inv.shift+2);
            ref_import(inverse,n+1,-1,8,0,0,inv.data);ref_mul(product,normalized,inverse);
            ref_set_ui(limit,1);ref_mul_2exp(limit,limit,128*n);ref_sub_ui(limit,limit,1);
            ref_add(next,product,normalized);
            assert(ref_cmp(product,limit)<=0&&ref_cmp(next,limit)>0);++inverse_cases;
        }
    }
    ref_clears(power,denominator,normalized,inverse,product,limit,next,nullptr);
    printf("radix published integer reciprocals: %u exact quotient bounds PASS\n",inverse_cases);
    assert(bases == 57); // 3..63 without 4, 8, 16, 32
    printf("radix leaf: %u bases, %zu words emit/parse against the scalar reference (values and alphabet), "
           "invalid bytes, %zu fragment extractions against exact arithmetic PASS\n",
           bases, words_checked, fragments_checked);
}
