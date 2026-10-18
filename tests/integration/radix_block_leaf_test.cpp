#include "radix/leaf.hpp"
#include "runtime/scratch.hpp"
#include <vector>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using namespace sbn::v3;
using namespace sbn::v3::radix;
int main(){
    uint64_t seed=0x519c326abbULL;auto random=[&]{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;};
    unsigned checked=0;
    for(unsigned base=3;base<=63;++base)if(base&(base-1))for(unsigned count=1;count<=8;++count)for(unsigned pattern=0;pattern<5;++pattern){
        uint8_t alphabet[64];for(unsigned j=0;j<64;++j)alphabet[j]=uint8_t(200-j);
        DigitPlan p;assert(digit_plan_init(p,base,pattern==2?alphabet:nullptr));
        std::array<std::vector<uint64_t>,8> input,first,overlap;
        std::array<std::vector<uint8_t>,8> output,expected;
        FractionBlock blocks[8];unsigned max_fragments=0;
        for(unsigned u=0;u<count;++u){const unsigned lane=0;
            const unsigned n=1+(3*base+5*lane+pattern)%fraction_block_tree_limbs,f=1+(base+lane+pattern)%16;
            input[u].resize(n);for(auto &v:input[u])v=pattern==1||pattern==4?UINT64_MAX:random();
            output[u].resize(64*f+64,0xee);first[u].resize(1);overlap[u].resize(1);expected[u].resize(64*f+8);
            auto value=input[u];
            for(auto &d:expected[u]){__uint128_t carry=0;for(auto &v:value){carry+=(__uint128_t)v*base;v=uint64_t(carry);carry>>=64;}d=uint8_t(carry);}
            blocks[u]={input[u].data(),n,f,output[u].data(),first[u].data(),overlap[u].data()};max_fragments=std::max(max_fragments,f);
        }
        const auto &plan=fraction_block_plan(base);
        const auto exact_first=expected[0];
        const size_t bytes=fraction_block_tree_scratch_bytes(blocks[0].limbs,max_fragments);
        std::vector<uint8_t> raw(bytes+64);void *at=reinterpret_cast<void *>((uintptr_t(raw.data())+63)&~uintptr_t(63));
        auto scratch=Frame::external(at,bytes);
        allocation_watch_start();
        emit_fraction_block_tree(blocks,count,p,plan,scratch);
        assert(!allocation_watch_stop());
        for(unsigned u=0;u<count;++u){const unsigned f=blocks[u].fragments;
            {auto got=expected[u];
                for(unsigned j=0;j<64*f;++j)got[j]=p.decode[output[u][j]];
                uint64_t tail=overlap[u][0];for(unsigned k=8;k-->0;){got[64*f+k]=tail%base;tail/=base;}
                if(got!=expected[u]){for(size_t j=expected[u].size();j-->0;){if(expected[u][j]){--expected[u][j];break;}expected[u][j]=base-1;}
                    assert(got==expected[u]);}
            }
            for(unsigned j=0;j<64*f;++j)assert(output[u][j]==p.encode[expected[u][j]]);
            for(unsigned j=64*f;j<output[u].size();++j)assert(output[u][j]==0xee);
            uint64_t head=0,tail=0;for(unsigned k=0;k<8;++k){head=head*base+expected[u][k];tail=tail*base+expected[u][64*f+k];}
            assert(first[u][0]==head&&overlap[u][0]==tail);
            ++checked;
        }
        const unsigned serial_digits=64*blocks[0].fragments;
        std::vector<uint64_t> serial_work(serial_fraction_words(blocks[0].limbs,serial_digits+8,base));
        std::vector<uint8_t> serial_out(serial_digits+64,0xee);uint8_t guard[8];
        allocation_watch_start();
        serial_fraction_emit(serial_out.data(),guard,blocks[0].value,blocks[0].limbs,serial_digits,8,p,serial_work.data());
        assert(!allocation_watch_stop());
        for(unsigned j=0;j<serial_digits;++j)assert(serial_out[j]==p.encode[exact_first[j]]);
        for(unsigned j=0;j<8;++j)assert(guard[j]==p.encode[exact_first[serial_digits+j]]);
        for(unsigned j=serial_digits;j<serial_out.size();++j)assert(serial_out[j]==0xee);
    }
    printf("batched SRT leaves: %u precision/digit-count/lane/alphabet cases, exact-or-one-low scalar oracle PASS\n",checked);
}
