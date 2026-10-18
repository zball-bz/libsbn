// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/source_window.hpp"
#include "common/checked.hpp"
#include <algorithm>
#include <cstring>
namespace sbn::v3::external {
void source_word_range(const SourceWindow&s,size_t offset,size_t count,size_t&begin,size_t&end) noexcept {
    const __int128 start=__int128(s.first_limb)+offset-s.low_zero_words;
    const auto clip=[&](__int128 value){return size_t(std::clamp(value,__int128(s.lower_limb),__int128(s.upper_limb)));};
    begin=clip(start-(s.bit_shift>0));end=clip(start+count+(s.bit_shift<0));
}
IoError read_source_words(ValueReader&reader,const SourceWindow&input,size_t offset,size_t count,std::span<uint8_t>out) noexcept {
    auto s=input;
    if(input.storage_limbs==SIZE_MAX||input.storage_limbs>SIZE_MAX/8||input.limbs==SIZE_MAX||
       !resolve_source_window(input.limbs,s)||offset>s.limbs||count>s.limbs-offset||count>SIZE_MAX/8||
       !out.data()||out.size()<count*8)return {IoCode::invalid,"source word view"};
    if(!count)return {};
    auto complement=[&](std::span<uint8_t>bytes){if(s.ones_complement)for(size_t i=0;i<bytes.size();i+=8){
        uint64_t word;std::memcpy(&word,bytes.data()+i,8);word=~word;std::memcpy(bytes.data()+i,&word,8);}};
    const __int128 start=__int128(s.first_limb)+offset-s.low_zero_words;
    if(!s.bit_shift&&!s.low_zero_words){
        if(auto e=reader.read(size_t(start)*8,out.first(count*8)))return e;complement(out.first(count*8));return {};
    }
    if(count>SIZE_MAX/8-2||out.size()<(count+2)*8)return {IoCode::capacity,"shifted source guard",0,(count+2)*8,out.size()};
    std::memset(out.data(),0,(count+2)*8);
    size_t begin,end;source_word_range(s,offset,count,begin,end);
    if(begin<end){
        const size_t at=size_t(__int128(begin)-start+1);
        if(auto e=reader.read(begin*8,out.subspan(at*8,(end-begin)*8)))return e;
        complement(out.subspan(at*8,(end-begin)*8));
    }
    const unsigned shift=unsigned(s.bit_shift<0?-s.bit_shift:s.bit_shift);
    // One leading and one trailing guard make a forward in-place pass safe.
    for(size_t j=0;j<count;++j){uint64_t current,adjacent,value;
        std::memcpy(&current,out.data()+8*(j+1),8);
        std::memcpy(&adjacent,out.data()+8*(s.bit_shift>0?j:j+2),8);
        value=!shift?current:s.bit_shift>0?(current<<shift)|(adjacent>>(64-shift)):(current>>shift)|(adjacent<<(64-shift));
        std::memcpy(out.data()+8*j,&value,8);
    }
    std::memset(out.data()+count*8,0,16);
    return {};
}
} // namespace sbn::v3::external
