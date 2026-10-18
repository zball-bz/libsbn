// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/window.hpp"
#include "external/trace.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include <algorithm>
#include <cstring>
namespace sbn::v3::external {
bool query_signed_lift(size_t words,product::Window window,product::MagnitudeBound bound,SignedLiftPlan&out,
                       size_t subtract_words,size_t subtract_shift) noexcept {
    if(!words||window.error_bits||window.origin>words||window.words>words-window.origin||
       !bound.high_exclusive||bound.high_exclusive>=(uint64_t(1)<<63)||bound.words>=words||
       subtract_shift>size_t(INT64_MAX/64))return false;
    SignedLiftPlan p{};p.modulus_words=words;p.window=window;p.bound=bound;
    p.subtract_words=subtract_words;p.subtract_shift=subtract_shift;
    size_t bytes;if(!mul_size(words,8,bytes)||!align_size(bytes,chunk_bytes,bytes)||!mul_size(window.words,8,p.output_bytes))return false;
    p.transfer_bytes=std::min(size_t(1)<<20,std::max(chunk_bytes,bytes));
    RegionLayout l{};if(region_layout(p.output_bytes,0,l))return false;p.output_allocation=l.allocation_bytes;p.write_bytes=l.data_bytes+l.index_bytes;
    size_t at=0;auto put=[&](size_t n,size_t&offset){return align_size(at,4096,offset)&&add_size(offset,n,at);};
    for(unsigned j=0;j<2+unsigned(subtract_words!=0);++j)if(!put(p.transfer_bytes,p.io[j])||!put(index_page_bytes,p.index[j]))return false;
    if(!put(p.transfer_bytes,p.data)||(subtract_words&&!put(p.transfer_bytes+16,p.subtract))||
       !align_size(at,4096,p.storage_bytes))return false;
    out=p;return true;
}
IoError run_signed_lift(const SignedLiftPlan&p,Transport&t,ValueStore&store,const StoredInteger&input,
                       StoredInteger&out,std::span<uint8_t>w,const LinearInput*sub) noexcept {
    OperationSpan trace(t,"signed_lift",p.modulus_words,p.window.words);
    if(!input.value||input.value.store()!=&store||!input.extent_known||input.first_limb||input.limbs()!=p.modulus_words||
       input.negative||input.exponent2||out.value||!w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes)
        return {IoCode::invalid,"signed lift binding"};
    if(bool(p.subtract_words)!=bool(sub))return {IoCode::invalid,"signed lift subtraction presence"};
    if(sub){auto source=sub->source;
        if(sub->negative||sub->limbs()!=p.subtract_words||sub->exponent2!=int64_t(64*p.subtract_shift)||
           (sub->value&&(!*sub->value||sub->value->store()!=&store||
            source.storage_limbs!=sub->value->region().layout.logical_bytes/8||!resolve_source_window(sub->limbs(),source))))
            return {IoCode::invalid,"signed lift subtraction view"};
    }
    if(t.counters.write_submitted>t.write_limit||p.write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"signed lift write allowance",0,p.write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    RegionLease result;if(auto e=store.reserve(p.output_bytes,RegionKind::value,result))return e;
    StoreBuffers read_io{w.subspan(p.io[0],p.transfer_bytes),w.subspan(p.index[0],index_page_bytes)};
    StoreBuffers write_io{w.subspan(p.io[1],p.transfer_bytes),w.subspan(p.index[1],index_page_bytes)};
    ValueReader reader,subtract_reader;ValueWriter writer;WordExtent extent;
    if(auto e=reader.open(t,input.value.region(),read_io))return e;
    if(sub&&sub->value){StoreBuffers io{w.subspan(p.io[2],p.transfer_bytes),w.subspan(p.index[2],index_page_bytes)};
        if(auto e=subtract_reader.open(t,sub->value->region(),io))return e;
    }
    auto data=w.subspan(p.data,p.transfer_bytes);
    auto subtract=sub?w.subspan(p.subtract,p.transfer_bytes+16):std::span<uint8_t>{};
    auto read_subtract=[&](size_t first,size_t count)->IoError {
        std::memset(subtract.data(),0,count*8+16);
        const size_t from=first>p.subtract_shift?first-p.subtract_shift:0;
        const size_t end=first+count>p.subtract_shift?std::min(p.subtract_words,first+count-p.subtract_shift):0;
        if(from>=end)return {};
        const size_t offset=from+p.subtract_shift-first;
        if(sub->value)return read_source_words(subtract_reader,sub->source,from,end-from,subtract.subspan(offset*8));
        require(!from&&end==1,SBN3_FATAL_MATH,"signed lift scalar range");
        std::memcpy(subtract.data()+offset*8,&sub->scalar,8);return {};
    };
    uint64_t high=input.extent.end==p.modulus_words?input.extent.top_word:0;
    if(sub){
        if(auto e=read_subtract(p.modulus_words-1,1))return e;
        uint64_t word;std::memcpy(&word,subtract.data(),8);high-=word;
        // A lower borrow changes the high sign bit only at these two values.
        // Compare the lower spans from high to low; cancellation normally
        // resolves inside the first integrity chunk. Equality scans fully.
        if(high==0||high==(uint64_t(1)<<63)){
            for(size_t end=p.modulus_words-1;end;){const size_t count=std::min(p.transfer_bytes/8,end),first=end-count;
                if(auto e=reader.read(first*8,data.first(count*8)))return e;
                if(auto e=read_subtract(first,count))return e;
                int comparison=0;
                for(size_t j=count;j-- >0;){uint64_t x,y;std::memcpy(&x,data.data()+8*j,8);std::memcpy(&y,subtract.data()+8*j,8);
                    if(x!=y){comparison=x<y?-1:1;break;}}
                if(comparison){high-=comparison<0;break;}end=first;
            }
        }
    }
    const bool negative=high>>63;
    if(auto e=writer.open(t,result.region(),write_io,&extent))return e;
    uint64_t carry=negative,borrow=0,raw=0,sub_residue=0,lifted=0;
    for(size_t first=0;first<p.modulus_words;){const size_t count=std::min(p.transfer_bytes/8,p.modulus_words-first);
        if(auto e=reader.read(first*8,data.first(count*8)))return e;
        m61::add_at(raw,data.data(),count,first);
        if(sub){if(auto e=read_subtract(first,count))return e;m61::add_at(sub_residue,subtract.data(),count,first);}
        for(size_t j=0;j<count;++j){uint64_t v;std::memcpy(&v,data.data()+8*j,8);
            if(sub){uint64_t word;std::memcpy(&word,subtract.data()+8*j,8);const __uint128_t x=__uint128_t(word)+borrow;
                borrow=__uint128_t(v)<x;v-=uint64_t(x);}
            if(negative){const __uint128_t x=__uint128_t(~v)+carry;v=uint64_t(x);carry=uint64_t(x>>64);}
            std::memcpy(data.data()+8*j,&v,8);
            const size_t at=first+j;
            require(at<p.bound.words||(at==p.bound.words?v<p.bound.high_exclusive:!v),SBN3_FATAL_MATH,"signed lift magnitude");
        }
        m61::add_at(lifted,data.data(),count,first);
        const size_t begin=std::max(first,p.window.origin),end=std::min(first+count,p.window.origin+p.window.words);
        if(begin<end)if(auto e=writer.append(data.subspan((begin-first)*8,(end-begin)*8)))return e;
        first+=count;
    }
    const uint64_t power=uint64_t(1)<<((3*(p.modulus_words%61))%61);
    raw=m61::reduce(__uint128_t(raw)+m61::modulus-sub_residue+__uint128_t(borrow)*power);
    const uint64_t expected=negative?(power>=raw?power-raw:power+m61::modulus-raw):raw;
    require(!carry&&lifted==expected,SBN3_FATAL_MATH,"signed lift carry/m61");
    if(auto e=writer.seal())return e;if(auto e=result.publish(out.value))return e;
    out.negative=negative;out.extent=extent;out.extent_known=true;return {};
}
IoError write_word_pattern(Transport&t,ValueStore&store,size_t words,uint64_t low,uint64_t high,
                          StoredInteger&out,std::span<uint8_t>w) noexcept {
    constexpr size_t bytes=4*chunk_bytes;
    if(!words||words>SIZE_MAX/8||out.value||!w.data()||(uintptr_t(w.data())&4095)||w.size()<2*bytes+index_page_bytes)
        return {IoCode::invalid,"constant word pattern"};
    RegionLease result;if(auto e=store.reserve(words*8,RegionKind::value,result))return e;
    StoreBuffers io{w.first(bytes),w.subspan(bytes,index_page_bytes)};auto data=w.subspan(bytes+index_page_bytes,bytes);
    ValueWriter writer;WordExtent extent;if(auto e=writer.open(t,result.region(),io,&extent))return e;
    auto*buffer=reinterpret_cast<uint64_t *>(data.data());std::fill_n(buffer,bytes/8,low);
    for(size_t first=0;first<words;){const size_t count=std::min(bytes/8,words-first);
        if(first+count==words)buffer[count-1]=high;
        if(auto e=writer.append(data.first(count*8)))return e;first+=count;
    }
    if(auto e=writer.seal())return e;if(auto e=result.publish(out.value))return e;
    out.extent=extent;out.extent_known=true;return {};
}
bool query_fixed_output(const StoredInteger&input,size_t fractional,uint64_t error,FixedOutputPlan&out,uint64_t integer_addend) noexcept {
    if(!input.value||input.negative||!input.extent_known||!fractional||fractional>(size_t(1)<<31)||
       input.extent.end>input.first_limb+input.limbs()||
       (input.extent.end>input.first_limb&&!input.extent.top_word))return false;
    const __int128 discard=-(__int128(input.exponent2)+64*__int128(fractional));
    const __int128 bits=input.extent.end>input.first_limb?
        64*__int128(input.extent.end-input.first_limb-1)+64-__builtin_clzll(input.extent.top_word):0;
    if(bits-discard>64*__int128(fractional+1))return false;
    return query_fixed_output_source(input.source(),input.exponent2,fractional,error,out,integer_addend);
}
bool query_fixed_output_source(SourceWindow source,int64_t exponent,size_t fractional,uint64_t error,FixedOutputPlan&out,uint64_t integer_addend) noexcept {
    if(!fractional||fractional>(size_t(1)<<31)||source.limbs==SIZE_MAX||!resolve_source_window(source.limbs,source))return false;
    FixedOutputPlan p{};p.input=source;p.exponent2=exponent;p.fractional=fractional;p.error_units=error;p.integer_addend=integer_addend;
    const __int128 discard=-(__int128(exponent)+64*__int128(fractional));
    if(discard<1||discard>INT64_MAX||(discard<65&&__uint128_t(error)*2>=(__uint128_t(1)<<unsigned(discard))))return false;
    p.discarded_bits=uint64_t(discard);
    if(!mul_size(fractional+1,8,p.output_bytes))return false;RegionLayout layout{};
    if(region_layout(p.output_bytes,0,layout))return false;
    p.output_allocation=layout.allocation_bytes;p.write_bytes=layout.data_bytes+layout.index_bytes;
    size_t bytes;if(!align_size(size_t(p.output_bytes),chunk_bytes,bytes))return false;
    p.transfer_bytes=std::min(size_t(1)<<20,std::max(chunk_bytes,bytes));
    size_t at=0;auto put=[&](size_t n,size_t&offset){return align_size(at,4096,offset)&&add_size(offset,n,at);};
    for(unsigned i=0;i<2;++i)if(!put(p.transfer_bytes,p.io[i])||!put(index_page_bytes,p.index[i]))return false;
    if(!put(p.transfer_bytes+16,p.data)||!align_size(at,4096,p.storage_bytes))return false;
    out=p;return true;
}
IoError run_fixed_output(const FixedOutputPlan&p,Transport&t,ValueStore&store,const StoredInteger&input,
                         StoredInteger&out,std::span<uint8_t>w) noexcept {
    if(!input.value||input.value.store()!=&store||input.negative||!input.extent_known||input.source()!=p.input||input.exponent2!=p.exponent2||
       out.value||!w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes)return {IoCode::invalid,"fixed output binding"};
    if(input.extent.end>input.first_limb+input.limbs()||
       (input.extent.end>input.first_limb&&!input.extent.top_word))return {IoCode::invalid,"fixed output magnitude"};
    const uint64_t input_bits=input.extent.end>input.first_limb?
        64*(input.extent.end-input.first_limb-1)+64-unsigned(__builtin_clzll(input.extent.top_word)):0;
    if(__int128(input_bits)-p.discarded_bits>64*__int128(p.fractional+1))return {IoCode::invalid,"fixed output high support"};
    if(t.counters.write_submitted>t.write_limit||p.write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"fixed output write admission",0,p.write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    StoreBuffers read_io{w.subspan(p.io[0],p.transfer_bytes),w.subspan(p.index[0],index_page_bytes)};
    StoreBuffers write_io{w.subspan(p.io[1],p.transfer_bytes),w.subspan(p.index[1],index_page_bytes)};
    ValueReader reader;if(auto e=reader.open(t,input.value.region(),read_io))return e;
    auto data=w.subspan(p.data,p.transfer_bytes+16);const uint64_t g=p.discarded_bits;
    const uint64_t guard_words=g/64+(g%64!=0),available=input.limbs();uint64_t low=0;
    bool any_high=false,all_high=guard_words<=available;
    if(p.error_units){
        for(size_t first=0;first<std::min(guard_words,available);){const size_t count=std::min(p.transfer_bytes/8,size_t(std::min(guard_words,available)-first));
            if(auto e=read_source_words(reader,p.input,first,count,data))return e;
            for(size_t j=0;j<count;++j){const size_t at=first+j;uint64_t v;std::memcpy(&v,data.data()+8*j,8);
                const uint64_t mask=at+1==guard_words&&g%64?(uint64_t(1)<<(g%64))-1:UINT64_MAX;
                v&=mask;if(!at)low=v;else {any_high|=v!=0;all_high&=v==mask;}
            }
            first+=count;
        }
        const bool retained_zero=input_bits<=g;
        const bool lower=retained_zero||any_high||low>=p.error_units;
        const bool upper=g<=64?__uint128_t(low)+p.error_units<=(__uint128_t(1)<<unsigned(g)):
            !all_high||__uint128_t(low)+p.error_units<=(__uint128_t(1)<<64);
        require(lower&&upper,SBN3_FATAL_MATH,"stored constant guard separation");
    }
    RegionLease result;if(auto e=store.reserve(p.output_bytes,RegionKind::value,result))return e;
    ValueWriter writer;WordExtent extent;if(auto e=writer.open(t,result.region(),write_io,&extent))return e;
    const size_t skip=size_t(g/64),out_words=p.fractional+1;
    SourceWindow source=p.input;source.first_limb+=std::min(skip,input.limbs());source.limbs=skip<input.limbs()?input.limbs()-skip:0;source.bit_shift=-int(g%64);
    for(size_t first=0;first<out_words;){const size_t count=std::min(p.transfer_bytes/8,out_words-first);
        std::memset(data.data(),0,count*8+16);
        const size_t read=first<source.limbs?std::min(count,source.limbs-first):0;
        if(read)if(auto e=read_source_words(reader,source,first,read,data))return e;
        if(first+count==out_words&&p.integer_addend){uint64_t high;std::memcpy(&high,data.data()+8*(count-1),8);
            const __uint128_t sum=__uint128_t(high)+p.integer_addend;require(!(sum>>64),SBN3_FATAL_MATH,"fixed output integer addend");
            high=uint64_t(sum);std::memcpy(data.data()+8*(count-1),&high,8);}
        if(auto e=writer.append(data.first(count*8)))return e;first+=count;
    }
    if(auto e=writer.seal())return e;if(auto e=result.publish(out.value))return e;
    out.exponent2=-int64_t(64*p.fractional);out.extent=extent;out.extent_known=true;return {};
}
} // namespace sbn::v3::external
