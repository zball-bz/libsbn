// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/bytes.hpp"
#include "common/checked.hpp"
#include "value/bitwise.hpp"
#include <bit>
namespace sbn {
namespace {
using U=std::uint64_t;
void valid(byte_format f){detail::require(unsigned(f.order)<=1&&unsigned(f.encoding)<=1,SBN3_FATAL_ARGUMENT,"integer byte format");}
unsigned octet(std::byte b){return std::to_integer<unsigned>(b);}
std::byte top(std::span<const std::byte> b,byte_order o,std::size_t k=0){return b[o==byte_order::big?k:b.size()-1-k];}
std::span<const std::byte> stripped(std::span<const std::byte>b,byte_format f,bool&negative){
    valid(f);negative=!b.empty()&&f.encoding==integer_encoding::twos_complement&&bool(octet(top(b,f.order))&128);
    const unsigned fill=negative?255:0;
    auto removable=[&](std::size_t n){return f.encoding==integer_encoding::magnitude||bool(octet(top(b,f.order,n))&128)==negative;};
    while(b.size()>8){U w;const auto*p=f.order==byte_order::big?b.data():b.data()+b.size()-8;std::memcpy(&w,p,8);
        if(w!=(negative?UINT64_MAX:0)||!removable(8))break;
        b=f.order==byte_order::big?b.subspan(8):b.first(b.size()-8);
    }
    while(b.size()>1&&octet(top(b,f.order))==fill&&removable(1))b=f.order==byte_order::big?b.subspan(1):b.first(b.size()-1);
    return b;
}
U load(std::span<const std::byte>b,std::size_t offset,byte_order order){
    const auto count=std::min(std::size_t(8),b.size()-offset);U v=0;
    if(count==8){const auto*p=order==byte_order::little?b.data()+offset:b.data()+b.size()-offset-8;std::memcpy(&v,p,8);
        if((order==byte_order::little)!=(std::endian::native==std::endian::little))v=__builtin_bswap64(v);
    }else for(std::size_t j=0;j<count;++j)v|=U(octet(b[order==byte_order::little?offset+j:b.size()-1-offset-j]))<<(8*j);
    return v;
}
void decode(integer_ref dst,std::span<const std::byte>b,byte_format f,bool negative){
    const auto n=b.size()/8+unsigned(b.size()%8!=0);auto*r=dst.begin_write();
    detail::require(n<=r->capacity,SBN3_FATAL_WORKSPACE,"integer import capacity",n,r->capacity);
    detail::require(!sbn::v3::overlaps(r->data,n*8,b.data(),b.size()),SBN3_FATAL_ARGUMENT,"integer import alias");
    bool carry=negative;
    for(std::size_t j=0;j<n;++j){U w=load(b,8*j,f.order);
        if(negative){w=~w;const U old=w;w+=carry;carry=carry&&w<old;
            const auto tail=b.size()-8*j;if(tail<8)w&=(U(1)<<(8*tail))-1;}
        r->data[j]=w;
    }
    r->size=sbn::v3::bits::normalized_size(r->data,n);r->negative=r->size&&negative;dst.finish_write();
}
}
std::size_t export_bytes_size(integer_view v,integer_encoding encoding)noexcept{
    detail::require(unsigned(encoding)<=1,SBN3_FATAL_ARGUMENT,"integer byte encoding");
    const std::size_t bits=sbn3_int_bit_length(v.c_view());if(!bits)return 1;
    if(encoding==integer_encoding::magnitude)return bits/8+unsigned(bits%8!=0);
    if(!v.negative())return bits/8+1;
    bool power=false;if(!(bits%8)){const auto at=sbn::v3::bits::first_nonzero(v.data(),v.size());power=at+1==v.size()&&!(v.data()[at]&(v.data()[at]-1));}
    return bits/8+1-unsigned(power);
}
byte_result export_bytes(std::span<std::byte>dst,integer_view v,byte_format f)noexcept{
    valid(f);const auto n=export_bytes_size(v,f.encoding);if(dst.size()<n)return {dst.data(),std::errc::value_too_large};
    detail::require(!sbn::v3::overlaps(dst.data(),n,v.data(),v.size()*8),SBN3_FATAL_ARGUMENT,"integer export alias");
    const bool negative=v.negative()&&f.encoding==integer_encoding::twos_complement;bool carry=negative;
    for(std::size_t i=0;i<n;){const auto j=i/8;U w=j<v.size()?v.data()[j]:0;
        if(negative){w=~w;const U old=w;w+=carry;carry=carry&&w<old;}
        const auto take=std::min(std::size_t(8),n-i);
        if(take==8){U bits=w;if((f.order==byte_order::little)!=(std::endian::native==std::endian::little))bits=__builtin_bswap64(bits);
            auto*p=f.order==byte_order::little?dst.data()+i:dst.data()+n-i-8;std::memcpy(p,&bits,8);
        }else for(std::size_t k=0;k<take;++k)dst[f.order==byte_order::little?i+k:n-1-i-k]=std::byte(w>>(8*k));
        i+=take;
    }
    return {dst.data()+n,{}};
}
void import_bytes_into(integer_ref r,std::span<const std::byte> b,byte_format f)noexcept{bool sign;const auto s=stripped(b,f,sign);decode(r,s,f,sign);}
integer from_bytes(std::span<const std::byte>b,byte_format f,memory_resource&r)noexcept{
    bool sign;const auto s=stripped(b,f,sign);integer out(r);out.reserve(s.size()/8+unsigned(s.size()%8!=0));decode(out.ref(),s,f,sign);return out;
}
}
