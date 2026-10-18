// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/integer.hpp"
#include "value/bitwise.hpp"
namespace sbn {
void integer::dispose()noexcept{if(value_.data!=small_)resource().deallocate({value_.data,value_.capacity*8,64});}
void integer::move_from(integer&x)noexcept{
    detail::writable(&x.state_);
    if(x.value_.data==x.small_){small_[0]=x.small_[0];small_[1]=x.small_[1];value_.size=x.size();value_.negative=x.negative();}
    else value_=x.value_;
    x.empty_inline();detail::changed(&x.state_);
}
integer&integer::operator=(const integer&x)noexcept{if(this!=&x)assign(x.view());return *this;}
integer&integer::operator=(integer&&x)noexcept{
    if(this==&x)return *this;detail::writable(&state_);detail::writable(&x.state_);
    if(&resource()!=&x.resource()){assign(x.view());x.reset();return *this;}
    dispose();empty_inline();move_from(x);detail::changed(&state_);return *this;
}
resource_error integer::try_reserve(std::size_t n)noexcept{
    if(n<=capacity())return {};
    if(n>SIZE_MAX/64)return {resource_errc::capacity,n,SIZE_MAX/64,0,"integer limb extent"};
    if(__atomic_load_n(&state_,__ATOMIC_ACQUIRE)&detail::borrow_mask)return {resource_errc::busy,n,capacity(),0,"reserve borrowed integer"};
    allocation next;const auto error=resource().try_allocate(n*8,64,next);if(error)return error;
    detail::require(next.data&&next.bytes>=n*8&&next.bytes%8==0&&!(std::uintptr_t(next.data)&63),SBN3_FATAL_WORKSPACE,"integer resource contract",n*8,next.bytes);
    if(size())std::memcpy(next.data,data(),size()*8);dispose();value_.data=static_cast<std::uint64_t*>(next.data);value_.capacity=next.bytes/8;detail::changed(&state_);return {};
}
void integer::assign(integer_view v)noexcept{
    detail::writable(&state_);reserve(v.size());if(v.size())std::memmove(value_.data,v.data(),v.size()*8);value_.size=v.size();value_.negative=v.negative();detail::changed(&state_);
}
void integer::swap(integer&x)noexcept{
    if(this==&x)return;detail::writable(&state_);detail::writable(&x.state_);
    const bool a_inline=value_.data==small_,b_inline=x.value_.data==x.small_;
    std::swap(value_,x.value_);std::swap(small_[0],x.small_[0]);std::swap(small_[1],x.small_[1]);std::swap(resource_,x.resource_);
    if(b_inline)value_.data=small_;if(a_inline)x.value_.data=x.small_;
    detail::changed(&state_);detail::changed(&x.state_);
}
namespace detail {
integer large_sum(const integer&a,const integer&b,bool subtract)noexcept{
    integer r=integer::with_resource_of(a);r.reserve(std::max(a.size(),b.size())+1);
    if(subtract)subtract_into(r.ref(),a.view(),b.view());else add_into(r.ref(),a.view(),b.view());return r;
}
integer large_bitwise(const integer&a,const integer&b,sbn3_bit_operation op)noexcept{
    const auto shape=sbn::v3::bits::shape(op,a.size(),a.negative(),b.size(),b.negative());
    integer r=integer::with_resource_of(a);r.reserve(shape.words+unsigned(shape.negative));
    if(op==SBN3_BIT_AND)and_into(r.ref(),a.view(),b.view());else if(op==SBN3_BIT_OR)or_into(r.ref(),a.view(),b.view());else xor_into(r.ref(),a.view(),b.view());return r;
}
}
integer&integer::operator+=(const integer&x)noexcept{
    if(x.zero())return *this;
    if(size()<=2&&x.size()<=2){*this=*this+x;return *this;}reserve(std::max(size(),x.size())+1);add_into(ref(),view(),x.view());return *this;
}
integer&integer::operator-=(const integer&x)noexcept{
    if(x.zero())return *this;if(this==&x){clear();return *this;}
    if(size()<=2&&x.size()<=2){*this=*this-x;return *this;}reserve(std::max(size(),x.size())+1);subtract_into(ref(),view(),x.view());return *this;
}
integer&integer::operator&=(const integer&x)noexcept{
    if(this==&x)return *this;if(x.zero()){clear();return *this;}
    if(size()<=2&&x.size()<=2){*this=*this&x;return *this;}
    const auto shape=sbn::v3::bits::shape(SBN3_BIT_AND,size(),negative(),x.size(),x.negative());reserve(shape.words+unsigned(shape.negative));and_into(ref(),view(),x.view());return *this;
}
integer&integer::operator|=(const integer&x)noexcept{
    if(this==&x||x.zero())return *this;if(size()<=2&&x.size()<=2){*this=*this|x;return *this;}
    const auto shape=sbn::v3::bits::shape(SBN3_BIT_OR,size(),negative(),x.size(),x.negative());reserve(shape.words+unsigned(shape.negative));or_into(ref(),view(),x.view());return *this;
}
integer&integer::operator^=(const integer&x)noexcept{
    if(x.zero())return *this;if(this==&x){clear();return *this;}if(size()<=2&&x.size()<=2){*this=*this^x;return *this;}
    const auto shape=sbn::v3::bits::shape(SBN3_BIT_XOR,size(),negative(),x.size(),x.negative());reserve(shape.words+unsigned(shape.negative));xor_into(ref(),view(),x.view());return *this;
}
integer&integer::operator<<=(std::size_t bits)noexcept{
    if(zero())return *this;const auto words=bits/64;detail::require(words<=SIZE_MAX/64-size()-1,SBN3_FATAL_SIZE,"integer left shift extent");reserve(size()+words+unsigned(bits%64!=0));lshift_into(ref(),view(),bits);return *this;
}
integer&integer::operator>>=(std::size_t bits)noexcept{rshift_into(ref(),view(),bits);return *this;}
integer&integer::set_bit(std::size_t bit,bool value)noexcept{
    if(test_bit(bit)==value)return *this;const std::size_t word=bit/64;detail::require(word<SIZE_MAX/64-1,SBN3_FATAL_SIZE,"integer bit index");
    reserve(std::max(size(),word+1)+1);auto r=ref();sbn3_int_edit_bit(r.begin_write(),view().c_view(),bit,value?SBN3_BIT_SET:SBN3_BIT_CLEAR);r.finish_write();return *this;
}
}
