// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/division.hpp"
namespace sbn {
// An immutable divisor snapshot and an exclusive, reusable execution workspace.
// Move-only; executions are sequential on the preparing thread. The original
// divisor may change/die after preparation. A Team, resource or borrowed span
// supplied by the caller must outlive the handle. No repeated divisor scan.
class prepared_divisor {
    void*state_=nullptr;memory_resource*resource_=nullptr;allocation allocation_{};
    void close()noexcept;
    explicit prepared_divisor(void*p)noexcept:state_(p){}
public:
    prepared_divisor()noexcept=default;
    prepared_divisor(integer_view,std::size_t max_numerator_limbs,memory_resource& =default_resource(),
                     sbn3_team*team=nullptr,unsigned reuse_hint=0)noexcept;
    prepared_divisor(const integer&d,std::size_t max_numerator_limbs,unsigned reuse_hint=0)noexcept
        :prepared_divisor(d.view(),max_numerator_limbs,d.resource(),nullptr,reuse_hint){}
    prepared_divisor(const prepared_divisor&)=delete;
    prepared_divisor&operator=(const prepared_divisor&)=delete;
    prepared_divisor(prepared_divisor&&x)noexcept:state_(x.state_),resource_(x.resource_),allocation_(x.allocation_){x.state_=nullptr;x.resource_=nullptr;x.allocation_={};}
    prepared_divisor&operator=(prepared_divisor&&x)noexcept{if(this!=&x){close();state_=x.state_;resource_=x.resource_;allocation_=x.allocation_;x.state_=nullptr;x.resource_=nullptr;x.allocation_={};}return *this;}
    ~prepared_divisor(){close();}
    static requirements storage_requirements(std::size_t divisor_limbs,std::size_t max_numerator_limbs,
                                              unsigned workers=1,unsigned reuse_hint=0)noexcept;
    static prepared_divisor bind(integer_view,std::size_t max_numerator_limbs,std::span<std::byte>,
                                 sbn3_team*team=nullptr,unsigned reuse_hint=0)noexcept;
    bool ready()const noexcept{return state_!=nullptr;}
    std::size_t storage_bytes()const noexcept;
    division_requirements query(std::optional<integer_ref>q,std::optional<integer_ref>r,integer_view,
                                 division_rounding=division_rounding::trunc)const noexcept;
    void execute(std::optional<integer_ref>q,std::optional<integer_ref>r,integer_view,execution={},
                  division_rounding=division_rounding::trunc)noexcept;
    division_result divrem(const integer&,division_rounding=division_rounding::trunc)noexcept;
    integer divide(const integer&,division_rounding=division_rounding::trunc)noexcept;
    integer remainder(const integer&,division_rounding=division_rounding::trunc)noexcept;
};
}
