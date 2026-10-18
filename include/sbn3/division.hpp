// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/integer.hpp"
#include "sbn3/detail/division.hpp"
namespace sbn {
enum class division_rounding {trunc,floor,ceil,euclidean};
struct division_result {integer quotient,remainder;};
struct division_requirements : requirements {std::size_t quotient_limbs=0,remainder_limbs=0;};
// Missing outputs are not published. The quote includes any required private
// Q/R storage, alias staging and existing backend preparation. Outputs must
// not overlap each other. They may alias either input; scratch is disjoint.
division_requirements divrem_requirements(std::optional<integer_ref> quotient,std::optional<integer_ref> remainder,
    integer_view numerator,integer_view divisor,division_rounding=division_rounding::trunc,unsigned workers=1)noexcept;
namespace detail {
void general_divrem_into(std::optional<integer_ref> quotient,std::optional<integer_ref> remainder,
    integer_view numerator,integer_view divisor,execution={},division_rounding=division_rounding::trunc)noexcept;
}
inline void divrem_into(std::optional<integer_ref>q,std::optional<integer_ref>r,integer_view n,integer_view d,
                       execution ex={},division_rounding mode=division_rounding::trunc)noexcept{
    if(q&&r&&mode==division_rounding::trunc&&d.size()&&n.size()>=d.size()&&d.size()<=32768&&
       (d.size()<=2||(!ex.scope&&ex.workers()==1))){
        const auto qn=n.size()-d.size()+1,nn=n.size()*8,dn=d.size()*8;
        const auto qp=std::uintptr_t(q->view().data()),rp=std::uintptr_t(r->view().data()),np=std::uintptr_t(n.data()),dp=std::uintptr_t(d.data());
        const bool inputs_separate=np+nn<=dp||dp+dn<=np;
        const bool results_separate=qp+qn*8<=rp||rp+dn<=qp;
        const auto input_low=std::min(np,dp),input_high=std::max(np+nn,dp+dn);
        const auto output_low=std::min(qp,rp),output_high=std::max(qp+qn*8,rp+dn);
        // A cheap bounding-interval proof covers the usual disjoint values.
        // Interleaved buffers, all aliases, and alternate modes use full layout.
        if(inputs_separate&&results_separate&&(output_high<=input_low||input_high<=output_low)){
            detail::require(q->capacity()>=qn&&r->capacity()>=d.size(),SBN3_FATAL_WORKSPACE,"local integer division outputs");
            auto*Q=q->begin_write();auto*R=r->begin_write();
            const auto sizes=detail::local_divrem_checked(Q->data,R->data,n.data(),n.size(),d.data(),d.size(),ex.scratch.data(),ex.scratch.size());
            Q->size=sizes.quotient;Q->negative=Q->size&&(n.negative()!=d.negative());R->size=sizes.remainder;R->negative=R->size&&n.negative();q->finish_write();r->finish_write();return;
        }
    }
    detail::general_divrem_into(q,r,n,d,ex,mode);
}
division_result divrem(const integer&,const integer&,division_rounding=division_rounding::trunc)noexcept;
integer divide(const integer&,const integer&,division_rounding=division_rounding::trunc)noexcept;
integer remainder(const integer&,const integer&,division_rounding=division_rounding::trunc)noexcept;
// A nonnegative remainder for either sign of the nonzero divisor.
inline integer mod(const integer&a,const integer&d)noexcept{return remainder(a,d,division_rounding::euclidean);}
// Exactness is a mathematical precondition, checked before returning.
integer exact_divide(const integer&,const integer&)noexcept;
// Empty result for a nonzero remainder; zero divisor is still invalid.
std::optional<integer> try_exact_divide(const integer&,const integer&)noexcept;
// 0 divides only 0; congruence modulo 0 is equality.
bool is_divisible(const integer&numerator,const integer&divisor)noexcept;
bool is_congruent(const integer&,const integer&,const integer&modulus)noexcept;
}
