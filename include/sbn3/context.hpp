// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/expression.hpp"
#include "sbn3/prepared_divisor.hpp"
#include "sbn3/float_constants.hpp"
namespace sbn {
template<expr::expression E>class prepared_expression {
    E expression_;memory_resource*values_;workspace work_;sbn3_team*team_;expression_requirements need_;
public:
    prepared_expression(E e,memory_resource&values,memory_resource&scratch,sbn3_team*team)noexcept
      :expression_(std::move(e)),values_(&values),work_(scratch),team_(team){
        expr::instruction_list<E> list(expression_);need_=detail::expression_query(list.view(),team?sbn3_team_workers(team):1);
        detail::require(need_.supported(),SBN3_FATAL_ARGUMENT,need_.reason);work_.reserve(need_.scratch_bytes,need_.scratch_alignment);
        detail::expression_compile(list.view(),work_.on(team_));
    }
    prepared_expression(const prepared_expression&)=delete;prepared_expression&operator=(const prepared_expression&)=delete;
    prepared_expression(prepared_expression&&)=default;prepared_expression&operator=(prepared_expression&&)=default;
    const expression_requirements&requirements()const noexcept{return need_;}
    void execute_into(integer_ref out,std::span<const integer_view>inputs={})noexcept{expr::instruction_list<E> list(expression_);detail::expression_execute(out,list.view(),work_.on(team_),inputs);}
    template<detail::expression_argument... V>requires(sizeof...(V)>0)
    void execute_into(integer_ref out,const V&...v)noexcept{std::array<integer_view,sizeof...(V)>inputs{detail::argument_view(v)...};execute_into(out,std::span<const integer_view>{inputs});}
    integer evaluate(std::span<const integer_view>inputs={})noexcept{integer out(*values_);out.reserve(need_.output_limbs);execute_into(out.ref(),inputs);return out;}
    template<detail::expression_argument... V>integer operator()(const V&...v)noexcept{
        std::array<integer_view,sizeof...(V)>inputs{detail::argument_view(v)...};return evaluate(std::span<const integer_view>{inputs});
    }
};
// A resident resource/execution authority. It owns scratch or borrows a fixed
// span, and never creates an implicit worker pool. Values, scratch providers
// and the optional existing Team have independent caller-controlled lifetimes.
class context {
    memory_resource*values_;memory_resource*scratch_resource_;workspace work_;sbn3_team*team_;
    constant_bank constants_;
    std::span<std::byte>external_{};bool borrowed_=false;
public:
    explicit context(memory_resource&r=default_resource(),sbn3_team*team=nullptr)noexcept:context(r,r,team){}
    context(memory_resource&values,memory_resource&scratch,sbn3_team*team=nullptr)noexcept:values_(&values),scratch_resource_(&scratch),work_(scratch),team_(team),constants_(scratch){}
    context(std::span<std::byte>scratch,memory_resource&values=default_resource(),sbn3_team*team=nullptr)noexcept:context(values,values,team){external_=scratch;borrowed_=true;}
    context(const context&)=delete;context&operator=(const context&)=delete;
    context(context&&)=default;context&operator=(context&&)=default;
    unsigned workers()const noexcept{return team_?sbn3_team_workers(team_):1;}
    sbn::execution execution()&noexcept{return borrowed_?sbn::execution{external_,team_,nullptr}:work_.on(team_);}
    sbn::execution execution()&&=delete;
    resource_error try_reserve(const requirements&r)noexcept{
        if(!r.supported())return {resource_errc::capacity,r.scratch_bytes,0,0,r.reason};
        if(!borrowed_)return work_.try_reserve(r.scratch_bytes,r.scratch_alignment);
        if(external_.size()<r.scratch_bytes||(r.scratch_bytes&&(uintptr_t(external_.data())&(r.scratch_alignment-1))))
            return {resource_errc::capacity,r.scratch_bytes,external_.size(),0,"borrowed context scratch"};
        return {};
    }
    void reserve(const requirements&r)noexcept{auto error=try_reserve(r);detail::require(!error,SBN3_FATAL_WORKSPACE,error.where,error.need,error.have);}
    // Explicit preparation boundary. Constants retain separate immutable
    // storage after this shared construction scratch is reused by arithmetic.
    resource_error try_prepare_constants(constant_demand demand)noexcept{
        if(!constants_.view().empty()&&constants_.view().supports(demand))return {};
        const auto need=constants_requirements(demand,workers());
        if(auto error=try_reserve(need.work))return error;
        return constants_.try_prepare(demand,execution());
    }
    void prepare_constants(constant_demand demand)noexcept{
        const auto error=try_prepare_constants(demand);
        detail::require(!error,SBN3_FATAL_WORKSPACE,error.where,error.need,error.have);
    }
    constant_view constants()const&noexcept{return constants_.view();}
    constant_view constants()const&&=delete;
    template<expr::expression E>expression_requirements query(const E&e)const noexcept{
        expr::instruction_list<E> list(e);return detail::expression_query(list.view(),workers());
    }
    template<expr::expression E>expression_requirements query_into(integer_ref out,const E&e)const noexcept{
        auto r=query(e);if(r.supported()&&out.capacity()<r.output_limbs){r.status=support::capacity;r.reason="expression output capacity";}return r;
    }
    // Strict: neither value nor scratch storage grows. The output may alias a
    // raw input view; managed owner borrows deliberately forbid such mutation.
    template<expr::expression E>void evaluate_into(integer_ref out,const E&e,std::span<const integer_view>inputs={})noexcept{
        expr::instruction_list<E> list(e);detail::expression_compile(list.view(),execution(),inputs);detail::expression_execute(out,list.view(),execution(),inputs);
    }
    template<expr::expression E,detail::expression_argument... V>requires(sizeof...(V)>0)
    void evaluate_into(integer_ref out,const E&e,const V&...v)noexcept{
        std::array<integer_view,sizeof...(V)>inputs{detail::argument_view(v)...};evaluate_into(out,e,std::span<const integer_view>{inputs});
    }
    // Consumes temporary requests before publishing an owning result. Thus
    // a = ctx.evaluate(expr::mul(a,b)) releases its temporary reader guards
    // before a's assignment. An explicitly retained request keeps its borrows.
    template<expr::expression E,detail::expression_argument... V>integer evaluate(E e,const V&...v)noexcept{
        integer result(*values_);
        std::array<integer_view,sizeof...(V)>inputs{detail::argument_view(v)...};
        {E request(std::move(e));expr::instruction_list<E> list(request);const auto r=detail::expression_query(list.view(),workers());
            detail::require(r.supported(),SBN3_FATAL_ARGUMENT,r.reason);reserve(r);result.reserve(r.output_limbs);
            detail::expression_compile(list.view(),execution(),inputs);detail::expression_execute(result.ref(),list.view(),execution(),inputs);}
        return result;
    }
    template<expr::expression E>auto prepare(E e)noexcept{return prepared_expression<E>(std::move(e),*values_,*scratch_resource_,team_);}
    prepared_divisor prepare_divisor(const integer&d,std::size_t max_numerator_limbs,unsigned reuse_hint=0)noexcept{
        return prepared_divisor(d.view(),max_numerator_limbs,*scratch_resource_,team_,reuse_hint);
    }
};
}
