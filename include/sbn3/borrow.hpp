// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/integer.hpp"
namespace sbn {
// A managed immutable borrow for explicit requests/prepared readers. Owning
// integers cannot be moved, mutated or destroyed while such a reader lives.
// Ordinary integer_view remains a lightweight, caller-managed C++ borrow.
class borrowed_integer {
    integer_view view_{};std::uint64_t*state_=nullptr;
    void acquire()noexcept{
        if(!state_)return;
        auto value=__atomic_load_n(state_,__ATOMIC_ACQUIRE);
        for(;;){detail::require((value&detail::borrow_mask)!=detail::borrow_mask,SBN3_FATAL_LIFETIME,"too many integer readers");
            if(__atomic_compare_exchange_n(state_,&value,value+1,false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE))break;}
    }
    void release()noexcept{if(state_){__atomic_fetch_sub(state_,1,__ATOMIC_RELEASE);state_=nullptr;}}
public:
    explicit borrowed_integer(const integer&x)noexcept:view_(x.view()),state_(&x.state_){acquire();}
    borrowed_integer(integer&&)=delete;borrowed_integer(const integer&&)=delete;
    borrowed_integer(const borrowed_integer&x)noexcept:view_(x.view_),state_(x.state_){acquire();}
    borrowed_integer(borrowed_integer&&x)noexcept:view_(x.view_),state_(x.state_){x.state_=nullptr;x.view_={};}
    borrowed_integer&operator=(const borrowed_integer&x)noexcept{if(this!=&x){release();view_=x.view_;state_=x.state_;acquire();}return *this;}
    borrowed_integer&operator=(borrowed_integer&&x)noexcept{if(this!=&x){release();view_=x.view_;state_=x.state_;x.state_=nullptr;x.view_={};}return *this;}
    ~borrowed_integer(){release();}
    integer_view view()const&noexcept{return view_;}integer_view view()const&&=delete;
    std::uint64_t generation()const noexcept{return state_?__atomic_load_n(state_,__ATOMIC_ACQUIRE)/detail::version_step:0;}
};
inline borrowed_integer integer::borrow()const&noexcept{return borrowed_integer(*this);}
}
