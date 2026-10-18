// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/memory.hpp"
#include "sbn3/team.h"
namespace sbn {
// Resident computation view. Inputs/output are separate caller-owned spans.
// Scratch is exclusive throughout the call. A supplied Team is already idle;
// a supplied scope is its current leader's admitted subteam. Neither is owned.
struct execution {
    std::span<std::byte> scratch{};
    sbn3_team*team=nullptr;
    sbn3_team_scope*scope=nullptr;
    unsigned workers()const noexcept{return scope?sbn3_team_width(scope):team?sbn3_team_workers(team):1;}
};
class workspace {
    memory_resource*resource_;allocation block_{};
public:
    explicit workspace(memory_resource&r=default_resource())noexcept:resource_(&r){}
    workspace(const workspace&)=delete;workspace&operator=(const workspace&)=delete;
    workspace(workspace&&x)noexcept:resource_(x.resource_),block_(x.block_){x.block_={};}
    workspace&operator=(workspace&&x)noexcept{if(this!=&x){resource_->deallocate(block_);resource_=x.resource_;block_=x.block_;x.block_={};}return *this;}
    ~workspace(){resource_->deallocate(block_);}
    resource_error try_reserve(std::size_t n,std::size_t alignment=128)noexcept{
        if(!alignment||(alignment&(alignment-1)))return {resource_errc::capacity,alignment,0,0,"workspace alignment"};
        if(n<=block_.bytes&&(!n||!(std::uintptr_t(block_.data)&(alignment-1))))return {};
        allocation a;auto error=resource_->try_allocate(n,alignment,a);if(error)return error;
        resource_->deallocate(block_);block_=a;return {};
    }
    void reserve(std::size_t n,std::size_t alignment=128)noexcept{auto e=try_reserve(n,alignment);detail::require(!e,SBN3_FATAL_WORKSPACE,e.where,e.need,e.have);}
    std::size_t capacity()const noexcept{return block_.bytes;}
    execution on(sbn3_team*team=nullptr)&noexcept{return {{static_cast<std::byte*>(block_.data),block_.bytes},team,nullptr};}
    execution on(sbn3_team*)&&=delete;
    execution on_scope(sbn3_team_scope*scope)&noexcept{return {{static_cast<std::byte*>(block_.data),block_.bytes},nullptr,scope};}
    execution on_scope(sbn3_team_scope*)&&=delete;
};
enum class support {yes,unsupported,capacity};
struct requirements {
    support status=support::yes;
    std::size_t scratch_bytes=0,scratch_alignment=128,output_limbs=0;
    const char*reason="";
    bool supported()const noexcept{return status==support::yes;}
};
}
