// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "runtime/team.hpp"
#include <span>
#include <cstddef>

namespace sbn::v3 {
// Adapts an already-provisioned exclusive span to the existing service lease
// protocol. No mapping, page operation or worker creation. This descriptor
// must outlive every binding and spectrum made from it. It is only needed by
// general services; leaf calls continue to use raw Frames/direct kernels.
class ResidentSession {
    sbn3_arena arena_{};
    sbn3_team serial_{};
    sbn3_team *team_;
public:
    ResidentSession(std::span<std::byte> bytes,sbn3_team *team=nullptr) noexcept :team_(team?team:&serial_) {
        valid_span(bytes.data(),bytes.size(),"resident service range");
        require(bytes.size(),SBN3_FATAL_WORKSPACE,"empty resident service range");
        arena_.base=reinterpret_cast<uint8_t *>(bytes.data());arena_.virtual_bytes=bytes.size();
        arena_.budget=arena_.resident_bytes=arena_.peak_bytes=bytes.size();
        arena_.regions[0]={0,bytes.size()};arena_.region_count=1;arena_.borrowed=true;
        require(!pthread_mutex_init(&arena_.mutex,nullptr),SBN3_FATAL_LIFETIME,"resident service mutex");
        if(team){
            require(team->width&&!team->busy&&pthread_equal(team->creator,pthread_self()),SBN3_FATAL_TEAM,"resident service owner");
            arena_.worker_arena=team->arena;
            for(unsigned i=0;i<team->width;++i){
                const auto &s=i?team->stacks[i]:team->storage;
                const auto &g=team->guards[i];
                require(!overlaps(bytes.data(),bytes.size(),s.data,s.bytes)&&
                        !overlaps(bytes.data(),bytes.size(),g.data,g.bytes),SBN3_FATAL_WORKSPACE,"resident service team alias");
            }
        }else{serial_.arena=&arena_;serial_.width=1;serial_.creator=pthread_self();}
    }
    ResidentSession(const ResidentSession&)=delete;
    ResidentSession&operator=(const ResidentSession&)=delete;
    ~ResidentSession(){
        sbn3_arena_stats s{};arena_.stats(s);
        require(!s.lease_references&&!s.active_computations&&!team_->busy,SBN3_FATAL_LIFETIME,"live resident service");
        require(!pthread_mutex_destroy(&arena_.mutex),SBN3_FATAL_LIFETIME,"resident service mutex destroy");
    }
    sbn3_arena *arena() noexcept{return &arena_;}
    sbn3_team *team() noexcept{return team_;}
};

// Exclusive resident storage borrowed by an active scope leader. The arena
// owns no pages or workers. Its exact Team identity also admits serial_team_run
// (whose temporary Team intentionally has no arena) without weakening ordinary
// Arena::accepts_team. All bindings/leases must close before destruction.
class ScopedResidentSession {
    sbn3_arena arena_{};
    sbn3_team_scope scope_{};
public:
    ScopedResidentSession(std::span<std::byte> bytes,sbn3_team_scope* scope)noexcept{
        require_scope_leader(scope);valid_span(bytes.data(),bytes.size(),"scoped resident range");
        require(bytes.size(),SBN3_FATAL_WORKSPACE,"empty scoped resident range");
        scope_=*scope;scope_.active=false;
        arena_.base=reinterpret_cast<uint8_t*>(bytes.data());arena_.virtual_bytes=bytes.size();
        arena_.budget=arena_.resident_bytes=arena_.peak_bytes=bytes.size();
        arena_.regions[0]={0,bytes.size()};arena_.region_count=1;arena_.borrowed=true;
        arena_.worker_arena=scope->team->arena;arena_.borrowed_team=scope->team;
        require(!pthread_mutex_init(&arena_.mutex,nullptr),SBN3_FATAL_LIFETIME,"scoped resident mutex");
        for(unsigned i=0;i<scope->team->width;++i){
            const auto& s=i?scope->team->stacks[i]:scope->team->storage;
            const auto& g=scope->team->guards[i];
            require(!overlaps(bytes.data(),bytes.size(),s.data,s.bytes)&&
                !overlaps(bytes.data(),bytes.size(),g.data,g.bytes),SBN3_FATAL_WORKSPACE,"scoped resident Team alias");
        }
    }
    ScopedResidentSession(const ScopedResidentSession&)=delete;
    ScopedResidentSession&operator=(const ScopedResidentSession&)=delete;
    ~ScopedResidentSession(){
        require_scope_leader(&scope_);sbn3_arena_stats stats{};arena_.stats(stats);
        require(!stats.lease_references&&!stats.active_computations,SBN3_FATAL_LIFETIME,"live scoped resident service");
        require(!pthread_mutex_destroy(&arena_.mutex),SBN3_FATAL_LIFETIME,"scoped resident mutex destroy");
    }
    sbn3_arena* arena()noexcept{return &arena_;}
    sbn3_team* team()noexcept{return static_cast<sbn3_team*>(scope_.team);}
    sbn3_team_scope* scope()noexcept{return &scope_;}
};
}
