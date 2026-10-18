// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "runtime/team.hpp"

namespace sbn::v3 {
// A product binding belongs either to its historical idle controller or to
// one explicitly borrowed, joined scope episode. No Team state is forged or
// changed. The borrowed scope and its bindings must close before that episode
// returns; plans and immutable metadata may of course outlive it.
class BoundOwner {
    mutable sbn3_team_scope scope_{};
    bool scoped_=false;
public:
    bool scoped()const noexcept{return scoped_;}
    void bind(Arena& arena,sbn3_team& team,sbn3_team_scope* scope,unsigned workers)noexcept{
        require(workers&&workers<=team.width,SBN3_FATAL_TEAM,"binding owner worker count");
        if(scope){
            require_scope_leader(scope);
            require(scope->team==&team&&scope->width>=workers&&
                (arena.accepts_team(team.arena)||(arena.borrowed&&arena.borrowed_team==&team)),
                SBN3_FATAL_TEAM,"scoped binding arena/team");
            scope_=*scope;scope_.active=false;scoped_=true;
        }else{
            require(arena.accepts_team(team.arena)&&pthread_equal(team.creator,pthread_self())&&!team.busy,
                SBN3_FATAL_TEAM,"idle binding arena/team");
            scope_={};scoped_=false;
        }
    }
    void check(const sbn3_team& team)const noexcept{
        if(scoped_){require(scope_.team==&team,SBN3_FATAL_TEAM,"scoped binding team identity");require_scope_leader(&scope_);}
        else require(pthread_equal(team.creator,pthread_self())&&!team.busy,SBN3_FATAL_TEAM,"binding owner/active");
    }
    sbn3_team_scope* resolve(const sbn3_team& team,sbn3_team_scope* provided,unsigned workers)const noexcept{
        if(scoped_)check(team);
        auto* result=provided?provided:scoped_?&scope_:nullptr;
        if(result){
            require_scope_leader(result);
            require(result->team==&team&&result->width>=workers,SBN3_FATAL_TEAM,"binding execution scope");
            if(scoped_)require(result->first==scope_.first&&result->width<=scope_.width&&result->epoch==scope_.epoch,
                SBN3_FATAL_TEAM,"binding scope ownership");
        }else check(team);
        return result;
    }
    void run(sbn3_team& team,sbn3_action_fn action,void* arg)const noexcept{
        check(team);
        if(scoped_)action(arg,&scope_);else sbn3_team_run(&team,action,arg);
    }
};
}
