// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "product/backend.hpp"
#include "runtime/bound_owner.hpp"

namespace sbn::v3 {
inline void product_bind_owned(const sbn3_mul_plan& p,sbn3_arena& arena,
    const sbn3_lease& tables,const sbn3_lease& work,sbn3_team& team,sbn3_team_scope* scope,
    const sbn3_spectrum* a,const sbn3_spectrum* b,sbn3_mul_binding** out){
    if(!scope){sbn3_product_bind(&p,&arena,&tables,&work,&team,a,b,out);return;}
    const auto* backend=backend_lookup(p.opaque[1]);
    require(backend&&backend->product_bind_scope,SBN3_FATAL_ARGUMENT,"scoped product provider");
    require_scope_leader(scope);require(scope->team==&team,SBN3_FATAL_TEAM,"scoped product Team");
    backend->product_bind_scope(p,arena,tables,work,scope,a,b,out);
}
}
