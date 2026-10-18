// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "product/program.hpp"
#include "backend/u52/kernels.hpp"
#include <new>
#include <algorithm>

namespace sbn::v3::product {
struct IndependentProduct {sbn3_const_limbs a,b;sbn3_limbs out;};
// A bounded batch of independent, nonaliasing products. Each job uses a
// serial backend recipe; parallelism is across jobs. Plans and all storage
// are acquired inside the caller's operation, never retained across calls.
inline bool independent_products(const IndependentProduct *jobs,unsigned count,
                                 Frame &space,sbn3_team_scope *scope,unsigned limit=4){
    if(!scope||scope->width<2||count<2||count>8)return false;
    const unsigned workers=std::min({scope->width,count,limit});
    ProductProgramPlan plans[8]{};bool native[8]{};size_t per_worker=0;
    for(unsigned j=0;j<count;++j){
        const auto &x=jobs[j];const size_t n=x.a.count+x.b.count;
        require(n<=x.out.capacity,SBN3_FATAL_ARGUMENT,"independent product output");
        if(!x.a.count||!x.b.count)continue;
        sbn3_mul_options o{};o.workers=1;o.borrow_output=1;
        native[j]=std::min(x.a.count,x.b.count)>=96&&
            product_program_query(x.a.count,x.b.count,o,plans[j])==SBN3_SUPPORTED;
        const auto &q=plans[j];
        const size_t staging=native[j]&&q.info.output_limbs>x.out.capacity?q.info.output_limbs*8+64:0;
        const size_t bytes=native[j]?q.prepared_bytes+q.info.workspace_alignment+q.info.workspace_bytes+staging+512:
            u52::scratch_bytes(x.a.count,x.b.count)+256;
        per_worker=std::max(per_worker,bytes);
    }
    if(workers*(per_worker+sizeof(Frame)+128)+256>space.capacity()-space.used())return false;
    FrameMark lifetime(space);Frame *frames[8]{};
    for(unsigned w=0;w<workers;++w){auto *at=space.allocate(sizeof(Frame));frames[w]=::new(at)Frame(space.subframe(per_worker));}
    struct Batch {const IndependentProduct *jobs;const ProductProgramPlan *plans;const bool *native;
        Frame *const *frames;sbn3_team_scope *scope;};
    sbn3_team_scope sub{scope->team,scope->first,workers,false,scope->epoch};
    Batch batch{jobs,plans,native,frames,&sub};
    sbn3_team_for(&sub,0,count,1,SBN3_STATIC,[](void *ptr,uint64_t lo,uint64_t hi,unsigned rank){
        const auto &b=*static_cast<Batch*>(ptr);Frame &f=*b.frames[rank];
        // A disjoint one-worker child, never recursive dispatch on the
        // active parent scope. No task can recruit another batch worker.
        sbn3_team_scope serial{b.scope->team,b.scope->first+rank,1,false,b.scope->epoch};
        for(auto j=lo;j<hi;++j){FrameMark task(f);const auto &x=b.jobs[j];const size_t n=x.a.count+x.b.count;
            if(!x.a.count||!x.b.count){if(n)memset(x.out.data,0,n*8);continue;}
            if(b.native[j]){const auto &q=b.plans[j];
                auto tables=f.subframe(q.prepared_bytes+128,128);const auto p=product_program_prepare(q,tables);
                auto work=f.subframe(q.info.workspace_bytes,q.info.workspace_alignment);
                const bool staging=q.info.output_limbs>x.out.capacity;
                auto *out=staging?f.alloc<uint64_t>(q.info.output_limbs):x.out.data;
                product_program_execute(p,work,&serial,x.a,x.b,{out,q.info.output_limbs});
                if(staging)memcpy(x.out.data,out,n*8);
            }else u52::multiply(x.out.data,x.a.data,x.a.count,x.b.data,x.b.count,f);
        }
    },&batch);
    for(unsigned w=0;w<workers;++w)frames[w]->~Frame();
    return true;
}
}
