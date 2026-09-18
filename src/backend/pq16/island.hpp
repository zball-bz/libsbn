#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "backend/u52/lane.hpp"
#include "runtime/team.hpp"
#include "backend/pq16/kernels.hpp"
namespace sbn::v3::pq16 {
using namespace sbn::v3::u52;
using scratch=::sbn::v3::Frame;
struct sbn_team {sbn3_team_scope *scope;};
constexpr int SBN_FOR_STATIC=0;
using RangeFn=void(*)(void *,uint64_t,uint64_t,int,scratch *);
using TaskFn=void(*)(void *,int,int,scratch *);
static void sbn_parallel_for(sbn_team *t,int nthr,uint64_t lo,uint64_t hi,uint64_t grain,int,RangeFn fn,void *arg){
    require(t && t->scope && t->scope->width==unsigned(nthr),SBN3_FATAL_TEAM,"pq16 scope");
    struct C {RangeFn fn;void *arg;unsigned first;} c{fn,arg,t->scope->first};
    auto call=[](void *v,uint64_t a,uint64_t b,unsigned rank){auto &c=*static_cast<C *>(v);c.fn(c.arg,a,b,int(c.first+rank),nullptr);};
    sbn3_team_for(t->scope,lo,hi,grain,SBN3_STATIC,call,&c);
}
static void sbn_run_tasks(sbn_team *t,int nthr,int n,TaskFn fn,void *arg){
    require(t && t->scope && t->scope->width==unsigned(nthr),SBN3_FATAL_TEAM,"pq16 task scope");
    struct C {TaskFn fn;void *arg;unsigned first;} c{fn,arg,t->scope->first};
    auto call=[](void *v,uint64_t a,uint64_t b,unsigned rank){auto &c=*static_cast<C *>(v);for(auto j=a;j<b;++j)c.fn(c.arg,int(j),int(c.first+rank),nullptr);};
    sbn3_team_for(t->scope,0,n,1,SBN3_STATIC,call,&c);
}
}
#define SCRATCH(s) ::sbn::v3::FrameMark pq16_mark(*(s))
#define SALLOC(s,T,n) (s)->alloc<T>(n)
#include "backend/pq16/engine.hpp"
#undef SCRATCH
#undef SALLOC
