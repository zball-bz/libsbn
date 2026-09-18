#pragma once
#include "product/cohort_pair_tuning.hpp"
#include "product/program.hpp"
#include <algorithm>
namespace sbn::v3::cohort_pair {
// Observed timing ranges filter noisy alternatives; they are not correctness
// bounds or a guarantee about interpolated performance.
struct Cost {double middle=0,low=0,high=0;};
inline bool price(const Group &g,double x,Cost &out){
    const Point *begin=points.data()+g.first,*end=begin+g.count;
    if(x<begin->x||x>(end-1)->x)return false;
    const Point *lo=begin,*hi=begin;
    while(hi+1<end && hi->x<x){lo=hi;++hi;}
    const double f=hi==lo?0:(x-lo->x)/(hi->x-lo->x);
    auto interpolate=[&](double a,double b){return a+f*(b-a);};
    out={};for(unsigned j=0;j<4;++j)out.middle+=interpolate(lo->phase[j],hi->phase[j]);
    out.low=interpolate(lo->low,hi->low);out.high=interpolate(lo->high,hi->high);return true;
}
inline bool price(const ProductProgramPlan &p,Cost &out){
    if(!p.info.np||p.info.M2<8)return false;const auto k=key(p.info);
    for(const auto &g:groups)if(g.key==k&&price(g,double(p.an+p.bn),out))return true;
    return false;
}
inline void select(ProductProgramPlan &p,unsigned cohort){
    if(cohort!=32 || !p.pair_workspace_bytes || !p.an || !p.bn ||
       std::min(p.an,p.bn)*32<std::max(p.an,p.bn)*31)return;
    Cost best{};if(!price(p,best))return;const auto reference=best;const auto original=p;
    for(const auto &g:groups){
        if(g.seed.workers!=p.info.workers)continue;
        // All alternatives must beat the same original observed range;
        // choose the least estimated cost among those admitted alternatives.
        // Comparing with the last accepted candidate would depend on table order.
        Cost estimate{};if(!price(g,double(p.an+p.bn),estimate) || estimate.high>=reference.low || estimate.middle>=best.middle)continue;
        ProductProgramPlan candidate{};
        if(product_program_replay(p.an,p.bn,g.seed,g.seed.prime_count,candidate)!=SBN3_SUPPORTED ||
           !candidate.pair_workspace_bytes || candidate.pair_workspace_bytes>original.pair_workspace_bytes || key(candidate.info)!=g.key)continue;
        p=candidate;best=estimate;
    }
}
}
