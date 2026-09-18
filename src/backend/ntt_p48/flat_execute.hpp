/* Included in an instance's private namespace after Binding and emit_range. */
#if CR_FLAT_TAIL_PAIR
#include "flat_tail.hpp"
#endif
void flat_prepare_bound(Binding &b,Spectrum &s,sbn3_team_scope *scope,sbn3_const_limbs a,p::PassCounts *counts=nullptr){
    const auto &i=b.plan.execution.info;const auto &g=b.plan.arithmetic.transform;
    sbn3_team_scope sub{scope->team,scope->first,i.workers,false,scope->epoch};
    p::FlatCtx c{};c.pl=&g;c.counts=counts;c.primes=&s.primes;c.constants=b.run.flat_constants;c.a[0]=a.data;c.an[0]=a.count;
    for(unsigned q=0;q<PN;++q)c.output[q]=s.planes[q];
    kernel_for(&sub,0,PN,1,SBN3_STATIC,p::flat_prepare_fn,&c,b.run.workers);
}
void execute_flat(Binding &b,sbn3_team_scope *scope,const sbn3_product_inputs &in,sbn3_limbs out,bool counters,
                  uint8_t *const *program_cache=nullptr){
    require_scope_leader(scope);const auto &i=b.plan.execution.info;const auto &r=b.plan.recipe;const auto &g=b.plan.arithmetic.transform;
    require(scope->team==b.run.team && scope->width>=i.workers,SBN3_FATAL_TEAM,"flat scope width");
    sbn3_team_scope sub{scope->team,scope->first,i.workers,false,scope->epoch};scope=&sub;
    uint32_t expected=0;require(__atomic_compare_exchange_n(&b.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent flat binding");
    for(auto &v:b.counts)v={};const uint64_t t0=tick_ns();memset(b.run.tail,0,b.plan.execution.tail_bytes);
    p::FlatCtx c{};c.pl=&g;c.primes=b.run.primes;c.constants=b.run.flat_constants;c.counts=counters?b.counts:nullptr;
#ifdef CR_FLAT_PROFILE
    memset(b.flat_phase,0,sizeof b.flat_phase);c.phase=b.flat_phase;
#endif
    c.a[0]=in.a.data;c.a[1]=in.a1.data;c.b[0]=in.b.data;c.b[1]=in.b1.data;
    c.an[0]=in.a.count;c.an[1]=in.a1.count;c.bn[0]=in.b.count;c.bn[1]=in.b1.count;
    for(unsigned term=0;term<2;++term){c.k[term]=r.k[term];c.rec[term]=r.kr[term];
        if(b.run.cached[term])for(unsigned q=0;q<PN;++q)c.cached[term][q]=reinterpret_cast<const p::V *>(spectrum(b.run.cached[term]).planes[q]);}
    if(program_cache)for(unsigned q=0;q<PN;++q)c.cached[0][q]=reinterpret_cast<const p::V *>(program_cache[q]);
    for(unsigned q=0;q<PN;++q)c.output[q]=b.run.b_planes[q];
    KernelForFn fn=nullptr;
    if(program_cache){
        require(r.kind==SBN3_PRODUCT_MUL && !r.cached_mask && !r.scaled,SBN3_FATAL_MATH,"private packed flat pair");
        fn=p::flat_program_apply48;
    } else switch(r.kind){
        case SBN3_PRODUCT_MUL:fn=r.scaled?p::flat_product_fn<0,true>:p::flat_product_fn<0,false>;break;
        case SBN3_PRODUCT_SQR:fn=r.scaled?p::flat_product_fn<1,true>:p::flat_product_fn<1,false>;break;
        case SBN3_PRODUCT_TMP:fn=r.scaled?p::flat_product_fn<2,true>:p::flat_product_fn<2,false>;break;
        case SBN3_PRODUCT_MAC2:fn=r.scaled?p::flat_product_fn<3,true>:p::flat_product_fn<3,false>;break;
        default:fatal(SBN3_FATAL_ARGUMENT,"flat product kind");
    }
#if CR_FLAT_TAIL_PAIR
    const unsigned tail=PN%i.workers,prefix=PN-tail;
    const bool tail_enabled=CR_FLAT_TAIL_PAIR==1 || (PN==10 && g.M2==16384 && (i.workers==4 || i.workers==8));
    if(tail_enabled && !program_cache && r.kind==SBN3_PRODUCT_MUL && !r.cached_mask && !r.scaled && tail && i.workers>=2*tail && g.M2>=8192){
        if(prefix)kernel_for(scope,0,prefix,1,SBN3_STATIC,fn,&c,b.run.workers);
        FlatTail job{&b,&c,prefix,tail,scope->first};flat_tail_action(&job,scope);
    }else
#endif
#if CR_FLAT_PRIME_SPREAD
    if(i.workers==16 && (PN==8 || PN==10) && g.M2>=16384){
        struct Launch {KernelForFn fn;p::FlatCtx *context;unsigned width;} launch{fn,&c,i.workers};
        auto spread=[](void *arg,uint64_t lo,uint64_t hi,int worker,Frame *frame){auto &l=*static_cast<Launch *>(arg);
            for(auto rank=lo;rank<hi;++rank){const uint64_t a=rank*PN/l.width,z=(rank+1)*PN/l.width;if(a<z)l.fn(l.context,a,z,worker,frame);}};
        kernel_for(scope,0,i.workers,1,SBN3_STATIC,spread,&launch,b.run.workers);
    }else
#endif
    {kernel_for(scope,0,PN,1,SBN3_STATIC,fn,&c,b.run.workers);}
    const uint64_t t1=tick_ns();
    p::Plan eg=g;const size_t low_limit=r.kind!=SBN3_PRODUCT_TMP?r.window.width_bits/64:0;
    if(low_limit){eg.outcap=low_limit;eg.ntp=((low_limit+g.OL-1)/g.OL)*g.OT;eg.nl=eg.ntp*g.T/64;}
    p::EmitCtx e{};e.pl=&eg;e.PS=b.run.primes;e.G=&b.garner;e.rp=out.data;e.tail=b.run.tail;e.spill=b.run.spill;
    e.nch=(eg.ntp+p::OCH-1)/p::OCH;e.EK=&b.emit_tables.EK;e.RG=&b.emit_tables.RG;e.reversed=r.kind==SBN3_PRODUCT_TMP;e.vlim=g.Zw;
    for(unsigned q=0;q<PN;++q)e.plane[q]=b.run.b_planes[q];p::EmitMap map;e.map=&map;const auto tasks=p::emit_tasks(map,eg,e.nch,e.reversed);
    kernel_for(scope,0,tasks,1,SBN3_STATIC,emit_range,&e,b.run.workers);p::emit_join(e);
    if(g.ring_rn&&!r.window.width_bits)p::fold_cyc(out.data,g.ring_rn,b.run.tail,g.nl-g.outcap);
    const uint64_t t2=tick_ns();
    b.last_stage_ns[0]=t1-t0;b.last_stage_ns[1]=b.last_stage_ns[2]=0;b.last_stage_ns[3]=t2-t1;++b.executions;__atomic_store_n(&b.active,0,__ATOMIC_RELEASE);
}
