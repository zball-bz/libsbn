/* Experimental schedule: spare workers run the final primes' A/B forwards
 * concurrently. Both arrays are allocated by the parent and live to the join. */
struct FlatTail {Binding *binding;p::FlatCtx *context;unsigned q,count,origin;};
void flat_tail_action(void *arg,sbn3_team_scope *scope){
    auto &job=*static_cast<FlatTail *>(arg);auto &b=*job.binding;auto &c=*job.context;const auto &g=*c.pl;
    require_scope_leader(scope);
    if(job.count>1){const unsigned left=job.count/2,width=scope->width*left/job.count;
        FlatTail a{&b,&c,job.q,left,job.origin},y{&b,&c,job.q+left,job.count-left,job.origin};
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,width,flat_tail_action,&a,flat_tail_action,&y);return;}
    const unsigned split=scope->width/2,rank=scope->first-job.origin;
    Frame &af=*b.run.workers[rank],&bf=*b.run.workers[rank+split];FrameMark am(af),bm(bf);
    p::V *a=af.alloc<p::V>(g.M2+16),*y=bf.alloc<p::V>(g.M2+16);
    struct Forward {p::FlatCtx *c;p::V *data;bool second;unsigned q;};
    Forward ax{&c,a,false,job.q},yx{&c,y,true,job.q};
    auto forward=[](void *arg,sbn3_team_scope *){auto &f=*static_cast<Forward *>(arg);auto &c=*f.c;const auto &g=*c.pl;
        const size_t n=f.second?c.bn[0]:c.an[0];const auto *src=f.second?c.b[0]:c.a[0];const auto &dec=f.second?c.constants->b:c.constants->a;
        p::flat_decode(f.data,src,n,dec,g,f.q);p::flat_forward(f.data,g,p::flat_input_vectors(n,g.T),c.primes->P[f.q],c.primes->V[f.q]);p::flat_pad_frequency(f.data,g);};
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,split,forward,&ax,forward,&yx);
    const auto &prime=c.primes->P[job.q];const auto &pv=c.primes->V[job.q];
    for(size_t v=0;v<g.lbw;v+=8)p::conv8x8(y+v,a+v,p::tower8<false>(prime,v),pv);
    p::flat_inverse(y,g,prime,pv);for(size_t v=g.lbv;v<g.lbw;++v)y[v]=_mm512_setzero_si512();
    const auto pack=p::pk52_mk();for(size_t v=0;v<g.lbw;++v)p::st52(c.output[job.q]+v*SLOT,y[v],pack,pv);
    if(c.counts){c.counts[scope->first].row_forward+=2;++c.counts[scope->first].row_inverse;c.counts[scope->first].leaf_products+=g.lbw/8;}
}
