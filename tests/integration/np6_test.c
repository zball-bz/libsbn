#include "sbn3/product.h"
#include <assert.h>
#include "../oracle/oracle.h"
#include <stdio.h>
#include <string.h>

static size_t align_up(size_t n,size_t a){return (n+a-1)&~(a-1);}
static uint64_t seed=0xe1815a31a0bc003dULL;
static uint64_t rnd(void){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static sbn3_lease acquire(sbn3_arena *a,size_t off,size_t bytes){
    sbn3_error e={0};sbn3_lease l={0};
    sbn3_status status=sbn3_arena_prepare(a,off,bytes,&e);
    if(status!=SBN3_OK)fprintf(stderr,"prepare: %s code=%d sys=%d need=%zu have=%zu\n",e.where,status,e.system_error,e.need,e.have);
    assert(status==SBN3_OK);sbn3_arena_acquire(a,off,bytes,&l);return l;
}
static void run_case(size_t an,size_t bn,unsigned workers,unsigned group,int T,unsigned borrow){
    sbn3_product_spec spec={an,bn};sbn3_mul_options options={workers,group,T,borrow,0,0,0,6,1,1,0,0};
    sbn3_mul_plan plan;sbn3_mul_info info={0};
    assert(sbn3_mul_query(&spec,&options,&plan,&info)==SBN3_SUPPORTED);
    assert(info.np==6 && info.workers==workers && info.output_limbs==an+bn);
    sbn3_mul_plan refused;memset(&refused,0xa3,sizeof refused);options.workspace_budget=1;
    sbn3_mul_info need={0};assert(sbn3_mul_query(&spec,&options,&refused,&need)==SBN3_QUERY_CAPACITY);
    assert(need.workspace_bytes==info.workspace_bytes && refused.opaque[0]==0xa3a3a3a3a3a3a3a3ULL);
    options.workspace_budget=0;
    sbn3_arena_config ac={(size_t)64<<20,(size_t)8<<20};sbn3_arena *arena=NULL;sbn3_error e={0};
    assert(sbn3_arena_create(&ac,&arena,&e)==SBN3_OK);
    sbn3_lease control=acquire(arena,0,sbn3_team_storage_bytes());
    sbn3_team_config tc={0};tc.workers=workers;tc.pin_threads=1;tc.stack_offset=64u<<10;
    for(unsigned i=0;i<32;++i)tc.cpu_ids[i]=-1;
    sbn3_team *team=NULL;assert(sbn3_team_create(arena,&control,&tc,&team,&e)==SBN3_OK);
    size_t off=tc.stack_offset+sbn3_team_stack_virtual_bytes(workers);
    sbn3_lease tables=acquire(arena,off,info.table_bytes);off+=info.table_bytes;
    const uintptr_t base=(uintptr_t)control.data;
    off=align_up(base+off,info.workspace_alignment)-base;
    sbn3_lease work=acquire(arena,off,info.workspace_bytes);off=align_up(off+info.workspace_bytes+4096,4096);
    sbn3_lease a=acquire(arena,off,align_up((an+8)*8,64));off=align_up(off+a.bytes+4096,4096);
    sbn3_lease b=acquire(arena,off,align_up((bn+8)*8,64));off=align_up(off+b.bytes+4096,4096);
    sbn3_lease r=acquire(arena,off,align_up((an+bn+8)*8,64));
    sbn3_mul_binding *binding=NULL;sbn3_mul_bind(&plan,arena,&tables,&work,team,&binding);
    sbn3_arena_stats before={0};sbn3_arena_get_stats(arena,&before);
    uint64_t *ap=a.data,*bp=b.data,*rp=r.data;
    ref_int za,zb,zp,zr;ref_inits(za,zb,zp,zr,NULL);
    for(int pattern=0;pattern<4;++pattern){
        for(size_t i=0;i<an;++i)ap[i]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(1ULL<<((i*17)&63)):rnd();
        for(size_t i=0;i<bn;++i)bp[i]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(i&1?0:UINT64_MAX):rnd();
        memset(rp,0xa5,r.bytes);
        ref_import(za,an,-1,8,0,0,ap);ref_import(zb,bn,-1,8,0,0,bp);ref_mul(zp,za,zb);
        if(pattern&1)sbn3_mul_execute_ptrs(binding,ap,bp,rp);
        else sbn3_mul_execute(binding,(sbn3_const_limbs){ap,an},(sbn3_const_limbs){bp,bn},(sbn3_limbs){rp,an+bn});
        ref_import(zr,an+bn,-1,8,0,0,rp);
        if(ref_cmp(zr,zp))fprintf(stderr,"NP6 mismatch an=%zu bn=%zu W=%u T=%u pattern=%d\n",an,bn,workers,info.trunk_bits,pattern);
        assert(ref_cmp(zr,zp)==0);assert(rp[an+bn]==0xa5a5a5a5a5a5a5a5ULL);
        ref_import(zr,an,-1,8,0,0,ap);assert(ref_cmp(zr,za)==0);
        ref_import(zr,bn,-1,8,0,0,bp);assert(ref_cmp(zr,zb)==0);
    }
    ref_clears(za,zb,zp,zr,NULL);
    sbn3_mul_metrics metrics={0};sbn3_mul_get_metrics(binding,&metrics);
    assert(metrics.executions==4 && metrics.worker_peak_bytes<=info.per_worker_bytes &&
           metrics.table_used_bytes<=info.table_bytes && metrics.workspace_used_bytes<=info.workspace_bytes);
    sbn3_arena_stats after={0};sbn3_arena_get_stats(arena,&after);
    assert(after.prepare_calls==before.prepare_calls && after.trim_calls==before.trim_calls &&
           after.payload_resident_bytes==before.payload_resident_bytes);
    printf("NP6 LIN %zu x %zu W=%u T=%u g=%u borrow=%u: reference/modular oracle/preserve/guard OK; worker %zu/%zu, table %zu, work %zu\n",
           an,bn,workers,info.trunk_bits,group,info.borrow_output,metrics.worker_peak_bytes,info.per_worker_bytes,info.table_bytes,info.workspace_bytes);
    sbn3_mul_unbind(binding);
    sbn3_arena_release(arena,&a);sbn3_arena_release(arena,&b);sbn3_arena_release(arena,&r);
    sbn3_arena_release(arena,&tables);sbn3_arena_release(arena,&work);
    sbn3_team_destroy(team);sbn3_arena_release(arena,&control);
    sbn3_arena_get_stats(arena,&after);assert(!after.lease_references && !after.active_computations);
    sbn3_arena_destroy(arena);
}
int main(void){
    run_case(1,1,1,1,0,0);
    run_case(9,17,1,2,112,0);
    run_case(257,511,1,3,120,1);
    run_case(4097,1021,1,6,128,1);
    run_case(16384,21845,1,1,136,1);
    run_case(4097,1021,3,1,0,1);
    puts("NP6 query/bind/execute integration gates OK");
}
