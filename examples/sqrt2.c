#define _POSIX_C_SOURCE 200809L
#include "sbn3/newton.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static size_t align_up(size_t n,size_t a){return (n+a-1)&~(a-1);}
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static size_t number(const char *s){char *end=NULL;errno=0;unsigned long long n=strtoull(s,&end,10);if(errno||!*s||*end){fprintf(stderr,"invalid number: %s\n",s);exit(2);}return (size_t)n;}
static int fail(sbn3_error *e){fprintf(stderr,"resource error: %s, errno=%d, need=%zu, have=%zu\n",e->where,e->system_error,e->need,e->have);return 1;}
static uint64_t fingerprint(const uint64_t *p,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t j=0;j<n;++j)for(unsigned k=0;k<8;++k){h=(h^((p[j]>>(8*k))&255))*UINT64_C(1099511628211);}return h;}
static void usage(void){fputs("sqrt2 [--limbs N] [--workers W] [--route both|rsqrt|rational] [--budget-mib M] [--query] [--output FILE]\n",stderr);}
int main(int argc,char **argv){
    size_t n=16384,budget=0;unsigned requested_workers=16,mask=3;int query_only=0;const char *output_path=NULL;
    for(int j=1;j<argc;++j){
        if(!strcmp(argv[j],"--query")){query_only=1;continue;}
        if(!strcmp(argv[j],"--help")){usage();return 0;}
        if(j+1>=argc){usage();return 2;}const char *arg=argv[++j];
        if(!strcmp(argv[j-1],"--limbs"))n=number(arg);
        else if(!strcmp(argv[j-1],"--workers")){const size_t w=number(arg);if(w>32){usage();return 2;}requested_workers=(unsigned)w;}
        else if(!strcmp(argv[j-1],"--budget-mib")){const size_t m=number(arg);if(m>(SIZE_MAX>>20)){usage();return 2;}budget=m*(size_t)(1u<<20);}
        else if(!strcmp(argv[j-1],"--route")){if(!strcmp(arg,"both"))mask=3;else if(!strcmp(arg,"rsqrt"))mask=1;else if(!strcmp(arg,"rational"))mask=2;else{usage();return 2;}}
        else if(!strcmp(argv[j-1],"--output"))output_path=arg;
        else{usage();return 2;}
    }
    if(!n || n>((size_t)1<<28) || !requested_workers || requested_workers>32){usage();return 2;}
    const char *names[2]={"rsqrt","rational"};const sbn3_newton_kind kinds[2]={SBN3_SQRT2_RSQRT,SBN3_SQRT2_RATIONAL};
    sbn3_newton_plan plans[2];sbn3_newton_info info[2];size_t storage=0,alignment=128;unsigned workers=1;
    const sbn3_newton_options options={requested_workers,0,budget,1};
    for(unsigned k=0;k<2;++k)if(mask&(1u<<k)){
        sbn3_query_result rc=sbn3_newton_query(kinds[k],n,&options,&plans[k],&info[k]);
        if(rc!=SBN3_SUPPORTED){fprintf(stderr,"unsupported/capacity route=%s result=%u need=%zu\n",names[k],rc,info[k].storage_bytes);return 1;}
        if(info[k].storage_bytes>storage)storage=info[k].storage_bytes;if(info[k].storage_alignment>alignment)alignment=info[k].storage_alignment;
        if(info[k].workers>workers)workers=info[k].workers;
        printf("{\"kind\":\"plan\",\"route\":\"%s\",\"fractional_limbs\":%zu,\"workers\":%u,\"storage_bytes\":%zu,\"table_bytes\":%zu,\"spectrum_bytes\":%zu,\"product_workspace_bytes\":%zu,\"value_bytes\":%zu,\"shared_bytes\":%zu,\"setup_bytes\":%zu,\"products\":%u,\"spectra\":%u,\"lease_peak\":%u,\"plan_id\":\"%016" PRIx64 "\"}\n",
            names[k],n,info[k].workers,info[k].storage_bytes,info[k].table_bytes,info[k].spectrum_bytes,info[k].product_workspace_bytes,info[k].value_bytes,info[k].shared_bytes,info[k].setup_bytes,info[k].products,info[k].spectra,info[k].lease_peak,info[k].plan_id);
    }
    const size_t result_bytes=align_up((n+1)*8,64),control_bytes=sbn3_team_storage_bytes();
    // Conservative allowance for arena control and page rounding, not an
    // unreported claim that all admitted bytes are actual payload.
    const size_t admission=align_up(storage,4096)+2*align_up(result_bytes,4096)+align_up(control_bytes,4096)+sbn3_team_stack_resident_bytes(workers)+(1u<<20);
    printf("{\"kind\":\"admission\",\"required_lock_limit_bytes\":%zu,\"actual_workers\":%u}\n",admission,workers);fflush(stdout);
    if(budget && admission>budget){fprintf(stderr,"overall RAM budget: need at most %zu, have %zu\n",admission,budget);return 1;}
    if(query_only)return 0;
    sbn3_arena_config ac={storage+2*result_bytes+sbn3_team_stack_virtual_bytes(workers)+2*alignment+(16u<<20),admission};
    sbn3_arena *arena=NULL;sbn3_team *team=NULL;sbn3_error error={0};sbn3_lease control={0},results[2]={{0},{0}};
    if(sbn3_arena_create(&ac,&arena,&error)!=SBN3_OK)return fail(&error);
    if(sbn3_arena_prepare(arena,0,control_bytes,&error)!=SBN3_OK)return fail(&error);sbn3_arena_acquire(arena,0,control_bytes,&control);
    sbn3_team_config tc={0};tc.workers=workers;tc.pin_threads=1;tc.stack_offset=65536;for(unsigned j=0;j<32;++j)tc.cpu_ids[j]=-1;
    if(sbn3_team_create(arena,&control,&tc,&team,&error)!=SBN3_OK)return fail(&error);
    const uintptr_t base=(uintptr_t)control.data;size_t offset=align_up(base+tc.stack_offset+sbn3_team_stack_virtual_bytes(workers)+4096,alignment)-base;
    if(sbn3_arena_prepare(arena,offset,storage,&error)!=SBN3_OK)return fail(&error);
    size_t result_offset=align_up(offset+storage,64);
    if(sbn3_arena_prepare(arena,result_offset,2*result_bytes,&error)!=SBN3_OK)return fail(&error);
    for(unsigned k=0;k<2;++k)sbn3_arena_acquire(arena,result_offset+k*result_bytes,result_bytes,&results[k]);
    for(unsigned k=0;k<2;++k)if(mask&(1u<<k)){
        const uint64_t begin=now();sbn3_newton_binding *binding=NULL;sbn3_newton_bind(&plans[k],arena,offset,team,&binding);const uint64_t prepared=now();
        sbn3_newton_execute(binding,NULL,(sbn3_limbs){results[k].data,n+1});const uint64_t end=now();sbn3_newton_metrics metrics;sbn3_newton_get_metrics(binding,&metrics);
        const uint64_t *z=results[k].data;sbn3_arena_stats stats;sbn3_arena_get_stats(arena,&stats);
        printf("{\"kind\":\"run\",\"route\":\"%s\",\"fractional_limbs\":%zu,\"workers\":%u,\"bind_ns\":%" PRIu64 ",\"compute_ns\":%" PRIu64 ",\"verify_ns\":%" PRIu64 ",\"bind_execute_ns\":%" PRIu64 ",\"products_executed\":%u,\"spectra_computed\":%u,\"unit_corrections\":%u,\"locked_peak_bytes\":%zu,\"prefix\":\"%" PRIx64 ".%016" PRIx64 "\",\"fnv1a64\":\"%016" PRIx64 "\"}\n",
            names[k],n,info[k].workers,prepared-begin,metrics.compute_ns,metrics.verify_ns,end-begin,metrics.products_executed,metrics.spectra_computed,metrics.unit_corrections,stats.peak_resident_bytes,z[n],z[n-1],fingerprint(z,n+1));fflush(stdout);
        sbn3_newton_unbind(binding);
    }
    if(mask==3){const int equal=memcmp(results[0].data,results[1].data,(n+1)*8)==0;
        printf("{\"kind\":\"comparison\",\"equal\":%s,\"limbs_compared\":%zu}\n",equal?"true":"false",n+1);if(!equal)return 1;}
    if(output_path){FILE *f=fopen(output_path,"wb");if(!f){perror(output_path);return 1;}
        const unsigned char header[8]={'S','B','N','3','S','Q','2',0};const uint64_t size=n;const unsigned result=mask&1?0:1;
        const int ok=fwrite(header,1,8,f)==8 && fwrite(&size,8,1,f)==1 && fwrite(results[result].data,8,n+1,f)==n+1;
        const int closed=fclose(f);if(!ok||closed){fprintf(stderr,"output write failed\n");return 1;}}
    for(unsigned k=0;k<2;++k)sbn3_arena_release(arena,&results[k]);sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);return 0;
}
