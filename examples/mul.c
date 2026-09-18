/* Build: clang -std=c11 -O2 -Iinclude examples/mul.c build/native/libsbn_v3.a
 *        -pthread -lm -o build/native/example_mul
 * No SIMD compiler flags, C++ compiler or GMP dependency are needed. */
#include "sbn3/mul.h"
#include <stdio.h>
#include <stdlib.h>

static size_t round_up(size_t n,size_t alignment){return (n+alignment-1)&~(alignment-1);}
static sbn3_lease prepare(sbn3_arena *a,size_t offset,size_t bytes){
    sbn3_error e={0};sbn3_lease out={0};
    if(sbn3_arena_prepare(a,offset,bytes,&e)!=SBN3_OK){
        fprintf(stderr,"resource preparation: %s (OS error %d)\n",e.where,e.system_error);exit(1);
    }
    sbn3_arena_acquire(a,offset,bytes,&out);return out;
}
int main(void){
    sbn3_product_spec product={1,1};sbn3_mul_options options={0};options.workers=1;
    sbn3_mul_plan plan; sbn3_mul_info info;
    if(sbn3_mul_query(&product,&options,&plan,&info)!=SBN3_SUPPORTED)return 1;
    sbn3_arena_config config={(size_t)16<<20,(size_t)8<<20};sbn3_error error={0};sbn3_arena *arena=NULL;
    if(sbn3_arena_create(&config,&arena,&error)!=SBN3_OK){fprintf(stderr,"%s\n",error.where);return 1;}
    sbn3_lease control=prepare(arena,0,sbn3_team_storage_bytes());
    sbn3_team_config tc={0};tc.workers=1;tc.pin_threads=1;tc.cpu_ids[0]=-1;tc.stack_offset=65536;
    sbn3_team *team=NULL;
    if(sbn3_team_create(arena,&control,&tc,&team,&error)!=SBN3_OK){fprintf(stderr,"%s\n",error.where);return 1;}
    size_t offset=65536;sbn3_lease tables=prepare(arena,offset,info.table_bytes);
    const uintptr_t base=(uintptr_t)control.data;
    offset=round_up(base+offset+info.table_bytes,info.workspace_alignment)-base;
    sbn3_lease work=prepare(arena,offset,info.workspace_bytes);
    sbn3_mul_binding *bound=NULL;sbn3_mul_bind(&plan,arena,&tables,&work,team,&bound);
    _Alignas(64) uint64_t a[1]={UINT64_MAX},b[1]={UINT64_MAX},r[2]={0,0};
    sbn3_mul_execute(bound,(sbn3_const_limbs){a,1},(sbn3_const_limbs){b,1},(sbn3_limbs){r,2});
    printf("%016llx%016llx\n",(unsigned long long)r[1],(unsigned long long)r[0]);
    const int correct=r[0]==1 && r[1]==UINT64_MAX-1;
    sbn3_mul_unbind(bound);sbn3_arena_release(arena,&tables);sbn3_arena_release(arena,&work);
    sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);
    return correct?0:1;
}
