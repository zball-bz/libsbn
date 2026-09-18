#define _POSIX_C_SOURCE 200809L
#include "sbn3/bitwindow.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint64_t tick(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;
}
int main(int argc,char **argv) {
    uint64_t offset=0; unsigned workers=16,window=192; int catalan=0;
    for(int j=1;j<argc;j+=2) {
        if(!strcmp(argv[j],"--help")) {
            puts("bbp --constant pi|catalan --offset BINARY_BITS --workers W --window 192|384"); return 0;
        }
        if(j+1==argc)return 2;
        if(!strcmp(argv[j],"--constant")) {
            if(!strcmp(argv[j+1],"catalan"))catalan=1;
            else if(!strcmp(argv[j+1],"pi"))catalan=0;
            else return 2;
            continue;
        }
        char *end; errno=0; const uint64_t x=strtoull(argv[j+1],&end,10);
        if(errno || !*argv[j+1] || *end || argv[j+1][0]=='-')return 2;
        if(!strcmp(argv[j],"--offset"))offset=x;
        else if(!strcmp(argv[j],"--workers") && x>=1 && x<=32)workers=(unsigned)x;
        else if(!strcmp(argv[j],"--window") && (x==192 || x==384))window=(unsigned)x;
        else return 2;
    }
    if(catalan?!sbn3_catalan_bbp_supported(offset,window):!sbn3_pi_bbp_supported(offset,window))return 2;
    sbn3_arena_config ac={(size_t)1<<28,(size_t)64<<20};
    sbn3_arena *arena=NULL; sbn3_team *team=NULL; sbn3_lease control={0}; sbn3_error error={0};
    if(sbn3_arena_create(&ac,&arena,&error)!=SBN3_OK)goto failed;
    if(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)!=SBN3_OK)goto failed;
    sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
    sbn3_team_config tc={0};tc.workers=workers;tc.pin_threads=1;tc.stack_offset=65536;
    for(unsigned j=0;j<32;++j)tc.cpu_ids[j]=-1;
    if(sbn3_team_create(arena,&control,&tc,&team,&error)!=SBN3_OK)goto failed;
    sbn3_bbp_result result={0};const uint64_t start=tick();
    if(catalan)sbn3_catalan_bbp(team,offset,window,&result);
    else sbn3_pi_bbp(team,offset,window,&result);
    const uint64_t ns=tick()-start;
    printf("{\"kind\":\"bbp\",\"constant\":\"%s\",\"bit_offset\":%" PRIu64 ",\"window_bits\":%u,\"workers\":%u,"
           "\"terms\":%" PRIu64 ",\"error_ulps\":%" PRIu64 ",\"stable\":%u,\"bits\":\"%016" PRIx64 "%016" PRIx64 "\","
           "\"verify_ns\":%" PRIu64 "}\n",catalan?"catalan":"pi",offset,window,workers,result.terms,result.error_ulps,result.stable,
           result.bits[0],result.bits[1],ns);
    sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);
    return result.stable?0:1;
failed:
    fprintf(stderr,"BBP setup error: %s errno=%d need=%zu have=%zu\n",error.where,error.system_error,error.need,error.have);
    if(team)sbn3_team_destroy(team);
    if(control.bytes)sbn3_arena_release(arena,&control);
    if(arena)sbn3_arena_destroy(arena);
    return 1;
}
