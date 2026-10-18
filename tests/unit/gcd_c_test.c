#include "sbn3/gcd.h"
#include <assert.h>
int main(void){
    assert(sbn3_gcd_u64(48,18)==6);
    uint64_t a[2]={0,3},b[2]={0,5},out[2];
    assert(sbn3_gcd_small_scratch_bytes(2,2)==0);
    assert(sbn3_gcd_small(out,a,2,b,2,0)==2&&out[0]==0&&out[1]==1);
    uint64_t x[5]={0,8,35,91,48},y[5]={7,12,14,13,18},g[5];
    sbn3_gcd_u64_batch(g,x,y,5);
    assert(g[0]==7&&g[1]==4&&g[2]==7&&g[3]==13&&g[4]==6);
    sbn3_gcd_info info;
    assert(sbn3_exgcd_query(1,1,3,0,&info)==SBN3_SUPPORTED&&info.scratch_bytes==0);
    uint64_t av=30,bv=21,gv=0,xv=0,yv=0;
    sbn3_int gg={&gv,1,0,0},xx={&xv,1,0,0},yy={&yv,1,0,0};
    sbn3_int_view aa={&av,1,1},bb={&bv,1,0};
    sbn3_int_exgcd(&gg,&xx,&yy,aa,bb,0,0,0);
    assert(gv==3&&gg.size==1&&!gg.negative);
    int64_t sx=xx.negative?-(int64_t)xv:(int64_t)xv,sy=yy.negative?-(int64_t)yv:(int64_t)yv;
    assert(-30*sx+21*sy==3);
    assert(sbn3_gcd(&gv,&av,1,&bv,1,0,0,0)==1&&gv==3);
    assert(sbn3_exgcd_query(1,1,4,0,&info)==SBN3_UNSUPPORTED);
    return 0;
}
