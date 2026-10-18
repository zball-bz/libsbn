/* Original libsbn tests; LGPL-3.0-or-later. */
#include "sbn3/jacobi.h"
#include <assert.h>
int main(void){
    sbn3_jacobi_info i;
    assert(sbn3_jacobi_query(1,1,0,&i)==SBN3_SUPPORTED&&i.scratch_bytes==0);
    assert(sbn3_jacobi_query(1,0,0,&i)==SBN3_UNSUPPORTED);
    assert(sbn3_jacobi_scratch_bytes(1,0,1)==SIZE_MAX);
    assert(sbn3_jacobi_scratch_bytes(1,1,33)==SIZE_MAX);
    assert(sbn3_jacobi_scratch_bytes((1u<<20)+1,1,1)==SIZE_MAX);
    uint64_t a=3,n=7;
    assert(sbn3_jacobi_u64(a,n)==-1);
    assert(sbn3_jacobi(&a,1,&n,1,0,0,0)==-1);
    sbn3_int_view av={&a,1,1},nv={&n,1,0};
    assert(sbn3_int_jacobi(av,nv,0,0,0)==1);
    assert(sbn3_jacobi_u64(0,1)==1);
    assert(sbn3_jacobi_u64(7,21)==0);
    return 0;
}
