#include "product_support.hpp"
#include <sys/mman.h>
#include "runtime/scratch.hpp"
#include "backend/pq16/kernels.hpp"
using namespace sbn::v3;
extern "C" void test_pq16_pack();
extern "C" void test_pq16_ct();
extern "C" void test_pq16_variable();
extern "C" void test_pq16_rac();
extern "C" void test_pq16_balanced();
struct PQTask {const pq16::Tables *table;Frame *work;uint64_t *r;const uint64_t *a,*b;size_t an,bn;};
static void pqtask(void *arg,sbn3_team_scope *scope){auto &t=*static_cast<PQTask *>(arg);pq16::multiply(t.r,t.a,t.an,t.b,t.bn,*t.table,*t.work,scope);}
static void one(size_t an,size_t bn,unsigned w,unsigned minimum=128){
    const auto shape=pq16::execution_shape(pq16::query(an,bn,minimum),w);assert(shape.nfull);Fixture f(w);
    auto tl=f.allocate(pq16::table_bytes(shape),128),wl=f.allocate(pq16::scratch_bytes(shape,an,bn,w),128);
    Frame tables(*f.arena,tl),work(*f.arena,wl);allocation_watch_start();auto *plan=pq16::prepare(tables,shape);assert(!allocation_watch_stop());
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ));
    auto *a=f.guarded(an),*b=f.guarded(bn),*r=f.guarded(an+bn);PQTask task{plan,&work,r,a,b,an,bn};
    for(unsigned k=0;k<4;++k){for(size_t j=0;j<an;++j)a[j]=k==0?UINT64_MAX:random_word();for(size_t j=0;j<bn;++j)b[j]=k==0?UINT64_MAX:k==1?0:random_word();memset(r,0xaa,(an+bn)*8);
        // Seed free arena bytes, then restore poison: allocation must still
        // make each live slice accessible. Dead NaNs must not enter the FFT.
        SBN3_FRAME_UNPOISON(work.data(),work.capacity());
        memset(work.data(),0xff,work.capacity());
        SBN3_FRAME_POISON(work.data(),work.capacity());
        allocation_watch_start();sbn3_team_run(f.team,pqtask,&task);assert(!allocation_watch_stop());assert(!work.used());verify_product(a,an,b,bn,r);}
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ|PROT_WRITE));
    assert(tables.peak()<=pq16::table_bytes(shape)&&work.peak()<=pq16::scratch_bytes(shape,an,bn,w));
    printf("pq16 %zu x %zu W%u N%u M%u centered%d recipe%u GMP/guard/Frame/read-only-plan PASS\n",an,bn,w,shape.nfull,shape.radix,shape.centered,unsigned(shape.recipe));
}
int main(){
    test_pq16_pack();
    test_pq16_ct();
    test_pq16_variable();
    test_pq16_rac();
    test_pq16_balanced();
    for(size_t n:{1u,2u,7u,25u,26u,32u,64u,128u,129u,256u,257u,448u,470u,511u,512u,513u,768u,1024u,4096u,8192u,16384u,32768u,32769u,35734u,49152u,65536u,90000u})one(n,n,1);
    for(size_t n:{129u,1024u,8192u,32768u}){one(n,n/3,3);one(n,n,4);}
    one(35734,35734,2);one(35734,35734,3);one(65537,12001,2);one(65537,12001,4);one(90000,90000,4);
    // Largest pow2 seed grid, with legal unbalanced centered inputs.
    one(163840,98304,1);one(200000,60000,4);
    assert(!pq16::query(131072,131072).nfull);assert(!pq16::query(130560,130560).nfull);assert(!pq16::query(1u<<20,1u<<20).nfull);assert(!pq16::query(0,1).nfull);
    for(unsigned minimum:{128u,256u})for(size_t n:{1u,2u,7u,25u,26u,31u,32u,33u,63u,64u,65u,96u,128u})one(n,n,1,minimum);
    puts("pq16 numerical band guards PASS");
}
