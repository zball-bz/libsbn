// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/integer.hpp"
#include "sbn3/borrow.hpp"
#include "../oracle/oracle.h"
#include <cassert>
#include <climits>
#include <cstdio>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;using U=uint64_t;
static U rng=9812756;static U random_word(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static void import(ref_int v,const integer&a){ref_import(v,a.size(),-1,8,0,0,a.data());if(a.negative())ref_neg(v,v);}
static void matches(const integer&a,ref_int b){ref_int t;ref_init(t);import(t,a);assert(!ref_cmp(t,b));ref_clear(t);assert(!a.zero()?!a.negative()||a.size():!a.negative());}
template<class T>concept HasRvalueView=requires(T&& x){std::move(x).view();};
static_assert(!HasRvalueView<integer>);
static_assert(std::same_as<decltype(integer{1}+integer{2}),integer>);
static_assert(sizeof(integer)==64);
template<class Fn>static void rejects(Fn fn){const auto pid=fork();assert(pid>=0);if(!pid){const auto fd=open("/dev/null",O_WRONLY);if(fd>=0){dup2(fd,2);close(fd);}fn();_exit(0);}int status;assert(waitpid(pid,&status,0)==pid);assert(WIFSIGNALED(status)&&WTERMSIG(status)==SIGABRT);}
int main(){
    // A bounded mask can shrink a large value. Its capacity requirement must
    // follow the result shape, not spuriously grow the larger operand.
    {struct Resource final:sbn::memory_resource {
        unsigned allocations=0;sbn::resource_error try_allocate(size_t n,size_t a,sbn::allocation&out)noexcept override{++allocations;return sbn::default_resource().try_allocate(n,a,out);}
        void deallocate(sbn::allocation a)noexcept override{sbn::default_resource().deallocate(a);}
        sbn::resource_statistics statistics()const noexcept override{return {};}
    } resource;
    std::vector<U> words(1024,UINT64_MAX);integer x{sbn::integer_view{words},resource};const auto before=resource.allocations;
    x+=0;x-=0;x|=0;x^=0;x&=255;assert(x==255&&resource.allocations==before);
    x.assign(sbn::integer_view{words});const auto again=resource.allocations;x-=x;assert(x.zero()&&resource.allocations==again);}
    rejects([]{integer x=0;x<<=-1;});rejects([]{integer x=0;x>>=-1;});
    rejects([]{integer x=0;(void)x.test_bit(-1);});rejects([]{integer x=0;x.clear_bit(-1);});
    rejects([]{integer x=0;(void)(x<<(__uint128_t(1)<<64));});rejects([]{integer x=0;(void)x.scan_bit(__uint128_t(1)<<64,true);});
    rejects([]{integer x=0;(void)x.count_ones(0,-1);});
    rejects([]{integer x=7;auto read=x.borrow();x+=1;});
    rejects([]{integer x=7;auto read=x.borrow();auto moved=std::move(x);(void)moved;});
    {integer x=7;{auto read=x.borrow();auto second=read;auto moved=std::move(read);assert(sbn::integer(second.view())==7&&sbn::integer(moved.view())==7);}x+=1;assert(x==8);}
    {integer x=-1;assert(x++==-1&&x==0);assert(++x==1&&x--==1&&--x==-1);assert(+x==x);sbn::fixed_integer<2> fixed=__uint128_t(1)<<127;assert(fixed.view().size()==2);fixed=-7;assert(fixed.view().negative());}
    for(int a=-64;a<=64;++a)for(int b=-64;b<=64;++b){allocation_watch_start();integer x=a,y=b;auto add=x+y,sub=x-y,aa=x&y,oo=x|y,xx=x^y,nn=~x;
        assert(add.try_to<int>()==a+b&&sub.try_to<int>()==a-b&&aa.try_to<int>()==(a&b)&&oo.try_to<int>()==(a|b)&&xx.try_to<int>()==(a^b)&&nn.try_to<int>()==~a);
        assert((x==y)==(a==b)&&((x<=>y)<0)==(a<b));x+=y;x-=y;x^=y;x^=y;assert(x.try_to<int>()==a);assert(!allocation_watch_stop());}
    assert(integer{INT64_MIN}.try_to<int64_t>()==INT64_MIN);assert(integer{UINT64_MAX}.try_to<uint64_t>()==UINT64_MAX);
    assert(!integer{-1}.try_to<uint64_t>()&&!integer{UINT64_MAX}.try_to<int64_t>()&&!integer{256}.try_to<uint8_t>());
    const __uint128_t max=~__uint128_t(0);integer a=max;assert(a.try_to<__uint128_t>()==max);integer b=a+1;assert(b.size()==3&&b.data()[2]==1&&!b.data()[0]&&!b.data()[1]);
    assert((~a)==-b);assert((a|(-a)).try_to<int>()==-1); // max128 | -max128 = -1
    auto copy=b;copy+=1;assert(copy!=b);auto moved=std::move(copy);assert(copy.zero()&&moved==b+1);swap(moved,a);assert(a==b+1&&moved.try_to<__uint128_t>()==max);
    integer zero=0;zero<<=SIZE_MAX;assert(zero.zero());assert((integer{-3}>>1)==-2&&sbn::trunc_shift(integer{-3},1)==-1);
    for(size_t n:{3u,4u,8u,31u,64u,257u,1025u}){
        std::vector<U> av(n),bv(n/2+1);for(auto&v:av)v=random_word();for(auto&v:bv)v=random_word();
        for(unsigned sign=0;sign<4;++sign){integer x{sbn::integer_view{av,bool(sign&1)}},y{sbn::integer_view{bv,bool(sign&2)}};ref_int A,B,R;ref_inits(A,B,R,nullptr);import(A,x);import(B,y);
            ref_add(R,A,B);matches(x+y,R);ref_sub(R,A,B);matches(x-y,R);
            auto q=x;q+=y;q-=y;assert(q==x);auto*alias=&q;q=*alias;assert(q==x);q=std::move(*alias);assert(q==x);
            for(size_t shift:{0u,1u,63u,64u,127u,257u}){ref_mul_2exp(R,A,shift);matches(x<<shift,R);ref_fdiv_q_2exp(R,A,shift);matches(x>>shift,R);ref_tdiv_q_2exp(R,A,shift);matches(sbn::trunc_shift(x,shift),R);}
            assert((x&y)==(y&x)&&(x|y)==(y|x)&&(x^y)==(y^x));assert((x^x)==0&&((~x)+x)==-1);assert(((x&y)|(x^y))==(x|y));
            integer dst;dst.reserve(std::max(x.size(),y.size())+1);
            allocation_watch_start();sbn::and_into(dst.ref(),x.view(),y.view());assert(!allocation_watch_stop());assert(dst==(x&y));
            allocation_watch_start();sbn::or_into(dst.ref(),x.view(),y.view());assert(!allocation_watch_stop());assert(dst==(x|y));
            allocation_watch_start();sbn::xor_into(dst.ref(),x.view(),y.view());assert(!allocation_watch_stop());assert(dst==(x^y));
            auto aliased=x;aliased.reserve(std::max(x.size(),y.size())+1);allocation_watch_start();sbn::xor_into(aliased.ref(),aliased.view(),y.view());assert(!allocation_watch_stop());assert(aliased==(x^y));
            auto edit=x;edit.set_bit(65);assert(edit.test_bit(65));edit.clear_bit(65);assert(!edit.test_bit(65));edit.flip_bit(65);assert(edit.test_bit(65));
            ref_clears(A,B,R,nullptr);
        }
    }
    // Frame-based product lowering keeps the existing scalar/u52/FFT/NTT
    // selection and square recipes. Re-query/prepare remains in strict calls.
    for(std::size_t n:{1u,2u,6u,7u,31u,161u,513u,1025u,4097u,16385u}){
        std::vector<U> av(n),bv(n/2+1);for(auto&v:av)v=random_word();for(auto&v:bv)v=random_word();
        integer x{sbn::integer_view{av,true}},y{sbn::integer_view{bv,false}};
        auto p=x*y;assert(p.negative()&&ref_product_equal(x.data(),x.size(),y.data(),y.size(),p.data(),p.size()));
        auto sq=sbn::square(x);assert(!sq.negative()&&ref_product_equal(x.data(),x.size(),x.data(),x.size(),sq.data(),sq.size()));
        integer out;out.reserve(x.size()+y.size());const auto need=sbn::multiply_requirements(out.ref(),x.view(),y.view());assert(need.supported());
        sbn::workspace work;assert(!work.try_reserve(need.scratch_bytes,need.scratch_alignment));
        allocation_watch_start();sbn::mul_into(out.ref(),x.view(),y.view(),work.on());assert(!allocation_watch_stop());assert(out==p);
        integer alias=x;alias.reserve(2*x.size());const auto bytes=sbn::square_requirements(alias.ref(),alias.view());assert(bytes.supported());
        work.reserve(bytes.scratch_bytes,bytes.scratch_alignment);
        allocation_watch_start();sbn::square_into(alias.ref(),alias.view(),work.on());assert(!allocation_watch_stop());assert(alias==sq);
        if(n<=513){auto self=x;self*=self;assert(self==sq);}
    }
    // Fixed external storage is an explicit resource boundary; failure must
    // preserve the previous value and allocation. Free order coalesces holes.
    alignas(128) std::byte backing[65536];
    {sbn::buffer_resource resource(backing);std::vector<integer> values;values.reserve(64);
        for(unsigned i=0;i<64;++i){integer v(resource);v=1;v<<=128+64*i;values.push_back(std::move(v));}
        for(unsigned i=0;i<64;i+=2)values[i].reset();
        for(unsigned i=0;i<64;i+=2){values[i]=1;values[i]<<=192+32*i;}
        integer v(resource);v=71;const auto before=v;auto err=v.try_reserve(1u<<20);assert(err&&v==before);
        integer other=123;other<<=512;const auto want=other;v=std::move(other);assert(v==want&&other.zero()&&&v.resource()==&resource);
        integer four=4,five=5;allocation_watch_start();sbn::fixed_integer<8> out;sbn::add_into(out.ref(),four.view(),five.view());assert(!allocation_watch_stop());
    }
    // A pool can serve many owning values without spending arena lease IDs.
    {std::vector<integer> values;values.reserve(6000);for(unsigned i=0;i<6000;++i){integer x=i+1;x<<=128;values.push_back(std::move(x));}
        for(unsigned i=0;i<6000;++i)assert((values[i]>>128).try_to<unsigned>()==i+1);}
    {auto worker=[](){for(unsigned i=0;i<500;++i){integer x=i+1;x<<=256;auto y=x+7;assert((y-x)==7);}};std::thread x(worker),y(worker);x.join();y.join();}
    // Leave long-lived objects a/b/moved alive; only free slabs are reclaimable.
    auto &res=sbn::default_resource();auto live=res.statistics().live_allocations;assert(!res.release_unused());assert(res.statistics().live_allocations==live);
    puts("C++ integer values/copy/move/resources/SBO/arithmetic/bits/shifts and allocation-free into PASS");
}
