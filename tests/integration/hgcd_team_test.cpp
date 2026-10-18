// Original libsbn tests; LGPL-3.0-or-later.
#include "gcd/internal.hpp"
#include "backend/u52/kernels.hpp"
#include <vector>
#include <cassert>
#include <cstdio>
#include <algorithm>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using U=uint64_t;
static void matrix_gate(sbn3_team *team){
    namespace h=sbn::v3::gcd;
    for(size_t n:{127u,128u,129u,256u,513u,1025u})for(unsigned kind=0;kind<3;++kind){
        struct Case {std::vector<U>a[4],b[4],out[4];size_t used[4]{};void *work;size_t bytes,cap,n;};Case c;c.n=n;
        U state=87625613;auto random=[&](){state^=state<<13;state^=state>>7;state^=state<<17;return state;};
        c.cap=2*n+8;std::vector<U> expected[4];
        for(unsigned k=0;k<4;++k){c.a[k].resize(kind==2&&k==1?0:n);c.b[k].resize(n);c.out[k].resize(c.cap);expected[k].resize(c.cap);
            for(auto &v:c.a[k])v=kind==1?UINT64_MAX:random();for(auto &v:c.b[k])v=kind==1?UINT64_MAX:random();}
        for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)for(unsigned k=0;k<2;++k){
            const auto &a=c.a[2*i+k],&b=c.b[2*k+j];auto &out=expected[2*i+j];
            for(size_t x=0;x<a.size();++x){U carry=0;
                for(size_t y=0;y<b.size();++y){const __uint128_t z=__uint128_t(a[x])*b[y]+out[x+y]+carry;out[x+y]=U(z);carry=U(z>>64);}
                for(size_t at=x+b.size();carry;++at){const __uint128_t z=__uint128_t(out[at])+carry;out[at]=U(z);carry=U(z>>64);}
            }
        }
        sbn3_gcd_info q{};h::policy_query(8*n,8*n,sbn3_team_workers(team),q);c.bytes=q.scratch_bytes;
        std::vector<U> memory(c.bytes/8+8);c.work=reinterpret_cast<void*>((uintptr_t(memory.data())+63)&~uintptr_t(63));
        allocation_watch_start();sbn3_team_run(team,[](void *ptr,sbn3_team_scope *scope){
            auto &c=*static_cast<Case*>(ptr);auto f=sbn::v3::Frame::external(c.work,c.bytes);h::Engine e{f,scope};h::Metrics m{};e.metrics=&m;
            e.policy.parallel_seven=true;e.policy.batch_min=256;auto a=h::make_matrix(f,c.cap,c.cap),b=h::make_matrix(f,c.cap,c.cap);
            for(unsigned k=0;k<4;++k){h::assign(a.e[k/2][k%2],{c.a[k].data(),c.a[k].size()});h::assign(b.e[k/2][k%2],{c.b[k].data(),c.b[k].size()});}
            h::compose(a,b,e);assert(m.batch_compositions==unsigned(c.n>=128));
            for(unsigned k=0;k<4;++k){const auto &v=a.e[k/2][k%2];c.used[k]=v.n;memcpy(c.out[k].data(),v.p,v.n*8);}
        },&c);assert(!allocation_watch_stop());
        for(unsigned k=0;k<4;++k)assert(c.out[k]==expected[k]);
    }
}
int main(){
    constexpr size_t n=65537;std::vector<U>a(n),b(n),want(n),out(n);U seed=925178;
    auto random=[&]{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;};
    for(auto &v:a)v=random();for(auto &v:b)v=random();a.back()|=1ULL<<63;b.back()|=1ULL<<63;
    sbn3_gcd_info q{};assert(sbn3_gcd_query(n,n,nullptr,&q)==SBN3_SUPPORTED);
    std::vector<U> first(q.scratch_bytes/8+8);auto *f=reinterpret_cast<void*>((uintptr_t(first.data())+63)&~uintptr_t(63));
    const size_t gn=sbn3_gcd(want.data(),a.data(),n,b.data(),n,f,q.scratch_bytes,nullptr);
    for(unsigned w:{2u,4u,16u}) {
        sbn3_arena *arena=nullptr;sbn3_arena_config ac{size_t(1)<<30,8u<<20};sbn3_error error{};
        assert(sbn3_arena_create(&ac,&arena,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_lease control{};sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
        sbn3_team_config cfg{};cfg.workers=w;cfg.pin_threads=1;cfg.stack_offset=65536;for(int&c:cfg.cpu_ids)c=-1;
        sbn3_team *team=nullptr;assert(sbn3_team_create(arena,&control,&cfg,&team,&error)==SBN3_OK);
        sbn3_gcd_options options{w};assert(sbn3_gcd_query(n,n,&options,&q)==SBN3_SUPPORTED);
        sbn3_gcd_info policy_info{};sbn::v3::gcd::policy_query(n,n,w,policy_info);
        const size_t bytes=std::max(q.scratch_bytes,policy_info.scratch_bytes);
        std::vector<U> memory(bytes/8+8);auto *p=reinterpret_cast<void*>((uintptr_t(memory.data())+63)&~uintptr_t(63));
        allocation_watch_start();const size_t used=sbn3_gcd(out.data(),a.data(),n,b.data(),n,p,bytes,team);assert(!allocation_watch_stop());
        assert(used==gn&&std::equal(want.begin(),want.begin()+gn,out.begin()));
        sbn::v3::gcd::Policy policy{};policy.parallel_rows=true;policy.parallel=512;
        allocation_watch_start();const size_t count=sbn::v3::gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},p,bytes,policy,nullptr,team);assert(!allocation_watch_stop());
        assert(count==gn&&std::equal(want.begin(),want.begin()+gn,out.begin()));
        policy.parallel_rows=false;policy.cyclic=true;
        allocation_watch_start();const size_t cyc=sbn::v3::gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},p,bytes,policy,nullptr,team);assert(!allocation_watch_stop());
        assert(cyc==gn&&std::equal(want.begin(),want.begin()+gn,out.begin()));
        policy.cyclic=false;policy.projected=policy.matrix_fft=policy.middle=true;
        allocation_watch_start();const size_t projected=sbn::v3::gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},p,bytes,policy,nullptr,team);assert(!allocation_watch_stop());
        assert(projected==gn&&std::equal(want.begin(),want.begin()+gn,out.begin()));
        policy={};policy.middle_recursive=policy.middle_cache=policy.word_ifma=policy.partial_tail=true;
        sbn::v3::gcd::Metrics metrics{};
        allocation_watch_start();const size_t cached=sbn::v3::gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},p,bytes,policy,&metrics,team);assert(!allocation_watch_stop());
        assert(cached==gn&&std::equal(want.begin(),want.begin()+gn,out.begin()));
        assert(metrics.middle_recursive_windows);
        policy.parallel_seven=policy.parallel_adjust=true;
        allocation_watch_start();const size_t batched=sbn::v3::gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},p,bytes,policy,nullptr,team);assert(!allocation_watch_stop());
        assert(batched==gn&&std::equal(want.begin(),want.begin()+gn,out.begin()));
        // A deliberately larger test-only leaf wraps the bounded stream.
        // The public leaf remains 96. Compare against the independent serial
        // result and require a real flush beyond final drain.
        policy.word32=policy.leaf_pipeline=true;policy.leaf=384;metrics={};
        allocation_watch_start();const size_t streamed=sbn::v3::gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},p,bytes,policy,&metrics,team);assert(!allocation_watch_stop());
        assert(streamed==gn&&std::equal(want.begin(),want.begin()+gn,out.begin()));
        assert(metrics.leaf_streams&&metrics.leaf_stream_wraps&&metrics.leaf_stream_flushes>metrics.leaf_streams);
        {
            constexpr size_t en=16385;std::vector<U> ea(a.begin(),a.begin()+en),eb(b.begin(),b.begin()+en),gx(en),xx(en+3),yy(en+3),gold(en);
            ea.back()|=U(1)<<63;eb.back()|=U(1)<<63;
            sbn3_gcd_info eq{};assert(sbn3_exgcd_query(en,en,3,&options,&eq)==SBN3_SUPPORTED);
            std::vector<U> ew(eq.scratch_bytes/8+8);void *ep=reinterpret_cast<void*>((uintptr_t(ew.data())+63)&~uintptr_t(63));
            const size_t expected=sbn3_gcd(gold.data(),ea.data(),en,eb.data(),en,ep,eq.scratch_bytes,nullptr);
            sbn3_int eg{gx.data(),en,0,0},ex{xx.data(),en+3,0,0},ey{yy.data(),en+3,0,0};
            policy=sbn::v3::gcd::default_policy(w,true);policy.middle_recursive=policy.middle_cache=policy.deferred=policy.column_fft=policy.compact_forward=true;policy.outer=3;metrics={};
            allocation_watch_start();sbn::v3::gcd::policy_exgcd(&eg,&ex,&ey,{ea.data(),en,0},{eb.data(),en,0},ep,eq.scratch_bytes,team,policy,&metrics);assert(!allocation_watch_stop());
            assert(eg.size==expected&&std::equal(gx.begin(),gx.begin()+eg.size,gold.begin()));assert(metrics.middle_recursive_windows&&metrics.deferred_calls&&metrics.history_peak_nodes&&metrics.column_episodes);
            std::vector<U> ax(2*en+4),by(2*en+4);
            const size_t mulbytes=std::max(sbn::v3::u52::scratch_bytes(en,ex.size),sbn::v3::u52::scratch_bytes(en,ey.size));
            std::vector<U> mw(mulbytes/8+8);void *mp=reinterpret_cast<void*>((uintptr_t(mw.data())+63)&~uintptr_t(63));
            {auto f=sbn::v3::Frame::external(mp,mulbytes);sbn::v3::u52::multiply(ax.data(),ea.data(),en,xx.data(),ex.size,f);}
            {auto f=sbn::v3::Frame::external(mp,mulbytes);sbn::v3::u52::multiply(by.data(),eb.data(),en,yy.data(),ey.size,f);}
            assert(ex.negative!=ey.negative);if(ex.negative)ax.swap(by);
            assert(!sbn3i_sub_n(ax.data(),ax.data(),by.data(),long(ax.size())));
            assert(std::equal(ax.begin(),ax.begin()+eg.size,gx.begin()));for(size_t j=eg.size;j<ax.size();++j)assert(!ax[j]);
        }
        for(unsigned kind=0;kind<3;++kind){
            constexpr size_t sn=1025;std::vector<U> sa(a.begin(),a.begin()+sn),sb=sa;
            sa.back()|=U(1)<<63;sa[0]|=1;sb=sa;
            if(kind==1)--sb[0];
            if(kind==2){std::fill(sa.begin(),sa.end(),0);std::fill(sb.begin(),sb.end(),0);sa[511]=1;sb[512]=2;}
            const auto aa=sa,bb=sb;
            allocation_watch_start();const size_t result=sbn3_gcd(out.data(),sa.data(),sn,sb.data(),sn,p,bytes,team);assert(!allocation_watch_stop());
            assert(sa==aa&&sb==bb);
            if(kind==1)assert(result==1&&out[0]==1);
            else{const size_t length=kind==2?512:sn;assert(result==length&&std::equal(out.begin(),out.begin()+length,sa.begin()));}
        }
        matrix_gate(team);
        sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_stats st{};sbn3_arena_get_stats(arena,&st);assert(!st.lease_references&&!st.active_computations);sbn3_arena_destroy(arena);
        printf("HGCD W%u product scopes, disjoint rows and cyclic MAC PASS\n",w);
    }
}
