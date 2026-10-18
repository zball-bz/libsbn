// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/gcd.h"
#include "gcd/internal.hpp"
#include "product/matrix_program.hpp"
#include <cassert>
#include <vector>
#include <numeric>
#include <cstdio>
#include <algorithm>
#include <cstring>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using U=uint64_t;using W=__uint128_t;using Big=std::vector<U>;
static U seed=0x87160926ac654213ULL;
static U rnd(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static void trim(Big &a){while(!a.empty()&&!a.back())a.pop_back();}
static int cmp(const Big&a,const Big&b){if(a.size()!=b.size())return a.size()>b.size()?1:-1;for(size_t i=a.size();i-->0;)if(a[i]!=b[i])return a[i]>b[i]?1:-1;return 0;}
static Big mul(const Big&a,const Big&b){Big r(a.size()+b.size());
    for(size_t i=0;i<a.size();++i){U carry=0;for(size_t j=0;j<b.size();++j){W t=W(a[i])*b[j]+r[i+j]+carry;r[i+j]=U(t);carry=U(t>>64);}r[i+b.size()]=carry;}trim(r);return r;}
static Big add(const Big&a,const Big&b){Big r(std::max(a.size(),b.size())+1);W c=0;
    for(size_t i=0;i+1<r.size();++i){c+=(i<a.size()?a[i]:0);c+=(i<b.size()?b[i]:0);r[i]=U(c);c>>=64;}r.back()=U(c);trim(r);return r;}
static Big sub(Big a,const Big&b){assert(cmp(a,b)>=0);U c=0;
    for(size_t i=0;i<a.size();++i){W t=W(i<b.size()?b[i]:0)+c;U old=a[i];a[i]-=U(t);c=W(old)<t;}assert(!c);trim(a);return a;}
static unsigned zeros(U x){return unsigned(__builtin_ctzll(x));}
static size_t shift(Big&a){size_t words=0;while(words<a.size()&&!a[words])++words;assert(words<a.size());a.erase(a.begin(),a.begin()+words);unsigned bits=zeros(a[0]);
    if(bits)for(size_t i=0;i<a.size();++i)a[i]=(a[i]>>bits)|(i+1<a.size()?a[i+1]<<(64-bits):0);trim(a);return words*64+bits;}
static Big reference(Big a,Big b){trim(a);trim(b);if(a.empty())return b;if(b.empty())return a;const size_t common=std::min(shift(a),shift(b));
    while(cmp(a,b)){if(cmp(a,b)>0)std::swap(a,b);b=sub(b,a);shift(b);}
    const unsigned s=common%64;U c=0;if(s){for(auto&v:a){const U old=v;v=(v<<s)|c;c=old>>(64-s);}if(c)a.push_back(c);}a.insert(a.begin(),common/64,0);return a;}
struct Scratch {Big backing;void *p;size_t bytes;explicit Scratch(size_t n):backing((n+127)/8+2,0x93abcd24ebed4712ULL),bytes(n){p=reinterpret_cast<void*>((uintptr_t(backing.data())+63)&~uintptr_t(63));}
    void guards(){const auto *tail=static_cast<const unsigned char*>(p)+bytes;const auto *end=reinterpret_cast<const unsigned char*>(backing.data()+backing.size());
        constexpr U sentinel=0x93abcd24ebed4712ULL;
        for(auto q=tail;q<end;++q)assert(*q==reinterpret_cast<const unsigned char*>(&sentinel)[size_t(q-reinterpret_cast<const unsigned char*>(backing.data()))%8]);}
};
static size_t count=0;
static void check(Big a,Big b,Big expected,bool extended=true,const sbn::v3::gcd::Policy *policy=nullptr,bool history_fallback=false){
    const auto aa=a,bb=b;sbn3_gcd_info i{};assert(sbn3_gcd_query(a.size(),b.size(),nullptr,&i)==SBN3_SUPPORTED);
    Scratch scratch(i.scratch_bytes);Big out(std::max(a.size(),b.size())+1,0xdeadbeef);allocation_watch_start();
    size_t n=sbn3_gcd(out.data(),a.data(),a.size(),b.data(),b.size(),scratch.p,i.scratch_bytes,nullptr);assert(!allocation_watch_stop());
    assert(n==expected.size()&&std::equal(expected.begin(),expected.end(),out.begin()));assert(out[n]==0xdeadbeef&&a==aa&&b==bb);++count;
    if(!extended)return;
    Big known_x[4],known_y[4];unsigned known_xsign[4]{},known_ysign[4]{};
    for(unsigned mask:{3u,1u,2u})for(unsigned signs=0;signs<4;++signs){
        assert(sbn3_exgcd_query(a.size(),b.size(),mask,nullptr,&i)==SBN3_SUPPORTED);Scratch work(i.scratch_bytes);
        Big gm(i.gcd_capacity+1,0xdeadbeef),xm(i.x_capacity+1,0xdeadbeef),ym(i.y_capacity+1,0xdeadbeef);
        sbn3_int g{gm.data(),i.gcd_capacity,0,0},x{xm.data(),i.x_capacity,0,0},y{ym.data(),i.y_capacity,0,0};
        sbn::v3::gcd::Metrics measured{};
        allocation_watch_start();
        if(policy)sbn::v3::gcd::policy_exgcd(&g,mask&1?&x:nullptr,mask&2?&y:nullptr,
            {a.data(),a.size(),signs&1},{b.data(),b.size(),(signs>>1)&1},work.p,i.scratch_bytes,nullptr,*policy,history_fallback?&measured:nullptr);
        else sbn3_int_exgcd(&g,mask&1?&x:nullptr,mask&2?&y:nullptr,
            {a.data(),a.size(),signs&1},{b.data(),b.size(),(signs>>1)&1},work.p,i.scratch_bytes,nullptr);
        assert(!allocation_watch_stop());if(history_fallback&&mask==3)assert(measured.deferred_calls&&measured.history_peak_nodes&&measured.deferred_updates&&measured.deferred_fallbacks);assert(g.size==expected.size()&&!g.negative&&std::equal(expected.begin(),expected.end(),gm.begin()));
        assert(gm[g.size]==0xdeadbeef&&xm[x.size]==0xdeadbeef&&ym[y.size]==0xdeadbeef&&a==aa&&b==bb);
        if(mask==3){Big ax=mul(a,Big(xm.begin(),xm.begin()+x.size)),by=mul(b,Big(ym.begin(),ym.begin()+y.size));
            const bool as=x.negative!=bool(signs&1),bs=y.negative!=bool(signs&2);Big z;
            if(as==bs){z=add(ax,by);assert(z.empty()||!as);}else if(cmp(ax,by)>=0){z=sub(ax,by);assert(z.empty()||!as);}else{z=sub(by,ax);assert(z.empty()||!bs);}
            assert(z==expected);
            known_x[signs]=Big(xm.begin(),xm.begin()+x.size);known_y[signs]=Big(ym.begin(),ym.begin()+y.size);
            known_xsign[signs]=x.negative;known_ysign[signs]=y.negative;
        }else if(mask==1)assert(Big(xm.begin(),xm.begin()+x.size)==known_x[signs]&&x.negative==known_xsign[signs]);
        else assert(Big(ym.begin(),ym.begin()+y.size)==known_y[signs]&&y.negative==known_ysign[signs]);
        ++count;
    }
}
struct ReduceCase {Big a,b;sbn::v3::gcd::Matrix m{};sbn::v3::gcd::Pair pair{};sbn::v3::Frame *frame=nullptr;bool only=false,recursive=false;sbn::v3::gcd::ResidualBound bound{};sbn::v3::gcd::Metrics metrics{};};
static void reduce_body(void *arg,sbn3_team_scope *s){auto &v=*static_cast<ReduceCase*>(arg);namespace h=sbn::v3::gcd;
    h::Engine e{*v.frame,s};e.policy.leaf=64;e.policy.word_ifma=true;v.m=h::make_matrix(*v.frame,v.a.size());
    e.policy.middle_recursive=e.policy.middle_cache=v.recursive;e.metrics=&v.metrics;
    v.pair={h::allocate(*v.frame,v.a.size()+3),h::allocate(*v.frame,v.a.size()+3)};
    h::assign(v.pair.a,{v.a.data(),v.a.size()});h::assign(v.pair.b,{v.b.data(),v.b.size()});
    if(v.only)h::hgcd_matrix_only(v.pair,v.m,e,&v.bound);else h::hgcd(v.pair,v.m,e);
}
struct CyclicCase {Big m[2][2],a,b,u,v;void *scratch;size_t bytes;};
struct MatrixCase {Big a[2][2],b[2][2];unsigned rows;void *scratch;size_t bytes;};
struct MiddleCase {sbn::v3::product::MatrixMiddlePlan plan;Big a[2][2],b[2],out[2];void *scratch;size_t bytes;size_t origin=SIZE_MAX;};
static void middle_body(void *arg,sbn3_team_scope *scope){
    auto &c=*static_cast<MiddleCase*>(arg);using namespace sbn::v3;const auto &q=c.plan.matrix.product;
    auto frame=Frame::external(c.scratch,c.bytes);auto tables=frame.subframe(q.prepared_bytes+128,128);
    const auto program=product_program_prepare(q,tables);auto work=frame.subframe(c.plan.matrix.workspace_bytes);
    MatrixProductInputs in{};in.middle=true;in.middle_first_vector=c.origin;const size_t d=c.plan.half_words;
    for(unsigned i=0;i<2;++i){
        for(unsigned j=0;j<2;++j)in.left[i][j]={c.a[i][j].data(),c.a[i][j].size()};
        in.right[i][0]={c.b[i].data(),c.b[i].size()};in.right[i][1]={c.b[i].data()+d,c.b[i].size()-d};
        in.out[i][0]={c.out[i].data(),c.out[i].size()};
    }
    allocation_watch_start();product::matrix_program_execute(program,work,scope,in);assert(!allocation_watch_stop());
}
static void matrix_body(void *arg,sbn3_team_scope *scope){
    auto &c=*static_cast<MatrixCase*>(arg);namespace h=sbn::v3::gcd;
    auto frame=sbn::v3::Frame::external(c.scratch,c.bytes);h::Engine e{frame,scope};h::Metrics metrics{};
    e.metrics=&metrics;e.policy.matrix_fft=true;
    const size_t cap=c.a[0][0].size()+c.b[0][0].size()+8;
    auto a=h::make_matrix(frame,cap,cap,c.rows),b=h::make_matrix(frame,cap,cap,3);
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
        if(c.rows&(1u<<i))h::assign(a.e[i][j],{c.a[i][j].data(),c.a[i][j].size()});
        h::assign(b.e[i][j],{c.b[i][j].data(),c.b[i][j].size()});
    }
    allocation_watch_start();h::compose(a,b,e);assert(!allocation_watch_stop());assert(metrics.matrix_episodes==1);
    for(unsigned i=0;i<2;++i)if(c.rows&(1u<<i))for(unsigned j=0;j<2;++j){
        const auto expected=add(mul(c.a[i][0],c.b[0][j]),mul(c.a[i][1],c.b[1][j]));
        assert(Big(a.e[i][j].p,a.e[i][j].p+a.e[i][j].n)==expected);
    }
}
static void cyclic_body(void *arg,sbn3_team_scope *scope) {
    auto &c=*static_cast<CyclicCase*>(arg);namespace h=sbn::v3::gcd;
    auto frame=sbn::v3::Frame::external(c.scratch,c.bytes);h::Engine e{frame,scope};h::Metrics metrics{};e.metrics=&metrics;
    auto m=h::make_matrix(frame,256,256,3);for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)h::assign(m.e[i][j],{c.m[i][j].data(),c.m[i][j].size()});
    h::Pair p{h::allocate(frame,c.a.size()+3),h::allocate(frame,c.b.size()+3)};
    h::assign(p.a,{c.a.data(),c.a.size()});h::assign(p.b,{c.b.data(),c.b.size()});
    assert(h::cyclic_apply(p,m,e));assert(metrics.cyclic==1);
    assert(metrics.cyclic_sums==1);
    assert(Big(p.a.p,p.a.p+p.a.n)==c.u&&Big(p.b.p,p.b.p+p.b.n)==c.v);
}
int main(){
    { // A one-word inverse matrix reuses the exact leaf kernel, including
      // 63-bit coefficients, unequal active lengths and vector end carries.
        namespace h=sbn::v3::gcd;
        for(size_t n:{15u,16u,17u,127u,257u,4097u})for(U q:{U(3),(U(1)<<40)-1,(U(1)<<52)-1,(U(1)<<63)-2})for(unsigned kind=0;kind<3;++kind){
            Big u(n),v(n);for(auto&x:u)x=rnd();for(auto&x:v)x=kind==1?UINT64_MAX:rnd();u.back()|=1;v.back()|=1;
            const U words[3][4]{{q+1,q,1,1},{1,q,0,1},{1,0,q,1}};
            Big a=add(mul({words[kind][0]},u),mul({words[kind][1]},v));
            Big b=add(mul({words[kind][2]},u),mul({words[kind][3]},v));
            const size_t cap=std::max(a.size(),b.size())+3;Scratch memory(32*cap+4096);
            auto f=sbn::v3::Frame::external(memory.p,memory.bytes);h::Engine e{f};e.policy.word32=true;h::Metrics metrics{};e.metrics=&metrics;
            auto m=h::make_matrix(f,8,8,3);for(unsigned i=0;i<4;++i)h::word(m.e[i/2][i%2],words[kind][i]);
            h::Pair pair{h::allocate(f,cap),h::allocate(f,cap)};h::assign(pair.a,{a.data(),a.size()});h::assign(pair.b,{b.data(),b.size()});
            allocation_watch_start();h::apply_matrix(pair,m,e);assert(!allocation_watch_stop()&&!metrics.products);
            assert(Big(pair.a.p,pair.a.p+pair.a.n)==u&&Big(pair.b.p,pair.b.p+pair.b.n)==v);memory.guards();
        }
        Big a(8193);for(auto&x:a)x=rnd();a.back()|=U(1)<<63;
        check(a,a,a);check(a,sub(a,{1}),{1});
        Big b(8193);a.assign(8193,0);a.back()=U(1)<<63;b.back()=U(1)<<61;check(a,b,b);
        // Equal lengths bound the seed quotient by one full unsigned word,
        // not merely the u63 domain of a certified Lehmer head.
        b.assign(129,0);b.back()=1;a=add(mul(b,{UINT64_MAX}),{1});assert(a.size()==b.size());check(a,b,{1});
    }
    { // Streamed division honors the minimal quotient capacity in its contract.
        Big b{0xabcdef0123456789ULL,0xfedcba9876543210ULL},q(257);for(auto&v:q)v=rnd();q.back()|=1;
        const auto a=add(mul(b,q),{1});const size_t qn=a.size()-b.size()+1;
        Big out(qn+1,0xfeedbabe12345678ULL),rem(3);sbn3_gcd_info info{};sbn::v3::gcd::policy_query(a.size(),b.size(),1,info);Scratch scratch(info.scratch_bytes);
        auto frame=sbn::v3::Frame::external(scratch.p,scratch.bytes);
        struct Case{const Big*a,*b;Big*q,*r;sbn::v3::Frame*f;size_t cap,qn=0,rn=0;}c{&a,&b,&out,&rem,&frame,qn};
        allocation_watch_start();sbn::v3::serial_team_run([](void*ptr,sbn3_team_scope*scope){auto&c=*static_cast<Case*>(ptr);namespace h=sbn::v3::gcd;
            h::Engine e{*c.f,scope};h::Nat q{c.q->data(),0,c.cap},r{c.r->data(),0,c.r->size()};h::stream_remainder(r,{c.a->data(),c.a->size()},{c.b->data(),c.b->size()},e,&q);c.qn=q.n;c.rn=r.n;
        },&c);assert(!allocation_watch_stop());assert(c.qn==q.size()&&std::equal(q.begin(),q.end(),out.begin())&&out[qn]==0xfeedbabe12345678ULL&&c.rn==1&&rem[0]==1);scratch.guards();
    }

    for(unsigned outer:{0u,2u,4u}){
        auto policy=sbn::v3::gcd::default_policy(1,true);policy.deferred=policy.column_fft=policy.compact_forward=policy.middle_recursive=policy.middle_cache=true;policy.outer=outer;
        for(size_t n:{97u,128u,257u,1025u,4097u}){
            Big a(n),b(n);for(auto&x:a)x=rnd();for(auto&x:b)x=rnd();a.back()|=U(1)<<63;b.back()|=U(1)<<63;
            check(a,b,reference(a,b),true,&policy);
            if(n==1025){check(a,a,a,true,&policy);auto low=sub(a,{1});check(a,low,{1},true,&policy);}
        }
        { // A useful prefix followed by a nearly equal suffix exercises
          // eager continuation with a retained parent, then reverse replay.
            Big b(257),c(65);for(auto&v:b)v=rnd();for(auto&v:c)v=rnd();b.back()|=U(1)<<62;c.back()|=U(1)<<62;
            const auto u=add(add(b,b),{1}),cm=sub(c,{1}),cp=add(c,{1});
            auto a0=add(mul(c,u),mul(cm,b)),b0=add(mul(cp,u),mul(c,b));
            check(a0,b0,{1},true,&policy,true);
        }
        Big b(257),q(130);for(auto&v:b)v=rnd();b.back()|=U(1)<<63;q.back()=1;
        const auto exact=mul(b,q),off=add(exact,{1});
        check(exact,b,b,true,&policy);check(b,exact,b,true,&policy);
        check(off,b,{1},true,&policy);check(b,off,{1},true,&policy);
    }

    // MAC2 must be admitted during geometry selection, not rejected after
    // choosing a basis under the one-product coefficient bound.
    for(size_t n:{262144u,333333u,524288u}){
        sbn::v3::product::MatrixMiddlePlan p{};
        assert(sbn::v3::product::matrix_middle_query(n,1,p,64));
        assert(p.half_words>=n/4&&3*p.half_words<=n);
    }
    // Matrix-only middle recursion must carry an honest lower bound, even
    // though its scratch inputs are not returned reduced values. Verify
    // against independent whole integer products, including cancellation
    // and abnormal first quotients; no multiplication backend is its oracle.
    for(size_t n:{4096u,5313u,8192u})for(unsigned kind=0;kind<3;++kind){
        ReduceCase r{};r.a.resize(n);r.b.resize(n);r.only=r.recursive=true;
        for(auto &x:r.a)x=rnd();for(auto &x:r.b)x=rnd();r.a.back()|=U(1)<<63;r.b.back()|=U(1)<<63;
        if(kind==1)r.b=sub(r.a,{1});
        if(kind==2){r.b.resize(n/2+7);r.b.back()|=1;}
        sbn3_gcd_info info{};sbn::v3::gcd::policy_query(n,n,1,info);Scratch scratch(info.scratch_bytes);
        auto f=sbn::v3::Frame::external(scratch.p,scratch.bytes);r.frame=&f;sbn::v3::serial_team_run(reduce_body,&r);
        auto big=[](sbn::v3::gcd::Span x){return Big(x.p,x.p+x.n);};
        const auto m00=big(r.m.e[0][0]),m01=big(r.m.e[0][1]),m10=big(r.m.e[1][0]),m11=big(r.m.e[1][1]);
        const auto u=sub(mul(m11,r.a),mul(m01,r.b)),v=sub(mul(m00,r.b),mul(m10,r.a));
        assert(sub(mul(m00,m11),mul(m01,m10))==Big{1});
        auto bits=[](const Big &a){return a.empty()?size_t(0):64*a.size()-__builtin_clzll(a.back());};
        assert(bits(u)>r.bound.lower_bits&&bits(v)>r.bound.lower_bits);
        if(!kind){assert(r.metrics.middle_recursive_windows);if(n==8192)assert(r.metrics.middle_cache_compositions);}
        scratch.guards();
    }
    // The optimized local division can need more scratch than the bounded
    // u52 fallback. Use the complete GCD quote at this former failure and
    // its neighbours; quotient discard streams the original numerator.
    for(size_t dn:{5792u,5793u,5794u}){
        Big d(dn),q(dn-1);for(auto&x:d)x=rnd();for(auto&x:q)x=rnd();
        d.back()|=U(1)<<63;q.back()=UINT64_MAX;
        const auto a=add(mul(d,q),{1});check(a,d,{1},false);
    }
    {
        Big zero(1025);sbn3_gcd_info info{};assert(sbn3_gcd_query(zero.size(),zero.size(),nullptr,&info)==SBN3_SUPPORTED);
        Scratch work(info.scratch_bytes);
        assert(sbn3_gcd(nullptr,zero.data(),zero.size(),zero.data(),zero.size(),work.p,work.bytes,nullptr)==0);
    }
    for(size_t n:{15u,16u,17u,31u,32u,63u,64u,257u})for(unsigned mode=0;mode<4;++mode){
        Big a(n),b(n);for(auto&x:a)x=mode==0?UINT64_MAX:mode==1?1:rnd();for(auto&x:b)x=mode==0?UINT64_MAX:rnd();
        sbn::v3::gcd_small::Matrix m;for(auto &row:m.u)for(auto&x:row)x=mode<2?(U(1)<<63)-1:rnd()&((U(1)<<52)-1);
        const auto x=add(mul(a,{m.u[0][0]}),mul(b,{m.u[1][0]})),y=add(mul(a,{m.u[0][1]}),mul(b,{m.u[1][1]}));
        a.push_back(0);b.push_back(0);sbn::v3::gcd::compose_head(a.data(),b.data(),n+1,m);trim(a);trim(b);assert(a==x&&b==y);
    }
    // Construct exact positive residuals and a determinant-one word matrix.
    // This covers signed borrow chains, every vector/tail boundary and the
    // high halves of 63-bit coefficients without relying on random heads.
    for(size_t n:{7u,14u,15u,16u,17u,23u,31u,32u,47u,48u,49u,63u,64u,65u,96u,128u})
    for(unsigned bits:{1u,31u,32u,51u,52u,62u,63u})for(unsigned kind=0;kind<3;++kind){
        Big r(n),t(n);for(auto&x:r)x=kind==0?UINT64_MAX:kind==1?0:rnd();for(auto&x:t)x=kind==0?0:kind==1?UINT64_MAX:rnd();r.back()|=1;t.back()|=1;trim(r);trim(t);
        const U q=(U(1)<<bits)-2;sbn::v3::gcd_small::Matrix m{{{1,q},{1,q+1}}};
        Big a=add(r,mul(t,{q})),b=add(r,mul(t,{q+1}));const size_t len=std::max(a.size(),b.size());a.resize(len);b.resize(len);
        allocation_watch_start();sbn::v3::gcd::apply_head_word32(a.data(),b.data(),len,m);assert(!allocation_watch_stop());trim(a);trim(b);assert(a==r&&b==t);
        a=r;b=t;const size_t count=std::max(a.size(),b.size())+1;const auto x=add(a,b),y=add(mul(a,{q}),mul(b,{q+1}));a.resize(count);b.resize(count);
        allocation_watch_start();sbn::v3::gcd::compose_head_word32(a.data(),b.data(),count,m);assert(!allocation_watch_stop());trim(a);trim(b);assert(a==x&&b==y);
    }
    for(size_t n:{4096u,5312u,6173u,16384u}){
        MiddleCase c{};assert(sbn::v3::product::matrix_middle_query(n,1,c.plan));const size_t d=c.plan.half_words;
        const size_t coefficient=n==5312?d/2:n==6173?d-37:d-2;
        size_t discard=d;
        if(n==5312||n==6173){const size_t unit=c.plan.digit_bits%8?2:1;
            c.origin=((8*coefficient+c.plan.digit_bits-1)/c.plan.digit_bits+unit-1)/unit*unit;
            discard=c.origin*c.plan.digit_bits/8;
            c.plan.matrix.workspace_bytes=sbn::v3::product::matrix_program_workspace(c.plan.matrix,3,c.origin);}
        for(unsigned i=0;i<2;++i){
            c.b[i].resize(2*d);for(auto&x:c.b[i])x=rnd();
            c.out[i].resize(sbn::v3::product::matrix_middle_output_words(c.plan.matrix,c.origin==SIZE_MAX?c.plan.matrix.product.info.lbw/2:c.origin));
            for(unsigned j=0;j<2;++j){c.a[i][j].resize(coefficient);for(auto&x:c.a[i][j])x=n==4096?UINT64_MAX:rnd();}
        }
        Scratch work(c.plan.matrix.product.prepared_bytes+c.plan.matrix.workspace_bytes+512);c.scratch=work.p;c.bytes=work.bytes;
        sbn::v3::serial_team_run(middle_body,&c);
        for(unsigned i=0;i<2;++i){
            Big expected=add(mul(c.a[i][0],c.b[0]),mul(c.a[i][1],c.b[1]));expected.erase(expected.begin(),expected.begin()+discard);trim(c.out[i]);
            const auto error=sub(expected,c.out[i]);
            const size_t bits=error.empty()?0:64*(error.size()-1)+64-__builtin_clzll(error.back());
            assert(bits<=c.plan.error_bits);
        }
    }
    for(size_t n:{4097u,8191u}){
        Big a(n),b(n),out(n+1);for(auto&x:a)x=rnd();for(auto&x:b)x=rnd();a.back()|=U(1)<<63;b.back()|=U(1)<<63;
        const auto want=reference(a,b);sbn3_gcd_info info{};sbn::v3::gcd::policy_query(n,n,1,info);Scratch work(info.scratch_bytes);
        sbn::v3::gcd::Policy policy{};policy.projected=policy.matrix_fft=policy.middle=policy.word_ifma=policy.partial_tail=true;
        sbn::v3::gcd::Metrics metrics{};allocation_watch_start();
        const auto used=sbn::v3::gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},work.p,work.bytes,policy,&metrics);
        assert(!allocation_watch_stop());assert(metrics.projected&&metrics.cyclic);
        if(n==8191)assert(metrics.middle_windows);
        assert(used==want.size()&&std::equal(want.begin(),want.end(),out.begin()));
        // Exercise owned terminal overlap after removal of whole-word 2-adic
        // factors. The GCD is shifted as well; low zeros are not padding.
        a.insert(a.begin(),8,0);b.insert(b.begin(),8,0);Big shifted=want;shifted.insert(shifted.begin(),8,0);
        check(a,b,shifted,false);
    }
    for(size_t n:{512u,513u,745u,1024u})for(unsigned rows:{1u,2u,3u}){
        MatrixCase c{};c.rows=rows;
        for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
            c.a[i][j].resize(n);c.b[i][j].resize(n-3);
            for(auto &x:c.a[i][j])x=rnd();for(auto &x:c.b[i][j])x=rnd();
            if(n==1024)std::fill(c.a[i][j].begin(),c.a[i][j].end(),UINT64_MAX);
        }
        // Keep both maxima above the experiment's shared-transform boundary.
        c.b[0][0].resize(n);
        Scratch work(8u<<20);c.scratch=work.p;c.bytes=work.bytes;
        sbn::v3::serial_team_run(matrix_body,&c);
    }
    for(U a:{0,1,2,3,7,17,193})for(U b:{0,1,2,5,11,29,127})check(a?Big{a}:Big{},b?Big{b}:Big{},a||b?Big{std::gcd(a,b)}:Big{});
    for(size_t n:{2u,3u,8u,15u,16u,17u,31u,64u,95u,96u,97u,128u,255u,512u,769u,1024u,1025u,2048u}){
        Big a(n),b(n);for(auto&x:a)x=rnd();for(auto&x:b)x=rnd();a.back()|=U(1)<<63;b.back()|=U(1)<<63;
        auto want=reference(a,b);check(a,b,want);
        check(a,a,a);check(a,{},a,false);
        Big c(n/2+1);for(auto&x:c)x=rnd();c.back()|=1;
        check(a,c,reference(a,c),n<=128);check(c,a,reference(a,c),n<=128);
        if(n<=512)for(bool only:{false,true}){ReduceCase r{a,b};r.only=only;Scratch scratch(1024*n+65536);auto f=sbn::v3::Frame::external(scratch.p,scratch.bytes);r.frame=&f;
            sbn::v3::serial_team_run(reduce_body,&r);
            auto big=[](sbn::v3::gcd::Span x){return Big(x.p,x.p+x.n);};Big u=big(r.pair.a),v=big(r.pair.b);
            Big m00=big(r.m.e[0][0]),m01=big(r.m.e[0][1]),m10=big(r.m.e[1][0]),m11=big(r.m.e[1][1]);
            if(only){u=sub(mul(m11,a),mul(m01,b));v=sub(mul(m00,b),mul(m10,a));}
            assert(add(mul(m00,u),mul(m01,v))==a&&add(mul(m10,u),mul(m11,v))==b);
            assert(sub(mul(m00,m11),mul(m01,m10))==Big{1});
            if(n>2){assert(u.size()>n/2+1&&v.size()>n/2+1);if(!only)assert((cmp(u,v)>0?sub(u,v):sub(v,u)).size()<=n/2+1);}
        }
    }
    // A balanced positive unimodular matrix with large coefficients makes
    // the cyclic output bound smaller than the individual products.
    {
        Big f0{0},f1{1};unsigned index=1;
        while(f1.size()<128||(index&1)){auto next=add(f0,f1);f0=f1;f1=next;++index;}
        CyclicCase c{};c.m[0][0]=add(f0,f1);c.m[0][1]=f1;c.m[1][0]=f1;c.m[1][1]=f0;
        assert(sub(mul(c.m[0][0],c.m[1][1]),mul(f1,f1))==Big{1});
        c.u.resize(200);c.v.resize(200);for(auto&x:c.u)x=rnd();for(auto&x:c.v)x=rnd();c.u.back()|=1;c.v.back()|=1;
        c.a=add(mul(c.m[0][0],c.u),mul(c.m[0][1],c.v));c.b=add(mul(c.m[1][0],c.u),mul(c.m[1][1],c.v));
        Scratch work(2u<<20);c.scratch=work.p;c.bytes=work.bytes;sbn::v3::serial_team_run(cyclic_body,&c);
    }
    puts("HGCD reconstruction/determinant and signed optional Bezout outputs PASS");
    printf("GCD/exgcd %zu array cases, no allocation PASS\n",count);
}
