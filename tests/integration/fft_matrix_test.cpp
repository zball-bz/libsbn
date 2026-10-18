// Original libsbn tests; LGPL-3.0-or-later.
#include "product/matrix_program.hpp"
#include "gcd/internal.hpp"
#include <vector>
#include <cassert>
#include <cstdio>
#include <algorithm>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using U=uint64_t;using W=__uint128_t;using Big=std::vector<U>;
static U seed=9827131;
static U random_word(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static void trim(Big&a){while(!a.empty()&&!a.back())a.pop_back();}
static Big add(Big a,const Big&b){a.resize(std::max(a.size(),b.size())+1);W c=0;for(size_t j=0;j<a.size();++j){c+=a[j]+W(j<b.size()?b[j]:0);a[j]=U(c);c>>=64;}trim(a);return a;}
static Big mul(const Big&a,const Big&b){Big z(a.size()+b.size());for(size_t i=0;i<a.size();++i){U c=0;for(size_t j=0;j<b.size();++j){const W t=W(a[i])*b[j]+z[i+j]+c;z[i+j]=U(t);c=U(t>>64);}z[i+b.size()]=c;}trim(z);return z;}
static int compare(const Big&a,const Big&b){if(a.size()!=b.size())return a.size()>b.size()?1:-1;for(size_t i=a.size();i-->0;)if(a[i]!=b[i])return a[i]>b[i]?1:-1;return 0;}
static Big difference(Big a,Big b){if(compare(a,b)<0)a.swap(b);U cy=0;for(size_t i=0;i<a.size();++i){const W t=W(i<b.size()?b[i]:0)+cy;const U old=a[i];a[i]-=U(t);cy=W(old)<t;}assert(!cy);trim(a);return a;}
static Big residue(const Big&x,size_t ring){Big out(ring);
    for(size_t at=0;at<x.size();at+=ring){W c=0;for(size_t j=0;j<ring;++j){c+=out[j]+W(at+j<x.size()?x[at+j]:0);out[j]=U(c);c>>=64;}
        while(c)for(size_t j=0;j<ring&&c;++j){c+=out[j];out[j]=U(c);c>>=64;}}
    if(std::all_of(out.begin(),out.end(),[](U x){return x==UINT64_MAX;}))std::fill(out.begin(),out.end(),0);trim(out);return out;
}
struct Case {
    sbn::v3::product::MatrixProgramPlan plan{};
    Big a[2][2],b[2][2],out[2][2];
    void *scratch=nullptr;size_t bytes=0,origin=0;unsigned rows=3;bool middle=false,cache=false,vector=false,column=false,compact=false,alias_right=false;size_t b_live[2]{};
};
struct TeamOwner {
    sbn3_arena *arena=nullptr;sbn3_team *team=nullptr;sbn3_lease control{};
    explicit TeamOwner(unsigned w){if(w==1)return;sbn3_error error{};sbn3_arena_config ac{size_t(1)<<30,16u<<20};
        assert(sbn3_arena_create(&ac,&arena,&error)==SBN3_OK);assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);sbn3_team_config c{};c.workers=w;c.stack_offset=65536;for(auto&cpu:c.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena,&control,&c,&team,&error)==SBN3_OK);}
    ~TeamOwner(){if(team){sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}}
};
static void body(void *ptr,sbn3_team_scope *scope){
    using namespace sbn::v3;auto &c=*static_cast<Case*>(ptr);auto f=Frame::external(c.scratch,c.bytes);
    const auto&q=c.plan.product;auto tables=f.subframe(q.prepared_bytes+128,128);const auto p=product_program_prepare(q,tables);
    MatrixSpectrumCache cache{};if(c.cache){cache.bytes=product::matrix_spectrum_bytes(c.plan);cache.data=f.alloc<uint8_t>(cache.bytes);}
    auto work=f.subframe(c.plan.workspace_bytes,64);MatrixProductInputs in{};in.rows=c.rows;in.middle=c.middle;in.middle_first_vector=c.origin;
    in.vector=c.vector||c.column;in.negative_left=c.vector?6:0;in.compact_forward=c.compact;
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){in.left[i][j]={c.a[i][j].data(),c.a[i][j].size()};in.right[i][j]={c.b[i][j].data(),c.b[i][j].size()};in.out[i][j]={c.out[i][j].data(),c.out[i][j].size()};}
    if(c.alias_right)for(unsigned row=0;row<2;++row){in.right[row][0].count=c.b_live[row];in.out[row][0]={c.b[row][0].data(),c.b[row][0].size()};}
    if(c.cache)in.left_cache=&cache;
    allocation_watch_start();product::matrix_program_execute(p,work,scope,in);assert(!allocation_watch_stop());
    if(c.cache){
        // Reuse without recomputing any source spectrum, including zero entries.
        in.use_cached_left=true;allocation_watch_start();product::matrix_program_execute(p,work,scope,in);assert(!allocation_watch_stop());
    }
}
int main(){
    using namespace sbn::v3;
    {
        pq16::Shape a{2048,2048,1,false,pq16::Recipe::CooleyTukeyPQ,16,false},b;
        memset(&b,0xa5,sizeof b);b.nfull=a.nfull;b.branch=a.branch;b.radix=a.radix;b.centered=a.centered;
        b.recipe=a.recipe;b.bits=a.bits;b.balanced=a.balanced;
        ProductProgramPlan x{},y{};
        assert(fft_matrix_query(127,127,0,16,a,x.plan,x.info)&&fft_matrix_query(127,127,0,16,b,y.plan,y.info));
        assert(!memcmp(&x.plan,&y.plan,sizeof x.plan)&&x.info.C==1&&x.info.M2==2048&&x.info.transform_trunks==4096);
        assert(product_program_describe(127,127,x)==SBN3_UNSUPPORTED);
        assert(product_program_describe(127,127,x,ProgramCapability::matrix)==SBN3_SUPPORTED);
    }
    size_t cases=0;
    for(unsigned workers:{1u,2u,4u,16u}){TeamOwner team(workers);
    for(unsigned bits=16;bits<=20;++bits)for(unsigned radix:{1u,3u,5u,7u})for(bool balanced:{false,true})for(unsigned kind=0;kind<4;++kind){
        const unsigned branch=bits<19?512:128;
        pq16::Shape s{radix*branch,branch,radix,false,pq16::Recipe::CooleyTukeyPQ,bits,balanced};
        if(!pq16::matrix_supported(s))continue;
        const size_t ring=pq16::cyclic_period(s),d=ring/2;
        for(bool middle:{false,true})for(unsigned rows:{1u,2u,3u}){
            if(middle&&rows!=3)continue;
            Case c;c.rows=rows;c.middle=middle;c.cache=(kind&1)&&rows==3;c.compact=true;
            const unsigned quantum=8u/(1u<<std::min(3u,unsigned(__builtin_ctz(bits))));
            c.origin=(((64*(d-7)+bits-1)/bits+unsigned(balanced)+7)/8+quantum-1)/quantum*quantum;
            const size_t discard=c.origin*bits/8;
            const unsigned error_bits=bits+2+unsigned(64-__builtin_clzll(2*s.nfull));
            const size_t an=d-7,bn=middle?ring:d-9;
            auto&q=c.plan.product;assert(fft_matrix_query(an,bn,middle?ring:0,workers,s,q.plan,q.info));
            assert(product_program_describe(an,bn,q,ProgramCapability::matrix)==SBN3_SUPPORTED);
            c.plan.workspace_bytes=product::matrix_program_workspace(c.plan,rows,middle?c.origin:SIZE_MAX);
            Big sources[2];
            for(unsigned i=0;i<2;++i){if(middle){sources[i].resize(3*d);for(auto&v:sources[i])v=kind==2?UINT64_MAX:random_word();}
                for(unsigned j=0;j<2;++j){c.a[i][j].resize(kind==3&&i==1&&j==0?0:an);for(auto&v:c.a[i][j])v=kind==2?UINT64_MAX:random_word();
                    if(middle)c.b[i][j]=Big(sources[i].begin()+j*d,sources[i].begin()+(j+2)*d);
                    else{c.b[i][j].resize(bn);for(auto&v:c.b[i][j])v=kind==2?UINT64_MAX:random_word();}
                    c.out[i][j].resize(middle?product::matrix_middle_output_words(c.plan,c.origin):an+bn+1,0);
                }
            }
            c.bytes=q.prepared_bytes+c.plan.workspace_bytes+(c.cache?product::matrix_spectrum_bytes(c.plan):0)+512;
            Big storage((c.bytes+127)/8+16,0x24bced691abc3221);c.scratch=reinterpret_cast<void*>((uintptr_t(storage.data())+63)&~uintptr_t(63));
            if(team.team)sbn3_team_run(team.team,body,&c);else serial_team_run(body,&c);
            for(unsigned i=0;i<2;++i)if(rows&(1u<<i))for(unsigned j=0;j<(middle?1u:2u);++j){
                auto expected=add(mul(c.a[i][0],middle?sources[0]:c.b[0][j]),mul(c.a[i][1],middle?sources[1]:c.b[1][j]));
                if(middle){expected.resize(3*d);expected.erase(expected.begin(),expected.begin()+discard);c.out[i][j].resize(3*d-discard);trim(expected);}
                trim(c.out[i][j]);auto error=difference(expected,c.out[i][j]);
                if(middle&&!error.empty()){
                    // A balanced cyclic window is an approximate residue.
                    // Near zero the omitted signed carry can cross the
                    // window modulus; HGCD's lower/upper bounds lift it.
                    Big modulus(3*d-discard+1);modulus.back()=1;
                    auto other=difference(modulus,error);if(compare(other,error)<0)error=std::move(other);
                }
                if(!middle)assert(error.empty());else {
                    if(!(error.size()<=1&&(error.empty()||error[0]<(U(1)<<error_bits)))){
                        fprintf(stderr,"middle M%u balanced=%u kind=%u row=%u origin=%zu sizes=%zu/%zu error=%zu low=%llx/%llx\n",radix,unsigned(balanced),kind,i,c.origin,expected.size(),c.out[i][j].size(),error.size(),(unsigned long long)(expected.empty()?0:expected[0]),(unsigned long long)(c.out[i][j].empty()?0:c.out[i][j][0]));
                    }
                    assert(error.size()<=1&&(error.empty()||error[0]<(U(1)<<error_bits)));
                }
            }
            if(!middle&&rows==3){
                c.cache=false;c.column=true;c.compact=true;
                c.plan.workspace_bytes=product::matrix_program_workspace(c.plan,3,matrix_vector_layout);
                c.bytes=q.prepared_bytes+c.plan.workspace_bytes+512;
                Big column_storage((c.bytes+127)/8+16,0x3edcbb69ac13b247ULL);c.scratch=reinterpret_cast<void*>((uintptr_t(column_storage.data())+63)&~uintptr_t(63));
                const auto unused0=c.out[0][1],unused1=c.out[1][1];
                for(unsigned row=0;row<2;++row)c.out[row][0].assign(q.info.output_limbs,0);
                if(team.team)sbn3_team_run(team.team,body,&c);else serial_team_run(body,&c);
                for(unsigned row=0;row<2;++row){auto expected=add(mul(c.a[row][0],c.b[0][0]),mul(c.a[row][1],c.b[1][0]));trim(c.out[row][0]);assert(c.out[row][0]==expected);}
                assert(c.out[0][1]==unused0&&c.out[1][1]==unused1);++cases;
                c.alias_right=true;
                for(unsigned row=0;row<2;++row){c.b_live[row]=c.b[row][0].size();c.b[row][0].resize(q.info.output_limbs,0xbbeeddaa22446688ULL);}
                if(team.team)sbn3_team_run(team.team,body,&c);else serial_team_run(body,&c);
                for(unsigned row=0;row<2;++row){trim(c.b[row][0]);assert(c.b[row][0]==c.out[row][0]);}++cases;
            }
            ++cases;
        }
    }
    for(unsigned np=4;np<=10;++np)for(unsigned crt:{1u,2u})for(unsigned kind=0;kind<3;++kind){
        constexpr size_t an=127,bn=1029;FixedNttGeometry geometry{};assert(fixed_ntt_geometry(an,bn,0,workers,np,geometry));
        const auto*backend=backend_lookup(np);auto options=fixed_ntt_options(geometry);options.crt_mode=crt;options.trunk_bits=0;options.borrow_output=0;
        sbn3_product_request request{};request.kind=SBN3_PRODUCT_MAC2;request.a_limbs=request.a1_limbs=an;request.b_limbs=request.b1_limbs=bn;
        Case c;c.column=true;c.compact=true;auto&q=c.plan.product;sbn3_product_info info{};
        assert(backend->product_query(request,options,q.plan,info)==SBN3_SUPPORTED);q.info=info.mul;q.a1=an;q.b1=bn;
        assert(product_program_describe(an,bn,q,ProgramCapability::matrix)==SBN3_SUPPORTED);
        c.plan.workspace_bytes=product::matrix_program_workspace(c.plan,3,matrix_vector_layout);assert(c.plan.workspace_bytes);
        for(unsigned row=0;row<2;++row){c.b[row][0].resize(bn);for(auto&v:c.b[row][0])v=kind==1?UINT64_MAX:random_word();
            for(unsigned col=0;col<2;++col){c.a[row][col].resize(kind==2&&row==col?0:an);for(auto&v:c.a[row][col])v=kind==1?UINT64_MAX:random_word();}
            c.out[row][0].resize(q.info.output_limbs);}
        c.bytes=q.prepared_bytes+c.plan.workspace_bytes+512;Big storage((c.bytes+127)/8+16);c.scratch=reinterpret_cast<void*>((uintptr_t(storage.data())+63)&~uintptr_t(63));
        if(team.team)sbn3_team_run(team.team,body,&c);else serial_team_run(body,&c);
        for(unsigned row=0;row<2;++row){auto expected=add(mul(c.a[row][0],c.b[0][0]),mul(c.a[row][1],c.b[1][0]));trim(c.out[row][0]);assert(c.out[row][0]==expected);}++cases;
        c.alias_right=true;for(unsigned row=0;row<2;++row){c.b_live[row]=c.b[row][0].size();c.b[row][0].resize(q.info.output_limbs,0xaabbccddeeff1234ULL);}
        if(team.team)sbn3_team_run(team.team,body,&c);else serial_team_run(body,&c);
        for(unsigned row=0;row<2;++row){trim(c.b[row][0]);assert(c.b[row][0]==c.out[row][0]);}++cases;
    }
    // Signed cyclic DOT2 uses a nonnegative CRT bias which vanishes in
    // the minus ring. Exercise every prime set and both CRT scale modes.
    for(unsigned np=4;np<=10;++np)for(unsigned crt:{1u,2u})for(unsigned kind=0;kind<3;++kind){
        FixedNttGeometry g{};assert(fixed_ntt_geometry(1536,127,1536,workers,np,g,size_t(1)<<19,0,false,FixedNttProduct::sum2));
        auto opt=fixed_ntt_options(g);opt.crt_mode=crt;opt.borrow_output=0;
        sbn3_product_request req{};req.kind=SBN3_PRODUCT_MAC2;req.a_limbs=req.a1_limbs=127;req.b_limbs=req.b1_limbs=g.ring;req.cyclic_limbs=g.ring;
        Case c;c.vector=true;auto&q=c.plan.product;sbn3_product_info info{};const auto*backend=backend_lookup(np);
        assert(backend->product_query(req,opt,q.plan,info)==SBN3_SUPPORTED);q.info=info.mul;
        assert(product_program_describe(127,g.ring,q,ProgramCapability::matrix)==SBN3_SUPPORTED);c.plan.workspace_bytes=product::matrix_program_workspace(c.plan,3,matrix_vector_layout);
        for(unsigned i=0;i<2;++i){c.b[i][0].resize(g.ring);for(auto&v:c.b[i][0])v=kind==1?UINT64_MAX:random_word();c.out[i][0].resize(g.ring);
            for(unsigned j=0;j<2;++j){c.a[i][j].resize(kind==2?0:127);for(auto&v:c.a[i][j])v=kind==1?UINT64_MAX:random_word();}}
        c.bytes=q.prepared_bytes+c.plan.workspace_bytes+512;Big storage((c.bytes+127)/8+16);c.scratch=reinterpret_cast<void*>((uintptr_t(storage.data())+63)&~uintptr_t(63));
        if(team.team)sbn3_team_run(team.team,body,&c);else serial_team_run(body,&c);
        for(unsigned i=0;i<2;++i){auto a=mul(c.a[i][0],c.b[0][0]),b=mul(c.a[i][1],c.b[1][0]);if(i)a.swap(b);
            const bool negative=compare(a,b)<0;auto gold=residue(difference(a,b),g.ring);
            if(negative&&!gold.empty())gold=difference(Big(g.ring,UINT64_MAX),gold);
            trim(c.out[i][0]);assert(c.out[i][0]==gold);}
        ++cases;
    }
    }
    // Reconstructed integer residuals exercise the two
    // consumers together, through the existing complete caller scratch quote.
    for(size_t n:{4096u,5313u,8192u,16385u}){
        Big a(n),b(n),want(n),out(n);for(auto&v:a)v=random_word();for(auto&v:b)v=random_word();a.back()|=U(1)<<63;b.back()|=U(1)<<63;
        sbn3_gcd_info i{};sbn::v3::gcd::policy_query(n,n,1,i);Big mem(i.scratch_bytes/8+16);void*p=reinterpret_cast<void*>((uintptr_t(mem.data())+63)&~uintptr_t(63));
        const size_t gn=sbn3_gcd(want.data(),a.data(),n,b.data(),n,p,i.scratch_bytes,nullptr);
        gcd::Policy policy{};policy.middle_recursive=policy.middle_cache=policy.word_ifma=policy.partial_tail=true;
        policy.fft_matrix=policy.fft_vector=policy.joint_vector=true;gcd::Metrics m{};
        allocation_watch_start();const size_t got=gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},p,i.scratch_bytes,policy,&m);assert(!allocation_watch_stop());
        assert(got==gn&&std::equal(out.begin(),out.begin()+got,want.begin()));++cases;
        policy.fft_vector=false;const size_t ntt=gcd::policy_gcd(out.data(),{a.data(),n},{b.data(),n},p,i.scratch_bytes,policy,&m);
        assert(ntt==gn&&std::equal(out.begin(),out.begin()+ntt,want.begin()));++cases;
    }
    printf("FFT matrix/middle/cache/integer gates: %zu cases\n",cases);
}
