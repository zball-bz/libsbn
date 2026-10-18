// Original libsbn tests; LGPL-3.0-or-later.
#include "float/elliptic.hpp"
#include "float/constants.hpp"
#include "sbn3/arena.h"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
namespace f=sbn::v3::floating;
using sbn::rounding;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x31bed7a98ce051f6);
std::size_t checks=0;
struct Ref {ref_int v;Ref(){ref_init(v);}~Ref(){ref_clear(v);}Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;};
struct Ball {Ref center;std::uint64_t radius=0;std::int64_t scale=0;};
// ref_mul may select a production-backed witness. This file never calls it.
void multiply(ref_number* out,const ref_number* a,const ref_number* b){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    std::vector<std::uint64_t> words((ref_sizeinbase(b,2)+63)/64);std::size_t n=0;
    ref_export(words.data(),&n,-1,8,0,0,b);Ref sum,part;ref_set_ui(sum.v,0);
    for(std::size_t j=0;j<n;++j)if(words[j]){ref_mul_ui(part.v,a,words[j]);
        if(j)ref_mul_2exp(part.v,part.v,64*j);ref_add(sum.v,sum.v,part.v);}
    ref_set(out,sum.v);
}
void shift(ref_number* out,const ref_number* in,std::int64_t n){assert(n>=0&&n<1000000);ref_mul_2exp(out,in,std::size_t(n));}
void power2(ref_number* out,std::size_t n){ref_set_ui(out,1);ref_mul_2exp(out,out,n);}
std::uint64_t atan_fixed(ref_number* out,unsigned denominator,std::size_t bits,bool alternating){
    Ref unit,power,divisor,term;power2(unit.v,bits);ref_set_ui(power.v,denominator);ref_set_ui(out,0);
    std::size_t j=0;
    for(;;++j){
        ref_mul_ui(divisor.v,power.v,2*j+1);ref_fdiv_q(term.v,unit.v,divisor.v);
        if(!ref_sgn(term.v))break;
        if(alternating&&(j&1))ref_sub(out,out,term.v);else ref_add(out,out,term.v);
        ref_mul_ui(power.v,power.v,std::uint64_t(denominator)*denominator);
        assert(j<bits+1);
    }
    // One unit per term's floor, plus two for a first omitted <1-unit term
    // and its geometric tail (ratio at most 1/9 for the positive series).
    return j+2;
}
void pi_ball(Ball& out,std::size_t bits){
    Ref a,b;const auto ea=atan_fixed(a.v,5,bits,true),eb=atan_fixed(b.v,239,bits,true);
    ref_mul_ui(a.v,a.v,16);ref_mul_ui(b.v,b.v,4);ref_sub(out.center.v,a.v,b.v);
    out.radius=16*ea+4*eb;out.scale=-std::int64_t(bits);
}
void log2_ball(Ball& out,std::size_t bits){
    out.radius=2*atan_fixed(out.center.v,3,bits,false);ref_mul_ui(out.center.v,out.center.v,2);
    out.scale=-std::int64_t(bits);
}
// Independent fixed-point hypergeometric sums. m=num/2^denbits<=3/4.
// The positive term recurrence contracts by at most3/4: every term floor
// is within4 grid units; the E decrement within5. The omitted tail<=16.
void ke_series(Ball& k,Ball& e,unsigned num,unsigned denbits,std::size_t bits){
    assert(num*4<=3*(1u<<denbits));Ball pi;pi_ball(pi,bits);
    Ref term,ks,es,den,dec,prod;power2(term.v,bits);ref_set(ks.v,term.v);ref_set(es.v,term.v);
    std::size_t j=1;
    for(;;++j){
        ref_mul_ui(term.v,term.v,(2*j-1)*(2*j-1));ref_mul_ui(term.v,term.v,num);
        ref_set_ui(den.v,4*j*j);ref_fdiv_q(term.v,term.v,den.v);ref_fdiv_q_2exp(term.v,term.v,denbits);
        if(!ref_sgn(term.v))break;
        ref_add(ks.v,ks.v,term.v);ref_set_ui(den.v,2*j-1);ref_fdiv_q(dec.v,term.v,den.v);ref_sub(es.v,es.v,dec.v);
        assert(j<4*bits+64);
    }
    multiply(prod.v,pi.center.v,ks.v);ref_fdiv_q_2exp(k.center.v,prod.v,bits+1);
    multiply(prod.v,pi.center.v,es.v);ref_fdiv_q_2exp(e.center.v,prod.v,bits+1);
    k.radius=4*pi.radius+4*(4*j+16)+8;e.radius=4*pi.radius+4*(5*j+16)+8;
    k.scale=e.scale=-std::int64_t(bits);
}
void sqrt2_fixed(ref_number* out,std::size_t bits){
    // Do not use ref_sqrt: its large sqrt2 branch calls production Newton.
    Ref radicand,x,y,q,check;power2(radicand.v,2*bits+1);power2(x.v,bits+1);
    for(;;){ref_fdiv_q(q.v,radicand.v,x.v);ref_add(y.v,x.v,q.v);ref_fdiv_q_2exp(y.v,y.v,1);
        if(ref_cmp(y.v,x.v)>=0)break;ref_set(x.v,y.v);}
    multiply(check.v,x.v,x.v);assert(ref_cmp(check.v,radicand.v)<=0);
    ref_add_ui(y.v,x.v,1);multiply(check.v,y.v,y.v);assert(ref_cmp(check.v,radicand.v)>0);ref_set(out,x.v);
}
void negative_one(Ball& k,Ball& e,std::size_t bits){
    Ball base_k,base_e;ke_series(base_k,base_e,1,1,bits);Ref root,scaled,product;
    sqrt2_fixed(root.v,bits);ref_mul_2exp(scaled.v,base_k.center.v,bits);ref_fdiv_q(k.center.v,scaled.v,root.v);
    multiply(product.v,base_e.center.v,root.v);ref_fdiv_q_2exp(e.center.v,product.v,bits);
    k.radius=4*base_k.radius+16;e.radius=4*base_e.radius+16;k.scale=e.scale=-std::int64_t(bits);
}
void singular(Ball& k,Ball& e,std::size_t half_complement_bits,std::size_t bits){
    Ball log2;log2_ball(log2,bits);const auto a=half_complement_bits+2;
    ref_mul_ui(k.center.v,log2.center.v,a);power2(e.center.v,bits);
    const auto shift=std::int64_t(bits)+2+std::int64_t(std::bit_width(a))-2*std::int64_t(half_complement_bits);
    assert(shift<63);const auto tail=shift<=0?UINT64_C(1):UINT64_C(1)<<unsigned(shift);
    k.radius=a*log2.radius+tail;e.radius=tail;k.scale=e.scale=-std::int64_t(bits);
}
void certify(sbn::float_view y,sbn::float_report report,const Ball& truth,std::size_t precision,rounding mode){
    assert(y.finite()&&!y.zero()&&!y.negative()&&report.error.kind!=sbn::error_kind::unbounded);
    Ref center,rad,lo,hi,expected,er,elo,ehi;
    ref_import(center.v,y.magnitude().size(),-1,8,0,0,y.magnitude().data());
    const auto bitlength=ref_sizeinbase(center.v,2);assert(bitlength<=precision);
    const auto grid=y.exponent2()+std::int64_t(bitlength)-std::int64_t(precision);
    assert(report.error.exact()||report.error.exponent2<=grid+(mode==rounding::nearest_even?0:1));
    auto scale=std::min(y.exponent2(),truth.scale);if(!report.error.exact())scale=std::min(scale,report.error.exponent2);
    shift(center.v,center.v,y.exponent2()-scale);ref_set_ui(rad.v,report.error.exact()?0:1);
    if(!report.error.exact())shift(rad.v,rad.v,report.error.exponent2-scale);
    ref_sub(lo.v,center.v,rad.v);ref_add(hi.v,center.v,rad.v);
    shift(expected.v,truth.center.v,truth.scale-scale);ref_set_ui(er.v,truth.radius);shift(er.v,er.v,truth.scale-scale);
    ref_sub(elo.v,expected.v,er.v);ref_add(ehi.v,expected.v,er.v);
    if(ref_cmp(lo.v,elo.v)>0||ref_cmp(hi.v,ehi.v)<0){
        std::fprintf(stderr,"elliptic enclosure failed p=%zu mode=%u exponent=%lld radius=%lld\n",precision,unsigned(mode),
            (long long)y.exponent2(),(long long)report.error.exponent2);assert(false);}
    ++checks;
}
struct Team {
    sbn3_arena* arena=nullptr;sbn3_lease control{};sbn3_team* team=nullptr;
    explicit Team(unsigned n){sbn3_error err{};sbn3_arena_config c{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&c,&arena,&err)==SBN3_OK);assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&err)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);sbn3_team_config t{};t.workers=n;t.stack_offset=65536;
        for(auto& cpu:t.cpu_ids)cpu=-1;assert(sbn3_team_create(arena,&control,&t,&team,&err)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}
};
struct Scoped {const f::FloatEllipticPrepared* program;sbn::float_ref k,e;sbn::float_view input;
    f::EllipticParameter kind;rounding mode;f::EllipticReport report{};};
void scoped(void* opaque,sbn3_team_scope* scope){auto& c=*static_cast<Scoped*>(opaque);
    c.report=f::elliptic_program_execute(*c.program,c.k,c.e,c.input,c.kind,{{},nullptr,scope},c.mode);}
void pair_case(const f::FloatEllipticPrepared& program,std::size_t p,std::uint64_t input,std::int64_t exponent,bool negative,
               f::EllipticParameter kind,const Ball& ktruth,const Ball& etruth,rounding mode,sbn3_team* team,bool active_scope,bool alias){
    const auto limbs=(p+63)/64;std::vector<std::uint64_t> kv(limbs+2,guard),ev(limbs+2,guard);kv[0]=input;
    sbn::float_buffer k({kv.data(),limbs},sbn::bits{p}),e({ev.data(),limbs},sbn::bits{p});
    const auto x=sbn::float_view::from_magnitude({alias?kv.data():&input,1},exponent,negative);f::EllipticReport report;
    allocation_watch_start();
    if(active_scope){Scoped c{&program,k.ref(),e.ref(),x,kind,mode,{}};sbn3_team_run(team,scoped,&c);report=c.report;}
    else report=f::elliptic_program_execute(program,k.ref(),e.ref(),x,kind,{{},team,nullptr},mode);
    assert(!allocation_watch_stop());assert(kv[limbs]==guard&&kv[limbs+1]==guard&&ev[limbs]==guard&&ev[limbs+1]==guard);
    certify(k.view(),report.k,ktruth,p,mode);certify(e.view(),report.e,etruth,p,mode);
}
void constants(std::size_t p,sbn3_team* team){
    const auto workers=team?sbn3_team_workers(team):1;const auto limbs=(p+63)/64;std::vector<std::uint64_t> words(limbs);
    sbn::float_buffer out(words,sbn::bits{p});sbn::workspace storage;
    for(auto kind:{f::ConstantKind::pi,f::ConstantKind::log2}){
        allocation_watch_start();const auto need=f::cold_constant_requirements(sbn::bits{p},workers,kind);assert(!allocation_watch_stop()&&need.supported());
        storage.reserve(need.scratch_bytes,need.scratch_alignment);allocation_watch_start();
        const auto prepared=f::cold_constant_prepare(sbn::bits{p},workers,kind,storage.on().scratch.first(need.scratch_bytes));
        assert(!allocation_watch_stop());Ball truth;if(kind==f::ConstantKind::pi)pi_ball(truth,p+512);else log2_ball(truth,p+512);
        for(auto mode:modes){allocation_watch_start();const auto report=f::cold_constant_execute(prepared,out.ref(),{{},team,nullptr},mode);
            assert(!allocation_watch_stop());certify(out.view(),report,truth,p,mode);}
    }
}
void tall_alias(const f::FloatEllipticPrepared& program,std::size_t p,sbn3_team* team){
    const auto input_limbs=(2*(p+512)+63)/64,limbs=(p+63)/64;
    std::vector<std::uint64_t> words(input_limbs+2,UINT64_MAX),ew(limbs+2,guard);
    words[input_limbs]=words[input_limbs+1]=guard;
    sbn::float_buffer k({words.data(),input_limbs},sbn::bits{p}),e({ew.data(),limbs},sbn::bits{p});
    const auto input=sbn::float_view::from_magnitude({words.data(),input_limbs},-std::int64_t(64*input_limbs));
    Ball kt,et;singular(kt,et,32*input_limbs,p+512);
    allocation_watch_start();const auto report=f::elliptic_program_execute(program,k.ref(),e.ref(),input,
        f::EllipticParameter::parameter,{{},team,nullptr});assert(!allocation_watch_stop());
    assert(words[input_limbs]==guard&&words[input_limbs+1]==guard&&ew[limbs]==guard&&ew[limbs+1]==guard);
    for(std::size_t j=limbs;j<input_limbs;++j)assert(words[j]==UINT64_MAX);
    certify(k.view(),report.k,kt,p,rounding::nearest_even);certify(e.view(),report.e,et,p,rounding::nearest_even);
}
void cases(std::size_t p,sbn3_team* team,bool many){
    const auto workers=team?sbn3_team_workers(team):1;allocation_watch_start();
    const auto need=f::elliptic_program_requirements(sbn::bits{p},workers);assert(!allocation_watch_stop()&&need.supported());
    sbn::workspace storage;storage.reserve(need.scratch_bytes,need.scratch_alignment);allocation_watch_start();
    const auto prepared=f::elliptic_program_prepare(sbn::bits{p},workers,storage.on().scratch.first(need.scratch_bytes));assert(!allocation_watch_stop());
    const auto bits=p+512;Ball k,e;ke_series(k,e,1,4,bits);
    for(auto mode:modes){pair_case(prepared,p,1,-4,false,f::EllipticParameter::parameter,k,e,mode,team,team!=nullptr,true);
        pair_case(prepared,p,15,-4,false,f::EllipticParameter::complement,k,e,mode,team,false,false);}
    if(!many)return;
    ke_series(k,e,0,1,bits);
    pair_case(prepared,p,0,0,false,f::EllipticParameter::parameter,k,e,rounding::nearest_even,team,false,false);
    ke_series(k,e,1,1,bits);
    for(auto mode:modes)pair_case(prepared,p,1,-1,false,f::EllipticParameter::parameter,k,e,mode,team,false,false);
    negative_one(k,e,bits);
    pair_case(prepared,p,1,0,true,f::EllipticParameter::parameter,k,e,rounding::nearest_even,team,false,true);
    pair_case(prepared,p,2,0,false,f::EllipticParameter::complement,k,e,rounding::downward,team,false,false);
    ke_series(k,e,3,2,bits);--k.scale;++e.scale;
    pair_case(prepared,p,3,0,true,f::EllipticParameter::parameter,k,e,rounding::upward,team,false,false);
    const auto r=p+512;singular(k,e,r,bits);
    pair_case(prepared,p,1,-2*std::int64_t(r),false,f::EllipticParameter::complement,k,e,rounding::nearest_even,team,false,false);
    k.scale-=std::int64_t(r);e.scale+=std::int64_t(r);
    pair_case(prepared,p,1,2*std::int64_t(r),false,f::EllipticParameter::complement,k,e,rounding::nearest_even,team,false,false);
    // Fresh-call coverage includes every preparation, not only prepared reuse.
    const auto limbs=(p+63)/64;std::vector<std::uint64_t> kw(limbs),ew(limbs);sbn::float_buffer ko(kw,sbn::bits{p}),eo(ew,sbn::bits{p});
    const std::uint64_t three=3;const auto input=sbn::float_view::from_magnitude({&three,1},-3);
    ke_series(k,e,3,3,bits);sbn::workspace fresh;const auto fresh_need=f::elliptic_ke_requirements(ko.ref(),eo.ref(),input,f::EllipticParameter::parameter,workers);
    fresh.reserve(fresh_need.scratch_bytes,fresh_need.scratch_alignment);allocation_watch_start();
    const auto report=f::elliptic_ke_into(ko.ref(),eo.ref(),input,f::EllipticParameter::parameter,fresh.on(team));
    assert(!allocation_watch_stop());certify(ko.view(),report.k,k,p,rounding::nearest_even);certify(eo.view(),report.e,e,p,rounding::nearest_even);
    tall_alias(prepared,p,team);
}
void specials(){
    const auto need=f::elliptic_program_requirements(sbn::bits{65});sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);
    const auto prepared=f::elliptic_program_prepare(sbn::bits{65},1,work.on().scratch.first(need.scratch_bytes));
    std::array<std::uint64_t,2> kw{},ew{};sbn::float_buffer k(kw,sbn::bits{65}),e(ew,sbn::bits{65});const std::uint64_t one=1,two=2;
    auto report=f::elliptic_program_execute(prepared,k.ref(),e.ref(),sbn::float_view::from_magnitude({&one,1},0));
    assert(!k.view().finite()&&e.view().finite()&&e.view().magnitude().data()[0]==1&&report.e.error.exact());
    report=f::elliptic_program_execute(prepared,k.ref(),e.ref(),{},f::EllipticParameter::complement);assert(!k.view().finite()&&report.e.error.exact());
    report=f::elliptic_program_execute(prepared,k.ref(),e.ref(),sbn::float_view::from_magnitude({&two,1},0));
    assert(k.view().is_nan()&&e.view().is_nan()&&sbn::has_flag(report.k.flags,sbn::float_flag::invalid));
    report=f::elliptic_program_execute(prepared,k.ref(),e.ref(),sbn::float_view::infinity(true));assert(k.view().zero()&&!e.view().finite());
    report=f::elliptic_program_execute(prepared,k.ref(),e.ref(),sbn::float_view::nan());assert(k.view().is_nan()&&e.view().is_nan());
}
}
int main(){
    for(auto p:{1u,3u,17u,63u,65u,127u,193u,257u,509u}){constants(p,nullptr);cases(p,nullptr,true);}
    for(auto p:{1025u,4097u,6145u}){constants(p,nullptr);cases(p,nullptr,false);}
    {Team team(1);cases(129,team.team,true);}{Team team(4);constants(257,team.team);cases(769,team.team,true);}
    specials();std::printf("float elliptic/constants independent enclosures: %zu\n",checks);
}
