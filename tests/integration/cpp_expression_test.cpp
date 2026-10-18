// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/sbn.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;namespace ex=sbn::expr;
template<unsigned N>static auto chain(const integer&a){if constexpr(!N)return ex::value(a);else return ex::add(chain<N-1>(a),a);}
template<ex::expression E>static void check(sbn::context&ctx,E request,const integer&expected){
    const auto need=ctx.query(request);assert(need.supported());
    integer result;result.reserve(need.output_limbs);ctx.reserve(need);
    allocation_watch_start();ctx.evaluate_into(result.ref(),request);assert(!allocation_watch_stop());assert(result==expected);
    auto prepared=ctx.prepare(std::move(request));auto moved=std::move(prepared);
    allocation_watch_start();moved.execute_into(result.ref());assert(!allocation_watch_stop());assert(result==expected);
    assert(moved.evaluate()==expected);
}
int main(){
    sbn::context ctx;
    for(int a=-19;a<=19;++a)for(int b=-7;b<=7;++b){
        integer A=a,B=b;
        assert(ctx.evaluate(ex::mul(A,B)+7)==a*b+7);
        assert(ctx.evaluate((ex::value(A)&B)^(ex::value(A)|B))==((a&b)^(a|b)));
        assert(ctx.evaluate(~ex::value(A))==~A);
        assert(ctx.evaluate(ex::value(A)>>3)==(A>>3));
        assert(ctx.evaluate(ex::trunc_shift(A,3))==sbn::trunc_shift(A,3));
        if(b){for(auto round:{sbn::division_rounding::trunc,sbn::division_rounding::floor,sbn::division_rounding::ceil,sbn::division_rounding::euclidean}){
            auto denominator=ex::sub(ex::add(B,100),100);
            assert(ctx.evaluate(ex::divide(A,denominator,round))==sbn::divide(A,B,round));
            assert(ctx.evaluate(ex::remainder(A,denominator,round))==sbn::remainder(A,B,round));}}
    }
    {integer a=7,b=11;a=ctx.evaluate(ex::mul(a,b)+3);assert(a==80);}
    {integer a=7;auto request=ex::value(a);assert(ctx.evaluate(-request)==-7&&ctx.evaluate(~request)==-8&&ctx.evaluate(request<<17)==(a<<17));}
    for(size_t n:{1u,2u,7u,33u,161u,513u,1025u,4097u}){
        uint64_t state=91354127;std::vector<uint64_t>v(n),w(n/2+1);
        for(auto&x:v){state^=state<<13;state^=state>>7;state^=state<<17;x=state;}for(auto&x:w){state^=state<<13;state^=state>>7;state^=state<<17;x=state;}
        v.back()|=uint64_t(1)<<63;w.back()|=1;
        integer a{sbn::integer_view{v,true}},b{sbn::integer_view{w,false}};
        check(ctx,ex::square(ex::add(a,b))-ex::square(a)-ex::square(b),2*a*b);
        check(ctx,ex::mul(a,b)+(ex::value(a)>>31),a*b+(a>>31));
        check(ctx,ex::pow(a,3)-ex::pow(b,2),sbn::pow(a,3)-sbn::pow(b,2));
        check(ctx,ex::gcd(ex::mul(a,b),a),sbn::abs(a));
        check(ctx,ex::exact_divide(ex::mul(a,b),a),b);
        // The denominator's actual length is 1 despite an n-limb capacity.
        check(ctx,ex::divide(ex::mul(a,b),ex::sub(a,a)+3),(a*b)/3);
        check(ctx,ex::remainder(ex::mul(a,b),ex::sub(a,a)-3,sbn::division_rounding::euclidean),sbn::mod(a*b,-3));
        // Raw views can alias output: the compiled program publishes only
        // after its last input read. Managed requests use immutable guards.
        auto alias=a;alias.reserve(2*(a.size()+b.size())+4);auto raw=ex::mul(alias.view(),b.view())+5;
        const auto need=ctx.query_into(alias.ref(),raw);assert(need.supported());ctx.reserve(need);
        allocation_watch_start();ctx.evaluate_into(alias.ref(),raw);assert(!allocation_watch_stop());assert(alias==a*b+5);
    }
    {auto request=ex::mul(integer(123456789),integer(987654321))+7;auto prepared=ctx.prepare(std::move(request));auto moved=std::move(prepared);assert(moved.evaluate()==integer(123456789)*987654321+7);}
    {
        auto formula=(ex::input<0>(257)*ex::input<1>(257)+ex::square(ex::input<0>(257))) / ex::input<2>(129);
        auto prepared=ctx.prepare(std::move(formula));auto moved=std::move(prepared);
        integer out;out.reserve(moved.requirements().output_limbs);
        for(unsigned n:{0u,1u,2u,17u,64u,257u}){
            integer a=n?((integer(1)<<(64*n-1))+7):integer(0),b=n?-(integer(1)<<(32*n)):integer(3),d=(integer(1)<<(n/2))+3;
            auto expected=(a*b+sbn::square(a))/d;
            allocation_watch_start();moved.execute_into(out.ref(),a,b,d);assert(!allocation_watch_stop());assert(out==expected);assert(moved(a,b,d)==expected);
            a=moved(a,b,d);assert(a==expected);
        }
        auto bits=ex::input_bits<0>(17)<<123;integer n=-98765;assert(ctx.evaluate(std::move(bits),n)==(n<<123));
    }
    {integer a=integer(1)<<32768;auto request=chain<64>(a);const auto need=ctx.query(request);assert(need.supported());assert(need.value_bytes<10*need.output_limbs*8+4096);check(ctx,std::move(request),a*65);}
    {integer a=77,b=91;auto r=ex::mul(a,b);const auto need=ctx.query(r);sbn::workspace storage;storage.reserve(need.scratch_bytes,need.scratch_alignment);sbn::context borrowed(storage.on().scratch);integer out;out.reserve(need.output_limbs);
        allocation_watch_start();borrowed.reserve(need);borrowed.evaluate_into(out.ref(),r);assert(!allocation_watch_stop());assert(out==7007);
        sbn::context empty(std::span<std::byte>{});assert(empty.try_reserve(need));}
    assert(ctx.evaluate(ex::pow(0,0))==1);
    puts("C++ explicit expressions/context/compiled replay/moves/aliases/normalized divisor embedding and no-allocation execution PASS");
}
