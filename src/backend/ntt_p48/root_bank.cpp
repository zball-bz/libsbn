#include "root_bank.hpp"
namespace sbn::v3::p48_root_bank {
namespace {
using Wide=__uint128_t;
constexpr uint64_t mul(uint64_t a,uint64_t b,uint64_t p){return uint64_t(Wide(a)*b%p);}
constexpr uint64_t power(uint64_t a,uint64_t n,uint64_t p){uint64_t v=1;
    for(;n;n>>=1,a=mul(a,a,p))if(n&1)v=mul(v,a,p);return v;}
constexpr uint64_t primitive(uint64_t p){uint64_t factors[16]{},n=p-1;unsigned count=0;
    for(uint64_t d=2;d*d<=n;d+=d==2?1:2)if(n%d==0){factors[count++]=d;while(n%d==0)n/=d;}
    if(n>1)factors[count++]=n;
    for(uint64_t g=2;;++g){bool ok=true;for(unsigned j=0;j<count;++j)if(power(g,(p-1)/factors[j],p)==1)ok=false;if(ok)return g;}}
struct alignas(64) Bank {
    uint64_t even[even_count+8]{},zero[64]{};
    uint64_t factor[2][(factor_log+1)*factor_stride+8]{};
};
constexpr void fill(uint64_t *out,size_t count,const uint64_t *z,uint64_t p){
    // Generate field elements independently, then convert to the stored
    // reciprocal. Readable padding stays zero.
    out[0]=1;unsigned t=1;
    for(size_t block=1;block<count;block*=2,++t)
        for(size_t j=0;j<block&&block+j<count;++j)out[block+j]=mul(z[t],out[j],p);
    for(size_t j=0;j<count;++j)out[j]=uint64_t((Wide(out[j])<<52)/p);
}
template<unsigned Q>constexpr Bank make(){
    Bank out{};constexpr uint64_t p=primes[Q];uint64_t z[34]{};
    const uint64_t root=power(primitive(p),(p-1)>>32,p);
    z[32]=root;for(unsigned t=32;t>1;--t)z[t-1]=mul(z[t],z[t],p);
    fill(out.even,even_count,z+1,p);
    constexpr uint64_t mask=(uint64_t(1)<<52)-1;
    for(size_t k=1;k<=8;k*=2)for(size_t c=0;c<k;++c){size_t b=1;while(2*b<=c)b*=2;
        out.zero[8*k-1-c]=c?out.even[3*b-1-c]:mask-out.even[0];}
    for(unsigned d=0;d<2;++d){z[32]=d?power(root,p-2,p):root;
        for(unsigned t=32;t>1;--t)z[t-1]=mul(z[t],z[t],p);
        for(unsigned k=0;k<=factor_log;++k)fill(out.factor[d]+k*factor_stride,factor_stride,z+k,p);}
    return out;
}
template<unsigned Q>inline constexpr auto bank=make<Q>();
constexpr const Bank *directory[prime_count]={&bank<0>,&bank<1>,&bank<2>,&bank<3>,&bank<4>,
    &bank<5>,&bank<6>,&bank<7>,&bank<8>,&bank<9>};
}
const uint64_t *tower(unsigned q,unsigned log)noexcept{return q<prime_count&&log<=tower_log?directory[q]->even:nullptr;}
const uint64_t *zero_image(unsigned q)noexcept{return q<prime_count?directory[q]->zero:nullptr;}
const uint64_t *factors(unsigned q,unsigned d,size_t columns,unsigned log_rows)noexcept{
    return q<prime_count&&d<2&&columns<=factor_stride&&log_rows<=factor_log?directory[q]->factor[d]:nullptr;}
size_t bytes()noexcept{return sizeof(Bank)*prime_count+sizeof(directory);}
}
