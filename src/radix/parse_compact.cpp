#include "radix/parse_compact.hpp"
#include "radix/power_bank.hpp"
#include "radix/bits.hpp"
#include "algorithms/small_division.hpp"
#include "common/identity.hpp"
#include <algorithm>
#include <cstring>
#include <new>
namespace sbn::v3::radix {
namespace {
constexpr size_t up(size_t n){return (n+127)&~size_t(127);}
size_t words(const CompactParsePlan &p,int i){return i<0?p.leaf.limbs[-i]:p.nodes[i].words;}
size_t work(const CompactParsePlan &p,int i){return i<0?0:p.nodes[i].work;}
size_t product_bytes(const product::LocalProductPlan &p){return p.persistent_bytes+p.work_bytes+256;}
int add_product(CompactParsePlan &p,size_t a,size_t b){
    for(unsigned j=0;j<p.product_count;++j)if(p.products[j].an==a&&p.products[j].bn==b)return int(j);
    if(p.product_count==24)return -1;
    if(std::min(a,b)<=2||a*b<tree_policy().group_u52_work){
        auto &q=p.products[p.product_count];q.an=a;q.bn=b;
    }else p.products[p.product_count]=product::local_product_query(a,b,false,false);
    return int(p.product_count++);
}
int node(CompactParsePlan &p,const BaseInfo &base,unsigned digits){
    const unsigned fragments=(digits+63)/64;
    if(fragments<=group_fragments)return -int(fragments);
    for(unsigned j=0;j<p.node_count;++j)if(p.nodes[j].digits==digits)return int(j);
    const unsigned k=split_level(fragments),low=64u<<k;
    const int l=node(p,base,low),h=node(p,base,digits-low);
    if(l==INT32_MIN||h==INT32_MIN||p.node_count==12)return INT32_MIN;
    const size_t hn=words(p,h),an=limbs_for_bits(power_bits(base.base_bound,digits-low)),bn=rail_limbs(base,k);
    const int product=add_product(p,an,bn);if(product<0)return INT32_MIN;
    auto &v=p.nodes[p.node_count];v.digits=digits;v.level=k;v.low=l;v.high=h;v.product=unsigned(product);
    v.words=limbs_for_bits(power_bits(base.base_bound,digits));v.shift=size_t(base.twos)<<k;
    v.work=up(hn*8)+std::max({work(p,l),work(p,h),up((an+bn)*8)+product_bytes(p.products[product])});
    return int(p.node_count++);
}
void multiply(const product::LocalProductPlan &p,Frame &frame,const uint64_t *a,const uint64_t *b,uint64_t *out){
    if(!p.work_bytes){sbn3_mul_basecase(out,p.an+p.bn,a,p.an,b,p.bn);return;}
    FrameMark mark(frame);
    const auto prepared=product::local_product_prepare(p,a,frame);
    product::local_product_apply(p,prepared,out,b,p.bn,frame);
}
bool run_node(const CompactParsePlan &p,int i,const DigitPlan &digits,const uint8_t *in,size_t count,
              uint64_t *out,Frame &frame){
    if(i<0)return parse_leaf(p.leaf,digits,in,count,out);
    const auto &n=p.nodes[i];const size_t low_digits=(size_t(1)<<n.level)*fragment_digits;
    FrameMark mark(frame);auto *high=frame.alloc<uint64_t>(words(p,n.high));
    if(!run_node(p,n.low,digits,in+count-low_digits,low_digits,out,frame)||
       !run_node(p,n.high,digits,in,count-low_digits,high,frame))return false;
    const auto &mul=p.products[n.product];auto *z=frame.alloc<uint64_t>(mul.an+mul.bn);
    multiply(mul,frame,high,fixed_power(p.base>>p.leaf.twos,n.level).data,z);
    const size_t ln=words(p,n.low);std::fill(out+ln,out+n.words,uint64_t(0));
    const size_t used=std::min(mul.an+mul.bn,n.words-n.shift);uint64_t carry=0;
    for(size_t j=0;j<used;++j){const __uint128_t v=__uint128_t(out[n.shift+j])+z[j]+carry;
        out[n.shift+j]=uint64_t(v);carry=uint64_t(v>>64);}
    for(size_t j=n.shift+used;carry&&j<n.words;++j)carry=++out[j]==0;
    require(!carry,SBN3_FATAL_MATH,"compact parse merge capacity");return true;
}
}
bool compact_parse_query(unsigned radix,size_t digits,size_t fractional,uint64_t bits,CompactParsePlan &p) noexcept {
    BaseInfo base;
    if(!base_info(radix,base)||base.odd==1||!digits||digits>compact_parse_fragments*fragment_digits||
       (fractional&&bits>64*compact_parse_precision))return false;
    // This value bound also keeps the exact quotient below the shared
    // division crossover to Barrett; word/U52 division suffices throughout.
    if(integer_limbs(base,(digits+63)/64)>compact_parse_precision)return false;
    ::new(&p)CompactParsePlan{};
    p.base=radix;p.leaf=parse_leaf_plan(radix);
    p.root=node(p,base,unsigned(digits));if(p.root==INT32_MIN)return false;
    p.number_words=words(p,p.root);p.work_bytes=up(p.number_words*8)+work(p,p.root)+256;
    if(!fractional||!bits)return true;
    // Cancel the known power of two before constructing the denominator.
    p.shift=int64_t(bits)-int64_t(base.twos*fractional);
    const uint64_t lo=uint64_t((__uint128_t(base.odd_bound.floor_q56)*fractional)>>56);
    const uint64_t hi=uint64_t((__uint128_t(base.odd_bound.floor_q56+1)*fractional)>>56);
    if(lo!=hi)return false; // ambiguous logarithm: the general recipe remains available
    p.denominator_words=limbs_for_bits(lo+1);
    p.numerator_words=limbs_for_bits(uint64_t(int64_t(power_bits(base.base_bound,digits))+p.shift));
    if(local_division_block(p.numerator_words,p.denominator_words))return false;
    p.native_division=small_division_u52(p.numerator_words,p.denominator_words);
    p.persistent_bytes=up(p.native_division?u52::divide_divisor_bytes(p.denominator_words):p.denominator_words*8);
    unsigned rest=unsigned(fractional%64);uint64_t exponent=rest;
    p.seed[0]=1;p.seed_words=1;
    while(rest){const unsigned n=std::min(8u,rest);uint64_t factor=1;
        for(unsigned j=0;j<n;++j)factor*=base.odd;
        uint64_t carry=0;for(size_t j=0;j<p.seed_words;++j){const __uint128_t v=__uint128_t(p.seed[j])*factor+carry;
            p.seed[j]=uint64_t(v);carry=uint64_t(v>>64);}
        if(carry)p.seed[p.seed_words++]=carry;rest-=n;
    }
    size_t current=p.seed_words,peak=0;const uint64_t fragments=fractional/64;
    for(unsigned k=0;(uint64_t(1)<<k)<=fragments;++k)if(fragments>>k&1){
        const size_t next=rail_limbs(base,k);const uint64_t add=uint64_t(64)<<k;
        if(!exponent){p.seed_level=int(k);p.seed_words=current=next;exponent=add;continue;}
        if(p.power_count==9)return false;
        const int product=add_product(p,current,next);if(product<0)return false;
        p.power_words=std::max(p.power_words,current+next);peak=std::max(peak,product_bytes(p.products[product]));
        exponent+=add;current=limbs_for_bits(power_bits(base.odd_bound,exponent));
        p.powers[p.power_count++]={k,unsigned(product),current};
    }
    require(exponent==fractional,SBN3_FATAL_MATH,"compact denominator exponent");
    p.work_bytes=std::max(p.work_bytes,2*up(p.power_words*8)+peak+256);
    const size_t q=p.numerator_words>=p.denominator_words?p.numerator_words-p.denominator_words+1:1;
    const size_t division=p.native_division?u52::divide_work_bytes(p.numerator_words,p.denominator_words):8*(p.numerator_words+1);
    p.work_bytes=std::max(p.work_bytes,up(p.number_words*8)+up(p.numerator_words*8)+up(q*8)+up(p.denominator_words*8)+division+512);
    p.work_bytes=up(p.work_bytes);return true;
}
uint64_t compact_parse_identity(const CompactParsePlan &p) noexcept {
    require(p.node_count<=12&&p.product_count<=24&&p.power_count<=9,
            SBN3_FATAL_ARGUMENT,"compact parse directory");
    uint64_t h=identity::fnv_seed;
    const uint64_t fields[]={p.base,p.node_count,p.product_count,p.power_count,uint64_t(p.root),uint64_t(p.seed_level),
        p.seed_words,p.power_words,p.number_words,p.numerator_words,p.denominator_words,p.persistent_bytes,p.work_bytes,
        uint64_t(p.shift),uint64_t(p.native_division)};
    for(auto v:fields)h=identity::word(h,v);
    for(auto v:p.seed)h=identity::word(h,v);
    h=identity::word(h,p.leaf.twos);
    for(auto v:p.leaf.limbs)h=identity::word(h,v);
    for(auto v:p.leaf.power_limbs)h=identity::word(h,v);
    // Placement is not arithmetic identity. Checked plans still reject a
    // damaged published-table pointer instead of hashing an ASLR address.
    if constexpr(SBN3_CHECK_SMALL){
        require(p.base>=3&&p.base<=63&&p.leaf.twos==unsigned(__builtin_ctz(p.base)),
                SBN3_FATAL_ARGUMENT,"compact parse radix");
        for(unsigned j=0;j<3;++j)require(p.leaf.powers[j]==fixed_power(p.base>>p.leaf.twos,j).data,
                                        SBN3_FATAL_ARGUMENT,"compact parse power pointer");
    }
    for(unsigned j=0;j<p.node_count;++j){const auto &n=p.nodes[j];
        const uint64_t f[]={n.digits,n.level,uint64_t(n.low),uint64_t(n.high),n.product,n.words,n.shift,n.work};
        for(auto v:f)h=identity::word(h,v);}
    for(unsigned j=0;j<p.product_count;++j){const auto &q=p.products[j];const auto &s=q.shape;
        const uint64_t f[]={q.an,q.bn,q.ring,uint64_t(q.cached),q.persistent_bytes,q.work_bytes,s.nfull,s.branch,s.radix,s.bits,
                            uint64_t(s.recipe),uint64_t(s.centered),uint64_t(s.balanced)};
        for(auto v:f)h=identity::word(h,v);}
    for(unsigned j=0;j<p.power_count;++j){const auto &q=p.powers[j];h=identity::word(h,q.level);h=identity::word(h,q.product);h=identity::word(h,q.words);}
    return h;
}
void compact_parse_prepare(const CompactParsePlan &p,CompactParseState &s,void *storage) noexcept {
    s={};if(!p.denominator_words)return;
    auto *at=static_cast<uint8_t*>(storage);auto temporary=Frame::external(at+p.persistent_bytes,p.work_bytes);
    const uint64_t *power=p.seed_level<0?p.seed:fixed_power(p.base>>p.leaf.twos,unsigned(p.seed_level)).data;
    size_t count=p.seed_words;
    auto *a=temporary.alloc<uint64_t>(p.power_words),*b=temporary.alloc<uint64_t>(p.power_words);
    for(unsigned j=0;j<p.power_count;++j){const auto &step=p.powers[j];auto *out=j&1?b:a;
        multiply(p.products[step.product],temporary,power,fixed_power(p.base>>p.leaf.twos,step.level).data,out);
        power=out;count=step.words;}
    require(count>=p.denominator_words&&power[p.denominator_words-1],SBN3_FATAL_MATH,"compact denominator length");
    for(size_t j=p.denominator_words;j<count;++j)require(!power[j],SBN3_FATAL_MATH,"compact denominator bound");
    if(p.native_division){auto persistent=Frame::external(at,p.persistent_bytes);s.native=u52::divide_prepare(power,p.denominator_words,persistent);}
    else s.word=divrem_words::prepare(power,p.denominator_words,reinterpret_cast<uint64_t*>(at),true);
}
bool compact_parse_execute(const CompactParsePlan &p,const CompactParseState &s,const DigitPlan &digits,
                           const uint8_t *in,size_t count,uint64_t bits,uint64_t *out,size_t out_words,void *storage) noexcept {
    auto space=Frame::external(static_cast<uint8_t*>(storage)+p.persistent_bytes,p.work_bytes);
    auto *number=space.alloc<uint64_t>(p.number_words);
    {FrameMark mark(space);if(!run_node(p,p.root,digits,in,count,number,space))return false;}
    if(!p.denominator_words){dyadic_window(out,out_words,number,p.number_words,-int64_t(bits),0,64*p.number_words);return true;}
    auto *n=space.alloc<uint64_t>(p.numerator_words);
    dyadic_window(n,p.numerator_words,number,p.number_words,-p.shift,0,64*p.number_words);
    size_t nn=p.numerator_words;while(nn&&!n[nn-1])--nn;
    const size_t qn=nn>=p.denominator_words?nn-p.denominator_words+1:0;
    auto *q=space.alloc<uint64_t>(std::max<size_t>(1,qn)),*r=space.alloc<uint64_t>(p.denominator_words);
    if(p.native_division)u52::divide_prepared(q,r,n,nn,s.native,space);
    else{auto *scratch=space.alloc<uint64_t>(nn+1);divrem_words::schoolbook_prepared(q,r,n,nn,s.word,scratch);}
    const size_t used=std::min(out_words,qn);if(used)memcpy(out,q,used*8);std::fill(out+used,out+out_words,uint64_t(0));
    for(size_t j=out_words;j<qn;++j)require(!q[j],SBN3_FATAL_MATH,"compact quotient capacity");
    return true;
}
}
