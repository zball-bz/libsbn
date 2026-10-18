// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/detail/expression.hpp"
#include "sbn3/division.hpp"
#include "sbn3/gcd.h"
#include "algorithms/divrem_prepared.hpp"
#include "runtime/resident.hpp"
#include "product/program.hpp"
#include "value/bitwise.hpp"
#include <algorithm>
#include <cstring>
#include <new>
namespace sbn::detail {
namespace {
using namespace sbn::v3;
using Op=expression_op;using Instruction=expression_instruction;
constexpr uint64_t marker=0x53424e3343505031ULL;
constexpr size_t max_nodes=max_expression_instructions,max_words=(size_t(1)<<31)-16;
struct Header {
    uint64_t magic;size_t count,bytes,output_limbs,control_bytes,values_at,kernel_at,kernel_bytes,alignment;
    unsigned workers,busy;size_t parameters;
};
struct Node {
    Op op;unsigned left,right;size_t argument,bits,logical,capacity,alignment,value_at,last_use,payload_at,kernel_bytes;
    bool leaf_kernel; sbn3_int value;
};
struct Division {
    sbn3_divrem_plan plan;sbn3_divrem_info info;
    size_t nn,dn,n_at,d_at,q_at,r_at,session_at,work_at,bytes,aux_at,aux_bytes;bool compact,fixed_divisor;
};
size_t align(size_t n,size_t a)noexcept{size_t v;require(align_size(n,a,v),SBN3_FATAL_SIZE,"expression alignment");return v;}
size_t limbs(size_t bits)noexcept{return (bits+63)/64;}
bool binary(Op o)noexcept{return o==Op::add||o==Op::subtract||o==Op::multiply||o==Op::bit_and||o==Op::bit_or||o==Op::bit_xor||o==Op::quotient||o==Op::remainder||o==Op::exact_quotient||o==Op::gcd;}
bool division(Op o)noexcept{return o==Op::quotient||o==Op::remainder||o==Op::exact_quotient;}
size_t bits_of(sbn3_int_view v)noexcept{return v.size?64*(v.size-1)+64-unsigned(__builtin_clzll(v.data[v.size-1])):0;}
struct Range {size_t begin,end;};
// Linear lifetime allocation: retired values contribute coalescing free
// ranges. No numerical candidate search and no allocation during planning.
struct Slots {
    Range free[2*max_nodes];size_t count=0,end=0;
    void release(size_t at,size_t bytes)noexcept{
        if(!bytes)return;size_t i=0;while(i<count&&free[i].begin<at)++i;
        require(count<2*max_nodes,SBN3_FATAL_SIZE,"expression free ranges");
        for(size_t j=count;j>i;--j)free[j]=free[j-1];free[i]={at,at+bytes};++count;
        if(i&&free[i-1].end==free[i].begin){free[i-1].end=free[i].end;for(size_t j=i;j+1<count;++j)free[j]=free[j+1];--count;--i;}
        if(i+1<count&&free[i].end==free[i+1].begin){free[i].end=free[i+1].end;for(size_t j=i+1;j+1<count;++j)free[j]=free[j+1];--count;}
    }
    size_t allocate(size_t bytes,size_t alignment)noexcept{
        if(!bytes)return 0;
        for(size_t i=0;i<count;++i){const auto r=free[i];const auto at=align(r.begin,alignment);
            if(at>r.end||bytes>r.end-at)continue;
            for(size_t j=i;j+1<count;++j)free[j]=free[j+1];--count;
            release(r.begin,at-r.begin);release(at+bytes,r.end-at-bytes);return at;
        }
        const auto at=align(end,alignment);end=at+bytes;return at;
    }
};
struct Compiler {
    std::span<const Instruction>input;unsigned workers;std::span<std::byte>storage;
    Node nodes[max_nodes];expression_requirements need{};size_t cursor=0,kernel_alignment=128,value_alignment=64;
    Compiler(std::span<const Instruction>i,unsigned w,std::span<std::byte>s)noexcept:input(i),workers(w),storage(s){}
    bool reject(const char*why,support s=support::unsupported)noexcept{need.status=s;need.reason=why;return false;}
    template<class T>size_t save(const T&value)noexcept{
        static_assert(std::is_trivially_copyable_v<T>);
        const auto at=align(cursor,std::max(size_t(128),alignof(T)));cursor=at+sizeof(T);
        if(storage.data()){require(cursor<=storage.size(),SBN3_FATAL_WORKSPACE,"expression plan storage",cursor,storage.size());std::memcpy(storage.data()+at,&value,sizeof(T));}return at;
    }
    bool compile_product(Node&n,const Node&a,const Node&b,bool sq)noexcept{
        if(!a.logical||!b.logical){n.leaf_kernel=true;return true;}
        n.capacity=std::max(n.capacity,a.logical+b.logical);
        if((a.logical<=6&&b.logical<=6)||std::min(a.logical,b.logical)<=1){n.leaf_kernel=true;return true;}
        ProductProgramPlan p;sbn3_mul_options opts{};opts.workers=workers;opts.borrow_output=1;
        if(sq){sbn3_product_request r{};r.kind=SBN3_PRODUCT_SQR;r.a_limbs=a.logical;sbn3_product_info info;
            if(sbn3_product_query(&r,&opts,&p.plan,&info)!=SBN3_SUPPORTED)return reject("expression square backend");p.info=info.mul;
            if(product_program_describe(a.logical,0,p)!=SBN3_SUPPORTED)return reject("expression square program");
        }else if(product_program_query(a.logical,b.logical,opts,p)!=SBN3_SUPPORTED)return reject("expression product backend");
        n.capacity=std::max(n.capacity,sbn3_mul_output_capacity(&p.info));n.alignment=std::max(size_t(64),p.info.output_alignment);
        n.kernel_bytes=p.prepared_bytes+p.info.workspace_bytes+p.info.workspace_alignment+384;
        kernel_alignment=std::max(kernel_alignment,p.info.workspace_alignment);n.payload_at=save(p);return true;
    }
    bool compile_division(Node&n,const Node&a,const Node&d)noexcept{
        if(!d.logical)return reject("expression zero divisor");
        if(n.argument>unsigned(division_rounding::euclidean))return reject("expression division rounding");
        // A word divisor has no transform plan. Reserve the two output
        // journals and the native word kernel's numerator scratch only.
        const auto word_bytes=n.op==Op::remainder?size_t(128):8*(2*a.logical+5)+256;
        if(d.logical<=2){n.leaf_kernel=true;n.kernel_bytes=word_bytes;return true;}
        Division p; p.dn=d.logical;p.fixed_divisor=d.op==Op::value;
        p.nn=p.fixed_divisor?a.logical:a.logical+p.dn-1;
        if(p.nn>max_words)return reject("expression division padded shape");
        const sbn3_divrem_request request{p.nn,p.dn};sbn3_divrem_options options{};options.workers=workers;
        const auto status=sbn3_divrem_query(&request,&options,&p.plan,&p.info);
        if(status!=SBN3_SUPPORTED)return reject("expression division backend",status==SBN3_QUERY_CAPACITY?support::capacity:support::unsupported);
        p.compact=p.plan.opaque[0]==divrem_prepared::magic;size_t at=0;
        p.n_at=at;at+=8*std::max(size_t(1),p.nn);p.d_at=at;at+=8*p.dn;
        p.q_at=at;if(n.op!=Op::remainder)at+=8*(std::max(size_t(1),p.info.quotient_limbs)+1);p.r_at=at;at+=8*p.dn;
        p.aux_bytes=n.op==Op::remainder?sbn3_remainder_plan_workspace_bytes(&p.plan):0;p.aux_at=align(at,64);if(p.aux_bytes)at=p.aux_at+p.aux_bytes;
        p.session_at=align(at,alignof(ResidentSession));if(!p.compact)at=p.session_at+sizeof(ResidentSession);
        p.work_at=align(at,p.info.storage_alignment);p.bytes=p.work_at+p.info.storage_bytes;
        n.kernel_bytes=std::max(p.bytes,word_bytes);kernel_alignment=std::max(kernel_alignment,p.info.storage_alignment);n.payload_at=save(p);return true;
    }
    bool node(size_t i)noexcept{
        const auto&in=input[i];auto&n=nodes[i];n={};n.op=in.op;n.left=in.left;n.right=in.right;n.argument=in.argument;n.alignment=64;n.last_use=i;
        if(n.op==Op::value){n.bits=bits_of(in.input);n.logical=in.input.size;n.capacity=n.logical;return n.logical<=max_words||reject("expression input domain");}
        if(n.op==Op::parameter){
            if(n.argument>=max_nodes||in.bound_bits>max_words*64)return reject("expression parameter domain");
            need.parameters=std::max(need.parameters,n.argument+1);n.bits=in.bound_bits;n.capacity=n.logical=limbs(n.bits);return true;
        }
        if(n.left>=i||(binary(n.op)&&n.right>=i))return reject("expression dependency order");
        auto&a=nodes[n.left];auto&b=binary(n.op)?nodes[n.right]:a;a.last_use=i;if(binary(n.op))b.last_use=i;
        __uint128_t bound=0;
        switch(n.op){
        case Op::add:case Op::subtract:bound=std::max(a.bits,b.bits)+1;n.capacity=std::max(a.logical,b.logical)+1;break;
        case Op::multiply:bound=a.bits&&b.bits?__uint128_t(a.bits)+b.bits:0;break;
        case Op::square:bound=__uint128_t(a.bits)*2;break;
        case Op::negate:case Op::absolute:bound=a.bits;n.capacity=a.logical;break;
        case Op::bit_and:case Op::bit_or:case Op::bit_xor:bound=std::max(a.bits,b.bits)+1;n.capacity=std::max(a.logical,b.logical)+1;break;
        case Op::complement:bound=__uint128_t(a.bits)+1;n.capacity=a.logical+1;break;
        case Op::left_shift:bound=a.bits?__uint128_t(a.bits)+n.argument:0;if(a.bits&&bound<=__uint128_t(max_words)*64)n.capacity=a.logical+n.argument/64+1;break;
        case Op::right_shift:bound=n.argument?std::min(a.bits,std::max(size_t(1),a.bits>n.argument?a.bits-n.argument+1:1)):a.bits;n.capacity=a.logical;break;
        case Op::trunc_shift:bound=a.bits>n.argument?a.bits-n.argument:0;n.capacity=a.logical;break;
        case Op::quotient:case Op::exact_quotient:bound=__uint128_t(a.bits)+unsigned(n.argument!=0);break;
        case Op::remainder:bound=n.argument?b.bits:std::min(a.bits,b.bits);break;
        case Op::gcd:bound=std::max(a.bits,b.bits);n.capacity=std::max(a.logical,b.logical);break;
        default:return reject("unknown expression operation");
        }
        if(bound>__uint128_t(max_words)*64)return reject("expression result domain");
        n.bits=size_t(bound);n.logical=limbs(n.bits);n.capacity=std::max(n.capacity,n.logical);
        if(n.op==Op::multiply||n.op==Op::square){if(!compile_product(n,a,b,n.op==Op::square))return false;}
        else if(division(n.op)){if(!compile_division(n,a,b))return false;}
        else if(n.op==Op::gcd){sbn3_gcd_info info;sbn3_gcd_options opts{workers};const auto status=sbn3_gcd_query(a.logical,b.logical,&opts,&info);
            if(status!=SBN3_SUPPORTED)return reject("expression GCD domain");n.capacity=std::max(n.capacity,info.gcd_capacity);n.kernel_bytes=info.scratch_bytes;kernel_alignment=std::max(kernel_alignment,info.scratch_alignment);}
        value_alignment=std::max(value_alignment,n.alignment);need.kernel_bytes=std::max(need.kernel_bytes,n.kernel_bytes);return true;
    }
    expression_requirements run()noexcept{
        need.scratch_alignment=128;need.instructions=input.size();
        if(input.empty()||input.size()>max_nodes||!workers||workers>32){reject("expression size/workers");return need;}
        cursor=align(sizeof(Header),alignof(Node))+input.size()*sizeof(Node);
        for(size_t i=0;i<input.size();++i)if(!node(i))return need;
        nodes[input.size()-1].last_use=input.size();Slots slots;bool retired[max_nodes]{};
        for(size_t i=0;i<input.size();++i){
            for(size_t j=0;j<i;++j)if(!retired[j]&&nodes[j].op!=Op::value&&nodes[j].last_use<i){slots.release(nodes[j].value_at,8*nodes[j].capacity);retired[j]=true;}
            if(nodes[i].op!=Op::value)nodes[i].value_at=slots.allocate(nodes[i].capacity*8,nodes[i].alignment);
        }
        need.control_bytes=cursor;need.value_bytes=slots.end;need.output_limbs=nodes[input.size()-1].logical;
        const auto values_at=align(cursor,value_alignment),kernel_at=align(values_at+slots.end,kernel_alignment);
        need.scratch_alignment=std::max(value_alignment,kernel_alignment);need.scratch_bytes=kernel_at+need.kernel_bytes;
        if(storage.data()){
            require(storage.size()>=need.scratch_bytes&&!(uintptr_t(storage.data())&(need.scratch_alignment-1)),SBN3_FATAL_WORKSPACE,"compiled expression storage",need.scratch_bytes,storage.size());
            auto*records=reinterpret_cast<Node*>(storage.data()+align(sizeof(Header),alignof(Node)));
            for(size_t i=0;i<input.size();++i)::new(records+i)Node(nodes[i]);
            ::new(storage.data())Header{marker,input.size(),need.scratch_bytes,need.output_limbs,need.control_bytes,values_at,kernel_at,need.kernel_bytes,need.scratch_alignment,workers,0,need.parameters};
        }
        return need;
    }
};
void set_value(sbn3_int&out,const uint64_t*p,size_t n,bool sign)noexcept{
    require(n<=out.capacity,SBN3_FATAL_MATH,"expression value bound");if(n&&p!=out.data)std::memmove(out.data,p,n*8);out.size=n;out.negative=n&&sign;
}
struct ProductRun {const ProductProgramPlan*plan;const sbn3_int*a,*b;sbn3_int*out;size_t an,bn;void*work;size_t bytes;bool square;};
void product_action(void*raw,sbn3_team_scope*scope){auto&r=*static_cast<ProductRun*>(raw);auto frame=Frame::external(r.work,r.bytes);const auto&p=*r.plan;
    auto tables=frame.subframe(p.prepared_bytes+128,128);auto program=product_program_prepare(p,tables);auto work=frame.subframe(p.info.workspace_bytes,p.info.workspace_alignment);
    product_program_execute(program,work,scope,{r.a->data,r.an},r.square?sbn3_const_limbs{}:sbn3_const_limbs{r.b->data,r.bn},{r.out->data,r.out->capacity});
}
void finish_division(const Node&node,sbn3_int&out,const sbn3_int&a,const sbn3_int&d,uint64_t*Q,uint64_t*R,
                     const uint64_t*D,size_t dn,size_t shift,sbn3_divrem_result r)noexcept{
    if(node.op==Op::exact_quotient)require(!r.remainder_limbs,SBN3_FATAL_MATH,"expression nonexact division");
    const auto mode=division_rounding(node.argument);bool rn=a.negative;
    const bool adjust=r.remainder_limbs&&(mode==division_rounding::floor?a.negative!=d.negative:mode==division_rounding::ceil?a.negative==d.negative:mode==division_rounding::euclidean?bool(a.negative):false);
    if(adjust){if(Q){size_t i=0;for(;i<r.quotient_limbs&&Q[i]==UINT64_MAX;++i)Q[i]=0;if(i==r.quotient_limbs)Q[r.quotient_limbs++]=1;else ++Q[i];}
        uint64_t borrow=0;for(size_t j=0;j<dn;++j){const auto x=D[j],y=j<r.remainder_limbs?R[j]:0,t=x-y,v=t-borrow;borrow=(x<y)|(t<borrow);R[j]=v;}
        require(!borrow,SBN3_FATAL_MATH,"expression rounded remainder");r.remainder_limbs=bits::normalized_size(R,dn);rn=!rn;}
    if(node.op==Op::remainder){require(!r.remainder_limbs||r.remainder_limbs>shift,SBN3_FATAL_MATH,"expression remainder scale");set_value(out,R+shift,r.remainder_limbs?r.remainder_limbs-shift:0,rn);}
    else set_value(out,Q,r.quotient_limbs,a.negative!=d.negative);
}
void divide_word(const Node&node,sbn3_int&out,const sbn3_int&a,const sbn3_int&d,void*work,size_t bytes)noexcept{
    require(d.size&&d.size<=2,SBN3_FATAL_MATH,"expression word divisor");auto frame=Frame::external(work,bytes);
    auto*Q=node.op==Op::remainder?nullptr:frame.alloc<uint64_t>(a.size+1),*R=frame.alloc<uint64_t>(2),*scratch=node.op==Op::remainder?nullptr:frame.alloc<uint64_t>(a.size+1);
    const uint64_t D[2]{d.data[0],d.size>1?d.data[1]:0};
    size_t qn=0;if(Q)qn=sbn3_divrem_small(Q,R,a.data,a.size,D,d.size,scratch);else sbn3_remainder_small(R,a.data,a.size,D,d.size,nullptr);
    finish_division(node,out,a,d,Q,R,D,d.size,0,{qn,bits::normalized_size(R,d.size),0});
}
void divide_node(const Division&p,const Node&node,sbn3_int&out,const sbn3_int&a,const sbn3_int&d,void*work,sbn3_team*team)noexcept{
    require(d.size,SBN3_FATAL_MATH,"expression computed zero divisor");const auto shift=p.dn-d.size;
    require(!p.fixed_divisor||!shift,SBN3_FATAL_LIFETIME,"expression fixed divisor width changed");
    auto*base=static_cast<std::byte*>(work);auto*N=reinterpret_cast<uint64_t*>(base+p.n_at),*D=reinterpret_cast<uint64_t*>(base+p.d_at),*Q=node.op==Op::remainder?nullptr:reinterpret_cast<uint64_t*>(base+p.q_at),*R=reinterpret_cast<uint64_t*>(base+p.r_at);
    std::memset(D,0,shift*8);std::memcpy(D+shift,d.data,d.size*8);
    const size_t nn=a.size?a.size+shift:0;if(a.size){std::memset(N,0,shift*8);std::memcpy(N+shift,a.data,a.size*8);}
    sbn3_divrem_binding*b;ResidentSession*session=nullptr;void*state=base+p.work_at;
    if(p.compact)b=divrem_prepared::bind_raw(p.plan,state,p.info.storage_bytes);
    else{session=::new(base+p.session_at)ResidentSession({static_cast<std::byte*>(state),p.info.storage_bytes},team);sbn3_divrem_bind(&p.plan,session->arena(),0,session->team(),&b);}
    sbn3_divrem_prepare(b,{D,p.dn});sbn3_divrem_result r{};
    if(Q)sbn3_divrem_execute(b,{N,nn},{Q,std::max(size_t(1),p.info.quotient_limbs)+1},{R,p.dn},&r);
    else sbn3_remainder_execute(b,{N,nn},{R,p.dn},p.aux_bytes?base+p.aux_at:nullptr,p.aux_bytes,&r);
    sbn3_divrem_unbind(b);if(session)session->~ResidentSession();
    finish_division(node,out,a,d,Q,R,D,p.dn,shift,r);
}
}
expression_requirements expression_query(std::span<const Instruction>input,unsigned workers)noexcept{
    Compiler c{input,workers,{}};return c.run();
}
expression_requirements expression_compile(std::span<const Instruction>input,execution ex,std::span<const integer_view>parameters)noexcept{
    require(!ex.scope,SBN3_FATAL_TEAM,"expression compiler needs root execution");
    require(ex.scratch.data()&&!(uintptr_t(ex.scratch.data())&127),SBN3_FATAL_WORKSPACE,"expression control storage");
    require(!overlaps(input.data(),input.size_bytes(),ex.scratch.data(),ex.scratch.size()),SBN3_FATAL_ARGUMENT,"expression instructions/work overlap");
    for(const auto&i:input)if(i.op==Op::value)require(!overlaps(i.input.data,i.input.size*8,ex.scratch.data(),ex.scratch.size()),SBN3_FATAL_ARGUMENT,"expression input/work overlap");
    for(const auto&v:parameters)require(!overlaps(v.data(),v.size()*8,ex.scratch.data(),ex.scratch.size()),SBN3_FATAL_ARGUMENT,"expression parameter/work overlap");
    Compiler c{input,ex.workers(),ex.scratch};auto need=c.run();require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);return need;
}
void expression_execute(integer_ref destination,std::span<const Instruction>input,execution ex,std::span<const integer_view>parameters)noexcept{
    require(!ex.scope&&ex.scratch.size()>=sizeof(Header),SBN3_FATAL_WORKSPACE,"expression compiled header");
    auto&h=*reinterpret_cast<Header*>(ex.scratch.data());require(h.magic==marker&&h.count==input.size()&&h.workers==ex.workers()&&h.bytes<=ex.scratch.size()&&!(uintptr_t(ex.scratch.data())&(h.alignment-1)),SBN3_FATAL_LIFETIME,"expression compiled identity");
    require(parameters.size()==h.parameters,SBN3_FATAL_ARGUMENT,"expression parameter count",h.parameters,parameters.size());
    require(destination.capacity()>=h.output_limbs,SBN3_FATAL_WORKSPACE,"expression result capacity",h.output_limbs,destination.capacity());
    require(!overlaps(destination.view().data(),destination.capacity()*8,ex.scratch.data(),h.bytes),SBN3_FATAL_ARGUMENT,"expression result/work overlap");
    unsigned expected=0;require(__atomic_compare_exchange_n(&h.busy,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent expression execution");
    auto*nodes=reinterpret_cast<Node*>(ex.scratch.data()+align(sizeof(Header),alignof(Node)));void*kernel=ex.scratch.data()+h.kernel_at;
    for(size_t i=0;i<h.count;++i){auto&n=nodes[i];const auto&in=input[i];require(n.op==in.op&&n.left==in.left&&n.right==in.right&&n.argument==in.argument,SBN3_FATAL_LIFETIME,"expression graph changed");
        if(n.op==Op::value){const auto v=in.input;require(v.size==n.logical&&bits_of(v)==n.bits&&!overlaps(v.data,v.size*8,ex.scratch.data(),h.bytes),SBN3_FATAL_LIFETIME,"expression input changed/work alias");n.value={const_cast<uint64_t*>(v.data),v.size,v.size,v.negative};}
        if(n.op==Op::parameter){const auto v=parameters[n.argument].c_view();require(n.bits==in.bound_bits&&v.size<=n.logical&&bits_of(v)<=n.bits&&!overlaps(v.data,v.size*8,ex.scratch.data(),h.bytes),SBN3_FATAL_ARGUMENT,"expression parameter bound/work alias");}}
    for(size_t i=0;i<h.count;++i){auto&n=nodes[i];if(n.op==Op::value)continue;
        if(n.op==Op::parameter){const auto v=parameters[n.argument].c_view();
            if(v.size==n.logical)n.value={const_cast<uint64_t*>(v.data),v.size,v.size,v.negative};
            else{n.value={reinterpret_cast<uint64_t*>(ex.scratch.data()+h.values_at+n.value_at),n.capacity,v.size,v.negative};if(v.size)std::memcpy(n.value.data,v.data,v.size*8);if(n.capacity>v.size)std::memset(n.value.data+v.size,0,(n.capacity-v.size)*8);}continue;}
        auto&a=nodes[n.left].value;auto&b=binary(n.op)?nodes[n.right].value:a;auto&r=n.value;
        r={reinterpret_cast<uint64_t*>(ex.scratch.data()+h.values_at+n.value_at),n.capacity,0,0};const sbn3_int_view av{a.data,a.size,a.negative},bv{b.data,b.size,b.negative};
        switch(n.op){
        case Op::add:sbn3_int_add(&r,av,bv);break;
        case Op::subtract:sbn3_int_sub(&r,av,bv);break;
        case Op::negate:set_value(r,a.data,a.size,!a.negative);break;
        case Op::absolute:set_value(r,a.data,a.size,false);break;
        case Op::bit_and:sbn3_int_and(&r,av,bv);break;
        case Op::bit_or:sbn3_int_or(&r,av,bv);break;
        case Op::bit_xor:sbn3_int_xor(&r,av,bv);break;
        case Op::complement:sbn3_int_complement(&r,av);break;
        case Op::left_shift:sbn3_int_lshift(&r,av,n.argument);break;
        case Op::right_shift:sbn3_int_arshift(&r,av,n.argument);break;
        case Op::trunc_shift:sbn3_int_rshift(&r,av,n.argument);break;
        case Op::multiply:case Op::square:{
            if(!a.size||!b.size){r.size=r.negative=0;break;}
            const auto an=nodes[n.left].logical,bn=n.op==Op::square?an:nodes[n.right].logical;
            if(n.leaf_kernel)sbn3_mul_basecase(r.data,r.capacity,a.data,an,b.data,bn);
            else{const auto*p=reinterpret_cast<const ProductProgramPlan*>(ex.scratch.data()+n.payload_at);ProductRun call{p,&a,&b,&r,an,bn,kernel,h.kernel_bytes,n.op==Op::square};
                if(ex.team)sbn3_team_run(ex.team,product_action,&call);else serial_team_run(product_action,&call);}
            r.size=bits::normalized_size(r.data,an+bn);r.negative=r.size&&(a.negative!=b.negative);break;}
        case Op::quotient:case Op::remainder:case Op::exact_quotient:
            if(b.size<=2)divide_word(n,r,a,b,kernel,n.kernel_bytes);
            else divide_node(*reinterpret_cast<const Division*>(ex.scratch.data()+n.payload_at),n,r,a,b,kernel,ex.team);break;
        case Op::gcd:r.size=sbn3_gcd(r.data,a.data,nodes[n.left].logical,b.data,nodes[n.right].logical,kernel,n.kernel_bytes,ex.team);r.negative=0;break;
        default:fatal(SBN3_FATAL_MATH,"expression opcode");
        }
        require(r.size<=n.logical&&bits_of({r.data,r.size,r.negative})<=n.bits,SBN3_FATAL_MATH,"expression magnitude certificate");
        if(n.capacity>r.size)std::memset(r.data+r.size,0,(n.capacity-r.size)*8);
    }
    const auto&answer=nodes[h.count-1].value;auto*out=destination.begin_write();set_value(*out,answer.data,answer.size,answer.negative);destination.finish_write();
    __atomic_store_n(&h.busy,0,__ATOMIC_RELEASE);
}
}
