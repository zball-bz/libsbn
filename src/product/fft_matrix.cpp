// Original libsbn code; LGPL-3.0-or-later.
#include "product/fft_matrix.hpp"
#include "common/identity.hpp"
#include "backend/pq16/cost_model.hpp"
#include "product/native_capabilities.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <new>
#include <cstring>
namespace sbn::v3 {
namespace {
constexpr uint64_t id=102,magic=0x53424e334d465431ULL;
struct Plan {
    uint64_t tag=magic,backend=id;
    pq16::Shape shape{};
    size_t a=0,b=0,ring=0;
    sbn3_mul_info info{};
};
static_assert(sizeof(Plan)<=sizeof(sbn3_mul_plan));
struct Prepared {Plan plan;const pq16::Tables *tables;};
Plan load(const sbn3_mul_plan &p){Plan q;memcpy(&q,&p,sizeof q);
    require(q.tag==magic&&q.backend==id&&pq16::matrix_supported(q.shape),SBN3_FATAL_ARGUMENT,"FFT matrix plan");return q;}
size_t preparation(const sbn3_mul_plan&p){return ((sizeof(Prepared)+127)&~size_t(127))+load(p).info.table_bytes+128;}
const void *prepare(const sbn3_mul_plan&p,Frame&f){auto *v=::new(f.allocate(sizeof(Prepared),128))Prepared{load(p),nullptr};v->tables=pq16::prepare(f,v->plan.shape);return v;}
size_t workspace(const sbn3_mul_plan&p,unsigned rows,size_t first){
    const auto q=load(p);const bool vector=first==matrix_vector_layout;
    if(!rows||rows>3||(!vector&&first!=SIZE_MAX&&(!q.ring||first>q.info.lbw/2)))return 0;
    return ((vector?2:4)+2*unsigned(__builtin_popcount(rows)))*(16*size_t(q.shape.nfull)+128)+4096;
}
void execute(const void *ptr,Frame&f,sbn3_team_scope*scope,const MatrixProductInputs&in){
    const auto &v=*static_cast<const Prepared*>(ptr);const auto &q=v.plan;
    require(scope&&scope->width>=q.info.workers&&in.rows&&in.rows<=3&&
            !(in.middle&&in.vector)&&(!in.negative_left||in.vector),SBN3_FATAL_ARGUMENT,"FFT matrix episode");
    require(!in.vector||(in.rows==3&&!in.left_cache&&((q.ring&&in.negative_left==6)||(!q.ring&&!in.negative_left))),SBN3_FATAL_ARGUMENT,"FFT inverse vector contract");
    require(!in.middle||q.ring,SBN3_FATAL_ARGUMENT,"FFT matrix middle ring");
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
        if(in.rows&(1u<<i))require(in.left[i][j].count<=q.a,SBN3_FATAL_ARGUMENT,"FFT matrix left support");
        if(!in.vector||!j)require(in.right[i][j].count<=q.b,SBN3_FATAL_ARGUMENT,"FFT matrix right support");
    }
    if(q.ring&&!in.middle&&!in.vector)for(unsigned i=0;i<2;++i)if(in.rows&(1u<<i))for(unsigned j=0;j<2;++j)for(unsigned k=0;k<2;++k)
        require(in.left[i][k].count+in.right[k][j].count+1<=q.ring,SBN3_FATAL_MATH,"FFT matrix linear ring support");
    sbn3_team_scope sub{scope->team,scope->first,q.info.workers,false,scope->epoch};
    pq16::matrix_execute(*v.tables,f,&sub,in,q.info.output_limbs,q.info.arithmetic_id);
}
unsigned contract(const sbn3_mul_plan&){return program_bounded_inputs|program_consume_inputs;}
}
pq16::Shape fft_matrix_shape(size_t minimum,size_t maximum) noexcept {
    pq16::Shape best{};if(!native_available()||!minimum||minimum>maximum||minimum>(1u<<17))return best;
    double cost=INFINITY;
    // One capacity-derived order per radix/codec, using the existing native
    // kernel cost model. No runtime measurements or caller-specific tuning.
    for(unsigned bits=16;bits<=20;++bits)for(unsigned m:{1u,3u,5u,7u}){
        unsigned n=128;while(size_t(m)*n*bits/32<minimum)n*=2;
        pq16::Shape s{m*n,n,m,false,pq16::Recipe::CooleyTukeyPQ,bits,false};
        if(!pq16::matrix_supported(s))s.balanced=true;
        const size_t ring=pq16::cyclic_period(s);
        if(ring<=maximum&&pq16::matrix_supported(s)){
            const double score=pq16::native_cost(s,ring/2,ring);
            if(score<cost){cost=score;best=s;}
        }
    }
    return best;
}
bool fft_matrix_query(size_t a,size_t b,size_t ring,unsigned workers,pq16::Shape s,sbn3_mul_plan&out,sbn3_mul_info&info) noexcept {
    if(!native_available()||!pq16::matrix_supported(s)||!workers||workers>32)return false;
    const size_t period=pq16::cyclic_period(s);
    if(a>period/2||b>period||(ring?ring!=period:a+b+1>period))return false;
    Plan q{};memset(&q,0,sizeof q);q.tag=magic;q.backend=id;
    q.shape.nfull=s.nfull;q.shape.branch=s.branch;q.shape.radix=s.radix;q.shape.centered=s.centered;
    q.shape.recipe=s.recipe;q.shape.bits=s.bits;q.shape.balanced=s.balanced;
    q.a=a;q.b=b;q.ring=ring;auto &i=q.info;
    // Four independent inverse/output streams. Keep their serial FFT jobs
    // in one small team; wider pointwise scattering does not create more
    // inverse parallelism and moves executable spectra between cache domains.
    i.algorithm=SBN3_MUL_PQ16;i.trunk_bits=s.bits;i.digit_words=1;i.workers=std::min(workers,4u);
    i.C=s.radix;i.M2=s.branch;i.table_entries=s.branch;
    // Completed coefficient windows use eight-digit vectors. These counts
    // are distinct from the FFT's radix/branch geometry above.
    i.lbv=i.lbw=s.nfull/4;i.transform_trunks=2*s.nfull;i.full=ring!=0;
    i.nat=(64*a+s.bits-1)/s.bits;i.nyt=(64*b+s.bits-1)/s.bits;i.output_limbs=ring?ring:a+b+1;
    const unsigned alignment=1u<<std::min(6u,unsigned(__builtin_ctz(s.bits)));
    i.emit_quantum_trunks=64/alignment;i.emit_quantum_limbs=s.bits/alignment;i.emit_limbs=period;i.emit_trunks=2*s.nfull;
    i.output_alignment=8;i.workspace_alignment=64;i.table_bytes=pq16::table_bytes(s);
    i.codec_mode=1+8*unsigned(s.balanced);
    uint64_t h=identity::word(1469598103934665603ULL,id);
    for(uint64_t x:{uint64_t(s.nfull),uint64_t(s.radix),uint64_t(s.bits),uint64_t(s.balanced),uint64_t(a),uint64_t(b),uint64_t(ring)})h=identity::word(h,x);
    i.basis_id=i.arithmetic_id=h;i.execution_id=identity::word(h,i.workers);
    i.workspace_bytes=8*(16*size_t(s.nfull)+128)+4096;
    memset(&out,0,sizeof out);memcpy(&out,&q,sizeof q);info=i;return true;
}
const Backend &fft_matrix_backend() noexcept {
    static constexpr Backend b=[] {Backend v{};v.id=id;v.name="private FFT matrix";
        v.program_bytes=preparation;v.program_prepare=prepare;v.program_contract=contract;
        v.program_matrix_bytes=workspace;v.program_matrix_execute=execute;return v;}();return b;
}
}
