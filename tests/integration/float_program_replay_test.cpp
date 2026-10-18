// Original libsbn tests; LGPL-3.0-or-later.
#include "common/query_audit.hpp"
#include <cstdio>

#if !SBN3_CHECK_SMALL
int main(){
    std::puts("float program replay: SKIP (requires a checked library and checked test)");
}
#else
#include "float/roots.hpp"
#include "float/rootn.hpp"
#include "float/power.hpp"
#include "float/division.hpp"
#include "float/short_division.hpp"
#include "float/agm.hpp"
#include "float/complex.hpp"
#include "float/exp_series.hpp"
#include "float/trig_series.hpp"
#include "float/exponential.hpp"
#include "float/logarithm.hpp"
#include "float/derived_explog.hpp"
#include "algorithms/normalized_newton.hpp"
#include "algorithms/refinement.hpp"
#include "product/local_product.hpp"
#include "product/local_windows.hpp"
#include "product/compact_windows.hpp"
#include "backend/pq16/kernels.hpp"
#include "runtime/scratch.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();

// Coverage is deliberately explicit, not a universal query interposer:
// sbn3_mul_query, sbn3_product_query, pq16::query, local_windows_query,
// peeled_windows_query, compact_window_detail::query, local_product_query.
// NOT individually intercepted: Backend::query/product_query/geometry_query;
// product_program_{choose,chosen,replay,rebatch} direct backend routes;
// leaf_windows_query, direct cyclic_shape/linear_shape geometry arithmetic;
// unrelated public constant/radix/external-service requirement functions.
// Some of these call an observed entry point, but that is not assumed here.
// This gate covers selection/lifetime only. Independent numerical suites own
// accuracy. It brackets PREPARED replay, never a complete fresh operation.
// In particular, fresh general trig range-reduction wrappers are not watched.
namespace {
constexpr const char* tags[]{"sbn3_mul_query","sbn3_product_query","pq16::query",
    "local_windows_query","peeled_windows_query","compact_window_detail::query","local_product_query"};
unsigned watching=0;
std::uint64_t query_count=0,tag_counts[7]{};
const char* first_tag=nullptr;
std::size_t checked_replays=0;
void start_queries() noexcept {
    __atomic_store_n(&query_count,0,__ATOMIC_RELAXED);__atomic_store_n(&first_tag,nullptr,__ATOMIC_RELAXED);
    for(auto& count:tag_counts)__atomic_store_n(&count,0,__ATOMIC_RELAXED);
    __atomic_store_n(&watching,1,__ATOMIC_RELEASE);
}
std::uint64_t stop_queries() noexcept {
    __atomic_store_n(&watching,0,__ATOMIC_RELEASE);return __atomic_load_n(&query_count,__ATOMIC_ACQUIRE);
}
}
extern "C" void sbn3_query_audit_observer(const char* tag) noexcept {
    if(!__atomic_load_n(&watching,__ATOMIC_ACQUIRE))return;
    __atomic_add_fetch(&query_count,1,__ATOMIC_RELAXED);
    const char* expected=nullptr;
    __atomic_compare_exchange_n(&first_tag,&expected,tag,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED);
    for(unsigned i=0;i<7;++i)if(!std::strcmp(tag,tags[i])){__atomic_add_fetch(tag_counts+i,1,__ATOMIC_RELAXED);break;}
}
namespace {
namespace f=sbn::v3::floating;
using sbn::v3::Frame;
void calibrate(){
    // Every hook must fire, including out-of-line library entry points.
    // This rejects a checked test accidentally linked to an uninstrumented
    // library instead of reporting a meaningless zero-query success.
    start_queries();
    sbn3_mul_plan plan{};sbn3_mul_info info{};sbn3_product_info product{};
    const sbn3_product_spec spec{17,19};sbn3_mul_options options{};options.workers=1;
    (void)sbn3_mul_query(&spec,&options,&plan,&info);
    sbn3_product_request request{};request.kind=SBN3_PRODUCT_SQR;request.a_limbs=17;
    (void)sbn3_product_query(&request,&options,&plan,&product);
    (void)sbn::v3::pq16::query(129,137);
    const auto shape=sbn::v3::refinement_products<sbn::v3::RefinementKind::Inverse>(65,129);
    const auto base=sbn::v3::product::local_windows_query(shape);
    (void)sbn::v3::product::peeled_windows_query(shape,base);
    (void)sbn::v3::product::compact_window_detail::query(shape);
    (void)sbn::v3::product::local_product_query(129,137,false,true);
    const auto count=stop_queries();assert(count>=7);
    for(unsigned i=0;i<7;++i){
        if(!__atomic_load_n(tag_counts+i,__ATOMIC_RELAXED))std::fprintf(stderr,"missing checked query hook: %s\n",tags[i]);
        assert(__atomic_load_n(tag_counts+i,__ATOMIC_RELAXED));
    }
}
class Team {
    sbn3_arena* arena=nullptr;sbn3_lease control{};
public:
    sbn3_team* value=nullptr;
    Team(){sbn3_error error{};sbn3_arena_config config{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&config,&arena,&error)==SBN3_OK);assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena,&control,&tc,&value,&error)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}
};
template<class Function>void replay(Team& team,const char* label,Function&& function){
    using F=std::remove_reference_t<Function>;
    struct Payload {F* function;};Payload payload{&function};
    allocation_watch_start();start_queries();
    sbn3_team_run(team.value,[](void* raw,sbn3_team_scope* scope){
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,[](void*,sbn3_team_scope*){},nullptr,
            [](void* raw_child,sbn3_team_scope* child){
                assert(sbn3_team_first_worker(child)==2&&sbn3_team_width(child)==2);
                auto& p=*static_cast<Payload*>(raw_child);(*p.function)(child);
            },raw);
    },&payload);
    const auto count=stop_queries();const auto allocations=allocation_watch_stop();
    if(count||allocations)std::fprintf(stderr,"prepared replay %s: %llu query calls (first %s), %llu allocations\n",label,
        static_cast<unsigned long long>(count),first_tag?first_tag:"none",static_cast<unsigned long long>(allocations));
    assert(!count&&!allocations);++checked_replays;
}
struct Value {
    std::vector<std::uint64_t> words;sbn::float_buffer buffer;
    explicit Value(std::size_t bits):words((bits+63)/64),buffer(words,sbn::bits{bits}){}
};
std::span<std::byte> reserve(sbn::workspace& storage,const sbn::requirements& need){
    assert(need.supported());storage.reserve(need.scratch_bytes,need.scratch_alignment);
    return storage.on().scratch.first(need.scratch_bytes);
}
void bounded(sbn::float_report report){assert(report.error.kind!=sbn::error_kind::unbounded);}
void scalar_programs(Team& team,std::size_t P){
    const std::uint64_t three=3,five=5,fifteen=15,one=1;
    const auto x=sbn::float_view::from_magnitude({&three,1},-1),a=sbn::float_view::from_magnitude({&five,1},-2);
    Value out(P);
    for(auto kind:{f::RootKind::square_root,f::RootKind::reciprocal_square_root}){
        sbn::workspace storage;const auto region=reserve(storage,f::root_program_requirements(sbn::bits{P},2,kind));
        const auto prepared=f::root_program_prepare(sbn::bits{P},2,kind,region);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"root",[&](sbn3_team_scope* scope){
            bounded(f::root_program_execute(prepared,out.buffer.ref(),x,{{},nullptr,scope}));});
    }
    const std::uint64_t degrees[]{17,UINT64_MAX};
    for(auto degree:degrees){
        sbn::workspace storage;const auto region=reserve(storage,f::rootn_program_requirements(sbn::bits{P},degree,2));
        const auto prepared=f::rootn_program_prepare(sbn::bits{P},degree,2,region);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"rootn",[&](sbn3_team_scope* scope){
            bounded(f::rootn_program_execute(prepared,out.buffer.ref(),sbn::float_view::from_magnitude({&fifteen,1},-4),{{},nullptr,scope}));});
    }
    {
        sbn::workspace storage;const auto region=reserve(storage,f::power_program_requirements(sbn::bits{P},-17,2));
        const auto prepared=f::power_program_prepare(sbn::bits{P},-17,2,region);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"signed power",[&](sbn3_team_scope* scope){
            bounded(f::power_program_execute(prepared,out.buffer.ref(),x,{{},nullptr,scope}));});
    }
    for(auto kind:{SBN3_NEWTON_DIVIDE,SBN3_NEWTON_INVERSE}){
        sbn::workspace storage;const auto region=reserve(storage,f::division_program_requirements(sbn::bits{P},2,kind));
        const auto prepared=f::division_program_prepare(sbn::bits{P},2,kind,region);
        const std::uint64_t d[]{7,11,UINT64_C(0xc000000000000003)};
        const auto denominator=sbn::float_view::from_magnitude(d,-191);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"float division",[&](sbn3_team_scope* scope){
            bounded(f::division_program_execute(prepared,out.buffer.ref(),a,denominator,{{},nullptr,scope}));});
    }
    {
        sbn::workspace storage;const auto region=reserve(storage,f::agm_program_requirements(sbn::bits{P},2));
        const auto prepared=f::agm_program_prepare(sbn::bits{P},2,region);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"AGM",[&](sbn3_team_scope* scope){
            bounded(f::agm_program_execute(prepared,out.buffer.ref(),x,a,{{},nullptr,scope}));});
    }
    for(bool negative:{false,true})for(auto kind:{f::ExpSeriesKind::exp,f::ExpSeriesKind::expm1}){
        sbn::workspace storage;const auto region=reserve(storage,f::exp_series_program_requirements(sbn::bits{P},kind,negative,2));
        const auto prepared=f::exp_series_program_prepare(sbn::bits{P},kind,negative,2,region);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"exp series",[&](sbn3_team_scope* scope){
            bounded(f::exp_series_program_execute(prepared,out.buffer.ref(),sbn::float_view::from_magnitude({&one,1},-1,negative),{{},nullptr,scope}));});
    }
    {
        Value cosine(P);sbn::workspace storage;const auto region=reserve(storage,f::trig_series_program_requirements(sbn::bits{P},2));
        const auto prepared=f::trig_series_program_prepare(sbn::bits{P},2,region);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"trig series",[&](sbn3_team_scope* scope){
            const auto reports=f::sincos_series_execute(prepared,out.buffer.ref(),cosine.buffer.ref(),sbn::float_view::from_magnitude({&one,1},-1),{{},nullptr,scope});
            bounded(reports.sine);bounded(reports.cosine);});
    }
}
void normalized(Team& team){
    using namespace sbn::v3;
    for(auto kind:{SBN3_NEWTON_DIVIDE,SBN3_NEWTON_INVERSE})for(std::size_t n:{129u,8193u}){
        const unsigned workers=n==129?1:2;NormalizedNewtonInfo info;
        assert(normalized_newton_query(kind,n,workers,info)==SBN3_SUPPORTED);
        assert(info.local==(n==129));
        sbn::workspace metadata,storage,values;metadata.reserve(info.metadata_bytes,info.alignment);
        auto mf=Frame::external(metadata.on().scratch.data(),info.metadata_bytes);
        const auto prepared=normalized_newton_prepare(kind,n,workers,mf);
        storage.reserve(info.workspace_bytes,info.alignment);values.reserve((3*n+8)*8+256,128);
        auto vf=Frame::external(values.on().scratch.data(),values.on().scratch.size());
        auto* a=vf.alloc<std::uint64_t>(n+1);auto* d=vf.alloc<std::uint64_t>(n);auto* q=vf.alloc<std::uint64_t>(n+1);
        std::fill_n(a,n,UINT64_C(0x941acf8371d52be7));a[n]=0;
        std::fill_n(d,n,UINT64_C(0xa814973b63210ecd));
        for(unsigned repeat=0;repeat<2;++repeat){
            auto wf=Frame::external(storage.on().scratch.data(),info.workspace_bytes);
            replay(team,info.local?"normalized local":"normalized native",[&](sbn3_team_scope* scope){
                normalized_newton_execute(prepared,kind==SBN3_NEWTON_DIVIDE?sbn3_const_limbs{a,n+1}:sbn3_const_limbs{},
                                          {d,n},{q,n+1},wf,scope);});
            assert(!wf.used());
        }
    }
}
void short_division(Team& team){
    for(auto kind:{SBN3_NEWTON_DIVIDE,SBN3_NEWTON_INVERSE})for(std::size_t k:{129u,32769u}){
        const auto n=2*k+3;f::ShortDivisionInfo info;
        assert(f::short_division_query(kind,n,k,2,info)==SBN3_SUPPORTED);
        sbn::workspace metadata,work;metadata.reserve(info.metadata_bytes,info.alignment);work.reserve(info.workspace_bytes,info.alignment);
        auto mf=Frame::external(metadata.on().scratch.data(),info.metadata_bytes);
        const auto prepared=f::short_division_prepare(kind,n,k,2,mf);
        std::vector<std::uint64_t> a(n,UINT64_C(0xa847b120d53e96cf)),d(k,UINT64_MAX),q(n+2);d.back()=3;
        for(unsigned repeat=0;repeat<2;++repeat){auto wf=Frame::external(work.on().scratch.data(),info.workspace_bytes);
            replay(team,info.local?"short local":"short native",[&](sbn3_team_scope* scope){
                f::short_division_execute(prepared,kind==SBN3_NEWTON_DIVIDE?sbn3_const_limbs{a.data(),a.size()}:sbn3_const_limbs{},
                    {d.data(),d.size()},{q.data(),q.size()},wf,scope);});assert(!wf.used());}
    }
}
void complex(Team& team){
    constexpr std::size_t P=257;const auto scalar=f::complex_scalar_precision(sbn::bits{P});Value real(scalar),imag(scalar);
    sbn::workspace storage;const auto region=reserve(storage,f::complex_program_requirements(sbn::bits{P},2));
    const auto prepared=f::complex_program_prepare(sbn::bits{P},2,region);
    const std::uint64_t one=1,three=3;
    const f::ScaledComplexView a{sbn::float_view::from_magnitude({&one,1},0),sbn::float_view::from_magnitude({&one,1},-2),0};
    const f::ScaledComplexView b{sbn::float_view::from_magnitude({&three,1},-1),sbn::float_view::from_magnitude({&one,1},-3,true),0};
    std::int64_t scale=0;
    for(auto kind:{f::ComplexKind::mean,f::ComplexKind::multiply,f::ComplexKind::square_root,f::ComplexKind::reciprocal})
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"complex",[&](sbn3_team_scope* scope){
            bounded(f::complex_program_execute(prepared,kind,{real.buffer.ref(),imag.buffer.ref(),&scale},a,b,{{},nullptr,scope}));});
}
void transcendental_programs(Team& team){
    constexpr std::size_t P=129;Value out(P),cosine(P);const std::uint64_t one=1,three=3;
    const auto half=sbn::float_view::from_magnitude({&one,1},-1),one_half=sbn::float_view::from_magnitude({&three,1},-1);
    {
        sbn::workspace storage;const auto region=reserve(storage,f::logarithm_program_requirements(sbn::bits{P},2));
        const auto prepared=f::logarithm_program_prepare(sbn::bits{P},2,region);
        // The first call is cold: no constant initialization is smuggled
        // outside the observed prepared-execution interval.
        replay(team,"cold AGM logarithm",[&](sbn3_team_scope* scope){
            bounded(f::logarithm_program_execute(prepared,out.buffer.ref(),one_half,f::LogarithmKind::log,{{},nullptr,scope}));});
        replay(team,"retained log1p",[&](sbn3_team_scope* scope){
            bounded(f::logarithm_program_execute(prepared,out.buffer.ref(),half,f::LogarithmKind::log1p,{{},nullptr,scope}));});
        replay(team,"local log1p",[&](sbn3_team_scope* scope){
            bounded(f::logarithm_program_execute(prepared,out.buffer.ref(),sbn::float_view::from_magnitude({&one,1},-80),
                f::LogarithmKind::log1p,{{},nullptr,scope}));});
    }
    {
        sbn::workspace storage;const auto region=reserve(storage,f::exponential_program_requirements(sbn::bits{P},2));
        const auto prepared=f::exponential_program_prepare(sbn::bits{P},2,region);
        replay(team,"cold inverse-Newton exp",[&](sbn3_team_scope* scope){
            bounded(f::exponential_program_execute(prepared,out.buffer.ref(),one_half,f::ExponentialKind::exp,{{},nullptr,scope}));});
        replay(team,"retained inverse-Newton expm1",[&](sbn3_team_scope* scope){
            bounded(f::exponential_program_execute(prepared,out.buffer.ref(),half.negated(),f::ExponentialKind::expm1,{{},nullptr,scope}));});
    }
    for(auto kind:{f::DerivedExpLogKind::exp2,f::DerivedExpLogKind::log2,f::DerivedExpLogKind::log10}){
        sbn::workspace storage;const auto region=reserve(storage,f::derived_explog_program_requirements(sbn::bits{P},2,kind));
        const auto prepared=f::derived_explog_program_prepare(sbn::bits{P},2,kind,region);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"derived exp/log",[&](sbn3_team_scope* scope){
            bounded(f::derived_explog_program_execute(prepared,out.buffer.ref(),one_half,kind,{{},nullptr,scope}));});
    }
    for(auto kind:{f::TrigAgmKind::atan,f::TrigAgmKind::reduced_sincos}){
        sbn::workspace storage;const auto region=reserve(storage,f::trig_agm_requirements(sbn::bits{P},2,kind));
        const auto prepared=f::trig_agm_prepare(sbn::bits{P},2,kind,region);
        for(unsigned repeat=0;repeat<2;++repeat)replay(team,"AGM trig",[&](sbn3_team_scope* scope){
            if(kind==f::TrigAgmKind::atan)bounded(f::atan_agm_execute(prepared,out.buffer.ref(),half,{{},nullptr,scope}));
            else{const auto reports=f::sincos_agm_execute(prepared,out.buffer.ref(),cosine.buffer.ref(),half,{{},nullptr,scope});
                bounded(reports.sine);bounded(reports.cosine);}
        });
    }
}
}
int main(){
    calibrate();Team team;
    // The 1025-bit case crosses the <=6-limb power leaf and exercises its
    // compiled nonleaf products, including inside general integer roots.
    for(std::size_t p:{257u,1025u})scalar_programs(team,p);
    normalized(team);short_division(team);complex(team);transcendental_programs(team);
    std::printf("float prepared replay: %zu checked executions on nonzero-worker scopes; zero calls at seven audited query entry points and zero ordinary allocations PASS\n",checked_replays);
}
#endif
