// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/product.h"
#include "runtime/resident.hpp"
#include "product/scoped_binding.hpp"
#include "product/build_apply.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();

namespace {
using namespace sbn::v3;
constexpr unsigned char guard=0x9b;
constexpr std::uint64_t rho_mask=UINT64_C(0xd35a87c19246f0b1);
std::uint64_t state=UINT64_C(0x5427be963c8d10af);
std::size_t completed=0;
std::uint64_t random_word(){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
std::size_t align_up(std::size_t n,std::size_t a){assert(a&&!(a&(a-1)));return (n+a-1)&~(a-1);}

struct Ref {
    ref_int value;
    Ref(){ref_init(value);}~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;
};
// All reference arithmetic happens before Team execution. Scalar products
// avoid tests/oracle's large ref_mul production-witness shortcut.
std::vector<std::uint64_t> reference(const std::vector<std::uint64_t>& a,
    const std::vector<std::uint64_t>& b,std::size_t ring,std::size_t output_words,bool identity){
    std::vector<std::uint64_t> out(output_words);
    if(identity){
        assert(b[0]==1&&std::all_of(b.begin()+1,b.end(),[](auto x){return !x;}));
        assert(a.size()<=output_words);std::copy(a.begin(),a.end(),out.begin());return out;
    }
    Ref av,product,term,low,high,modulus;
    ref_import(av.value,a.size(),-1,8,0,0,a.data());ref_set_ui(product.value,0);
    for(std::size_t j=0;j<b.size();++j)if(b[j]){
        ref_mul_ui(term.value,av.value,b[j]);if(j)ref_mul_2exp(term.value,term.value,64*j);
        ref_add(product.value,product.value,term.value);
    }
    if(ring){
        const auto bits=64*ring;
        // Modulo 2^bits-1 by exact folding; no reference division fallback.
        for(;;){
            ref_fdiv_q_2exp(high.value,product.value,bits);
            if(!ref_sgn(high.value))break;
            ref_fdiv_r_2exp(low.value,product.value,bits);ref_add(product.value,low.value,high.value);
        }
        ref_set_ui(modulus.value,1);ref_mul_2exp(modulus.value,modulus.value,bits);ref_sub_ui(modulus.value,modulus.value,1);
        if(ref_cmp(product.value,modulus.value)==0)ref_set_ui(product.value,0);
        assert(ref_cmp(product.value,modulus.value)<0);
    }
    std::size_t count=0;ref_export(out.data(),&count,-1,8,0,0,product.value);assert(count<=out.size());return out;
}

struct Config {
    const char* name;
    unsigned algorithm,workers,np=0;
    int bits=0;
    std::size_t an=0,bn=0,ring=0,correction_words=0;
    bool cached=true,separate_build=false,identity=false;
    unsigned column_log2=0,row_log2=0;
};
struct Slot {std::size_t offset=0,bytes=0;};
struct Scenario {
    Config config{};
    sbn3_mul_plan producer{},first{},correction{};
    sbn3_product_info producer_info{},first_info{},correction_info{};
    sbn3_spectrum_desc future{};
    std::vector<std::uint64_t> original_a,original_b,expected0,expected_rho,expected1;
    std::vector<unsigned char> backing;
    unsigned char* base=nullptr;
    std::size_t bytes=0,alignment=128,table_bytes=0,work_bytes=0,work_alignment=128,capacity=0;
    Slot a{},b{},rho{},result0{},result1{},cache{},tables{},work{};
    std::array<Slot*,8> slots{&a,&b,&rho,&result0,&result1,&cache,&tables,&work};
    unsigned expected_first=0,seen_first=0,seen_width=0;
    bool expect_null_arena=false,finished=false,used_loan=false;
    sbn3_product_metrics build_metrics{},apply_metrics{};

    void put(Slot& slot,std::size_t n,std::size_t align){
        if(!n)return;bytes=align_up(bytes,align);slot={bytes,n};bytes+=n+64;
    }
    std::uint64_t* words(const Slot& slot){return reinterpret_cast<std::uint64_t*>(base+slot.offset);}
    void check_guards()const {
        for(const auto* slot:slots)if(slot->bytes)
            for(std::size_t i=0;i<64;++i)assert(base[slot->offset+slot->bytes+i]==guard);
    }
    bool prepare(Config cfg,bool optional=false){
        config=cfg;sbn3_mul_options options{};options.workers=cfg.workers;options.algorithm=cfg.algorithm;
        options.prime_count=cfg.np;options.trunk_bits=cfg.bits;options.borrow_output=0;
        options.column_log2=cfg.column_log2;options.row_log2=cfg.row_log2;
        sbn3_product_request request{};request.kind=SBN3_PRODUCT_MUL;
        request.a_limbs=cfg.an;request.b_limbs=cfg.bn;request.cyclic_limbs=cfg.ring;
        allocation_watch_start();
        auto rc=sbn3_product_query(&request,&options,&producer,&producer_info);
        assert(!allocation_watch_stop());
        if(rc!=SBN3_SUPPORTED){if(optional)return false;std::fprintf(stderr,"scoped producer unsupported: %s\n",cfg.name);assert(false);}
        const auto* producer_backend=backend_lookup(producer.opaque[1]);
        assert(producer_backend&&producer_backend->product_bind_scope);
        if(cfg.cached){
            allocation_watch_start();rc=sbn3_spectrum_query(&producer,SBN3_SPECTRUM_COLUMNS,37,&future);
            assert(!allocation_watch_stop());assert(rc==SBN3_SUPPORTED);
            request.cached_a[0]=&future;
            allocation_watch_start();rc=sbn3_product_query(&request,&options,&first,&first_info);
            assert(!allocation_watch_stop());assert(rc==SBN3_SUPPORTED);
            request.b_limbs=cfg.correction_words;
            allocation_watch_start();rc=sbn3_product_query(&request,&options,&correction,&correction_info);
            assert(!allocation_watch_stop());assert(rc==SBN3_SUPPORTED);
            assert(backend_lookup(first.opaque[1])->product_bind_scope&&backend_lookup(correction.opaque[1])->product_bind_scope);
        }else{first=producer;first_info=producer_info;}
        table_bytes=std::max({producer_info.mul.table_bytes,first_info.mul.table_bytes,correction_info.mul.table_bytes});
        work_bytes=std::max({producer_info.mul.workspace_bytes,first_info.mul.workspace_bytes,correction_info.mul.workspace_bytes});
        work_alignment=std::max({std::size_t(128),producer_info.mul.workspace_alignment,
            first_info.mul.workspace_alignment,correction_info.mul.workspace_alignment});
        alignment=std::max({work_alignment,producer_info.mul.output_alignment,first_info.mul.output_alignment,
            correction_info.mul.output_alignment,std::size_t(128)});
        capacity=std::max({sbn3_mul_output_capacity(&producer_info.mul),sbn3_mul_output_capacity(&first_info.mul),
            sbn3_mul_output_capacity(&correction_info.mul)});
        put(a,cfg.an*8,64);put(b,cfg.bn*8,64);put(rho,cfg.cached?cfg.correction_words*8:0,64);
        put(result0,capacity*8,alignment);put(result1,cfg.cached?capacity*8:0,alignment);
        put(cache,cfg.cached?future.storage_bytes:0,128);put(tables,table_bytes,128);put(work,work_bytes,work_alignment);
        // Optional large test has an explicit resource ceiling, including
        // all numeric values and the retained cache, below the requested 1 GiB.
        if(bytes>(std::size_t(512)<<20)){assert(optional);return false;}
        backing.assign(bytes+alignment+64,guard);
        base=reinterpret_cast<unsigned char*>(align_up(reinterpret_cast<std::uintptr_t>(backing.data()),alignment));
        original_a.resize(cfg.an);original_b.resize(cfg.bn);
        for(auto& x:original_a)x=random_word();
        if(cfg.identity){std::fill(original_b.begin(),original_b.end(),0);original_b[0]=1;}
        else for(auto& x:original_b)x=random_word();
        std::copy(original_a.begin(),original_a.end(),words(a));std::copy(original_b.begin(),original_b.end(),words(b));
        expected0=reference(original_a,original_b,cfg.ring,first_info.mul.output_limbs,cfg.identity);
        if(cfg.cached){
            expected_rho.resize(cfg.correction_words);
            if(cfg.identity){expected_rho[0]=1;}
            else{
                assert(cfg.correction_words<=expected0.size());
                std::copy_n(expected0.begin(),cfg.correction_words,expected_rho.begin());expected_rho[0]^=rho_mask;
            }
            expected1=reference(original_a,expected_rho,cfg.ring,correction_info.mul.output_limbs,cfg.identity);
        }
        return true;
    }
};

std::uint64_t cache_digest(const sbn3_spectrum* cache,std::size_t bytes){
    const auto* p=reinterpret_cast<const unsigned char*>(cache);std::uint64_t h=1;
    // Lifecycle/refcount fields precede the immutable root tables and planes.
    for(std::size_t i=1024;i<bytes;++i)h=(h^p[i])*1099511628211ULL;return h;
}
void equal_words(const std::uint64_t* got,const std::vector<std::uint64_t>& expected){
    assert(!std::memcmp(got,expected.data(),expected.size()*8));
}
void bind(Scenario& c,const sbn3_mul_plan& plan,ScopedResidentSession& session,
    const sbn3_lease& tables,const sbn3_lease& work,const sbn3_spectrum* cache,sbn3_mul_binding** result){
    product_bind_owned(plan,*session.arena(),tables,work,*session.team(),session.scope(),cache,nullptr,result);
    assert((*result)->backend->product_bind_scope);
    assert(session.team()->busy);(void)c;
}
void run_pipeline(Scenario& c,sbn3_team_scope* scope){
    assert(scope->team->busy&&scope->first==c.expected_first);
    c.seen_first=scope->first;c.seen_width=scope->width;
    if(c.expect_null_arena)assert(!scope->team->arena);
    {
        ScopedResidentSession session({reinterpret_cast<std::byte*>(c.base),c.bytes},scope);
        auto* arena=session.arena();
        const auto tables=arena->acquire(c.tables.offset,c.tables.bytes);
        const auto work=arena->acquire(c.work.offset,c.work.bytes);
        sbn3_mul_binding* binding=nullptr;
        const sbn3_const_limbs a{c.words(c.a),c.config.an},b{c.words(c.b),c.config.bn};
        const sbn3_limbs output0{c.words(c.result0),c.capacity},output1{c.words(c.result1),c.capacity};
        if(!c.config.cached){
            bind(c,c.producer,session,tables,work,nullptr,&binding);
            const sbn3_product_inputs input{a,b,{},{}};
            sbn3_product_execute_on_scope(binding,session.scope(),&input,output0);
            equal_words(output0.data,c.expected0);
            // Also exercise the default no-scope entry while the Team is busy.
            sbn3_mul_execute(binding,a,b,output0);equal_words(output0.data,c.expected0);
            sbn3_product_get_metrics(binding,&c.apply_metrics);
            assert(c.apply_metrics.mul.worker_peak_bytes<=c.first_info.mul.per_worker_bytes);
            assert(session.team()->busy);sbn3_mul_unbind(binding);
        }else{
            const auto storage=arena->acquire(c.cache.offset,c.cache.bytes);
            sbn3_spectrum* cache=nullptr;
            sbn3_spectrum_reserve_plan(&c.producer,SBN3_SPECTRUM_COLUMNS,37,arena,&storage,&cache);
            assert(!sbn3_spectrum_can_apply(&c.first,cache,0));
            if(c.config.separate_build){
                // Includes the short PQ16 producer -> FFT owned spectrum seam.
                bind(c,c.producer,session,tables,work,nullptr,&binding);
                const sbn3_product_inputs input{a,b,{},{}};
                sbn3_product_execute(binding,&input,output0);equal_words(output0.data,c.expected0);
                sbn3_spectrum_compute(binding,cache,a);
                sbn3_product_get_metrics(binding,&c.build_metrics);sbn3_mul_unbind(binding);
                bind(c,c.first,session,tables,work,cache,&binding);
                const sbn3_product_inputs cached{{},b,{},{}};
                sbn3_product_execute(binding,&cached,output0);
            }else{
                bind(c,c.first,session,tables,work,cache,&binding);
                spectrum_compute_multiply(binding,cache,a,b,output0);
                sbn3_product_get_metrics(binding,&c.build_metrics);
            }
            assert(sbn3_spectrum_can_apply(&c.first,cache,0));equal_words(output0.data,c.expected0);
            sbn3_spectrum_desc descriptor{};sbn3_spectrum_describe(cache,&descriptor);
            const auto hash=cache_digest(cache,descriptor.storage_bytes);
            if(c.config.identity){std::fill_n(c.words(c.rho),c.config.correction_words,0);c.words(c.rho)[0]=1;}
            else{std::copy_n(output0.data,c.config.correction_words,c.words(c.rho));c.words(c.rho)[0]^=rho_mask;}
            equal_words(c.words(c.rho),c.expected_rho);
            const auto* backend=binding->backend;
            if(backend->product_scratch_bytes&&backend->with_product_scratch&&backend->product_scratch_bytes(c.first)){
                backend->with_product_scratch(binding,nullptr,[](void*,Frame& frame){
                    const auto bytes=frame.capacity();auto* p=frame.allocate(bytes,1);std::memset(p,0xa5,bytes);
                });
                c.used_loan=true;assert(hash==cache_digest(cache,descriptor.storage_bytes));
                // Verify that poisoned dead scratch is fully redefined.
                const sbn3_product_inputs cached{{},b,{},{}};sbn3_product_execute(binding,&cached,output0);
                equal_words(output0.data,c.expected0);
            }
            sbn3_product_get_metrics(binding,&c.apply_metrics);
            assert(c.apply_metrics.mul.executions&&
                c.apply_metrics.mul.worker_peak_bytes<=c.first_info.mul.per_worker_bytes);
            assert(session.team()->busy);sbn3_mul_unbind(binding);
            // The same max-sized phase pool is reused; U's cache remains live.
            bind(c,c.correction,session,tables,work,cache,&binding);
            const sbn3_product_inputs correction{{},{c.words(c.rho),c.config.correction_words},{},{}};
            sbn3_product_execute_on_scope(binding,session.scope(),&correction,output1);
            equal_words(output1.data,c.expected1);
            sbn3_product_get_metrics(binding,&c.apply_metrics);
            assert(c.apply_metrics.mul.executions&&
                c.apply_metrics.mul.worker_peak_bytes<=c.correction_info.mul.per_worker_bytes);
            assert(hash==cache_digest(cache,descriptor.storage_bytes)&&session.team()->busy);
            sbn3_mul_unbind(binding);sbn3_spectrum_release(cache);arena->release(storage);
        }
        arena->release(work);arena->release(tables);
        sbn3_arena_stats stats{};arena->stats(stats);assert(!stats.lease_references&&!stats.active_computations);
    }
    assert(scope->team->busy);equal_words(c.words(c.a),c.original_a);equal_words(c.words(c.b),c.original_b);
    c.check_guards();c.finished=true;
}

struct Launch {Scenario* scenario;std::atomic<unsigned>* arrivals=nullptr;};
void action(void* opaque,sbn3_team_scope* scope){
    auto& c=*static_cast<Launch*>(opaque);
    if(c.arrivals){c.arrivals->fetch_add(1,std::memory_order_release);while(c.arrivals->load(std::memory_order_acquire)<2){}}
    run_pipeline(*c.scenario,scope);
}
void noop(void*,sbn3_team_scope*){}
struct Pair {Launch first,second;unsigned split=2;bool only_right=false;};
void pair_action(void* opaque,sbn3_team_scope* scope){
    auto& p=*static_cast<Pair*>(opaque);
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,p.split,p.only_right?noop:action,&p.first,action,&p.second);
}

class Team {
    sbn3_arena* arena_=nullptr;sbn3_lease control_{};
public:
    sbn3_team* value=nullptr;
    Team(){
        sbn3_arena_config config{std::size_t(1)<<28,32u<<20};sbn3_error error{};
        assert(sbn3_arena_create(&config,&arena_,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena_,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena_,0,sbn3_team_storage_bytes(),&control_);
        sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& x:tc.cpu_ids)x=-1;
        assert(sbn3_team_create(arena_,&control_,&tc,&value,&error)==SBN3_OK);
    }
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena_,&control_);sbn3_arena_stats stats{};
        sbn3_arena_get_stats(arena_,&stats);assert(!stats.lease_references&&!stats.active_computations);sbn3_arena_destroy(arena_);}
    Team(const Team&)=delete;Team&operator=(const Team&)=delete;
};
void finished(Scenario& c){
    assert(c.finished);++completed;
    std::printf("scoped %s first=%u width=%u admitted=%zu loan=%u PASS\n",
        c.config.name,c.seen_first,c.seen_width,c.bytes,unsigned(c.used_loan));
}
Config p48(unsigned workers){return {"p48 cached CYC",SBN3_MUL_FLAT,workers,4,80,419,1269,1280,431};}
Config fft(unsigned workers){return {"FFT cached CYC",SBN3_MUL_PQ16,workers,0,16,127,509,512,131};}

void serial_cases(){
    Scenario p,f,short_scalar,short_u52,short_fft;
    assert(p.prepare(p48(1)));assert(f.prepare(fft(1)));
    assert(short_scalar.prepare({"short scalar",SBN3_MUL_SCALAR,1,0,0,3,4,0,0,false}));
    assert(short_u52.prepare({"short u52",SBN3_MUL_U52,1,0,0,31,29,0,0,false}));
    assert(short_fft.prepare({"short PQ16 spectrum",SBN3_MUL_PQ16,1,0,16,257,193,0,97,true,true}));
    for(auto* c:{&p,&f,&short_scalar,&short_u52,&short_fft}){
        c->expect_null_arena=true;Launch launch{c};
        allocation_watch_start();serial_team_run(action,&launch);assert(!allocation_watch_stop());finished(*c);
    }
    assert(p.used_loan);
}
void child_cases(){
    Team team;
    Scenario p,f;assert(p.prepare(p48(2)));assert(f.prepare(fft(2)));
    p.expected_first=0;f.expected_first=2;std::atomic<unsigned> arrivals{0};
    Pair pair{{&p,&arrivals},{&f,&arrivals},2,false};
    allocation_watch_start();sbn3_team_run(team.value,pair_action,&pair);assert(!allocation_watch_stop());
    assert(arrivals.load()==2);finished(p);finished(f);
    Scenario right;assert(right.prepare(p48(1)));right.expected_first=3;
    Pair one{{nullptr},{&right},3,true};
    allocation_watch_start();sbn3_team_run(team.value,pair_action,&one);assert(!allocation_watch_stop());
    assert(right.used_loan);finished(right);
    Scenario short_case;assert(short_case.prepare({"short child scope",SBN3_MUL_U52,1,0,0,73,51,0,0,false}));
    short_case.expected_first=3;one.second={&short_case};
    allocation_watch_start();sbn3_team_run(team.value,pair_action,&one);assert(!allocation_watch_stop());finished(short_case);
}
void bailey_case(){
    Scenario big;
    // Existing Bailey geometry: 64 columns x 1024 rows, T80; a full cyclic
    // transform with RHS=1 has an independent exact identity reference.
    const std::size_t ring=std::size_t(64)*1024*80/8;
    Config config{"p48 Bailey RHS1",SBN3_MUL_BAILEY,2,4,80,ring/2-3,ring-7,ring,1,true,false,true,6,10};
    if(!big.prepare(config,true)){
        std::puts("scoped Bailey optional case unavailable or exceeds 512 MiB admission cap");return;
    }
    Team team;big.expected_first=2;Pair pair{{nullptr},{&big},2,true};
    allocation_watch_start();sbn3_team_run(team.value,pair_action,&pair);assert(!allocation_watch_stop());finished(big);
}
}

int main(){
    serial_cases();child_cases();bailey_case();
    std::printf("scoped product: %zu cases; genuine active scopes, retained caches, concurrent children, phase reuse and allocation gates PASS\n",completed);
}
