// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "formula_reference.hpp"
#include "external/series_merge.hpp"
#include "external/owned_product.hpp"
#include "series/arithmetic.hpp"
#include "series/precision_policy.hpp"
#include "common/m61.hpp"
#include <utility>
using namespace formula_reference;
static unsigned delayed_product_loans=0,paired_merges=0;
static unsigned resident_input_sides=0,merge_sequence=0;

static StoredInteger upload(ValueStore&s,Transport&t,StoreBuffers io,const uint64_t*p,size_t n,int64_t e=0,bool negative=false){
    StoredInteger out;RegionLease lease;allocation_watch_start();
    assert(!s.reserve(n*8,RegionKind::value,lease));ValueWriter writer;assert(!writer.open(t,lease.region(),io,&out.extent));
    assert(!writer.append({reinterpret_cast<const uint8_t *>(p),n*8}));assert(!writer.seal());
    assert(!lease.publish(out.value));assert(!allocation_watch_stop());out.extent_known=true;out.exponent2=e;out.negative=negative;return out;
}
static void load(Transport&t,StoreBuffers io,const StoredInteger&v,ref_number*out){
    std::vector<uint64_t> data(v.limbs());ValueReader reader;assert(!reader.open(t,v.value.region(),io));
    assert(!reader.read(v.first_limb*8,{reinterpret_cast<uint8_t *>(data.data()),data.size()*8}));
    ref_import(out,data.size(),-1,8,0,0,data.data());if(v.negative)ref_neg(out,out);
}
static void ownership(){
    ValueStoreStorage q{};assert(query_value_store(8,q));Aligned metadata(q.bytes),data(chunk_bytes),page(index_page_bytes),source(37),out(37);
    Device disk(16*extent_alignment);auto t=disk.transport();StoreBuffers io{data.span(),page.span()};ValueStore store;
    assert(!store.open(71,81,0,disk.bytes.size(),metadata.span(),8));fill(source.span());
    RegionLease a,b,c;allocation_watch_start();assert(!store.reserve(37,RegionKind::value,a));
    ValueWriter writer;assert(!writer.open(t,a.region(),io));assert(!writer.append(source.span()));assert(!writer.seal());
    const auto id=a.region().id;const auto first=a.region().layout.data_offset;
    StoredValue value,pin;assert(!a.publish(value));value.share(pin);value.reset();
    assert(!store.reserve(37,RegionKind::scratch,b));assert(b.region().layout.data_offset!=first);
    ValueReader reader;assert(!reader.open(t,pin.region(),io));assert(!reader.read(0,out.span()));assert(!memcmp(out.data,source.data,37));
    pin.reset();assert(!store.reserve(37,RegionKind::scratch,c));
    assert(c.region().layout.data_offset==first&&!(c.region().id==id));
    assert(reader.read(0,out.span()));b.reset();c.reset();
    assert(store.usage().live_regions==0&&store.usage().largest_free==disk.bytes.size());
    // Empty values own metadata, but neither consume nor fragment backing.
    assert(!store.reserve(0,RegionKind::value,a));assert(!writer.open(t,a.region(),io));assert(!writer.seal());
    assert(!a.publish(value));value.reset();assert(store.usage().largest_free==disk.bytes.size());
    assert(store.reserve(disk.bytes.size(),RegionKind::value,a).code==IoCode::capacity); // index extent also needs backing
    assert(!a&&store.usage().live_regions==0);
    RegionLease all[8];for(auto &lease:all)assert(!store.reserve(0,RegionKind::scratch,lease));
    assert(store.reserve(0,RegionKind::scratch,a).code==IoCode::capacity);
    for(auto &lease:all)lease.reset();
    assert(!allocation_watch_stop());
    puts("Value ownership: immutable pins, stale readers, reuse identities, coalescing and metadata bounds PASS");
}
static void sums(){
    ValueStoreStorage q{};assert(query_value_store(16,q));Aligned metadata(q.bytes),data(4*chunk_bytes),page(index_page_bytes);
    Device disk(8u<<20);auto t=disk.transport();StoreBuffers io{data.span(),page.span()};ValueStore store;
    assert(!store.open(72,82,0,disk.bytes.size(),metadata.span(),16));
    for(unsigned mode=0;mode<12;++mode){
        const size_t an=mode<4?7:13011,bn=mode<4?5:9013;
        std::vector<uint64_t>a(an),b(bn);
        for(auto&v:a)v=mode==6?UINT64_MAX:random_word();for(auto&v:b)v=mode==6?1:random_word();
        const int64_t ae=mode<4?0:-79,be=mode<4?63:int64_t(mode*19);
        auto x=upload(store,t,io,a.data(),a.size(),ae,mode&1),y=upload(store,t,io,b.data(),b.size(),be,mode&2);
        StoredSumPlan plan{};assert(query_stored_sum(an,ae,bn,be,plan));Aligned work(plan.storage_bytes);StoredInteger z;
        const auto before=t.counters;allocation_watch_start();assert(!run_stored_sum(plan,t,store,x,y,z,work.span()));assert(!allocation_watch_stop());
        assert(t.counters.write_submitted-before.write_submitted==plan.write_bytes);
        ref_int want,term,got;ref_inits(want,term,got,nullptr);
        ref_import(want,an,-1,8,0,0,a.data());ref_import(term,bn,-1,8,0,0,b.data());
        if(x.negative)ref_neg(want,want);if(y.negative)ref_neg(term,term);
        ref_mul_2exp(want,want,size_t(ae-plan.exponent));ref_mul_2exp(term,term,size_t(be-plan.exponent));ref_add(want,want,term);
        load(t,io,z,got);assert(!ref_cmp(want,got)&&z.exponent2==plan.exponent);
        ref_clears(want,term,got,nullptr);
    }
    assert(!store.usage().live_regions&&store.usage().largest_free==disk.bytes.size());
    // Exact cancellation and an IO failure must not leak either output.
    uint64_t one=1;auto x=upload(store,t,io,&one,1),y=upload(store,t,io,&one,1,0,true);StoredInteger z;
    StoredSumPlan p{};assert(query_stored_sum(1,0,1,0,p));Aligned work(p.storage_bytes);
    assert(!run_stored_sum(p,t,store,x,y,z,work.span()));assert(!z.negative);z.value.reset();
    const auto held=store.usage().live_bytes;disk.fail_write=disk.writes+1;
    assert(run_stored_sum(p,t,store,x,y,z,work.span()).code==IoCode::io);
    assert(!z.value&&store.usage().live_bytes==held);disk.fail_write=0;
    auto zero=upload(store,t,io,nullptr,0,INT64_MIN);
    assert(query_stored_sum(0,INT64_MIN,1,0,p)&&p.result_limbs==2&&p.exponent==0);
    assert(!run_stored_sum(p,t,store,zero,x,z,work.span()));ref_int got;ref_init(got);load(t,io,z,got);assert(!ref_cmp_ui(got,1));ref_clear(got);
    puts("Stored integers: arbitrary binary shifts, signed multi-batch sum/subtract, cancellation and failure cleanup PASS");
}
static void pipelined_sums(){
    constexpr size_t n=(1u<<18)+29;
    for(unsigned pattern=0;pattern<7;++pattern){
        ValueStoreStorage q{};assert(query_value_store(16,q));Aligned metadata(q.bytes),data(1u<<20),page(index_page_bytes);
        Device disk(24u<<20);auto t=disk.transport();StoreBuffers io{data.span(),page.span()};ValueStore store;
        assert(!store.open(781+pattern,881+pattern,0,disk.bytes.size(),metadata.span(),16));
        std::vector<uint64_t>a(n),b(n);for(auto&v:a)v=random_word();for(auto&v:b)v=random_word();
        bool negative=pattern==1||pattern==3||pattern==4||pattern==5;
        const int64_t ae=pattern==1?13:0,be=pattern==1?-37:0;
        if(pattern==2){std::fill(a.begin(),a.end(),UINT64_MAX);b.assign(1,1);}
        if(pattern==3){a.assign(n+1,0);a.back()=1;b.assign(1,1);}
        if(pattern==4)b=a;
        if(pattern==5){a.assign(1,1);b.assign(n+1,0);b.back()=1;}
        auto x=upload(store,t,io,a.data(),a.size(),ae),y=upload(store,t,io,b.data(),b.size(),be,negative);
        StoredSumPlan plan;assert(query_stored_sum(a.size(),ae,b.size(),be,plan,0,4)&&plan.workers>1);
        if(pattern==6)assert(project_stored_sum(plan,{n/4,n/2,0}));
        Fixture fixture(4,false,32u<<20);auto work=fixture.allocate(plan.storage_bytes,4096);StoredInteger out;
        const auto before=t.counters;allocation_watch_start();
        const auto bytes=std::span(static_cast<uint8_t*>(work.data),work.bytes);
        if(pattern&1){
            auto action=[&](sbn3_team_scope*scope){assert(!run_stored_linear(plan,t,store,linear_input(x),linear_input(y),out,bytes,nullptr,scope));};
            sbn3_team_run(fixture.team,[](void*ptr,sbn3_team_scope*scope){(*static_cast<decltype(action)*>(ptr))(scope);},&action);
        }else assert(!run_stored_linear(plan,t,store,linear_input(x),linear_input(y),out,bytes,fixture.team));
        assert(!allocation_watch_stop()&&t.counters.write_submitted-before.write_submitted==plan.write_bytes);
        ref_int expected,rhs,got;ref_inits(expected,rhs,got,nullptr);
        ref_import(expected,a.size(),-1,8,0,0,a.data());ref_import(rhs,b.size(),-1,8,0,0,b.data());
        ref_mul_2exp(expected,expected,size_t(ae-plan.exponent));ref_mul_2exp(rhs,rhs,size_t(be-plan.exponent));
        if(negative)ref_sub(expected,expected,rhs);else ref_add(expected,expected,rhs);
        const bool sign=ref_sgn(expected)<0;ref_abs(expected,expected);
        ref_fdiv_q_2exp(expected,expected,plan.first_output_limb*64);ref_fdiv_r_2exp(expected,expected,plan.output_bytes*8);
        load(t,io,out,got);ref_abs(got,got);assert(!ref_cmp(got,expected)&&out.negative==sign);
        assert(out.extent.residue_m61==ref_fdiv_ui(expected,sbn::v3::m61::modulus));
        ref_clears(expected,rhs,got,nullptr);out.value.reset();const auto live=store.usage().live_bytes;
        for(unsigned failure=0;failure<3;++failure){
            if(failure==0)disk.fail_read=disk.reads+3;
            if(failure==1)disk.fail_write=disk.writes+2;
            if(failure==2)disk.cancel_after_write=disk.writes+1;
            const auto e=run_stored_linear(plan,t,store,linear_input(x),linear_input(y),out,{static_cast<uint8_t*>(work.data),work.bytes},fixture.team);
            assert(e.code==(failure==2?IoCode::cancelled:IoCode::io)&&!out.value&&store.usage().live_bytes==live);
            disk.fail_read=disk.fail_write=disk.cancel_after_write=0;disk.cancelled=false;
        }
    }
    puts("Pipelined linear arithmetic: exact shifts/signs/projection, cross-bank long carries/borrows, m61 and joined IO-failure cleanup PASS");
}
static StoredTriple subtree(const FiniteFormula&formula,uint64_t begin,uint64_t end,unsigned need,
                           Fixture&fixture,ValueStore&store,Transport&t,StoreBuffers io){
    FinitePlan plan{};sbn3_series_options options{1,8,0,0,0};
    assert(finite_query_prefix(formula,{begin,end},need,options,0,plan)==SBN3_SUPPORTED);
    const auto meta=fixture.allocate(plan.info.plan_bytes,plan.info.plan_alignment);
    const auto prepared=fixture.allocate(plan.info.prepared_bytes,plan.info.prepared_alignment);
    const auto scratch=fixture.allocate(plan.info.workspace_bytes,plan.info.workspace_alignment);
    std::array<std::unique_ptr<Aligned>,3> data;StoredTriple out;sbn3_series_values v{};
    for(unsigned j=0;j<3;++j)if(need&(1u<<j)){
        data[j]=std::make_unique<Aligned>(plan.info.output.limbs[j]*8);
        v.value[j].mantissa={reinterpret_cast<uint64_t *>(data[j]->data),plan.info.output.limbs[j],0,0};
    }
    FiniteBinding binding;allocation_watch_start();finite_prepare(plan,*fixture.arena,*fixture.team,meta,prepared,scratch,binding);
    finite_execute(binding,v);assert(!allocation_watch_stop());
    for(unsigned j=0;j<3;++j)if(need&(1u<<j))out[j]=upload(store,t,io,v.value[j].mantissa.data,v.value[j].mantissa.size,v.value[j].exponent2,v.value[j].mantissa.negative);
    return out;
}
static StoredTriple subtree(const DataFormula&f,uint64_t begin,uint64_t end,unsigned need,
                           Fixture&fixture,ValueStore&store,Transport&t,StoreBuffers io){
    return subtree(data_finite_formula(f),begin,end,need,fixture,store,t,io);
}
static StoredTriple merge(sbn3_series_recipe recipe,unsigned need,StoredTriple&a,StoredTriple&b,
                         Fixture&f,ValueStore&store,Transport&t,std::span<uint8_t>work,uint64_t &peak_bound,size_t ram=8u<<20){
    std::array<IntegerShape,6> shape{};uint64_t input=0;
    for(unsigned j=0;j<6;++j){auto&v=j<3?a[j]:b[j-3];shape[j]={v.limbs(),v.exponent2,v.source()};if(v.value)input+=v.value.region().layout.allocation_bytes;}
    planning::ExternalProductBudget budget{ram,4u<<20,32u<<20,store.usage().live_bytes-input};
    StoredMergePlan p{};allocation_watch_start();const auto rc=query_stored_merge(recipe,need,shape,1,budget,{},p);
    assert(!allocation_watch_stop());
    if(rc!=planning::ExternalPlanStatus::ready)fprintf(stderr,"stored merge query status %u\n",unsigned(rc));
    assert(rc==planning::ExternalPlanStatus::ready&&p.storage_bytes<=work.size());
    // Exercise both incoming sides independently of whole-tree admission.
    // Uploaded values are only fixture setup; replace one child by the exact
    // resident words before entering the consuming merge.
    std::array<ResidentInteger,6> loans{};const unsigned side=(merge_sequence++%2)*3;unsigned mask=0;
    std::array<size_t,6> bytes{};
    for(unsigned i=side;i<side+3;++i){const auto&v=i<3?a[i]:b[i-3];
        if(p.uses[i]&&v.value&&v.limbs()){assert(v.first_limb==0);mask|=1u<<i;bytes[i]=v.value.region().layout.logical_bytes;}
    }
    paired_merges+=p.pair_first>=0;peak_bound=std::max(peak_bound,p.peak_backing_bytes);
    const size_t limit=std::min(work.size(),ram-budget.other_live_bytes-p.shared_bytes)&~(p.alignment-1);
    auto resident=p;
    if(mask&&lend_stored_merge_inputs(resident,mask,bytes,limit,budget,{})){
        Aligned data(4*chunk_bytes),page(index_page_bytes);StoreBuffers io{data.span(),page.span()};
        for(unsigned i=side;i<side+3;++i)if(mask&(1u<<i)){
            auto&v=i<3?a[i]:b[i-3];ValueReader reader;assert(!reader.open(t,v.value.region(),io));
            auto words=work.subspan(resident.input_offsets[i],bytes[i]);assert(!reader.read(0,words));
            loans[i]={{reinterpret_cast<const uint64_t *>(words.data()),words.size()/8},v.extent,v.exponent2,v.negative};
            v.value.reset();
        }
        p=resident;resident_input_sides|=1u<<(side/3);
    }
    for(unsigned i=0;i<p.stage_count;++i)if(p.stage[i].borrow_result&&i+1<p.stage_count&&
        p.stage[i+1].kind==StoredMergeStage::Kind::product)++delayed_product_loans;
    if(ram==16u<<20)assert(p.cached_pair||std::any_of(p.stage,p.stage+p.stage_count,[](const auto&s){return s.use_resident;}));
    paired_merges+=p.pair_first>=0;peak_bound=std::max(peak_bound,p.peak_backing_bytes);
    const auto before=t.counters;StoredTriple out;allocation_watch_start();assert(!run_stored_merge(p,t,store,a,b,out,work,*f.team,
        p.resident_inputs?std::span<const ResidentInteger>(loans):std::span<const ResidentInteger>{}));assert(!allocation_watch_stop());
    assert(t.counters.write_submitted-before.write_submitted==p.normal_write_bytes);
    assert(store.usage().peak_bytes<=peak_bound);
    for(const auto*v:{&a,&b})for(const auto&x:*v)assert(!x.value);
    return out;
}
static void series_case(const char*name,const FormulaDef&def,size_t ram=8u<<20){
    constexpr uint64_t span=149;const uint64_t start=def.begin,end=start+span;
    auto data=std::make_unique<DataFormula>();assert(data_formula_prepare(def,end,*data));
    const unsigned need=def.recipe==SBN3_SERIES_COMMON_P2B3?7:3;
    ValueStoreStorage q{};assert(query_value_store(64,q));Aligned metadata(q.bytes),io_data(4*chunk_bytes),page(index_page_bytes);
    Device disk(32u<<20);auto t=disk.transport();StoreBuffers io{io_data.span(),page.span()};ValueStore store;
    assert(!store.open(73,83,0,disk.bytes.size(),metadata.span(),64));
    // The query may spend available RAM on delayed result lifetimes. Supply
    // the declared working grant instead of the old incidental 3/4 MiB bound.
    Fixture f(1,false,32u<<20);const auto memory=f.allocate(ram-(4u<<20),4096);auto work=std::span(static_cast<uint8_t *>(memory.data),memory.bytes);
    uint64_t peak_bound=0;
    auto a=subtree(*data,start,start+17,need,f,store,t,io),b=subtree(*data,start+17,start+41,need,f,store,t,io);
    auto ab=merge(def.recipe,need,a,b,f,store,t,work,peak_bound,ram);
    auto c=subtree(*data,start+41,start+83,need,f,store,t,io),d=subtree(*data,start+83,end,need,f,store,t,io);
    auto cd=merge(def.recipe,need,c,d,f,store,t,work,peak_bound,ram);
    auto result=merge(def.recipe,need,ab,cd,f,store,t,work,peak_bound,ram);
    Ref expected;reference(data->def,start,end,expected);ref_int got;ref_init(got);
    const ref_number*want[]{expected.t,expected.d,expected.u};
    for(unsigned j=0;j<3;++j)if(need&(1u<<j)){
        load(t,io,result[j],got);assert(!ref_cmp(got,want[j]));
        assert(result[j].exponent2==(j==0&&def.recipe==SBN3_SERIES_BINARY_BBP?expected.exponent:0));
        result[j].value.reset();
    }
    ref_clear(got);assert(!store.usage().live_regions&&store.usage().largest_free==disk.bytes.size());
    printf("Native leaves -> stored exact %-10s: two-level merge, input consumption, exact writes and reference result PASS\n",name);
}
static void merge_failures(){
    ValueStoreStorage q{};assert(query_value_store(48,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(16u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(74,84,0,disk.bytes.size(),metadata.span(),48));
    Fixture f(1,false,32u<<20);const auto memory=f.allocate(3u<<20,4096);const auto work=std::span(static_cast<uint8_t *>(memory.data),memory.bytes);
    for(unsigned mode=0;mode<5;++mode){
        uint64_t a=2,b=3,c=5,d=7;StoredTriple left,right,out;StoredValue pin;
        left[0]=upload(store,t,io,&a,mode?1:0);left[1]=upload(store,t,io,&b,1);
        right[0]=upload(store,t,io,&c,1);right[1]=upload(store,t,io,&d,1);
        std::array<IntegerShape,6> shapes{};for(unsigned i=0;i<6;++i){const auto&v=i<3?left[i]:right[i-3];shapes[i]={v.limbs(),v.exponent2};}
        StoredMergePlan p{};assert(query_stored_merge(SBN3_SERIES_HYPERDESCENT,3,shapes,1,{8u<<20,4u<<20,16u<<20},{},p,mode?2:0)==planning::ExternalPlanStatus::ready);
        if(mode)left[1].value.share(pin);
        if(mode==1)disk.fail_write=disk.writes+1;
        if(mode==2)disk.cancel_after_write=disk.writes+1;
        if(mode==3)t.write_limit=t.counters.write_submitted+p.normal_write_bytes-1;
        allocation_watch_start();const auto error=run_stored_merge(p,t,store,left,right,out,work,*f.team);assert(!allocation_watch_stop());
        if(!mode||mode==4){
            assert(!error);ref_int got;ref_init(got);load(t,io,out[0],got);assert(ref_cmp_ui(got,mode==4?19:5)==0);
            load(t,io,out[1],got);assert(ref_cmp_ui(got,21)==0);ref_clear(got);
            assert(store.usage().peak_bytes<=p.peak_backing_bytes);
        }else{
            assert(error.code==(mode==1?IoCode::io:mode==2?IoCode::cancelled:IoCode::capacity));
            for(const auto&v:out)assert(!v.value);
            if(mode==3){assert(left[0].value&&right[0].value);} // refusal precedes take
            else assert(store.usage().live_regions==1); // only the checkpoint/read pin
            disk.fail_write=disk.cancel_after_write=0;disk.cancelled=false;t.write_limit=UINT64_MAX;
            ValueReader reader;assert(!reader.open(t,pin.region(),io));uint64_t word=0;
            assert(!reader.read(0,{reinterpret_cast<uint8_t *>(&word),8}));assert(word==3);
        }
    }
    assert(!store.usage().live_regions&&store.usage().largest_free==disk.bytes.size());
    puts("Stored merge: zero products, pre-take write refusal, in-flight IO/cancel cleanup and surviving pins PASS");
}
static void window_product(){
    constexpr size_t an=70001,bn=62003,af=17,bf=2051,ac=20007,bc=17001;
    ValueStoreStorage q{};assert(query_value_store(32,q));Aligned metadata(q.bytes),data(4*chunk_bytes),page(index_page_bytes);
    Device disk(16u<<20);auto t=disk.transport();StoreBuffers io{data.span(),page.span()};ValueStore store;
    assert(!store.open(75,85,0,disk.bytes.size(),metadata.span(),32));
    std::vector<uint64_t>a(an),b(bn);for(auto&v:a)v=random_word();for(auto&v:b)v=random_word();
    std::fill_n(a.data()+af,3,0);std::fill_n(b.data()+bf,4,0);a[af+3]|=1;b[bf+4]|=1;
    a[af+ac-1]=a[af+ac-2]=b[bf+bc-1]=b[bf+bc-2]=0;
    auto av=upload(store,t,io,a.data(),an),bv=upload(store,t,io,b.data(),bn);StoredInteger result;
    SpilledProductSpec spec{20011,17013,4,2048,1024,6,128,3,4,false,6*chunk_bytes};
    spec.a_source={an,0,spec.an};spec.b_source={bn,0,spec.bn};SpilledProductPlan p{};assert(query_spilled_product(spec,p));
    const auto original=p;assert(rebind_spilled_sources(p,{an,af,ac},{bn,bf,bc}));
    assert(p.storage_bytes==original.storage_bytes&&p.pool==original.pool&&p.spec.trunk_bits==original.spec.trunk_bits);
    auto rejected=p;assert(!rebind_spilled_sources(rejected,{an+1,af,ac},{bn,bf,bc}));
    assert(rejected.forward[0].source_capacity_bytes==p.forward[0].source_capacity_bytes&&rejected.normal_read_bytes==p.normal_read_bytes);
    Fixture f(4,false,32u<<20);auto memory=f.allocate(p.storage_bytes,4096);const auto before=t.counters;
    allocation_watch_start();assert(!run_owned_product(p,t,store,av.value,bv.value,result.value,
        {static_cast<uint8_t *>(memory.data),memory.bytes},*f.team,&result.extent));assert(!allocation_watch_stop());result.extent_known=true;
    assert(t.counters.read_submitted-before.read_submitted==p.normal_read_bytes&&t.counters.write_submitted-before.write_submitted==p.normal_write_bytes);
    std::vector<uint64_t> product(p.output_bytes/8);ValueReader reader;assert(!reader.open(t,result.value.region(),io));
    assert(!reader.read(0,{reinterpret_cast<uint8_t *>(product.data()),product.size()*8}));
    assert(ref_product_equal(a.data()+af,ac,b.data()+bf,bc,product.data(),ac+bc));
    for(size_t i=ac+bc;i<product.size();++i)assert(!product[i]);
    size_t high=product.size();while(high&&!product[high-1])--high;assert(result.extent.first==7&&result.extent.end==high);
    const auto counts=t.counters;assert(limit_stored_integer(result,31));assert(result.limbs()==31&&result.first_limb==high-31);
    assert(t.counters.read_submitted==counts.read_submitted&&t.counters.write_submitted==counts.write_submitted);
    ref_int got,want;ref_inits(got,want,nullptr);load(t,io,result,got);ref_import(want,31,-1,8,0,0,product.data()+high-31);assert(!ref_cmp(got,want));ref_clears(got,want,nullptr);
    auto projected=p;const size_t first=(high-31)&~(chunk_bytes/8-1);
    assert(project_spilled_output(projected,first,high));StoredInteger smaller;const auto prior=t.counters;
    allocation_watch_start();assert(!run_owned_product(projected,t,store,av.value,bv.value,smaller.value,
        {static_cast<uint8_t *>(memory.data),memory.bytes},*f.team,&smaller.extent));assert(!allocation_watch_stop());
    assert(t.counters.read_submitted-prior.read_submitted==projected.normal_read_bytes&&
           t.counters.write_submitted-prior.write_submitted==projected.normal_write_bytes);
    assert(projected.normal_write_bytes<p.normal_write_bytes&&smaller.value.region().layout.logical_bytes==(high-first)*8);
    ref_inits(got,want,nullptr);load(t,io,smaller,got);ref_import(want,high-first,-1,8,0,0,product.data()+first);assert(!ref_cmp(got,want));ref_clears(got,want,nullptr);
    printf("Projected product output: %llu -> %llu stored bytes; %llu fewer writes, same full m61/carry PASS\n",
        (unsigned long long)p.stored_output_bytes,(unsigned long long)projected.stored_output_bytes,
        (unsigned long long)(p.normal_write_bytes-projected.normal_write_bytes));
    puts("Stored product windows: unaligned edges, fixed transform, zero extension, exact IO and output extent PASS");
}
static void sparse_limit(){
    ValueStoreStorage q{};assert(query_value_store(8,q));Aligned metadata(q.bytes),data(4*chunk_bytes),page(index_page_bytes);
    Device disk(2u<<20);auto t=disk.transport();StoreBuffers io{data.span(),page.span()};ValueStore store;
    assert(!store.open(77,87,0,disk.bytes.size(),metadata.span(),8));
    uint64_t sparse[10]{1,0,0,0,0,0,0,0,9,0},one=1;
    auto x=upload(store,t,io,sparse,10);const auto before=t.counters;
    assert(limit_stored_integer(x,4)&&x.first_limb==5&&x.limbs()==4&&x.exponent2==320);
    assert(t.counters.read_submitted==before.read_submitted&&t.counters.write_submitted==before.write_submitted);
    auto y=upload(store,t,io,&one,1,512,true);StoredInteger z;StoredSumPlan p{};
    assert(query_stored_sum(x.limbs(),x.exponent2,y.limbs(),y.exponent2,p,2));Aligned work(p.storage_bytes);
    assert(!run_stored_sum(p,t,store,x,y,z,work.span()));ref_int got,want;ref_inits(got,want,nullptr);
    load(t,io,z,got);ref_set_ui(want,8);const int64_t low=std::min(z.exponent2,int64_t(512));
    ref_mul_2exp(got,got,size_t(z.exponent2-low));ref_mul_2exp(want,want,size_t(512-low));assert(!ref_cmp(got,want));ref_clears(got,want,nullptr);
    puts("Sparse retained window: low zero words, signed cancellation and limited alignment PASS");
}
static uint64_t decay(const void*p,uint64_t at){return static_cast<const DataFormula *>(p)->attenuation_bits(at);}
static void resident_state(const PsrSpec&s,Ref&expected,int64_t (&exponents)[2]){
    PsrInfo info{};assert(psr_query(s,info)==SBN3_SUPPORTED);Fixture f(1,false,32u<<20);
    const size_t offset=up(f.base+f.cursor,info.storage_alignment)-f.base;f.cursor=up(offset+info.storage_bytes,4096)+4096;
    sbn3_error e{};assert(sbn3_arena_prepare(f.arena,offset,info.storage_bytes,&e)==SBN3_OK);
    auto memory=f.allocate(info.output_limbs*16,128);sbn3_series_values values{};
    for(unsigned j=0;j<2;++j)values.value[j].mantissa={static_cast<uint64_t *>(memory.data)+j*info.output_limbs,info.output_limbs,0,0};
    PsrBinding binding;allocation_watch_start();psr_prepare(s,info,*f.arena,*f.team,offset,binding);psr_execute(binding,values);psr_release(binding);assert(!allocation_watch_stop());
    ref_number *result[]{expected.t,expected.d};
    for(unsigned j=0;j<2;++j){const auto&v=values.value[j];ref_import(result[j],v.mantissa.size,-1,8,0,0,v.mantissa.data);if(v.mantissa.negative)ref_neg(result[j],result[j]);exponents[j]=v.exponent2;}
}
static void precision_case(const char*label,const FormulaDef&def,uint64_t terms,size_t precision,uint64_t max_terms=83){
    StoredMergePlan invalid{};
    assert(query_stored_merge(SBN3_SERIES_COMMON_P2B3,7,{},1,{},{},invalid,0,precision)==planning::ExternalPlanStatus::invalid);
    auto data=std::make_unique<DataFormula>();const uint64_t end=def.begin+terms;assert(data_formula_prepare(def,end,*data));
    const PsrSpec spec{data_finite_formula(*data),{def.begin,end},{1,8,0,0,0},{data.get(),decay,4,max_terms,0},precision};
    Ref expected;int64_t exponents[2];resident_state(spec,expected,exponents);
    std::vector<PrecisionBlock> blocks;for(uint64_t at=def.begin;at<end;){PrecisionBlock b{};assert(precision_block_query(spec,at,b)==SBN3_SUPPORTED);blocks.push_back(b);at=b.range.end;}
    ValueStoreStorage q{};assert(query_value_store(48,q));Aligned metadata(q.bytes),io_data(4*chunk_bytes),page(index_page_bytes);
    Device disk(24u<<20);auto t=disk.transport();StoreBuffers io{io_data.span(),page.span()};ValueStore store;
    assert(!store.open(76,86,0,disk.bytes.size(),metadata.span(),48));
    Fixture f(1,false,32u<<20);const auto memory=f.allocate(3u<<20,4096);const auto work=std::span(static_cast<uint8_t *>(memory.data),memory.bytes);StoredTriple tail;
    for(size_t i=blocks.size();i-- >0;){const auto&b=blocks[i];const auto formula=precision_formula(spec,b.exact_words?&b.exact_words:nullptr);
        auto finite=subtree(formula,b.range.begin,b.range.end,b.need,f,store,t,io);
        const size_t keep=b.precision+precision_guard_words;
        if(!b.next_precision){
            for(unsigned j=0;j<2;++j){tail[j]=std::move(finite[j]);assert(limit_stored_integer(tail[j],keep));}
        }else{
            std::array<IntegerShape,6> shape{};for(unsigned j=0;j<6;++j){const auto&v=j<3?finite[j]:tail[j-3];shape[j]={v.limbs(),v.exponent2,v.source()};}
            StoredMergePlan plan{};allocation_watch_start();assert(query_stored_merge(def.recipe,3,shape,1,{8u<<20,4u<<20,24u<<20},{},plan,0,keep)==planning::ExternalPlanStatus::ready);assert(!allocation_watch_stop());
            assert(plan.storage_bytes<=work.size());const auto before=t.counters;StoredTriple out;
            allocation_watch_start();assert(!run_stored_merge(plan,t,store,finite,tail,out,work,*f.team));assert(!allocation_watch_stop());
            assert(t.counters.write_submitted-before.write_submitted<=plan.normal_write_bytes);tail=std::move(out);
        }
        assert(tail[0].limbs()<=keep&&tail[1].limbs()<=keep);
    }
    ref_int got,want;ref_inits(got,want,nullptr);const ref_number *reference[]{expected.t,expected.d};
    for(unsigned j=0;j<2;++j){load(t,io,tail[j],got);ref_set(want,reference[j]);const int64_t low=std::min(tail[j].exponent2,exponents[j]);
        ref_mul_2exp(got,got,size_t(tail[j].exponent2-low));ref_mul_2exp(want,want,size_t(exponents[j]-low));
        if(ref_cmp(got,want))fprintf(stderr,"limited %s component %u disagrees\n",label,j);
        assert(!ref_cmp(got,want));
    }
    ref_clears(got,want,nullptr);printf("Stored precision %-10s: %zu blocks, N/D equal resident PSR, no copied windows PASS\n",label,blocks.size());
}
struct ScaledReference {
    ref_int value;
    int64_t exponent=0;
    ScaledReference(){ref_init(value);}
    ~ScaledReference(){ref_clear(value);}
    ScaledReference(const ScaledReference &)=delete;
};
static size_t reference_words(const ScaledReference&r){return ref_sgn(r.value)?(ref_sizeinbase(r.value,2)+63)/64:0;}
static void reference_limit(ScaledReference&r,size_t keep){
    const size_t n=reference_words(r);if(n>keep){ref_tdiv_q_2exp(r.value,r.value,64*(n-keep));r.exponent+=int64_t(64*(n-keep));}
}
static void reference_mul(ScaledReference&r,const ScaledReference&a,const ScaledReference&b,size_t keep){
    ref_mul(r.value,a.value,b.value);r.exponent=a.exponent+b.exponent;reference_limit(r,keep);
}
static void reference_add(ScaledReference&r,ScaledReference&a,ScaledReference&b,size_t keep){
    const int64_t high=std::max(a.exponent+int64_t(reference_words(a)*64),b.exponent+int64_t(reference_words(b)*64));
    const int64_t low=std::max(std::min(a.exponent,b.exponent),high-int64_t((keep+1)*64));
    for(auto*x:{&a,&b}){
        if(x->exponent<low)ref_tdiv_q_2exp(x->value,x->value,size_t(low-x->exponent));
        else ref_mul_2exp(x->value,x->value,size_t(x->exponent-low));
        x->exponent=low;
    }
    ref_add(r.value,a.value,b.value);r.exponent=low;reference_limit(r,keep);
}
static void large_precision_window(){
    constexpr size_t P=5001;const size_t n[]{4991,4957,4993,4979,4921};
    ValueStoreStorage q{};assert(query_value_store(48,q));Aligned metadata(q.bytes),data(4*chunk_bytes),page(index_page_bytes);
    Device disk(16u<<20);auto t=disk.transport();StoreBuffers io{data.span(),page.span()};ValueStore store;
    assert(!store.open(78,88,0,disk.bytes.size(),metadata.span(),48));
    StoredTriple left,right,out;ScaledReference input[5],x,y,numerator,denominator;
    for(unsigned i=0;i<5;++i){std::vector<uint64_t> raw(n[i]);for(auto&v:raw)v=random_word();raw.back()|=uint64_t(1)<<63;
        auto value=upload(store,t,io,raw.data(),raw.size(),0,i==3);
        if(i<3)left[i]=std::move(value);else right[i-3]=std::move(value);
        ref_import(input[i].value,raw.size(),-1,8,0,0,raw.data());if(i==3)ref_neg(input[i].value,input[i].value);
    }
    reference_mul(y,input[2],input[3],P+1);reference_limit(input[0],P);
    reference_mul(x,input[0],input[4],P+1);reference_add(numerator,x,y,P);
    reference_mul(denominator,input[1],input[4],P);
    std::array<IntegerShape,6> shape{};for(unsigned i=0;i<6;++i){const auto&v=i<3?left[i]:right[i-3];shape[i]={v.limbs(),v.exponent2,v.source()};}
    StoredMergePlan plan{};assert(query_stored_merge(SBN3_SERIES_COMMON_P2B3,3,shape,1,{16u<<20,4u<<20,16u<<20},{},plan,0,P)==planning::ExternalPlanStatus::ready);
    paired_merges+=plan.pair_first>=0;assert(plan.summary_quotes<=26&&plan.basis_queries<=9);
    assert(plan.cached_pair||std::any_of(plan.stage,plan.stage+plan.stage_count,[](const auto&s){return s.use_resident&&s.borrow_result;}));
    StoredMergePlan tight_ram{},tight_writes{};
    assert(query_stored_merge(SBN3_SERIES_COMMON_P2B3,3,shape,1,{8u<<20,4u<<20,16u<<20},{},tight_ram,0,P)==planning::ExternalPlanStatus::ready);
    assert(tight_ram.total_resident_bytes<=(8u<<20)&&tight_ram.normal_write_bytes>=plan.normal_write_bytes);
    assert(query_stored_merge(SBN3_SERIES_COMMON_P2B3,3,shape,1,{16u<<20,4u<<20,16u<<20,0,plan.normal_write_bytes},{},tight_writes,0,P)==planning::ExternalPlanStatus::ready);
    assert(tight_writes.normal_write_bytes<=plan.normal_write_bytes);
    uint64_t saved_bound=0;unsigned clipped=0;
    for(unsigned i=0;i<plan.stage_count;++i){const auto&s=plan.stage[i];if(s.kind!=StoredMergeStage::Kind::product)continue;
        RegionLayout full{},kept{};assert(!region_layout(s.product.output_bytes,0,full)&&!region_layout(s.product.stored_output_bytes,0,kept));
        saved_bound+=full.data_bytes+full.index_bytes-kept.data_bytes-kept.index_bytes;clipped+=kept.allocation_bytes<full.allocation_bytes;
    }
    assert(saved_bound&&clipped==3);
    Fixture f(1,false,32u<<20);const auto memory=f.allocate(plan.storage_bytes,plan.alignment);const auto before=t.counters;
    allocation_watch_start();assert(!run_stored_merge(plan,t,store,left,right,out,{static_cast<uint8_t *>(memory.data),memory.bytes},*f.team));assert(!allocation_watch_stop());
    assert(t.counters.write_submitted-before.write_submitted<=plan.normal_write_bytes&&store.usage().peak_bytes<=plan.peak_backing_bytes);
    ref_int got,want;ref_inits(got,want,nullptr);const ScaledReference*expected[]{&numerator,&denominator};
    for(unsigned j=0;j<2;++j){load(t,io,out[j],got);ref_set(want,expected[j]->value);
        const int64_t low=std::min(out[j].exponent2,expected[j]->exponent);
        ref_mul_2exp(got,got,size_t(out[j].exponent2-low));ref_mul_2exp(want,want,size_t(expected[j]->exponent-low));assert(!ref_cmp(got,want));
    }
    ref_clears(got,want,nullptr);printf("Projected precision P%zu: native/cached placement, 3 projected products, write bound saves %llu bytes, independent signed reference PASS\n",P,(unsigned long long)saved_bound);
}
int main(){
    setvbuf(stdout,nullptr,_IOLBF,0);
    ownership();sums();pipelined_sums();
    series_case("e",euler_definition());series_case("exp(-1)",exp_minus_one_definition());
    series_case("pi",chudnovsky_definition());series_case("binary log",binary_log_definition(3));
    series_case("pi cached",chudnovsky_definition(),16u<<20);
    series_case("BBP cached",binary_log_definition(3),16u<<20);
    merge_failures();
    window_product();
    sparse_limit();
    precision_case("e exact",euler_definition(),181,64,0);
    precision_case("e limited",euler_definition(),361,12);
    precision_case("exp(-1)",exp_minus_one_definition(),413,16);
    precision_case("pi",chudnovsky_definition(),221,60);
    precision_case("binary log",binary_log_definition(3),511,20);
    large_precision_window();
    assert(delayed_product_loans);
    assert(resident_input_sides==3);
    printf("Delayed numerator loans across another product: %u independently checked merges PASS\n",delayed_product_loans);
    printf("Native common-operand pairs: %u independently checked mergers\n",paired_merges);
    assert(paired_merges);
    puts("OOC stored values and generic exact series mergers PASS");
}
