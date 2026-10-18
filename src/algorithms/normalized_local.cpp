// Original libsbn code; LGPL-3.0-or-later.
#include "algorithms/normalized_local.hpp"
#include "algorithms/local_inverse.hpp"
#include "algorithms/newton_limits.hpp"
#include "algorithms/reciprocal.hpp"
#include "product/local_program.hpp"
#include "product/compact_windows.hpp"
#include "product/native_capabilities.hpp"
#include <algorithm>
#include <cstdint>
#include <new>

namespace sbn::v3 {
namespace {
using namespace product;
enum class Recipe { tiny_inverse, word_divide, native_divide, ladder };
enum class GroupKind { leaf, local, peeled, compact };
struct Layout {
    Recipe recipe=Recipe::ladder;
    bool compact=false;
    unsigned records=0,seed=0;
};
struct Record {
    GroupKind kind=GroupKind::leaf;
    std::size_t n=0,bytes=0;
    unsigned count=0;
    LeafWindowPlan leaf{};
    LocalWindowPlan local{};
    PeeledWindowPlan peeled{};
    compact_window_detail::Plan compact{};
    pq16::Shape full[3]{},low[3]{};
};
struct State {
    std::uint64_t marker=0;
    NormalizedLocalInfo info{};
    Layout layout{};
    const Record* records=nullptr;
    std::size_t metadata_extent=0;
};
constexpr std::uint64_t magic=UINT64_C(0x53424e334e4c4f43);

bool choose_layout(NormalizedLocalKind kind,std::size_t n,Layout& layout) noexcept {
    if(!n||n>newton_limits::precision_words||
       (kind!=NormalizedLocalKind::inverse&&kind!=NormalizedLocalKind::divide))return false;
    if(kind==NormalizedLocalKind::inverse&&n<=4){layout.recipe=Recipe::tiny_inverse;layout.seed=unsigned(n);return true;}
    if(kind==NormalizedLocalKind::divide&&!local_program_eligible(n)&&local_dyadic_divide_supported(n)){
        layout.recipe=local_dyadic_divide_native(n)?Recipe::native_divide:Recipe::word_divide;
        return true;
    }
    if(kind==NormalizedLocalKind::divide&&n<4)return false;
    if(n>15&&!native_available())return false;
    if(n>local_inverse_max_limbs){
        const auto m=newton_contract::next_precision(n);
        const auto shape=kind==NormalizedLocalKind::inverse?
            refinement_products<RefinementKind::Inverse>(m,n):refinement_products<RefinementKind::Quotient>(m,n);
        layout.compact=bool(compact_window_detail::query(shape).ring);
        if(!layout.compact&&(kind==NormalizedLocalKind::inverse||n>local_divide_max_limbs))return false;
    }
    std::size_t target=kind==NormalizedLocalKind::inverse?n:newton_contract::next_precision(n);
    while(target>15){++layout.records;target=newton_contract::next_precision(target);}
    layout.seed=unsigned(target);
    if(kind==NormalizedLocalKind::divide)++layout.records;
    return true;
}
Record select_record(const WindowGroupShape& shape,bool use_compact) noexcept {
    Record result;result.n=shape.products[shape.count-2].bound.words;result.count=shape.count;
    if(use_compact){
        result.compact=compact_window_detail::query(shape);
        if(result.compact.ring){
            result.kind=GroupKind::compact;
            std::size_t original_low=0,selected_low=0;
            for(unsigned j=0;j<shape.count;++j){
                const auto& r=shape.products[j];
                const auto g=r.bound.words>=result.compact.ring?r.bound.words-result.compact.ring+1:1;
                if(g<=8)continue;
                const auto an=std::min(r.a.words,g),bn=std::min(r.b.words,g);
                result.low[j]=local_window_detail::linear_shape(an,bn);
                selected_low=std::max(selected_low,local_window_detail::full_selected_bytes(result.low[j],an,bn));
                original_low=std::max(original_low,local_window_detail::full_bytes(g,g));
            }
            // The current compact query prices g-by-g low witnesses. Keep
            // that quote, expanding it if a nonmonotone smaller shape needs
            // more scratch; never assume backend sizes are monotone.
            if(selected_low>original_low)result.compact.bytes+=selected_low-original_low;
            result.bytes=result.compact.bytes;return result;
        }
    }
    if(leaf_windows_supported(shape)){
        result.kind=GroupKind::leaf;result.leaf=leaf_windows_query(shape);
        result.bytes=result.leaf.bytes;return result;
    }
    result.local=local_windows_query(shape);
    if(peel_window_possible(shape))result.peeled=peeled_windows_query(shape,result.local);
    if(result.peeled.ring){result.kind=GroupKind::peeled;result.bytes=result.peeled.bytes;return result;}
    result.kind=GroupKind::local;result.bytes=result.local.storage_bytes;
    // Capture all unshared full products, including the ones below the FFT
    // eligibility used by the older LocalWindowProgram. A zero shape is an
    // explicit u52 choice, not an execution-time fallback query.
    for(unsigned j=0;j<shape.count;++j){
        const auto& r=shape.products[j];
        const bool full=!result.local.shared.nfull&&
            (!r.cancellation||(!result.local.middle&&!result.local.cancel_shape.nfull));
        if(full)result.full[j]=local_window_detail::linear_shape(r.a.words,r.b.words);
    }
    return result;
}
struct Compiler {
    Record* records=nullptr;
    unsigned capacity=0;
    bool compact=false;
    mutable unsigned count=0;
    std::size_t bytes(const WindowGroupShape& shape) const noexcept {
        require(count<capacity,SBN3_FATAL_MATH,"normalized local record count");
        const auto record=select_record(shape,compact);
        if(records)records[count]=record;
        ++count;return record.bytes;
    }
    static std::size_t retained_words(const WindowGroupShape& shape) noexcept {
        return shape.count==2?0:shape.products[shape.count-2].window.words;
    }
};
bool compile_info(NormalizedLocalKind kind,std::size_t n,const Layout& layout,
                  Record* records,NormalizedLocalInfo& info) noexcept {
    info={};info.kind=kind;info.n=n;info.record_count=layout.records;info.seed_words=layout.seed;
    switch(layout.recipe){
        case Recipe::tiny_inverse:info.workspace_bytes=local_inverse_bytes(n);break;
        case Recipe::word_divide:info.workspace_bytes=local_dyadic_divide_bytes(n,false);break;
        case Recipe::native_divide:info.workspace_bytes=local_dyadic_divide_bytes(n,true);break;
        case Recipe::ladder:{
            const Compiler compiler{records,layout.records,layout.compact};
            info.workspace_bytes=kind==NormalizedLocalKind::inverse?reciprocal_bytes(n,compiler):quotient_bytes(n,compiler);
            require(compiler.count==layout.records,SBN3_FATAL_MATH,"normalized local complete program");
            info.products=2*layout.records+(kind==NormalizedLocalKind::divide);
            break;
        }
    }
    std::size_t bytes=0;
    if(!mul_size(layout.records,sizeof(Record),bytes)||!add_size(bytes,sizeof(State)+256,info.metadata_bytes)||
       !add_size(info.metadata_bytes,info.workspace_bytes,bytes)||!add_size(bytes,127,info.storage_bytes))return false;
    return true;
}
struct Replay {
    const State& state;
    static std::size_t retained_words(const WindowGroupShape& shape) noexcept {
        return Compiler::retained_words(shape);
    }
    template<class Function>void with_group(const WindowGroupShape& shape,Frame& space,
                                           std::uint64_t* output,Function&& function) const noexcept {
        const auto n=shape.products[shape.count-2].bound.words;
        for(unsigned j=0;j<state.info.record_count;++j){
            const auto& record=state.records[j];
            if(record.n!=n||record.count!=shape.count)continue;
            switch(record.kind){
                case GroupKind::leaf:{
                    LeafWindowProducts products(record.leaf,space);function(products);return;
                }
                case GroupKind::local:{
                    LocalWindowProducts products(record.local,space,output,record.full);function(products);return;
                }
                case GroupKind::peeled:{
                    FrameMark lifetime(space);PeeledWindows products(record.peeled,space);function(products);return;
                }
                case GroupKind::compact:{
                    FrameMark lifetime(space);compact_window_detail::Products products(record.compact,space,record.low);
                    function(products);return;
                }
            }
        }
        fatal(SBN3_FATAL_MATH,"normalized local missing compiled group");
    }
};
}

sbn3_query_result normalized_local_query(NormalizedLocalKind kind,std::size_t n,NormalizedLocalInfo& info) noexcept {
    info={};Layout layout;
    if(!choose_layout(kind,n,layout))return SBN3_UNSUPPORTED;
    return compile_info(kind,n,layout,nullptr,info)?SBN3_SUPPORTED:SBN3_QUERY_CAPACITY;
}
NormalizedLocalPrepared normalized_local_prepare(NormalizedLocalKind kind,std::size_t n,Frame& metadata) noexcept {
    Layout layout;require(choose_layout(kind,n,layout),SBN3_FATAL_ARGUMENT,"normalized local shape");
    std::size_t records_bytes=0,metadata_bytes=0;
    require(mul_size(layout.records,sizeof(Record),records_bytes)&&
            add_size(records_bytes,sizeof(State)+256,metadata_bytes),SBN3_FATAL_SIZE,"normalized local metadata size");
    require(metadata.capacity()-metadata.used()>=metadata_bytes,SBN3_FATAL_WORKSPACE,"normalized local metadata",
            metadata_bytes,metadata.capacity()-metadata.used());
    auto* state=::new(metadata.allocate(sizeof(State),128))State{};
    auto* records=metadata.alloc<Record>(layout.records);
    for(unsigned j=0;j<layout.records;++j)::new(records+j)Record{};
    require(compile_info(kind,n,layout,records,state->info),SBN3_FATAL_SIZE,"normalized local complete quote");
    state->layout=layout;state->records=records;
    state->metadata_extent=std::size_t(metadata.data()+metadata.used()-reinterpret_cast<std::uint8_t*>(state));
    state->marker=magic;
    return {state,state->info};
}
void normalized_local_execute(const NormalizedLocalPrepared& prepared,sbn3_const_limbs A,sbn3_const_limbs D,
                              sbn3_limbs Q,Frame& work) noexcept {
    require(prepared.state,SBN3_FATAL_LIFETIME,"normalized local prepared state");
    const auto& state=*static_cast<const State*>(prepared.state);const auto& info=state.info;
    require(state.marker==magic&&prepared.info.n==info.n&&prepared.info.kind==info.kind&&
            prepared.info.workspace_bytes==info.workspace_bytes,SBN3_FATAL_LIFETIME,"normalized local identity");
    const auto n=info.n;const bool division=info.kind==NormalizedLocalKind::divide;
    require(D.data&&D.count==n&&(D.data[n-1]>>63)&&Q.data&&Q.capacity>=n+1&&
            !(std::uintptr_t(D.data)&7)&&!(std::uintptr_t(Q.data)&7),SBN3_FATAL_ARGUMENT,"normalized local values");
    require(division?(A.data&&A.count==n+1&&A.data[n]<=1&&!(std::uintptr_t(A.data)&7)):(!A.data&&!A.count),
            SBN3_FATAL_ARGUMENT,"normalized local numerator");
    require(!overlaps(D.data,n*8,Q.data,(n+1)*8)&&
            (!division||((A.data==Q.data||!overlaps(A.data,(n+1)*8,Q.data,(n+1)*8))&&
                          !overlaps(A.data,(n+1)*8,D.data,n*8))),SBN3_FATAL_ARGUMENT,"normalized local value alias");
    const auto available=work.capacity()-work.used();
    require(available>=info.workspace_bytes,SBN3_FATAL_WORKSPACE,"normalized local workspace",info.workspace_bytes,available);
    const auto* work_begin=work.used()?work.data()+work.used():work.data();
    require(!info.workspace_bytes||!(std::uintptr_t(work_begin)&(info.alignment-1)),
            SBN3_FATAL_WORKSPACE,"normalized local workspace alignment");
    require(!overlaps(state.records,info.record_count*sizeof(Record),work_begin,available)&&
            !overlaps(&state,state.metadata_extent,work_begin,available),SBN3_FATAL_ARGUMENT,"normalized local metadata/work alias");
    for(auto value:{A,D,sbn3_const_limbs{Q.data,n+1}})if(value.count){
        valid_span(value.data,value.count*8,"normalized local span");
        require(!overlaps(value.data,value.count*8,&state,state.metadata_extent)&&
                !overlaps(value.data,value.count*8,work_begin,available),SBN3_FATAL_ARGUMENT,"normalized local scratch alias");
    }
    FrameMark lifetime(work);
    switch(state.layout.recipe){
        case Recipe::tiny_inverse:local_inverse(Q.data,D.data,n,work);return;
        case Recipe::word_divide:
        case Recipe::native_divide:{
            auto* scratch=info.workspace_bytes?work.allocate(info.workspace_bytes,64):nullptr;
            local_dyadic_divide(Q.data,A.data,D.data,n,state.layout.recipe==Recipe::native_divide,scratch,info.workspace_bytes);
            return;
        }
        case Recipe::ladder:{
            Replay replay{state};
            if(division)quotient(Q.data,A.data,D.data,n,work,replay);
            else reciprocal(Q.data,D.data,n,work,replay);
            return;
        }
    }
    fatal(SBN3_FATAL_MATH,"normalized local recipe");
}
}
