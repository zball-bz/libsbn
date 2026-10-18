// SPDX-License-Identifier: LGPL-3.0-or-later
// Reference application for the private OOC providers; not a public ABI.
#include "external/job.hpp"
#include "external/checkpoint.hpp"
#include "runtime/arena.hpp"
#include "sbn3/verify.h"
#include "sbn3/bitwindow.h"
#include <algorithm>
#include <cerrno>
#include <cfenv>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <glob.h>
#include <inttypes.h>
#include <linux/magic.h>
#include <new>
#include <string>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>
using namespace sbn::v3;
using namespace sbn::v3::external;
namespace {
constexpr uint64_t MiB=1u<<20,GiB=uint64_t(1)<<30;
volatile sig_atomic_t cancelled=0;
void signal_stop(int){cancelled=1;}
bool cancel(void*)noexcept{return cancelled!=0;}
uint64_t tick(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
uint64_t align(uint64_t n,uint64_t a){return (n+a-1)&~(a-1);}
std::string quote(const std::string&s){std::string out="\"";for(unsigned char c:s){if(c=='\\'||c=='\"')out+='\\';
    if(c<32){char b[8];snprintf(b,sizeof b,"\\u%04x",c);out+=b;}else out+=char(c);}return out+'\"';}
struct Block {
    void*data=nullptr;size_t bytes=0;
    bool allocate(size_t n){bytes=size_t(align(n,4096));return !posix_memalign(&data,4096,std::max(size_t(4096),bytes));}
    void reset(){free(data);data=nullptr;bytes=0;}
    std::span<uint8_t>span(){return {static_cast<uint8_t *>(data),bytes};}
    ~Block(){reset();}
};
struct Options {
    std::string constant="pi",directory,lock="build/benchmark.lock";
    uint64_t digits=0,ram=4*GiB,other=64*MiB,backing=UINT64_MAX,writes=UINT64_MAX,metadata_writes=MiB,host_reserve=4*GiB;
    unsigned workers=16,force=0,io_depth=16;uint64_t seed=SIZE_MAX,frontier=UINT64_MAX;
    bool run=false,resume=false,checkpoint=false,external_lock=false,diagnostic=false,write_given=false,describe=false,trace=false,allow_disk=false;
    CheckpointKey key{};
};
bool number(const char*p,uint64_t&n){if(!p||!*p||*p=='-')return false;char*end;errno=0;n=strtoull(p,&end,10);return !errno&&!*end;}
bool parse(int argc,char**argv,Options&o){
    for(int i=1;i<argc;++i){const char*a=argv[i];
        if(!std::strcmp(a,"--run")){o.run=true;continue;}if(!std::strcmp(a,"--resume")){o.run=o.resume=o.checkpoint=true;continue;}
        if(!std::strcmp(a,"--externally-locked")){o.external_lock=true;continue;}if(!std::strcmp(a,"--diagnostic-small")){o.diagnostic=true;continue;}
        if(!std::strcmp(a,"--describe-series")){o.describe=true;continue;}
        if(!std::strcmp(a,"--trace-operations")){o.trace=true;continue;}
        if(!std::strcmp(a,"--allow-disk")){o.allow_disk=true;continue;}
        if(i+1==argc)return false;const char*v=argv[++i];
        if(!std::strcmp(a,"--constant")){o.constant=v;continue;}if(!std::strcmp(a,"--directory")){o.directory=v;continue;}
        if(!std::strcmp(a,"--lock")){o.lock=v;continue;}
        if(!std::strcmp(a,"--checkpoint-key")){if(std::strlen(v)!=64)return false;o.checkpoint=true;
            for(unsigned j=0;j<4;++j){uint64_t word=0;for(unsigned k=0;k<16;++k){const char c=v[16*j+k];
                const int d=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;if(d<0)return false;word=(word<<4)|unsigned(d);}o.key.word[j]=word;}continue;}
        uint64_t n;if(!number(v,n))return false;
        if(!std::strcmp(a,"--digits"))o.digits=n;else if(!std::strcmp(a,"--ram-bytes"))o.ram=n;
        else if(!std::strcmp(a,"--other-live-bytes"))o.other=n;else if(!std::strcmp(a,"--backing-limit"))o.backing=n;
        else if(!std::strcmp(a,"--write-limit")){o.writes=n;o.write_given=true;}else if(!std::strcmp(a,"--metadata-write-limit"))o.metadata_writes=n;
        else if(!std::strcmp(a,"--host-reserve"))o.host_reserve=n;
        else if(!std::strcmp(a,"--workers")&&n>=1&&n<=32)o.workers=unsigned(n);
        else if(!std::strcmp(a,"--io-depth")&&(n==0||(n<=os::Uring::max_entries&&!(n&(n-1)))))o.io_depth=unsigned(n);
        else if(!std::strcmp(a,"--force-stored-depth")&&n<=8)o.force=unsigned(n);
        else if(!std::strcmp(a,"--maximum-native-seed"))o.seed=n;
        else if(!std::strcmp(a,"--maximum-resident-fragments"))o.frontier=n;else return false;
    }
    return o.digits&&o.digits<=(uint64_t(1)<<40)&&(o.ram>=4*GiB||o.diagnostic)&&o.ram<=uint64_t(SIZE_MAX)&&
        (o.constant=="pi"||o.constant=="e"||o.constant=="log8")&&(!o.run||(o.write_given&&!o.directory.empty()))&&
        (!o.checkpoint||((o.key.word[0]|o.key.word[1]|o.key.word[2]|o.key.word[3])&&o.metadata_writes<=UINT64_MAX-8192));
}
uint64_t available_memory(){FILE*f=fopen("/proc/meminfo","r");if(!f)return 0;char line[256];uint64_t value=0;
    while(fgets(line,sizeof line,f))if(sscanf(line,"MemAvailable: %" SCNu64 " kB",&value)==1){value*=1024;break;}fclose(f);return value;}
double temperature(){glob_t paths{};double result=-1;if(glob("/sys/class/hwmon/hwmon*/temp*_label",0,nullptr,&paths))return result;
    for(size_t i=0;i<paths.gl_pathc;++i){FILE*f=fopen(paths.gl_pathv[i],"r");if(!f)continue;char label[64]{};const bool match=fgets(label,sizeof label,f)&&!std::strncmp(label,"Tctl",4);fclose(f);if(!match)continue;
        std::string path=paths.gl_pathv[i];path.replace(path.size()-5,5,"input");f=fopen(path.c_str(),"r");long raw;if(f){if(fscanf(f,"%ld",&raw)==1)result=double(raw)/1000;fclose(f);}break;}
    globfree(&paths);return result;}
const char*phase_name(JobPhase p){const char*names[]{"series","quotient","root","fixed","radix","complete"};return unsigned(p)<6?names[unsigned(p)]:"invalid";}
uint64_t cpu_ns(const timeval&t){return uint64_t(t.tv_sec)*1000000000+uint64_t(t.tv_usec)*1000;}
uint64_t term_count(series::FormulaKind kind,size_t precision){
    if(kind==series::FormulaKind::Chudnovsky)return (64*uint64_t(precision)+160)/47;
    if(kind==series::FormulaKind::BinaryLog)return (64*uint64_t(precision)+67)/3;
    const uint64_t target=64*uint64_t(precision)+8;uint64_t lo=1,hi=2;
    while(series::factorial_log_bounds(0,hi).lower<target)hi*=2;
    while(lo<hi){const uint64_t mid=lo+(hi-lo)/2;if(series::factorial_log_bounds(0,mid).lower>=target)hi=mid;else lo=mid+1;}return lo;
}
uint64_t decay(const void*p,uint64_t at){const auto&f=*static_cast<const series::Formula *>(p);
    if(f.kind==series::FormulaKind::Chudnovsky)return at?47*(at-1):0;
    if(f.kind==series::FormulaKind::BinaryLog)return at?3*(at-1):0;
    return at?series::factorial_log_bounds(0,at-1).lower:0;
}
struct Outcome {
    const char*status="failed";uint64_t query_ns=0,ready_ns=0,verify_ns=0,dispose_ns=0;
    uint64_t resident=0,backing=0,write_bound=0,locked_need=0,host_available=0,backing_available=0;
    IoCounters io{};FileStoreUsage files{};CheckpointUsage checkpoint{};sbn3_arena_stats arena{};
    uint64_t integer=0,certified=0,bit_offset=0;uint64_t actual[2]{},expected[2]{};
    bool bbp=false,verified=false,tmpfs=false;double before=-1,after=-1;
};
int execute(const Options&o,Outcome&r,uint64_t began){
    constexpr size_t slots=512;FileStoreStorage file_layout{};ValueStoreStorage value_layout{};
    if(!query_file_store(slots,file_layout)||!query_value_store(slots,value_layout))return 2;
    const size_t arena_control=size_t(align(sizeof(sbn3_arena),4096));
    const size_t queue_bytes=o.io_depth?FileStore::uring_memory_bound(o.io_depth):0;
    const size_t known=sbn3_team_storage_bytes()+sbn3_team_stack_resident_bytes(o.workers)+arena_control+file_layout.bytes+value_layout.bytes+sizeof(Checkpoint)+sizeof(JobState)+queue_bytes;
    if(o.other>SIZE_MAX-known)return 2;const size_t other=size_t(o.other)+known;
    TerminalSpec terminal=o.constant=="pi"?TerminalSpec{23,24,4270934400ULL,10005,0,true}:
        o.constant=="e"?TerminalSpec{0,1,1,1,1,false}:TerminalSpec{-3,-2,1,1,0,false};
    JobPrecision precision;SeriesMetadata layout{};
    if(!query_job_precision(10,o.digits,terminal,128,1,precision)||!query_series_metadata(128,256,layout))return 2;
    series::Formula formula{o.constant=="pi"?series::FormulaKind::Chudnovsky:o.constant=="e"?series::FormulaKind::Euler:series::FormulaKind::BinaryLog,3};
    const uint64_t count=term_count(formula.kind,precision.series);const uint64_t begin=formula.kind==series::FormulaKind::Chudnovsky?0:1;
    JobSpec spec;spec.series={series::finite_formula(formula),{begin,begin+count},{o.workers,8,0,0,0},
        {&formula,decay,128,0,64.*16384*o.workers},precision.series};spec.terminal=terminal;spec.digits=o.digits;spec.force_stored_depth=o.force;
    spec.newton.maximum_native_seed=size_t(o.seed);spec.format.maximum_resident_fragments=o.frontier;spec.format.radix.use_alphabet=1;sbn3_radix_alphabet(10,spec.format.radix.alphabet);
    Block plan_memory,metadata,values_memory,files_memory;
    if(!plan_memory.allocate(sizeof(JobPlan))||!metadata.allocate(layout.bytes)||!values_memory.allocate(value_layout.bytes)||!files_memory.allocate(file_layout.bytes))return 2;
    auto*plan=::new(plan_memory.data)JobPlan{};const uint64_t qbegin=tick();
    const uint64_t metadata_disk=o.checkpoint?o.metadata_writes+8192:0;
    if(o.backing<metadata_disk)return 2;
    const auto status=compile_job(spec,{size_t(o.ram),other,o.backing-metadata_disk,0,o.writes},{},layout,metadata.span(),*plan);r.query_ns=tick()-qbegin;
    if(status!=planning::ExternalPlanStatus::ready){fprintf(stderr,"OOC query refused: status=%u\n",unsigned(status));r.status="query_refused";return 2;}
    if(plan->minimum_regions>slots||checkpoint_storage_bytes(slots)>plan->storage_bytes)return 2;
    r.resident=plan->total_resident_bytes;r.backing=plan->peak_backing_bytes+metadata_disk;r.write_bound=plan->write_bytes;
    r.locked_need=plan->storage_bytes+arena_control+sbn3_team_stack_resident_bytes(o.workers)+align(sbn3_team_storage_bytes(),4096);
    std::string path="/tmp";
    if(!o.directory.empty()){const auto slash=o.directory.find_last_of('/');
        path=o.resume?o.directory:slash==std::string::npos?".":slash==0?"/":o.directory.substr(0,slash);}
    struct statfs fs{};struct statvfs space{};if(statfs(path.empty()?".":path.c_str(),&fs)||statvfs(path.empty()?".":path.c_str(),&space)){perror("backing filesystem");return 2;}
    r.tmpfs=fs.f_type==TMPFS_MAGIC;r.backing_available=uint64_t(space.f_bavail)*space.f_frsize;r.host_available=available_memory();
    const __uint128_t demand=__uint128_t(r.resident)+(r.tmpfs?r.backing:0)+o.host_reserve;
    const bool host=demand<=r.host_available,backing=r.backing<=r.backing_available;
    rlimit locked{};getrlimit(RLIMIT_MEMLOCK,&locked);const bool lock_ok=locked.rlim_cur==RLIM_INFINITY||r.locked_need<=locked.rlim_cur;
    printf("{\"kind\":\"query\",\"constant\":%s,\"digits\":%" PRIu64 ",\"workers\":%u,\"ram_budget\":%" PRIu64 ",\"resident_bytes\":%" PRIu64 ",\"working_bytes\":%zu,\"backing_bytes\":%" PRIu64 ",\"data_write_bound\":%" PRIu64 ",\"metadata_write_allowance\":%" PRIu64 ",\"locked_need\":%" PRIu64 ",\"memlock_soft\":%llu,\"memlock_hard\":%llu,\"memlock_ok\":%s,\"host_admitted\":%s,\"backing_admitted\":%s,\"tmpfs\":%s,\"series_prefix\":%zu,\"radix_frontiers\":%" PRIu64 ",\"fractional_words\":%zu,\"terms\":%" PRIu64 ",\"query_ns\":%" PRIu64 "}\n",
        quote(o.constant).c_str(),o.digits,o.workers,o.ram,r.resident,plan->storage_bytes,r.backing,r.write_bound,o.checkpoint?o.metadata_writes:0,r.locked_need,
        (unsigned long long)locked.rlim_cur,(unsigned long long)locked.rlim_max,lock_ok?"true":"false",host?"true":"false",backing?"true":"false",r.tmpfs?"true":"false",
        plan->series.resident.first,plan->format.node[0].frontiers,precision.fractional,count,r.query_ns);
    size_t cyclic_inverse=0;for(size_t i=0;i<plan->quotient.inverse.rung_count;++i)cyclic_inverse+=bool(plan->quotient.inverse.rungs[i].residual_product.spec.cyclic);
    unsigned native_products=0,borrowed_products=0,raw_products=0,raw_radix=0,completed_caches=0,return_loans=0,native_pairs=0,middle_radix=0;
    auto count_cache=[&](const RowCachePlan&c){completed_caches+=c.kind==RowCacheKind::completed_columns;};
    auto count_merge=[&](const StoredMergePlan&m){native_pairs+=m.pair_first>=0;if(m.cached_pair)count_cache(m.cache);for(unsigned i=0;i<m.stage_count;++i){const auto&s=m.stage[i];
        native_products+=s.use_resident;borrowed_products+=s.borrow_result;
        raw_products+=s.kind==StoredMergeStage::Kind::product&&!s.use_resident&&s.product.spec.raw_operand<2;}};
    for(const auto&node:plan->series.nodes)if(!node.native)count_merge(node.merge);
    for(const auto&node:plan->series.nodes)return_loans+=unsigned(__builtin_popcount(node.loan_mask));
    for(size_t i=0;i<plan->series.resident.first;++i)if(plan->series.blocks[i].block.next_precision)count_merge(plan->series.blocks[i].merge);
    for(unsigned i=0;i<plan->format.classes;++i){raw_radix+=!plan->format.node[i].resident&&plan->format.node[i].split.spec.raw_operand<2;middle_radix+=bool(plan->format.node[i].split.spec.middle);}
    for(size_t i=0;i<plan->quotient.inverse.rung_count;++i)count_cache(plan->quotient.inverse.rungs[i].cache);
    count_cache(plan->quotient.refinement.cache);
    for(size_t i=0;i<plan->root.half.rung_count;++i)count_cache(plan->root.half.rungs[i].cache);
    printf("{\"kind\":\"selection\",\"plan_id\":%" PRIu64 ",\"diagnostic\":%s,\"force_stored_depth\":%u,\"maximum_native_seed\":%" PRIu64 ",\"maximum_resident_fragments\":%" PRIu64 ",\"other_live_allowance\":%zu,\"inverse_rungs\":%zu,\"cyclic_inverse_rungs\":%zu,\"cyclic_terminal_square\":%s,\"terminal_square_np\":%u,\"terminal_square_T\":%u,\"terminal_square_K\":%llu,\"inverse_shapes\":[",
        plan->plan_id,o.diagnostic?"true":"false",o.force,o.seed,o.frontier,other,
        plan->quotient.inverse.rung_count,cyclic_inverse,plan->root.square.spec.cyclic?"true":"false",
        plan->root.square.spec.primes,plan->root.square.spec.trunk_bits,(unsigned long long)(plan->root.square.output_bytes/8));
    for(size_t i=0;i<plan->quotient.inverse.rung_count;++i){const auto&q=plan->quotient.inverse.rungs[i];const auto&s=q.residual_product.spec;
        printf("%s{\"m\":%zu,\"n\":%zu,\"cyclic\":%s,\"np\":%u,\"T\":%u,\"K\":%llu,\"C\":%zu,\"M\":%zu,\"stripe\":%zu,\"g\":%u,\"middle_pipeline\":%s}",
            i?",":"",q.m,q.n,s.cyclic?"true":"false",s.primes,s.trunk_bits,(unsigned long long)(q.residual_product.output_bytes/8),
            s.outer_cells,s.inner_cells,s.stripe_cells,s.prime_batch,q.residual_product.middle.pipeline?"true":"false");
    }
    const auto&qr=plan->quotient.refinement.residual_product;
    printf("],\"resident_merge_products\":%u,\"borrowed_merge_products\":%u,\"raw_merge_products\":%u,\"raw_radix_classes\":%u,\"completed_caches\":%u,\"native_return_loans\":%u,\"native_shared_pairs\":%u,\"middle_radix_classes\":%u,\"radix_normal_writes\":%llu,"
        "\"quotient_residual\":{\"cyclic\":%s,\"stored_subtract\":%s,\"np\":%u,\"T\":%u,\"K\":%llu,\"stored_words\":%llu}}\n",
        native_products,borrowed_products,raw_products,raw_radix,completed_caches,return_loans,native_pairs,middle_radix,(unsigned long long)plan->format.normal_write_bytes,qr.spec.cyclic?"true":"false",qr.spec.cyclic.stored_subtract?"true":"false",
        qr.spec.primes,qr.spec.trunk_bits,(unsigned long long)(qr.output_bytes/8),(unsigned long long)(qr.stored_output_bytes/8));
    printf("{\"kind\":\"io_provider\",\"depth\":%u,\"memory_bound\":%zu,\"fallback_workers_per_ring\":%u}\n",o.io_depth,queue_bytes,o.io_depth?1:0);
    for(unsigned i=0;i<5;++i)printf("{\"kind\":\"phase_plan\",\"phase\":\"%s\",\"write_bound\":%" PRIu64 ",\"backing_bound\":%" PRIu64 "}\n",
        phase_name(JobPhase(i)),plan->phase_writes[i],plan->phase_backing[i]);
    if(o.describe){
        auto describe_merge=[&](const StoredMergePlan&m,const char*owner,size_t id){
            for(unsigned j=0;j<m.stage_count;++j){const auto&s=m.stage[j];
                printf("{\"kind\":\"merge_stage\",\"owner\":\"%s\",\"id\":%zu,\"stage\":%u,\"op\":%u,\"a\":%u,\"b\":%u,\"out\":%u,\"an\":%zu,\"bn\":%zu,\"out_limbs\":%zu,\"precision\":%zu,\"resident\":%s,\"borrowed\":%s,\"cache\":%s,\"workspace\":%zu,\"merge_workspace\":%zu}\n",
                    owner,id,j,unsigned(s.kind),s.a,s.b,s.out,m.shape[s.a].limbs,m.shape[s.b].limbs,m.shape[s.out].limbs,m.precision,
                    s.use_resident?"true":"false",s.borrow_result?"true":"false",m.cached_pair?"true":"false",
                    s.use_resident?s.resident.storage_bytes:s.kind==StoredMergeStage::Kind::sum?s.sum.storage_bytes:s.product.storage_bytes,m.storage_bytes);
            }
        };
        for(size_t i=0;i<plan->series.nodes.size();++i){const auto&n=plan->series.nodes[i];
            printf("{\"kind\":\"series_node\",\"id\":%zu,\"begin\":%" PRIu64 ",\"end\":%" PRIu64 ",\"need\":%u,\"native\":%s,\"return_bytes\":%" PRIu64 ",\"workspace\":%zu}\n",
                i,n.range.begin,n.range.end,n.need,n.native?"true":"false",n.return_bytes,n.native?n.layout.bytes:n.merge.storage_bytes);
            if(!n.native)describe_merge(n.merge,"finite",i);
        }
        for(size_t i=0;i<plan->series.resident.first;++i)if(plan->series.blocks[i].block.next_precision)describe_merge(plan->series.blocks[i].merge,"tail",i);
    }
    if(!o.run){r.status="query_only";return 0;}
    // Disk execution is explicit and still bounded by the cumulative transport
    // write quota. A pathname alone must never turn a tmpfs trial into SSD IO.
    if((!r.tmpfs&&(!o.allow_disk||o.writes==UINT64_MAX))||!host||!backing||!lock_ok){r.status="admission_refused";return 3;}
    if(!o.resume&&mkdir(o.directory.c_str(),0700)){perror("exclusive run directory");return 3;}
    int directory=open(o.directory.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);if(directory<0){perror("run directory");return 3;}
    SlotStorePlan slot_plan;if(!query_slot_store(slots,plan->peak_backing_bytes,slot_plan)){::close(directory);return 3;}
    FileStore files;ValueStore store;Checkpoint checkpoint;JobState state;bool files_open=false,checkpoint_open=false;
    sbn3_arena*arena=nullptr;sbn3_team*team=nullptr;sbn3_lease control{};sbn3_error error{};IoError failure{};int code=1;
    const size_t stack_offset=size_t(align(sbn3_team_storage_bytes(),65536));const size_t after_stacks=stack_offset+sbn3_team_stack_virtual_bytes(o.workers);
    const sbn3_arena_config arena_config{after_stacks+2*plan->alignment+plan->storage_bytes,size_t(o.ram)};size_t work_at=0;
    Transport transport{};
    auto release=[&]{const uint64_t start=tick();
        if(checkpoint_open){const auto e=checkpoint.close_preserving();if(e){fprintf(stderr,"checkpoint close: %s\n",e.where);fflush(nullptr);std::_Exit(5);}checkpoint_open=false;}
        state={};if(files_open){const auto e=files.close();if(e){fprintf(stderr,"file cleanup: %s errno=%d\n",e.where,e.system_error);fflush(nullptr);std::_Exit(5);}files_open=false;}
        r.files=files.usage();r.checkpoint=checkpoint.usage();r.io=transport.counters;
        if(arena)sbn3_arena_get_stats(arena,&r.arena);if(team)sbn3_team_destroy(team);if(control.token)sbn3_arena_release(arena,&control);if(arena)sbn3_arena_destroy(arena);
        ::close(directory);if(!o.checkpoint&&!rmdir(o.directory.c_str())){}else if(!o.checkpoint)perror("run directory cleanup");r.dispose_ns=tick()-start;
    };
    if((failure=files.open_directory(slot_plan,files_memory.span(),directory,true)))goto cleanup;files_open=true;
    if(o.io_depth&&(failure=files.enable_uring(o.io_depth,1)))goto cleanup;
    {uint64_t uuid[2];if(getrandom(uuid,sizeof uuid,0)!=sizeof uuid){failure={IoCode::io,"creation UUID",0,0,0,errno};goto cleanup;}
        if((failure=store.open_slots(uuid[0],uuid[1],slot_plan,values_memory.span(),files.lifecycle())))goto cleanup;}
    transport=files.transport({nullptr,cancel});transport.write_limit=o.writes;
    if(o.trace)transport.trace=[](void*,const OperationTrace&e)noexcept{
        printf("{\"kind\":\"operation_trace\",\"operation\":\"%s\",\"n\":%" PRIu64 ",\"m\":%" PRIu64 ",\"ns\":%" PRIu64 ",\"reads\":%" PRIu64 ",\"writes\":%" PRIu64 ",\"np\":%u,\"T\":%u,\"C\":%zu,\"M\":%zu}\n",
            e.kind,e.n,e.m,e.elapsed_ns,e.read_bytes,e.write_bytes,e.primes,e.trunk_bits,e.outer,e.inner);
    };
    if(sbn3_arena_create(&arena_config,&arena,&error)!=SBN3_OK)goto setup_failure;
    if(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)!=SBN3_OK)goto setup_failure;
    sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
    {sbn3_team_config config{};config.workers=o.workers;config.pin_threads=1;config.stack_offset=stack_offset;for(auto&cpu:config.cpu_ids)cpu=-1;
        if(sbn3_team_create(arena,&control,&config,&team,&error)!=SBN3_OK)goto setup_failure;}
    work_at=size_t(align(uintptr_t(arena->base)+after_stacks,plan->alignment)-uintptr_t(arena->base));
    if(sbn3_arena_prepare(arena,work_at,plan->storage_bytes,&error)!=SBN3_OK)goto setup_failure;
    if(o.checkpoint){if((failure=checkpoint.open(files,o.key)))goto cleanup;checkpoint_open=true;checkpoint.set_manifest_write_limit(o.metadata_writes);
        if(o.resume){if((failure=checkpoint.restore(*plan,state,transport,store,{arena->base+work_at,plan->storage_bytes})))goto cleanup;}
    }
    do{const auto phase=state.next;const auto before=transport.counters;rusage cpu_before{},cpu_after{};getrusage(RUSAGE_SELF,&cpu_before);const uint64_t start=tick();
        failure=o.checkpoint?checkpoint.advance(*plan,state,transport,store,*arena,work_at,*team):advance_job(*plan,state,transport,store,*arena,work_at,*team);
        const uint64_t elapsed=tick()-start;getrusage(RUSAGE_SELF,&cpu_after);
        printf("{\"kind\":\"phase\",\"phase\":\"%s\",\"next\":\"%s\",\"elapsed_ns\":%" PRIu64 ",\"read_bytes\":%" PRIu64 ",\"write_bytes\":%" PRIu64 ",\"read_calls\":%" PRIu64 ",\"write_calls\":%" PRIu64 ",\"user_ns\":%" PRIu64 ",\"system_ns\":%" PRIu64 ",\"live_backing\":%" PRIu64 ",\"error\":%u}\n",
            phase_name(phase),phase_name(state.next),elapsed,transport.counters.read_submitted-before.read_submitted,transport.counters.write_submitted-before.write_submitted,
            transport.counters.reads-before.reads,transport.counters.writes-before.writes,
            cpu_ns(cpu_after.ru_utime)-cpu_ns(cpu_before.ru_utime),cpu_ns(cpu_after.ru_stime)-cpu_ns(cpu_before.ru_stime),store.usage().live_bytes,unsigned(failure.code));
        if(failure){if(failure.code==IoCode::commit_unknown){const auto e=checkpoint.resolve();fprintf(stderr,"publication resolution: code=%u\n",unsigned(e.code));}goto cleanup;}
    }while(state.next!=JobPhase::complete);
    {const auto reads=files.uring_read_usage(),writes=files.uring_write_usage();
        printf("{\"kind\":\"io_queue_usage\",\"depth\":%u,\"mapped_bytes\":%zu,\"read_requests\":%" PRIu64 ",\"write_requests\":%" PRIu64 "}\n",
            o.io_depth,files.uring_mapped_bytes(),reads.requests,writes.requests);}
    printf("{\"kind\":\"radix_arithmetic\",\"middle_products\":%" PRIu64 ",\"carry_witnesses\":%" PRIu64 "}\n",
        state.text.middle_products,state.text.middle_fallbacks);
    r.integer=state.text.integer_part;r.certified=state.text.certified_digits;
    if(r.certified<o.digits){failure={IoCode::incomplete,"short certified radix prefix",0,o.digits,r.certified};goto cleanup;}
    if(checkpoint_open){if((failure=checkpoint.close_preserving()))goto cleanup;checkpoint_open=false;}
    if((failure=files.collect()))goto cleanup;
    // Every operation-owned arithmetic plan/cache is now retired. Output
    // values and reusable context stay alive for the independent verifier.
    plan->~JobPlan();plan=nullptr;plan_memory.reset();metadata.reset();r.ready_ns=tick()-began;
    if(o.constant!="e"){
        const uint64_t start=tick();auto lease=arena->acquire(work_at,(1u<<20)+index_page_bytes);
        StoreBuffers io{{static_cast<uint8_t *>(lease.data),1u<<20},{static_cast<uint8_t *>(lease.data)+(1u<<20),index_page_bytes}};
        ValueReader reader;uint64_t words[2]{},high=0;
        failure=reader.open(transport,state.binary.value.region(),io);
        if(!failure)failure=reader.read(0,{reinterpret_cast<uint8_t *>(words),sizeof words});
        if(!failure)failure=reader.read(8*precision.fractional,{reinterpret_cast<uint8_t *>(&high),8});arena->release(lease);
        if(failure)goto cleanup;r.bit_offset=64*precision.fractional-128;r.actual[0]=words[1];r.actual[1]=words[0];
        sbn3_bbp_result result{};
        for(unsigned window:{192u,384u}){
            if(o.constant=="pi")sbn3_pi_bbp(team,r.bit_offset,window,&result);
            else {const sbn3_bbp_stream term{.numerator=1,.a=1,.b=1,.denominator_scale=1,
                    .shift=-3,.stride=3,.power=1,.alternating=0};sbn3_bbp_window(team,&term,1,r.bit_offset,window,&result);}
            if(result.stable)break;
        }
        r.expected[0]=result.bits[0];r.expected[1]=result.bits[1];r.bbp=true;r.verified=result.stable&&words[1]==result.bits[0]&&words[0]==result.bits[1]&&high==(o.constant=="pi"?3:0);
        r.verify_ns=tick()-start;if(!r.verified){failure={IoCode::corrupt,"independent BBP tail"};goto cleanup;}
    }
    r.status="ok";code=0;goto cleanup;
setup_failure:
    fprintf(stderr,"arena/team preparation: %s errno=%d need=%zu have=%zu\n",error.where,error.system_error,error.need,error.have);
    r.status="setup_failed";code=3;
cleanup:
    if(failure){r.status=failure.code==IoCode::cancelled?"cancelled":"failed";fprintf(stderr,"OOC error %u: %s errno=%d need=%" PRIu64 " have=%" PRIu64 "\n",unsigned(failure.code),failure.where,failure.system_error,failure.need,failure.have);code=failure.code==IoCode::cancelled?130:1;}
    release();return code;
}
}
int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IOLBF,0);Options options;
    if(argc==1||(argc==2&&!std::strcmp(argv[1],"--help"))){puts("ooc_constant --constant pi|e|log8 --digits N [--ram-bytes B --workers W]\nDefault: query only; io_uring QD16. --run --directory NEW_DIR --write-limit B executes on tmpfs. Add --allow-disk for explicitly capped real-device execution.\nOptional: --io-depth 0|POWER_OF_TWO_UP_TO_64 (0: scalar IO), --checkpoint-key 64_HEX_DIGITS [--resume], --metadata-write-limit B, --backing-limit B, --other-live-bytes B, --host-reserve B.\nDiagnostics: --describe-series, --diagnostic-small --force-stored-depth N --maximum-native-seed N --maximum-resident-fragments N.\nSerialization: --lock PATH or --externally-locked (supervisor owns the lock).");return 0;}
    if(!parse(argc,argv,options)){fprintf(stderr,"invalid OOC runner options; see --help\n");return 2;}
    int lock=-1;if(!options.external_lock){lock=open(options.lock.c_str(),O_WRONLY|O_CREAT|O_CLOEXEC,0600);if(lock<0||flock(lock,LOCK_EX|LOCK_NB)){perror("benchmark lock");return 3;}}
    struct sigaction action{};action.sa_handler=signal_stop;action.sa_flags=SA_RESTART;sigemptyset(&action.sa_mask);sigaction(SIGINT,&action,nullptr);sigaction(SIGTERM,&action,nullptr);
    std::fesetround(FE_TONEAREST);Outcome result;result.before=temperature();
    if(options.run&&result.before>62){fprintf(stderr,"thermal admission refused: Tctl %.1f C\n",result.before);if(lock>=0)::close(lock);return 3;}
    const uint64_t start=tick();const int code=execute(options,result,start);const uint64_t lifecycle=tick()-start;result.after=temperature();
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    printf("{\"kind\":\"result\",\"status\":\"%s\",\"constant\":%s,\"digits\":%" PRIu64 ",\"mode\":\"%s\",\"directory\":%s,\"temperature_before\":%.3f,\"temperature_after\":%.3f,\"cold_fresh_to_output_ns\":%" PRIu64 ",\"resume_to_output_ns\":%" PRIu64 ",\"verification_ns\":%" PRIu64 ",\"disposal_ns\":%" PRIu64 ",\"whole_session_ns\":%" PRIu64 ",\"rss_peak_bytes\":%llu,\"arena_peak_bytes\":%zu,\"file_peak_claimed_bytes\":%" PRIu64 ",\"read_submitted\":%" PRIu64 ",\"write_submitted\":%" PRIu64 ",\"write_completed\":%" PRIu64 ",\"manifest_submitted\":%" PRIu64 ",\"certified_digits\":%" PRIu64 ",\"integer_part\":%" PRIu64 ",\"bbp_checked\":%s,\"bbp_passed\":%s,\"bit_offset\":%" PRIu64 ",\"tail\":\"%016" PRIx64 "%016" PRIx64 "\",\"bbp_tail\":\"%016" PRIx64 "%016" PRIx64 "\"}\n",
        result.status,quote(options.constant).c_str(),options.digits,options.resume?"resume":options.run?"fresh":"query",quote(options.directory).c_str(),result.before,result.after,
        options.resume?0:result.ready_ns,options.resume?result.ready_ns:0,result.verify_ns,result.dispose_ns,lifecycle,(unsigned long long)usage.ru_maxrss*1024,result.arena.peak_resident_bytes,result.files.peak_claimed_bytes,
        result.io.read_submitted,result.io.write_submitted,result.io.write_completed,result.checkpoint.manifest_bytes_submitted,result.certified,result.integer,
        result.bbp?"true":"false",result.verified?"true":"false",result.bit_offset,result.actual[0],result.actual[1],result.expected[0],result.expected[1]);
    if(lock>=0)::close(lock);return code;
}
