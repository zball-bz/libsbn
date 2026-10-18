// SPDX-License-Identifier: LGPL-3.0-or-later
#include "file_store_support.hpp"
#include "external/checkpoint.hpp"
#include "formula_reference.hpp"
#include <dirent.h>
#include <sys/stat.h>
#include <string>
using namespace sbn::v3;
static constexpr CheckpointKey key{{0x434850544255494cULL,0x444b455930303031ULL,37,53}};
struct Directory {
    char path[128]{};
    Directory(){struct statfs st{};assert(!statfs("/tmp",&st)&&st.f_type==TMPFS_MAGIC);
        std::strcpy(path,"/tmp/libsbn-ckpt-test-XXXXXX");assert(mkdtemp(path));}
    void clean(){const int fd=open(path,O_RDONLY|O_DIRECTORY|O_CLOEXEC);assert(fd>=0);DIR*d=fdopendir(fd);assert(d);
        while(auto*e=readdir(d)){if(!std::strcmp(e->d_name,".")||!std::strcmp(e->d_name,".."))continue;assert(!unlinkat(fd,e->d_name,0));}
        assert(!closedir(d));assert(!rmdir(path));path[0]=0;}
    ~Directory(){assert(!path[0]);}
};
struct FormulaJob {
    series::FormulaDef def;
    bool pi=false;
    std::unique_ptr<series::DataFormula>formula=std::make_unique<series::DataFormula>();
    JobPrecision precision{};SeriesMetadata layout{};std::unique_ptr<Aligned>metadata;
    std::unique_ptr<JobPlan>plan=std::make_unique<JobPlan>();uint64_t terms=0;
    FormulaJob(bool use_pi=false):def(use_pi?series::chudnovsky_definition():series::euler_definition()),pi(use_pi){
        const TerminalSpec terminal=pi?TerminalSpec{23,24,4270934400ULL,10005,0,true}:TerminalSpec{0,1,1,1,1,false};
        assert(query_job_precision(10,200,terminal,128,1,precision));terms=(pi?2:32)*precision.series;
        assert(series::data_formula_prepare(def,def.begin+terms,*formula));assert(query_series_metadata(128,64,layout));metadata=std::make_unique<Aligned>(layout.bytes);
        JobSpec spec{};spec.series={series::data_finite_formula(*formula),{def.begin,def.begin+terms},{4,8,0,0,0},
            {formula.get(),[](const void*p,uint64_t i){return static_cast<const series::DataFormula *>(p)->attenuation_bits(i);},4,83,0},precision.series};
        spec.terminal=terminal;spec.digits=200;spec.newton.maximum_native_seed=7;spec.force_stored_depth=2;spec.format.maximum_resident_fragments=16;
        spec.format.radix.use_alphabet=1;sbn3_radix_alphabet(10,spec.format.radix.alphabet);
        assert(compile_job(spec,{64u<<20,8u<<20,64u<<20},{},layout,metadata->span(),*plan)==planning::ExternalPlanStatus::ready);
    }
};
struct Hooks {
    int fail=-1,crash=-1,cancel=-1;unsigned seen=0,crash_visit=1;bool cancelled=false;
    static IoError before(void*p,CheckpointEvent event) noexcept {auto&h=*static_cast<Hooks *>(p);
        if(int(event)==h.crash&&++h.seen==h.crash_visit)kill(getpid(),SIGKILL);
        if(int(event)==h.cancel)h.cancelled=true;
        if(int(event)==h.fail)return {IoCode::io,"injected checkpoint event",0,0,0,EIO};return {};}
    static bool test(void*p)noexcept{return static_cast<Hooks *>(p)->cancelled;}
};
struct Runtime {
    SlotStorePlan slots=FileDevice::layout(192,64u<<20);
    Aligned metadata{slots.metadata.bytes},registry{FileDevice::storage(192).bytes},scratch{checkpoint_storage_bytes(192)};
    FileStore files;ValueStore store;Transport transport{};Checkpoint checkpoint;JobState state;
    std::unique_ptr<Fixture>runtime;size_t offset=0;bool closed=false;
    Runtime(const char*path,uint64_t nonce,FormulaJob&job,Hooks&hooks,CheckpointKey compatibility=key){
        const int fd=open(path,O_RDONLY|O_DIRECTORY|O_CLOEXEC);assert(fd>=0);assert(!files.open_directory(slots,registry.span(),fd,true));assert(!::close(fd));
        assert(!store.open_slots(nonce,0xdead,slots,metadata.span(),files.lifecycle()));transport=files.transport({&hooks,Hooks::test});
        assert(!checkpoint.open(files,compatibility,{&hooks,Hooks::before}));runtime=std::make_unique<Fixture>(4,false,64u<<20);
        auto lease=runtime->allocate(job.plan->storage_bytes,job.plan->alignment);offset=static_cast<uint8_t *>(lease.data)-runtime->arena->base;
        runtime->leases.pop_back();sbn3_arena_release(runtime->arena,&lease);
    }
    ~Runtime(){assert(closed);}
    IoError step(FormulaJob&job){return checkpoint.advance(*job.plan,state,transport,store,*runtime->arena,offset,*runtime->team);}
    void finish(FormulaJob&job){while(state.next!=JobPhase::complete){auto e=step(job);if(e)fprintf(stderr,"checkpoint phase%u error%u %s\n",unsigned(state.next),unsigned(e.code),e.where);assert(!e);}}
    void close(){assert(!checkpoint.close_preserving());state={};assert(!store.usage().live_regions);assert(!files.close());closed=true;}
};
static void verify(Runtime&r,FormulaJob&job){
    assert(r.state.next==JobPhase::complete&&r.state.text.certified_digits==200&&r.state.text.integer_part==(job.pi?3:2));
    StoreBuffers io{r.scratch.span().first(1u<<20),r.scratch.span().subspan(1u<<20,index_page_bytes)};ValueReader reader;
    std::vector<uint64_t>binary(job.precision.fractional+1);assert(!reader.open(r.transport,r.state.binary.value.region(),io));
    assert(!reader.read(0,{reinterpret_cast<uint8_t *>(binary.data()),binary.size()*8}));
    ref_int got,want,one;ref_inits(got,want,one,nullptr);ref_import(got,binary.size(),-1,8,0,0,binary.data());
    if(job.pi){auto atan=[&](ref_number*out,uint64_t q){ref_int power,term,divisor;ref_inits(power,term,divisor,nullptr);
            ref_set_ui(power,1);ref_mul_2exp(power,power,64*(job.precision.fractional+3));ref_set_ui(divisor,q);ref_fdiv_q(power,power,divisor);ref_set_ui(out,0);
            for(uint64_t k=0;ref_sgn(power);++k){ref_set_ui(divisor,2*k+1);ref_fdiv_q(term,power,divisor);if(k&1)ref_sub(out,out,term);else ref_add(out,out,term);
                ref_set_ui(divisor,q*q);ref_fdiv_q(power,power,divisor);}ref_clears(power,term,divisor,nullptr);};
        atan(want,5);atan(one,239);ref_mul_ui(want,want,16);ref_mul_ui(one,one,4);ref_sub(want,want,one);ref_fdiv_q_2exp(want,want,192);
    }else{formula_reference::Ref ref;formula_reference::reference(job.def,job.def.begin,job.def.begin+job.terms,ref);
        ref_mul_2exp(want,ref.t,64*job.precision.fractional);ref_fdiv_q(want,want,ref.d);ref_set_ui(one,1);ref_mul_2exp(one,one,64*job.precision.fractional);ref_add(want,want,one);}
    assert(!ref_cmp(got,want));ref_clears(got,want,one,nullptr);
    std::vector<uint8_t>digits(200);assert(!reader.open(r.transport,r.state.text.digits.region(),io));assert(!reader.read(0,digits));binary.resize(job.precision.fractional);
    for(size_t i=0;i<digits.size();++i){uint64_t carry=0;for(auto&v:binary){const __uint128_t x=__uint128_t(v)*10+carry;v=uint64_t(x);carry=uint64_t(x>>64);}assert(digits[i]=='0'+carry);}
}
static void roundtrip(){
    Directory dir;FormulaJob job;Hooks hooks;uint64_t before_restore=0;
    {Runtime r(dir.path,1,job,hooks);allocation_watch_start();assert(!r.step(job));assert(!allocation_watch_stop());assert(r.state.next==JobPhase::quotient);
        const auto reads=r.files.usage().read_bytes;assert(!r.checkpoint.commit(*job.plan,r.state));assert(r.files.usage().read_bytes==reads);before_restore=r.state.charged_write_bytes;
        assert(!r.checkpoint.collect(r.scratch.span()));r.close();}
    {Runtime r(dir.path,2,job,hooks);allocation_watch_start();const auto e=r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span());assert(!allocation_watch_stop());
        if(e)fprintf(stderr,"restore error%u %s\n",unsigned(e.code),e.where);assert(!e&&r.state.next==JobPhase::quotient&&r.state.charged_write_bytes==before_restore);
        allocation_watch_start();r.finish(job);assert(!allocation_watch_stop());verify(r,job);r.close();}
    dir.clean();puts("Checkpoint restart: remapped readonly files, XXH3+m61 verification, unchanged commit reads and exact final e PASS");
}
static void all_pi_phases(){
    Directory dir;Hooks hooks;JobPhase expected=JobPhase::series;
    for(unsigned run=0;run<6;++run){FormulaJob job(true);Runtime r(dir.path,600+run,job,hooks);
        if(run){assert(!r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span()));assert(r.state.next==expected);}
        if(expected==JobPhase::complete){verify(r,job);const auto writes=r.transport.counters.write_submitted;
            assert(!r.step(job)&&r.transport.counters.write_submitted==writes);r.close();break;}
        assert(!r.step(job));expected=r.state.next;r.close();
    }
    assert(expected==JobPhase::complete);dir.clean();puts("Pi restart after every phase: fresh compiled plans, quotient/root/binary/digit views and independent Machin result PASS");
}
static void failures(){
    Directory dir;FormulaJob job;Hooks hooks;
    {Runtime r(dir.path,3,job,hooks);
        r.state.plan_id=job.plan->plan_id;hooks.fail=int(CheckpointEvent::directory_sync);
        assert(r.checkpoint.commit(*job.plan,r.state).code==IoCode::commit_unknown);
        assert(!unlinkat(r.files.directory_descriptor(),"sbn-checkpoint.head",0));hooks.fail=-1;
        assert(!r.checkpoint.resolve()&&!r.checkpoint.unknown()&&!r.checkpoint.generation());
        assert(!r.step(job));const auto generation=r.checkpoint.generation();
        r.checkpoint.set_manifest_write_limit(r.checkpoint.usage().manifest_bytes_submitted+4095);
        assert(r.checkpoint.commit(*job.plan,r.state).code==IoCode::capacity&&r.checkpoint.generation()==generation);
        r.checkpoint.set_manifest_write_limit(UINT64_MAX);
        std::array<uint8_t,4096>old_image{};const int oldfd=openat(r.files.directory_descriptor(),"sbn-checkpoint.head",O_RDONLY|O_CLOEXEC);
        assert(oldfd>=0&&pread(oldfd,old_image.data(),old_image.size(),0)==ssize_t(old_image.size()));assert(!::close(oldfd));
        // Build a distinct replacement root, while Checkpoint pins keep both
        // series inputs alive. This tests values as well as manifest counters.
        assert(!advance_job(*job.plan,r.state,r.transport,r.store,*r.runtime->arena,r.offset,*r.runtime->team));assert(r.state.next==JobPhase::fixed);
        for(int event=0;event<int(CheckpointEvent::directory_sync);++event){hooks.fail=event;
            const auto e=r.checkpoint.commit(*job.plan,r.state);assert(e&&e.code!=IoCode::commit_unknown&&!r.checkpoint.unknown()&&r.checkpoint.generation()==generation);}
        hooks.fail=int(CheckpointEvent::directory_sync);assert(r.checkpoint.commit(*job.plan,r.state).code==IoCode::commit_unknown&&r.checkpoint.unknown());
        RegionLease blocked;assert(r.store.reserve(16,RegionKind::value,blocked).code==IoCode::commit_unknown);
        // Storage-model rollback of an unsynchronized rename: the prior HEAD
        // remains valid and its pinned files must not have been deleted.
        const int rollback=openat(r.files.directory_descriptor(),"sbn-checkpoint.head",O_WRONLY|O_TRUNC|O_CLOEXEC);
        assert(rollback>=0&&pwrite(rollback,old_image.data(),old_image.size(),0)==ssize_t(old_image.size()));assert(!fsync(rollback)&&!::close(rollback));
        hooks.fail=-1;assert(!r.checkpoint.resolve()&&!r.checkpoint.unknown()&&r.checkpoint.generation()==generation);
        hooks.fail=int(CheckpointEvent::directory_sync);assert(r.checkpoint.commit(*job.plan,r.state).code==IoCode::commit_unknown);
        hooks.fail=-1;assert(!r.checkpoint.resolve()&&!r.checkpoint.unknown()&&r.checkpoint.generation()==generation+1);
        hooks.cancel=int(CheckpointEvent::rename_root);assert(!r.checkpoint.commit(*job.plan,r.state,{&hooks,Hooks::test}));assert(hooks.cancelled&&!r.checkpoint.unknown());
        const auto committed=r.checkpoint.generation();assert(r.checkpoint.commit(*job.plan,r.state,{&hooks,Hooks::test}).code==IoCode::cancelled&&r.checkpoint.generation()==committed);
        hooks.cancel=-1;hooks.cancelled=false;r.close();}
    {auto wrong=key;wrong.word[1]^=1;Runtime r(dir.path,4,job,hooks,wrong);assert(r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span()).code==IoCode::incompatible);
        assert(!r.state.plan_id&&!r.store.usage().live_regions);r.close();}
    {Runtime r(dir.path,5,job,hooks);assert(!r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span()));r.finish(job);verify(r,job);r.close();}
    dir.clean();puts("Checkpoint failures: pre-rename refusal, UNKNOWN pins/create block/resolve, deferred cancellation and wrong-build rejection PASS");
}
static void corrupt_restore(){
    Directory dir;FormulaJob job;Hooks hooks;std::array<uint8_t,4096>manifest{};RegionId id{};
    {Runtime r(dir.path,401,job,hooks);assert(!r.step(job));id=r.state.pair[0].value.region().id;
        const int fd=openat(r.files.directory_descriptor(),"sbn-checkpoint.head",O_RDONLY|O_CLOEXEC);assert(fd>=0);
        assert(pread(fd,manifest.data(),manifest.size(),0)==ssize_t(manifest.size()));assert(!::close(fd));r.close();}
    const int directory=open(dir.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC);assert(directory>=0);
    auto install=[&](std::span<const uint8_t>image){const int fd=openat(directory,"sbn-checkpoint.head",O_WRONLY|O_TRUNC|O_CLOEXEC);assert(fd>=0);
        assert(pwrite(fd,image.data(),image.size(),0)==ssize_t(image.size())&&!fsync(fd)&&!::close(fd));};
    install(std::span(manifest).first(2048));
    {Runtime r(dir.path,402,job,hooks);assert(r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span()).code==IoCode::corrupt);
        assert(!r.state.plan_id&&!r.store.usage().live_regions);r.close();}
    install(manifest);auto bad_summary=manifest;put64(bad_summary.data()+256+112,get64(bad_summary.data()+256+112)^1);
    put64(bad_summary.data()+4088,metadata_checksum(std::span(bad_summary).first(4088)));install(bad_summary);
    {Runtime r(dir.path,403,job,hooks);const auto e=r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span());
        assert(e.code==IoCode::corrupt&&!std::strcmp(e.where,"checkpoint integer summary/m61"));assert(!r.state.plan_id&&!r.store.usage().live_regions);r.close();}
    install(manifest);char name[64];snprintf(name,sizeof name,"sbn-%016llx-%016llx-%016llx",(unsigned long long)id.uuid_high,(unsigned long long)id.uuid_low,(unsigned long long)id.incarnation);
    const int value=openat(directory,name,O_RDWR|O_CLOEXEC);assert(value>=0);uint8_t original;assert(pread(value,&original,1,0)==1);const uint8_t changed=original^0x80;
    assert(pwrite(value,&changed,1,0)==1&&!fsync(value));
    {Runtime r(dir.path,404,job,hooks);assert(r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span()).code==IoCode::corrupt);
        assert(!r.state.plan_id&&!r.store.usage().live_regions);r.close();}
    // Failed import must preserve the preexisting file, including after its
    // protected descriptor has been retired and explicitly closed.
    struct stat st{};assert(!fstatat(directory,name,&st,0));assert(pwrite(value,&original,1,0)==1&&!fsync(value));assert(!::close(value)&&!::close(directory));
    {Runtime r(dir.path,405,job,hooks);assert(!r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span()));r.finish(job);verify(r,job);r.close();}
    dir.clean();puts("Restore rejection: torn manifest, independent m61 mismatch, payload corruption and non-destructive failed import PASS");
}
struct KillIo {
    Transport inner{};unsigned writes=0;
    static Completion read(void*p,uint64_t at,std::span<uint8_t>b)noexcept{auto&x=*static_cast<KillIo *>(p);return x.inner.read(x.inner.context,at,b);}
    static Completion write(void*p,uint64_t at,std::span<const uint8_t>b)noexcept{auto&x=*static_cast<KillIo *>(p);
        if(++x.writes==3){const size_t part=std::max(size_t(4096),(b.size()/2)&~size_t(4095));
            const auto done=x.inner.write(x.inner.context,at,b.first(std::min(part,b.size())));assert(done.code==IoCode::ok);kill(getpid(),SIGKILL);}
        return x.inner.write(x.inner.context,at,b);}
};
static void writer(const char*path,int event){
    FormulaJob job;Hooks hooks;Runtime r(path,100+unsigned(event),job,hooks);assert(!r.step(job));
    // The first committed series pair is the old root. Kill inside the next
    // publication; prior data and new candidates remain named for recovery.
    KillIo io{r.transport};if(event==7){r.transport.context=&io;r.transport.read=KillIo::read;r.transport.write=KillIo::write;}
    else {hooks.crash=event;hooks.crash_visit=2;}
    (void)r.step(job);_exit(92);
}
static void crashes(const char*exe){
    for(int event=0;event<=7;++event){Directory dir;
        const pid_t child=fork();assert(child>=0);if(!child){char number[16];snprintf(number,sizeof number,"%d",event);execl(exe,exe,"--writer",dir.path,number,nullptr);_exit(93);}
        int status=0;assert(waitpid(child,&status,0)==child&&WIFSIGNALED(status)&&WTERMSIG(status)==SIGKILL);
        FormulaJob job;Hooks hooks;Runtime r(dir.path,300+unsigned(event),job,hooks);
        const auto e=r.checkpoint.restore(*job.plan,r.state,r.transport,r.store,r.scratch.span());if(e)fprintf(stderr,"crash restore event%d error%u %s\n",event,unsigned(e.code),e.where);assert(!e);
        const bool replacement_visible=event==int(CheckpointEvent::directory_sync);
        assert(r.state.next==(replacement_visible?JobPhase::fixed:JobPhase::quotient));
        if(!replacement_visible)assert(r.state.charged_write_bytes>=job.plan->phase_writes[1]);
        r.finish(job);verify(r,job);r.close();dir.clean();
        printf("Process SIGKILL publication event%d: %s value root, crash writes accounted, orphan GC and exact resumed output PASS\n",event,replacement_visible?"replacement":"prior");
    }
}
int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IOLBF,0);if(argc==4&&!std::strcmp(argv[1],"--writer")){assert(!std::strncmp(argv[2],"/tmp/libsbn-ckpt-test-",20));writer(argv[2],atoi(argv[3]));return 1;}
    roundtrip();all_pi_phases();failures();corrupt_restore();crashes("/proc/self/exe");
    puts("Durable job phases: values+directory before manifest, atomic root, UNKNOWN, charged crash attempts and tmpfs process recovery PASS");
}
