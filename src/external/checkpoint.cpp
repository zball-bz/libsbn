// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/checkpoint.hpp"
#include "common/checked.hpp"
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>
namespace sbn::v3::external {
namespace {
constexpr uint64_t magic=0x3154504b434e4253ULL,version=1;
constexpr size_t manifest_bytes=4096,header_bytes=256,record_bytes=160,transfer_bytes=1u<<20;
constexpr const char*head_name="sbn-checkpoint.head";
IoError failure(const char*where,int error=errno){return {IoCode::io,where,0,0,0,error};}
IoError sync_fd(int fd,const char*where){int rc;do{rc=fsync(fd);}while(rc&&errno==EINTR);return rc?failure(where):IoError{};}
void clear(StoredValue*p){for(unsigned i=0;i<5;++i)p[i].reset();}
const StoredInteger*integer(const JobState&s,unsigned role){return role<2?&s.pair[role]:role==2?&s.working:role==3?&s.binary:nullptr;}
StoredInteger*integer(JobState&s,unsigned role){return const_cast<StoredInteger *>(integer(static_cast<const JobState&>(s),role));}
const StoredValue&value(const JobState&s,unsigned role){return role==4?s.text.digits:integer(s,role)->value;}
unsigned required(JobPhase phase,bool fused){
    switch(phase){case JobPhase::series:return 0;case JobPhase::quotient:return 3;
        case JobPhase::root:return fused?4:UINT32_MAX;case JobPhase::fixed:return 4;
        case JobPhase::radix:return 8;case JobPhase::complete:return 24;}return UINT32_MAX;
}
}
size_t checkpoint_storage_bytes(size_t slots) noexcept {
    size_t names,bytes;if(slots>SIZE_MAX-16||!mul_size(slots+16,64,names)||!add_size(2*transfer_bytes+index_page_bytes+4096,names,bytes)||
                         !align_size(bytes,4096,bytes))return 0;
    return bytes;
}
Checkpoint::~Checkpoint(){require(directory_<0&&lock_<0,SBN3_FATAL_LIFETIME,"checkpoint requires explicit close");}
IoError Checkpoint::event(CheckpointEvent e) noexcept{return hook_.before?hook_.before(hook_.context,e):IoError{};}
IoError Checkpoint::open(FileStore&files,CheckpointKey key,CheckpointHook hook) noexcept {
    if(directory_>=0||files.directory_descriptor()<0||!(key.word[0]|key.word[1]|key.word[2]|key.word[3]))return {IoCode::invalid,"checkpoint directory/key"};
    uint64_t nonce;const ssize_t entropy=getrandom(&nonce,sizeof nonce,0);if(entropy!=sizeof nonce)return failure("checkpoint temporary nonce",entropy<0?errno:EIO);
    const int dir=openat(files.directory_descriptor(),".",O_RDONLY|O_DIRECTORY|O_CLOEXEC);if(dir<0)return failure("checkpoint directory open");
    const int lock=openat(dir,"sbn-checkpoint.lock",O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600);
    if(lock<0){const auto e=failure("checkpoint lock open");::close(dir);return e;}
    if(flock(lock,LOCK_EX|LOCK_NB)){const auto e=failure("checkpoint exclusive lock");::close(lock);::close(dir);return e;}
    struct stat st{};const int rc=fstatat(dir,head_name,&st,AT_SYMLINK_NOFOLLOW);const int saved=errno;
    if(rc&&saved!=ENOENT){::close(lock);::close(dir);return failure("checkpoint head stat",saved);}
    if(!rc&&!S_ISREG(st.st_mode)){::close(lock);::close(dir);return {IoCode::corrupt,"checkpoint head type"};}
    current_={};pending_={};current_image_.fill(0);pending_image_.fill(0);usage_={};
    files_=&files;directory_=dir;lock_=lock;key_=key;hook_=hook;nonce_=nonce;attempt_=0;exists_=!rc;loaded_=!exists_;unknown_=needs_gc_=false;
    files.suspend_creates(exists_);return {};
}
IoError Checkpoint::capture(const JobPlan&p,const JobState&s,uint64_t reservation,Snapshot&out,StoredValue*pins) noexcept {
    if(!p.ready||s.plan_id!=p.plan_id||s.pair[2].value||s.charged_write_bytes>UINT64_MAX-reservation)return {IoCode::invalid,"checkpoint job state"};
    const unsigned mask=required(s.next,p.fused_root);if(mask==UINT32_MAX)return {IoCode::invalid,"checkpoint phase"};
    Snapshot snap{};snap.generation=current_.generation+1;if(!snap.generation)return {IoCode::overflow,"checkpoint generation"};
    snap.job=p.plan_id;snap.next=s.next;snap.charged=s.charged_write_bytes+reservation;snap.reserved=reservation;snap.slots=files_->slot_capacity();
    snap.certified=s.text.certified_digits;snap.integer=s.text.integer_part;
    if(snap.certified>p.spec.digits||(s.next!=JobPhase::complete&&(snap.certified||snap.integer)))return {IoCode::invalid,"checkpoint digit metadata"};
    for(unsigned role=0;role<5;++role){const auto&v=value(s,role);
        if(bool(v)!=bool(mask&(1u<<role)))return {IoCode::invalid,"checkpoint value roles"};if(!v)continue;
        auto&r=snap.records[snap.count];r.role=role;r.region=v.region();
        if(files_->descriptor(r.region)<0)return {IoCode::invalid,"checkpoint file mapping"};
        if(const auto*x=integer(s,role)){
            if(!x->extent_known||r.region.layout.logical_bytes%8||x->extent.end>r.region.layout.logical_bytes/8||
               x->first_limb>r.region.layout.logical_bytes/8||x->limbs()>r.region.layout.logical_bytes/8-x->first_limb)
                return {IoCode::invalid,"checkpoint integer view"};
            r.exponent=x->exponent2;r.negative=x->negative;r.first=x->first_limb;r.words=x->limbs();r.extent=x->extent;
        }else if(r.region.layout.logical_bytes!=p.spec.digits)return {IoCode::invalid,"checkpoint digit length"};
        ++snap.count;
    }
    for(unsigned i=0;i<snap.count;++i)value(s,snap.records[i].role).share(pins[i]);out=snap;return {};
}
void Checkpoint::encode(const Snapshot&s,std::span<uint8_t>b) const noexcept {
    std::fill(b.begin(),b.end(),uint8_t(0));const uint64_t h[]={magic,version,manifest_bytes,s.generation,s.job,key_.word[0],key_.word[1],key_.word[2],key_.word[3],
        uint64_t(s.next),s.charged,s.reserved,s.certified,s.integer,s.count,s.slots};
    for(size_t i=0;i<std::size(h);++i)put64(b.data()+8*i,h[i]);
    for(unsigned i=0;i<s.count;++i){const auto&r=s.records[i];const auto&id=r.region.id;
        const uint64_t words[]={r.role,id.uuid_low,id.uuid_high,id.incarnation,r.region.index_instance,r.region.layout.logical_bytes,r.region.revision,
            std::bit_cast<uint64_t>(r.exponent),r.negative,r.first,r.words,r.extent.first,r.extent.end,r.extent.top_word,r.extent.residue_m61};
        for(size_t j=0;j<std::size(words);++j)put64(b.data()+header_bytes+i*record_bytes+8*j,words[j]);
    }
    put64(b.data()+manifest_bytes-8,metadata_checksum(b.first(manifest_bytes-8)));
}
IoError Checkpoint::decode(const JobPlan&p,std::span<uint8_t>b,Snapshot&out) const noexcept {
    if(b.size()!=manifest_bytes||get64(b.data())!=magic||get64(b.data()+8)!=version||get64(b.data()+16)!=manifest_bytes||
       get64(b.data()+manifest_bytes-8)!=metadata_checksum(b.first(manifest_bytes-8)))return {IoCode::corrupt,"checkpoint header/checksum"};
    if(get64(b.data()+32)!=p.plan_id)return {IoCode::incompatible,"checkpoint job identity"};
    for(unsigned i=0;i<4;++i)if(get64(b.data()+40+8*i)!=key_.word[i])return {IoCode::incompatible,"checkpoint build/schema key"};
    Snapshot s{};s.generation=get64(b.data()+24);s.job=p.plan_id;
    const uint64_t phase=get64(b.data()+72),count=get64(b.data()+112);
    if(!s.generation||phase>uint64_t(JobPhase::complete)||count>5)return {IoCode::corrupt,"checkpoint generation/phase/count"};
    s.next=JobPhase(phase);s.charged=get64(b.data()+80);s.reserved=get64(b.data()+88);s.certified=get64(b.data()+96);s.integer=get64(b.data()+104);
    s.count=unsigned(count);s.slots=get64(b.data()+120);
    if(!s.slots||s.slots>SIZE_MAX/64||s.reserved>s.charged||s.certified>p.spec.digits||
       (s.next!=JobPhase::complete&&(s.certified||s.integer))||
       (s.reserved&&(s.next==JobPhase::complete||s.reserved!=p.phase_writes[unsigned(s.next)])))return {IoCode::corrupt,"checkpoint ledger/certificate"};
    unsigned mask=0;
    for(unsigned i=0;i<s.count;++i){auto&r=s.records[i];const uint8_t*x=b.data()+header_bytes+i*record_bytes;
        const uint64_t role=get64(x);if(role>4||(mask&(1u<<role)))return {IoCode::corrupt,"checkpoint duplicate role"};r.role=unsigned(role);mask|=1u<<role;
        r.region.id={get64(x+8),get64(x+16),get64(x+24)};r.region.index_instance=get64(x+32);r.region.revision=get64(x+48);
        r.region.kind=RegionKind::value;r.region.state=RegionState::sealed;
        if(!r.region.id.incarnation||!r.region.index_instance||!r.region.revision||region_layout(get64(x+40),0,r.region.layout))return {IoCode::corrupt,"checkpoint value descriptor"};
        r.exponent=std::bit_cast<int64_t>(get64(x+56));if(get64(x+64)>1)return {IoCode::corrupt,"checkpoint sign"};r.negative=get64(x+64);
        r.first=size_t(get64(x+72));r.words=size_t(get64(x+80));r.extent={get64(x+88),get64(x+96),get64(x+104),get64(x+112)};
        if(role<4){const size_t n=size_t(r.region.layout.logical_bytes/8);
            if(r.region.layout.logical_bytes%8||r.first>n||r.words>n-r.first||r.extent.end>n||
               (r.extent.end&&(!r.extent.top_word||r.extent.first>=r.extent.end)))return {IoCode::corrupt,"checkpoint integer range"};
        }else if(r.region.layout.logical_bytes!=p.spec.digits)return {IoCode::corrupt,"checkpoint digit capacity"};
    }
    if(mask!=required(s.next,p.fused_root))return {IoCode::corrupt,"checkpoint phase values"};out=s;return {};
}
IoError Checkpoint::read_head(std::span<uint8_t>b,bool&found) noexcept {
    found=false;const int fd=openat(directory_,head_name,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return errno==ENOENT?IoError{}:failure("checkpoint head open");
    struct End {int fd;~End(){::close(fd);}} end{fd};struct stat st{};if(fstat(fd,&st))return failure("checkpoint head stat");
    if(!S_ISREG(st.st_mode)||st.st_size!=manifest_bytes)return {IoCode::corrupt,"checkpoint head length"};
    ssize_t n;do{n=pread(fd,b.data(),manifest_bytes,0);}while(n<0&&errno==EINTR);
    ++usage_.manifest_reads;if(n>0)usage_.manifest_bytes_read+=uint64_t(n);
    if(n<0)return failure("checkpoint head read");if(n!=manifest_bytes)return {IoCode::short_io,"checkpoint head short read"};found=true;return {};
}
void Checkpoint::promote() noexcept {
    clear(pins_);for(unsigned i=0;i<5;++i)pins_[i]=std::move(pending_pins_[i]);current_=pending_;current_image_=pending_image_;
    pending_={};unknown_=false;exists_=loaded_=true;files_->suspend_creates(needs_gc_);++usage_.commits;
}
IoError Checkpoint::publish(const JobPlan&p,const JobState&s,uint64_t reservation,bool ignore_cancel,Cancel cancel) noexcept {
    if(directory_<0||!loaded_||needs_gc_)return {IoCode::invalid,"checkpoint requires restore/GC"};
    if(unknown_)return {IoCode::commit_unknown,"checkpoint unresolved publication"};
    if(!ignore_cancel&&cancel.test())return {IoCode::cancelled,"checkpoint before critical section"};
    if(usage_.manifest_bytes_submitted>manifest_write_limit_||manifest_bytes>manifest_write_limit_-usage_.manifest_bytes_submitted)
        return {IoCode::capacity,"checkpoint manifest write allowance",0,manifest_bytes,
                usage_.manifest_bytes_submitted<=manifest_write_limit_?manifest_write_limit_-usage_.manifest_bytes_submitted:0};
    clear(pending_pins_);if(auto e=capture(p,s,reservation,pending_,pending_pins_))return e;encode(pending_,pending_image_);
    int fd=-1;bool renamed=false,owned=false;char temporary[96]{};
    if(attempt_==UINT64_MAX){clear(pending_pins_);return {IoCode::overflow,"checkpoint temporary sequence"};}
    snprintf(temporary,sizeof temporary,"sbn-ckpt-%016llx-%016llx.next",(unsigned long long)nonce_,(unsigned long long)++attempt_);
    auto abort=[&](IoError e){if(fd>=0){::close(fd);fd=-1;}if(renamed){unknown_=true;files_->suspend_creates(true);return IoError{IoCode::commit_unknown,e.where,e.offset,e.need,e.have,e.system_error};}
        if(owned)unlinkat(directory_,temporary,0);clear(pending_pins_);pending_={};return e;};
    if(auto e=files_->collect())return abort(e);
    if(auto e=event(CheckpointEvent::values_sync))return abort(e);
    for(unsigned i=0;i<pending_.count;++i){IoError e;do{e=files_->sync(pending_pins_[i].region());}while(e&&e.system_error==EINTR);if(e)return abort(e);}
    // File data sync alone does not make newly created filenames durable.
    // Commit their directory entries before a manifest can point at them.
    if(auto e=event(CheckpointEvent::data_directory_sync))return abort(e);
    if(auto e=sync_fd(directory_,"checkpoint data directory sync"))return abort(e);
    if(auto e=event(CheckpointEvent::manifest_open))return abort(e);
    fd=openat(directory_,temporary,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);if(fd<0)return abort(failure("checkpoint temporary open"));owned=true;
    if(auto e=event(CheckpointEvent::manifest_write))return abort(e);
    ssize_t n;
    do{
        if(usage_.manifest_bytes_submitted>manifest_write_limit_||manifest_bytes>manifest_write_limit_-usage_.manifest_bytes_submitted)
            return abort({IoCode::capacity,"checkpoint manifest retry allowance"});
        usage_.manifest_bytes_submitted+=manifest_bytes;++usage_.manifest_writes;
        n=pwrite(fd,pending_image_.data(),manifest_bytes,0);if(n>0)usage_.manifest_bytes_written+=uint64_t(n);
    }while(n<0&&errno==EINTR);
    if(n<0)return abort(failure("checkpoint manifest write"));if(n!=manifest_bytes)return abort({IoCode::short_io,"checkpoint manifest short write"});
    if(auto e=event(CheckpointEvent::manifest_sync))return abort(e);
    if(auto e=sync_fd(fd,"checkpoint manifest sync"))return abort(e);
    {const int closed=fd;fd=-1;if(::close(closed))return abort(failure("checkpoint manifest close"));}
    if(auto e=event(CheckpointEvent::rename_root))return abort(e);
    // Once rename is attempted, conservatively retain both roots even if the
    // syscall reports an error. Resolution determines which name is visible.
    renamed=true;if(renameat(directory_,temporary,directory_,head_name))return abort(failure("checkpoint rename"));
    if(auto e=event(CheckpointEvent::directory_sync))return abort(e);
    if(auto e=sync_fd(directory_,"checkpoint publication directory sync"))return abort(e);
    promote();return {};
}
IoError Checkpoint::commit(const JobPlan&p,const JobState&s,Cancel cancel) noexcept{return publish(p,s,0,false,cancel);}
IoError Checkpoint::resolve() noexcept {
    if(!unknown_)return {};
    std::array<uint8_t,manifest_bytes>image{};bool found=false;if(auto e=read_head(image,found))return e;
    const bool absent=!found&&!current_.generation;
    if(!found&&!absent)return {IoCode::commit_unknown,"checkpoint root disappeared"};
    const bool next=found&&image==pending_image_,old=absent||(current_.generation&&image==current_image_);
    if(!next&&!old)return {IoCode::commit_unknown,"checkpoint root differs from both generations"};
    if(auto e=sync_fd(directory_,"checkpoint resolution directory sync"))return {IoCode::commit_unknown,e.where,0,0,0,e.system_error};
    ++usage_.resolves;if(next)promote();else{clear(pending_pins_);pending_={};unknown_=false;files_->suspend_creates(needs_gc_);}return {};
}
bool Checkpoint::matches(const JobState&s) const noexcept {
    if(!current_.generation||current_.reserved||current_.job!=s.plan_id||current_.next!=s.next||current_.charged!=s.charged_write_bytes||
       current_.certified!=s.text.certified_digits||current_.integer!=s.text.integer_part)return false;
    for(unsigned i=0;i<current_.count;++i){const auto&r=current_.records[i];const auto&v=value(s,r.role);
        if(!v||v.region().id!=r.region.id||v.region().index_instance!=r.region.index_instance||v.region().layout.logical_bytes!=r.region.layout.logical_bytes)return false;
        if(const auto*x=integer(s,r.role))if(x->exponent2!=r.exponent||x->negative!=r.negative||x->first_limb!=r.first||x->limbs()!=r.words||
            x->extent.first!=r.extent.first||x->extent.end!=r.extent.end||x->extent.top_word!=r.extent.top_word||x->extent.residue_m61!=r.extent.residue_m61)return false;
    }return true;
}
IoError Checkpoint::restore(const JobPlan&p,JobState&out,Transport&t,ValueStore&store,std::span<uint8_t>w) noexcept {
    if(directory_<0||!p.ready||unknown_||out.plan_id||out.working.value||out.binary.value||out.text.digits||store.usage().live_regions)
        return {IoCode::invalid,"checkpoint restore requires empty state/store"};
    for(const auto&v:out.pair)if(v.value)return {IoCode::invalid,"checkpoint restore pair"};
    bool found=false;std::array<uint8_t,manifest_bytes>image{};if(auto e=read_head(image,found))return e;
    if(!found){exists_=false;loaded_=true;files_->suspend_creates(false);return {};}
    Snapshot snapshot;if(auto e=decode(p,image,snapshot))return e;
    const size_t count=std::max(size_t(snapshot.slots),files_->slot_capacity()),needed=checkpoint_storage_bytes(count);
    if(!needed||!w.data()||(uintptr_t(w.data())&4095)||w.size()<needed)return {IoCode::capacity,"checkpoint restore workspace",0,needed,w.size()};
    JobState state;state.plan_id=snapshot.job;state.next=snapshot.next;state.charged_write_bytes=snapshot.charged;
    state.text.certified_digits=snapshot.certified;state.text.integer_part=snapshot.integer;
    StoreBuffers io{w.first(transfer_bytes),w.subspan(transfer_bytes,index_page_bytes)};auto data=w.subspan(transfer_bytes+index_page_bytes,transfer_bytes);
    for(unsigned i=0;i<snapshot.count;++i){const auto&r=snapshot.records[i];StoredValue loaded;
        if(auto e=store.import(r.region,loaded))return e;
        ValueReader reader;if(auto e=reader.open(t,loaded.region(),io))return e;WordExtent observed;
        for(uint64_t at=0;at<r.region.layout.logical_bytes;){const size_t n=size_t(std::min(uint64_t(data.size()),r.region.layout.logical_bytes-at));
            if(auto e=reader.read(at,data.first(n)))return e;if(r.role<4)observed.observe(at/8,data.first(n));at+=n;}
        if(r.role<4){
            if(observed.first!=r.extent.first||observed.end!=r.extent.end||observed.top_word!=r.extent.top_word||observed.residue_m61!=r.extent.residue_m61)
                return {IoCode::corrupt,"checkpoint integer summary/m61"};
            auto*x=integer(state,r.role);x->value=std::move(loaded);x->exponent2=r.exponent;x->negative=r.negative;x->first_limb=r.first;x->word_count=r.words;x->extent=r.extent;x->extent_known=true;
        }else state.text.digits=std::move(loaded);
    }
    // A visible HEAD left by an interrupted rename must become durable before
    // orphan GC is allowed to remove the previous root's files.
    if(auto e=sync_fd(directory_,"checkpoint restored namespace sync"))return e;
    for(unsigned i=0;i<snapshot.count;++i)if(auto e=files_->adopt(value(state,snapshot.records[i].role).region()))return e;
    clear(pins_);for(unsigned i=0;i<snapshot.count;++i)value(state,snapshot.records[i].role).share(pins_[i]);
    current_=snapshot;current_image_=image;loaded_=exists_=true;needs_gc_=true;
    out=std::move(state);return collect(w);
}
IoError Checkpoint::collect(std::span<uint8_t>w) noexcept {
    if(!loaded_||!current_.generation||unknown_)return {IoCode::invalid,"checkpoint GC requires resolved root"};
    const size_t count=std::max(size_t(current_.slots),files_->slot_capacity());if(count>SIZE_MAX-16)return {IoCode::overflow,"checkpoint GC capacity"};
    if(auto e=files_->collect_orphans(w,count+16,true))return e;needs_gc_=false;files_->suspend_creates(false);return {};
}
IoError Checkpoint::advance(JobPlan&p,JobState&s,Transport&t,ValueStore&store,sbn3_arena&arena,size_t at,sbn3_team&team) noexcept {
    if(unknown_)return {IoCode::commit_unknown,"checkpoint advance unresolved"};
    if(needs_gc_)return {IoCode::incomplete,"checkpoint advance requires GC"};
    if(!s.plan_id&&s.next==JobPhase::series)s.plan_id=p.plan_id;
    if(current_.generation&&!matches(s))return publish(p,s,0,false,t.cancel);
    if(s.next==JobPhase::complete)return current_.generation?IoError{}:commit(p,s,t.cancel);
    if(unsigned(s.next)>=5||s.plan_id!=p.plan_id)return {IoCode::invalid,"checkpoint advance state"};
    uint64_t required_bytes=0;for(unsigned i=unsigned(s.next);i<5;++i){if(__builtin_add_overflow(required_bytes,p.phase_writes[i],&required_bytes))return {IoCode::overflow,"checkpoint phase write sum"};}
    if(s.charged_write_bytes>p.write_limit||required_bytes>p.write_limit-s.charged_write_bytes)return {IoCode::capacity,"checkpoint next attempt allowance"};
    if(auto e=publish(p,s,p.phase_writes[unsigned(s.next)],false,t.cancel))return e;
    const auto execution=advance_job(p,s,t,store,arena,at,team);
    // Reconcile even when cancellation was latched during arithmetic/IO.
    if(auto e=publish(p,s,0,true,{}))return e;return execution;
}
IoError Checkpoint::close_preserving() noexcept {
    if(directory_<0)return {};
    for(auto*array:{pins_,pending_pins_})for(unsigned i=0;i<5;++i)if(array[i])if(auto e=files_->preserve(array[i].region()))return e;
    clear(pins_);clear(pending_pins_);const int lock=lock_,dir=directory_;lock_=directory_=-1;
    const int a=::close(lock),error=a?errno:0;const int b=::close(dir);files_=nullptr;
    if(error)return failure("checkpoint lock close",error);if(b)return failure("checkpoint directory close");return {};
}
} // namespace sbn::v3::external
