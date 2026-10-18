// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/io.hpp"
#include "runtime/linux/sync.hpp"

namespace sbn::v3::external {
namespace {
struct Ledger {
    uint32_t&word;bool locked;
    Ledger(uint32_t&w,bool use):word(w),locked(use){if(locked)
        while(__atomic_exchange_n(&word,1,__ATOMIC_ACQUIRE))os::pause();}
    ~Ledger(){if(locked)__atomic_store_n(&word,0,__ATOMIC_RELEASE);}
};
IoError check(const Transport &t, uint64_t offset, const void *data, size_t bytes) {
    if (!t.alignment || (t.alignment & (t.alignment-1)) || !data || !bytes ||
        bytes > t.max_transfer || (uintptr_t(data) & (t.alignment-1)) ||
        (offset & (t.alignment-1)) || (bytes & (t.alignment-1)))
        return {IoCode::invalid, "transport request", offset, bytes, t.max_transfer};
    if (offset > t.capacity || bytes > t.capacity-offset)
        return {IoCode::capacity, "transport extent", offset, bytes, offset <= t.capacity ? t.capacity-offset : 0};
    if (t.cancel.test()) return {IoCode::cancelled, "transport cancellation", offset};
    return {};
}
IoError completed(Completion c, uint64_t offset, size_t requested) {
    if (c.bytes > requested) return {IoCode::invalid, "transport completion", offset, requested, c.bytes};
    if (c.code != IoCode::ok) return {c.code, "transport IO", offset, requested, c.bytes, c.system_error};
    if (c.bytes != requested) return {IoCode::short_io, "transport short IO", offset, requested, c.bytes};
    return {};
}
}
IoError read_exact(Transport &t, uint64_t offset, std::span<uint8_t> data) noexcept {
    if (!t.read) return {IoCode::invalid, "transport missing read"};
    if (auto error = check(t, offset, data.data(), data.size())) return error;
    {Ledger lock(t.read_ledger_lock,t.parallel_reads>1);
        if (data.size() > UINT64_MAX-t.counters.read_submitted || t.counters.reads == UINT64_MAX)
            return {IoCode::overflow, "transport read ledger"};
        t.counters.read_submitted += data.size(); ++t.counters.reads;
    }
    const auto c = t.read(t.context, offset, data);
    // Completed bytes cannot exceed cumulative submitted bytes for a valid provider.
    if (c.bytes <= data.size()){Ledger lock(t.read_ledger_lock,t.parallel_reads>1);t.counters.read_completed += c.bytes;}
    return completed(c, offset, data.size());
}
IoError write_exact(Transport &t, uint64_t offset, std::span<const uint8_t> data) noexcept {
    if (!t.write) return {IoCode::invalid, "transport missing write"};
    if (auto error = check(t, offset, data.data(), data.size())) return error;
    {Ledger lock(t.write_ledger_lock,t.parallel_writes>1);
        if (t.counters.write_submitted > t.write_limit || data.size() > t.write_limit-t.counters.write_submitted)
            return {IoCode::capacity, "transport write budget", offset, data.size(),
                    t.counters.write_submitted <= t.write_limit ? t.write_limit-t.counters.write_submitted : 0};
        if (data.size() > UINT64_MAX-t.counters.write_submitted || t.counters.writes == UINT64_MAX)
            return {IoCode::overflow, "transport write ledger"};
        t.counters.write_submitted += data.size(); ++t.counters.writes;
    }
    const auto c = t.write(t.context, offset, data);
    if (c.bytes <= data.size()){Ledger lock(t.write_ledger_lock,t.parallel_writes>1);t.counters.write_completed += c.bytes;}
    return completed(c, offset, data.size());
}
IoError read_batch_exact(Transport&t,std::span<ReadRequest> requests) noexcept {
    if(requests.empty())return {};
    if(!t.read||requests.size()>io_batch_max||(t.read_batch&&requests.size()>t.batch_limit))return {IoCode::invalid,"read batch capability"};
    uint64_t bytes=0;
    for(size_t i=0;i<requests.size();++i){auto&r=requests[i];r.result={IoCode::incomplete,0,0};
        if(auto e=check(t,r.offset,r.data.data(),r.data.size()))return e;
        const uintptr_t at=uintptr_t(r.data.data());if(at>UINTPTR_MAX-r.data.size())return {IoCode::overflow,"read batch pointer"};
        for(size_t k=0;k<i;++k){const auto&s=requests[k];const uintptr_t other=uintptr_t(s.data.data());
            if(at<other+s.data.size()&&other<at+r.data.size())return {IoCode::invalid,"read batch destination alias"};}
        if(r.data.size()>UINT64_MAX-bytes)return {IoCode::overflow,"read batch byte sum"};bytes+=r.data.size();
    }
    {Ledger lock(t.read_ledger_lock,t.parallel_reads>1);
        if(bytes>UINT64_MAX-t.counters.read_submitted||requests.size()>UINT64_MAX-t.counters.reads)return {IoCode::overflow,"read batch ledger"};
        t.counters.read_submitted+=bytes;t.counters.reads+=requests.size();}
    if(t.read_batch)t.read_batch(t.context,requests);
    else for(auto&r:requests)r.result=t.cancel.test()?Completion{IoCode::cancelled,0,0}:t.read(t.context,r.offset,r.data);
    IoError error{};uint64_t done=0;
    for(auto&r:requests){if(r.result.bytes<=r.data.size())done+=r.result.bytes;if(auto e=completed(r.result,r.offset,r.data.size());!error&&e)error=e;}
    {Ledger lock(t.read_ledger_lock,t.parallel_reads>1);t.counters.read_completed+=done;}
    if(!error&&t.cancel.test())error={IoCode::cancelled,"joined read batch cancellation"};return error;
}
IoError write_batch_exact(Transport&t,std::span<WriteRequest> requests) noexcept {
    if(requests.empty())return {};
    if(!t.write||requests.size()>io_batch_max||(t.write_batch&&requests.size()>t.batch_limit))return {IoCode::invalid,"write batch capability"};
    uint64_t bytes=0;
    for(size_t i=0;i<requests.size();++i){auto&r=requests[i];r.result={IoCode::incomplete,0,0};
        if(auto e=check(t,r.offset,r.data.data(),r.data.size()))return e;
        for(size_t k=0;k<i;++k){const auto&s=requests[k];if(r.offset<s.offset+s.data.size()&&s.offset<r.offset+r.data.size())return {IoCode::invalid,"write batch extent alias"};}
        if(r.data.size()>UINT64_MAX-bytes)return {IoCode::overflow,"write batch byte sum"};bytes+=r.data.size();
    }
    {Ledger lock(t.write_ledger_lock,t.parallel_writes>1);
        if(t.counters.write_submitted>t.write_limit||bytes>t.write_limit-t.counters.write_submitted)return {IoCode::capacity,"write batch budget",0,bytes,t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
        if(bytes>UINT64_MAX-t.counters.write_submitted||requests.size()>UINT64_MAX-t.counters.writes)return {IoCode::overflow,"write batch ledger"};
        t.counters.write_submitted+=bytes;t.counters.writes+=requests.size();}
    if(t.write_batch)t.write_batch(t.context,requests);
    else for(auto&r:requests)r.result=t.cancel.test()?Completion{IoCode::cancelled,0,0}:t.write(t.context,r.offset,r.data);
    IoError error{};uint64_t done=0;
    for(auto&r:requests){if(r.result.bytes<=r.data.size())done+=r.result.bytes;if(auto e=completed(r.result,r.offset,r.data.size());!error&&e)error=e;}
    {Ledger lock(t.write_ledger_lock,t.parallel_writes>1);t.counters.write_completed+=done;}
    if(!error&&t.cancel.test())error={IoCode::cancelled,"joined write batch cancellation"};return error;
}
} // namespace sbn::v3::external
