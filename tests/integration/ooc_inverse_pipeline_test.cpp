// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/inverse_planes.hpp"
#include "product/modular_row.hpp"
#include "runtime/team.hpp"

struct ReadCancelDevice : Device {
    uint64_t cancel_after_read=0;
    explicit ReadCancelDevice(size_t n):Device(n){}
    static Completion read_then_cancel(void*p,uint64_t at,std::span<uint8_t>w) noexcept {
        auto&d=*static_cast<ReadCancelDevice *>(static_cast<Device *>(p));const auto c=Device::read(p,at,w);
        if(d.cancel_after_read&&d.reads==d.cancel_after_read)d.cancelled=true;return c;
    }
};
static void one(unsigned np,unsigned batch,unsigned fault=0){
    product::OuterTransformSpec spec{8,2048,3,1024,0,128,np,batch,4};
    InversePlanesPlan plan;assert(query_inverse_planes(spec,plan));auto serial=plan;serial.pipeline=false;
    assert(plan.pipeline==(batch<np)&&plan.storage_bytes==serial.storage_bytes);
    std::array<Region,10> regions;uint64_t end=0;
    for(unsigned q=0;q<np;++q){regions[q]=fresh(size_t(plan.plane_bytes),end,700+q,RegionKind::scratch);end+=regions[q].layout.allocation_bytes;}
    ReadCancelDevice device{size_t(end)};auto t=device.transport();t.read=ReadCancelDevice::read_then_cancel;
    Aligned io(4*chunk_bytes),page(index_page_bytes),wire(size_t(plan.plane_bytes));StoreBuffers buffers{io.span(),page.span()};
    std::array<ScratchPlane,10> planes;std::array<std::unique_ptr<Aligned>,10> indexes,dirty;
    PlaneIndexStorage index;assert(!plane_index_storage(regions[0].layout,index));
    for(unsigned q=0;q<np;++q){const uint64_t prime=product::p48_row_prime(q);
        for(size_t i=0;i<wire.bytes/6;++i){const uint64_t value=random_word()%prime;std::memcpy(wire.data+6*i,&value,6);}
        write_value(t,regions[q],buffers,wire.span());indexes[q]=std::make_unique<Aligned>(index.image_bytes);dirty[q]=std::make_unique<Aligned>(index.dirty_bytes);
        assert(!planes[q].take(t,regions[q],buffers,indexes[q]->span(),dirty[q]->span()));}
    const auto preserved=device.bytes;
    Fixture f(4,false,64u<<20);auto a=f.allocate(plan.storage_bytes,4096),b=f.allocate(plan.storage_bytes,4096);
    InversePlanesBinding baseline{},pipeline{};
    prepare_inverse_planes(serial,{static_cast<uint8_t *>(a.data),a.bytes},*f.team,baseline);
    prepare_inverse_planes(plan,{static_cast<uint8_t *>(b.data),b.bytes},*f.team,pipeline);
    if(fault==1)device.fail_read=device.reads+5;
    if(fault==2)device.short_read=device.reads+5;
    if(fault==3){device.corrupt_at=regions[1].layout.data_offset+17;device.corrupt_reads=-1;}
    if(fault==4)device.cancel_after_read=device.reads+4;
    if(fault){CoefficientStripe result{};allocation_watch_start();
        const auto error=read_inverse_stripe(pipeline,0,{planes.data(),np},result);
        assert(!allocation_watch_stop()&&error&&result.data.empty()&&!f.team->busy);
    }else for(size_t column=0;column<spec.inner_cells;column+=spec.stripe_cells){
        CoefficientStripe left{},right{};const auto before=t.counters;allocation_watch_start();
        assert(!read_inverse_stripe(baseline,column,{planes.data(),np},left));const auto middle=t.counters;
        assert(!read_inverse_stripe(pipeline,column,{planes.data(),np},right));assert(!allocation_watch_stop());
        assert(left.data.size()==right.data.size()&&!memcmp(left.data.data(),right.data.data(),left.data.size_bytes()));
        assert(left.first_column==right.first_column&&left.rows==right.rows&&left.primes==right.primes);
        assert(middle.read_submitted-before.read_submitted==t.counters.read_submitted-middle.read_submitted);
        assert(t.counters.write_submitted==before.write_submitted);
    }
    assert(device.bytes==preserved);for(unsigned q=0;q<np;++q)planes[q].discard();
    printf("Inverse pipeline NP%u batch%u fault%u: exact bank, unchanged reads/storage, immutable source, joined failure PASS\n",np,batch,fault);
}
int main(){setvbuf(stdout,nullptr,_IOLBF,0);one(6,1);one(10,4);one(6,6);for(unsigned fault=1;fault<=4;++fault)one(6,1,fault);}
