// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "runtime/linux/sync.hpp"

namespace sbn::v3::external {
// One ordered IO owner and a fixed compute owner per slot. One compute owner
// may alternate several slots. Payloads live in the caller's quoted grant.
// Each direction owns its park flag: a previous wake
// can still be unwinding while the peer starts its next wait on the same slot.
template<unsigned Count>
class SlotExchange {
    static_assert(Count>0);
    struct alignas(64) Slot {uint32_t state=0,io_parked=0,compute_parked=0;} slots_[Count]{};
    bool stopped_=false;
public:
    enum State : uint32_t { empty=0,ready=1,computed=2,abandoned=3 };
    bool stopped() const noexcept {return __atomic_load_n(&stopped_,__ATOMIC_ACQUIRE);}
    bool wait(unsigned i,State desired) noexcept {
        for(;;){
            if(stopped())return false;
            const auto state=__atomic_load_n(&slots_[i].state,__ATOMIC_ACQUIRE);
            if(state==abandoned)return false;
            if(state==desired)return true;
            os::wait_epoch(&slots_[i].state,state,desired==computed?&slots_[i].io_parked:&slots_[i].compute_parked);
        }
    }
    void publish(unsigned i,State state) noexcept {
        __atomic_store_n(&slots_[i].state,uint32_t(state),__ATOMIC_SEQ_CST);
        if(__atomic_load_n(&slots_[i].io_parked,__ATOMIC_SEQ_CST)||__atomic_load_n(&slots_[i].compute_parked,__ATOMIC_SEQ_CST))
            os::wake(&slots_[i].state,2);
    }
    void stop() noexcept {__atomic_store_n(&stopped_,true,__ATOMIC_RELEASE);for(unsigned i=0;i<Count;++i)publish(i,abandoned);}
};
using TwoSlotExchange=SlotExchange<2>;
// Three actors require three park flags: no returning waiter may clear another
// actor's newly parked state. Read -> compute -> write -> empty is ordered.
class ThreeSlotExchange {
    struct alignas(64) Slot {uint32_t state=0,parked[3]{};} slots_[3]{};
    bool stopped_=false;
public:
    enum State:uint32_t {empty,ready,computed,abandoned};
    enum Role:unsigned {reader,computer,writer};
    bool stopped()const noexcept{return __atomic_load_n(&stopped_,__ATOMIC_ACQUIRE);}
    bool wait(unsigned i,State desired,Role role) noexcept {
        for(;;){if(stopped())return false;const auto at=__atomic_load_n(&slots_[i].state,__ATOMIC_ACQUIRE);
            if(at==abandoned)return false;if(at==desired)return true;
            os::wait_epoch(&slots_[i].state,at,&slots_[i].parked[role],10000);}
    }
    void publish(unsigned i,State state) noexcept {
        __atomic_store_n(&slots_[i].state,uint32_t(state),__ATOMIC_SEQ_CST);
        bool wake=false;for(unsigned r=0;r<3;++r)wake|=__atomic_load_n(&slots_[i].parked[r],__ATOMIC_SEQ_CST)!=0;
        if(wake)os::wake(&slots_[i].state,3);
    }
    void stop()noexcept{__atomic_store_n(&stopped_,true,__ATOMIC_RELEASE);for(unsigned i=0;i<3;++i)publish(i,abandoned);}
};
} // namespace sbn::v3::external
