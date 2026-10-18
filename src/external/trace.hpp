// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/io.hpp"
#include <time.h>
namespace sbn::v3::external {
class OperationSpan {
    Transport&t_;OperationTrace event_;uint64_t began_=0,reads_=0,writes_=0;
    static uint64_t tick(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
public:
    OperationSpan(Transport&t,const char*kind,uint64_t n=0,uint64_t m=0,unsigned np=0,unsigned T=0,size_t C=0,size_t M=0):
        t_(t),event_{kind,n,m,0,0,0,np,T,C,M}{
        if(t.trace){began_=tick();reads_=t.counters.read_submitted;writes_=t.counters.write_submitted;}
    }
    ~OperationSpan(){if(t_.trace){event_.elapsed_ns=tick()-began_;event_.read_bytes=t_.counters.read_submitted-reads_;
        event_.write_bytes=t_.counters.write_submitted-writes_;t_.trace(t_.trace_context,event_);}}
};
}
