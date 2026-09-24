#pragma once
#include "product/local_windows.hpp"

namespace sbn::v3::product {
// Process-local plan records have an explicit word layout: no pointers or
// struct padding are copied into the public opaque plan. Kernel choices are
// exactly those of LocalWindowFactory; execution only decodes these records.
struct LocalWindowStep {
    uint64_t words[11]{};
    static void shape(uint64_t *w,pq16::Shape s) noexcept {
        w[0]=s.nfull|(uint64_t(s.branch)<<32);
        w[1]=s.radix|(uint64_t(s.recipe)<<8)|(uint64_t(s.bits)<<16)|
             (uint64_t(s.centered)<<24)|(uint64_t(s.balanced)<<25);
    }
    static pq16::Shape shape(const uint64_t *w) noexcept {
        return {uint32_t(w[0]),uint32_t(w[0]>>32),unsigned(w[1]&255),bool(w[1]>>24&1),
                pq16::Recipe(w[1]>>8&255),unsigned(w[1]>>16&255),bool(w[1]>>25&1)};
    }
    static LocalWindowStep compile(const WindowGroupShape &s) noexcept {
        LocalWindowStep out{};auto *w=out.words;w[0]=s.products[s.count-2].bound.words;
        w[1]=uint64_t(s.count)<<8;
        const auto base=local_windows_query(s);
        const auto peel=peel_window_possible(s)?peeled_windows_query(s,base):PeeledWindowPlan{};
        if(peel.ring){
            w[1]|=1;shape(w+2,peel.fft);w[4]=peel.ring;w[5]=peel.capacity;w[6]=peel.bytes;
            w[7]=peel.residual_words;w[8]=peel.shared_mask|(uint64_t(peel.count)<<32);
        }else{
            shape(w+2,base.shared);shape(w+4,base.cancel_shape);w[6]=base.value_words;
            w[7]=base.correction_words;w[8]=base.work_bytes;w[9]=base.storage_bytes;
            w[10]=base.count|(uint64_t(base.cancellation)<<8)|(uint64_t(base.shared_mask)<<16)|
                  (uint64_t(base.middle)<<24);
        }
        return out;
    }
    size_t bytes() const noexcept {return words[1]&1?words[6]:words[9];}
    bool matches(const WindowGroupShape &s) const noexcept {
        return words[0]==s.products[s.count-2].bound.words&&words[1]>>8==s.count;
    }
    PeeledWindowPlan peeled_plan() const noexcept {
        const auto*w=words;
        return {shape(w+2),w[4],w[5],w[6],w[7],unsigned(w[8]),unsigned(w[8]>>32)};
    }
    LocalWindowPlan local_plan() const noexcept {
        const auto*w=words;
        return {shape(w+2),shape(w+4),w[6],w[7],w[8],w[9],unsigned(w[10]&255),
                unsigned(w[10]>>8&255),unsigned(w[10]>>16&255),bool(w[10]>>24&1)};
    }
    template<class F>void execute(Frame &space,uint64_t *out,F &&f)const noexcept {
        if(words[1]&1){
            const auto plan=peeled_plan();
            FrameMark mark(space);PeeledWindows products(plan,space);f(products);
        }else{
            const auto plan=local_plan();
            LocalWindowProducts products(plan,space,out);f(products);
        }
    }
};
inline bool local_program_eligible(size_t repeated_operand_words) noexcept {
    return repeated_operand_words>=local_window_detail::fft_min_words;
}
struct LocalWindowProgram {
    static constexpr unsigned capacity=10;
    // Conservative bound for the halving ladder below the local ceiling.
    static_assert(native_policy::window_inline_words<=
                  (local_window_detail::fft_min_words<<(capacity-1)));
    LocalWindowStep steps[capacity]{};
    unsigned count=0;
};
struct LocalWindowCompiler {
    LocalWindowProgram &program;
    size_t bytes(const WindowGroupShape &s)const noexcept {
        if(!local_program_eligible(s.products[0].a.words))return LocalWindowFactory{}.bytes(s);
        require(program.count<LocalWindowProgram::capacity,SBN3_FATAL_WORKSPACE,"local product program capacity");
        auto &step=program.steps[program.count++];step=LocalWindowStep::compile(s);return step.bytes();
    }
    size_t retained_words(const WindowGroupShape&s)const noexcept {return LocalWindowFactory{}.retained_words(s);}
};
struct LocalWindowReplay {
    const LocalWindowStep *steps=nullptr;
    unsigned count=0;
    size_t retained_words(const WindowGroupShape&s)const noexcept {return LocalWindowFactory{}.retained_words(s);}
    template<class F>void with_group(const WindowGroupShape&s,Frame&space,uint64_t*out,F&&f)const noexcept {
        if(!local_program_eligible(s.products[0].a.words)){LocalWindowFactory{}.with_group(s,space,out,f);return;}
        for(unsigned j=0;j<count;++j)if(steps[j].matches(s)){steps[j].execute(space,out,f);return;}
        fatal(SBN3_FATAL_MATH,"local product program does not cover refinement");
    }
};
}
