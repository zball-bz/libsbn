// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
namespace sbn::v3::floating {
enum class ConstantKind { pi, log2 };
struct ConstantBankDemand {std::size_t pi_bits=0,log2_bits=0;};
struct ConstantBankQuote {
    support status=support::yes;
    std::size_t value_bytes=0,value_alignment=128,workspace_bytes=0,workspace_alignment=128;
    std::size_t pi_bits=0,log2_bits=0;
    unsigned workers=1;const char* reason="";
    bool supported()const noexcept{return status==support::yes;}
};
// The value span is immutable after publication. It is independent of the
// construction workspace, and lives until its owner next rebuilds/reserves
// or destroys it. Moving the owner without relocating storage is harmless.
struct ConstantBankView {
    const void* state=nullptr;std::span<const std::byte> storage{};
    std::size_t pi_bits=0,log2_bits=0;
};
struct ConstantValueView {float_view point{};error_bound error{};std::size_t precision=0;};
struct ConstantBankPrepared {void* state=nullptr;std::span<std::byte> workspace{};ConstantBankView pending{};};
class ConstantBankBinding {
    ConstantBankView values_{};bool pending_=false,query_=false;
    ConstantBankBinding(ConstantBankView v,bool pending)noexcept:values_(v),pending_(pending){}
    friend ConstantBankBinding constant_bank_pending_binding(const ConstantBankPrepared&)noexcept;
    friend ConstantBankBinding constant_bank_query_binding(ConstantBankDemand)noexcept;
public:
    ConstantBankBinding()noexcept=default;
    // External views are always checked as fully published at bind.
    ConstantBankBinding(ConstantBankView v)noexcept:values_(v){}
    ConstantBankView values()const noexcept{return values_;}
    bool pending()const noexcept{return pending_;}
    bool query_only()const noexcept{return query_;}
    bool supplied()const noexcept{return query_||values_.state!=nullptr;}
};
ConstantBankDemand constant_bank_merge(ConstantBankDemand,ConstantBankDemand) noexcept;
ConstantBankQuote constant_bank_requirements(ConstantBankDemand,unsigned=1) noexcept;
ConstantBankPrepared constant_bank_prepare(const ConstantBankQuote&,std::span<std::byte> values,
                                          std::span<std::byte> workspace) noexcept;
ConstantBankBinding constant_bank_pending_binding(const ConstantBankPrepared&) noexcept;
// Admission-only facts. Prepare must reject this token: it has no storage.
ConstantBankBinding constant_bank_query_binding(ConstantBankDemand) noexcept;
ConstantBankView constant_bank_execute(const ConstantBankPrepared&,execution={}) noexcept;
bool constant_bank_supports(ConstantBankBinding,ConstantBankDemand) noexcept;
void constant_bank_check_output(ConstantBankBinding,float_ref) noexcept;
ConstantValueView constant_bank_borrow(ConstantBankView,ConstantKind) noexcept;
// Reads only O(out.precision) high words, never scans the omitted low tail.
float_report constant_bank_prefix(ConstantBankBinding,ConstantKind,float_ref) noexcept;
}
