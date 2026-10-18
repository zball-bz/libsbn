// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/float_constants.hpp"
#include "float/constant_bank.hpp"
namespace sbn::detail {
struct constant_access {
    static v3::floating::ConstantBankView internal(constant_view view)noexcept{
        return {view.state_,view.storage_,view.provided_.pi_bits,view.provided_.ln2_bits};
    }
    static constant_view external(v3::floating::ConstantBankView view)noexcept{
        constant_view out;out.state_=view.state;out.storage_=view.storage;
        out.provided_={view.pi_bits,view.log2_bits};return out;
    }
};
}
