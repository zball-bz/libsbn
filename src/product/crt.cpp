// SPDX-License-Identifier: LGPL-3.0-or-later
#include "product/crt.hpp"
namespace sbn::v3::product {
const CrtBackend &p48_crt_4() noexcept;
const CrtBackend &p48_crt_5() noexcept;
const CrtBackend &p48_crt_6() noexcept;
const CrtBackend &p48_crt_7() noexcept;
const CrtBackend &p48_crt_8() noexcept;
const CrtBackend &p48_crt_9() noexcept;
const CrtBackend &p48_crt_10() noexcept;
const CrtBackend *p48_crt_backend(unsigned n) noexcept {
    switch(n){case 4:return &p48_crt_4();case 5:return &p48_crt_5();case 6:return &p48_crt_6();
    case 7:return &p48_crt_7();case 8:return &p48_crt_8();case 9:return &p48_crt_9();case 10:return &p48_crt_10();default:return nullptr;}
}
}
