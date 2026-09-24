#pragma once
#include "sbn3/divrem.h"
namespace sbn::v3 {
// Private experiment/recipe entry; normal bind/prepare/execute/unbind apply.
// No runtime candidate search. Prepare retains D and the half reciprocal.
sbn3_query_result fused_divrem_query(const sbn3_divrem_request *,const sbn3_divrem_options *,
                                     sbn3_divrem_plan *,sbn3_divrem_info *);
}
