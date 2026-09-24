#include "sbn3/newton.h"
#include "algorithms/newton_limits.hpp"
#include "algorithms/newton_contract.hpp"
#include "algorithms/newton_planner.hpp"
#include "algorithms/newton_tuning.hpp"
#include "product/local_program.hpp"
#include <math.h>
#include <assert.h>
#include <initializer_list>
#include <stdio.h>
#include <string.h>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();

int main() {
    using namespace sbn::v3;
    unsigned cases = 0;
    for (auto kind : {SBN3_NEWTON_INVERSE, SBN3_NEWTON_RSQRT, SBN3_NEWTON_DIVIDE}) {
        sbn3_newton_options o{16, 0, 0, 0};
        sbn3_newton_plan p{};
        sbn3_newton_info i{};
        assert(sbn3_newton_query(kind, size_t(1) << 24, &o, &p, &i) == SBN3_SUPPORTED);
        // A long ladder owns only current-phase mutable leases, not one set
        // per rung. This gate catches accidental restoration of accumulation.
        assert(i.stages > 10 && i.lease_peak <= 8);
    }
    for (unsigned kind = 0; kind <= SBN3_SQRT2_RATIONAL; ++kind)
        for (unsigned workers : {1u, 32u})
            for (size_t n : {size_t(0), size_t(1), size_t(3), size_t(4), size_t(15), size_t(16),
                             newton_limits::precision_words - 1, newton_limits::precision_words,
                             newton_limits::precision_words + 1}) {
                sbn3_newton_options options{workers, 0, 0, 0};
                sbn3_newton_plan plan{};
                sbn3_newton_info info{};
                allocation_watch_start();
                auto result = sbn3_newton_query(sbn3_newton_kind(kind), n, &options, &plan, &info);
                assert(!allocation_watch_stop());
                if (!n)
                    assert(result == SBN3_UNSUPPORTED);
                else if (n > newton_limits::precision_words)
                    assert(result == SBN3_QUERY_CAPACITY);
                else {
                    assert(result == SBN3_SUPPORTED || result == SBN3_QUERY_CAPACITY);
                    if (result == SBN3_SUPPORTED) {
                        assert(info.products <= newton_limits::products &&
                               info.spectra <= newton_limits::spectra);
                        assert(info.lease_peak <= newton_limits::leases);
                        const bool dyadic=plan.opaque[0]==newton_detail::dyadic_plan_magic;
                        newton_detail::Plan internal{};if(!dyadic)memcpy(&internal,plan.opaque,sizeof internal);
                        if(dyadic){
                            newton_detail::DyadicPlan direct;memcpy(&direct,plan.opaque,sizeof direct);
                            assert(kind==SBN3_NEWTON_DIVIDE&&!info.spectra&&info.lease_peak<=1);
                            assert(direct.recipe==2?info.products>0:info.products==0);
                            assert(info.control_bytes<=256&&info.storage_bytes>=info.control_bytes+info.shared_bytes);
                        }else if(internal.local){
                            assert(info.value_bytes==0&&info.control_bytes<=256+128*((internal.local_steps*sizeof(product::LocalWindowStep)+127)/128)&&info.lease_peak==1);
                            assert(info.storage_bytes>=info.control_bytes+info.shared_bytes);
                        } else if (kind == SBN3_NEWTON_INVERSE || kind == SBN3_NEWTON_RSQRT) {
                            const size_t seed = kind == SBN3_NEWTON_INVERSE ? 15 : 3;
                            // Compact execution sends the last rung directly
                            // to caller output; internal values need no final n.
                            assert(info.value_bytes <= 16 * (n + 1));
                            assert(16 * newton_contract::minimum_approximation_words(n, seed) <=
                                   info.value_bytes);
                        }
                        const auto saved = plan;
                        options.memory_budget = info.storage_bytes - 1;
                        sbn3_newton_info required{};
                        allocation_watch_start();
                        result = sbn3_newton_query(sbn3_newton_kind(kind), n, &options, &plan, &required);
                        assert(!allocation_watch_stop());
                        assert(result == SBN3_QUERY_CAPACITY && required.storage_bytes == info.storage_bytes);
                        assert(!memcmp(&saved, &plan, sizeof plan));
                    }
                }
                ++cases;
            }
    printf("Newton query capacity and value-layout bounds: %u cases, including maximum precision, PASS\n",
           cases);
    // The cycle chooser skips a lattice candidate once a lower bound of its cost, known after the
    // producer query, leaves the selection window. That is exact only if the bound never exceeds the
    // full cold cost: checked over the whole lattice for short and long rungs of every cycle kind.
    {
        using namespace newton_detail;
        unsigned checked = 0;
        double tightest = 0;
        for (auto kind : {Cycle::Inverse, Cycle::Rsqrt, Cycle::Division})
            for (size_t n : {size_t(8), size_t(33), size_t(100), size_t(512), size_t(4097), size_t(70000),
                             size_t(1) << 20, size_t(5000000)})
                for (size_t m : {n / 2 + 1, n}) {
                    if (kind == Cycle::Division && m == n)
                        continue;
                    const size_t minimum = kind == Cycle::Inverse ? newton_contract::inverse_ring_min(n)
                                           : kind == Cycle::Rsqrt ? newton_contract::rsqrt_ring_min(m)
                                                                  : newton_contract::division_ring_min(m);
                    for (unsigned np = newton_limits::first_ntt_prime_count; np <= newton_limits::last_ntt_prime_count; ++np)
                        for (int T = np == 4 ? 88 : 24 * int(np) - 8; T >= (np == 4 ? 80 : 24 * int(np) - 32);
                             T -= np == 4 ? 4 : 8) {
                            size_t ring = 2 * size_t(T);
                            while (ring < minimum)
                                ring *= 2;
                            for (unsigned algorithm : {unsigned(SBN3_MUL_FLAT), unsigned(SBN3_MUL_BAILEY)})
                                for (unsigned workers : {1u, 16u})
                                    for (bool compact : {false, true}) {
                                        double bound = 0, cost = 0;
                                        const Choice c{np, algorithm, workers, T, ring};
                                        if (!cycle_bound_probe(kind, m, n, c, compact, bound, cost))
                                            continue;
                                        assert(isfinite(bound) && bound >= 0 && bound <= cost);
                                        tightest = fmax(tightest, bound / cost);
                                        ++checked;
                                    }
                        }
                }
        assert(checked > 1000);
        printf("Newton lattice pruning bound: %u candidates, bound <= cold cost everywhere (largest ratio %.3f) PASS\n",
               checked, tightest);
    }
}
