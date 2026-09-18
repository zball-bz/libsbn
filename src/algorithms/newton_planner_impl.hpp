#pragma once
// Private implementation fragment, included only by newton_service.cpp.
// Keep bind/replay and planning visible in one TU: the separate-TU version
// regressed short Newton calls. See docs/code-cleanup-results-2026-09-10.md.
#include "algorithms/newton_planner.hpp"
#include "algorithms/newton_contract.hpp"
#include "algorithms/newton_tuning.hpp"
#include "product/cost_model.hpp"
#include "product/root_prepare_cost.hpp"
#include "common/checked.hpp"
#include <algorithm>
#include <cmath>
namespace sbn::v3::newton_detail {
// Every cold cycle cost is the producer's forward share plus the producer's
// root preparation plus nonnegative terms (cycle_cost, cold_cycle_cost), and
// floating-point addition is monotone, so this never exceeds it.
double producer_lower_bound(Cycle kind, size_t m, size_t n, const Bundle &b) {
    const bool deep = (kind == Cycle::Inverse ? n : kind == Cycle::Rsqrt ? m + 1 : n + 1) >
                      native_policy::small_model_max_words;
    return cost_model::prepare_share(cost_model::cyclic_product(b.producer_info.mul, deep).nanoseconds) +
           root_prepare_cost(b.producer_info.mul);
}
// relevant/pruned (optional): after the producer query, the lower bound of
// this candidate's cold cost is compared with `relevant`; above it the
// remaining queries are skipped and *pruned is set. See choose_cycle.
bool cycle_candidate(Cycle kind, size_t m, size_t n, size_t ring, const Choice &c, Bundle &out,
                     double relevant = INFINITY, bool *pruned = nullptr) {
    sbn3_mul_options o{};
    o.workers = c.workers;
    o.prime_count = c.np;
    o.trunk_bits = c.T;
    o.algorithm = c.algorithm;
    // The division ring is temporary, so its backing can also hold E1 when
    // codec/row padding extends slightly beyond the mathematical ring.
    o.borrow_output = kind == Cycle::Division ? 2 : 1;
    sbn3_product_request r{};
    r.cyclic_limbs = ring;
    r.kind = kind == Cycle::Rsqrt ? SBN3_PRODUCT_SQR : SBN3_PRODUCT_MUL;
    r.window_limbs = kind == Cycle::Rsqrt ? m + newton_contract::guard_words : 0;
    r.a_limbs = kind == Cycle::Rsqrt ? m : m + 1;
    r.b_limbs = kind == Cycle::Rsqrt      ? 0
                : kind == Cycle::Division ? std::max(m + 1, newton_contract::residual_words(m, n))
                                          : n;
    if (sbn3_product_query(&r, &o, &out.producer, &out.producer_info) != SBN3_SUPPORTED)
        return false;
    if (c.algorithm == SBN3_MUL_SCALAR) {
        out.future = {};
        out.plans[0] = out.producer;
        out.infos[0] = out.producer_info;
        out.count = 1;
        if (kind != Cycle::Inverse) {
            if (kind == Cycle::Rsqrt) {
                r.kind = SBN3_PRODUCT_MUL;
                r.b_limbs = m + 1;
                r.window_limbs = 0;
            } else
                r.b_limbs = n;
            if (sbn3_product_query(&r, &o, &out.plans[1], &out.infos[1]) != SBN3_SUPPORTED)
                return false;
            out.count = 2;
        }
        return true;
    }
    if (pruned && producer_lower_bound(kind, m, n, out) > relevant) {
        *pruned = true;
        return false;
    }
    if (sbn3_spectrum_query(&out.producer, SBN3_SPECTRUM_COLUMNS, n, &out.future) != SBN3_SUPPORTED)
        return false;
    r.cached_a[0] = &out.future;
    if (sbn3_product_query(&r, &o, &out.plans[0], &out.infos[0]) != SBN3_SUPPORTED)
        return false;
    out.count = 1;
    if (kind == Cycle::Rsqrt) {
        r.kind = SBN3_PRODUCT_MUL;
        r.b_limbs = m + 1;
        r.window_limbs = 0;
    }
    if (kind == Cycle::Division) {
        r.cached_a[0] = nullptr;
        r.b_limbs = n;
    }
    if (kind != Cycle::Inverse) {
        if (sbn3_product_query(&r, &o, &out.plans[1], &out.infos[1]) != SBN3_SUPPORTED)
            return false;
        out.count = 2;
    }
    return true;
}
double cycle_cost(Cycle kind, const Bundle &b, bool deep) {
    // Existing measured MUL cost is an initial seed for the shared-transform
    // graph. Legality comes exclusively from the product queries above.
    if (b.producer_info.mul.algorithm == SBN3_MUL_SCALAR) {
        auto word_cost = [](const sbn3_product_info &i) {
            return scalar_product_cost(i.mul.nat, i.mul.nyt, 1) +
                   native_policy::scalar_ring_ns_per_word * i.cyclic_limbs +
                   native_policy::scalar_input_ns_per_word * i.mul.nyt;
        };
        const double first = word_cost(b.infos[0]);
        return kind == Cycle::Inverse ? 2 * first
               : kind == Cycle::Rsqrt ? first + word_cost(b.infos[1])
                                      : 2 * first + word_cost(b.infos[1]);
    }
    auto cost = [deep](const sbn3_mul_info &i) { return cost_model::cyclic_product(i, deep).nanoseconds; };
    const double producer = cost(b.producer_info.mul);
    if (kind == Cycle::Inverse)
        return cost_model::prepare_share(producer) + cost_model::cached_share(cost(b.infos[0].mul), 2);
    if (kind == Cycle::Rsqrt)
        return cost_model::prepare_share(producer) + cost_model::inverse_share(cost(b.infos[0].mul)) +
               cost_model::cached_share(cost(b.infos[1].mul));
    return cost_model::prepare_share(producer) + cost_model::cached_share(cost(b.infos[0].mul), 2) +
           cost(b.infos[1].mul);
}
double cold_cycle_cost(Cycle kind,const Bundle &b,bool deep,bool compact) {
    const double arithmetic=cycle_cost(kind,b,deep);
    if(!b.producer_info.mul.np)return arithmetic;
    const double root=root_prepare_cost(b.producer_info.mul);
    if(kind!=Cycle::Division)return arithmetic+root;
    const double extra_forward=compact?cost_model::prepare_share(cost_model::cyclic_product(b.producer_info.mul,deep).nanoseconds):0;
    return arithmetic+(compact?2:1)*root+root_prepare_cost(b.infos[1].mul)+extra_forward;
}
bool cycle_bound_probe(Cycle kind, size_t m, size_t n, const Choice &c, bool compact, double &bound, double &cost) {
    Bundle b{};
    if (!cycle_candidate(kind, m, n, c.ring, c, b))
        return false;
    const size_t largest = kind == Cycle::Inverse ? n : kind == Cycle::Rsqrt ? m + 1 : n + 1;
    bound = producer_lower_bound(kind, m, n, b);
    cost = cold_cycle_cost(kind, b, largest > native_policy::small_model_max_words, compact);
    return true;
}

bool choose_cycle(Plan &p, Cycle kind, size_t m, size_t n, unsigned index, bool replay, Bundle &out) {
    const size_t minimum = kind == Cycle::Inverse ? newton_contract::inverse_ring_min(n)
                           : kind == Cycle::Rsqrt ? newton_contract::rsqrt_ring_min(m)
                                                  : newton_contract::division_ring_min(m);
    auto period = [&](int T) {
        size_t r = 2 * size_t(T);
        while (r < minimum)
            r *= 2;
        return r;
    };
    if (replay) {
        require(index < p.choice_count, SBN3_FATAL_ARGUMENT, "Newton choice count");
        const auto c = p.choices[index];
        return cycle_candidate(kind, m, n, c.ring, c, out);
    }
    if (index >= newton_limits::stages)
        return false;
    const unsigned width_limit = newton_rung_workers(n, p.options.workers);
    const unsigned first =
                       p.options.prime_count ? p.options.prime_count : newton_limits::first_ntt_prime_count,
                   last = p.options.prime_count ? p.options.prime_count : newton_limits::last_ntt_prime_count;
    struct Candidate {
        Choice choice;
        double cost;
        size_t resident;
    };
    // Candidates are compared in one fixed order (scalar, NTT lattice, PQ16
    // radix ladder, PQ16 wide digits); ties go to the earlier one. The NTT
    // lattice is by far the largest family and, for short rungs, never close.
    // A candidate can be selected only if its cost is within
    // memory_trade_time_ratio of the final best, so one whose lower bound
    // already exceeds that window of the best seen so far cannot be chosen
    // and cannot change which other candidate is: skipping it is exact, not
    // a threshold. To make the window tight before the lattice is visited,
    // the cheap families are evaluated first and merged back in order.
    Candidate candidates[newton_limits::candidates]{}, late[newton_limits::candidates]{};
    unsigned count = 0, late_count = 0;
    double best = INFINITY;
    Bundle candidate{};
    auto consider = [&](const Choice &c, const Bundle &b, bool deferred = false) {
        const size_t largest = kind == Cycle::Inverse ? n : kind == Cycle::Rsqrt ? m + 1 : n + 1;
        const double cost = cold_cycle_cost(kind, b, largest > native_policy::small_model_max_words, p.compact);
        if (!std::isfinite(cost))
            return;
        size_t resident = b.future.storage_bytes + 16 * c.ring;
        for (unsigned j = 0; j < b.count; ++j)
            resident += b.infos[j].mul.workspace_bytes + b.infos[j].mul.table_bytes;
        require(count + late_count < newton_limits::candidates, SBN3_FATAL_ARGUMENT, "Newton candidate count");
        (deferred ? late[late_count++] : candidates[count++]) = {c, cost, resident};
        best = std::min(best, cost);
    };
    if (!p.options.prime_count && minimum <= 512) {
        const Choice c{0, SBN3_MUL_SCALAR, 1, 0, minimum};
        if (cycle_candidate(kind, m, n, minimum, c, candidate))
            consider(c, candidate);
    }
    auto lattice = [&] {
        for (unsigned np = first; np <= last; ++np)
            for (int T = np == 4 ? 88 : 24 * int(np) - 8; T >= (np == 4 ? 80 : 24 * int(np) - 32);
                 T -= np == 4 ? 4 : 8) {
                const size_t ring = period(T);
                for (unsigned algorithm : {unsigned(SBN3_MUL_FLAT), unsigned(SBN3_MUL_BAILEY)}) {
                    if (algorithm == SBN3_MUL_FLAT && ring > (size_t(1) << 19))
                        continue;
                    if (algorithm == SBN3_MUL_BAILEY && ring < 2048)
                        continue;
                    const Choice c{np, algorithm, width_limit, T, ring};
                    bool pruned = false;
                    if (!cycle_candidate(kind, m, n, ring, c, candidate, native_policy::memory_trade_time_ratio * best,
                                         &pruned))
                        continue;
                    consider(c, candidate);
                }
            }
    };
    const bool cheap_families = !p.options.prime_count && minimum <= 32768;
    if (!cheap_families)
        lattice();
    if (!p.options.prime_count && minimum <= 32768) {
        for (unsigned radix : {1u, 3u, 5u, 7u}) {
            size_t branch = radix == 3 ? 256 : 128;
            while (radix * branch < 2 * minimum)
                branch *= 2;
            const size_t ring = radix * branch / 2;
            if (ring > 32768)
                continue;
            for (unsigned width : {1u, width_limit}) {
                const Choice c{0, SBN3_MUL_PQ16, width, 16, ring};
                if (!cycle_candidate(kind, m, n, ring, c, candidate))
                    continue;
                consider(c, candidate, true);
            }
        }
    }
    if (!p.options.prime_count && minimum <= 16384) {
        for (unsigned bits : {17u, 18u, 19u, 20u})
            for (unsigned radix : {1u, 3u, 5u, 7u}) {
                size_t branch = 128;
                while (radix * branch * bits / 32 < minimum)
                    branch *= 2;
                const size_t ring = radix * branch * bits / 32;
                const Choice c{0, SBN3_MUL_PQ16, 1, int(bits), ring};
                if (cycle_candidate(kind, m, n, ring, c, candidate))
                    consider(c, candidate, true);
            }
    }
    if (cheap_families)
        lattice();
    for (unsigned j = 0; j < late_count; ++j)
        candidates[count++] = late[j];
    if (!count)
        return false;
    unsigned cheapest = 0;
    for (unsigned j = 1; j < count; ++j)
        if (candidates[j].cost < candidates[cheapest].cost ||
            (candidates[j].cost == candidates[cheapest].cost &&
             candidates[j].resident < candidates[cheapest].resident))
            cheapest = j;
    unsigned selected = cheapest;
    const double worthwhile =
        native_policy::memory_trade_resident_ratio * double(candidates[cheapest].resident);
    // Preserve the fastest prediction unless a near tie saves at least 1/8
    // of the resident rung. Small T-only space differences do not justify
    // changing plans; choose the cheapest among materially smaller options.
    for (unsigned j = 0; j < count; ++j)
        if (candidates[j].cost <= native_policy::memory_trade_time_ratio * best &&
            double(candidates[j].resident) <= worthwhile &&
            (selected == cheapest || candidates[j].cost < candidates[selected].cost))
            selected = j;
    const auto c = candidates[selected].choice;
    if (!cycle_candidate(kind, m, n, c.ring, c, out))
        return false;
    p.choices[index] = c;
    p.choice_count = index + 1;
    return true;
}
} // namespace sbn::v3::newton_detail
