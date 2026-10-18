// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/prepared_divisor.hpp"
#include "sbn3/radix.hpp"
#include "sbn3/context.hpp"
#include "runtime/resident.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;
int main(){
    sbn3_arena*arena=nullptr;sbn3_arena_config cfg{size_t(1)<<28,8u<<20};sbn3_error error{};
    assert(sbn3_arena_create(&cfg,&arena,&error)==SBN3_OK);assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
    sbn3_lease control{};sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
    sbn3_team_config tc{};tc.workers=2;tc.stack_offset=65536;for(auto&c:tc.cpu_ids)c=-1;
    sbn3_team*team=nullptr;assert(sbn3_team_create(arena,&control,&tc,&team,&error)==SBN3_OK);
    std::vector<uint64_t> words(32769);uint64_t seed=748926103;for(auto&w:words){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;w=seed;}words.back()|=1ULL<<63;
    const integer D{sbn::integer_view{words}},Q=(integer(1)<<500003)+579,N=D*Q+71;
    {
        integer q,r;q.reserve(N.size()+1);r.reserve(D.size());const auto need=sbn::divrem_requirements(q.ref(),r.ref(),N.view(),D.view(),sbn::division_rounding::trunc,2);assert(need.supported());
        sbn::workspace w;w.reserve(need.scratch_bytes,need.scratch_alignment);
        allocation_watch_start();sbn::divrem_into(q.ref(),r.ref(),N.view(),D.view(),w.on(team));assert(!allocation_watch_stop());assert(q==Q&&r==71);
        sbn::prepared_divisor pd(D.view(),N.size(),sbn::default_resource(),team,4);
        allocation_watch_start();pd.execute(q.ref(),r.ref(),N.view());assert(!allocation_watch_stop());assert(q==Q&&r==71);
        const auto fn=sbn::format_requirements(N.view(),10,2);assert(fn.supported());std::vector<char> chars(fn.characters);w.reserve(fn.scratch_bytes,fn.scratch_alignment);
        allocation_watch_start();auto text=sbn::format_into(chars,N.view(),10,w.on(team));assert(!allocation_watch_stop());assert(text);
        integer parsed;parsed.reserve(N.size()+1);const auto pn=sbn::parse_requirements(parsed.ref(),text.text,10,2);assert(pn.supported());w.reserve(pn.scratch_bytes,pn.scratch_alignment);
        allocation_watch_start();assert(!sbn::parse_integer_into(parsed.ref(),text.text,10,w.on(team)));assert(!allocation_watch_stop());assert(parsed==N);
        sbn::context ctx(sbn::default_resource(),team);auto expr=sbn::expr::square(sbn::expr::add(D,Q))-sbn::expr::square(D)-sbn::expr::square(Q);
        const auto req=ctx.query(expr);assert(req.supported());ctx.reserve(req);integer expression_out;expression_out.reserve(req.output_limbs);
        allocation_watch_start();ctx.evaluate_into(expression_out.ref(),expr);assert(!allocation_watch_stop());assert(expression_out==2*D*Q);
        auto division=sbn::expr::divide(sbn::expr::value(N),sbn::expr::sub(D,D)+97);const auto dr=ctx.query(division);assert(dr.supported());ctx.reserve(dr);expression_out.reserve(dr.output_limbs);
        allocation_watch_start();ctx.evaluate_into(expression_out.ref(),division);assert(!allocation_watch_stop());assert(expression_out==N/97);
        // Borrowed arena descriptors may never perform mapping/page work.
        sbn::v3::ResidentSession resident(w.on().scratch,team);
        allocation_watch_start();assert(sbn3_arena_prepare(resident.arena(),0,64,&error)==SBN3_EBUSY);assert(sbn3_arena_trim(resident.arena(),0,64,&error)==SBN3_EBUSY);assert(!allocation_watch_stop());
    }
    sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_stats stats{};sbn3_arena_get_stats(arena,&stats);assert(!stats.lease_references&&!stats.active_computations);sbn3_arena_destroy(arena);
    puts("C++ borrowed resident services with an existing W2 Team, no allocation or page work PASS");
}
