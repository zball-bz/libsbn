/* Included inside one native instance's anonymous namespace. No heap ownership. */
void spectrum_retain(const sbn3_spectrum *opaque){
    auto &s=const_cast<Spectrum &>(spectrum(opaque));
    const uint64_t previous=__atomic_fetch_add(&s.refs,1,__ATOMIC_RELAXED);
    require(previous && previous!=UINT64_MAX,SBN3_FATAL_LIFETIME,"spectrum retain");
}
void spectrum_release(const sbn3_spectrum *opaque){
    auto &s=const_cast<Spectrum &>(spectrum(opaque));
    const uint64_t previous=__atomic_fetch_sub(&s.refs,1,__ATOMIC_ACQ_REL);
    require(previous,SBN3_FATAL_LIFETIME,"spectrum release");
    if(previous==1){auto *owner=s.owner;const auto storage=s.storage;s.header.marker=0;s.~Spectrum();owner->release(storage);}
}
void describe(const sbn3_spectrum *opaque,sbn3_spectrum_desc &out){out=spectrum(opaque).desc;}
bool can_apply(const sbn3_mul_plan &opaque,const sbn3_spectrum *cache,unsigned term){
    const auto q=load_plan(opaque);
    return term<2 && cache && cache->backend==&SBN3_P48_BACKEND() && cache->marker==spectrum_magic &&
        __atomic_load_n(&spectrum(cache).state,__ATOMIC_ACQUIRE)==spectrum_contract::ready && (q.recipe.cached_mask&(1u<<term)) &&
        cache_matches(spectrum(cache).desc,q.recipe.cached[term],q.arithmetic.transform,q.recipe.lengths[2*term]);
}
sbn3_spectrum_desc planned_spectrum(const ProductPlan &q,unsigned frontier,uint64_t generation){
    const auto &g=q.arithmetic.transform;const auto &i=q.execution.info;sbn3_spectrum_desc d{};
    d.basis_id=i.basis_id;d.backend_id=PN;d.generation=generation;d.np=PN;d.trunk_bits=g.T;d.frontier=frontier;
    d.format_version=i.algorithm==SBN3_MUL_FLAT?spectrum_contract::flat_lazy64_format:spectrum_contract::blocked48_format;d.C=g.C;d.M2=g.M2;d.transform_trunks=g.N;
    d.live_slots=g.lbv;d.written_slots=g.lbw;d.source_limbs=q.recipe.lengths[0];d.source_trunks=(d.source_limbs*64+g.T-1)/g.T;
    d.block_stride=i.algorithm==SBN3_MUL_FLAT?64:g.bstride;
    for(unsigned k=0;k<PN;++k)d.scale[k]=q.arithmetic.scale_a[k];
    d.storage_bytes=q.recipe.spectrum_bytes;d.table_bytes=owned_table_bytes(i.table_entries,i.factor_levels,g.M2,i.algorithm==SBN3_MUL_FLAT);d.plane_bytes=i.algorithm==SBN3_MUL_FLAT?g.lbw*64+128:g.plane_bytes;
    d.seal=descriptor_seal(d);return d;
}
sbn3_query_result query_spectrum(const sbn3_mul_plan &opaque,unsigned frontier,uint64_t generation,sbn3_spectrum_desc &out){
    if constexpr(LEAF!=8)return SBN3_UNSUPPORTED;
    if(frontier>1)return SBN3_UNSUPPORTED;out=planned_spectrum(load_plan(opaque),frontier,generation);return SBN3_SUPPORTED;
}
struct PrepareCall {Binding *binding;Spectrum *spectrum;sbn3_const_limbs a;unsigned frontier;p::PassCounts *counts=nullptr;};
void prepare_action(void *argument,sbn3_team_scope *scope){
    auto &x=*static_cast<PrepareCall *>(argument);auto &b=*x.binding;auto &s=*x.spectrum;
    const auto &g=b.plan.arithmetic.transform;const auto &i=b.plan.execution.info;
    if(i.algorithm==SBN3_MUL_FLAT){flat_prepare_bound(b,s,scope,x.a,x.counts);return;}
    sbn3_team_scope sub{scope->team,scope->first,i.workers,false,scope->epoch};scope=&sub;
    p::Ctx c=b.context;c.PS=&s.primes;c.prm=0;c.counts=x.counts;c.reversed=0;
    for(unsigned k=0;k<PN;++k){c.fp[k]=s.planes[k];c.op[k]=nullptr;}
    auto t=transpose_shape(g,x.a.count);c.dsc=b.plan.arithmetic.scale_a;
    // A spectrum build may borrow the idle product pool, regardless of the
    // subsequent product's output alias contract.
    if(!i.transpose_bytes) forward_rows(b,scope,c,t,x.a.data,nullptr);
    else {
        t.a=x.a.data;t.xp=b.run.pool;
        kernel_for(scope,0,t.C/t.tr,1,SBN3_STATIC,p::xpose_fn,&t,b.run.workers);
        c.a=x.a.data;c.an=x.a.count;c.natv=t.natv;c.xp=t.xp;c.xrs=t.xrs;c.xs=t.xs;
        kernel_for(scope,0,g.C/CR_ROWW,UR,SBN3_STATIC,p::rows_fn<8,0>,&c,b.run.workers);c.xp=nullptr;
    }
    if(x.frontier)kernel_for(scope,0,PN*g.nblk,8,SBN3_DYNAMIC,p::block_fn<8,0,1>,&c,b.run.workers);
}
void reserve_plan_spectrum(const sbn3_mul_plan &opaque,unsigned frontier,uint64_t generation,
                           sbn3_arena &arena,const sbn3_lease &storage,sbn3_spectrum **out){
    require(LEAF==8,SBN3_FATAL_ARGUMENT,"unsupported spectrum leaf");
    const auto plan=load_plan(opaque);const auto &r=plan.recipe;const auto &i=plan.execution.info;const auto &g=plan.arithmetic.transform;
    require(frontier<=1 && storage.bytes>=r.spectrum_bytes &&
        !(reinterpret_cast<uintptr_t>(storage.data)&63),SBN3_FATAL_ARGUMENT,"spectrum prepare spans");
    // claim_unshared also rejects any overlapping live lease (including worker stacks).
    arena.claim_unshared(storage);Spectrum *sp=nullptr;
    {
        ComputeLease setup(arena);Frame mem(arena,storage);
        sp=::new(mem.allocate(sizeof(Spectrum)))Spectrum{};sp->owner=&arena;sp->storage=storage;
        sp->primes.init(mem,p::clog2(i.table_entries));
        if(i.algorithm!=SBN3_MUL_FLAT)sp->primes.factors_for(mem,g.M2,g.lgC);
        for(unsigned k=0;k<PN;++k)sp->planes[k]=static_cast<uint8_t *>(mem.allocate(i.algorithm==SBN3_MUL_FLAT?g.lbw*64+128:g.plane_bytes+128));
        static uint64_t sequence=0;auto &d=sp->desc;d=planned_spectrum(plan,frontier,generation);
        d.instance_id=__atomic_add_fetch(&sequence,1,__ATOMIC_RELAXED);
        require(d.instance_id,SBN3_FATAL_LIFETIME,"spectrum sequence exhausted");
        d.storage_bytes=storage.bytes;d.seal=descriptor_seal(d);
    }
    *out=&sp->header;
}
void reserve_spectrum(sbn3_mul_binding *opaque,unsigned frontier,uint64_t generation,
                      sbn3_arena &arena,const sbn3_lease &storage,sbn3_spectrum **out){
    auto &b=binding(opaque);idle_owner(b);require(&arena==b.run.arena,SBN3_FATAL_ARGUMENT,"spectrum reserve arena");
    sbn3_mul_plan plan{};memcpy(plan.opaque,&b.plan,sizeof b.plan);reserve_plan_spectrum(plan,frontier,generation,arena,storage,out);
}
void begin_spectrum(Binding &b,Spectrum &sp,sbn3_const_limbs a){
    idle_owner(b);const auto &g=b.plan.arithmetic.transform;
    require(sp.owner==b.run.arena && compatible(sp.desc,g,a.count) && sp.desc.live_slots==g.lbv && sp.desc.written_slots==g.lbw &&
        !memcmp(sp.desc.scale,b.plan.arithmetic.scale_a,sizeof b.plan.arithmetic.scale_a),SBN3_FATAL_ARGUMENT,"spectrum producer representation");
    valid_span(a.data,bytes_for(a.count,8),"spectrum input");require(!(reinterpret_cast<uintptr_t>(a.data)&7),SBN3_FATAL_ARGUMENT,"spectrum input alignment");
    for(const auto &l:{b.run.work_memory,b.run.table_memory,b.run.team->storage,sp.storage})
        require(!overlaps(l.data,l.bytes,a.data,a.count*8),SBN3_FATAL_ARGUMENT,"spectrum input alias");
    for(unsigned k=1;k<b.run.team->width;++k)require(!overlaps(b.run.team->stacks[k].data,b.run.team->stacks[k].bytes,a.data,a.count*8),SBN3_FATAL_ARGUMENT,"spectrum input/stack alias");
    uint32_t expected=spectrum_contract::reserved;require(__atomic_compare_exchange_n(&sp.state,&expected,spectrum_contract::computing,false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE),SBN3_FATAL_LIFETIME,"spectrum already computed");
 }
void compute_spectrum(sbn3_mul_binding *opaque,sbn3_spectrum *handle,sbn3_const_limbs a){
    auto &b=binding(opaque);auto &sp=const_cast<Spectrum &>(spectrum(handle));begin_spectrum(b,sp,a);
    PrepareCall call{&b,&sp,a,sp.desc.frontier};sbn3_team_run(b.run.team,prepare_action,&call);
    __atomic_store_n(&sp.state,spectrum_contract::ready,__ATOMIC_RELEASE);
}
void prepare(sbn3_mul_binding *opaque,sbn3_const_limbs a,unsigned frontier,uint64_t generation,
             sbn3_arena &arena,const sbn3_lease &storage,sbn3_spectrum **out){
    reserve_spectrum(opaque,frontier,generation,arena,storage,out);compute_spectrum(opaque,*out,a);
}
struct ProductCall {Binding *b;sbn3_product_inputs inputs;sbn3_limbs output;};
void product_rows(Binding &b,sbn3_team_scope *s,sbn3_const_limbs value,uint8_t *const *dest,
                  const uint64_t *scale,uint64_t *output,bool mid=false){
    auto &c=b.context;const auto &g=b.plan.arithmetic.transform;
    for(unsigned k=0;k<PN;++k)c.fp[k]=dest[k];c.dsc=scale;c.reversed=mid;c.Zv=g.Zv;
    if(mid){c.xp=nullptr;c.a=value.data;c.an=value.count;c.natv=(value.count*64+g.T-1)/g.T/8;
        kernel_for(s,0,g.C/CR_ROWW,UR,SBN3_STATIC,p::rows_fn<8,1>,&c,b.run.workers);
    }else{forward_rows(b,s,c,transpose_shape(g,value.count),value.data,output);}
    c.xp=nullptr;
}
void product_action(void *argument,sbn3_team_scope *scope){
    auto &x=*static_cast<ProductCall *>(argument);auto &b=*x.b;auto &c=b.context;
    const auto &e=b.plan.execution;const auto &i=e.info;const auto &g=b.plan.arithmetic.transform;const auto &r=b.plan.recipe;
    if(i.algorithm==SBN3_MUL_FLAT){execute_flat(b,scope,x.inputs,x.output,true);return;}
    require_scope_leader(scope);require(scope->team==b.run.team && scope->width>=i.workers,SBN3_FATAL_TEAM,"product scope width");
    sbn3_team_scope sub{scope->team,scope->first,i.workers,false,scope->epoch};scope=&sub;
    uint32_t expected=0;require(__atomic_compare_exchange_n(&b.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent binding execution");
    for(auto &n:b.counts)n={};c.counts=b.counts;
    const uint64_t t0=tick_ns();memset(b.run.tail,0,e.tail_bytes);if(e.journal_bytes)memset(b.run.journal,0,e.journal_bytes);
    for(auto &gate:b.gates){gate.next.store(0,std::memory_order_relaxed);gate.done.store(0,std::memory_order_relaxed);}
    if(!(r.cached_mask&1))product_rows(b,scope,x.inputs.a,b.run.a_planes,b.plan.arithmetic.scale_a,x.output.data);
    if(r.kind==SBN3_PRODUCT_MAC2 && !(r.cached_mask&2))product_rows(b,scope,x.inputs.a1,b.run.a1_planes,b.plan.arithmetic.scale_a,x.output.data);
    const uint64_t t1=tick_ns();
    if(r.kind!=SBN3_PRODUCT_SQR)product_rows(b,scope,x.inputs.b,b.run.b_planes,nullptr,x.output.data,r.kind==SBN3_PRODUCT_TMP);
    if(r.kind==SBN3_PRODUCT_MAC2)product_rows(b,scope,x.inputs.b1,b.run.b1_planes,nullptr,x.output.data);
    for(unsigned k=0;k<PN;++k){
        c.hp[k]=(r.cached_mask&1)?spectrum(b.run.cached[0]).planes[k]:b.run.a_planes[k];
        c.fp[k]=b.run.b_planes[k];c.op[k]=nullptr;
        if(r.kind==SBN3_PRODUCT_MAC2){c.hp1[k]=(r.cached_mask&2)?spectrum(b.run.cached[1]).planes[k]:b.run.a1_planes[k];c.fp1[k]=b.run.b1_planes[k];}
    }
    const uint64_t t2=tick_ns();
    KernelForFn tile=nullptr;
    if(r.kind==SBN3_PRODUCT_SQR)tile=r.scaled?p::block_fn4<8,0,0,1,true>:p::block_fn4<8,0,0,1,false>;
    else if(r.kind==SBN3_PRODUCT_MAC2)tile=r.scaled?p::block_fn4<8,0,0,2,true>:p::block_fn4<8,0,0,2,false>;
    else if(r.kind==SBN3_PRODUCT_TMP)tile=r.scaled?p::block_fn4<8,1,0,0,true>:p::block_fn4<8,1,0,0,false>;
    else if(c.writeback_a)tile=r.scaled?p::block_fn4<8,0,0,0,true,true>:p::block_fn4<8,0,0,0,false,true>;
    else tile=r.scaled?p::block_fn4<8,0,0,0,true>:p::block_fn4<8,0,0,0,false>;
    if(i.fused){
        uint8_t *spare[PN]{};for(unsigned k=0;k<i.prime_batch;++k)spare[k]=b.run.spares[k];
        const size_t nbi=(g.nblk+7)&~size_t(7);
        for(unsigned first=0;first<PN;first+=i.prime_batch){
            const unsigned count=first+i.prime_batch<=PN?i.prime_batch:PN-first;
            for(unsigned k=0;k<count;++k)c.op[first+k]=(r.kind==SBN3_PRODUCT_SQR && r.cached_mask)?b.run.b_planes[first+k]:spare[k];
            kernel_for(scope,first*nbi,(first+count)*nbi,8,SBN3_DYNAMIC,tile,&c,b.run.workers);
            for(unsigned k=0;k<count;++k)spare[k]=b.run.b_planes[first+k];
        }
    }else kernel_for(scope,0,PN*g.nblk,8,SBN3_DYNAMIC,tile,&c,b.run.workers);
    const uint64_t t3=tick_ns();
    if(i.fused){
        p::FuseCtx f{};f.c=&c;f.EK=&b.emit_tables.EK;f.rp=x.output.data;f.tail=b.run.tail;f.nts=g.outcap*8>=(size_t(1)<<20);f.W=i.fused_items;f.prefix=r.kind!=SBN3_PRODUCT_TMP?r.window.width_bits/64:0;f.jrn=b.run.journal;
        kernel_for(scope,0,i.fused_items,1,SBN3_STATIC,p::irowemit_fn<8>,&f,b.run.workers);p::fuse_join(f,g);
    }else{
        const KernelForFn inverse=g.arm==p::ARM_CYC?p::irow_fn<p::ARM_CYC>:r.kind==SBN3_PRODUCT_TMP?p::irow_fn<p::ARM_TMP>:p::irow_fn<p::ARM_LIN>;
        kernel_for(scope,0,PN*(g.nrows/IROW_G),1,SBN3_STATIC,inverse,&c,b.run.workers);
        p::Plan eg=g;const size_t prefix=r.kind!=SBN3_PRODUCT_TMP?r.window.width_bits/64:0;
        if(prefix){eg.outcap=prefix;eg.ntp=((prefix+g.OL-1)/g.OL)*g.OT;eg.nl=eg.ntp*g.T/64;}
        p::EmitCtx ec{};ec.pl=&eg;ec.PS=b.run.primes;ec.G=&b.garner;ec.rp=x.output.data;ec.tail=b.run.tail;
        ec.spill=b.run.spill;ec.nch=(eg.ntp+p::OCH-1)/p::OCH;ec.EK=&b.emit_tables.EK;ec.RG=&b.emit_tables.RG;
        ec.reversed=r.kind==SBN3_PRODUCT_TMP;ec.vlim=g.Zw;
        for(unsigned k=0;k<PN;++k)ec.plane[k]=c.fp[k];
        p::EmitMap map;ec.map=&map;const int tasks=p::emit_tasks(map,eg,ec.nch,ec.reversed);
        kernel_for(scope,0,tasks,1,SBN3_DYNAMIC,emit_range,&ec,b.run.workers);p::emit_join(ec);
    }
    if(g.ring_rn&&!r.window.width_bits)p::fold_cyc(x.output.data,g.ring_rn,b.run.tail,g.nl-g.outcap);
    const uint64_t t4=tick_ns();b.last_stage_ns[0]=t1-t0;b.last_stage_ns[1]=t2-t1;b.last_stage_ns[2]=t3-t2;b.last_stage_ns[3]=t4-t3;
    ++b.executions;__atomic_store_n(&b.active,0,__ATOMIC_RELEASE);
}
void product_execute(sbn3_mul_binding *opaque,sbn3_team_scope *scope,const sbn3_product_inputs &in,sbn3_limbs out){
    auto &b=binding(opaque);if(!scope)idle_owner(b);else require_scope_leader(scope);
    const auto &r=b.plan.recipe;const size_t capacity=sbn3_mul_output_capacity(&b.plan.execution.info),rb=bytes_for(capacity,8);
    require(out.capacity>=capacity && !(reinterpret_cast<uintptr_t>(out.data)&63),SBN3_FATAL_ARGUMENT,"product output span");valid_span(out.data,rb,"product output");
    const sbn3_const_limbs values[4]{in.a,in.b,in.a1,in.b1};
    for(unsigned k=0;k<4;++k){const bool absent=(k==0 && (r.cached_mask&1)) || (k==2 && (r.cached_mask&2)) ||
            (k==1 && r.kind==SBN3_PRODUCT_SQR) || (k>=2 && r.kind!=SBN3_PRODUCT_MAC2);
        require(values[k].count==(absent?0:r.lengths[k]) && (!absent || !values[k].data),SBN3_FATAL_ARGUMENT,"product input lengths");
        if(!absent){const size_t nb=bytes_for(values[k].count,8);valid_span(values[k].data,nb,"product input");
            require(!(reinterpret_cast<uintptr_t>(values[k].data)&7) && !overlaps(values[k].data,nb,out.data,rb),SBN3_FATAL_ARGUMENT,"product input alias");}}
    auto no_alias=[&](const sbn3_lease &l){require(!overlaps(l.data,l.bytes,out.data,rb),SBN3_FATAL_ARGUMENT,"product output/resource alias");
        for(const auto &v:values)if(v.count)require(!overlaps(l.data,l.bytes,v.data,v.count*8),SBN3_FATAL_ARGUMENT,"product input/resource alias");};
    for(const auto &l:{b.run.table_memory,b.run.work_memory,b.run.team->storage})no_alias(l);
    for(unsigned k=1;k<b.run.team->width;++k)no_alias(b.run.team->stacks[k]);
    for(auto *s:b.run.cached)if(s){require(__atomic_load_n(&spectrum(s).state,__ATOMIC_ACQUIRE)==spectrum_contract::ready,SBN3_FATAL_LIFETIME,"spectrum not completed");no_alias(spectrum(s).storage);}
    if constexpr(LEAF!=8){Execute call{&b,in.a,in.b,out};if(scope)execute_action(&call,scope);else sbn3_team_run(b.run.team,execute_action,&call);return;}
    ProductCall call{&b,in,out};if(scope)product_action(&call,scope);else sbn3_team_run(b.run.team,product_action,&call);
}
void compute_square(sbn3_mul_binding *opaque,sbn3_spectrum *handle,sbn3_const_limbs a,sbn3_limbs out){
    auto &b=binding(opaque);idle_owner(b);const auto &r=b.plan.recipe;
    require(r.kind==SBN3_PRODUCT_SQR&&r.cached_mask==1&&b.run.cached[0]==handle,SBN3_FATAL_ARGUMENT,"fused SQR cache binding");
    const size_t capacity=sbn3_mul_output_capacity(&b.plan.execution.info),bytes=bytes_for(capacity,8);
    require(out.capacity>=capacity&&!(uintptr_t(out.data)&63),SBN3_FATAL_ARGUMENT,"fused SQR output");valid_span(out.data,bytes,"fused SQR output");
    require(!overlaps(a.data,a.count*8,out.data,bytes),SBN3_FATAL_ARGUMENT,"fused SQR input/output alias");
    auto &sp=const_cast<Spectrum &>(spectrum(handle));
    for(auto l:{b.run.work_memory,b.run.table_memory,b.run.team->storage,sp.storage})require(!overlaps(l.data,l.bytes,out.data,bytes),SBN3_FATAL_ARGUMENT,"fused SQR output/resource alias");
    for(unsigned j=1;j<b.run.team->width;++j)require(!overlaps(b.run.team->stacks[j].data,b.run.team->stacks[j].bytes,out.data,bytes),SBN3_FATAL_ARGUMENT,"fused SQR output/stack alias");
    begin_spectrum(b,sp,a);
    struct Both {PrepareCall prep;ProductCall product;p::PassCounts counts[32]{};} call{{&b,&sp,a,sp.desc.frontier},{&b,{},out}};
    call.prep.counts=call.counts;
    auto action=[](void *ptr,sbn3_team_scope *scope){auto &v=*static_cast<Both *>(ptr);const auto begin=tick_ns();prepare_action(&v.prep,scope);const auto prepared=tick_ns();product_action(&v.product,scope);v.product.b->last_stage_ns[0]+=prepared-begin;
        for(unsigned j=0;j<32;++j){auto &dst=v.product.b->counts[j];const auto &src=v.counts[j];dst.row_forward+=src.row_forward;dst.column_forward+=src.column_forward;dst.row_mid+=src.row_mid;dst.row_inverse+=src.row_inverse;dst.column_inverse+=src.column_inverse;dst.row_transpose_forward+=src.row_transpose_forward;dst.leaf_products+=src.leaf_products;}};
    sbn3_team_run(b.run.team,action,&call);__atomic_store_n(&sp.state,spectrum_contract::ready,__ATOMIC_RELEASE);
}
void compute_multiply(sbn3_mul_binding *opaque,sbn3_spectrum *handle,sbn3_const_limbs a,
                      sbn3_const_limbs y,sbn3_limbs out){
    auto &b=binding(opaque);idle_owner(b);const auto &r=b.plan.recipe;
    require(r.kind==SBN3_PRODUCT_MUL && r.cached_mask==1 && b.run.cached[0]==handle &&
            a.count==r.lengths[0] && y.count==r.lengths[1],SBN3_FATAL_ARGUMENT,"build/apply MUL binding");
    const size_t capacity=sbn3_mul_output_capacity(&b.plan.execution.info),bytes=bytes_for(capacity,8);
    require(out.capacity>=capacity && !(uintptr_t(out.data)&63),SBN3_FATAL_ARGUMENT,"build/apply output");
    valid_span(out.data,bytes,"build/apply output");valid_span(y.data,bytes_for(y.count,8),"build/apply B");
    require(!(uintptr_t(y.data)&7) && !overlaps(a.data,a.count*8,out.data,bytes) &&
            !overlaps(y.data,y.count*8,out.data,bytes),SBN3_FATAL_ARGUMENT,"build/apply input/output alias");
    auto &sp=const_cast<Spectrum &>(spectrum(handle));
    auto disjoint=[&](const sbn3_lease &l){
        require(!overlaps(l.data,l.bytes,out.data,bytes) && !overlaps(l.data,l.bytes,y.data,y.count*8),
                SBN3_FATAL_ARGUMENT,"build/apply resources");
    };
    for(const auto &l:{b.run.work_memory,b.run.table_memory,b.run.team->storage,sp.storage})disjoint(l);
    for(unsigned j=1;j<b.run.team->width;++j)disjoint(b.run.team->stacks[j]);
    begin_spectrum(b,sp,a);
    const bool writeback=b.plan.execution.info.algorithm==SBN3_MUL_BAILEY && sp.desc.frontier==1;
    struct Both {PrepareCall prep;ProductCall product;p::PassCounts counts[32]{};bool writeback;} call{
        {&b,&sp,a,writeback?0:sp.desc.frontier},{&b,{{},y,{},{}},out},{},writeback};
    call.prep.counts=call.counts;
    auto action=[](void *ptr,sbn3_team_scope *scope){
        auto &v=*static_cast<Both *>(ptr);auto &b=*v.product.b;const auto start=tick_ns();
        prepare_action(&v.prep,scope);const auto prepared=tick_ns();
        if(v.writeback){b.context.frontier[0]=0;b.context.writeback_a=1;}
        product_action(&v.product,scope);
        b.context.frontier[0]=v.prep.spectrum->desc.frontier;b.context.writeback_a=0;
        b.last_stage_ns[0]+=prepared-start;
        for(unsigned j=0;j<32;++j){auto &dst=b.counts[j];const auto &src=v.counts[j];
            dst.row_forward+=src.row_forward;dst.column_forward+=src.column_forward;
            dst.row_mid+=src.row_mid;dst.row_inverse+=src.row_inverse;
            dst.column_inverse+=src.column_inverse;dst.row_transpose_forward+=src.row_transpose_forward;
            dst.leaf_products+=src.leaf_products;}
    };
    sbn3_team_run(b.run.team,action,&call);
    __atomic_store_n(&sp.state,spectrum_contract::ready,__ATOMIC_RELEASE);
}
void product_metrics(const sbn3_mul_binding *opaque,sbn3_product_metrics &out){
    const auto &b=binding(opaque);out={};metrics(opaque,out.mul);
    for(const auto &n:b.counts){out.row_forward+=n.row_forward;out.row_mid+=n.row_mid;out.column_forward+=n.column_forward;
        out.column_inverse+=n.column_inverse;out.row_inverse+=n.row_inverse;out.row_transpose_forward+=n.row_transpose_forward;out.leaf_products+=n.leaf_products;}
}
