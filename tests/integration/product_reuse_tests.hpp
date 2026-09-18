struct CachedTask {sbn3_mul_binding *b;sbn3_product_inputs inputs;uint64_t *out;size_t rn;};
static void cached_task(void *arg,sbn3_team_scope *scope){auto &t=*static_cast<CachedTask *>(arg);sbn3_product_execute_on_scope(t.b,scope,&t.inputs,{t.out,t.rn});}
static void concurrent_tasks(void *arg,sbn3_team_scope *scope){auto *t=static_cast<CachedTask *>(arg);sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,cached_task,t,cached_task,t+1);}
static void reuse_gate(unsigned frontier,unsigned algorithm=0){
    Fixture f(4);const size_t an=513,bn=algorithm==SBN3_MUL_FLAT?130559:15871;auto *a=f.guarded(an),*b=f.guarded(bn);
    for(size_t k=0;k<an;++k)a[k]=random_word();for(size_t k=0;k<bn;++k)b[k]=random_word();
    sbn3_mul_options opt{};opt.workers=2;opt.prime_count=6;opt.trunk_bits=128;opt.crt_mode=1;opt.borrow_output=1;opt.column_log2=6;opt.row_log2=4;opt.algorithm=algorithm;if(algorithm==SBN3_MUL_FLAT)opt.column_log2=opt.row_log2=0;
    sbn3_product_request build{};build.a_limbs=an;build.b_limbs=bn;
    sbn3_mul_plan plan{};sbn3_product_info info{};auto *seed=f.product(build,opt,info,plan);
    // Use full rows for a scale deliberately different from the small applies.
    assert(info.mul.full);sbn3_spectrum_desc desc{};auto *handle=f.cache(seed,{a,an},frontier,info,desc);f.unbind(seed);
    std::vector<uint64_t> original(a,a+an);memset(a,0xcd,an*8); // original value storage is now reusable
    sbn3_product_request req[2]{};sbn3_product_info infos[2]{};CachedTask tasks[2]{};
    for(unsigned j=0;j<2;++j){req[j].kind=j?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;req[j].a_limbs=an;req[j].b_limbs=j?0:601;req[j].cached_a[0]=&desc;
        opt.crt_mode=2;tasks[j].b=f.product(req[j],opt,infos[j],plan,handle);
        bool compensation=false;for(unsigned q=0;q<6;++q)compensation|=infos[j].leaf_scale[0][q]!=1;assert(compensation);
        tasks[j].inputs={};if(!j)tasks[j].inputs.b={b,601};tasks[j].rn=infos[j].mul.output_limbs;tasks[j].out=f.guarded(up(tasks[j].rn,8));
    }
    for(unsigned k=0;k<8;++k){auto bad=desc;switch(k){case 0:++bad.generation;break;case 1:++bad.instance_id;break;case 2:++bad.basis_id;break;case 3:++bad.scale[0];break;
        case 4:bad.live_slots=0;break;case 5:bad.frontier=2;break;case 6:bad.format_version=99;break;case 7:bad.np=5;break;}
        auto q=req[0];q.cached_a[0]=&bad;sbn3_product_info badinfo{};assert(sbn3_product_query(&q,&opt,&plan,&badinfo)==SBN3_UNSUPPORTED);}
    auto too_big=req[0];too_big.b_limbs=8*bn;sbn3_product_info ignored{};assert(sbn3_product_query(&too_big,&opt,&plan,&ignored)!=SBN3_SUPPORTED);
    const auto hash=hash_cache(handle,desc);
    // Drop the user reference; two independent bindings keep the immutable handle alive.
    sbn3_spectrum_release(handle);
    for(unsigned repeats:{1u,2u,4u,8u})for(unsigned k=0;k<repeats;++k){
        for(size_t j=0;j<601;++j)b[j]=random_word();allocation_watch_start();sbn3_team_run(f.team,concurrent_tasks,tasks);assert(!allocation_watch_stop());
        sbn3_product_inputs full{{original.data(),an},{b,601},{},{}};for(unsigned j=0;j<2;++j)check(req[j],full,infos[j],tasks[j].out);
    }
    assert(hash_cache(handle,desc)==hash);puts("same cache 1/2/4/8; concurrent MUL/SQR; Garner/full -> direct/truncated scale; owner unbound; rejection/refcount OK");
}
static void unequal_mac(){
    Fixture f(1);sbn3_mul_options o{};o.workers=1;o.prime_count=8;o.borrow_output=1;o.column_log2=6;o.row_log2=4;
    sbn3_product_request req{SBN3_PRODUCT_MAC2,111,901,609,37,{},0,0,0};sbn3_product_info info{};sbn3_mul_plan plan{};auto *p=f.product(req,o,info,plan);
    sbn3_product_inputs in{};sbn3_const_limbs *v[]{&in.a,&in.b,&in.a1,&in.b1};size_t lengths[]{111,901,609,37};
    for(unsigned j=0;j<4;++j){auto *a=f.guarded(lengths[j]);*v[j]={a,lengths[j]};for(size_t k=0;k<lengths[j];++k)a[k]=UINT64_MAX;}
    auto *out=f.guarded(up(info.mul.output_limbs,8));assert(info.mul.output_limbs==1013);
    sbn3_product_execute(p,&in,{out,info.mul.output_limbs});check(req,in,info,out);puts("unequal positive MAC/carry capacity OK");
}
static void capacity_gate(){
    sbn3_mul_options o{};o.workers=1;o.prime_count=4;o.trunk_bits=88;sbn3_mul_plan plan{};sbn3_product_info info{};
    sbn3_product_request r{SBN3_PRODUCT_MUL,80000,80000,0,0,{},0,0,0};
    assert(sbn3_product_query(&r,&o,&plan,&info)==SBN3_SUPPORTED);
    r.kind=SBN3_PRODUCT_MAC2;r.a1_limbs=r.b1_limbs=80000;memset(&plan,0x9c,sizeof plan);
    assert(sbn3_product_query(&r,&o,&plan,&info)==SBN3_QUERY_CAPACITY && plan.opaque[0]==0x9c9c9c9c9c9c9c9cULL);
    o.trunk_bits=0;assert(sbn3_product_query(&r,&o,&plan,&info)==SBN3_SUPPORTED && info.mul.trunk_bits==84);
    o.workspace_budget=1;assert(sbn3_product_query(&r,&o,&plan,&info)==SBN3_QUERY_CAPACITY);
    puts("MAC sum bound rejects legal single-product T88; auto T84; budget query OK");
}

static void distinct_mac_reuse(unsigned algorithm=0){
    Fixture f(3);sbn3_mul_options o{};o.workers=3;o.prime_count=6;o.trunk_bits=128;o.column_log2=6;o.row_log2=4;o.borrow_output=1;o.algorithm=algorithm;if(algorithm==SBN3_MUL_FLAT)o.column_log2=o.row_log2=0;
    constexpr size_t n=8192;uint64_t *a=f.guarded(n),*b=f.guarded(n),*a1=f.guarded(n),*b1=f.guarded(n);
    for(size_t k=0;k<n;++k){a[k]=random_word();b[k]=random_word();a1[k]=random_word();b1[k]=random_word();}
    sbn3_product_request req{SBN3_PRODUCT_MAC2,n,n,n,n,{},0,0,0};sbn3_mul_plan plan{};sbn3_product_info info{};
    auto *seed=f.product(req,o,info,plan);sbn3_spectrum_desc d[2]{};auto *s0=f.cache(seed,{a,n},1,info,d[0]),*s1=f.cache(seed,{a1,n},1,info,d[1]);f.unbind(seed);
    req.cached_a[0]=d;req.cached_a[1]=d+1;auto *bound=f.product(req,o,info,plan,s0,s1);
    auto *out=f.guarded(up(info.mul.output_limbs,8));sbn3_product_inputs full{{a,n},{b,n},{a1,n},{b1,n}},cached{{},{b,n},{},{b1,n}};
    for(unsigned round=0;round<8;++round){memset(out,0xa7,up(info.mul.output_limbs,8)*8);sbn3_product_execute(bound,&cached,{out,info.mul.output_limbs});check(req,full,info,out);
        for(size_t k=0;k<n;++k){b[k]=random_word();b1[k]=random_word();}}
    sbn3_spectrum_release(s0);sbn3_spectrum_release(s1);puts("distinct full cached MAC / nonzero output / repeated B replacement OK");
}
