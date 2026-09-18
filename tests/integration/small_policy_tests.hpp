#pragma once
// G0 compares fields, never C/C++ structure padding or private opaque layout.
static bool same_mul_info(const sbn3_mul_info &a,const sbn3_mul_info &b){
    if(a.np!=b.np)return false;
    if(a.trunk_bits!=b.trunk_bits)return false;
    if(a.digit_words!=b.digit_words)return false;
    if(a.workers!=b.workers)return false;
    if(a.prime_batch!=b.prime_batch)return false;
    if(a.fused!=b.fused)return false;
    if(a.row_major!=b.row_major)return false;
    if(a.borrow_output!=b.borrow_output)return false;
    if(a.full!=b.full)return false;
    if(a.C!=b.C)return false;
    if(a.M2!=b.M2)return false;
    if(a.lbv!=b.lbv)return false;
    if(a.lbw!=b.lbw)return false;
    if(a.nat!=b.nat)return false;
    if(a.nyt!=b.nyt)return false;
    if(a.transform_trunks!=b.transform_trunks)return false;
    if(a.output_limbs!=b.output_limbs)return false;
    if(a.output_alignment!=b.output_alignment)return false;
    if(a.table_bytes!=b.table_bytes)return false;
    if(a.workspace_bytes!=b.workspace_bytes)return false;
    if(a.workspace_alignment!=b.workspace_alignment)return false;
    if(a.per_worker_bytes!=b.per_worker_bytes)return false;
    if(a.plane_pitch!=b.plane_pitch)return false;
    if(a.pool_bytes!=b.pool_bytes)return false;
    if(a.transpose_bytes!=b.transpose_bytes)return false;
    if(a.table_entries!=b.table_entries)return false;
    if(a.factor_levels!=b.factor_levels)return false;
    if(a.root_order_log2!=b.root_order_log2)return false;
    if(a.block_stride!=b.block_stride)return false;
    if(a.product_row_stride!=b.product_row_stride)return false;
    if(a.emit_trunks!=b.emit_trunks)return false;
    if(a.emit_limbs!=b.emit_limbs)return false;
    if(a.emit_quantum_trunks!=b.emit_quantum_trunks)return false;
    if(a.emit_quantum_limbs!=b.emit_quantum_limbs)return false;
    if(a.rowscale!=b.rowscale)return false;
    if(a.row_task_grain!=b.row_task_grain)return false;
    if(a.fused_row_grain!=b.fused_row_grain)return false;
    if(a.format_slot_bytes!=b.format_slot_bytes)return false;
    if(a.format_block_slots!=b.format_block_slots)return false;
    if(a.fused_items!=b.fused_items)return false;
    if(a.crt_mode!=b.crt_mode)return false;
    if(a.codec_mode!=b.codec_mode)return false;
    if(a.basis_id!=b.basis_id)return false;
    if(a.arithmetic_id!=b.arithmetic_id)return false;
    if(a.execution_id!=b.execution_id)return false;
    if(a.algorithm!=b.algorithm)return false;
    if(a.fused_start_skew_us!=b.fused_start_skew_us)return false;
    for(unsigned q=0;q<10;++q)if(a.scale_a[q]!=b.scale_a[q])return false;
    return true;
}
static void small_policy_query_gate(){
    for(size_t n:{size_t(32768),size_t(46341),size_t(65536),size_t(185363),size_t(1048576),size_t(4194304)})
    for(unsigned w:{1u,2u,4u,8u,16u}){
        sbn3_mul_options o{};o.workers=w;o.borrow_output=1;sbn3_product_spec s{n,n};sbn3_mul_plan p{},p2{};sbn3_mul_info i{},i2{};
        allocation_watch_start();assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_SUPPORTED);
        assert(sbn3_mul_query(&s,&o,&p2,&i2)==SBN3_SUPPORTED);assert(!allocation_watch_stop());
        assert(same_mul_info(i,i2) && i.workers==w && i.algorithm && !i.fused_start_skew_us);
        sbn3_product_request r{SBN3_PRODUCT_MUL,n,n,0,0,{},0,0,0};sbn3_product_info ri{};
        assert(sbn3_product_query(&r,&o,&p2,&ri)==SBN3_SUPPORTED && same_mul_info(i,ri.mul));
        o.workspace_budget=i.workspace_bytes;assert(sbn3_mul_query(&s,&o,&p2,&i2)==SBN3_SUPPORTED && i2.workspace_bytes<=o.workspace_budget);
        o.workspace_budget=1;memset(&p2,0x93,sizeof p2);assert(sbn3_mul_query(&s,&o,&p2,&i2)==SBN3_QUERY_CAPACITY && p2.opaque[0]==0x9393939393939393ULL);
    }
    sbn3_product_spec s{32768,32768};sbn3_mul_options o{};o.workers=4;o.prime_count=6;o.trunk_bits=128;o.column_log2=7;o.row_log2=5;o.fused_start_skew_us=-1;
    sbn3_mul_plan p{};sbn3_mul_info a{},b{};assert(sbn3_mul_query(&s,&o,&p,&a)==SBN3_SUPPORTED);
    o.fused_start_skew_us=40;assert(sbn3_mul_query(&s,&o,&p,&b)==SBN3_SUPPORTED);
    assert(a.basis_id==b.basis_id && a.arithmetic_id==b.arithmetic_id && a.execution_id!=b.execution_id && b.fused_start_skew_us==40);
    puts("small policy: deterministic mul/product identity, exact width, budget filtering, execution-only skew OK");
}
struct SmallBatch {CachedTask *tasks;unsigned count;};
static void small_batch_task(void *arg,sbn3_team_scope *scope){
    auto b=*static_cast<SmallBatch *>(arg);if(b.count==1){cached_task(b.tasks,scope);return;}
    SmallBatch left{b.tasks,b.count/2},right{b.tasks+b.count/2,b.count/2};
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,sbn3_team_width(scope)/2,small_batch_task,&left,small_batch_task,&right);
}
static void small_batch_gate(unsigned w){
    Fixture f(16);constexpr size_t n=32768;const unsigned count=16/w;CachedTask tasks[16]{};
    sbn3_mul_options o{};o.workers=w;o.borrow_output=1;sbn3_product_request req{SBN3_PRODUCT_MUL,n,n,0,0,{},0,0,0};
    for(unsigned j=0;j<count;++j){auto &t=tasks[j];sbn3_product_info i{};sbn3_mul_plan p{};t.b=f.product(req,o,i,p);t.rn=2*n;t.out=f.guarded(2*n);
        auto *a=f.guarded(n),*b=f.guarded(n);for(size_t k=0;k<n;++k){a[k]=random_word();b[k]=random_word();}t.inputs.a={a,n};t.inputs.b={b,n};}
    SmallBatch batch{tasks,count};allocation_watch_start();sbn3_team_run(f.team,small_batch_task,&batch);assert(!allocation_watch_stop());
    for(unsigned j=0;j<count;++j){const auto &t=tasks[j];verify_product(t.inputs.a.data,n,t.inputs.b.data,n,t.out);}
    printf("small B=%u W=%u: disjoint products, recursive team, GMP, protected tails, no allocation OK\n",count,w);
}
