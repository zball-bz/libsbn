// Original libsbn code; LGPL-3.0-or-later.
// Included in the instance's private namespace after Program. Integer matrix
// multiplication using one transform per entry and one inverse per output.
// Products are summed before CRT; the MAC2 plan certifies the two-term bound.
size_t program_matrix_bytes(const sbn3_mul_plan &opaque,unsigned rows,size_t first) {
    const auto q=load_plan(opaque);const auto &g=q.arithmetic.transform;
    if(LEAF!=8||!rows||rows>3||q.recipe.kind!=SBN3_PRODUCT_MAC2||q.recipe.cached_mask||
       q.recipe.window.width_bits||q.execution.info.algorithm!=SBN3_MUL_FLAT)return 0;
    const bool vector=first==matrix_vector_layout;
    const unsigned outputs=(vector?1:2)*unsigned(__builtin_popcount(rows));
    const size_t transforms=(vector?4:4+2*unsigned(__builtin_popcount(rows)))*(g.M2+16)*64;
    const size_t emission=PN*((p::OCH+32+7)/8)*64+4096;
    const unsigned workers=std::min<unsigned>(PN,q.execution.info.workers);
    const bool middle=g.ring_rn&&first!=SIZE_MAX&&!vector;
    if(middle&&first>g.lbw/2)return 0;
    const unsigned planes=middle?outputs/2:outputs;
    const size_t plane_vectors=middle?3*g.lbw/2-first:g.lbw;
    const size_t emit_trunks=middle?((8*plane_vectors+16+g.OT-1)/g.OT)*g.OT:g.ntp;
    if(q.execution.info.workers>1){
        const size_t w=q.execution.info.workers;
        const size_t transform_phase=workers*(transforms+512+sizeof(Frame));
        const size_t emit_phase=w*(emission+256+sizeof(Frame))+planes*(emit_trunks*g.T/8+((emit_trunks+p::OCH-1)/p::OCH)*64+256);
        return planes*PN*(plane_vectors*48+128)+std::max(transform_phase,emit_phase)+4096;
    }
    return planes*PN*(plane_vectors*48+128)+workers*(std::max(transforms,emission)+256)+
        sizeof(Frame)*workers+emit_trunks*g.T/8+((emit_trunks+p::OCH-1)/p::OCH)*64+2048;
}
struct MatrixEpisode {
    const Program *program;
    const MatrixProductInputs *input;
    uint8_t *planes[2][2][PN]{};
    uint64_t bias[PN]{};
};
inline void matrix_leaf(p::V *a[2][2],p::V *b[2][2],size_t at,unsigned rows,p::V roots,const p::PrimeV &pv){
    p::V right[2][2][8];
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
        for(unsigned k=0;k<8;++k)right[i][j][k]=b[i][j][at+k];
        p::tr8(right[i][j]);for(auto &x:right[i][j])x=p::sh2(x,pv.p2);
    }
    const p::V wc=_mm512_madd52hi_epu64(p::vset(1),roots,pv.p);
    for(unsigned row=0;row<2;++row)if(rows&(1u<<row)){
        p::V left[2][8],wrapped[2][8];
        for(unsigned j=0;j<2;++j){
            for(unsigned k=0;k<8;++k)left[j][k]=a[row][j][at+k];
            p::tr8(left[j]);for(auto &x:left[j])x=p::shp(p::sh2(x,pv.p2),pv.p);
            for(unsigned k=1;k<8;++k)wrapped[j][k]=p::shp(p::b52(left[j][k],wc,roots,pv.pn),pv.p);
        }
        for(unsigned col=0;col<2;++col){
            p::V out[8];
            for(unsigned k=0;k<8;++k){
                p::V lo0{},hi0{},lo1{},hi1{};
                for(unsigned i=0;i<8;++i){
                    const p::V x=k>=i?left[0][k-i]:wrapped[0][k+8-i];
                    const p::V y=k>=i?left[1][k-i]:wrapped[1][k+8-i];
                    lo0=_mm512_madd52lo_epu64(lo0,x,right[0][col][i]);
                    hi0=_mm512_madd52hi_epu64(hi0,x,right[0][col][i]);
                    lo1=_mm512_madd52lo_epu64(lo1,y,right[1][col][i]);
                    hi1=_mm512_madd52hi_epu64(hi1,y,right[1][col][i]);
                }
                // Sixteen products <p*2p give REDC <3p (p<2^48,
                // R=2^52). A single fold restores the inverse's <2p input.
                out[k]=p::sh2(p::redc(_mm512_add_epi64(lo0,lo1),_mm512_add_epi64(hi0,hi1),pv.J,pv.p,pv.M),pv.p2);
            }
            p::tr8(out);for(unsigned k=0;k<8;++k)a[row][col][at+k]=out[k];
        }
    }
}
struct MatrixBuffers {p::V *a[2][2]{},*b[2][2]{};};
void matrix_allocate(MatrixBuffers &x,Frame &frame,const p::Plan &g,unsigned rows){
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
        if(rows&(1u<<i))x.a[i][j]=frame.alloc<p::V>(g.M2+16);
        x.b[i][j]=frame.alloc<p::V>(g.M2+16);
    }
}
void matrix_forward_entry(const MatrixEpisode &c,MatrixBuffers &x,unsigned q,unsigned slot){
    const auto &v=*c.program;const auto &in=*c.input;const auto &g=v.plan.arithmetic.transform;
    const bool left=slot<4;const unsigned i=(slot%4)/2,j=slot%2;
    auto *dest=left?x.a[i][j]:x.b[i][j];if(!dest)return;
    const auto &pv=v.primes->V[q];const auto pack=p::pk52_mk();const auto *cache=left?in.left_cache:nullptr;
    if(left&&in.use_cached_left){
        unsigned ci=i,cj=j;if(cache->inverse_order!=in.middle&&i==j){ci=1-i;cj=1-j;}
        const auto *data=cache->data+((2*ci+cj)*PN+q)*(g.lbw*48+128);
        for(size_t k=0;k<g.lbw;++k)dest[k]=p::ld52(data+k*SLOT,pack);
    }else{
        const auto src=left?in.left[i][j]:in.right[i][j];
        p::flat_decode(dest,src.data,src.count,left?v.constants->a:v.constants->b,g,q);
        p::flat_forward(dest,g,p::flat_input_vectors(src.count,g.T),v.primes->P[q],pv);
        p::flat_pad_frequency(dest,g);
        if(cache){auto *data=cache->data+((2*i+j)*PN+q)*(g.lbw*48+128);
            for(size_t k=0;k<g.lbw;++k)p::st52(data+k*SLOT,dest[k],pack,pv);}
    }
}
void matrix_inverse_entry(const MatrixEpisode &c,MatrixBuffers &x,unsigned q,unsigned slot){
    const auto &v=*c.program;const auto &in=*c.input;const auto &g=v.plan.arithmetic.transform;
    const unsigned i=slot/2,j=slot%2;if(!x.a[i][j]||(in.vector&&j))return;
    const auto &pv=v.primes->V[q];const auto pack=p::pk52_mk();
    p::flat_inverse(x.a[i][j],g,v.primes->P[q],pv);p::flat_pad_frequency(x.a[i][j],g);
    const size_t first=in.middle?(j||in.middle_first_vector==SIZE_MAX?g.lbw/2:in.middle_first_vector):0;
    if(in.vector&&in.negative_left){const auto bias=p::vset(c.bias[q]);
        for(size_t k=first;k<g.lbw;++k){const auto value=_mm512_add_epi64(p::shp(p::sh2(x.a[i][j][k],pv.p2),pv.p),bias);
            p::st52(c.planes[i][j][q]+(k-first)*SLOT,value,pack,pv);}
    }else for(size_t k=first;k<g.lbw;++k)p::st52(c.planes[i][j][q]+(k-first)*SLOT,x.a[i][j][k],pack,pv);
}
// Keep the two source spectra, but stream the coefficient rows through
// the same two buffers. Four live transforms, six forwards, two inverses;
// use the existing tuned conv8x8 kernel instead of a second DOT2 kernel.
void matrix_vector_prime(const MatrixEpisode &c,unsigned lo,unsigned hi,Frame&frame){
    const auto &v=*c.program;const auto &g=v.plan.arithmetic.transform;MatrixBuffers x;
    for(unsigned j=0;j<2;++j){x.a[0][j]=frame.alloc<p::V>(g.M2+16);x.a[1][j]=x.a[0][j];x.b[j][0]=frame.alloc<p::V>(g.M2+16);}
    for(unsigned q=lo;q<hi;++q){const auto &pv=v.primes->V[q];
        matrix_forward_entry(c,x,q,4);matrix_forward_entry(c,x,q,6);
        for(unsigned row=0;row<2;++row){matrix_forward_entry(c,x,q,2*row);matrix_forward_entry(c,x,q,2*row+1);
            for(size_t at=0;at<g.lbw;at+=8){const auto roots=p::tower8<false>(v.primes->P[q],at);
                p::conv8x8(x.a[row][0]+at,x.b[0][0]+at,roots,pv);
                p::conv8x8(x.a[row][1]+at,x.b[1][0]+at,roots,pv);
                for(unsigned k=0;k<8;++k){const auto a=p::sh2(x.a[row][0][at+k],pv.p2),b=p::sh2(x.a[row][1][at+k],pv.p2);
                    x.a[row][0][at+k]=c.input->negative_left?
                        p::sh2(_mm512_sub_epi64(_mm512_add_epi64(row?b:a,pv.p2),row?a:b),pv.p2):
                        p::sh2(_mm512_add_epi64(a,b),pv.p2);}
            }
            matrix_inverse_entry(c,x,q,2*row);
        }
    }
}
void matrix_prime(void *ptr,uint64_t lo,uint64_t hi,int,Frame *frame) {
    const auto &c=*static_cast<MatrixEpisode *>(ptr);const auto &v=*c.program;
    const auto &g=v.plan.arithmetic.transform;
    if(c.input->vector){matrix_vector_prime(c,unsigned(lo),unsigned(hi),*frame);return;}
    MatrixBuffers x;matrix_allocate(x,*frame,g,c.input->rows);
    for(unsigned q=lo;q<hi;++q){
        // Preserve the prime-local pipeline and its original input ordering.
        for(unsigned j=0;j<4;++j){matrix_forward_entry(c,x,q,j);matrix_forward_entry(c,x,q,j+4);}
        for(size_t k=0;k<g.lbw;k+=8)matrix_leaf(x.a,x.b,k,c.input->rows,p::tower8<false>(v.primes->P[q],k),v.primes->V[q]);
        for(unsigned j=0;j<4;++j)matrix_inverse_entry(c,x,q,j);
    }
}
void matrix_parallel(const MatrixEpisode &c,Frame &space,sbn3_team_scope *scope,const p::Plan &emit_plan){
    const auto &v=*c.program;const auto &g=v.plan.arithmetic.transform;const auto &in=*c.input;
    const unsigned workers=v.plan.execution.info.workers,primes=std::min<unsigned>(PN,workers);
    sbn3_team_scope full{scope->team,scope->first,workers,false,scope->epoch};
    {
        FrameMark phase(space);
        Frame *frames[32]{};const size_t bytes=(in.vector?4:4+2*unsigned(__builtin_popcount(in.rows)))*(g.M2+16)*64+128;
        for(unsigned w=0;w<primes;++w){auto *at=space.allocate(sizeof(Frame));frames[w]=::new(at)Frame(space.subframe(bytes));}
        sbn3_team_scope sub{scope->team,scope->first,primes,false,scope->epoch};
        kernel_for(&sub,0,PN,1,SBN3_STATIC,matrix_prime,const_cast<MatrixEpisode*>(&c),frames);
        for(unsigned w=0;w<primes;++w)frames[w]->~Frame();
    }
    if(in.left_cache&&!in.use_cached_left){in.left_cache->arithmetic_id=v.plan.execution.info.arithmetic_id;
        in.left_cache->rows=in.rows;in.left_cache->inverse_order=in.middle;}
    // All input reads have completed. Reuse the transform pool for a wider
    // CRT team and independent carry journals for each output entry.
    FrameMark emission(space);Frame *frames[32]{};
    const size_t bytes=PN*((p::OCH+32+7)/8)*64+4096+128;
    for(unsigned w=0;w<workers;++w){auto *at=space.allocate(sizeof(Frame));frames[w]=::new(at)Frame(space.subframe(bytes));}
    struct EmitBatch {p::Plan plans[4];p::EmitCtx ctx[4]{};p::EmitMap maps[4];uint64_t end[5]{};unsigned count=0;};
    EmitBatch batch;
    for(unsigned i=0;i<2;++i)if(in.rows&(1u<<i))for(unsigned j=0;j<(in.middle||in.vector?1u:2u);++j){
        const unsigned k=batch.count++;auto &eg=batch.plans[k];eg=emit_plan;
        require(!in.middle||in.out[i][j].capacity>=eg.nl,SBN3_FATAL_ARGUMENT,"middle coefficient carry capacity",eg.nl,in.out[i][j].capacity);
        eg.outcap=std::min(in.middle?eg.nl:eg.outcap,in.out[i][j].capacity);
        auto &e=batch.ctx[k];e.pl=&eg;e.PS=v.primes;e.G=&v.garner;e.rp=in.out[i][j].data;
        e.tail=space.alloc<uint64_t>(eg.nl);e.spill=reinterpret_cast<uint64_t (*)[8]>(space.alloc<uint64_t>((eg.ntp+p::OCH-1)/p::OCH*8));
        e.nch=(eg.ntp+p::OCH-1)/p::OCH;e.EK=&v.emit.EK;e.RG=&v.emit.RG;e.map=&batch.maps[k];
        for(unsigned q=0;q<PN;++q)e.plane[q]=c.planes[i][j][q];
        batch.end[k+1]=batch.end[k]+p::emit_tasks(batch.maps[k],eg,e.nch,0);
    }
    kernel_for(&full,0,batch.end[batch.count],1,SBN3_STATIC,[](void *p,uint64_t a,uint64_t b,int rank,Frame *f){
        auto &c=*static_cast<EmitBatch*>(p);
        for(unsigned k=0;k<c.count;++k){const auto lo=std::max(a,c.end[k]),hi=std::min(b,c.end[k+1]);
            if(lo<hi)emit_range(&c.ctx[k],lo-c.end[k],hi-c.end[k],rank,f);}
    },&batch,frames);
    for(unsigned k=0;k<batch.count;++k){auto &e=batch.ctx[k];p::emit_join(e);
        if(in.vector&&g.ring_rn)p::fold_cyc(e.rp,g.ring_rn,e.tail,e.pl->nl-e.pl->outcap);
        uint64_t overflow=0;for(size_t j=e.pl->outcap;j<g.outcap;++j)overflow|=e.tail[j-e.pl->outcap];
        require(!overflow,SBN3_FATAL_MATH,"matrix product output bound");}
    for(unsigned w=0;w<workers;++w)frames[w]->~Frame();
}
void program_matrix_execute(const void *ptr,Frame &space,sbn3_team_scope *scope,const MatrixProductInputs &in){
    const auto &v=*static_cast<const Program *>(ptr);const auto &g=v.plan.arithmetic.transform;
    require(!in.middle||g.ring_rn,SBN3_FATAL_ARGUMENT,"matrix coefficient window recipe");
    require(!in.vector||(!in.middle&&in.rows==3&&!in.left_cache&&((g.ring_rn&&in.negative_left==6)||(!g.ring_rn&&!in.negative_left))),SBN3_FATAL_ARGUMENT,"NTT signed cyclic vector recipe");
    require(in.vector||!in.negative_left,SBN3_FATAL_ARGUMENT,"NTT matrix nonnegative entries");
    if(in.left_cache){
        const auto &cache=*in.left_cache;
        require(in.rows==3,SBN3_FATAL_ARGUMENT,"matrix spectrum requires both rows");
        require(cache.data&&cache.bytes>=4*PN*(g.lbw*48+128),SBN3_FATAL_WORKSPACE,"matrix spectrum storage");
        if(in.use_cached_left)require(cache.arithmetic_id==v.plan.execution.info.arithmetic_id&&cache.rows==in.rows,
                                      SBN3_FATAL_ARGUMENT,"matrix spectrum arithmetic identity");
    }else require(!in.use_cached_left,SBN3_FATAL_ARGUMENT,"matrix spectrum missing");
    if(g.ring_rn&&!in.middle&&!in.vector){
        require(in.use_cached_left,SBN3_FATAL_ARGUMENT,"linear matrix with ring geometry");
        for(unsigned i=0;i<2;++i)if(in.rows&(1u<<i))for(unsigned j=0;j<2;++j)for(unsigned k=0;k<2;++k)
            require(in.left[i][k].count+in.right[k][j].count+1<=g.ring_rn,
                    SBN3_FATAL_MATH,"matrix linear support inside ring");
    }
    const unsigned workers=std::min<unsigned>(PN,v.plan.execution.info.workers);
    require(scope&&scope->width>=workers&&in.rows&&in.rows<=3,SBN3_FATAL_ARGUMENT,"matrix program scope/rows");
    sbn3_team_scope sub{scope->team,scope->first,workers,false,scope->epoch};
    MatrixEpisode c{&v,&in};Frame *frames[32]{};
    if(in.vector&&in.negative_left){
        // |coefficient| <= m*(beta-1)^2. Adding that same C to every
        // coefficient leaves a nonnegative value below the MAC2 bound.
        // C*sum(beta^j)=m*(beta-1)*(beta^N-1), zero in the minus ring.
        // Convert C into the physical inverse-plane scale before CRT.
        const uint64_t m=(64*v.plan.product.a_limbs+g.T-1)/g.T;
        for(unsigned q=0;q<PN;++q){const uint64_t prime=p::PR[q],digit=(p::powm(2,g.T,prime)+prime-1)%prime;
            const uint64_t bias=p::mulm(m,p::mulm(digit,digit,prime),prime);
            const uint64_t physical=p::mulm(v.plan.arithmetic.product.scale[q],(g.rowscale*(g.Lg>1?8:1))%prime,prime);
            c.bias[q]=p::mulm(bias,physical,prime);
        }
    }
    const size_t first=in.middle&&in.middle_first_vector!=SIZE_MAX?in.middle_first_vector:g.lbw/2;
    require(!in.middle||first<=g.lbw/2,SBN3_FATAL_ARGUMENT,"middle first block origin");
    const size_t plane_vectors=in.middle?3*g.lbw/2-first:g.lbw;
    p::Plan emit_plan=g;
    if(in.middle){
        emit_plan.lbv=emit_plan.lbw=plane_vectors;emit_plan.vtrunks=8*plane_vectors;
        emit_plan.ntp=((emit_plan.vtrunks+16+g.OT-1)/g.OT)*g.OT;
        emit_plan.nl=emit_plan.ntp*g.T/64;
    }
    const unsigned outputs=2*unsigned(__builtin_popcount(in.rows));
    const size_t per_worker=std::max<size_t>((in.vector?4:4+outputs)*(g.M2+16)*64,PN*((p::OCH+32+7)/8)*64+4096)+128;
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
        require(in.right[i][j].count<=v.plan.product.b_limbs,SBN3_FATAL_ARGUMENT,"matrix right bound");
        if(in.rows&(1u<<i)){
            require(in.left[i][j].count<=v.plan.product.a_limbs,SBN3_FATAL_ARGUMENT,"matrix left bound");
            if(in.middle){
                require(in.left[i][j].count<=g.ring_rn/2,SBN3_FATAL_ARGUMENT,"middle matrix coefficient support");
                require((64*in.left[i][j].count+g.T-1)/g.T<=8*first,
                        SBN3_FATAL_ARGUMENT,"middle cyclic uncontaminated range");
                if(!j)for(unsigned q=0;q<PN;++q){
                    c.planes[i][0][q]=space.alloc<uint8_t>(plane_vectors*48+128);
                    c.planes[i][1][q]=c.planes[i][0][q]+(g.lbw-first)*48;
                }
            }else if(!in.vector||!j)for(unsigned q=0;q<PN;++q)c.planes[i][j][q]=space.alloc<uint8_t>(g.lbw*48+128);
        }
    }
    if(v.plan.execution.info.workers>1){
        require(scope->width>=v.plan.execution.info.workers,SBN3_FATAL_TEAM,"parallel matrix scope");
        matrix_parallel(c,space,scope,emit_plan);return;
    }
    for(unsigned w=0;w<workers;++w){auto *at=space.allocate(sizeof(Frame));frames[w]=::new(at)Frame(space.subframe(per_worker));}
    auto *tail=space.alloc<uint64_t>(emit_plan.nl);
    auto *spill=space.alloc<uint64_t>((emit_plan.ntp+p::OCH-1)/p::OCH*8);
    kernel_for(&sub,0,PN,1,SBN3_STATIC,matrix_prime,&c,frames);
    if(in.left_cache&&!in.use_cached_left){in.left_cache->arithmetic_id=v.plan.execution.info.arithmetic_id;
        in.left_cache->rows=in.rows;in.left_cache->inverse_order=in.middle;}
    // Canonicalize only after every prime has consumed every input. A
    // caller may place these outputs over the old left matrix entries.
    for(unsigned i=0;i<2;++i)if(in.rows&(1u<<i))for(unsigned j=0;j<(in.middle||in.vector?1u:2u);++j){
        p::Plan eg=emit_plan;
        require(!in.middle||in.out[i][j].capacity>=eg.nl,SBN3_FATAL_ARGUMENT,"middle coefficient carry capacity",eg.nl,in.out[i][j].capacity);
        eg.outcap=std::min(in.middle?eg.nl:eg.outcap,in.out[i][j].capacity);
        p::EmitCtx e{};e.pl=&eg;e.PS=v.primes;e.G=&v.garner;e.rp=in.out[i][j].data;
        e.tail=tail;e.spill=reinterpret_cast<uint64_t (*)[8]>(spill);
        e.nch=(eg.ntp+p::OCH-1)/p::OCH;e.EK=&v.emit.EK;e.RG=&v.emit.RG;
        for(unsigned q=0;q<PN;++q)e.plane[q]=c.planes[i][j][q];
        p::EmitMap map;e.map=&map;const auto tasks=p::emit_tasks(map,eg,e.nch,0);
        kernel_for(&sub,0,tasks,1,SBN3_STATIC,emit_range,&e,frames);p::emit_join(e);
        if(in.vector&&g.ring_rn)p::fold_cyc(e.rp,g.ring_rn,e.tail,eg.nl-eg.outcap);
        uint64_t overflow=0;for(size_t k=eg.outcap;k<g.outcap;++k)overflow|=tail[k-eg.outcap];
        require(!overflow,SBN3_FATAL_MATH,"matrix product output bound");
    }
    for(unsigned w=0;w<workers;++w)frames[w]->~Frame();
}
