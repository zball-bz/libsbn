// Original libsbn code; LGPL-3.0-or-later.
// Included in the private ISA island after Tables. Canonical DOT2 entries;
// each distinct input is forwarded once and each output inverted once.
bool matrix_supported(Shape s) noexcept {
    if(s.bits<16||s.bits>20||s.centered||s.recipe!=Recipe::CooleyTukeyPQ||s.branch<128||
       (s.branch&(s.branch-1))||s.nfull!=s.branch*s.radix||
       (s.radix!=1&&s.radix!=3&&s.radix!=5&&s.radix!=7))return false;
    // Separate DOT2 envelope: initially half the one-product cyclic cap.
    // The accumulated coefficient bound includes both product terms.
    return s.nfull<=matrix_cap(s.bits,s.radix,s.balanced);
}
namespace matrix_detail {
struct Episode {
    const Tables &t;const MatrixProductInputs &in;
    double *a[2][2]{},*b[2][2]{};
    unsigned columns()const{return in.vector?1:2;}
};
void forward(double*p,sbn3_const_limbs a,const Tables&t){forward_spectrum(p,a.data,a.count,t,nullptr);}
qcv signed_sum(qcv x,qcv y,unsigned signs,unsigned row){
    if(signs&(1u<<(2*row)))x=q_sub({sb_dzero(),sb_dzero()},x);
    return signs&(2u<<(2*row))?q_sub(x,y):q_add(x,y);
}
// DC/Nyquist have the existing PQ head's lane permutations. Use fixed
// stack blocks here only; all four results precede destructive stores.
void head(Episode &e){
    alignas(64) double out[2][2][128],term[2][128];
    const auto sc=sb_set1_d(1./e.t.shape.nfull),qsc=sb_set1_d(.25/e.t.shape.nfull);
    for(unsigned r=0;r<2;++r)if(e.in.rows&(1u<<r))for(unsigned c=0;c<e.columns();++c){
        for(unsigned k=0;k<2;++k){memcpy(term[k],e.a[r][k],sizeof term[k]);q_pw_head(term[k],e.b[k][c],&e.t.plan,sc,qsc);}
        for(unsigned j=0;j<128;j+=16)q_st(out[r][c]+j,signed_sum(q_ld(term[0]+j),q_ld(term[1]+j),e.in.negative_left,r));
    }
    for(unsigned r=0;r<2;++r)if(e.in.rows&(1u<<r))for(unsigned c=0;c<e.columns();++c)memcpy(e.b[r][c],out[r][c],sizeof out[r][c]);
}
void group(Episode &e,unsigned gl,unsigned gr,unsigned bl=0,unsigned br=0){
    const auto &t=e.t;const unsigned n=t.shape.branch;
    const auto sc=sb_set1_d(1./t.shape.nfull),qsc=sb_set1_d(.25/t.shape.nfull);
    const auto gs=bl?q_pq_gsel_rot(t.plan.pq+32*size_t(gl),t.ct->phase_re[bl],t.ct->phase_im[bl]):q_pq_gsel(t.plan.pq+32*size_t(gl));
    for(unsigned j=0;j<((!bl&&gl==gr)?4u:8u);++j){
        const size_t l=2*size_t(bl)*n+16*(8*size_t(gl)+j),r=2*size_t(br)*n+16*(8*size_t(gr)+7-j);
        qcv b[2][2],bn[2][2];
        for(unsigned k=0;k<2;++k)for(unsigned c=0;c<e.columns();++c){b[k][c]=q_ld(e.b[k][c]+l);bn[k][c]=q_rev(q_ld(e.b[k][c]+r));}
        const qcv w=q_pq_w(&gs,j);
        for(unsigned row=0;row<2;++row)if(e.in.rows&(1u<<row)){
            qcv a[2],an[2];for(unsigned k=0;k<2;++k){a[k]=q_ld(e.a[row][k]+l);an[k]=q_rev(q_ld(e.a[row][k]+r));}
            for(unsigned c=0;c<e.columns();++c){qcv x[2],y[2];
                for(unsigned k=0;k<2;++k)q_pq_pair(x+k,y+k,a[k],an[k],b[k][c],bn[k][c],w,sc,qsc);
                q_st(e.b[row][c]+l,signed_sum(x[0],x[1],e.in.negative_left,row));
                q_st(e.b[row][c]+r,q_rev(signed_sum(y[0],y[1],e.in.negative_left,row)));
            }
        }
    }
}
void leaves(Episode&e,unsigned branch,unsigned g,unsigned leaf){
    for(unsigned r=0;r<2;++r)if(e.in.rows&(1u<<r))for(unsigned c=0;c<e.columns();++c)
        q_ileaf_group(e.b[r][c]+2*size_t(branch)*e.t.shape.branch,g,leaf,&e.t.plan);
}
void pointwise(Episode&e,sbn3_team_scope *scope){
    const unsigned n=e.t.shape.branch,M=e.t.shape.radix,leaf=M==1?pq16_shape_of(n).leaf:pq16_odd_leaf(n);
    head(e);group(e,1,1);leaves(e,0,0,leaf);if(leaf!=128)leaves(e,0,1,leaf);
    // One dispatch, disjoint mirror-block pairs. A task owns both halves
    // of every 128-point inverse leaf that it retires.
    struct Tasks {Episode*e;unsigned n,M,leaf,pow2;};Tasks tasks{&e,n,M,leaf,n/256};
    sbn3_team_for(scope,0,n/256+(M/2)*(n/128),1,SBN3_STATIC,[](void*p,uint64_t lo,uint64_t hi,unsigned){
        auto &c=*static_cast<Tasks*>(p);auto &e=*c.e;
        for(auto t=lo;t<hi;++t){
            if(t<c.pow2){
                const unsigned block=t?1u<<(63-__builtin_clzll(t)):0;
                const unsigned base=t?256*block:128,g0=base/64,first=t?g0+2*(unsigned(t)-block):2;
                for(unsigned gl=first;gl<first+(t?2:1);++gl){const unsigned gr=3*g0-1-gl;group(e,gl,gr);
                    if(c.leaf!=128){leaves(e,0,gl,c.leaf);leaves(e,0,gr,c.leaf);}
                    else if(!t)leaves(e,0,gl,c.leaf);
                    else if(gl&1){leaves(e,0,gl-1,c.leaf);leaves(e,0,gr,c.leaf);}
                }
            }else{
                const unsigned at=unsigned(t)-c.pow2,b=1+at/(c.n/128),first=2*(at%(c.n/128));
                for(unsigned gl=first;gl<first+2;++gl){const unsigned gr=c.n/64-1-gl;group(e,gl,gr,b,c.M-b);
                    if(c.leaf!=128){leaves(e,b,gl,c.leaf);leaves(e,c.M-b,gr,c.leaf);}
                    else if(gl&1){leaves(e,b,gl-1,c.leaf);leaves(e,c.M-b,gr,c.leaf);}
                }
            }
        }
    },&tasks);
}
// Complete the inverse in-place into natural AoSoA coefficient pairs. There
// is no carry here: the two middle windows must be joined before carrying.
template<unsigned M>void inverse(double *data,const Tables&t){
    const unsigned n=t.shape.branch;
    if constexpr(M==1){
        pq16_inv_r8_only(data,n,&t.plan);const unsigned len=n/4;const bool compact=pq16_twc(n);
        const auto *tw=t.plan.tw22[__builtin_ctz(n)];
        for(unsigned j=0;j<len;j+=16){qcv w1,w2,w3,w4;pq16_tw22_get(tw,compact,&w1,&w2,&w3,&w4);tw+=pq16_tw_step(compact);
            for(unsigned u=0;u<2;++u){qcv x[4];for(unsigned k=0;k<4;++k)x[k]=q_ld(data+2*size_t(k)*len+2*j+16*u);
                q_ibf4(x,u?w3:w1,u?w4:w2);for(unsigned k=0;k<4;++k)q_st(data+2*size_t(k)*len+2*j+16*u,x[k]);}
        }
    }else{
        for(unsigned b=0;b<M;++b)pq16_odd_inv(data+2*size_t(b)*n,n,&t.plan);
        for(unsigned j=0;j<n;j+=8){qcv x[M],y[M];for(unsigned b=0;b<M;++b){x[b]=q_ld(data+2*size_t(b)*n+2*j);if(b)x[b]=q_mulc(x[b],ct_cross_root(*t.ct,b,j));}
            q_pfa_bfly(x,y,M,0);for(unsigned b=0;b<M;++b)q_st(data+2*size_t(b)*n+2*j,y[b]);}
    }
}
void inverse(double*p,const Tables&t){switch(t.shape.radix){case 1:inverse<1>(p,t);break;case 3:inverse<3>(p,t);break;case 5:inverse<5>(p,t);break;case 7:inverse<7>(p,t);break;}}
int64_t coefficient(const double*p,size_t digit){const size_t c=digit/2;return _mm_cvtsd_si64(_mm_set_sd(p[16*(c/8)+(digit&1)*8+c%8]));}
template<unsigned B>void emit(sbn3_limbs out,const double *first,size_t start,size_t end,const double *second,size_t start2,size_t end2,bool cyclic,bool linear,uint64_t bias){
    const size_t digits=end-start+end2-start2,words=digits*B/64;
    require(!((digits*B)%64)&&(linear||out.capacity>=words),SBN3_FATAL_ARGUMENT,"FFT matrix carry extent");
    // K=bias*(beta-1) is a geometric-series bias. Inject bias at the low
    // end and remove it from the high carry after the SINGLE joined window.
    const uint64_t mask=(uint64_t(1)<<B)-1,K=bias*mask;
    __int128 carry=bias;size_t written=0;unsigned fill=0;__uint128_t buffer=0;
    const auto rc=sb_set1_64(0x4338000000000000ULL-K);
    auto scalar=[&](const double*p,size_t at){
        carry+=__int128(coefficient(p,at))+K;
        buffer|=__uint128_t(uint64_t(carry)&mask)<<fill;fill+=B;carry>>=B;
        if(fill>=64){if(written<out.capacity)out.data[written]=uint64_t(buffer);
            else require(!uint64_t(buffer),SBN3_FATAL_MATH,"FFT matrix output support");
            ++written;buffer>>=64;fill-=64;}
    };
    auto store=[&](sb_vec z,size_t at,unsigned count){
        const unsigned live=unsigned(std::min<size_t>(count,out.capacity>at?out.capacity-at:0));
        if(live)q_st_tail(out.data+at,z,live);
        if(live<count)require((unsigned(sb_eq(z,sb_zero()))|((1u<<live)-1)|~((1u<<count)-1))==~0u,SBN3_FATAL_MATH,"FFT matrix zero output tail");
    };
    auto span=[&](const double*p,size_t at,size_t limit){
        while(at<limit&&(at%64)){scalar(p,at);++at;}
        require(!fill,SBN3_FATAL_MATH,"FFT matrix digit/limb boundary");
        q_chain chain{};const uint64_t initial=uint64_t(carry);memcpy(reinterpret_cast<char*>(&chain.prev_hi)+56,&initial,8);
        while(at+64<=limit){
            const auto a=vx_pair<B>(q_ld(p+at),chain,rc),b=vx_pair<B>(q_ld(p+at+16),chain,rc),
                       c=vx_pair<B>(q_ld(p+at+32),chain,rc),d=vx_pair<B>(q_ld(p+at+48),chain,rc);
            constexpr unsigned mid=512/(2*B)-8,last=1024/(2*B)-24;
            store(vx_pair_words<B,0>(a,b),written,8);
            if constexpr(mid==8)store(vx_pair_words<B,8>(c,d),written+8,8);
            else store(vx_pair_words<B,8>(sb_alignr64(c,b,mid),sb_alignr64(d,c,mid)),written+8,8);
            if constexpr(B>16)store(vx_pair_words<B,16>(sb_alignr64(sb_zero(),d,last),sb_zero()),written+16,B-16);
            written+=B;at+=64;
        }
        uint64_t high;memcpy(&high,reinterpret_cast<const char*>(&chain.prev_hi)+56,8);carry=__int128(high)+chain.cin;
        while(at<limit){scalar(p,at);++at;}
        require(!fill,SBN3_FATAL_MATH,"FFT matrix whole-limb window");
    };
    span(first,start,end);if(second)span(second,start2,end2);carry-=bias;
    if(cyclic){
        while(carry)for(size_t w=0;w<words&&carry;++w){carry+=__int128(out.data[w]);out.data[w]=uint64_t(carry);carry>>=64;}
        uint64_t all=UINT64_MAX;for(size_t w=0;w<words;++w)all&=out.data[w];if(all==UINT64_MAX)memset(out.data,0,8*words);
    }else if(linear){require(!carry,SBN3_FATAL_MATH,"FFT matrix final carry");}
    else for(size_t w=words;w<out.capacity;++w){out.data[w]=uint64_t(carry);carry>>=64;}
}
void emit(unsigned bits,sbn3_limbs out,const double*first,size_t start,size_t end,const double*second,size_t start2,size_t end2,bool cyclic,bool linear,uint64_t bias){
#define MATRIX_EMIT(B) case B:emit<B>(out,first,start,end,second,start2,end2,cyclic,linear,bias);return
    switch(bits){MATRIX_EMIT(16);MATRIX_EMIT(17);MATRIX_EMIT(18);MATRIX_EMIT(19);MATRIX_EMIT(20);}
#undef MATRIX_EMIT
    fatal(SBN3_FATAL_ARGUMENT,"FFT matrix codec");
}
}
void matrix_execute(const Tables&t,Frame&space,sbn3_team_scope*scope,const MatrixProductInputs&in,size_t output_words,uint64_t arithmetic_id) noexcept {
    using namespace matrix_detail;FrameMark lifetime(space);const size_t N=t.shape.nfull,bytes=16*N;
    Episode e{t,in};const size_t first=in.middle_first_vector==SIZE_MAX?N/8:in.middle_first_vector;
    if(in.left_cache){auto &c=*in.left_cache;require(c.data&&c.bytes>=4*bytes&&in.rows==3,SBN3_FATAL_WORKSPACE,"FFT matrix spectrum storage");
        if(in.use_cached_left)require(c.arithmetic_id==arithmetic_id&&c.rows==in.rows,SBN3_FATAL_ARGUMENT,"FFT matrix spectrum identity");
    }else require(!in.use_cached_left,SBN3_FATAL_ARGUMENT,"FFT matrix spectrum absent");
    for(unsigned r=0;r<2;++r)for(unsigned c=0;c<2;++c){
        if(in.rows&(1u<<r)){
            unsigned cr=r,cc=c;if(in.use_cached_left&&in.left_cache->inverse_order!=in.middle&&r==c){cr=1-r;cc=1-c;}
            e.a[r][c]=in.left_cache?reinterpret_cast<double*>(in.left_cache->data+(2*cr+cc)*bytes):space.alloc<double>(2*N);
            if(in.middle)require((64*in.left[r][c].count+t.shape.bits-1)/t.shape.bits+unsigned(t.shape.balanced)<=8*first,SBN3_FATAL_ARGUMENT,"FFT middle unpolluted range");
        }
        if(!in.vector||!c)e.b[r][c]=space.alloc<double>(2*N);
    }
    struct Forwards {Episode*e;unsigned count=0,slots[8]{};} forwards{&e};
    for(unsigned slot=0;slot<8;++slot){const bool left=slot<4;const unsigned r=slot%4/2,c=slot%2;
        const bool needed=left?(e.a[r][c]&&!in.use_cached_left):e.b[r][c]!=nullptr;
        if(!in.compact_forward||needed)forwards.slots[forwards.count++]=slot;
    }
    sbn3_team_for(scope,0,forwards.count,1,SBN3_STATIC,[](void*p,uint64_t lo,uint64_t hi,unsigned){auto&f=*static_cast<Forwards*>(p);auto&e=*f.e;
        for(auto index=lo;index<hi;++index){const unsigned slot=f.slots[index];const bool left=slot<4;const unsigned r=slot%4/2,c=slot%2;
            if(left){if(e.a[r][c]&&!e.in.use_cached_left)forward(e.a[r][c],e.in.left[r][c],e.t);}
            else if(e.b[r][c])forward(e.b[r][c],e.in.right[r][c],e.t);
        }
    },&forwards);
    if(in.left_cache&&!in.use_cached_left){in.left_cache->arithmetic_id=arithmetic_id;in.left_cache->rows=in.rows;in.left_cache->inverse_order=in.middle;}
    pointwise(e,scope);
    sbn3_team_for(scope,0,4,1,SBN3_STATIC,[](void*p,uint64_t lo,uint64_t hi,unsigned){auto &e=*static_cast<Episode*>(p);
        for(auto s=lo;s<hi;++s){const unsigned r=unsigned(s)/2,c=unsigned(s)%2;if((e.in.rows&(1u<<r))&&c<e.columns())inverse(e.b[r][c],e.t);}
    },&e);
    // Every source read has joined. Aliased canonical outputs are now safe.
    const uint64_t bias=t.shape.balanced?vx_bias_m(t.shape.bits,unsigned(2*N+2)):in.negative_left?2*N*((uint64_t(1)<<t.shape.bits)-1)+1:0;
    struct Outputs {Episode *e;size_t first,words;uint64_t bias;};Outputs outputs{&e,first,output_words,bias};
    sbn3_team_for(scope,0,4,1,SBN3_STATIC,[](void*p,uint64_t lo,uint64_t hi,unsigned){auto &v=*static_cast<Outputs*>(p);auto&e=*v.e;const auto &in=e.in;const size_t N=e.t.shape.nfull;
        for(auto s=lo;s<hi;++s){const unsigned r=unsigned(s)/2,c=unsigned(s)%2;if(!(in.rows&(1u<<r)))continue;
            if(in.middle){if(!c)emit(e.t.shape.bits,in.out[r][0],e.b[r][0],8*v.first,2*N,e.b[r][1],N,2*N,false,false,v.bias);}
            else if(c<e.columns()){const bool cyclic=in.vector&&in.negative_left;const size_t words=cyclic?cyclic_period(e.t.shape):std::min(v.words,in.out[r][c].capacity);
                emit(e.t.shape.bits,{in.out[r][c].data,words},e.b[r][c],0,2*N,nullptr,0,0,cyclic,!cyclic,v.bias);}
        }
    },&outputs);
}
