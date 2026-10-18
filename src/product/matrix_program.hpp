// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "product/fixed_geometry.hpp"
#include "product/program.hpp"
#include "product/fft_matrix.hpp"
namespace sbn::v3::product {
struct MatrixProgramPlan {ProductProgramPlan product{};size_t workspace_bytes=0;};
struct MatrixMiddlePlan {MatrixProgramPlan matrix{};size_t half_words=0;unsigned digit_bits=0,error_bits=0;};
inline bool matrix_vector_query(size_t minimum,size_t maximum,size_t coefficient,unsigned workers,
                                bool fft,bool ntt,MatrixProgramPlan&out,size_t&period){
    if(fft){const auto s=fft_matrix_shape(std::max(minimum,2*coefficient),maximum);
        MatrixProgramPlan f{};auto&q=f.product;const size_t r=pq16::cyclic_period(s);
        if(s.nfull&&fft_matrix_query(coefficient,r,r,std::min(workers,2u),s,q.plan,q.info)&&
           product_program_describe(coefficient,r,q,ProgramCapability::matrix)==SBN3_SUPPORTED){
            f.workspace_bytes=fft_matrix_backend().program_matrix_bytes(q.plan,3,matrix_vector_layout);out=f;period=r;return true;
        }
    }
    if(!ntt)return false;FixedNttGeometry g{};
    if(!fixed_ntt_geometry(minimum,coefficient,minimum,workers,0,g,size_t(1)<<19,0,false,FixedNttProduct::sum2,maximum)||g.algorithm!=SBN3_MUL_FLAT)return false;
    const auto*b=backend_lookup(g.np);if(!b->program_matrix_bytes)return false;
    auto o=fixed_ntt_options(g);o.borrow_output=0;
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_MAC2;r.a_limbs=r.a1_limbs=coefficient;r.b_limbs=r.b1_limbs=g.ring;r.cyclic_limbs=g.ring;
    sbn3_product_info i{};MatrixProgramPlan f{};auto&q=f.product;
    if(b->product_query(r,o,q.plan,i)!=SBN3_SUPPORTED)return false;
    q.info=i.mul;q.a1=coefficient;q.b1=g.ring;
    if(product_program_describe(coefficient,g.ring,q,ProgramCapability::matrix)!=SBN3_SUPPORTED)return false;
    f.workspace_bytes=b->program_matrix_bytes(q.plan,3,matrix_vector_layout);if(!f.workspace_bytes)return false;
    out=f;period=g.ring;return true;
}
inline bool matrix_middle_query(size_t n,unsigned workers,MatrixMiddlePlan &out,size_t guard_words=0,bool legacy_capacity=false,bool native_fft=false){
    FixedNttGeometry g{};const size_t minimum=(n+guard_words+1)/2;
    if(native_fft){
        const auto s=fft_matrix_shape(minimum,2*n/3);const size_t ring=pq16::cyclic_period(s);
        MatrixMiddlePlan f{};auto &q=f.matrix.product;
        if(s.nfull&&fft_matrix_query(ring/2,ring,ring,workers,s,q.plan,q.info)&&
           product_program_describe(ring/2,ring,q,ProgramCapability::matrix)==SBN3_SUPPORTED){
            f.half_words=ring/2;f.digit_bits=s.bits;
            f.error_bits=s.bits+2+unsigned(64-__builtin_clzll(2*s.nfull));
            f.matrix.workspace_bytes=fft_matrix_backend().program_matrix_bytes(q.plan,3,q.info.lbw/2);
            out=f;return true;
        }
    }
    if(!fixed_ntt_geometry(minimum,(n+3)/4,minimum,workers,0,g,size_t(1)<<19,0,false,
                           legacy_capacity?FixedNttProduct::multiply:FixedNttProduct::middle_matrix,
                           legacy_capacity?SIZE_MAX:2*n/3)||g.algorithm!=SBN3_MUL_FLAT||g.ring>2*n/3)return false;
    const size_t coefficient=g.ring/2;
    const auto *backend=backend_lookup(g.np);
    if(!backend->program_matrix_bytes||!backend->program_matrix_execute)return false;
    auto o=fixed_ntt_options(g);o.borrow_output=0;
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_MAC2;r.a_limbs=r.a1_limbs=coefficient;
    r.b_limbs=r.b1_limbs=g.ring;r.cyclic_limbs=g.ring;
    sbn3_product_info i{};out={};auto &q=out.matrix.product;
    if(backend->product_query(r,o,q.plan,i)!=SBN3_SUPPORTED)return false;
    q.info=i.mul;q.a1=coefficient;q.b1=g.ring;
    if(product_program_describe(coefficient,g.ring,q,ProgramCapability::matrix)!=SBN3_SUPPORTED)return false;
    out.matrix.workspace_bytes=backend->program_matrix_bytes(q.plan,3,q.info.lbw/2);
    out.half_words=g.ring/2;out.digit_bits=g.T;
    const size_t m=g.ring*32/g.T;
    // Each omitted signed carry has |carry| < m*2^T. One extra bit covers
    // the fractional original tail when a returned prefix matrix is lifted.
    out.error_bits=unsigned(g.T)+unsigned(64-__builtin_clzll(m))+1;
    return out.matrix.workspace_bytes!=0;
}
inline bool matrix_program_query(size_t a,size_t b,unsigned workers,unsigned rows,MatrixProgramPlan &out,bool native_fft=false,bool column=false){
    if(column&&rows!=3)return false;
    if(native_fft){const auto s=fft_matrix_shape(std::max(a+b+1,2*a),SIZE_MAX);MatrixProgramPlan f{};
        if(s.nfull&&fft_matrix_query(a,b,0,std::min(workers,(column?1u:2u)*unsigned(__builtin_popcount(rows))),s,f.product.plan,f.product.info)&&
           product_program_describe(a,b,f.product,ProgramCapability::matrix)==SBN3_SUPPORTED){
            f.workspace_bytes=fft_matrix_backend().program_matrix_bytes(f.product.plan,rows,column?matrix_vector_layout:SIZE_MAX);out=f;return true;
        }
    }
    FixedNttGeometry geometry{};
    if(!fixed_ntt_geometry(a,b,0,workers,0,geometry)||geometry.algorithm!=SBN3_MUL_FLAT)return false;
    const auto *backend=backend_lookup(geometry.np);
    if(!backend->program_matrix_bytes||!backend->program_matrix_execute)return false;
    auto o=fixed_ntt_options(geometry);o.borrow_output=0;
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_MAC2;r.a_limbs=r.a1_limbs=a;r.b_limbs=r.b1_limbs=b;
    sbn3_product_info i{};out={};
    // MAC has one extra accumulation bit. Let the same backend select its
    // widest legal codec; no geometry/threshold grid is searched by GCD.
    o.trunk_bits=0;
    if(backend->product_query(r,o,out.product.plan,i)!=SBN3_SUPPORTED)return false;
    out.product.info=i.mul;out.product.a1=a;out.product.b1=b;
    if(product_program_describe(a,b,out.product,ProgramCapability::matrix)!=SBN3_SUPPORTED)return false;
    out.workspace_bytes=backend->program_matrix_bytes(out.product.plan,rows,column?matrix_vector_layout:SIZE_MAX);
    return out.workspace_bytes!=0;
}
inline size_t matrix_program_workspace(const MatrixProgramPlan &p,unsigned rows,size_t first_vector=SIZE_MAX,bool cached_left=false){
    const auto *backend=backend_lookup(p.product.plan.opaque[1]);
    const size_t bytes=backend->program_matrix_bytes(p.product.plan,rows,first_vector);
    // FFT caches are already executable arrays; unlike packed p48 they
    // need no unpacked copy in the transient transform pool.
    if(bytes&&cached_left&&p.product.info.algorithm==SBN3_MUL_PQ16)
        return bytes-2*unsigned(__builtin_popcount(rows))*(8*p.product.info.transform_trunks+128);
    return bytes;
}
inline size_t matrix_spectrum_bytes(const MatrixProgramPlan &p){
    if(p.product.info.algorithm==SBN3_MUL_PQ16)return 32*p.product.info.transform_trunks;
    return 4*p.product.info.np*(p.product.info.lbw*48+128);
}
inline size_t matrix_middle_output_words(const MatrixProgramPlan &p,size_t first_vector){
    const auto &i=p.product.info;const size_t trunks=8*(3*i.lbw/2-first_vector);
    return ((trunks+16+i.emit_quantum_trunks-1)/i.emit_quantum_trunks)*i.emit_quantum_limbs;
}
inline void matrix_program_execute(const ProductProgram &p,Frame &work,sbn3_team_scope *scope,const MatrixProductInputs &in){
    require(p.backend&&p.backend->program_matrix_execute&&scope,SBN3_FATAL_ARGUMENT,"matrix program capability");
    if(in.vector){
        require(in.rows==3&&!in.middle&&!in.left_cache&&(in.negative_left==6||in.negative_left==0),SBN3_FATAL_ARGUMENT,"matrix vector recipe");
        for(unsigned row=0;row<2;++row)require(in.out[row][0].data&&in.out[row][0].capacity>=p.output_capacity,SBN3_FATAL_ARGUMENT,"inverse vector output extent");
        require(!overlaps(in.out[0][0].data,8*p.output_capacity,in.out[1][0].data,8*p.output_capacity),SBN3_FATAL_ARGUMENT,"inverse vector outputs overlap");
    }
    FrameMark mark(work);p.backend->program_matrix_execute(p.prepared,work,scope,in);
}
}
