#pragma once
#include "product/program.hpp"
namespace sbn::v3 {
struct ProductTableRef {
    SharedPreparation request{};
    const void *data = nullptr;
};
// Engine/context-owned immutable table directory. The owner keeps both this
// directory and the backing arena leases alive until every consumer finishes.
// A view can be passed to unrelated products and algorithm schedules.
struct ProductTablePoolView {
    const ProductTableRef *entries = nullptr;
    size_t count = 0;
    const void *find(const SharedPreparation &request) const {
        if (request.bytes)
            for (size_t j = 0; j < count; ++j)
                if (entries[j].request == request) return entries[j].data;
        return nullptr;
    }
};
template<size_t Capacity = 512> class ProductTablePool {
    ProductTableRef entries_[Capacity]{};
    size_t count_ = 0;
    bool sealed_ = false;
public:
    // Query/prepare run at a quiescent engine boundary, never in a hot task.
    // Capacity is metadata only: callers choose it when reserving the context.
    size_t additional_bytes(const ProductProgramPlan &p) const {
        return ProductTablePoolView{entries_, count_}.find(product_program_tables(p)) ? 0 : product_program_tables(p).bytes;
    }
    const void *prepare(const ProductProgramPlan &p, Frame &storage) {
        require(!sealed_ && product_program_tables(p).bytes, SBN3_FATAL_LIFETIME, "table pool preparation phase");
        if (const auto *data = ProductTablePoolView{entries_, count_}.find(product_program_tables(p))) return data;
        require(count_ < Capacity, SBN3_FATAL_WORKSPACE, "table pool directory capacity", count_ + 1, Capacity);
        auto frame = storage.subframe(product_program_tables(p).bytes, product_program_tables(p).alignment);
        const auto *backend = backend_lookup(p.plan.opaque[1]);
        const void *data = backend->program_tables_prepare(p.plan, frame);
        require(data == frame.data(), SBN3_FATAL_MATH, "table pool object base");
        entries_[count_++] = {product_program_tables(p), data};
        return data;
    }
    ProductTablePoolView seal() {
        sealed_ = true;
        return {entries_, count_};
    }
};
inline ProductProgram product_program_prepare_pooled(const ProductProgramPlan &p, Frame &local,
                                                      ProductTablePoolView pool) {
    const void *shared = pool.find(product_program_tables(p));
    require(!product_program_tables(p).bytes || shared, SBN3_FATAL_LIFETIME, "planned engine table missing");
    return product_program_prepare(p, local, shared);
}
} // namespace sbn::v3
