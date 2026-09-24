#pragma once
#include "sbn3/product.h"
namespace sbn::v3 {
// Optional, preplanned lifetime controller. Every byte and arithmetic recipe
// is determined before execution. A switch may destroy a dead binding and
// construct another in the same resident arena span, never allocate pages.
struct ProductStageOps {
    void (*enter)(void *, unsigned);
    sbn3_mul_binding *(*bind)(void *, unsigned);
    sbn3_spectrum *(*cache)(void *);
    void (*leave)(void *);
};
struct ProductStage {
    void *context;
    unsigned index;
    const ProductStageOps *ops;
};
class ProductStageRun {
    const ProductStage *stage_;

  public:
    explicit ProductStageRun(const ProductStage *s) : stage_(s) {
        if (stage_)
            stage_->ops->enter(stage_->context, stage_->index);
    }
    ~ProductStageRun() {
        if (stage_)
            stage_->ops->leave(stage_->context);
    }
    ProductStageRun(const ProductStageRun &) = delete;
    ProductStageRun &operator=(const ProductStageRun &) = delete;
    sbn3_mul_binding *bind(sbn3_mul_binding *fallback, unsigned slot) {
        return stage_ ? stage_->ops->bind(stage_->context, slot) : fallback;
    }
    sbn3_spectrum *cache(sbn3_spectrum *fallback) {
        return stage_ ? stage_->ops->cache(stage_->context) : fallback;
    }
};
} // namespace sbn::v3
