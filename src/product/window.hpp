#pragma once
#include "common/checked.hpp"
#include "runtime/scratch.hpp"
#include <algorithm>
#include <string.h>

namespace sbn::v3::product {
// Word-aligned mathematical windows, B=2^64. Physical transform padding and
// representation are deliberately absent. The window is
// floor(|E|/B^origin) modulo B^words, with a separate sign. error_bits==0
// requests the exact window and sign. A bounded result may have a different
// sign close to zero; its signed modular error must be below 2^error_bits.
// When a magnitude bound proves that the window includes all high words,
// this is an ordinary signed truncation-error bound.
struct Window {
    size_t origin = 0, words = 0;
    unsigned error_bits = 0;
};
struct Span {
    const uint64_t *data = nullptr;
    size_t words = 0;
};
struct ShiftedSpan {
    Span value{};
    size_t shift = 0;
};
// |E| < high_exclusive * B^words. This is an arithmetic precondition, not a
// request to pad the input or output to this many physical transform slots.
struct MagnitudeBound {
    size_t words = 0;
    uint64_t high_exclusive = 0;
    bool verify = true; // false when already implied by the input bounds
};
struct WindowResult {
    const uint64_t *data = nullptr;
    size_t words = 0;
    unsigned error_bits = 0;
    bool negative = false;
};
struct OperandShape {
    size_t words = 0;
    unsigned value_id = 0; // equal IDs denote the same mathematical value/view
};
struct AddendShape {
    size_t words = 0, shift = 0;
    bool unit = false;
};
struct WindowProductShape {
    OperandShape a{}, b{};
    Window window{};
    MagnitudeBound bound{};
    AddendShape subtract{};
    bool cancellation = false;
};
// A fixed, ordered group. Products may depend on preceding returned windows;
// no task-graph interpreter is required. The provider owns geometry/reuse.
struct WindowGroupShape {
    WindowProductShape products[3]{};
    unsigned count = 0;
};

// Correctness fallback for any ordinary full-multiplication implementation.
// Multiply::operator()(out,a,an,b,bn,Frame&) writes exactly an+bn words and
// preserves both inputs. Its temporary allocations must rewind before return.
// The complete integer buffer is caller-provided planned storage. Returned
// windows are borrowed until the next product; retain() makes a future input
// live across that product. Optimized providers may avoid this copy when their
// output/input lifetime contract permits it.
template<class Multiply> class FullProductWindows {
    Multiply &multiply_;
    uint64_t *value_;
    size_t capacity_;
    Frame &scratch_;

    void product(Span a, Span b) noexcept {
        require(a.data && b.data && a.words && b.words &&
                    a.words <= capacity_ && b.words <= capacity_ - a.words,
                SBN3_FATAL_ARGUMENT, "full window product capacity");
        multiply_(value_, a.data, a.words, b.data, b.words, scratch_);
        memset(value_ + a.words + b.words, 0, (capacity_ - a.words - b.words) * 8);
    }
    WindowResult window(Window w, bool negative = false) const noexcept {
        require(w.words && w.origin <= capacity_ && w.words <= capacity_ - w.origin,
                SBN3_FATAL_ARGUMENT, "full product window range");
        return {value_ + w.origin, w.words, 0, negative};
    }
    void bound(MagnitudeBound b) const noexcept {
        if (!b.high_exclusive || !b.verify) return;
        require(b.words < capacity_ && value_[b.words] < b.high_exclusive,
                SBN3_FATAL_MATH, "cancellation magnitude bound");
        uint64_t high = 0;
        for (size_t j = b.words + 1; j < capacity_; ++j) high |= value_[j];
        require(!high, SBN3_FATAL_MATH, "cancellation high support");
    }

  public:
    FullProductWindows(Multiply &mul, uint64_t *integer, size_t capacity, Frame &scratch) noexcept
        : multiply_(mul), value_(integer), capacity_(capacity), scratch_(scratch) {}

    WindowResult multiply(unsigned, Span a, Span b, Window w, MagnitudeBound limit = {}) noexcept {
        product(a, b);
        bound(limit);
        return window(w);
    }
    WindowResult cancel(unsigned, Span a, Span b, ShiftedSpan subtrahend,
                        Window w, MagnitudeBound limit) noexcept {
        product(a, b);
        require(subtrahend.shift <= capacity_ &&
                    subtrahend.value.words < capacity_ - subtrahend.shift &&
                    (!subtrahend.value.words || subtrahend.value.data),
                SBN3_FATAL_ARGUMENT, "cancellation addend capacity");
        uint64_t borrow = 0;
        for (size_t j = subtrahend.shift; j < capacity_; ++j) {
            const size_t k = j - subtrahend.shift;
            const uint64_t x = k < subtrahend.value.words ? subtrahend.value.data[k] : 0;
            const unsigned __int128 amount = (unsigned __int128)x + borrow;
            const uint64_t old = value_[j];
            value_[j] = old - uint64_t(amount);
            borrow = (unsigned __int128)old < amount;
        }
        const bool negative = borrow;
        if (negative) {
            uint64_t carry = 1;
            for (size_t j = 0; j < capacity_; ++j) {
                const unsigned __int128 x = (unsigned __int128)(~value_[j]) + carry;
                value_[j] = uint64_t(x);
                carry = uint64_t(x >> 64);
            }
        }
        bound(limit);
        return window(w, negative);
    }
    WindowResult retain(WindowResult r, uint64_t *destination) const noexcept {
        require(destination, SBN3_FATAL_ARGUMENT, "retained product window");
        if (r.data != destination) memmove(destination, r.data, r.words * 8);
        r.data = destination;
        return r;
    }
};

// Baseline factory for a backend that implements only complete multiplication.
// The leaf supplies workspace_bytes(a,b); every arithmetic recipe remains the
// same as for optimized window/cyclic providers.
template<class Multiply> struct FullWindowFactory {
    Multiply &multiply;
    static size_t integer_words(const WindowGroupShape &s) noexcept {
        size_t words=0;
        for(unsigned j=0;j<s.count;++j){const auto &r=s.products[j];
            words=std::max(words,r.a.words+r.b.words);
            if(r.cancellation)words=std::max(words,r.subtract.shift+r.subtract.words);
        }
        return words+1;
    }
    size_t bytes(const WindowGroupShape &s) const noexcept {
        size_t work=0;
        for(unsigned j=0;j<s.count;++j)work=std::max(work,multiply.workspace_bytes(s.products[j].a.words,s.products[j].b.words));
        return 8*integer_words(s)+work+256;
    }
    static size_t retained_words(const WindowGroupShape &s) noexcept {return s.products[s.count-2].window.words;}
    template<class F> void with_group(const WindowGroupShape &s,Frame &space,uint64_t *,F &&f) const noexcept {
        FrameMark mark(space);const size_t words=integer_words(s);auto *integer=space.alloc<uint64_t>(words);
        FullProductWindows<Multiply> products(multiply,integer,words,space);f(products);
    }
};
} // namespace sbn::v3::product
