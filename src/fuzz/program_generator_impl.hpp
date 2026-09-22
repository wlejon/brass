#pragma once

// Internal state of the structured program generator, shared between the
// control-flow half (program_generator.cpp) and the memory half
// (program_generator_memory.cpp).

#include <brass/fuzz/program_generator.hpp>
#include <brass/fuzz/ir_mutator.hpp>
#include <brass/mir/builder.hpp>
#include <functional>
#include <string>
#include <vector>

namespace brass::fuzz::detail {

// A buffer every statement may touch: defined in the entry block, fully
// initialized before the first statement, `len` a power of two.
struct Buffer {
    Value* base = nullptr;  // ptr (alloca) or gcref (brass_gc_alloc)
    Type elem = Type::i64();
    uint8_t scale = 8;
    uint32_t len = 0;
    bool gc = false;
};

// A GC object whose slot 0 holds a gcref to one of the gc buffers of
// `len` elements of `elem` (all of which are interchangeable).
struct Holder {
    Value* obj = nullptr;
    Type elem = Type::i64();
    uint8_t scale = 8;
    uint32_t len = 0;
};

// Values an innermost loop carries besides the induction variable and acc.
struct LoopCtx {
    BasicBlock* exit = nullptr;
    std::vector<Value*> carried;
};

class GenState {
public:
    GenState(Module& mod, const ProgramGeneratorOptions& opts, uint64_t seed);

    Function* generate_entry(std::string_view name);
    void generate_helpers();

    // ---- values and scopes (program_generator.cpp)
    struct Mark {
        size_t n64 = 0;
        size_t n32 = 0;
    };
    Mark mark() const { return {i64s_.size(), i32s_.size()}; }
    void restore(const Mark& m) {
        i64s_.resize(m.n64);
        i32s_.resize(m.n32);
    }
    void add_value(Value* v);
    BasicBlock* new_block(const char* name);
    Value* konst(Type t, int64_t v);
    Value* interesting_const(Type t);
    Value* pick(Type t);
    Value* operand(Type t);
    Value* int_expr(Type t, int depth);
    Value* cond();
    Value* to_i64(Value* v);
    Value* safe_divisor(Value* d);
    void mix(Value* v);

    // ---- statements
    void stmts(int depth, uint32_t count);
    void stmt(int depth);
    void stmt_arith();
    void stmt_div();
    void stmt_diamond(int depth);
    void stmt_switch(int depth);
    void stmt_phi_consts(int depth);
    void stmt_loop(int depth);
    void stmt_call();
    void stmt_f64();
    void stmt_vector();

    // Emits `for (iv = 0; iv < n; iv += 1)` carrying acc and ctx.carried
    // around `body`. The body runs with the builder in the loop body and
    // must leave it in a block that falls through to the latch; it updates
    // acc_ and ctx.carried in place. After the call the builder sits in the
    // exit block with acc_ and ctx.carried rebound to its parameters.
    void emit_loop(Value* n, uint32_t max_trip, LoopCtx& ctx,
                   const std::function<void(Value* iv, LoopCtx& ctx)>& body);
    void emit_break_if(Value* c);
    // A trip count: a constant or a masked value, never above `max_trip`.
    Value* trip_count(uint32_t max_trip);
    uint32_t trip_budget() const;

    // ---- memory (program_generator_memory.cpp)
    void setup_buffers();
    void stmt_memory(int depth);
    void stmt_loop_group();
    void checksum_buffers();
    Value* in_bounds_index(const Buffer& buf);
    Value* elem_value(Type elem);

    Module& mod_;
    ProgramGeneratorOptions opts_;
    FuzzRng rng_;
    Builder b_;
    Function* fn_ = nullptr;
    std::vector<Value*> i64s_;
    std::vector<Value*> i32s_;
    Value* acc_ = nullptr;
    std::vector<Buffer> buffers_;
    std::vector<Holder> holders_;
    std::vector<std::string> helpers_;      // (i64, i64) -> i64
    std::vector<std::string> buf_helpers_;  // (gcref of >= 4 i64, i64) -> i64
    std::vector<LoopCtx*> loops_;
    uint32_t iter_product_ = 1;
    uint32_t stmts_left_ = 0;
    bool helper_mode_ = false;
};

} // namespace brass::fuzz::detail
