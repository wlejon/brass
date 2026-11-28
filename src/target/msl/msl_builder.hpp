#pragma once

#include "msl_isel.hpp"
#include <brass/mir/opcodes.hpp>
#include <brass/target/gpu_intrinsics.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::msl {

using brass::gpu::CfgNode;
using brass::gpu::kNoNode;
using brass::gpu::StructuredCfg;

[[noreturn]] inline void fail_msl(const Function& fn, const std::string& msg) {
    throw std::runtime_error("MslTarget: kernel '" + std::string(fn.name()) + "': " + msg);
}

class MslFnBuilder {
public:
    MslFnBuilder(const Function& fn, const target::MslOptions& opts)
        : fn_(fn), opts_(opts) {}

    MslEmitter::KernelResult run();

private:
    const Function& fn_;
    const target::MslOptions& opts_;
    const StructuredCfg* cfg_ = nullptr;
    MslEmitter::KernelResult res_;
    std::ostringstream body_;
    std::unordered_map<const Value*, std::string> val_names_;
    std::unordered_set<const Value*> shared_ptrs_;
    std::unordered_set<const Value*> inlined_consts_;
    std::unordered_set<const Value*> local_vals_;
    std::unordered_map<const Value*, std::pair<std::string, int>> coop_vars_; // name, use (0=A,1=B,2=Acc)
    uint32_t shared_count_ = 0;
    uint32_t coop_count_ = 0;
    uint32_t tmp_seq_ = 0;
    bool need_tid_ = false;
    bool need_ntid_ = false;
    bool need_ctaid_ = false;
    bool need_nctaid_ = false;
    bool need_lane_ = false;
    bool need_sg_ = false;

    void indent_line(int d) {
        for (int i = 0; i < d; ++i) body_ << "    ";
    }

    static bool is_shared_alloc(std::string_view raw) {
        std::string_view sym = target::canonical_gpu_intrinsic(raw);
        return sym == "ptx_shared_alloc_f32" || sym == "ptx_shared_alloc_i32" ||
               sym == "ptx_shared_alloc_f64" || sym == "ptx_shared_alloc_i64" ||
               sym == "ptx_shared_alloc_i32x4" || sym == "ptx_shared_alloc_u16";
    }

    static bool is_coop_alloc(std::string_view raw) {
        std::string_view sym = target::canonical_gpu_intrinsic(raw);
        return sym == "ptx_coop_alloc_a" || sym == "ptx_coop_alloc_b" || sym == "ptx_coop_alloc_acc";
    }
    static int coop_use_kind(std::string_view raw) {
        std::string_view sym = target::canonical_gpu_intrinsic(raw);
        if (sym == "ptx_coop_alloc_a") return 0;
        if (sym == "ptx_coop_alloc_b") return 1;
        return 2;
    }

    static Type shared_elem_type(std::string_view raw) {
        std::string_view sym = target::canonical_gpu_intrinsic(raw);
        if (sym == "ptx_shared_alloc_i32") return Type::i32();
        if (sym == "ptx_shared_alloc_f64") return Type::f64();
        if (sym == "ptx_shared_alloc_i64") return Type::i64();
        if (sym == "ptx_shared_alloc_i32x4") return Type::i32x4();
        if (sym == "ptx_shared_alloc_u16") return Type::i16();
        return Type::f32();
    }

    static std::string shared_decl_elem_type(std::string_view raw, const MslFnBuilder& self) {
        std::string_view sym = target::canonical_gpu_intrinsic(raw);
        if (sym == "ptx_shared_alloc_u16") return "uint16_t";
        return self.msl_type(shared_elem_type(sym));
    }

    uint32_t const_i32_arg(const Value* v, const Instruction& inst) const;
    void find_shared_allocs();
    std::string phi_name(size_t node_idx, size_t param_idx) const {
        return "_phi_" + std::to_string(node_idx) + "_" + std::to_string(param_idx);
    }
    std::string msl_type(Type t) const;
    std::string msl_utype(Type t) const;
    std::string msl_param_type(Type t) const;
    uint32_t scalar_param_size(Type t) const;
    std::string msl_val_type(Type t, const Value* v) const;
    std::string zero_init(Type t, const Value* v) const;
    std::string val_expr(const Value* v) const;
    std::string ptr_space(const Value* ptr) const {
        return shared_ptrs_.count(ptr) ? "threadgroup" : "device";
    }
    void emit_edge_args(size_t edge_idx, int d);
    void emit_node(size_t n, size_t stop_node, size_t loop_merge, size_t loop_cont, int d);
    void emit_assign(const Instruction& inst, const std::string& expr, int d);
    void lower_inst(const Instruction& inst, int d);
    void lower_call(const Instruction& inst, int d);
    void lower_coop_call(const Instruction& inst, int d);
};

} // namespace brass::msl
