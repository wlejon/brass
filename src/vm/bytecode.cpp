#include <brass/vm/bytecode.hpp>
#include <stdexcept>
#include <algorithm>

namespace brass {

std::string_view bytecode_op_name(BytecodeOp op) noexcept {
    switch (op) {
        case BytecodeOp::nop: return "nop";
        case BytecodeOp::unreachable: return "unreachable";

        case BytecodeOp::iconst32: return "iconst32";
        case BytecodeOp::load_const: return "load_const";
        case BytecodeOp::patchable_const32: return "patchable_const32";
        case BytecodeOp::patchable_const64: return "patchable_const64";

        case BytecodeOp::mov: return "mov";
        case BytecodeOp::mov_imm: return "mov_imm";

        case BytecodeOp::sext64: return "sext64";
        case BytecodeOp::zext64: return "zext64";
        case BytecodeOp::trunc32: return "trunc32";
        case BytecodeOp::trunc8: return "trunc8";
        case BytecodeOp::fptosi32: return "fptosi32";
        case BytecodeOp::fptosi64: return "fptosi64";
        case BytecodeOp::fptosi32_f32: return "fptosi32_f32";
        case BytecodeOp::fptosi64_f32: return "fptosi64_f32";
        case BytecodeOp::sitofp_f64: return "sitofp_f64";
        case BytecodeOp::sitofp_f32: return "sitofp_f32";
        case BytecodeOp::sitofp_f64_i64: return "sitofp_f64_i64";
        case BytecodeOp::sitofp_f32_i64: return "sitofp_f32_i64";
        case BytecodeOp::fptrunc_f32: return "fptrunc_f32";
        case BytecodeOp::fpext_f64: return "fpext_f64";
        case BytecodeOp::bitcast_i64_f64: return "bitcast_i64_f64";
        case BytecodeOp::bitcast_f64_i64: return "bitcast_f64_i64";

        case BytecodeOp::add_i32: return "add_i32";
        case BytecodeOp::add_i64: return "add_i64";
        case BytecodeOp::sub_i32: return "sub_i32";
        case BytecodeOp::sub_i64: return "sub_i64";
        case BytecodeOp::mul_i32: return "mul_i32";
        case BytecodeOp::mul_i64: return "mul_i64";
        case BytecodeOp::sdiv_i32: return "sdiv_i32";
        case BytecodeOp::sdiv_i64: return "sdiv_i64";
        case BytecodeOp::udiv_i32: return "udiv_i32";
        case BytecodeOp::udiv_i64: return "udiv_i64";
        case BytecodeOp::smod_i32: return "smod_i32";
        case BytecodeOp::smod_i64: return "smod_i64";
        case BytecodeOp::umod_i32: return "umod_i32";
        case BytecodeOp::umod_i64: return "umod_i64";
        case BytecodeOp::neg_i32: return "neg_i32";
        case BytecodeOp::neg_i64: return "neg_i64";

        case BytecodeOp::add_f32: return "add_f32";
        case BytecodeOp::add_f64: return "add_f64";
        case BytecodeOp::sub_f32: return "sub_f32";
        case BytecodeOp::sub_f64: return "sub_f64";
        case BytecodeOp::mul_f32: return "mul_f32";
        case BytecodeOp::mul_f64: return "mul_f64";
        case BytecodeOp::fdiv_f32: return "fdiv_f32";
        case BytecodeOp::fdiv_f64: return "fdiv_f64";
        case BytecodeOp::neg_f32: return "neg_f32";
        case BytecodeOp::neg_f64: return "neg_f64";
        case BytecodeOp::fma_f32: return "fma_f32";
        case BytecodeOp::fma_f64: return "fma_f64";
        case BytecodeOp::sqrt_f32: return "sqrt_f32";
        case BytecodeOp::sqrt_f64: return "sqrt_f64";
        case BytecodeOp::fabs_f32: return "fabs_f32";
        case BytecodeOp::fabs_f64: return "fabs_f64";
        case BytecodeOp::floor_f32: return "floor_f32";
        case BytecodeOp::floor_f64: return "floor_f64";
        case BytecodeOp::ceil_f32: return "ceil_f32";
        case BytecodeOp::ceil_f64: return "ceil_f64";
        case BytecodeOp::round_f32: return "round_f32";
        case BytecodeOp::round_f64: return "round_f64";
        case BytecodeOp::fmin_f32: return "fmin_f32";
        case BytecodeOp::fmin_f64: return "fmin_f64";
        case BytecodeOp::fmax_f32: return "fmax_f32";
        case BytecodeOp::fmax_f64: return "fmax_f64";

        case BytecodeOp::sadd_overflow_i32: return "sadd_overflow_i32";
        case BytecodeOp::sadd_overflow_i64: return "sadd_overflow_i64";
        case BytecodeOp::ssub_overflow_i32: return "ssub_overflow_i32";
        case BytecodeOp::ssub_overflow_i64: return "ssub_overflow_i64";
        case BytecodeOp::smul_overflow_i32: return "smul_overflow_i32";
        case BytecodeOp::smul_overflow_i64: return "smul_overflow_i64";
        case BytecodeOp::uadd_overflow_i32: return "uadd_overflow_i32";
        case BytecodeOp::uadd_overflow_i64: return "uadd_overflow_i64";
        case BytecodeOp::usub_overflow_i32: return "usub_overflow_i32";
        case BytecodeOp::usub_overflow_i64: return "usub_overflow_i64";
        case BytecodeOp::umul_overflow_i32: return "umul_overflow_i32";
        case BytecodeOp::umul_overflow_i64: return "umul_overflow_i64";

        case BytecodeOp::and_i32: return "and_i32";
        case BytecodeOp::and_i64: return "and_i64";
        case BytecodeOp::or_i32: return "or_i32";
        case BytecodeOp::or_i64: return "or_i64";
        case BytecodeOp::xor_i32: return "xor_i32";
        case BytecodeOp::xor_i64: return "xor_i64";
        case BytecodeOp::shl_i32: return "shl_i32";
        case BytecodeOp::shl_i64: return "shl_i64";
        case BytecodeOp::lshr_i32: return "lshr_i32";
        case BytecodeOp::lshr_i64: return "lshr_i64";
        case BytecodeOp::ashr_i32: return "ashr_i32";
        case BytecodeOp::ashr_i64: return "ashr_i64";
        case BytecodeOp::not_i32: return "not_i32";
        case BytecodeOp::not_i64: return "not_i64";
        case BytecodeOp::clz_i32: return "clz_i32";
        case BytecodeOp::clz_i64: return "clz_i64";
        case BytecodeOp::ctz_i32: return "ctz_i32";
        case BytecodeOp::ctz_i64: return "ctz_i64";
        case BytecodeOp::popcnt_i32: return "popcnt_i32";
        case BytecodeOp::popcnt_i64: return "popcnt_i64";

        case BytecodeOp::eq_i32: return "eq_i32";
        case BytecodeOp::eq_i64: return "eq_i64";
        case BytecodeOp::eq_f32: return "eq_f32";
        case BytecodeOp::eq_f64: return "eq_f64";
        case BytecodeOp::ne_i32: return "ne_i32";
        case BytecodeOp::ne_i64: return "ne_i64";
        case BytecodeOp::ne_f32: return "ne_f32";
        case BytecodeOp::ne_f64: return "ne_f64";
        case BytecodeOp::slt_i32: return "slt_i32";
        case BytecodeOp::slt_i64: return "slt_i64";
        case BytecodeOp::ult_i32: return "ult_i32";
        case BytecodeOp::ult_i64: return "ult_i64";
        case BytecodeOp::lt_f32: return "lt_f32";
        case BytecodeOp::lt_f64: return "lt_f64";
        case BytecodeOp::sle_i32: return "sle_i32";
        case BytecodeOp::sle_i64: return "sle_i64";
        case BytecodeOp::ule_i32: return "ule_i32";
        case BytecodeOp::ule_i64: return "ule_i64";
        case BytecodeOp::le_f32: return "le_f32";
        case BytecodeOp::le_f64: return "le_f64";
        case BytecodeOp::sgt_i32: return "sgt_i32";
        case BytecodeOp::sgt_i64: return "sgt_i64";
        case BytecodeOp::ugt_i32: return "ugt_i32";
        case BytecodeOp::ugt_i64: return "ugt_i64";
        case BytecodeOp::gt_f32: return "gt_f32";
        case BytecodeOp::gt_f64: return "gt_f64";
        case BytecodeOp::sge_i32: return "sge_i32";
        case BytecodeOp::sge_i64: return "sge_i64";
        case BytecodeOp::uge_i32: return "uge_i32";
        case BytecodeOp::uge_i64: return "uge_i64";
        case BytecodeOp::ge_f32: return "ge_f32";
        case BytecodeOp::ge_f64: return "ge_f64";
        case BytecodeOp::select: return "select";

        case BytecodeOp::load8: return "load8";
        case BytecodeOp::load16: return "load16";
        case BytecodeOp::load32: return "load32";
        case BytecodeOp::load64: return "load64";
        case BytecodeOp::store8: return "store8";
        case BytecodeOp::store16: return "store16";
        case BytecodeOp::store32: return "store32";
        case BytecodeOp::store64: return "store64";
        case BytecodeOp::alloca_: return "alloca";

        case BytecodeOp::jump: return "jump";
        case BytecodeOp::jump_if: return "jump_if";
        case BytecodeOp::jump_if_not: return "jump_if_not";
        case BytecodeOp::ret: return "ret";
        case BytecodeOp::ret_void: return "ret_void";
        case BytecodeOp::switch_: return "switch";

        case BytecodeOp::call: return "call";
        case BytecodeOp::call_indirect: return "call_indirect";
        case BytecodeOp::patchable_call: return "patchable_call";
        case BytecodeOp::func_addr: return "func_addr";

        case BytecodeOp::safepoint: return "safepoint";
        case BytecodeOp::write_barrier: return "write_barrier";
        case BytecodeOp::guard: return "guard";
        case BytecodeOp::resume_point: return "resume_point";
        case BytecodeOp::osr_entry: return "osr_entry";
        case BytecodeOp::pinned_tls_read: return "pinned_tls_read";
        case BytecodeOp::pinned_tls_write: return "pinned_tls_write";
        case BytecodeOp::read_sp: return "read_sp";

        case BytecodeOp::throw_: return "throw";
        case BytecodeOp::invoke: return "invoke";
        case BytecodeOp::landing_pad: return "landing_pad";
        case BytecodeOp::resume: return "resume";
        case BytecodeOp::coro_create: return "coro_create";
        case BytecodeOp::coro_suspend: return "coro_suspend";
        case BytecodeOp::coro_resume: return "coro_resume";
        case BytecodeOp::coro_destroy: return "coro_destroy";

        case BytecodeOp::vadd: return "vadd";
        case BytecodeOp::vsub: return "vsub";
        case BytecodeOp::vmul: return "vmul";
        case BytecodeOp::vdiv: return "vdiv";
        case BytecodeOp::vfma: return "vfma";
        case BytecodeOp::vneg: return "vneg";
        case BytecodeOp::vmin: return "vmin";
        case BytecodeOp::vmax: return "vmax";
        case BytecodeOp::vsqrt: return "vsqrt";
        case BytecodeOp::vand: return "vand";
        case BytecodeOp::vor: return "vor";
        case BytecodeOp::vxor: return "vxor";
        case BytecodeOp::vnot: return "vnot";
        case BytecodeOp::vload: return "vload";
        case BytecodeOp::vstore: return "vstore";
        case BytecodeOp::vbroadcast: return "vbroadcast";
        case BytecodeOp::vextract_lane: return "vextract_lane";
        case BytecodeOp::vinsert_lane: return "vinsert_lane";
        case BytecodeOp::vshuffle: return "vshuffle";
        case BytecodeOp::vzero: return "vzero";

        case BytecodeOp::vmov: return "vmov";
        case BytecodeOp::index_addr: return "index_addr";
        case BytecodeOp::br_eq_i32: return "br_eq_i32";
        case BytecodeOp::br_ne_i32: return "br_ne_i32";
        case BytecodeOp::br_slt_i32: return "br_slt_i32";
        case BytecodeOp::br_sle_i32: return "br_sle_i32";
        case BytecodeOp::br_ult_i32: return "br_ult_i32";
        case BytecodeOp::br_ule_i32: return "br_ule_i32";
        case BytecodeOp::br_eq_i64: return "br_eq_i64";
        case BytecodeOp::br_ne_i64: return "br_ne_i64";
        case BytecodeOp::br_slt_i64: return "br_slt_i64";
        case BytecodeOp::br_sle_i64: return "br_sle_i64";
        case BytecodeOp::br_ult_i64: return "br_ult_i64";
        case BytecodeOp::br_ule_i64: return "br_ule_i64";
        case BytecodeOp::add_imm_i32: return "add_imm_i32";
        case BytecodeOp::add_imm_i64: return "add_imm_i64";
        case BytecodeOp::op_count_: break;
    }
    return "unknown";
}

bool is_bytecode_cond_branch(BytecodeOp op) noexcept {
    return op >= BytecodeOp::br_eq_i32 && op <= BytecodeOp::br_ule_i64;
}

bool is_bytecode_jump(BytecodeOp op) noexcept {
    return op == BytecodeOp::jump || op == BytecodeOp::jump_if || op == BytecodeOp::jump_if_not ||
           is_bytecode_cond_branch(op);
}

int64_t bytecode_branch_target(BytecodeWord inst, size_t pc) noexcept {
    BytecodeOp op = decode_op(inst);
    int64_t off = is_bytecode_cond_branch(op) ? decode_imm24(inst) : decode_imm32(inst);
    return static_cast<int64_t>(pc) + off;
}

bool is_bytecode_call(BytecodeOp op) noexcept {
    return op == BytecodeOp::call || op == BytecodeOp::call_indirect || op == BytecodeOp::patchable_call;
}

bool is_bytecode_terminator(BytecodeOp op) noexcept {
    switch (op) {
        case BytecodeOp::jump:
        case BytecodeOp::ret:
        case BytecodeOp::ret_void:
        case BytecodeOp::switch_:
        case BytecodeOp::unreachable:
        case BytecodeOp::throw_:
        case BytecodeOp::resume:
            return true;
        default:
            return false;
    }
}

uint32_t BytecodeFunction::add_constant(uint64_t val) {
    for (size_t i = 0; i < constants.size(); ++i) {
        if (constants[i] == val) {
            return static_cast<uint32_t>(i);
        }
    }
    constants.push_back(val);
    return static_cast<uint32_t>(constants.size() - 1);
}

uint32_t BytecodeFunction::add_constant_f64(double val) {
    uint64_t bits = 0;
    std::memcpy(&bits, &val, sizeof(double));
    return add_constant(bits);
}

uint32_t BytecodeFunction::add_constant_f32(float val) {
    uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(float));
    return add_constant(static_cast<uint64_t>(bits));
}

uint32_t BytecodeFunction::add_string_constant(std::string_view str) {
    for (size_t i = 0; i < string_pool.size(); ++i) {
        if (string_pool[i] == str) {
            return static_cast<uint32_t>(i);
        }
    }
    string_pool.emplace_back(str);
    return static_cast<uint32_t>(string_pool.size() - 1);
}

uint32_t BytecodeFunction::add_call_site(CallSiteInfo info) {
    call_sites.push_back(std::move(info));
    return static_cast<uint32_t>(call_sites.size() - 1);
}

uint32_t BytecodeFunction::add_switch_table(SwitchTable table) {
    switch_tables.push_back(std::move(table));
    return static_cast<uint32_t>(switch_tables.size() - 1);
}

uint32_t BytecodeFunction::add_guard(GuardInfo guard) {
    guards.push_back(std::move(guard));
    return static_cast<uint32_t>(guards.size() - 1);
}

uint32_t BytecodeFunction::add_patch_const(PatchConstSite site) {
    patch_consts.push_back(std::move(site));
    return static_cast<uint32_t>(patch_consts.size() - 1);
}

void BytecodeFunction::set_line_info(uint32_t pc, DebugLoc loc) {
    line_info_table.push_back({pc, loc});
}

DebugLoc BytecodeFunction::get_line_info(uint32_t pc) const {
    DebugLoc best;
    for (const auto& entry : line_info_table) {
        if (entry.pc <= pc) {
            best = entry.loc;
        } else {
            break;
        }
    }
    return best;
}

void BytecodeModule::add_function(std::unique_ptr<BytecodeFunction> fn) {
    if (!fn) return;
    fn->parent = this;
    std::string fn_name = fn->name;
    uint32_t idx = static_cast<uint32_t>(functions_.size());
    functions_.push_back(std::move(fn));
    if (!fn_name.empty()) {
        symbol_table_[fn_name] = idx;
    }
}

BytecodeFunction* BytecodeModule::get_function(std::string_view name) const noexcept {
    auto it = symbol_table_.find(std::string(name));
    if (it != symbol_table_.end() && it->second < functions_.size()) {
        return functions_[it->second].get();
    }
    return nullptr;
}

BytecodeFunction* BytecodeModule::get_function(size_t index) const noexcept {
    if (index < functions_.size()) {
        return functions_[index].get();
    }
    return nullptr;
}

void BytecodeModule::add_symbol(std::string_view name, uint32_t func_index) {
    symbol_table_[std::string(name)] = func_index;
}

uint32_t BytecodeModule::find_symbol(std::string_view name) const {
    auto it = symbol_table_.find(std::string(name));
    if (it != symbol_table_.end()) {
        return it->second;
    }
    throw std::runtime_error("Symbol not found: " + std::string(name));
}

bool BytecodeModule::has_symbol(std::string_view name) const {
    return symbol_table_.find(std::string(name)) != symbol_table_.end();
}

} // namespace brass
