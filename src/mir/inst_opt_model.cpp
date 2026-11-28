#include <brass/mir/inst_opt_model.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace brass {

namespace {

#include "inst_opt_weights_builtin.inc"

constexpr uint32_t kBrimMagic = 0x4D495242u; // "BRIM"
constexpr uint32_t kBrimVersion = 1u;
constexpr uint32_t kBrimFlagFp32 = 0u;
constexpr uint32_t kBrimFlagInt8 = 1u;

uint32_t crc32_bytes(const uint8_t* data, size_t len) noexcept {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (uint32_t bit = 0; bit < 8; ++bit) {
            const uint32_t mask = static_cast<uint32_t>(-(static_cast<int32_t>(crc & 1u)));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

float target_to_scalar(InstOptTarget t) noexcept {
    switch (t) {
        case InstOptTarget::X64:     return 0.0f;
        case InstOptTarget::ARM64:   return 0.5f;
        case InstOptTarget::GPU_PTX: return 1.0f;
    }
    return 0.0f;
}

bool is_high_lat_eop(EOp op) noexcept {
    return op == EOp::Mul || op == EOp::SDiv || op == EOp::UDiv || op == EOp::SMod || op == EOp::UMod;
}

bool is_cmp_or_select_eop(EOp op) noexcept {
    return op == EOp::Eq || op == EOp::Ne || op == EOp::Slt || op == EOp::Ult ||
           op == EOp::Sle || op == EOp::Ule || op == EOp::Select;
}

bool map_mir_op_to_eop(const Instruction* inst, EOp& out_op) noexcept {
    if (!inst || inst->has_side_effects() || !inst->produces_value()) return false;
    const Type res_t = inst->type();
    if (res_t != Type::i32() && res_t != Type::i64()) return false;
    switch (inst->opcode()) {
        case Opcode::add:       out_op = EOp::Add;     return true;
        case Opcode::sub:       out_op = EOp::Sub;     return true;
        case Opcode::mul:       out_op = EOp::Mul;     return true;
        case Opcode::sdiv:      out_op = EOp::SDiv;    return true;
        case Opcode::udiv:      out_op = EOp::UDiv;    return true;
        case Opcode::smod:      out_op = EOp::SMod;    return true;
        case Opcode::umod:      out_op = EOp::UMod;    return true;
        case Opcode::and_:      out_op = EOp::And;     return true;
        case Opcode::or_:       out_op = EOp::Or;      return true;
        case Opcode::xor_:      out_op = EOp::Xor;     return true;
        case Opcode::shl:       out_op = EOp::Shl;     return true;
        case Opcode::lshr:      out_op = EOp::LShr;    return true;
        case Opcode::ashr:      out_op = EOp::AShr;    return true;
        case Opcode::neg:       out_op = EOp::Neg;     return true;
        case Opcode::not_:      out_op = EOp::Not;     return true;
        case Opcode::select:    out_op = EOp::Select;  return true;
        case Opcode::sext_i64:  out_op = EOp::SExt64;  return true;
        case Opcode::zext_i64:  out_op = EOp::ZExt64;  return true;
        case Opcode::trunc_i32: out_op = EOp::Trunc32; return true;
        case Opcode::eq:        out_op = EOp::Eq;      return true;
        case Opcode::ne:        out_op = EOp::Ne;      return true;
        case Opcode::slt:
        case Opcode::sgt:       out_op = EOp::Slt;     return true;
        case Opcode::ult:
        case Opcode::ugt:       out_op = EOp::Ult;     return true;
        case Opcode::sle:
        case Opcode::sge:       out_op = EOp::Sle;     return true;
        case Opcode::ule:
        case Opcode::uge:       out_op = EOp::Ule;     return true;
        default:                return false;
    }
}

struct FeatureAccum {
    std::array<uint32_t, kInstOptNumEOps> op_counts{};
    std::array<uint32_t, 16> var_ids{};
    uint32_t num_unique_vars = 0;
    uint32_t total_nodes = 0;
    uint32_t total_ops = 0;
    uint32_t ops_64bit = 0;
    uint32_t const_parent_ops = 0;
    uint32_t cmp_select_ops = 0;
    EOp root_op = EOp::Var;

    void add_var_id(uint32_t id) noexcept {
        for (uint32_t i = 0; i < num_unique_vars; ++i) {
            if (var_ids[i] == id) return;
        }
        if (num_unique_vars < var_ids.size()) {
            var_ids[num_unique_vars++] = id;
        }
    }

    InstOptFeatures finalize(uint32_t crit_depth,
                             uint32_t high_lat_depth,
                             InstOptTarget target,
                             bool has_runtime_guards) const noexcept {
        InstOptFeatures f;
        const float denom_nodes = static_cast<float>((std::max)(1u, total_nodes));
        for (uint32_t i = 0; i < kInstOptNumEOps; ++i) {
            f.x[i] = static_cast<float>(op_counts[i]) / denom_nodes;
        }
        const uint32_t root_idx = static_cast<uint32_t>(root_op);
        if (root_idx < kInstOptNumEOps) {
            f.x[kInstOptNumEOps + root_idx] = 1.0f;
        }
        const float f_ops = static_cast<float>(total_ops);
        const float f_depth = static_cast<float>((std::max)(1u, crit_depth));
        const float denom_ops = static_cast<float>((std::max)(1u, total_ops));
        f.x[54] = (std::min)(1.0f, f_ops / 16.0f);
        f.x[55] = (std::min)(1.0f, static_cast<float>(crit_depth) / 16.0f);
        f.x[56] = (total_ops == 0) ? 0.0f : (std::min)(1.0f, (f_ops / f_depth) / 4.0f);
        f.x[57] = (std::min)(1.0f, static_cast<float>(num_unique_vars) / 16.0f);
        f.x[58] = (total_ops == 0) ? 0.0f : static_cast<float>(ops_64bit) / denom_ops;
        f.x[59] = (total_ops == 0) ? 0.0f : static_cast<float>(const_parent_ops) / denom_ops;
        f.x[60] = (std::min)(1.0f, static_cast<float>(high_lat_depth) / 8.0f);
        f.x[61] = (std::min)(1.0f, static_cast<float>(cmp_select_ops) / 8.0f);
        f.x[62] = has_runtime_guards ? 1.0f : 0.0f;
        f.x[63] = target_to_scalar(target);
        return f;
    }
};

struct DepthPair {
    uint32_t depth = 0;
    uint32_t high_lat_depth = 0;
};

DepthPair walk_extracted_expr(const ExtractedExpr& expr, FeatureAccum& acc, uint32_t guard_depth) noexcept {
    if (guard_depth > 32) return {0, 0};
    const uint32_t idx = static_cast<uint32_t>(expr.node.op);
    if (idx < kInstOptNumEOps) {
        acc.op_counts[idx]++;
    }
    acc.total_nodes++;

    if (expr.node.op == EOp::Var) {
        acc.add_var_id(static_cast<uint32_t>(expr.node.imm));
        return {0, 0};
    }
    if (expr.node.op == EOp::Const) {
        return {0, 0};
    }

    acc.total_ops++;
    if (expr.node.width == 64) acc.ops_64bit++;
    if (is_cmp_or_select_eop(expr.node.op)) acc.cmp_select_ops++;

    bool has_const_child = false;
    uint32_t max_child_depth = 0;
    uint32_t max_child_hl = 0;
    const uint32_t ar = eop_arity(expr.node.op);
    const size_t n_children = (std::min)(static_cast<size_t>(ar), expr.children.size());
    for (size_t i = 0; i < n_children; ++i) {
        if (expr.children[i].node.op == EOp::Const) has_const_child = true;
        const DepthPair ch = walk_extracted_expr(expr.children[i], acc, guard_depth + 1);
        max_child_depth = (std::max)(max_child_depth, ch.depth);
        max_child_hl = (std::max)(max_child_hl, ch.high_lat_depth);
    }
    if (has_const_child) acc.const_parent_ops++;

    const uint32_t hl_add = is_high_lat_eop(expr.node.op) ? 1u : 0u;
    return {max_child_depth + 1u, max_child_hl + hl_add};
}

bool value_is_const_mir(const Value* v) noexcept {
    if (!v || !v->is_instruction()) return false;
    const Instruction* def = v->defining_instruction();
    return def && (def->opcode() == Opcode::iconst_i32 || def->opcode() == Opcode::iconst_i64);
}

DepthPair walk_mir_value(const Value* v, const BasicBlock* bb, FeatureAccum& acc, uint32_t depth) noexcept {
    if (!v) return {0, 0};
    if (!v->is_instruction() || depth >= 8) {
        acc.op_counts[static_cast<uint32_t>(EOp::Var)]++;
        acc.total_nodes++;
        acc.add_var_id(v->id());
        return {0, 0};
    }
    const Instruction* inst = v->defining_instruction();
    if (!inst || inst->parent() != bb) {
        acc.op_counts[static_cast<uint32_t>(EOp::Var)]++;
        acc.total_nodes++;
        acc.add_var_id(v->id());
        return {0, 0};
    }
    const Opcode oc = inst->opcode();
    if (oc == Opcode::iconst_i32 || oc == Opcode::iconst_i64 ||
        oc == Opcode::patchable_const_i32 || oc == Opcode::patchable_const_i64) {
        acc.op_counts[static_cast<uint32_t>(EOp::Const)]++;
        acc.total_nodes++;
        return {0, 0};
    }
    EOp eop = EOp::Var;
    if (!map_mir_op_to_eop(inst, eop)) {
        acc.op_counts[static_cast<uint32_t>(EOp::Var)]++;
        acc.total_nodes++;
        acc.add_var_id(v->id());
        return {0, 0};
    }

    acc.op_counts[static_cast<uint32_t>(eop)]++;
    acc.total_nodes++;
    acc.total_ops++;
    if (inst->type() == Type::i64() || inst->type() == Type::ptr()) acc.ops_64bit++;
    if (is_cmp_or_select_eop(eop)) acc.cmp_select_ops++;

    bool has_const_child = false;
    uint32_t max_child_depth = 0;
    uint32_t max_child_hl = 0;
    const uint32_t ar = eop_arity(eop);
    const size_t n_ops = (std::min)(static_cast<size_t>(ar), inst->operand_count());
    for (size_t i = 0; i < n_ops; ++i) {
        const Value* ov = inst->operand(i);
        if (value_is_const_mir(ov)) has_const_child = true;
        const DepthPair ch = walk_mir_value(ov, bb, acc, depth + 1);
        max_child_depth = (std::max)(max_child_depth, ch.depth);
        max_child_hl = (std::max)(max_child_hl, ch.high_lat_depth);
    }
    if (has_const_child) acc.const_parent_ops++;
    const uint32_t hl_add = is_high_lat_eop(eop) ? 1u : 0u;
    return {max_child_depth + 1u, max_child_hl + hl_add};
}

template <size_t Rows, size_t Cols>
void quantize_rows_int8(const std::array<float, Rows * Cols>& w,
                        std::vector<int8_t>& out_q,
                        std::vector<float>& out_s) {
    out_q.resize(Rows * Cols);
    out_s.resize(Rows);
    for (size_t r = 0; r < Rows; ++r) {
        float max_abs = 0.0f;
        const float* row = w.data() + r * Cols;
        for (size_t c = 0; c < Cols; ++c) {
            max_abs = (std::max)(max_abs, std::fabs(row[c]));
        }
        const float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;
        out_s[r] = scale;
        const float inv_scale = 1.0f / scale;
        for (size_t c = 0; c < Cols; ++c) {
            const int q = static_cast<int>(std::lround(row[c] * inv_scale));
            out_q[r * Cols + c] = static_cast<int8_t>(std::clamp(q, -127, 127));
        }
    }
}

template <size_t Rows, size_t Cols>
void dequantize_rows_int8(const int8_t* q, const float* s, std::array<float, Rows * Cols>& out_w) noexcept {
    for (size_t r = 0; r < Rows; ++r) {
        const float scale = s[r];
        for (size_t c = 0; c < Cols; ++c) {
            out_w[r * Cols + c] = static_cast<float>(q[r * Cols + c]) * scale;
        }
    }
}

template <typename T>
void append_bytes(std::vector<uint8_t>& buf, const T* data, size_t count) {
    const size_t nbytes = sizeof(T) * count;
    const size_t off = buf.size();
    buf.resize(off + nbytes);
    if (nbytes > 0) {
        std::memcpy(buf.data() + off, data, nbytes);
    }
}

template <typename T>
bool read_bytes(const std::vector<uint8_t>& buf, size_t& offset, T* out, size_t count) noexcept {
    const size_t nbytes = sizeof(T) * count;
    if (offset + nbytes > buf.size()) return false;
    if (nbytes > 0) {
        std::memcpy(out, buf.data() + offset, nbytes);
    }
    offset += nbytes;
    return true;
}

float heuristic_fallback_cost(const InstOptFeatures& feat) noexcept {
    const float ops = feat.x[54] * 16.0f;
    const float depth = feat.x[55] * 16.0f;
    const float hl = feat.x[60] * 8.0f;
    return (std::max)(0.25f, 0.4f * ops + 0.6f * depth + 2.5f * hl);
}

std::string get_env_str(const char* name) {
#ifdef _MSC_VER
    char* raw = nullptr;
    size_t len = 0;
    if (_dupenv_s(&raw, &len, name) != 0 || !raw) return {};
    std::string out(raw);
    std::free(raw);
    return out;
#else
    const char* raw = std::getenv(name);
    return raw ? std::string(raw) : std::string();
#endif
}

void format_cpp_float(std::ostream& os, float v) {
    const double d = static_cast<double>(v);
    os << std::setprecision(8) << d;
    if (std::isfinite(d) && std::floor(d) == d && std::fabs(d) < 1e7) {
        std::ostringstream chk;
        chk << std::setprecision(8) << d;
        const std::string s = chk.str();
        if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
            os << ".0";
        }
    }
    os << "f";
}

template <typename T>
void write_cpp_array(std::ostream& os, const char* type_name, const char* name,
                     const T* data, size_t count, size_t per_line, bool is_float) {
    os << "static constexpr " << type_name << " " << name << "[" << count << "] = {\n";
    for (size_t i = 0; i < count; ++i) {
        if (i % per_line == 0) os << "    ";
        if (is_float) {
            format_cpp_float(os, static_cast<float>(data[i]));
        } else {
            os << static_cast<int>(data[i]);
        }
        if (i + 1 < count) os << ",";
        if ((i + 1) % per_line == 0 || i + 1 == count) os << "\n";
    }
    os << "};\n";
}

} // namespace

InstOptFeatures extract_inst_opt_features(
    const ExtractedExpr& expr,
    InstOptTarget target,
    bool has_runtime_guards) noexcept {
    FeatureAccum acc;
    acc.root_op = expr.node.op;
    const DepthPair dp = walk_extracted_expr(expr, acc, 0);
    return acc.finalize(dp.depth, dp.high_lat_depth, target, has_runtime_guards);
}

InstOptFeatures extract_inst_opt_features(
    const Instruction* inst,
    const BasicBlock* bb,
    InstOptTarget target,
    bool has_runtime_guards) noexcept {
    FeatureAccum acc;
    if (!inst || !inst->produces_value() || !inst->result()) {
        return acc.finalize(0, 0, target, has_runtime_guards);
    }
    EOp root_op = EOp::Var;
    if (map_mir_op_to_eop(inst, root_op)) {
        acc.root_op = root_op;
    }
    const DepthPair dp = walk_mir_value(inst->result(), bb, acc, 0);
    return acc.finalize(dp.depth, dp.high_lat_depth, target, has_runtime_guards);
}

float OnlineCostCalibrator::adjust(const InstOptPrediction& pred) const noexcept {
    float delta = bias;
    for (uint32_t i = 0; i < kInstOptHidden3; ++i) {
        delta += w[i] * pred.hidden_embed[i];
    }
    return delta;
}

void OnlineCostCalibrator::observe(const InstOptPrediction& pred,
                                   float measured_cost,
                                   float lr,
                                   float l2) noexcept {
    const float current = pred.predicted_cost + adjust(pred);
    const float err = current - measured_cost;
    const float clipped_err = std::clamp(err, -8.0f, 8.0f);
    for (uint32_t i = 0; i < kInstOptHidden3; ++i) {
        const float g = clipped_err * pred.hidden_embed[i] + l2 * w[i];
        w[i] = std::clamp(w[i] - lr * g, -4.0f, 4.0f);
    }
    bias = std::clamp(bias - lr * (clipped_err + l2 * bias), -8.0f, 8.0f);
    updates++;
}

const InstOptModel& InstOptModel::builtin() {
    static const InstOptModel instance = []() {
        InstOptModel m;
        const std::string env_path = get_env_str("BRASS_INST_OPT_MODEL");
        if (!env_path.empty() && InstOptModel::load_file(env_path, m)) {
            return m;
        }
        if (kBuiltinWeightsTrained) {
            dequantize_rows_int8<kInstOptHidden1, kInstOptInputDim>(kBuiltinW1Q, kBuiltinW1S, m.w1);
            std::memcpy(m.b1.data(), kBuiltinB1, sizeof(float) * kInstOptHidden1);
            dequantize_rows_int8<kInstOptHidden2, kInstOptHidden1>(kBuiltinW2Q, kBuiltinW2S, m.w2);
            std::memcpy(m.b2.data(), kBuiltinB2, sizeof(float) * kInstOptHidden2);
            dequantize_rows_int8<kInstOptHidden3, kInstOptHidden2>(kBuiltinW3Q, kBuiltinW3S, m.w3);
            std::memcpy(m.b3.data(), kBuiltinB3, sizeof(float) * kInstOptHidden3);
            dequantize_rows_int8<1, kInstOptHidden3>(kBuiltinWCostQ, kBuiltinWCostS, m.w_cost);
            m.b_cost = kBuiltinBCost;
            dequantize_rows_int8<kNumInstOptActions, kInstOptHidden3>(kBuiltinWPolQ, kBuiltinWPolS, m.w_pol);
            std::memcpy(m.b_pol.data(), kBuiltinBPol, sizeof(float) * kNumInstOptActions);
            m.loaded = true;
        }
        return m;
    }();
    return instance;
}

InstOptPrediction InstOptModel::predict(const InstOptFeatures& feat) const noexcept {
    InstOptPrediction out;
    if (!loaded) {
        out.predicted_cost = heuristic_fallback_cost(feat);
        const float u = 1.0f / static_cast<float>(kNumInstOptActions);
        out.action_probs.fill(u);
        return out;
    }

    std::array<float, kInstOptHidden1> h1{};
    for (uint32_t r = 0; r < kInstOptHidden1; ++r) {
        float acc = b1[r];
        const float* row = w1.data() + static_cast<size_t>(r) * kInstOptInputDim;
        for (uint32_t c = 0; c < kInstOptInputDim; ++c) {
            acc += row[c] * feat.x[c];
        }
        h1[r] = (acc > 0.0f) ? acc : 0.0f;
    }

    std::array<float, kInstOptHidden2> h2{};
    for (uint32_t r = 0; r < kInstOptHidden2; ++r) {
        float acc = b2[r];
        const float* row = w2.data() + static_cast<size_t>(r) * kInstOptHidden1;
        for (uint32_t c = 0; c < kInstOptHidden1; ++c) {
            acc += row[c] * h1[c];
        }
        h2[r] = ((acc > 0.0f) ? acc : 0.0f) + h1[r];
    }

    for (uint32_t r = 0; r < kInstOptHidden3; ++r) {
        float acc = b3[r];
        const float* row = w3.data() + static_cast<size_t>(r) * kInstOptHidden2;
        for (uint32_t c = 0; c < kInstOptHidden2; ++c) {
            acc += row[c] * h2[c];
        }
        out.hidden_embed[r] = (acc > 0.0f) ? acc : 0.0f;
    }

    float z_cost = b_cost;
    for (uint32_t c = 0; c < kInstOptHidden3; ++c) {
        z_cost += w_cost[c] * out.hidden_embed[c];
    }
    const float zc = std::clamp(z_cost, -20.0f, 20.0f);
    out.predicted_cost = (zc > 12.0f) ? zc : std::log1pf(std::exp(zc));

    std::array<float, kNumInstOptActions> logits{};
    float max_logit = -1e30f;
    for (uint32_t r = 0; r < kNumInstOptActions; ++r) {
        float acc = b_pol[r];
        const float* row = w_pol.data() + static_cast<size_t>(r) * kInstOptHidden3;
        for (uint32_t c = 0; c < kInstOptHidden3; ++c) {
            acc += row[c] * out.hidden_embed[c];
        }
        logits[r] = acc;
        max_logit = (std::max)(max_logit, acc);
    }

    float sum_exp = 0.0f;
    for (uint32_t r = 0; r < kNumInstOptActions; ++r) {
        const float e = std::exp(std::clamp(logits[r] - max_logit, -30.0f, 0.0f));
        out.action_probs[r] = e;
        sum_exp += e;
    }
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : (1.0f / static_cast<float>(kNumInstOptActions));
    for (uint32_t r = 0; r < kNumInstOptActions; ++r) {
        out.action_probs[r] *= inv_sum;
    }
    return out;
}

float InstOptModel::predict_cost(const InstOptFeatures& feat,
                                 const OnlineCostCalibrator* cal) const noexcept {
    const InstOptPrediction pred = predict(feat);
    const float raw = cal ? (pred.predicted_cost + cal->adjust(pred)) : pred.predicted_cost;
    return (std::max)(0.05f, raw);
}

std::array<InstOptAction, 4> InstOptModel::top_actions(const InstOptFeatures& feat) const noexcept {
    const InstOptPrediction pred = predict(feat);
    std::array<uint32_t, kNumInstOptActions> idx{};
    for (uint32_t i = 0; i < kNumInstOptActions; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 4, idx.end(), [&](uint32_t a, uint32_t b) {
        return pred.action_probs[a] > pred.action_probs[b];
    });
    return {
        static_cast<InstOptAction>(idx[0]),
        static_cast<InstOptAction>(idx[1]),
        static_cast<InstOptAction>(idx[2]),
        static_cast<InstOptAction>(idx[3])
    };
}

bool InstOptModel::save_file(const std::string& path, bool quantize_int8) const {
    std::vector<uint8_t> payload;
    if (quantize_int8) {
        std::vector<int8_t> q;
        std::vector<float> s;
        quantize_rows_int8<kInstOptHidden1, kInstOptInputDim>(w1, q, s);
        append_bytes(payload, q.data(), q.size());
        append_bytes(payload, s.data(), s.size());
        append_bytes(payload, b1.data(), b1.size());

        quantize_rows_int8<kInstOptHidden2, kInstOptHidden1>(w2, q, s);
        append_bytes(payload, q.data(), q.size());
        append_bytes(payload, s.data(), s.size());
        append_bytes(payload, b2.data(), b2.size());

        quantize_rows_int8<kInstOptHidden3, kInstOptHidden2>(w3, q, s);
        append_bytes(payload, q.data(), q.size());
        append_bytes(payload, s.data(), s.size());
        append_bytes(payload, b3.data(), b3.size());

        quantize_rows_int8<1, kInstOptHidden3>(w_cost, q, s);
        append_bytes(payload, q.data(), q.size());
        append_bytes(payload, s.data(), s.size());
        append_bytes(payload, &b_cost, 1);

        quantize_rows_int8<kNumInstOptActions, kInstOptHidden3>(w_pol, q, s);
        append_bytes(payload, q.data(), q.size());
        append_bytes(payload, s.data(), s.size());
        append_bytes(payload, b_pol.data(), b_pol.size());
    } else {
        append_bytes(payload, w1.data(), w1.size());
        append_bytes(payload, b1.data(), b1.size());
        append_bytes(payload, w2.data(), w2.size());
        append_bytes(payload, b2.data(), b2.size());
        append_bytes(payload, w3.data(), w3.size());
        append_bytes(payload, b3.data(), b3.size());
        append_bytes(payload, w_cost.data(), w_cost.size());
        append_bytes(payload, &b_cost, 1);
        append_bytes(payload, w_pol.data(), w_pol.size());
        append_bytes(payload, b_pol.data(), b_pol.size());
    }

    const uint32_t header[9] = {
        kBrimMagic,
        kBrimVersion,
        kInstOptInputDim,
        kInstOptHidden1,
        kInstOptHidden2,
        kInstOptHidden3,
        kNumInstOptActions,
        quantize_int8 ? kBrimFlagInt8 : kBrimFlagFp32,
        crc32_bytes(payload.data(), payload.size())
    };

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(header), sizeof(header));
    ofs.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(ofs);
}

bool InstOptModel::load_file(const std::string& path, InstOptModel& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    uint32_t header[9] = {};
    ifs.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!ifs) return false;
    if (header[0] != kBrimMagic || header[1] != kBrimVersion ||
        header[2] != kInstOptInputDim || header[3] != kInstOptHidden1 ||
        header[4] != kInstOptHidden2 || header[5] != kInstOptHidden3 ||
        header[6] != kNumInstOptActions) {
        return false;
    }
    const uint32_t flags = header[7];
    if (flags != kBrimFlagFp32 && flags != kBrimFlagInt8) return false;

    std::vector<uint8_t> payload((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (crc32_bytes(payload.data(), payload.size()) != header[8]) return false;

    InstOptModel tmp;
    size_t off = 0;
    if (flags == kBrimFlagInt8) {
        std::vector<int8_t> q;
        std::vector<float> s;
        auto load_q = [&](auto& dst_w, size_t rows, size_t cols) -> bool {
            q.resize(rows * cols);
            s.resize(rows);
            if (!read_bytes(payload, off, q.data(), q.size())) return false;
            if (!read_bytes(payload, off, s.data(), s.size())) return false;
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                    dst_w[r * cols + c] = static_cast<float>(q[r * cols + c]) * s[r];
                }
            }
            return true;
        };
        if (!load_q(tmp.w1, kInstOptHidden1, kInstOptInputDim)) return false;
        if (!read_bytes(payload, off, tmp.b1.data(), tmp.b1.size())) return false;
        if (!load_q(tmp.w2, kInstOptHidden2, kInstOptHidden1)) return false;
        if (!read_bytes(payload, off, tmp.b2.data(), tmp.b2.size())) return false;
        if (!load_q(tmp.w3, kInstOptHidden3, kInstOptHidden2)) return false;
        if (!read_bytes(payload, off, tmp.b3.data(), tmp.b3.size())) return false;
        if (!load_q(tmp.w_cost, 1, kInstOptHidden3)) return false;
        if (!read_bytes(payload, off, &tmp.b_cost, 1)) return false;
        if (!load_q(tmp.w_pol, kNumInstOptActions, kInstOptHidden3)) return false;
        if (!read_bytes(payload, off, tmp.b_pol.data(), tmp.b_pol.size())) return false;
    } else {
        if (!read_bytes(payload, off, tmp.w1.data(), tmp.w1.size())) return false;
        if (!read_bytes(payload, off, tmp.b1.data(), tmp.b1.size())) return false;
        if (!read_bytes(payload, off, tmp.w2.data(), tmp.w2.size())) return false;
        if (!read_bytes(payload, off, tmp.b2.data(), tmp.b2.size())) return false;
        if (!read_bytes(payload, off, tmp.w3.data(), tmp.w3.size())) return false;
        if (!read_bytes(payload, off, tmp.b3.data(), tmp.b3.size())) return false;
        if (!read_bytes(payload, off, tmp.w_cost.data(), tmp.w_cost.size())) return false;
        if (!read_bytes(payload, off, &tmp.b_cost, 1)) return false;
        if (!read_bytes(payload, off, tmp.w_pol.data(), tmp.w_pol.size())) return false;
        if (!read_bytes(payload, off, tmp.b_pol.data(), tmp.b_pol.size())) return false;
    }
    if (off != payload.size()) return false;
    tmp.loaded = true;
    out = tmp;
    return true;
}

bool InstOptModel::export_cpp_inc(const std::string& path) const {
    std::ostringstream os;
    os << "// Auto-generated by brotensor_train_inst_opt_model — DO NOT EDIT BY HAND.\n";
    os << "// Per-row symmetric INT8 quantized weights + FP32 row scales + FP32 biases.\n";
    std::vector<int8_t> q;
    std::vector<float> s;

    quantize_rows_int8<kInstOptHidden1, kInstOptInputDim>(w1, q, s);
    write_cpp_array(os, "int8_t", "kBuiltinW1Q", q.data(), q.size(), 48, false);
    write_cpp_array(os, "float ", "kBuiltinW1S", s.data(), s.size(), 16, true);
    write_cpp_array(os, "float ", "kBuiltinB1",  b1.data(), b1.size(), 16, true);

    quantize_rows_int8<kInstOptHidden2, kInstOptHidden1>(w2, q, s);
    write_cpp_array(os, "int8_t", "kBuiltinW2Q", q.data(), q.size(), 48, false);
    write_cpp_array(os, "float ", "kBuiltinW2S", s.data(), s.size(), 16, true);
    write_cpp_array(os, "float ", "kBuiltinB2",  b2.data(), b2.size(), 16, true);

    quantize_rows_int8<kInstOptHidden3, kInstOptHidden2>(w3, q, s);
    write_cpp_array(os, "int8_t", "kBuiltinW3Q", q.data(), q.size(), 48, false);
    write_cpp_array(os, "float ", "kBuiltinW3S", s.data(), s.size(), 16, true);
    write_cpp_array(os, "float ", "kBuiltinB3",  b3.data(), b3.size(), 16, true);

    quantize_rows_int8<1, kInstOptHidden3>(w_cost, q, s);
    write_cpp_array(os, "int8_t", "kBuiltinWCostQ", q.data(), q.size(), 48, false);
    write_cpp_array(os, "float ", "kBuiltinWCostS", s.data(), s.size(), 8, true);
    os << "static constexpr float  kBuiltinBCost = ";
    format_cpp_float(os, b_cost);
    os << ";\n";

    quantize_rows_int8<kNumInstOptActions, kInstOptHidden3>(w_pol, q, s);
    write_cpp_array(os, "int8_t", "kBuiltinWPolQ", q.data(), q.size(), 48, false);
    write_cpp_array(os, "float ", "kBuiltinWPolS", s.data(), s.size(), 16, true);
    write_cpp_array(os, "float ", "kBuiltinBPol",  b_pol.data(), b_pol.size(), 16, true);
    os << "static constexpr bool   kBuiltinWeightsTrained = " << (loaded ? "true" : "false") << ";\n";

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    const std::string text = os.str();
    ofs.write(text.data(), static_cast<std::streamsize>(text.size()));
    return static_cast<bool>(ofs);
}

} // namespace brass
