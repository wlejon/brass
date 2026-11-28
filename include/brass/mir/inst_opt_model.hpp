#pragma once

// Zero-dependency neural instruction-level cost & rewrite policy model
// (`InstOptModel`) for Brass MIR / EGraph / Superopt.
//
// Architecture:
//   Input (64 floats): normalized EOp histogram (27), root EOp one-hot (27),
//                      and structural / microarchitectural features (10).
//   Trunk: 64 -> 96 (ReLU) -> 96 (residual ReLU) -> 48 (ReLU).
//   Cost head: scalar cycle cost via softplus(dot(w_cost, h3) + b_cost).
//   Policy head: 16 rewrite-family logits -> softmax probabilities.
//   Online calibration: linear residual adapter over `hidden_embed` (`OnlineCostCalibrator`).

#include <brass/mir/egraph.hpp>
#include <array>
#include <cstdint>
#include <string>

namespace brass {

class BasicBlock;
class Instruction;

inline constexpr uint32_t kInstOptNumEOps = 27;
inline constexpr uint32_t kInstOptInputDim = 64;
inline constexpr uint32_t kInstOptHidden1 = 96;
inline constexpr uint32_t kInstOptHidden2 = 96;
inline constexpr uint32_t kInstOptHidden3 = 48;
inline constexpr uint32_t kNumInstOptActions = 16;

enum class InstOptTarget : uint8_t {
    X64 = 0,
    ARM64 = 1,
    GPU_PTX = 2
};

enum class InstOptAction : uint8_t {
    NoOp = 0,
    AlgebraicIdentity = 1,
    BitwiseDeMorganDist = 2,
    BitwiseCarryAdd = 3,
    MulStrengthReduce = 4,
    DivModReduce = 5,
    TwosComplementFold = 6,
    CmpSelectCanon = 7,
    GuardMaskElide = 8,
    GuardExtTruncNarrow = 9,
    AssocConstCombine = 10,
    MulAddFactor = 11,
    TreeHeightBalance = 12,
    ShiftMaskFuse = 13,
    RuntimeSpeculateGuard = 14,
    LirPeephole = 15
};

inline constexpr const char* inst_opt_action_name(InstOptAction action) noexcept {
    switch (action) {
        case InstOptAction::NoOp:                  return "NoOp";
        case InstOptAction::AlgebraicIdentity:     return "AlgebraicIdentity";
        case InstOptAction::BitwiseDeMorganDist:   return "BitwiseDeMorganDist";
        case InstOptAction::BitwiseCarryAdd:       return "BitwiseCarryAdd";
        case InstOptAction::MulStrengthReduce:     return "MulStrengthReduce";
        case InstOptAction::DivModReduce:          return "DivModReduce";
        case InstOptAction::TwosComplementFold:    return "TwosComplementFold";
        case InstOptAction::CmpSelectCanon:        return "CmpSelectCanon";
        case InstOptAction::GuardMaskElide:        return "GuardMaskElide";
        case InstOptAction::GuardExtTruncNarrow:   return "GuardExtTruncNarrow";
        case InstOptAction::AssocConstCombine:     return "AssocConstCombine";
        case InstOptAction::MulAddFactor:          return "MulAddFactor";
        case InstOptAction::TreeHeightBalance:     return "TreeHeightBalance";
        case InstOptAction::ShiftMaskFuse:         return "ShiftMaskFuse";
        case InstOptAction::RuntimeSpeculateGuard: return "RuntimeSpeculateGuard";
        case InstOptAction::LirPeephole:           return "LirPeephole";
    }
    return "Unknown";
}

struct InstOptFeatures {
    std::array<float, kInstOptInputDim> x{};
};

struct InstOptPrediction {
    float predicted_cost = 0.0f;
    std::array<float, kNumInstOptActions> action_probs{};
    std::array<float, kInstOptHidden3> hidden_embed{};
};

/// Extract normalized architectural & DAG features from an extracted EGraph expression.
InstOptFeatures extract_inst_opt_features(
    const ExtractedExpr& expr,
    InstOptTarget target = InstOptTarget::X64,
    bool has_runtime_guards = false) noexcept;

/// Extract normalized architectural & DAG features from a MIR instruction window inside `bb`.
InstOptFeatures extract_inst_opt_features(
    const Instruction* inst,
    const BasicBlock* bb,
    InstOptTarget target = InstOptTarget::X64,
    bool has_runtime_guards = false) noexcept;

/// Lightweight online linear cost calibrator trained incrementally on actual host timings.
class OnlineCostCalibrator {
public:
    std::array<float, kInstOptHidden3> w{};
    float bias = 0.0f;
    uint32_t updates = 0;

    float adjust(const InstOptPrediction& pred) const noexcept;
    void observe(const InstOptPrediction& pred,
                 float measured_cost,
                 float lr = 0.05f,
                 float l2 = 1e-3f) noexcept;
};

/// Embedded/loadable MLP+residual instruction-level cost & rewrite policy model.
class InstOptModel {
public:
    std::array<float, kInstOptHidden1 * kInstOptInputDim> w1{};
    std::array<float, kInstOptHidden1> b1{};
    std::array<float, kInstOptHidden2 * kInstOptHidden1> w2{};
    std::array<float, kInstOptHidden2> b2{};
    std::array<float, kInstOptHidden3 * kInstOptHidden2> w3{};
    std::array<float, kInstOptHidden3> b3{};
    std::array<float, kInstOptHidden3> w_cost{};
    float b_cost = 0.0f;
    std::array<float, kNumInstOptActions * kInstOptHidden3> w_pol{};
    std::array<float, kNumInstOptActions> b_pol{};
    bool loaded = false;

    /// Returns the process-wide builtin model (honoring `BRASS_INST_OPT_MODEL` if set).
    static const InstOptModel& builtin();

    /// Load a `.brim` binary weight file (supports FP32 and per-row symmetric INT8).
    static bool load_file(const std::string& path, InstOptModel& out);

    /// Save weights to a `.brim` binary weight file.
    bool save_file(const std::string& path, bool quantize_int8 = true) const;

    /// Export compact INT8 quantized C++ initializer data for `inst_opt_weights_builtin.inc`.
    bool export_cpp_inc(const std::string& path) const;

    /// Evaluate cost and 16-class rewrite action probabilities.
    InstOptPrediction predict(const InstOptFeatures& feat) const noexcept;

    /// Convenience scalar cost evaluator (optionally applying `OnlineCostCalibrator`).
    float predict_cost(const InstOptFeatures& feat,
                       const OnlineCostCalibrator* cal = nullptr) const noexcept;

    /// Returns the top 4 predicted rewrite actions in descending probability order.
    std::array<InstOptAction, 4> top_actions(const InstOptFeatures& feat) const noexcept;
};

} // namespace brass
