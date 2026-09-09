#pragma once

#include <brass/mir/slp_vectorize.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <vector>
#include <unordered_set>

namespace brass {

struct SlpStoreBundle {
    std::vector<Instruction*> stores;
    Value* base = nullptr;
    int32_t base_offset = 0;
    Type elem_type;
    Type vec_type;
    uint32_t width = 0;
    std::vector<Value*> stored_values;
};

struct SlpLoadBundle {
    std::vector<Instruction*> loads;
    Value* base = nullptr;
    int32_t base_offset = 0;
    Type elem_type;
    Type vec_type;
    uint32_t width = 0;
    std::vector<Value*> loaded_values;
};

struct SlpArithBundle {
    std::vector<Instruction*> insts;
    Opcode op = Opcode::unreachable;
    Type elem_type;
    Type vec_type;
    uint32_t width = 0;
};

bool get_slp_type_info(
    Type elem_type,
    const SlpOptions& options,
    uint32_t& out_width,
    int32_t& out_stride,
    Type& out_vec_type
);

std::vector<SlpStoreBundle> find_slp_store_bundles(
    BasicBlock& bb,
    const SlpOptions& options
);

std::vector<SlpLoadBundle> find_slp_load_bundles(
    BasicBlock& bb,
    const SlpOptions& options
);

std::vector<SlpArithBundle> find_slp_arith_bundles(
    BasicBlock& bb,
    const SlpOptions& options,
    const std::unordered_set<Instruction*>& ignored_insts
);

} // namespace brass