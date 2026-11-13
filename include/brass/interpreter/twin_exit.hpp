#pragma once

// A guard exit into a resume twin (Function::is_resume_twin), as an
// interpreter takes it: the state values as the 8-byte words native code
// passes, and the twin's two arguments, (i32 resume id, ptr words).

#include <brass/interpreter/value.hpp>
#include <cstdint>
#include <vector>

namespace brass {

struct TwinExitArgs {
    std::vector<uint64_t> words;
    std::vector<RuntimeValue> args;

    TwinExitArgs(uint32_t resume_id, const std::vector<RuntimeValue>& state) {
        // One word at least, so the pointer is never null.
        words.reserve(state.size() + 1);
        for (const RuntimeValue& v : state) words.push_back(v.raw_bits());
        words.push_back(0);
        args.push_back(RuntimeValue::from_u32(resume_id));
        args.push_back(RuntimeValue::from_ptr(static_cast<const void*>(words.data())));
    }
};

} // namespace brass
