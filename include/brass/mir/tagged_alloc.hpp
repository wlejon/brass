#pragma once

// Objects from the module's tagged allocator (Module::tagged_allocator), as
// the optimizer sees them.
//
// A call `%o = call.tagged @alloc(%bytes, %header)` makes a fresh object and
// answers it NaN-boxed: its payload address ORed with the allocator's tag.
// The program then reaches the object only through values computed from %o
// by the tag arithmetic a NaN-boxing front end uses, which this analysis
// follows; each such value is *derived* from the call, in one of two forms:
//
//   Box      the object's bits, `address | tag` (as a tagged or i64 value):
//            %o, its bitcasts either way, `or`/`and` that keep the tag;
//   Address  its payload address plus a constant byte delta: a Box under
//            `and` with a mask that keeps the 48 address bits and drops the
//            tag, or `xor` with the tag, then `add`/`sub` of constants (an
//            Address of delta 0 under `or`/`xor` with the tag is a Box again).
//
// A block parameter is derived when every edge into it passes a value
// derived from the same call in the same form and delta. Such a parameter
// holds the call's most recent object: SSA dominance means any path from a
// later execution of the call to a use of the parameter passes the
// parameter's block again, which re-binds it.
//
// What the object's derived values may do without the object escaping:
//
//   - be the address of a `load` or `store` (a field access at the delta
//     plus the instruction's offset); the value stored is no derived value;
//   - be a `write_barrier`'s target (never its value);
//   - be kept alive by `keep_alive`;
//   - be tested in ways answered without the address (a *fold*): the tag
//     read off a Box (`and` with a mask of the tag bits, `lshr`/`ashr` by at
//     least 48), a Box compared unsigned or signed with a constant on one
//     side of every possible `address | tag`, and a Box compared for
//     identity: equal to a Box of the same call (the same object, above),
//     unequal to anything else (a derived value of another call is another
//     object; a value derived from none cannot be this object, whose
//     address nothing outside its derived values holds while it does not
//     escape; a constant with another tag never matches);
//   - flow into derived block parameters.
//
// Anything else makes it escape: being stored (into any memory, its own
// fields included), passed to a call or invoke, returned, thrown, held in
// deopt state (a guard's or any instruction's state map, which a lower tier
// would need the real object for), passed on an edge to a parameter that is
// not derived (or to a block entered from outside the CFG: the entry, a
// landing pad, a resume target), selected, compared as an Address, or used
// by any other instruction. Being live across a GC point is not an escape:
// a collection cannot observe an object nothing it scans refers to, and an
// object scalar replacement removes keeps its fields' words in tagged
// values, which are roots (sroa_tagged.cpp).
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>

#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brass {

// True when `inst` calls its module's tagged allocator.
bool is_tagged_allocation(const Instruction* inst) noexcept;

// True when an edge may enter `bb` with values no terminator names: the
// entry, a landing pad, a resume target.
bool entered_from_outside(const Function& fn, const BasicBlock* bb);

// The constant payload size of such a call, or -1.
int64_t tagged_allocation_bytes(const Instruction* inst) noexcept;

enum class TaggedForm : uint8_t { Box, Address };

struct TaggedDerivation {
    uint32_t site = 0;
    TaggedForm form = TaggedForm::Box;
    int64_t delta = 0;  // Address only
};

enum class TaggedUseKind : uint8_t {
    Derive,     // computes another derived value
    Load,       // a field read
    Store,      // a field write
    Barrier,    // write_barrier with the object as its target
    KeepAlive,
    Fold,       // answered without the object: `folded`
};

struct TaggedUse {
    Instruction* inst = nullptr;
    TaggedUseKind kind = TaggedUseKind::Derive;
    int64_t offset = 0;  // Load / Store: the byte offset into the payload
    int64_t folded = 0;  // Fold: the instruction's result
};

struct TaggedAllocSite {
    Instruction* call = nullptr;
    int64_t bytes = -1;  // constant payload size, or -1
    // Every value derived from the call, the call's result and derived
    // block parameters included.
    std::vector<const Value*> values;
    // Every use of them, each instruction once (a fold of two derived
    // operands, a load whose address is derived, ...).
    std::vector<TaggedUse> uses;
    // Why the object escapes, or empty when it does not; and where.
    std::string_view escape;
    const Instruction* escape_at = nullptr;

    bool escapes() const noexcept { return !escape.empty(); }
};

class TaggedAllocations {
public:
    explicit TaggedAllocations(const Function& fn);

    const std::vector<TaggedAllocSite>& sites() const noexcept { return sites_; }
    // How `v` is derived from a site, or null.
    const TaggedDerivation* derivation(const Value* v) const noexcept;
    // Non-escaping sites.
    size_t non_escaping() const noexcept;

private:
    std::vector<TaggedAllocSite> sites_;
    std::unordered_map<const Value*, TaggedDerivation> derived_;
};

} // namespace brass
