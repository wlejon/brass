#pragma once

#include <brass/mir/function.hpp>
#include <cstddef>

namespace brass {

// A word read only to be seen as a double is read as one. A NaN-boxing front
// end loads a slot's bits (`load.i64`) and, where the slot is known to hold a
// Number, reads them as the double (`bitcast.f64.i64`): on x64 a load into a
// general register and a `movq` across to an xmm, two instructions where
// `movsd xmm, [m]` is one. Where every use of the load is such a bitcast, the
// load becomes a `load.f64` in place and stands for them.
//
// Last in the pipeline: GVN and the store-to-load forwarding match loads and
// stores by the type they move, and a slot is stored as bits.
//
// Returns how many loads it retyped.
size_t read_doubles_as_doubles(Function& fn);

}  // namespace brass
