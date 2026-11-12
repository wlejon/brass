#pragma once

#include <brass/mir/function.hpp>
#include <cstddef>

namespace brass {

// One untagging of a reference serves every use no GC point separates from
// it, joins included. A NaN-boxing front end reads a reference's address as
// `xor(bitcast_i64_tagged x, K)` and, by the contract of gc_refs.hpp, reads
// it again after every call. GVN merges the reads a first one dominates with
// no GC point between (gvn_gc_free.hpp). What it leaves is the read after a
// join one of whose arms called: a fast path that passes a slow path's
// rejoin reads the address again though it holds it, at every such join.
//
// This pass carries the address through the join as a block parameter: an
// arm that still holds it passes it, an arm that does not (it called, or
// never read it) computes it before its branch, and the read after the join
// is the parameter. On the arm that held it nothing is computed; on the
// other the read moved from after the join to before it.
//
// What a block holds is found forwards, in reverse postorder: a GC point
// (may_trigger_gc) empties it, a block with one predecessor starts with what
// that one ended with, a join with what at least one predecessor ended with
// and the others can compute (the tagged value is defined above them, and
// each ends in a plain `br`). A parameter is made only when a read asks for
// it. A loop header, a deoptimization's resume target and a landing pad
// start with nothing.
//
// The address stays an i64 no stack map lists, as it was, and is live only
// where no GC point can run: every path from a definition to a use of it is
// one this analysis walked without meeting one. Critical edges are split
// first. Returns how many reads were replaced.
size_t reuse_untagged_addresses(Function& fn);

} // namespace brass
