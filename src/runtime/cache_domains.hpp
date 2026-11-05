#pragma once

// Where the compile pool's workers run relative to the program they compile
// for (compile_pool.cpp).
//
// A tier-2 compile walks tens of MB of compiler state for 5 to 200 ms. On a
// worker that shares the mutator's last-level cache it evicts the program's
// working set: on a two-CCD Ryzen (two 32 MB L3s) three.js's three_meshes
// went from ~90 demand fills from DRAM a frame to ~65,000 while six compiles
// ran, its frames 10.7 -> 17 ms with the same instructions retired, and one
// long compile alone cost 1-3 ms a frame. Linux keeps a process's threads in
// one LLC domain when it can (wake-affine placement), so the workers landed
// beside the mutator every time while the other CCD sat idle.
//
// So a worker moves itself, per job, onto the CPUs that share no last-level
// cache with the CPU the job's mutator was on when it asked. On a machine with
// one last-level cache (or where the topology cannot be read) nothing moves.

namespace brass::runtime::detail {

// The CPU the calling thread runs on now; -1 when the platform does not say.
int current_cpu() noexcept;

// Restricts the calling thread to the CPUs that share no last-level cache
// with `cpu`, among those the process may run on; with `cpu` < 0, no other
// domain to go to, or no topology, it may run on all of them again. Cheap
// when the thread is already where the call would put it.
void place_away_from(int cpu) noexcept;

} // namespace brass::runtime::detail
