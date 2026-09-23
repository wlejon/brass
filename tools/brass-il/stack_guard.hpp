#pragma once

// Windows: turns a native stack overflow on the calling thread into an error
// message on stderr and exit code 3. Elsewhere a no-op.
void brass_il_install_stack_overflow_guard();
