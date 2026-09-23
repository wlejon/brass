# Runs a command that must fail cleanly: a nonzero exit code that is a
# plain status (not a crash, which a Windows NTSTATUS such as 0xC0000094 or
# a POSIX signal would show as) and stderr matching a regular expression.
#
# Inputs (-D):
#   CMD        the command and its arguments, comma-separated
#   EXIT_CODE  the exit code expected
#   STDERR_RE  a regular expression stderr must match

string(REPLACE "," ";" CMD "${CMD}")
execute_process(
    COMMAND ${CMD}
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
)
if(NOT _rc STREQUAL "${EXIT_CODE}")
    message(FATAL_ERROR "${CMD}: exit code '${_rc}', expected ${EXIT_CODE}\nstdout:\n${_out}\nstderr:\n${_err}")
endif()
if(NOT _err MATCHES "${STDERR_RE}")
    message(FATAL_ERROR "${CMD}: stderr does not match '${STDERR_RE}'\nstdout:\n${_out}\nstderr:\n${_err}")
endif()
