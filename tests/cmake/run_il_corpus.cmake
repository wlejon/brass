# Runs brass-il on one bronze corpus program and compares its stdout with the
# program's .expected file. brass-il's own "[brass-il] ..." status lines are
# not program output and are dropped before the comparison.
#
# Inputs (-D):
#   BRASS_IL   the brass-il binary
#   IL_FILE    the .il program
#   EXPECTED   its .expected file
#   IL_ARGS    brass-il flags, comma-separated (e.g. --run,--enable-background-compile)

string(REPLACE "," ";" IL_ARGS "${IL_ARGS}")
execute_process(
    COMMAND "${BRASS_IL}" "${IL_FILE}" ${IL_ARGS}
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "brass-il ${IL_ARGS} ${IL_FILE} exited with ${_rc}\nstdout:\n${_out}\nstderr:\n${_err}")
endif()

file(READ "${EXPECTED}" _expected)
string(REPLACE "\r" "" _expected "${_expected}")
string(REPLACE "\r" "" _out "${_out}")
string(REGEX REPLACE "(^|\n)\\[brass-il\\][^\n]*" "" _out "${_out}")
string(STRIP "${_expected}" _expected)
string(STRIP "${_out}" _out)

if(NOT _out STREQUAL _expected)
    message(FATAL_ERROR "brass-il ${IL_ARGS} ${IL_FILE}: output differs\n--- expected\n${_expected}\n--- actual\n${_out}\nstderr:\n${_err}")
endif()
