# Prints one bronze corpus program as MIR text (brass-il --emit-mir), then
# checks that the text is a complete, standalone program:
#   1. brass-opt --check-roundtrip parses it and prints it back byte-identically
#      (quoted `@"..."` names included);
#   2. brass-opt runs `main` from the text alone under --baseline-jit and
#      --jit, and the output matches the program's .expected file (no runtime
#      state from the translating process, such as property-name pointers).
#
# Inputs (-D):
#   BRASS_IL   the brass-il binary
#   BRASS_OPT  the brass-opt binary
#   IL_FILE    the .il program
#   EXPECTED   its .expected file
#   WORK_DIR   scratch directory for the .mir and .obj files

get_filename_component(_name "${IL_FILE}" NAME_WE)
file(MAKE_DIRECTORY "${WORK_DIR}")
set(_mir "${WORK_DIR}/${_name}.mir")

# -o suppresses brass-il's own run of the program; the MIR goes to stdout.
execute_process(
    COMMAND "${BRASS_IL}" "${IL_FILE}" --emit-mir -o "${WORK_DIR}/${_name}.obj"
    OUTPUT_VARIABLE _text
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "brass-il --emit-mir ${IL_FILE} exited with ${_rc}\n${_err}")
endif()
string(REPLACE "\r" "" _text "${_text}")
string(REGEX REPLACE "(^|\n)\\[brass-il\\][^\n]*" "" _text "${_text}")
file(WRITE "${_mir}" "${_text}")

execute_process(
    COMMAND "${BRASS_OPT}" "${_mir}" --check-roundtrip
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "brass-opt --check-roundtrip ${_mir} exited with ${_rc}\n${_out}\n${_err}")
endif()

file(READ "${EXPECTED}" _expected)
string(REPLACE "\r" "" _expected "${_expected}")
string(STRIP "${_expected}" _expected)

foreach(_mode --baseline-jit --jit)
    execute_process(
        COMMAND "${BRASS_OPT}" "${_mir}" -r main ${_mode}
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "brass-opt ${_mir} -r main ${_mode} exited with ${_rc}\nstdout:\n${_out}\nstderr:\n${_err}")
    endif()
    string(REPLACE "\r" "" _out "${_out}")
    string(STRIP "${_out}" _out)
    if(NOT _out STREQUAL _expected)
        message(FATAL_ERROR "brass-opt ${_mir} -r main ${_mode}: output differs\n--- expected\n${_expected}\n--- actual\n${_out}\nstderr:\n${_err}")
    endif()
endforeach()
