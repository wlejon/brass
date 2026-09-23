# Runs the AOT exception host (tests/aot/eh_aot_host.c linked with the COFF
# object brass-opt -c made from tests/aot/eh_aot.mir, and brass.lib) and
# checks every "<function> <arg> <result>" line it prints against brass-opt
# running the same function in the interpreter and under --jit.
#
# Inputs (-D):
#   HOST_EXE   the linked host
#   BRASS_OPT  the brass-opt binary
#   MIR        tests/aot/eh_aot.mir

execute_process(
    COMMAND "${HOST_EXE}"
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "${HOST_EXE} exited with ${_rc}\nstdout:\n${_out}\nstderr:\n${_err}")
endif()
string(REPLACE "\r" "" _out "${_out}")
string(STRIP "${_out}" _out)
string(REPLACE "\n" ";" _lines "${_out}")

list(POP_BACK _lines _last)
if(NOT _last STREQUAL "done")
    message(FATAL_ERROR "${HOST_EXE} stopped early; last line: '${_last}'\nstdout:\n${_out}\nstderr:\n${_err}")
endif()
list(LENGTH _lines _count)
if(NOT _count EQUAL 32)
    message(FATAL_ERROR "expected 32 result lines from ${HOST_EXE}, got ${_count}\n${_out}")
endif()

set(_failures "")
foreach(_line IN LISTS _lines)
    if(NOT _line MATCHES "^([A-Za-z_]+) (-?[0-9]+) (-?[0-9]+)$")
        message(FATAL_ERROR "malformed host line '${_line}'")
    endif()
    set(_fn "${CMAKE_MATCH_1}")
    set(_arg "${CMAKE_MATCH_2}")
    set(_aot "${CMAKE_MATCH_3}")
    foreach(_mode interp jit)
        set(_flags "")
        if(_mode STREQUAL "jit")
            set(_flags "--jit")
        endif()
        execute_process(
            COMMAND "${BRASS_OPT}" "${MIR}" -r "${_fn}" --args "${_arg}" ${_flags}
            OUTPUT_VARIABLE _ref
            ERROR_VARIABLE _ref_err
            RESULT_VARIABLE _ref_rc
        )
        if(NOT _ref_rc EQUAL 0)
            message(FATAL_ERROR "brass-opt -r ${_fn} --args ${_arg} ${_flags} exited with ${_ref_rc}\n${_ref}\n${_ref_err}")
        endif()
        string(REPLACE "\r" "" _ref "${_ref}")
        string(STRIP "${_ref}" _ref)
        if(NOT _ref STREQUAL _aot)
            string(APPEND _failures "  ${_fn}(${_arg}): aot ${_aot}, ${_mode} ${_ref}\n")
        endif()
    endforeach()
endforeach()

if(_failures)
    message(FATAL_ERROR "AOT results differ:\n${_failures}")
endif()
