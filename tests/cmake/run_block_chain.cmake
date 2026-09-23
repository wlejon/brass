# Generates a MIR function with many blocks, runs it with brass-opt and
# checks the result. Analyses that recursed once per block (or once per
# dominator-tree or loop-nest level) overflowed the native stack on such
# functions; the input is generated here rather than checked in.
#
# Inputs (-D):
#   BRASS_OPT  path to brass-opt
#   OUT_DIR    directory for the generated .mir file
#   SHAPE      reversed  a chain bb0 -> bbN -> ... -> bb1, blocks written bb1..bbN,
#                        so every use is a forward reference in the text
#              ordered   the same CFG, blocks written in execution order
#              ifs       N nested if/else diamonds (dominator tree ~2N deep)
#              loops     N nested loops, each running once
#   N          the number of blocks / nesting levels
#   FLAGS      extra brass-opt flags, comma-separated (e.g. "--jit")

if(NOT DEFINED BRASS_OPT OR NOT DEFINED OUT_DIR OR NOT DEFINED SHAPE OR NOT DEFINED N)
    message(FATAL_ERROR "run_block_chain.cmake: BRASS_OPT, OUT_DIR, SHAPE and N are required")
endif()

# Appends each argument as a line of _text. string(APPEND) keeps generation
# linear; list(APPEND) on a list this long is quadratic.
set(_text "")
macro(emit)
    foreach(_line IN ITEMS ${ARGN})
        string(APPEND _text "${_line}\n")
    endforeach()
endmacro()

set(_arg 3)
emit("func @chain(%0: i64) -> i64 {" "bb0:")
if(SHAPE STREQUAL "reversed" OR SHAPE STREQUAL "ordered")
    # v_N = x + x, v_k = v_{k+1} + x: the result is (N + 1) * x.
    emit("  br bb${N}")
    foreach(_i RANGE 1 ${N})
        set(_k ${_i})
        if(SHAPE STREQUAL "ordered")
            math(EXPR _k "${N} + 1 - ${_i}")
        endif()
        math(EXPR _next "${_k} + 1")
        math(EXPR _prev "${_k} - 1")
        emit("bb${_k}:")
        if(_k EQUAL N)
            emit("  %v${_k} = add.i64 %0, %0")
        else()
            emit("  %v${_k} = add.i64 %v${_next}, %0")
        endif()
        if(_k EQUAL 1)
            emit("  ret %v1")
        else()
            emit("  br bb${_prev}")
        endif()
    endforeach()
    math(EXPR _expected "(${N} + 1) * ${_arg}")
elseif(SHAPE STREQUAL "ifs")
    # t_k tests x > k and nests into t_{k+1}; the join j_k adds 1. With
    # x > N every test holds and the result is x + N.
    math(EXPR _arg "${N} + 5")
    emit("  %one = iconst.i64 1" "  br t1")
    foreach(_k RANGE 1 ${N})
        math(EXPR _next "${_k} + 1")
        math(EXPR _prev "${_k} - 1")
        emit(
            "t${_k}:"
            "  %k${_k} = iconst.i64 ${_k}"
            "  %c${_k} = sgt.i64 %0, %k${_k}"
            "  br_if %c${_k}, t${_next}, e${_k}"
            "e${_k}:"
            "  br j${_k}(%0)"
            "j${_k}(%a${_k}: i64):"
            "  %b${_k} = add.i64 %a${_k}, %one")
        if(_k EQUAL 1)
            emit("  ret %b1")
        else()
            emit("  br j${_prev}(%b${_k})")
        endif()
    endforeach()
    math(EXPR _next "${N} + 1")
    emit("t${_next}:" "  br j${N}(%0)")
    math(EXPR _expected "${_arg} + ${N}")
elseif(SHAPE STREQUAL "loops")
    # h_k(i, a) runs its body (loop k+1) while i < 1, so once; the latch
    # l_k adds 1 to the accumulator. The innermost body adds x, and the
    # result is x + N.
    emit("  %zero = iconst.i64 0" "  %one = iconst.i64 1" "  br h1(%zero, %zero)")
    foreach(_k RANGE 1 ${N})
        math(EXPR _next "${_k} + 1")
        math(EXPR _prev "${_k} - 1")
        emit(
            "h${_k}(%i${_k}: i64, %a${_k}: i64):"
            "  %c${_k} = slt.i64 %i${_k}, %one")
        if(_k EQUAL 1)
            emit("  br_if %c1, h2(%zero, %a1), done(%a1)")
        else()
            emit("  br_if %c${_k}, h${_next}(%zero, %a${_k}), l${_prev}(%a${_k})")
        endif()
        emit(
            "l${_k}(%r${_k}: i64):"
            "  %n${_k} = add.i64 %i${_k}, %one"
            "  %s${_k} = add.i64 %r${_k}, %one"
            "  br h${_k}(%n${_k}, %s${_k})")
    endforeach()
    math(EXPR _next "${N} + 1")
    emit(
        "h${_next}(%i${_next}: i64, %a${_next}: i64):"
        "  %body = add.i64 %a${_next}, %0"
        "  br l${N}(%body)"
        "done(%res: i64):"
        "  ret %res")
    math(EXPR _expected "${_arg} + ${N}")
else()
    message(FATAL_ERROR "run_block_chain.cmake: unknown SHAPE '${SHAPE}'")
endif()
emit("}")

set(_mir "${OUT_DIR}/block_chain_${SHAPE}_${N}.mir")
file(MAKE_DIRECTORY "${OUT_DIR}")
file(WRITE "${_mir}" "${_text}")

string(REPLACE "," ";" _flags "${FLAGS}")
execute_process(
    COMMAND "${BRASS_OPT}" "${_mir}" -r chain --args ${_arg} ${_flags}
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
)
if(NOT _rc STREQUAL "0")
    message(FATAL_ERROR "brass-opt ${_mir} ${FLAGS}: exit code '${_rc}'\nstdout:\n${_out}\nstderr:\n${_err}")
endif()
if(NOT _out MATCHES "^${_expected}[\r\n]")
    message(FATAL_ERROR "brass-opt ${_mir} ${FLAGS}: expected ${_expected}, got\n${_out}\nstderr:\n${_err}")
endif()
