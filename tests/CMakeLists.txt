# Tests CMakeLists.txt

add_executable(brass_unit_tests
    unit/test_sanity.cpp
    unit/test_arena.cpp
    unit/test_core_utils.cpp
    unit/test_mir_builder.cpp
    unit/test_mir_verifier.cpp
    unit/test_mir_printer_parser.cpp
    unit/test_mir_parser_strictness.cpp
    unit/test_mir_loop_opt.cpp
    unit/test_mini_cheney.cpp
    unit/test_interpreter.cpp
    unit/test_bytecode_compiler.cpp
    unit/test_fast_interpreter.cpp
    unit/test_fast_interpreter_advanced.cpp
    unit/test_target_calling_conv.cpp
    unit/test_aarch64_registers.cpp
    unit/test_aarch64_operands.cpp
    unit/test_aarch64_encoder.cpp
    unit/test_aarch64_frame.cpp
    unit/test_aarch64_isel.cpp
    unit/test_aarch64_emit.cpp
    unit/test_aarch64_runtime.cpp
    unit/test_aarch64_objects.cpp
    unit/test_x64_encoder.cpp
    unit/test_lir_isel.cpp
    unit/test_linear_scan.cpp
    unit/test_regalloc_held_pregs.cpp
    unit/test_peephole.cpp
    unit/test_x64_codegen.cpp
    unit/test_coff_writer.cpp
    unit/test_elf_writer.cpp
    unit/test_macho_writer.cpp
    unit/test_stack_map.cpp
    unit/test_code_stack_maps.cpp
    unit/test_stack_walker.cpp
    unit/test_deopt_resume.cpp
    unit/test_deopt_complete.cpp
    unit/test_deopt_heap.cpp
    unit/test_deopt_fast_rebind.cpp
    unit/test_patching_concurrency.cpp
    unit/test_jit_engine_move.cpp
    unit/test_determinism_ratchet.cpp
    unit/test_frontend_interp_safety.cpp
    differential/test_differential_exec.cpp
    differential/test_fast_interpreter_diff.cpp
    differential/test_moving_gc_native.cpp
    differential/test_speculation_differential.cpp
    differential/test_fuzz_loops.cpp
    differential/test_fuzz_gc.cpp
    differential/test_fuzz_speculation.cpp
    differential/test_fuzz_memory.cpp
    differential/test_fuzz_patching.cpp
    differential/test_fuzz_matrix.cpp
    differential/test_fuzz_js_shapes.cpp
    differential/test_fuzz_optimizations.cpp
    differential/fuzz_generator.cpp
    benchmarks/bench_numeric_modules.cpp
    unit/test_msvc_milestone.cpp
    unit/test_embedding_api.cpp
    unit/test_aot_embedding.cpp
    unit/test_mir_unroll_select.cpp
    unit/test_mir_alias_safety.cpp
    unit/test_il_expressibility.cpp
    differential/test_embedding_differential.cpp
    differential/test_fuzz_js_loops.cpp
    unit/test_f64_demote.cpp
    unit/test_inliner.cpp
    differential/test_inlining_differential.cpp
    unit/test_simd_mir.cpp
    differential/test_simd_differential.cpp
    unit/test_vectorizer.cpp
    differential/test_vectorizer_differential.cpp
    unit/test_sroa.cpp
    differential/test_sroa_differential.cpp
    unit/test_gvn.cpp
    differential/test_gvn_differential.cpp
    unit/test_pe_dll_writer.cpp
    unit/test_elf_so_writer.cpp
    unit/test_aot_shared_library.cpp
    unit/test_aot_imports.cpp
    unit/test_pic_x64.cpp
    unit/test_loop_tile.cpp
    differential/test_loop_tile_differential.cpp
    unit/test_sccp.cpp
    differential/test_sccp_differential.cpp
    unit/test_loop_unswitch.cpp
    unit/test_jump_threading.cpp
    unit/test_block_layout.cpp
    differential/test_unswitch_differential.cpp
    unit/test_profile_serialization.cpp
    unit/test_pgo_instrument.cpp
    unit/test_branch_probability.cpp
    differential/test_pgo_differential.cpp
    unit/test_sched_dag.cpp
    unit/test_instruction_scheduler.cpp
    unit/test_lir_flags.cpp
    unit/test_backend_hard_errors.cpp
    unit/test_aarch64_relocs.cpp
    unit/test_aarch64_image_relocs.cpp
    unit/test_aarch64_branch_relax.cpp
    unit/test_aarch64_execution_regressions.cpp
    unit/test_stack_args_tiers.cpp
    unit/test_narrow_int_tiers.cpp
    unit/test_jit_unwind.cpp
    unit/test_software_pipeline.cpp
    differential/test_sched_differential.cpp
    unit/test_source_loc.cpp
    unit/test_source_map.cpp
    unit/test_symbolicator.cpp
    differential/test_debug_differential.cpp
    unit/test_tlab.cpp
    unit/test_host_heap.cpp
    unit/test_heap_roots_regressions.cpp
    unit/test_gc_sweep19_regressions.cpp
    unit/test_gc_sweep20_regressions.cpp
    unit/test_gc_sweep21_regressions.cpp
    unit/test_sweep22_regressions.cpp
    unit/test_sweep23_regressions.cpp
    unit/test_sweep25_regressions.cpp
    unit/test_sweep26_regressions.cpp
    unit/test_sweep27_regressions.cpp
    unit/test_sweep29_regressions.cpp
    unit/test_sweep30_regressions.cpp
    unit/test_sweep31_regressions.cpp
    unit/test_sweep32_regressions.cpp
    unit/test_sweep33_regressions.cpp
    unit/test_sweep34_regressions.cpp
    unit/test_sweep35b_regressions.cpp
    unit/test_interp_coro_regressions.cpp
    unit/test_coro_mir_frames.cpp
    unit/test_exceptions_mir.cpp
    unit/test_unwind_tables.cpp
    unit/test_exception_dispatch.cpp
    differential/test_exception_differential.cpp
    unit/test_coro_mir.cpp
    unit/test_coro_transform.cpp
    unit/test_coro_live_across_suspend.cpp
    unit/test_coro_runtime.cpp
    differential/test_coro_differential.cpp
    unit/test_partial_escape.cpp
    unit/test_allocation_sinking.cpp
    differential/test_pea_differential.cpp
    unit/test_card_table.cpp
    unit/test_generational_gc.cpp
    unit/test_write_barrier_elim.cpp
    differential/test_generational_differential.cpp
    unit/test_tiering.cpp
    unit/test_osr_entry.cpp
    unit/test_osr_runtime.cpp
    differential/test_osr_differential.cpp
    differential/test_osr_regressions.cpp
    differential/test_eh_coro_regressions.cpp
    unit/test_critical_edge.cpp
    unit/test_gvn_pre.cpp
    differential/test_gvn_pre_differential.cpp
    unit/test_loop_fusion.cpp
    unit/test_loop_distribution.cpp
    unit/test_array_contraction.cpp
    differential/test_loop_transform_differential.cpp
    unit/test_vex_encoder.cpp
    unit/test_v256_mir.cpp
    unit/test_fma_opt.cpp
    differential/test_avx2_differential.cpp
    differential/test_avx2_isel_regressions.cpp
    unit/test_background_compiler.cpp
    unit/test_atomic_code_install.cpp
    differential/test_async_jit_differential.cpp
    unit/test_loop_dependence.cpp
    unit/test_parallel_runtime.cpp
    unit/test_auto_parallel.cpp
    differential/test_parallel_differential.cpp
    unit/test_dwarf_emitter.cpp
    unit/test_codeview_emitter.cpp
    unit/test_debug_relocs.cpp
    unit/test_fuzz_mutator.cpp
    unit/test_fuzz_sandbox.cpp
    unit/test_delta_reducer.cpp
    unit/test_c_api.cpp
    unit/test_c_api_lifetime.cpp
    unit/test_range_analysis.cpp
    unit/test_bounds_check_elim.cpp
    unit/test_optimizer_soundness.cpp
    unit/test_optimizer_soundness_passes.cpp
    unit/test_loop_parallel_tile_soundness.cpp
    unit/test_fuzz_pipelines.cpp
    unit/test_mir_roundtrip_all.cpp
    unit/test_pass_manager.cpp
    unit/test_mir_uses.cpp
    unit/test_loop_ivsr.cpp
    unit/test_jit_fuzz_regressions.cpp
    unit/test_type_feedback.cpp
    unit/test_speculative_inlining.cpp
    unit/test_baseline_jit.cpp
    unit/test_baseline_jit_opcodes.cpp
    unit/test_baseline_jit_vector.cpp
    unit/test_tier2_vector_calls.cpp
    unit/test_invoke_return_types.cpp
    unit/test_baseline_lazy_symbols.cpp
    unit/test_tier1_cold_callees.cpp
    unit/test_tier1_link_order.cpp
    unit/test_function_pointer_tiers.cpp
    unit/test_gc_native_frames.cpp
    unit/test_baseline_frame.cpp
    unit/test_baseline_unwind.cpp
    unit/test_program_dispatch.cpp
    unit/test_program_runtime_state.cpp
    unit/test_kernel_jit.cpp
    unit/test_audio_intrinsics.cpp
    unit/test_sdf_builder.cpp
    unit/test_grammar_builder.cpp
    unit/test_image_builder.cpp
    unit/test_ptx_target.cpp
    unit/test_ptx_ir.cpp
    unit/test_ptx_isel.cpp
    unit/test_ptx_intrinsics.cpp
    unit/test_ptx_vector.cpp
    unit/test_ptx_cleanup.cpp
    unit/test_ml_fusion.cpp
    unit/test_gpu_execution.cpp
    unit/test_gpu_kernels.cpp
    unit/test_gpu_kernels_quant.cpp
    unit/test_phase1_concurrency.cpp
    unit/test_phase2_mir_opt.cpp
    unit/test_phase3_target_codegen.cpp
    unit/test_phase4_gc_frontend.cpp
    unit/test_backend_hardening.cpp
    unit/test_aarch64_backend_hardening.cpp
    unit/test_gc_concurrency_safety.cpp
    unit/test_gc_runtime_hardening.cpp
    unit/test_deopt_concurrency_hardening.cpp
    unit/test_runtime_activation.cpp
    unit/test_mir_optimizer_hardening.cpp
    unit/test_frontend_gpu_hardening.cpp
)

target_include_directories(brass_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/harness
        ${CMAKE_CURRENT_SOURCE_DIR}/differential
        ${BRASS_INCLUDE_DIR}
)

target_link_libraries(brass_unit_tests
    PRIVATE
        brass
)

# test_c_api_lifetime.cpp checks the shared library's C exports.
if(TARGET brass_shared)
    add_dependencies(brass_unit_tests brass_shared)
    target_compile_definitions(brass_unit_tests PRIVATE
        BRASS_TEST_SHARED_LIB="$<TARGET_FILE:brass_shared>")
endif()

# brass_windows_h_first: <brass/brass.hpp> compiles after <windows.h>
# (without NOMINMAX); it once failed on the COFF/PE writers' IMAGE_*
# constants and on std::min / std::max.
if(WIN32)
    add_executable(brass_windows_h_first unit/windows_h_first.cpp)
    target_include_directories(brass_windows_h_first PRIVATE ${BRASS_INCLUDE_DIR})
    target_link_libraries(brass_windows_h_first PRIVATE brass)
    add_test(NAME brass_windows_h_first COMMAND brass_windows_h_first)
    set_tests_properties(brass_windows_h_first PROPERTIES
        TIMEOUT 60 PASS_REGULAR_EXPRESSION "^OK[\r\n]" LABELS "correctness")
endif()

# One ctest per TEST_CASE, discovered from the binary each time ctest starts
# (see cmake/discover_unit_tests.cmake), so one crash cannot hide the rest.
# The per-config file carries the binary's path; the include below picks the
# file for the configuration ctest was asked to test.
set(BRASS_UNIT_TEST_TIMEOUT 600 CACHE STRING "Per-test timeout (seconds) for brass unit tests")
set(_brass_discovery_dir "${CMAKE_CURRENT_BINARY_DIR}/ctest_discovery")
file(GENERATE
    OUTPUT "${_brass_discovery_dir}/unit_tests_$<CONFIG>.cmake"
    CONTENT "set(BRASS_TEST_EXE [==[$<TARGET_FILE:brass_unit_tests>]==])
set(BRASS_TEST_WORKDIR [==[${CMAKE_SOURCE_DIR}]==])
set(BRASS_TEST_TIMEOUT ${BRASS_UNIT_TEST_TIMEOUT})
set(BRASS_TEST_EMULATOR [==[${CMAKE_CROSSCOMPILING_EMULATOR}]==])
include([==[${CMAKE_CURRENT_SOURCE_DIR}/cmake/discover_unit_tests.cmake]==])
")
file(WRITE "${_brass_discovery_dir}/include.cmake"
"if(CTEST_CONFIGURATION_TYPE)
    set(_brass_cfg \"\${CTEST_CONFIGURATION_TYPE}\")
else()
    set(_brass_cfg [==[${CMAKE_BUILD_TYPE}]==])
endif()
include(\"${_brass_discovery_dir}/unit_tests_\${_brass_cfg}.cmake\" OPTIONAL RESULT_VARIABLE _brass_found)
if(NOT _brass_found)
    add_test(unit_tests.no_config \"unit_tests_\${_brass_cfg}.cmake-not-generated\")
endif()
")
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_brass_discovery_dir}/include.cmake")

if(APPLE)
    add_custom_command(TARGET brass_unit_tests POST_BUILD
        COMMAND codesign -s - -f $<TARGET_FILE:brass_unit_tests>
        COMMENT "Ad-hoc code signing brass_unit_tests"
    )
endif()

# Consumers (bronze builds with MSVC /W4 /WX) compile brass's public headers
# with their own warning flags, not brass's suppressions. This object library
# includes every public header under consumer-style flags so a header warning
# breaks brass's build before it breaks theirs.
file(GLOB_RECURSE BRASS_PUBLIC_HEADERS CONFIGURE_DEPENDS
    "${BRASS_INCLUDE_DIR}/*.hpp" "${BRASS_INCLUDE_DIR}/*.h")
list(SORT BRASS_PUBLIC_HEADERS)
set(_brass_header_check_src "")
foreach(_hdr IN LISTS BRASS_PUBLIC_HEADERS)
    file(RELATIVE_PATH _rel "${BRASS_INCLUDE_DIR}" "${_hdr}")
    string(APPEND _brass_header_check_src "#include <${_rel}>\n")
endforeach()
string(APPEND _brass_header_check_src "int brass_public_headers_check_anchor = 0;\n")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/public_headers_check.cpp"
    CONTENT "${_brass_header_check_src}")
add_library(brass_public_headers_check OBJECT "${CMAKE_CURRENT_BINARY_DIR}/public_headers_check.cpp")
# Drop the directory-wide options (they carry brass's own /wd suppressions).
set_target_properties(brass_public_headers_check PROPERTIES COMPILE_OPTIONS "")
target_include_directories(brass_public_headers_check PRIVATE ${BRASS_INCLUDE_DIR})
target_compile_features(brass_public_headers_check PRIVATE cxx_std_20)
target_compile_definitions(brass_public_headers_check PRIVATE BRASS_HOST_BACKEND=$<BOOL:${BRASS_HOST_BACKEND}>)
if(MSVC)
    target_compile_options(brass_public_headers_check PRIVATE
        /W4 /WX /EHsc /permissive- /Zc:__cplusplus /utf-8)
else()
    target_compile_options(brass_public_headers_check PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_dependencies(brass_unit_tests brass_public_headers_check)

# Retrieve Git Commit SHA
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE BRASS_GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT BRASS_GIT_COMMIT_HASH)
    set(BRASS_GIT_COMMIT_HASH "unknown")
else()
    execute_process(
        COMMAND git status --porcelain
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE BRASS_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(BRASS_GIT_STATUS)
        set(BRASS_GIT_COMMIT_HASH "${BRASS_GIT_COMMIT_HASH}-dirty")
    endif()
endif()

# Benchmarks
add_executable(brass_benchmarks
    benchmarks/benchmark_suite.cpp
    benchmarks/bench_numeric.cpp
    benchmarks/bench_numeric_modules.cpp
    benchmarks/bench_js_shapes.cpp
    benchmarks/bench_js_shapes_modules.cpp
    benchmarks/bench_gc.cpp
    benchmarks/bench_compile_speed.cpp
    benchmarks/bench_simd_math.cpp
    benchmarks/bench_interpreter.cpp
)

target_include_directories(brass_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
        ${BRASS_INCLUDE_DIR}
)

target_compile_definitions(brass_benchmarks
    PRIVATE
        BRASS_GIT_COMMIT="${BRASS_GIT_COMMIT_HASH}"
        BRASS_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        BRASS_BENCH_RATCHET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../bench/ratchet.json"
)

target_link_libraries(brass_benchmarks
    PRIVATE
        brass
)

# Performance targets, not correctness: labelled "perf" so correctness runs
# (`ctest -L correctness`, as CI does) exclude them and a slow machine or a
# perf regression reads as exactly that.
add_test(NAME benchmarks COMMAND brass_benchmarks)
add_test(NAME performance_ratchet COMMAND brass_benchmarks --check-ratchet)
set_tests_properties(benchmarks performance_ratchet PROPERTIES
    TIMEOUT 600 RUN_SERIAL TRUE WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS "perf")

if(BRASS_BUILD_TOOLS)
    # brass_opt_cli.coro_args.*: brass-opt lowers coroutines before running
    # them (an unlowered body once yielded 1, 0, 0), and --args takes a
    # negative number (it was rejected as an unknown option).
    foreach(_mode interp jit)
        set(_flags "")
        if(_mode STREQUAL "jit")
            set(_flags "--jit")
        endif()
        add_test(NAME "brass_opt_cli.coro_args.${_mode}"
            COMMAND brass-opt "${CMAKE_CURRENT_SOURCE_DIR}/cli/coro_args.mir" -r main --args -5 ${_flags})
        set_tests_properties("brass_opt_cli.coro_args.${_mode}" PROPERTIES
            TIMEOUT 60 PASS_REGULAR_EXPRESSION "^1011026[\r\n]" LABELS "correctness")
    endforeach()
    # brass_opt_cli.div_zero.*: an integer division by zero is an error and
    # exit code 1 in every engine. JIT code carries no zero check; the #DE
    # it raised was unhandled and killed brass-opt with 0xC0000094 and no
    # message.
    foreach(_fn sdiv64 umod32)
        foreach(_mode interp jit baseline)
            set(_flags "")
            if(_mode STREQUAL "jit")
                set(_flags ",--jit")
            elseif(_mode STREQUAL "baseline")
                set(_flags ",--baseline-jit")
            endif()
            add_test(NAME "brass_opt_cli.div_zero.${_fn}.${_mode}"
                COMMAND ${CMAKE_COMMAND}
                    "-DCMD=$<TARGET_FILE:brass-opt>,${CMAKE_CURRENT_SOURCE_DIR}/cli/div_zero.mir,-r,${_fn},--args,7,0${_flags}"
                    "-DEXIT_CODE=1"
                    "-DSTDERR_RE=([Dd]ivision|Modulo) by zero"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_expect_error.cmake")
            set_tests_properties("brass_opt_cli.div_zero.${_fn}.${_mode}" PROPERTIES
                TIMEOUT 60 LABELS "correctness")
        endforeach()
    endforeach()
    # brass_opt_cli.guard.*: guard exits give the interpreter's answer in
    # every engine (docs/mir_reference.md, "Guard exits"). An exit stub is
    # called as stub(state values...) and its result, float or integer, is
    # the function's; a resume block's parameter i takes state value i; a
    # guard with both takes the stub, once. osr_resume returned 0 under OSR
    # (the resume block's parameters were filled past the state values),
    # osr_stub failed under OSR ("Resume target ID 0 not found") and returned
    # garbage under --jit (tier 2 called the stub as stub(resume_id, slots)),
    # and stub_f64 returned 0.5 under --jit (RAX overwrote the stub's XMM0).
    # Standalone tier 2 (--jit) has no lower tier to continue a resume block
    # in, so osr_resume is a fatal error there by design and is not run.
    # param0_dispatch: a guard that never fails, in a function whose first
    # parameter is i32 and which has a resume table; tier 2 returned the
    # resume block's 7777 for f(0, 1000) (its entry dispatched on %k).
    set(_guard_cases
        "param0_dispatch|f|0,1000|499500|interp baseline jit osr jit_osr"
        "param0_dispatch|f|5,1000|504500|interp baseline jit osr jit_osr"
        "osr_resume|f|1000|11175150|interp baseline osr"
        "osr_stub|f|1000|11175150|interp baseline jit osr jit_osr"
        "osr_both|main|1000|111751501|interp baseline jit osr jit_osr"
        "stub_f64|spec|3.0,0|4[.]5|interp baseline jit osr jit_osr"
        "stub_f64|spec|3.0,1|8[.]0|interp baseline jit osr jit_osr"
        "stub_many|spec|3,0|148978294|interp baseline jit osr jit_osr"
        "stub_many|spec|3,1|13|interp baseline jit osr jit_osr")
    foreach(_case IN LISTS _guard_cases)
        string(REPLACE "|" ";" _parts "${_case}")
        list(GET _parts 0 _file)
        list(GET _parts 1 _fn)
        list(GET _parts 2 _args)
        list(GET _parts 3 _want)
        list(GET _parts 4 _modes)
        string(REPLACE "," ";" _arg_list "${_args}")
        string(REPLACE "," "_" _arg_tag "${_args}")
        string(REPLACE " " ";" _modes "${_modes}")
        foreach(_mode IN LISTS _modes)
            set(_flags "")
            if(_mode STREQUAL "jit")
                set(_flags "--jit")
            elseif(_mode STREQUAL "baseline")
                set(_flags "--baseline-jit")
            elseif(_mode STREQUAL "osr")
                # The interpreter enters the hot loop's tier-2 code through OSR.
                set(_flags "--enable-osr;--osr-threshold=10")
            elseif(_mode STREQUAL "jit_osr")
                set(_flags "--jit;--enable-osr;--osr-threshold=10")
            endif()
            set(_name "brass_opt_cli.guard.${_file}.${_fn}.${_arg_tag}.${_mode}")
            add_test(NAME "${_name}"
                COMMAND brass-opt "${CMAKE_CURRENT_SOURCE_DIR}/cli/guard_${_file}.mir" -r ${_fn} --args ${_arg_list} ${_flags})
            set_tests_properties("${_name}" PROPERTIES
                TIMEOUT 60 PASS_REGULAR_EXPRESSION "^${_want}[\r\n]" LABELS "correctness")
        endforeach()
    endforeach()
    # brass_opt_cli.ret_i32.*: a negative i32 result prints as a negative
    # number in every engine. Tier-2 invoke zero-extended an i32 result into
    # an i64 when the function had one or two arguments (neg(5) printed
    # 4293967301), also when an exit stub produced it.
    set(_ret_i32_cases
        "neg|5|-999995"
        "guarded|5,1|-999995"
        "guarded|5,0|-999995"
        "loop|1000,150|-999550")
    foreach(_case IN LISTS _ret_i32_cases)
        string(REPLACE "|" ";" _parts "${_case}")
        list(GET _parts 0 _fn)
        list(GET _parts 1 _args)
        list(GET _parts 2 _want)
        string(REPLACE "," ";" _arg_list "${_args}")
        string(REPLACE "," "_" _arg_tag "${_args}")
        foreach(_mode interp baseline jit)
            set(_flags "")
            if(_mode STREQUAL "jit")
                set(_flags "--jit")
            elseif(_mode STREQUAL "baseline")
                set(_flags "--baseline-jit")
            endif()
            set(_name "brass_opt_cli.ret_i32.${_fn}.${_arg_tag}.${_mode}")
            add_test(NAME "${_name}"
                COMMAND brass-opt "${CMAKE_CURRENT_SOURCE_DIR}/cli/ret_i32.mir" -r ${_fn} --args ${_arg_list} ${_flags})
            set_tests_properties("${_name}" PROPERTIES
                TIMEOUT 60 PASS_REGULAR_EXPRESSION "^${_want}[\r\n]" LABELS "correctness")
        endforeach()
    endforeach()
    # brass_opt_cli.guard.stub_twin_abi.*: an exit stub whose signature is
    # not stub(state values...) (here the old generic-twin (resume_id, slots))
    # is rejected by the verifier in every engine. It dereferenced a null
    # pointer in the interpreter and crashed the baseline JIT.
    foreach(_mode interp baseline jit)
        set(_flags "")
        if(_mode STREQUAL "jit")
            set(_flags ",--jit")
        elseif(_mode STREQUAL "baseline")
            set(_flags ",--baseline-jit")
        endif()
        add_test(NAME "brass_opt_cli.guard.stub_twin_abi.${_mode}"
            COMMAND ${CMAKE_COMMAND}
                "-DCMD=$<TARGET_FILE:brass-opt>,${CMAKE_CURRENT_SOURCE_DIR}/cli/guard_stub_twin_abi.mir,-r,spec,--args,3,0${_flags}"
                "-DEXIT_CODE=1"
                "-DSTDERR_RE=exit stub 'twin' takes 2 parameters but the guard has 1 state values"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_expect_error.cmake")
        set_tests_properties("brass_opt_cli.guard.stub_twin_abi.${_mode}" PROPERTIES
            TIMEOUT 60 LABELS "correctness")
    endforeach()
    # brass_opt_cli.block_chain.*: a function of 20000 blocks chained in
    # reverse text order (bb0 -> bbN -> ... -> bb1), run through the tier-2
    # JIT. CFG walks, dominator-tree walks and loop-nest walks recursed once
    # per block or level and overflowed the native stack from ~10000 blocks.
    # The input is generated by the script.
    foreach(_shape reversed ordered)
        add_test(NAME "brass_opt_cli.block_chain.${_shape}.jit"
            COMMAND ${CMAKE_COMMAND}
                "-DBRASS_OPT=$<TARGET_FILE:brass-opt>"
                "-DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/block_chain"
                "-DSHAPE=${_shape}"
                "-DN=20000"
                "-DFLAGS=--jit"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_block_chain.cmake")
        set_tests_properties("brass_opt_cli.block_chain.${_shape}.jit" PROPERTIES
            TIMEOUT 120 LABELS "correctness")
    endforeach()
    # brass_opt_cli.compile_scaling.*: compile time must stay near-linear in
    # function size. The LIR peephole rescanned to the end of the block from
    # every move (an 80000-deep add chain in one block took most of a
    # minute), and MIR folding replaced each folded value's uses by walking
    # the whole function. Each case takes a few seconds; the timeout is the
    # check.
    foreach(_case "expr;80000" "loops;1000" "ifs;5000")
        list(GET _case 0 _shape)
        list(GET _case 1 _n)
        add_test(NAME "brass_opt_cli.compile_scaling.${_shape}.jit"
            COMMAND ${CMAKE_COMMAND}
                "-DBRASS_OPT=$<TARGET_FILE:brass-opt>"
                "-DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/compile_scaling"
                "-DSHAPE=${_shape}"
                "-DN=${_n}"
                "-DFLAGS=--jit"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_block_chain.cmake")
        set_tests_properties("brass_opt_cli.compile_scaling.${_shape}.jit" PROPERTIES
            TIMEOUT 60 LABELS "correctness")
    endforeach()
    # aot_coff.eh: exceptions in AOT COFF code on Win64. brass-opt -c makes
    # an object from tests/aot/eh_aot.mir, MSVC links it with a C host and
    # brass.lib, and every result must match the interpreter and the JIT:
    # throw to a landing pad, rethrow from a pad, resume, and values live
    # into a pad across a frame that uses callee-saved registers. The first
    # throw once ended the process silently (unrelocated scope tables, a
    # personality that never unwound, and a throw that only searched JIT
    # code).
    if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        set(_eh_aot_obj "${CMAKE_CURRENT_BINARY_DIR}/aot/eh_aot.obj")
        add_custom_command(
            OUTPUT "${_eh_aot_obj}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/aot"
            COMMAND brass-opt "${CMAKE_CURRENT_SOURCE_DIR}/aot/eh_aot.mir" -c -o "${_eh_aot_obj}"
            DEPENDS brass-opt "${CMAKE_CURRENT_SOURCE_DIR}/aot/eh_aot.mir"
            COMMENT "brass-opt -c eh_aot.mir"
            VERBATIM)
        set_source_files_properties("${_eh_aot_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        add_executable(brass_aot_eh_host aot/eh_aot_host.c "${_eh_aot_obj}")
        target_link_libraries(brass_aot_eh_host PRIVATE brass)
        add_test(NAME aot_coff.eh
            COMMAND ${CMAKE_COMMAND}
                "-DHOST_EXE=$<TARGET_FILE:brass_aot_eh_host>"
                "-DBRASS_OPT=$<TARGET_FILE:brass-opt>"
                "-DMIR=${CMAKE_CURRENT_SOURCE_DIR}/aot/eh_aot.mir"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_aot_eh.cmake")
        set_tests_properties(aot_coff.eh PROPERTIES TIMEOUT 300 LABELS "correctness")
    endif()
endif()

if(APPLE)
    add_custom_command(TARGET brass_benchmarks POST_BUILD
        COMMAND codesign -s - -f $<TARGET_FILE:brass_benchmarks>
        COMMENT "Ad-hoc code signing brass_benchmarks"
    )
endif()

