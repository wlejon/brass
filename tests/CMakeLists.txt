# Tests CMakeLists.txt

add_executable(brass_unit_tests
    unit/test_sanity.cpp
    unit/test_arena.cpp
    unit/test_core_utils.cpp
    unit/test_mir_builder.cpp
    unit/test_mir_verifier.cpp
    unit/test_mir_printer_parser.cpp
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
    unit/test_peephole.cpp
    unit/test_x64_codegen.cpp
    unit/test_coff_writer.cpp
    unit/test_elf_writer.cpp
    unit/test_macho_writer.cpp
    unit/test_stack_map.cpp
    unit/test_stack_walker.cpp
    unit/test_deopt_resume.cpp
    unit/test_deopt_complete.cpp
    unit/test_patching_concurrency.cpp
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
    unit/test_bronze_il_translator.cpp
    differential/test_fuzz_closures_gc.cpp
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
    unit/test_jit_unwind.cpp
    unit/test_software_pipeline.cpp
    differential/test_sched_differential.cpp
    unit/test_source_loc.cpp
    unit/test_source_map.cpp
    unit/test_symbolicator.cpp
    differential/test_debug_differential.cpp
    unit/test_shapes.cpp
    unit/test_inline_cache.cpp
    unit/test_property_lowering.cpp
    unit/test_inlined_fastpaths.cpp
    unit/test_tlab.cpp
    differential/test_pic_differential.cpp
    unit/test_exceptions_mir.cpp
    unit/test_unwind_tables.cpp
    unit/test_exception_dispatch.cpp
    differential/test_exception_differential.cpp
    unit/test_coro_mir.cpp
    unit/test_coro_transform.cpp
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
    unit/test_native_cabi.cpp
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
    unit/test_baseline_lazy_symbols.cpp
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

if(APPLE)
    add_custom_command(TARGET brass_benchmarks POST_BUILD
        COMMAND codesign -s - -f $<TARGET_FILE:brass_benchmarks>
        COMMENT "Ad-hoc code signing brass_benchmarks"
    )
endif()

