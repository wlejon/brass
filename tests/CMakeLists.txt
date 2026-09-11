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
    unit/test_target_calling_conv.cpp
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
    unit/test_patching_concurrency.cpp
    unit/test_determinism_ratchet.cpp
    differential/test_differential_exec.cpp
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
    unit/test_software_pipeline.cpp
    differential/test_sched_differential.cpp
    unit/test_source_loc.cpp
    unit/test_source_map.cpp
    unit/test_symbolicator.cpp
    differential/test_debug_differential.cpp
    unit/test_shapes.cpp
    unit/test_inline_cache.cpp
    unit/test_property_lowering.cpp
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

add_test(NAME unit_tests COMMAND brass_unit_tests)
set_tests_properties(unit_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if(APPLE)
    add_custom_command(TARGET brass_unit_tests POST_BUILD
        COMMAND codesign -s - -f $<TARGET_FILE:brass_unit_tests>
        COMMENT "Ad-hoc code signing brass_unit_tests"
    )
endif()

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

add_test(NAME benchmarks COMMAND brass_benchmarks)
add_test(NAME performance_ratchet COMMAND brass_benchmarks --check-ratchet)
set_tests_properties(benchmarks performance_ratchet PROPERTIES TIMEOUT 300 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if(APPLE)
    add_custom_command(TARGET brass_benchmarks POST_BUILD
        COMMAND codesign -s - -f $<TARGET_FILE:brass_benchmarks>
        COMMENT "Ad-hoc code signing brass_benchmarks"
    )
endif()

