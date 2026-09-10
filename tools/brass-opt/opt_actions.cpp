#include "opt_actions.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/pgo/instrument.hpp>
#include <iostream>
#include <fstream>

namespace brass {

namespace {

bool write_binary_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return true;
}

RuntimeValue parse_arg_for_type(Type type, const std::string& arg_str) {
    switch (type.kind()) {
        case TypeKind::I32:
            return RuntimeValue::from_i32(static_cast<int32_t>(std::stol(arg_str, nullptr, 0)));
        case TypeKind::I64:
            return RuntimeValue::from_i64(std::stoll(arg_str, nullptr, 0));
        case TypeKind::F32:
            return RuntimeValue::from_f32(std::stof(arg_str));
        case TypeKind::F64:
            return RuntimeValue::from_f64(std::stod(arg_str));
        case TypeKind::Ptr:
            return RuntimeValue::from_ptr(static_cast<uintptr_t>(std::stoull(arg_str, nullptr, 0)));
        case TypeKind::GCRef:
            return RuntimeValue::from_gcref(static_cast<uintptr_t>(std::stoull(arg_str, nullptr, 0)));
        case TypeKind::Void:
            return RuntimeValue::from_void();
        default:
            break;
    }
    return RuntimeValue::from_i64(std::stoll(arg_str, nullptr, 0));
}

} // namespace

bool execute_compile_object(const Module& mod, const CompileObjectOptions& opts) {
    Target target = Target::host();
    if (opts.obj_format == "coff") {
        target = Target::x64_windows();
    } else if (opts.obj_format == "elf") {
        target = Target::x64_linux();
    }

    codegen::SchedOptions sched_opts;
    sched_opts.enable_pre_ra = opts.enable_schedule_insns;
    sched_opts.enable_post_ra = opts.enable_schedule_insns;
    sched_opts.enable_software_pipelining = opts.enable_software_pipeline;

    auto obj = object::compile_module_to_object(mod, target, sched_opts);
    std::vector<uint8_t> binary_data;
    if (target.is_windows() || opts.obj_format == "coff") {
        binary_data = object::emit_coff_object(obj);
    } else {
        binary_data = object::emit_elf_object(obj);
    }

    std::string out_file = opts.output_file;
    if (out_file.empty()) {
        if (opts.input_file != "-") {
            size_t dot_pos = opts.input_file.find_last_of('.');
            std::string base = (dot_pos != std::string::npos) ? opts.input_file.substr(0, dot_pos) : opts.input_file;
            out_file = base + (target.is_windows() ? ".obj" : ".o");
        } else {
            out_file = target.is_windows() ? "out.obj" : "out.o";
        }
    }

    if (!write_binary_file(out_file, binary_data)) {
        std::cerr << "Error: Could not write object file to '" << out_file << "'\n";
        return false;
    }

    std::cout << "Successfully emitted object file '" << out_file << "' ("
              << binary_data.size() << " bytes, " << (target.is_windows() ? "COFF" : "ELF64") << ")\n";
    return true;
}

bool execute_emit_shared(const Module& mod, const EmitSharedOptions& opts) {
    Target target = Target::host();
    target::OutputFormat fmt = target::OutputFormat::Auto;
    if (opts.obj_format == "coff") {
        target = Target::x64_windows();
        fmt = target::OutputFormat::WindowsPeDll;
    } else if (opts.obj_format == "elf") {
        target = Target::x64_linux();
        fmt = target::OutputFormat::LinuxElfSo;
    }

    std::string final_output = opts.shared_output_file.empty() ? opts.output_file : opts.shared_output_file;
    if (final_output.empty()) {
        if (opts.input_file != "-") {
            size_t dot_pos = opts.input_file.find_last_of('.');
            std::string base = (dot_pos != std::string::npos) ? opts.input_file.substr(0, dot_pos) : opts.input_file;
            final_output = base + (target.is_windows() ? ".dll" : ".so");
        } else {
            final_output = target.is_windows() ? "out.dll" : "out.so";
        }
    }

    target::LinkerOptions link_opts;
    link_opts.format = fmt;
    link_opts.export_all_functions = true;

    if (!target::AotLinker::link_to_file(mod, final_output, target, link_opts)) {
        std::cerr << "Error: Could not link shared library to '" << final_output << "'\n";
        return false;
    }

    std::cout << "Successfully emitted shared library '" << final_output << "' ("
              << (target.is_windows() ? "PE32+ DLL" : "ELF64 SO") << ")\n";
    return true;
}

bool execute_run_function(Module& mod, const RunFunctionOptions& opts) {
    const Function* fn = mod.get_function(opts.run_fn);
    if (!fn) {
        std::cerr << "Error: Function '" << opts.run_fn << "' not found in module '" << mod.name() << "'\n";
        return false;
    }

    std::vector<RuntimeValue> run_args;
    for (size_t i = 0; i < fn->param_count(); ++i) {
        if (i < opts.run_arg_strings.size()) {
            try {
                run_args.push_back(parse_arg_for_type(fn->param_type(i), opts.run_arg_strings[i]));
            } catch (const std::exception& ex) {
                std::cerr << "Error: Could not parse argument " << i << " ('" << opts.run_arg_strings[i]
                          << "') for parameter type " << fn->param_type(i) << ": " << ex.what() << "\n";
                return false;
            }
        } else {
            run_args.push_back(RuntimeValue::from_i64(0));
        }
    }

    if (opts.use_jit) {
        codegen::JitExecutionEngine jit(Target::host());
        codegen::SchedOptions sched_opts;
        sched_opts.enable_pre_ra = opts.enable_schedule_insns;
        sched_opts.enable_post_ra = opts.enable_schedule_insns;
        sched_opts.enable_software_pipelining = opts.enable_software_pipeline;
        jit.set_sched_options(sched_opts);
        jit.register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
        if (!jit.compile_and_load(mod)) {
            std::cerr << "Error: JIT compilation/loading failed for module '" << mod.name() << "'\n";
            return false;
        }

        try {
            RuntimeValue result = jit.invoke(opts.run_fn, run_args);
            if (!fn->return_type().is_void()) {
                std::cout << result << "\n";
            }
            if (opts.dump_ic_stats) {
                runtime::ICRegistry::global().dump_stats(std::cout);
            }
        } catch (const std::exception& ex) {
            std::cerr << "JIT Execution error: " << ex.what() << "\n";
            return false;
        }
        return true;
    }

    Interpreter interp;
    interp.register_external_function("brass_pgo_inc", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (!args.empty()) {
            uint32_t idx = args[0].is_i32() ? args[0].as_u32() : static_cast<uint32_t>(args[0].as_u64());
            brass_pgo_inc(idx);
        }
        return RuntimeValue::from_void();
    });
    if (opts.gc_stress) {
        interp.gc().set_stress_mode(true);
    }
    if (opts.enable_osr) {
        runtime::OsrCoordinator::instance().set_enabled(true);
        runtime::OsrCoordinator::instance().set_threshold(opts.osr_threshold);
    }
    if (opts.enable_background_compile) {
        runtime::TieringRegistry::instance().set_background_compile_enabled(true);
        runtime::TieringRegistry::instance().set_jit_threads(opts.jit_threads);
        runtime::BackgroundCompiler::instance().start(opts.jit_threads);
    }

    try {
        RuntimeValue result = interp.run(mod, opts.run_fn, run_args);
        if (!fn->return_type().is_void()) {
            std::cout << result << "\n";
        }
        if (opts.enable_background_compile) {
            runtime::BackgroundCompiler::instance().wait_idle();
        }
        if (opts.dump_tiering_stats) {
            runtime::TieringRegistry::instance().dump_stats(std::cout);
        }
        if (opts.dump_jit_thread_stats) {
            runtime::BackgroundCompiler::instance().dump_stats(std::cout);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Runtime error during execution: " << ex.what() << "\n";
        return false;
    }

    return true;
}

} // namespace brass
