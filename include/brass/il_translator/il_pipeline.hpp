#pragma once

#include <brass/il_translator/il_translator.hpp>
#include <brass/mir/pass_pipeline.hpp>

namespace brass::il {

// The optimization pipeline configuration a translation with `options` runs.
PassPipelineOptions pass_pipeline_options(const TranslatorOptions& options);

// The TranslatorOptions Bronze (the JS engine) compiles with, for tools that
// need to reproduce Bronze's exact optimizer configuration. The source of
// truth is BrassBackend::buildMirModule in bronze's
// src/codegen-brass/brass_backend.cpp; keep this in step with it. Fields
// that depend on the module being compiled (IC sites, entry symbol, census)
// keep their defaults.
TranslatorOptions bronze_translator_options();

} // namespace brass::il
