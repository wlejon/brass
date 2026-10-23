#pragma once

#include <brass/mir/function.hpp>
#include <functional>
#include <string>

namespace brass {

// Independently verifies proof certificates for:
// 1. GC write-barrier elision (BarrierCertKind on store / store_indexed)
// 2. Guard elimination & implied check folding (GuardElimCert on Function)
void verify_proof_certificates(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
);

} // namespace brass
