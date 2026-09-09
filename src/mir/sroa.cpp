#include <brass/mir/sroa.hpp>
#include "sroa_transform.hpp"

namespace brass {

bool sroa_function(Function& fn) {
    SroaOptions opts;
    return sroa_function(fn, opts);
}

bool sroa_function(Function& fn, const SroaOptions& options) {
    SroaTransformer transformer(fn, options, options.stats);
    bool changed = transformer.run();
    if (changed) {
        fn.rebuild_cfg_predecessors();
    }
    return changed;
}

bool sroa_module(Module& mod) {
    SroaOptions opts;
    return sroa_module(mod, opts);
}

bool sroa_module(Module& mod, const SroaOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= sroa_function(*fn, options);
        }
    }
    return changed;
}

} // namespace brass
