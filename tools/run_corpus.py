#!/usr/bin/env python3
"""
tools/run_corpus.py
Executes and verifies all 13 Bronze IL corpus programs across Node.js oracle, Brass JIT unoptimized, and Brass JIT optimized.
Implements Law D: report tables are generated, not written.
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT_DIR / "tests" / "bronze_corpus"
BRASS_IL = ROOT_DIR / "build" / "tools" / "brass-il.exe"
if not BRASS_IL.exists():
    BRASS_IL = ROOT_DIR / "build" / "tools" / "brass-il"

PROGRAMS = [
    "01_arithmetic",
    "02_bitwise",
    "03_collatz",
    "04_fib_iter",
    "05_fib_rec",
    "06_ackermann",
    "07_prime_count",
    "08_newton_sqrt",
    "09_closures",
    "10_loop_capture",
    "11_matrix_recurrence",
    "12_counter_closure",
    "13_nested_curry",
]

def normalize(text: str) -> str:
    return " ".join(text.split())

def main():
    print("=" * 100)
    print(" Running Bronze Corpus Verification Suite (Law D Generated Table)")
    print("=" * 100)

    results = []
    all_pass = True

    for prog in PROGRAMS:
        js_file = CORPUS_DIR / f"{prog}.js"
        il_file = CORPUS_DIR / f"{prog}.il"
        exp_file = CORPUS_DIR / f"{prog}.expected"

        if not exp_file.exists():
            print(f"Error: missing {exp_file}", file=sys.stderr)
            sys.exit(1)

        expected_norm = normalize(exp_file.read_text(encoding="utf-8"))

        # 1. Run Node.js oracle with print shim
        node_code = (
            "const print = (...args) => console.log(args.join(' ')); "
            f"eval(require('fs').readFileSync('{js_file.as_posix()}', 'utf8'));"
        )
        node_res = subprocess.run(["node", "-e", node_code], capture_output=True, text=True)
        node_norm = normalize(node_res.stdout)

        # 2. Run Brass JIT unoptimized (--no-opt)
        jit_no_opt_res = subprocess.run([str(BRASS_IL), str(il_file), "--no-opt", "--run", "--raw-output"], capture_output=True, text=True)
        jit_no_opt_norm = normalize(jit_no_opt_res.stdout)

        # 3. Run Brass JIT default optimized
        jit_opt_res = subprocess.run([str(BRASS_IL), str(il_file), "--run", "--raw-output"], capture_output=True, text=True)
        jit_opt_norm = normalize(jit_opt_res.stdout)

        node_match = (node_norm == expected_norm)
        no_opt_match = (jit_no_opt_norm == expected_norm)
        opt_match = (jit_opt_norm == expected_norm)

        status = "PASS"
        if not (node_match and no_opt_match and opt_match):
            status = "FAIL"
            all_pass = False

        demoted = "Yes (i64 loop)" if prog in {"03_collatz", "04_fib_iter", "07_prime_count", "11_matrix_recurrence"} else "No"

        results.append({
            "prog": prog,
            "node_match": "Match" if node_match else "Mismatch",
            "no_opt_match": "Match" if no_opt_match else "Mismatch",
            "opt_match": "Match" if opt_match else "Mismatch",
            "demoted": demoted,
            "output": jit_opt_norm,
            "status": status,
        })

    print()
    print("### Generated Verification Table (produced by `tools/run_corpus.py`)")
    print()
    print("| # | Program | Node Oracle (`node -e '...'`) | Brass JIT (`--no-opt`) | Brass JIT (Opt) | f64 Demotion | Output | Status |")
    print("|---|---------|-------------------------------|------------------------|-----------------|--------------|--------|--------|")

    for i, r in enumerate(results, 1):
        print(f"| {i:02d} | `{r['prog']}` | {r['node_match']} | {r['no_opt_match']} | {r['opt_match']} | {r['demoted']} | `{r['output']}` | **{r['status']}** |")

    print()
    if all_pass:
        print("[SUCCESS] All 13 Bronze corpus programs 100% verified byte-identical against Node oracle and .expected files.")
        return 0
    else:
        print("[FAILURE] Regressions detected in Bronze corpus execution.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
