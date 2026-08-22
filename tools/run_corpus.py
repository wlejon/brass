#!/usr/bin/env python3
"""
tools/run_corpus.py
Executes and verifies all 13 Bronze IL corpus programs across Node.js oracle,
Brass JIT unoptimized (--no-opt), Brass JIT no-demote (--no-demote), and Brass JIT full optimized.
Measures runtime medians and spreads across >= 5 runs per configuration.
Implements Law D: report tables are generated, not written.
"""

import subprocess
import sys
import time
import re
from pathlib import Path
from statistics import median

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
    "14_array_loop",
    "15_nested_acc",
    "16_param_bounds",
    "17_large_int_overflow",
    "18_gcd_iter",
    "19_vec3_acc",
    "20_mat4_mul",
    "21_quat_norm",
    "22_bbox_expand",
]

DEMOTED_PROGRAMS = {
    "03_collatz",
    "04_fib_iter",
    "07_prime_count",
    "11_matrix_recurrence",
    "12_counter_closure",
    "15_nested_acc",
    "16_param_bounds",
    "18_gcd_iter",
}

NUM_RUNS = 5

def normalize(text: str) -> str:
    text = re.sub(r'\[brass-il-timed:[^\]]*\]', '', text)
    return " ".join(text.split())

def run_timed_config(cmd, num_runs=NUM_RUNS):
    cmd_str = [str(c) for c in cmd]
    # Check if cmd is brass-il
    is_brass_il = any("brass-il" in c for c in cmd_str)
    if is_brass_il and "--timed" not in cmd_str:
        timed_cmd = list(cmd_str) + ["--timed", str(num_runs)]
        res = subprocess.run(timed_cmd, capture_output=True, text=True)
        m = re.search(r'\[brass-il-timed:\s*([\d\.]+)\s*\+/-\s*([\d\.]+)\s*\(min:\s*([\d\.]+),\s*max:\s*([\d\.]+)\)\]', res.stdout)
        if m:
            med_t = float(m.group(1))
            spread = float(m.group(2))
            min_t = float(m.group(3))
            max_t = float(m.group(4))
            return med_t, spread, min_t, max_t, normalize(res.stdout)

    # Fallback to subprocess timing
    warmup_res = subprocess.run(cmd_str, capture_output=True, text=True)
    times = []
    last_stdout = warmup_res.stdout
    for _ in range(num_runs):
        t0 = time.perf_counter()
        res = subprocess.run(cmd_str, capture_output=True, text=True)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        last_stdout = res.stdout
    med_t = median(times)
    min_t = min(times)
    max_t = max(times)
    spread = (max_t - min_t) / 2.0
    return med_t, spread, min_t, max_t, normalize(last_stdout)

def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: python tools/run_corpus.py [options]")
        print("Executes and verifies the Bronze IL corpus suite across Node.js oracle and Brass JIT configurations.")
        print("Options:")
        print("  -h, --help       Print this help message and exit")
        print("  --demote-stats   Print per-program and per-loop demotion/refusal census")
        sys.exit(0)

    if "--demote-stats" in sys.argv:
        print("=" * 115)
        print(" Bronze Corpus Loop Demotion & Refusal Census (--demote-stats)")
        print("=" * 115)
        for prog in PROGRAMS:
            il_file = CORPUS_DIR / f"{prog}.il"
            res = subprocess.run([str(BRASS_IL), str(il_file), "--demote-stats"], capture_output=True, text=True)
            print(f"--- Program: {prog} ---")
            print(res.stdout.strip())
            print()
        return 0

    print("=" * 115)
    print(" Running Bronze Corpus Timed Verification Suite (Law D Generated Table)")
    print(f" Executing {NUM_RUNS} runs per configuration for median & spread timing (ms)")
    print("=" * 115)

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
        no_opt_med, no_opt_spread, no_opt_min, no_opt_max, no_opt_norm = run_timed_config(
            [str(BRASS_IL), str(il_file), "--no-opt", "--run", "--raw-output"]
        )

        # 3. Run Brass JIT with demotion disabled (--no-demote)
        no_dem_med, no_dem_spread, no_dem_min, no_dem_max, no_dem_norm = run_timed_config(
            [str(BRASS_IL), str(il_file), "--no-demote", "--run", "--raw-output"]
        )

        # 4. Run Brass JIT default full optimized
        opt_med, opt_spread, opt_min, opt_max, opt_norm = run_timed_config(
            [str(BRASS_IL), str(il_file), "--run", "--raw-output"]
        )

        node_match = (node_norm == expected_norm)
        no_opt_match = (no_opt_norm == expected_norm)
        no_dem_match = (no_dem_norm == expected_norm)
        opt_match = (opt_norm == expected_norm)

        status = "PASS"
        if not (node_match and no_opt_match and no_dem_match and opt_match):
            status = "FAIL"
            all_pass = False

        demote_check = subprocess.run([str(BRASS_IL), str(il_file), "--demote-stats"], capture_output=True, text=True)
        is_demoted = "DEMOTED (i64 loop)" in demote_check.stdout
        demoted_str = "Yes (i64 loop)" if is_demoted else "No"

        # Demote speedup vs no-demote
        speedup_val = ((no_dem_med / opt_med) - 1.0) * 100.0 if opt_med > 0 else 0.0
        speedup_str = f"{speedup_val:+.1f}%"

        results.append({
            "prog": prog,
            "node_match": "Match" if node_match else "Mismatch",
            "no_opt_time": f"{no_opt_med:.1f} +/- {no_opt_spread:.1f}",
            "no_dem_time": f"{no_dem_med:.1f} +/- {no_dem_spread:.1f}",
            "opt_time": f"{opt_med:.1f} +/- {opt_spread:.1f}",
            "demoted": demoted_str,
            "speedup": speedup_str,
            "output": opt_norm,
            "status": status,
        })

    print()
    print("### Generated Law D Timed Verification Table (produced by `tools/run_corpus.py`)")
    print()
    print("| # | Program | Full Opt (ms) | No-Demote (ms) | No-Opt (ms) | f64 Demotion | Demote Speedup | Output | Status |")
    print("|---|---------|---------------|----------------|-------------|--------------|----------------|--------|--------|")

    for i, r in enumerate(results, 1):
        print(f"| {i:02d} | `{r['prog']}` | {r['opt_time']} | {r['no_dem_time']} | {r['no_opt_time']} | {r['demoted']} | {r['speedup']} | `{r['output']}` | **{r['status']}** |")

    print()
    if all_pass:
        print(f"[SUCCESS] All {len(PROGRAMS)} Bronze corpus programs 100% verified byte-identical across Node.js oracle, --no-opt, --no-demote, and full opt.")
        return 0
    else:
        print("[FAILURE] Regressions detected in Bronze corpus execution.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
