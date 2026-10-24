#!/usr/bin/env python3
"""
Export the BVSolver #153 regression corpus to SMT-LIB2 files via test_smt_equiv
and cross-check every query against an external Z3 binary (if installed).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = Path(os.environ.get("BRASS_BUILD_DIR", repo_root / "build"))
    test_bin = build_dir / "tests" / "test_smt_equiv"
    if not test_bin.exists():
        print(f"error: {test_bin} not found; build test_smt_equiv first", file=sys.stderr)
        return 1

    z3_bin = shutil.which("z3")

    with tempfile.TemporaryDirectory(prefix="brass_smt2_corpus_") as tmpdir:
        env = os.environ.copy()
        env["BRASS_DUMP_SMT2_CORPUS_DIR"] = tmpdir
        cmd = [
            str(test_bin),
            "--exact=BVSolver #153 - Bitvector Identity & Edge-Case Regression Corpus (with SMT-LIB2 Export Check)",
        ]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr, file=sys.stderr)
            return res.returncode

        smt2_files = sorted(Path(tmpdir).glob("*.smt2"))
        if not smt2_files:
            print("error: no .smt2 files exported by corpus test", file=sys.stderr)
            return 1

        print(f"Exported {len(smt2_files)} SMT-LIB2 corpus queries.")
        if not z3_bin:
            print("z3 not found on PATH; skipping external solver cross-check (SMT-LIB2 export verified).")
            return 0

        for smt2_path in smt2_files:
            text = smt2_path.read_text(encoding="utf-8")
            expected = None
            for line in text.splitlines():
                if "EXPECTED: unsat" in line:
                    expected = "unsat"
                    break
                if "EXPECTED: sat" in line:
                    expected = "sat"
                    break
            if expected is None:
                print(f"error: missing EXPECTED header in {smt2_path.name}", file=sys.stderr)
                return 1

            z3_res = subprocess.run([z3_bin, str(smt2_path)], capture_output=True, text=True)
            first_line = z3_res.stdout.strip().splitlines()[0] if z3_res.stdout.strip() else ""
            if first_line != expected:
                print(
                    f"FAIL {smt2_path.name}: expected {expected}, z3 returned {first_line!r}",
                    file=sys.stderr,
                )
                return 1
            print(f"  [Z3 OK] {smt2_path.name}: {first_line}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
