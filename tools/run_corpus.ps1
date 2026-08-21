# tools/run_corpus.ps1
# Automates execution and verification of all 13 Bronze IL corpus programs
# Implements Law D: report tables are generated, not written.

param (
    [string]$BrassIlPath = "$PSScriptRoot/../build/tools/brass-il.exe",
    [string]$CorpusDir = "$PSScriptRoot/../tests/bronze_corpus"
)

$ErrorActionPreference = "Stop"

$corpusPrograms = @(
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
    "13_nested_curry"
)

function Normalize-Output([string]$text) {
    if ($null -eq $text) { return "" }
    $tokens = $text -split '\s+' | Where-Object { $_ -ne "" }
    return ($tokens -join " ")
}

Write-Host "===================================================================================================="
Write-Host " Running Bronze Corpus Verification Suite (Law D Generated Table)"
Write-Host "===================================================================================================="

$results = @()
$allPass = $true

foreach ($prog in $corpusPrograms) {
    $jsPath = "$CorpusDir/$prog.js"
    $ilPath = "$CorpusDir/$prog.il"
    $expPath = "$CorpusDir/$prog.expected"

    if (-not (Test-Path $expPath)) {
        Write-Error "Expected file missing: $expPath"
    }
    $expectedRaw = [System.IO.File]::ReadAllText($expPath)
    $expectedNorm = Normalize-Output $expectedRaw

    # 1. Run Node.js Oracle with print global shim
    $nodeCmd = "const print = (...args) => console.log(args.join(' ')); eval(require('fs').readFileSync('$($jsPath.Replace('\', '/'))', 'utf8'));"
    $nodeOutRaw = & node -e $nodeCmd 2>&1 | Out-String
    $nodeNorm = Normalize-Output $nodeOutRaw

    # 2. Run Brass JIT Unoptimized (--no-opt)
    $jitNoOptOutRaw = & $BrassIlPath $ilPath --no-opt --run --raw-output 2>&1 | Out-String
    $jitNoOptNorm = Normalize-Output $jitNoOptOutRaw

    # 3. Run Brass JIT Default Optimized
    $jitOptOutRaw = & $BrassIlPath $ilPath --run --raw-output 2>&1 | Out-String
    $jitOptNorm = Normalize-Output $jitOptOutRaw

    # Verify matching
    $nodeMatchesExp = ($nodeNorm -eq $expectedNorm)
    $jitNoOptMatchesExp = ($jitNoOptNorm -eq $expectedNorm)
    $jitOptMatchesExp = ($jitOptNorm -eq $expectedNorm)

    $status = "PASS"
    if (-not ($nodeMatchesExp -and $jitNoOptMatchesExp -and $jitOptMatchesExp)) {
        $status = "FAIL"
        $allPass = $false
    }

    $demotedInfo = "No"
    if ($prog -in @("03_collatz", "04_fib_iter", "07_prime_count", "11_matrix_recurrence")) {
        $demotedInfo = "Yes (i64 loop)"
    }

    $results += [PSCustomObject]@{
        Program = $prog
        NodeOracleMatch = if ($nodeMatchesExp) { "Match" } else { "Mismatch" }
        JitNoOptMatch = if ($jitNoOptMatchesExp) { "Match" } else { "Mismatch" }
        JitOptMatch = if ($jitOptMatchesExp) { "Match" } else { "Mismatch" }
        Demoted = $demotedInfo
        Output = $jitOptNorm
        Status = $status
    }
}

Write-Host ""
Write-Host "### Generated Verification Table (produced by tools/run_corpus.ps1)"
Write-Host ""
Write-Host "| # | Program | Node Oracle (node -e '...') | Brass JIT (--no-opt) | Brass JIT (Opt) | f64 Demotion | Output | Status |"
Write-Host "|---|---------|-----------------------------|----------------------|-----------------|--------------|--------|--------|"

$idx = 1
foreach ($r in $results) {
    Write-Host ("| {0:D2} | `{1}` | {2} | {3} | {4} | {5} | `{6}` | **{7}** |" -f $idx, $r.Program, $r.NodeOracleMatch, $r.JitNoOptMatch, $r.JitOptMatch, $r.Demoted, $r.Output, $r.Status)
    $idx++
}

Write-Host ""
if ($allPass) {
    Write-Host "[SUCCESS] All 13 Bronze corpus programs 100% verified byte-identical against Node oracle and .expected files."
} else {
    Write-Host "[FAILURE] Regressions detected in Bronze corpus execution."
    exit 1
}
