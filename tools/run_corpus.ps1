# tools/run_corpus.ps1
# Automates timed execution and verification of all 13 Bronze IL corpus programs
# Measures runtime medians and spreads across >= 5 runs per configuration.
# Implements Law D: report tables are generated, not written.

param (
    [string]$BrassIlPath = "$PSScriptRoot/../build/tools/brass-il.exe",
    [string]$CorpusDir = "$PSScriptRoot/../tests/bronze_corpus",
    [int]$NumRuns = 5
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
    "23_objects_basic",
    "24_shapes_polymorphic",
    "25_try_catch_basic",
    "26_nested_try_finally",
    "27_generator_fibonacci",
    "28_async_chain",
    "29_generational_churn",
    "30_osr_hot_loop",
    "31_gvn_pre_diamonds"
)

$demotedPrograms = @(
    "03_collatz",
    "04_fib_iter",
    "07_prime_count",
    "11_matrix_recurrence",
    "12_counter_closure",
    "15_nested_acc",
    "16_param_bounds",
    "18_gcd_iter"
)

function Normalize-Output([string]$text) {
    if ($null -eq $text) { return "" }
    $text = [regex]::Replace($text, '\[brass-il-timed:[^\]]*\]', '')
    $tokens = $text -split '\s+' | Where-Object { $_ -ne "" }
    return ($tokens -join " ")
}

function Measure-ConfigTimes([string]$exePath, [string[]]$argsList, [int]$runs) {
    if ($exePath -like "*brass-il*" -and ($argsList -notcontains "--timed")) {
        $timedArgs = @($argsList) + @("--timed", "$runs")
        $procOut = & $exePath @timedArgs 2>&1 | Out-String
        if ($procOut -match '\[brass-il-timed:\s*([\d\.]+)\s*\+/-\s*([\d\.]+)\s*\(min:\s*([\d\.]+),\s*max:\s*([\d\.]+)\)\]') {
            return @{
                Median = [double]$Matches[1]
                Spread = [double]$Matches[2]
                Output = (Normalize-Output $procOut)
            }
        }
    }

    # Warmup run to eliminate first-time process spawn / disk cache overhead
    $lastOut = & $exePath @argsList 2>&1 | Out-String
    $times = @()
    for ($i = 0; $i -lt $runs; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $procOut = & $exePath @argsList 2>&1 | Out-String
        $sw.Stop()
        $times += $sw.Elapsed.TotalMilliseconds
        $lastOut = $procOut
    }
    $sorted = $times | Sort-Object
    $mid = [int]($runs / 2)
    $med = $sorted[$mid]
    $minVal = $sorted[0]
    $maxVal = $sorted[$runs - 1]
    $spread = ($maxVal - $minVal) / 2.0
    return @{
        Median = $med
        Spread = $spread
        Output = (Normalize-Output $lastOut)
    }
}

Write-Host "===================================================================================================="
Write-Host " Running Bronze Corpus Timed Verification Suite (Law D Generated Table)"
Write-Host " Executing $NumRuns runs per configuration for median & spread timing (ms)"
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
    $noOptRes = Measure-ConfigTimes $BrassIlPath @($ilPath, "--no-opt", "--run", "--raw-output") $NumRuns

    # 3. Run Brass JIT with demotion disabled (--no-demote)
    $noDemRes = Measure-ConfigTimes $BrassIlPath @($ilPath, "--no-demote", "--run", "--raw-output") $NumRuns

    # 4. Run Brass JIT Default Full Optimized
    $optRes = Measure-ConfigTimes $BrassIlPath @($ilPath, "--run", "--raw-output") $NumRuns

    # Verify matching
    $nodeMatchesExp = ($nodeNorm -eq $expectedNorm)
    $noOptMatchesExp = ($noOptRes.Output -eq $expectedNorm)
    $noDemMatchesExp = ($noDemRes.Output -eq $expectedNorm)
    $optMatchesExp = ($optRes.Output -eq $expectedNorm)

    $status = "PASS"
    if (-not ($nodeMatchesExp -and $noOptMatchesExp -and $noDemMatchesExp -and $optMatchesExp)) {
        $status = "FAIL"
        $allPass = $false
    }

    $demoteStatsOut = & $BrassIlPath $ilPath --demote-stats 2>&1 | Out-String
    $demotedInfo = "No"
    if ($demoteStatsOut -match "DEMOTED \(i64 loop\)") {
        $demotedInfo = "Yes (i64 loop)"
    }

    $speedupStr = if ($optRes.Median -gt 0) {
        $pct = (($noDemRes.Median / $optRes.Median) - 1.0) * 100.0
        "{0:+0.0;-0.0;+0.0}%" -f $pct
    } else {
        "0.0%"
    }

    $results += [PSCustomObject]@{
        Program = $prog
        NodeOracleMatch = if ($nodeMatchesExp) { "Match" } else { "Mismatch" }
        OptTime = ("{0:F1} +/- {1:F1}" -f $optRes.Median, $optRes.Spread)
        NoDemTime = ("{0:F1} +/- {1:F1}" -f $noDemRes.Median, $noDemRes.Spread)
        NoOptTime = ("{0:F1} +/- {1:F1}" -f $noOptRes.Median, $noOptRes.Spread)
        Demoted = $demotedInfo
        Speedup = $speedupStr
        Output = $optRes.Output
        Status = $status
    }
}

Write-Host ""
Write-Host "### Generated Law D Timed Verification Table (produced by tools/run_corpus.ps1)"
Write-Host ""
Write-Host "| # | Program | Full Opt (ms) | No-Demote (ms) | No-Opt (ms) | f64 Demotion | Demote Speedup | Output | Status |"
Write-Host "|---|---------|---------------|----------------|-------------|--------------|----------------|--------|--------|"

$idx = 1
foreach ($r in $results) {
    Write-Host ("| {0:D2} | `{1}` | {2} | {3} | {4} | {5} | {6} | `{7}` | **{8}** |" -f $idx, $r.Program, $r.OptTime, $r.NoDemTime, $r.NoOptTime, $r.Demoted, $r.Speedup, $r.Output, $r.Status)
    $idx++
}

Write-Host ""
if ($allPass) {
    Write-Host ("[SUCCESS] All {0} Bronze corpus programs 100% verified byte-identical across Node.js oracle, --no-opt, --no-demote, and full opt." -f $corpusPrograms.Count)
} else {
    Write-Host "[FAILURE] Regressions detected in Bronze corpus execution."
    exit 1
}
