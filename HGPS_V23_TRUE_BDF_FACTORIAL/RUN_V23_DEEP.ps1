param(
  [string]$Root = (Join-Path $env:USERPROFILE 'Desktop\２６。０８。３０ 소아조로증 ＡＩ 신약개발 연구'),
  [string]$Base = '',
  [string]$Protocol = '',
  [int]$CasesPerSeed = 2048,
  [int]$Workers = 16,
  [int]$RefinementEvery = 128,
  [switch]$RetryFailed
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = (Get-Command python -ErrorAction Stop).Source
$out = Join-Path $Root 'HGPS_V23_TRUE_BDF_FACTORIAL_RESULTS'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
Write-Host 'REAL ORIGINAL BDF DEEP RUN (not statistics-only)' -ForegroundColor Cyan
Write-Host ('RESULTS: '+$out)
Write-Host ('Planned cases: '+(5*$CasesPerSeed)+' ; minimum ODE trajectories: '+(5*$CasesPerSeed*26))
Write-Host 'Scientific scope: synthetic scenarios; NOT patient treatment efficacy' -ForegroundColor Yellow
# Invoke a Python FILE, not `python -c`: Windows PowerShell 5.1 strips nested
# quote characters when marshalling native executable arguments.
$checkScript = Join-Path $here 'V23_PYTHON_ENV_CHECK.py'
& $py $checkScript
if ($LASTEXITCODE -ne 0) {
    throw ('V23 Python dependency check failed. Interpreter: ' + $py + '. See the Python error above; do not reinstall packages unless an actual ModuleNotFoundError appears.')
}
$argsList = @('-u', (Join-Path $here 'run_v23_true_bdf.py'), '--output', $out, '--cases-per-seed', "$CasesPerSeed", '--workers', "$Workers", '--refinement-every', "$RefinementEvery")
if ($Base -ne '') { $argsList += @('--base', $Base) }
if ($Protocol -ne '') { $argsList += @('--protocol', $Protocol) }
if ($RetryFailed) { $argsList += '--retry-failed' }
& $py @argsList
$rc = $LASTEXITCODE
if ($rc -eq 75) { Write-Host 'PAUSED; remove REQUEST_STOP.flag, rerun same script.'; exit 75 }
if ($rc -ne 0) { Write-Host ('HOLD/ERROR: actual physics not complete. Exit='+$rc) -ForegroundColor Red; exit $rc }
Write-Host 'FULL BDF run complete. Starting independent post-analysis...' -ForegroundColor Green
& $py -u (Join-Path $here 'summarize_v23.py') --output $out --include-0d
if ($LASTEXITCODE -ne 0) { throw 'BDF completed, but post-analysis requires review; check V23_DIRECT_BDF_FINAL_GATE.json' }
Write-Host ('COMPLETE: '+$out) -ForegroundColor Green
