param(
  [string]$Root = '',
  [string]$V22 = '',
  [string]$Snapshots = '',
  [string]$TransportConfig = '',
  [string]$SensitivityConfig = '',
  [string]$Stages = '5,6',
  [int]$Bootstrap = 4096,
  [int]$Workers = 16,
  [switch]$AllowExploratoryTransport,
  [switch]$AcceptDifferentV22Hash
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Root) {
  $items = @(Get-ChildItem -LiteralPath (Join-Path $env:USERPROFILE 'Desktop') -Directory -ErrorAction Stop)
  foreach ($item in $items) {
    if (Test-Path -LiteralPath (Join-Path $item.FullName 'V150_FINAL_REAL_3D\reference_surface.vtp')) {
      $Root = $item.FullName; break
    }
  }
}
if (-not $Root) { throw 'RESEARCH_ROOT_NOT_FOUND: supply -Root explicitly' }
$env:OPENBLAS_NUM_THREADS = '8'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:NUMEXPR_NUM_THREADS = '8'
$py = Join-Path $env:LOCALAPPDATA 'HGPS_CFD_ASCII\VTK_DIAGNOSTIC_PYENV\Scripts\python.exe'
if (!(Test-Path -LiteralPath $py)) { throw ('VTK_PYTHON_NOT_FOUND: ' + $py) }
& $py -c 'import numpy,pandas,scipy,vtk,matplotlib; print("SCIENTIFIC_DEPENDENCIES_PASS")'
if ($LASTEXITCODE -ne 0) {throw 'DEPENDENCIES_MISSING: python -m pip install numpy pandas scipy vtk matplotlib psutil'}
$out = Join-Path $Root 'FIG58_DEEP_RESULTS'
if (!(Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Force -Path $out | Out-Null }
$arglist = @((Join-Path $here 'run.py'),'--root',$Root,'--out',$out,'--stages',$Stages,'--v22-bootstrap',"$Bootstrap",'--fig8-workers',"$Workers",'--fig5-workers',"$Workers")
if ($V22) {$arglist += @('--v22',$V22)}
if ($Snapshots) {$arglist += @('--snapshots') + @($Snapshots.Split(';') | Where-Object { $_ })}
if ($TransportConfig) {$arglist += @('--fig7-config',$TransportConfig)}
if ($SensitivityConfig) {$arglist += @('--fig8-config',$SensitivityConfig)}
if ($AllowExploratoryTransport) {$arglist += '--fig7-exploratory'}
if ($AcceptDifferentV22Hash) {$arglist += '--accept-new-v22'}
Write-Host 'FIGURE_5_8_DEEP_SOURCE_ANALYSIS_START' -ForegroundColor Cyan
Write-Host ('ROOT=' + $Root)
Write-Host ('OUTPUT=' + $out)
$log = Join-Path $out 'RUN_CONSOLE_TRANSCRIPT.log'
Start-Transcript -Path $log -Append | Out-Null
try {
  & $py @arglist
  $code=$LASTEXITCODE
  Write-Host ('FIG58_PROCESS_EXIT=' + $code)
} finally { Stop-Transcript | Out-Null }
if ($code -ne 0) {Write-Host 'CHECK_MASTER_INPUT_AND_SOURCE_AUDIT_JSON_FOR_STAGE_HOLDS' -ForegroundColor Yellow}
exit $code
