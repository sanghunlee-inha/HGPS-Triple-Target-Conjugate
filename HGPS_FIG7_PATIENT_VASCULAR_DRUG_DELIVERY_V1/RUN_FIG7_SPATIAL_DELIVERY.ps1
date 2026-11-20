param(
  [ValidateRange(64,20000)][int]$Samples = 1024,
  [string]$V10Prefix = '',
  [string]$RoiCsv = '',
  [switch]$NoFigure
)
$ErrorActionPreference = 'Stop'
$Root = Join-Path $env:LOCALAPPDATA 'HGPS_CFD_ASCII'
if (-not (Test-Path -LiteralPath $Root)) { throw "SOURCE_ROOT_NOT_FOUND=$Root" }
$Py = Join-Path $env:USERPROFILE 'anaconda3\python.exe'
if (-not (Test-Path -LiteralPath $Py)) { $Py = (Get-Command python -ErrorAction Stop).Source }
$Script = Join-Path $PSScriptRoot 'fig7_delivery.py'
if (-not (Test-Path -LiteralPath $Script)) { throw "PY_SCRIPT_MISSING=$Script" }
if (-not $V10Prefix) {
    $Source = Get-ChildItem -LiteralPath $Root -Directory -Filter 'V16_FIG6_V10_TETGEN_NATURE_*' |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { Join-Path $_.FullName 'quality\patient_11patch.1.node' } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if (-not $Source) {
        throw 'PATIENT_V10_QUALITY_SOURCE_NOT_FOUND: supply -V10Prefix path without .node'
    }
    $V10Prefix = $Source.Substring(0,$Source.Length-5)
}
if (-not (Test-Path -LiteralPath ($V10Prefix + '.node'))) { throw "V10_NODE_MISSING=$V10Prefix" }
if (-not (Test-Path -LiteralPath ($V10Prefix + '.face'))) { throw "V10_FACE_MISSING=$V10Prefix" }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$Out = Join-Path $Root "FIG7_DELIVERY_PATIENT_SURFACE_$Stamp"
$Args = @($Script,'--prefix',$V10Prefix,'--out',$Out,'--samples',"$Samples")
if ($RoiCsv) { $Args += @('--roi-csv',$RoiCsv) }
if ($NoFigure.IsPresent) { $Args += '--no-figure' }
Write-Host "VERSION=HGPS_FIG7_PATIENT_SURFACE_DELIVERY_SCENARIO_V1"
Write-Host "PYTHON=$Py"
Write-Host "PATIENT_V10_SOURCE=$V10Prefix"
Write-Host "OUTPUT=$Out"
& $Py @Args
$rc=$LASTEXITCODE
Write-Host "FIG7_PROCESS_EXIT=$rc"
if ($rc -ne 0) { throw "FIG7_EXECUTION_FAILED rc=$rc OUTPUT=$Out" }
Write-Host "FIG7_SUMMARY=$Out\SUMMARY.txt"
Write-Host "FIG7_PNG=$Out\FIG7_PATIENT_SURFACE_DELIVERY_CONDITIONAL_a-f_600dpi.png"
