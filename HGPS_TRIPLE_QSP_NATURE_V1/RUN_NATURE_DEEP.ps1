param(
    [ValidateSet('all','audit','assays','v22','qsp','calibrate','self-test')]
    [string]$Mode='all',
    [int]$SobolM=10,
    [int]$Workers=18,
    [int]$Bootstraps=2000,
    [string]$ObservedAssays='',
    [string]$ObservedTimecourse='',
    [string]$V22='',
    [string]$V22Mapping='',
    [string]$ParameterFile='',
    [switch]$InstallDependencies
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$Dir=Split-Path -Parent $MyInvocation.MyCommand.Path
$Root=Join-Path $env:LOCALAPPDATA 'HGPS_CFD_ASCII'
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Out=Join-Path $Root "TRIPLE_CONJUGATE_NATURE_V1_$Stamp"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Py=(Get-Command python -ErrorAction Stop).Source
Write-Host "PIPELINE_SOURCE=$Dir"
Write-Host "PYTHON=$Py"
Write-Host "OUTPUT=$Out"
# Explicit isolation of BLAS threads avoids oversubscribing 18 concurrent BDF workers.
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
if ($InstallDependencies) {
    & $Py -m pip install -r (Join-Path $Dir 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'DEPENDENCY_INSTALL_FAILED' }
}
& $Py -c 'import numpy,pandas,scipy,matplotlib; print("DEPENDENCY_IMPORT_PASS")'
if ($LASTEXITCODE -ne 0) {
    throw 'Missing Python dependencies; re-run with -InstallDependencies if network/package policy permits.'
}
& $Py -m py_compile (Join-Path $Dir 'qsp_engine.py') (Join-Path $Dir 'analysis_pipeline.py') (Join-Path $Dir 'calibration.py')
if ($LASTEXITCODE -ne 0) { throw 'PYTHON_COMPILE_FAILED' }
Write-Host 'SOURCE_COMPILE_PASS'
$ArgsForPython=@((Join-Path $Dir 'analysis_pipeline.py'),'--mode',$Mode,
  '--home',$env:USERPROFILE,'--out',$Out,
  '--workers',[string]$Workers,'--sobol-m',[string]$SobolM,
  '--bootstraps',[string]$Bootstraps)
if ($ObservedAssays) { $ArgsForPython+=@('--assays',$ObservedAssays) }
if ($ObservedTimecourse) { $ArgsForPython+=@('--calibration-csv',$ObservedTimecourse) }
if ($V22) { $ArgsForPython+=@('--v22',$V22) }
if ($V22Mapping) { $ArgsForPython+=@('--v22-mapping',$V22Mapping) }
if ($ParameterFile) { $ArgsForPython+=@('--parameter-file',$ParameterFile) }
& $Py @ArgsForPython
$Exit=$LASTEXITCODE
Write-Host "NATURE_TRIPLE_V1_EXIT=$Exit"
Write-Host "NATURE_TRIPLE_V1_OUTPUT=$Out"
if ($Exit -ne 0) { throw "MODEL_HOLD_OR_PIPELINE_ERROR exit=$Exit; inspect RUN_STATUS.json" }
$Status=Join-Path $Out 'RUN_STATUS.json'
if (-not (Test-Path $Status)) { throw 'RUN_STATUS_MISSING' }
Write-Host 'PIPELINE_COMPLETED_CHECK_EVIDENCE_GATES_NOT_AUTOMATIC_PUBLICATION_APPROVAL'
