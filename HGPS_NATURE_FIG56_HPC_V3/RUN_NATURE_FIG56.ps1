#Requires -Version 5.1
param(
    [string]$ResearchRoot = '',
    [int]$Threads = 24,
    [string]$MmgExe = '',
    [string]$V22Csv = '',
    [string]$MolecularCsv = '',
    [string]$AblationCsv = '',
    [string]$MeasuredWallCsv = '',
    [string]$WallAnalysisPlan = '',
    [string]$CfdMetricsCsv = '',
    [string]$CfdEvidenceManifest = '',
    [string]$VerifiedCfdCasesManifest = '',
    [string]$ReferencePhysics = '',
    [ValidateSet('all','v22','mesh','cfd')][string]$Only = 'all',
    [switch]$UseCuda,
    [switch]$StrictV22SHA
)
$ErrorActionPreference = 'Stop'
if (-not $ResearchRoot) {
    $ResearchRoot = Join-Path $env:USERPROFILE 'Desktop\２６。０８。３０ 소아조로증 ＡＩ 신약개발 연구'
}
if ($Threads -lt 1 -or $Threads -gt [Environment]::ProcessorCount) {
    throw 'HOLD: MPI ranks must not exceed available logical CPUs'
}
if (!(Test-Path -LiteralPath $ResearchRoot)) {throw "HOLD: Research root missing: $ResearchRoot"}
$Package = Split-Path -Parent $MyInvocation.MyCommand.Path
$Ascii = Join-Path $env:LOCALAPPDATA 'HGPS_CFD_ASCII\NATURE_FIG56_HPC_V3'
$Code = Join-Path $Ascii 'CODE'
$Work = Join-Path $Ascii 'WORK'
New-Item -ItemType Directory -Force $Code,$Work | Out-Null
# LiteralPath wildcard is not expanded on Windows PowerShell; use explicit file enumeration.
Get-ChildItem -LiteralPath (Join-Path $Package 'code') -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Code $_.Name) -Force
}
$EnvV = Join-Path $env:LOCALAPPDATA 'HGPS_CFD_ASCII\VTK_DIAGNOSTIC_PYENV\Scripts\python.exe'
$Py = $EnvV
if (!(Test-Path -LiteralPath $Py)) {
    $BasePy = (Get-Command python -ErrorAction Stop).Source
    & $BasePy -m venv (Join-Path $Ascii 'PYENV')
    if ($LASTEXITCODE -ne 0) {throw 'HOLD: Python virtualenv creation failed'}
    $Py = Join-Path $Ascii 'PYENV\Scripts\python.exe'
}
& $Py -c 'import numpy,scipy,meshio,matplotlib,mmgpy'
if ($LASTEXITCODE -ne 0) {
    Write-Host '[INSTALL] Scientific dependencies for isolated audit environment'
    & $Py -m pip install --disable-pip-version-check 'meshio==5.3.5' scipy matplotlib 'mmgpy==0.18.0'
    if ($LASTEXITCODE -ne 0) {throw 'HOLD: Scientific package installation failed'}
}
& $Py -c 'import numpy,scipy,meshio,matplotlib,mmgpy'
if ($LASTEXITCODE -ne 0) {throw 'HOLD: Python scientific modules unavailable'}
# Maintain BLAS single-thread per MPI worker, avoiding 24x24 oversubscription.
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:PYTHONUTF8 = '1'
$GpuPy = ''
if ($UseCuda) {
    $BasePy = (Get-Command python -ErrorAction Stop).Source
    & $BasePy -c 'import torch,numpy,matplotlib;assert torch.cuda.is_available()'
    if ($LASTEXITCODE -eq 0) { $GpuPy = $BasePy }
    else {Write-Host 'CUDA_SOURCE_GATE: HOLD (missing compatible torch/numpy/matplotlib)'}
}
if (!$MmgExe) {
    $MmgCandidate = Join-Path (Split-Path -Parent $Py) 'mmg3d.exe'
    if (Test-Path -LiteralPath $MmgCandidate) { $MmgExe = $MmgCandidate }
}
$Args = @((Join-Path $Code 'run_deep_workflow.py'), '--research-root', $ResearchRoot, '--stage-dir', $Work,
          '--mpi-ranks', [string]$Threads, '--only', $Only)
if ($MmgExe) { $Args += @('--mmg-exe', $MmgExe) }
if ($V22Csv) { $Args += @('--v22', $V22Csv) }
if ($MolecularCsv) { $Args += @('--molecular-csv', $MolecularCsv) }
if ($AblationCsv) { $Args += @('--ablation-csv', $AblationCsv) }
if ($MeasuredWallCsv) {$Args += @('--wall-data',$MeasuredWallCsv)}
if ($WallAnalysisPlan) {$Args += @('--wall-plan',$WallAnalysisPlan)}
if ($CfdMetricsCsv) { $Args += @('--cfd-metrics', $CfdMetricsCsv) }
if ($CfdEvidenceManifest) { $Args += @('--cfd-evidence-manifest', $CfdEvidenceManifest) }
if ($VerifiedCfdCasesManifest) { $Args += @('--cfd-cases-manifest', $VerifiedCfdCasesManifest) }
if ($ReferencePhysics) {$Args += @('--reference-physics', $ReferencePhysics)}
if ($UseCuda -and $GpuPy) {$Args += @('--use-cuda','--gpu-python', $GpuPy)}
if ($StrictV22SHA) {$Args += '--strict-v22-sha'}
$LocalLog = Join-Path $Work 'MASTER_WINDOWS_STDOUT.log'
Write-Host "WORK=$Work; MPI=$Threads; GPU=$GpuPy; MMG=$MmgExe"
Write-Host 'Running real source-based analysis; mesh and CFD physical holds are enforced.'
& $Py @Args 2>&1 | Tee-Object -FilePath $LocalLog
$RC = $LASTEXITCODE
$ResearchOut = Join-Path $ResearchRoot 'NATURE_FIG5_FIG6_CONJUGATE_HPC_V3'
New-Item -ItemType Directory -Force -Path $ResearchOut | Out-Null
foreach ($file in @('STUDY_STATUS.json','ALL_MESH_GATES.csv','MASTER_WINDOWS_STDOUT.log')) {
    $candidate = Join-Path $Work $file
    if (Test-Path -LiteralPath $candidate) {Copy-Item -LiteralPath $candidate -Destination (Join-Path $ResearchOut $file) -Force}
}
Write-Host "PROCESS_EXIT=$RC"
Write-Host "ASCII_WORK=$Work"
Write-Host "RESEARCH_AUDIT=$ResearchOut"
if ($RC -ne 0) {Write-Host 'HOLD: verify stage reports; not a completed therapeutic Figure 6.'}
exit $RC
