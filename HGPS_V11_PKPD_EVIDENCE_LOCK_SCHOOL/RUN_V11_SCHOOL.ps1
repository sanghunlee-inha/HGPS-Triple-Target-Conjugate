param(
 [ValidateSet('audit','all')][string]$Mode='audit',
 [int]$Dpi=600
)
$ErrorActionPreference='Stop'
$Project=Split-Path -Parent $MyInvocation.MyCommand.Path
$Root=Split-Path -Parent $Project
$V9=Join-Path $Root 'HGPS_P1_TBK1_MIR4651_PKPD_V9_RECEPTOR_FIXED_SCHOOL'
$V10=Join-Path $Root 'HGPS_P1_TBK1_MIR4651_PKPD_V10_BREAKPOINT_SCHOOL'
$Py=Join-Path $env:USERPROFILE 'anaconda3\envs\hgps_twin\python.exe'
$Out=Join-Path $Project 'outputs\V11_SOURCE_LOCKED'
foreach($p in @($Py,$V9,$V10)){if(-not(Test-Path -LiteralPath $p)){throw "V11_MISSING_PATH: $p"}}
$env:PYTHONUTF8='1'
Write-Output "V11_PYTHON=$Py"
Write-Output "V11_V9_READONLY=$V9"
Write-Output "V11_V10_READONLY=$V10"
Write-Output "V11_OUTPUT=$Out"
& $Py -u (Join-Path $Project 'src\v11_main.py') --mode $Mode --v9-project $V9 --v10-project $V10 --out $Out --dpi $Dpi
if($LASTEXITCODE -ne 0){throw "V11_AUDIT_OR_REPLAY_FAILED_$LASTEXITCODE"}
if($Mode -eq 'all'){
 $Gate=Join-Path $Out 'V11_SCIENTIFIC_GATE.json'
 if(-not(Test-Path -LiteralPath $Gate)){throw 'V11_GATE_MISSING'}
 Write-Output "V11_RESULT_GATE=$Gate"
 Write-Output "V11_MAIN_PNG=$(Join-Path $Out 'V11_MAIN_FIGURE_SOURCE_VERIFIED.png')"
}
