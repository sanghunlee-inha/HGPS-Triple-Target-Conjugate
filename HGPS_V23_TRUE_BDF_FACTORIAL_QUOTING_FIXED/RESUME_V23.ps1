param([string]$Root=(Join-Path $env:USERPROFILE 'Desktop\２６。０８。３０ 소아조로증 ＡＩ 신약개발 연구'),[string]$Base='',[string]$Protocol='',[int]$CasesPerSeed=2048,[int]$Workers=16)
Remove-Item -LiteralPath (Join-Path $Root 'HGPS_V23_TRUE_BDF_FACTORIAL_RESULTS\REQUEST_STOP.flag') -ErrorAction SilentlyContinue
& (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'RUN_V23_DEEP.ps1') -Root $Root -Base $Base -Protocol $Protocol -CasesPerSeed $CasesPerSeed -Workers $Workers
exit $LASTEXITCODE
