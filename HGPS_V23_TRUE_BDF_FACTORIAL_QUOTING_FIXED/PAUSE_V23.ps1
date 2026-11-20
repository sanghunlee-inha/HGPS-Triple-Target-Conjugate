param([string]$Root=(Join-Path $env:USERPROFILE 'Desktop\２６。０８。３０ 소아조로증 ＡＩ 신약개발 연구'))
$out=Join-Path $Root 'HGPS_V23_TRUE_BDF_FACTORIAL_RESULTS'
New-Item -ItemType File -Path (Join-Path $out 'REQUEST_STOP.flag') -Force | Out-Null
Write-Host 'Pause requested; current BDF workers will finish safely before shutdown.'
