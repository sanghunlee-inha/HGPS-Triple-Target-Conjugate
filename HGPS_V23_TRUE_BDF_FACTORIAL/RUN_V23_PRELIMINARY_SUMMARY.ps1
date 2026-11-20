param([string]$Root=(Join-Path $env:USERPROFILE 'Desktop\２６。０８。３０ 소아조로증 ＡＩ 신약개발 연구'))
$here=Split-Path -Parent $MyInvocation.MyCommand.Path
$out=Join-Path $Root 'HGPS_V23_TRUE_BDF_FACTORIAL_RESULTS'
& python -u (Join-Path $here 'summarize_v23.py') --output $out --partial
exit $LASTEXITCODE
