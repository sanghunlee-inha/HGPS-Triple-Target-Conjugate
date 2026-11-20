#Requires -Version 5.1
$ResearchFolder=Join-Path ([Environment]::GetFolderPath('Desktop')) '２６。０８。３０ 소아조로증 ＡＩ 신약개발 연구'
$Out=Join-Path $ResearchFolder 'V22_LONAFARNIB_DEEP_BDF_SOURCE_LOCKED'
$Progress=Join-Path $Out 'LIVE_PROGRESS.json'
if(Test-Path -LiteralPath $Progress){Get-Content -LiteralPath $Progress -Raw -Encoding UTF8}
else {Write-Host 'No BDF replay progress yet (0D audit may be running).'}
if(Test-Path -LiteralPath $Out){Start-Process explorer.exe -ArgumentList ('"'+$Out+'"')}
