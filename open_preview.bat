@echo off
rem One-click: ensure the preview service is running, then open the browser.
cd /d "%~dp0"
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; if (-not $c) { Write-Host 'Starting preview service (about 40s on first start)...'; Start-Process -WindowStyle Hidden -FilePath '%~dp0keepalive_preview.bat'; for ($i = 0; $i -lt 90; $i++) { Start-Sleep -Seconds 1; if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { break } } }"
start "" "http://127.0.0.1:8000"
