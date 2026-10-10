# Start ThermaScope backend and frontend (two new windows)
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; py -m backend"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Write-Host "Started backend (8000) and frontend (5173) in separate windows."
