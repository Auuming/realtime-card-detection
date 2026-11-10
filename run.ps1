Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython -PathType Leaf)) {
    throw 'Project Python environment is missing. Run .\setup.ps1 first, then run .\run.ps1 again.'
}
& $projectPython app.py @args
