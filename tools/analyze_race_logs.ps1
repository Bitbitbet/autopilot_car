param(
    [Parameter(Mandatory = $true)][string]$RunDirectory,
    [string]$OutputDirectory,
    [string]$Annotations
)
$ErrorActionPreference = 'Stop'
$RuntimePython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (-not (Test-Path -LiteralPath $RuntimePython)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) { throw '没有找到现有 Python 运行时；此脚本不会安装组件。' }
    $RuntimePython = $PythonCommand.Source
}
$ReportScript = Join-Path $PSScriptRoot 'race_log_report.py'
$ReportArguments = @($ReportScript, $RunDirectory)
if ($OutputDirectory) { $ReportArguments += @('--output', $OutputDirectory) }
if ($Annotations) { $ReportArguments += @('--annotations', $Annotations) }
& $RuntimePython @ReportArguments
exit $LASTEXITCODE
