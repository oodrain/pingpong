[CmdletBinding(DefaultParameterSetName = 'Sample')]
param(
    [Parameter(ParameterSetName = 'Sample')]
    [ValidateSet('course50', 'course25')]
    [string]$Sample = 'course50',
    [Parameter(Mandatory = $true, ParameterSetName = 'Video')]
    [string]$Video,
    [string]$AnalysisRuntime,
    [switch]$Check,
    [string]$Output
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv-annotation\Scripts\python.exe'
$pythonw = Join-Path $root '.venv-annotation\Scripts\pythonw.exe'
$app = Join-Path $root 'development\video_studio\app.py'
$inputMode = $PSCmdlet.ParameterSetName
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $root 'workspace'
} elseif (-not [IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $root $Output
}
$Output = [IO.Path]::GetFullPath($Output)

function Invoke-Preflight([string]$inputPath, [string]$runtimePath) {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw 'Python environment missing. Follow course-workbench/README.md before starting.'
    }

    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    # Capture native stderr without Windows PowerShell 5.1 truncating it.
    $ErrorActionPreference = 'Continue'
    $tkOutput = & $python -B -c 'import tkinter; w = tkinter.Tk(); w.withdraw(); w.destroy()' 2>&1
    if ($LASTEXITCODE -ne 0) {
        # Some embedded distributions need a local copy of Tcl/Tk.
        $base = (& $python -B -c 'import sys; print(sys.base_prefix)' | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Unable to locate the Python runtime.' }
        $local = Join-Path $root '.venv-annotation\tcl'
        $source = Join-Path $base 'tcl'
        if (-not (Test-Path -LiteralPath (Join-Path $source 'tcl8.6\init.tcl') -PathType Leaf)) {
            throw ("Python Tcl/Tk is missing. Reinstall Python with Tcl/Tk enabled." + [Environment]::NewLine + ($tkOutput -join [Environment]::NewLine))
        }
        if (-not (Test-Path -LiteralPath (Join-Path $local 'tcl8.6\init.tcl') -PathType Leaf) -or
            -not (Test-Path -LiteralPath (Join-Path $local 'tk8.6\tk.tcl') -PathType Leaf)) {
            $null = New-Item -ItemType Directory -Path $local -Force -ErrorAction Stop
            foreach ($item in Get-ChildItem -LiteralPath $source) {
                Copy-Item -LiteralPath $item.FullName -Destination $local -Recurse -Force -ErrorAction Stop
            }
        }
        $env:TCL_LIBRARY = Join-Path $local 'tcl8.6'
        $env:TK_LIBRARY = Join-Path $local 'tk8.6'
    }

    $code = @'
import json
from pathlib import Path
import sys
import tkinter

root_path, mode, input_path, *runtime_args = sys.argv[1:]
root = Path(root_path)
source = Path(input_path)
sys.path.insert(0, str(root / "development" / "video_studio"))
import app

if runtime_args:
    runtime = Path(runtime_args[0])
    for name in app.CPU_RUNTIME_FILES:
        if not (runtime / name).is_file():
            raise FileNotFoundError("The CPU runtime directory is missing " + name)
if mode == "Sample":
    import table_view

    case = json.loads(source.read_text(encoding="utf-8-sig"))
    result = root / case["result"]
    for path in (root / case["video"], *(result / name for name in ("run.json", "tracks.json", "landings.json"))):
        if not path.is_file():
            raise FileNotFoundError("Generate the samples first with tools/generate_samples.py. Missing: " + str(path))
    label = case["title"]
else:
    if not source.is_file():
        raise FileNotFoundError("Expected a local video file: " + str(source))
    label = source.name
window = tkinter.Tk()
window.withdraw()
version = window.tk.call("info", "patchlevel")
window.destroy()
print("VIDEO_STUDIO_CHECK_OK. Tk", version, "|", mode + ":", label)
'@
    $checkArgs = @('-B', '-', $root, $inputMode, $inputPath)
    if ($runtimePath) { $checkArgs += $runtimePath }
    $result = $code | & $python @checkArgs 2>&1
    if ($LASTEXITCODE -ne 0) { throw ($result -join [Environment]::NewLine) }
    $result
}

function Quote-Argument([string]$value) {
    $escaped = [regex]::Replace($value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

try {
    $runtimePath = $null
    if ($AnalysisRuntime) {
        $runtimePath = (Resolve-Path -LiteralPath $AnalysisRuntime).ProviderPath
    }
    if ($inputMode -eq 'Video') {
        $sourcePath = (Resolve-Path -LiteralPath $Video).ProviderPath
        $inputFlag = '--video'
    } else {
        $sourcePath = Join-Path $root "development\video_studio\table_demo_$Sample.json"
        $inputFlag = '--table-demo'
    }
    Invoke-Preflight $sourcePath $runtimePath
    if ($Check) { exit 0 }
    $logDir = Join-Path $Output 'launcher_logs'
    $null = New-Item -ItemType Directory -Path $logDir -Force
    $run = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + $PID
    $arguments = @('-B', $app, '--output', $Output, $inputFlag, $sourcePath)
    if ($runtimePath) { $arguments += @('--analysis-runtime', $runtimePath) }
    $commandLine = ($arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    Start-Process -FilePath $pythonw -ArgumentList $commandLine -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "$run.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "$run.stderr.log")
} catch {
    $message = $_.Exception.Message
    if (-not $Check) {
        Add-Type -AssemblyName System.Windows.Forms
        $null = [System.Windows.Forms.MessageBox]::Show($message, 'Pingpong Course Workbench')
    }
    Write-Error -Message $message -ErrorAction Continue
    exit 1
}
