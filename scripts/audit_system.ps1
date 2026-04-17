# audit_system.ps1 - PowerShell wrapper around audit_system.py
#
# Prints a concise, human-readable environment summary and refreshes
# docs/system_audit_report.json. Safe to run on any Windows host.

[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "============================================================"
Write-Host "  agri-drone system audit  (PowerShell)"
Write-Host "============================================================"
Write-Host ("  Host       : " + $env:COMPUTERNAME + "  OS=" + [System.Environment]::OSVersion.VersionString)
Write-Host ("  User       : " + $env:USERNAME)

# Python + venv
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($py) {
    $pyVer = (& python -c "import sys; print(sys.version.split()[0])") 2>$null
    $venv = (& python -c "import sys; print(sys.prefix != getattr(sys,'base_prefix',sys.prefix))") 2>$null
    Write-Host ("  Python     : $pyVer  venv=$venv  ($py)")
} else {
    Write-Host "  Python     : NOT FOUND ON PATH"
}

# Git
if (Get-Command git -ErrorAction SilentlyContinue) {
    Push-Location $root
    $branch = (git branch --show-current) 2>$null
    $sha    = (git rev-parse --short HEAD) 2>$null
    $ahead  = (git rev-list --count main..HEAD) 2>$null
    $dirty  = (git status --short) 2>$null
    $dirtyCount = if ($dirty) { ($dirty -split "`n").Count } else { 0 }
    Write-Host ("  Git        : ${branch}@${sha}  ahead-of-main=$ahead  dirty=$dirtyCount")
    Pop-Location
}

# GPU
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) 2>$null
    Write-Host ("  GPU        : " + $gpu)
} else {
    Write-Host "  GPU        : No GPU (nvidia-smi not found)"
}

# Disk + RAM
$disk = Get-PSDrive -Name ($root.Substring(0,1)) -ErrorAction SilentlyContinue
if ($disk) {
    $freeGb = [math]::Round($disk.Free / 1GB, 2)
    Write-Host ("  Disk free  : $freeGb GB on drive " + $disk.Name)
}
$ram = (Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue)
if ($ram) {
    $totalGb = [math]::Round($ram.TotalVisibleMemorySize / 1MB, 2)
    $freeRamGb = [math]::Round($ram.FreePhysicalMemory / 1MB, 2)
    Write-Host ("  RAM        : free=$freeRamGb GB  total=$totalGb GB")
}

# Ports
foreach ($p in 8000, 9000) {
    $t = Test-NetConnection -ComputerName localhost -Port $p -WarningAction SilentlyContinue -InformationLevel Quiet
    $state = if ($t) { "in-use" } else { "free" }
    Write-Host ("  Port $p   : $state")
}

Write-Host "------------------------------------------------------------"
Write-Host "  Running Python audit to refresh JSON report ..."
if ($py) {
    & python (Join-Path $PSScriptRoot "audit_system.py")
} else {
    Write-Host "  (skipped - python not available)"
}
