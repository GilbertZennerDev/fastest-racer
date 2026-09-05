# Builds the Docker image locally, ships it to the Hetzner server, and starts it there —
# avoids building on the low-RAM server (numba/scipy wheels + JIT warmup are heavy).
# Run from the repo root in PowerShell.
#
# Usage: .\deploy.ps1

$ErrorActionPreference = "Stop"

$ServerUser = "root"
$ServerHost = "89.167.25.230"
$ServerPath = "/opt/fastest-racer"
$ImageName  = "fastest-racer:latest"
$TarFile    = "fastest-racer.tar"
# Windows OpenSSH does not reliably expand "~" when the path is built inside
# a script variable and passed through as an argument, so resolve it via
# $HOME explicitly.
$SshKey     = Join-Path $HOME ".ssh/id_rsa"

function Invoke-Step {
    param([string]$Description, [scriptblock]$Command)
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed: $Description (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Step "Building Docker image" {
    docker build -t $ImageName .
}
Invoke-Step "Saving image to $TarFile" { docker save -o $TarFile $ImageName }
Invoke-Step "Copying to server" { scp -i $SshKey $TarFile "${ServerUser}@${ServerHost}:${ServerPath}/" }
Invoke-Step "Loading image and restarting container on server" {
    # --force-recreate is required: `docker compose up -d` alone only
    # recreates a container when the *resolved compose config* changes, not
    # when a mutable tag like `fastest-racer:latest` starts pointing at
    # different image content — without it, `docker load` silently updates
    # the local image while the running container keeps serving the old one.
    ssh -i $SshKey "${ServerUser}@${ServerHost}" "cd $ServerPath && docker load -i $TarFile && docker compose up -d --force-recreate && rm $TarFile"
}

if (Test-Path $TarFile) {
    Write-Host "==> Cleaning up local tar" -ForegroundColor Cyan
    Remove-Item $TarFile
}

Write-Host "==> Done" -ForegroundColor Green
