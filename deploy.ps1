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
    # --pull refreshes base layers so a stale local cache doesn't silently
    # skip security patches.
    docker build --pull -t $ImageName .
}
Invoke-Step "Deploying to server" {
    # Streams the image straight into the server's Docker daemon over SSH —
    # no local .tar file and no separate scp hop. --force-recreate is
    # required: `docker compose up -d` alone only recreates a container when
    # the *resolved compose config* changes, not when a mutable tag like
    # `fastest-racer:latest` starts pointing at different image content.
    # `docker image prune -f` clears the now-dangling previous `:latest`
    # layer so repeated deploys don't slowly fill the server's disk.
    docker save $ImageName | ssh -i $SshKey "${ServerUser}@${ServerHost}" "cd $ServerPath && docker load && docker compose up -d --force-recreate && docker image prune -f"
}

Write-Host "==> Done" -ForegroundColor Green
