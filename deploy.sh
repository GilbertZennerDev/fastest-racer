#!/usr/bin/env bash
set -euo pipefail

IDENTITY_FILE="${IDENTITY_FILE:-$HOME/.ssh/pic_allier_id_rsa}"

step() {
    local desc="$1"; shift
    echo -e "\033[0;36m==> ${desc}\033[0m"
    if ! eval "$1"; then
        echo -e "\033[0;31m✘ Failed: ${desc}\033[0m" >&2
        exit 1
    fi
}

SERVER_USER="root"
SERVER_HOST="89.167.25.230"
SERVER_PATH="/opt/fastest-racer"
IMAGE_NAME="fastest-racer:latest"

step "Building Docker image" 'docker build --pull -t "$IMAGE_NAME" .'
step "Deploying to server" 'docker save "$IMAGE_NAME" | ssh -i "$IDENTITY_FILE" "${SERVER_USER}@${SERVER_HOST}" "cd $SERVER_PATH && docker load && docker compose up -d --force-recreate && docker image prune -f"'

echo -e "\033[0;32m==> Done\033[0m"
