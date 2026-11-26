#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

destroy_user_data=false
for arg in "$@"; do
    case "$arg" in
        --destroy-user-data)
            destroy_user_data=true
            ;;
        -h|--help)
            cat <<'EOF'
Usage: scripts/reset-local.sh [--destroy-user-data]

By default, stop and recreate the v1 core services (PostgreSQL, Redis, API)
while preserving all named Docker volumes.

--destroy-user-data requires typing DELETE_LOCAL_USER_DATA and removes
the Compose project's named volumes. Bind-mounted repository files are not
removed.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or is not accessible." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Error: Docker Compose v2 is required (docker compose)." >&2
    exit 1
fi

if ! docker compose config --quiet >/dev/null 2>&1; then
    echo "Error: docker-compose.yml is not valid." >&2
    exit 1
fi

if "$destroy_user_data"; then
    echo "WARNING: this permanently removes named project volumes."
    echo "Repository bind mounts and files outside Docker volumes will remain."
    read -r -p "Type DELETE_LOCAL_USER_DATA to continue: " confirmation
    if [[ "$confirmation" != "DELETE_LOCAL_USER_DATA" ]]; then
        echo "Reset cancelled; no data was removed." >&2
        exit 2
    fi
fi

echo "Stopping containers and removing orphaned containers..."
if "$destroy_user_data"; then
    docker compose down --remove-orphans --volumes
else
    docker compose down --remove-orphans
fi

echo "Starting core v1 services: postgres, redis, api..."
docker compose up -d --build postgres redis api

echo "Local reset complete."
