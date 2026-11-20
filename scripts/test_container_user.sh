#!/usr/bin/env bash
# test_container_user.sh — Assert the API container does not run as root (issue #1249)
set -euo pipefail

IMAGE_TAG="astroml-api-test:$(date +%s)"

echo "Building image..."
docker build -t "$IMAGE_TAG" -f api/Dockerfile .

echo "Checking runtime user..."
RUNTIME_USER=$(docker run --rm --entrypoint id "$IMAGE_TAG" -u)

docker rmi "$IMAGE_TAG" >/dev/null 2>&1 || true

if [ "$RUNTIME_USER" = "0" ]; then
  echo "FAIL: container is running as root (uid=0)"
  exit 1
fi

if [ "$RUNTIME_USER" != "1000" ]; then
  echo "FAIL: container runtime user uid=$RUNTIME_USER, expected 1000"
  exit 1
fi

echo "PASS: container runtime user is uid=$RUNTIME_USER (not root)"
exit 0
