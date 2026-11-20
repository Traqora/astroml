# test_container_user.ps1 — Assert the API container does not run as root (issue #1249)
$ErrorActionPreference = 'Stop'

$ImageTag = "astroml-api-test:$(Get-Date -UFormat %s)"

Write-Host "Building image..."
docker build -t $ImageTag -f api/Dockerfile .

Write-Host "Checking runtime user..."
$RuntimeUser = docker run --rm --entrypoint id $ImageTag -u

docker rmi $ImageTag 2>$null

if ($RuntimeUser -eq '0') {
    Write-Host "FAIL: container is running as root (uid=0)"
    exit 1
}

if ($RuntimeUser -ne '1000') {
    Write-Host "FAIL: container runtime user uid=$RuntimeUser, expected 1000"
    exit 1
}

Write-Host "PASS: container runtime user is uid=$RuntimeUser (not root)"
exit 0
