# Container Runtime Users

All AstroML service containers run as a dedicated non-root user. This document covers the convention, the rationale, and how to work with it locally.

## Convention

- **User**: `appuser` (UID 1000) in the API service (`api/Dockerfile`)
- **Group**: `appgroup` (GID 1000)
- **User in root Dockerfile stages**: `astroml`/`astroml` (created with `groupadd -r`/`useradd -r` in the base stage)
- **User in Soroban stages**: `soroban`/`soroban`

The `USER` instruction appears immediately before the final `CMD`/`ENTRYPOINT` in every runtime stage. Build-time steps (package installs, directory creation, `chown`) run as root before the user switch.

## Why root containers are disallowed in v1

Running container processes as root violates the principle of least privilege and creates unnecessary risk:

1. **Credential exposure**: The API service handles JWT secrets and database credentials passed via environment variables. A root process that escapes container isolation has full host access.
2. **Public-chain data integrity**: A root process could modify application files at runtime; a non-root process cannot overwrite its own code.
3. **Service boundary enforcement**: Non-root UIDs make it straightforward to apply Kubernetes `securityContext` / `runAsNonRoot: true` and Linux namespace policies without per-service exceptions.
4. **Defence in depth**: Container escapes are rare but not impossible. A non-root process limits post-escape lateral movement.

## Directory permissions

Only directories the application writes to at runtime (`/app/logs`, and service-specific paths like `/app/feature_store`) are `chown`-ed to the application user. Read-only paths (config, source code) are owned by root and readable by all users, limiting the blast radius of a compromised process.

## Verifying the runtime user

After building the API image:

```bash
# Linux/macOS
bash scripts/test_container_user.sh

# Windows PowerShell
pwsh scripts/test_container_user.ps1
```

The CI pipeline (`docker.yml`) runs this check automatically on every push and pull request.

## Overriding for local development

Running as a non-root user is strongly recommended even locally. The main case where you might consider overriding is bind-mounting a host directory that is owned by your local user and whose UID does not match 1000.

**Preferred solution**: match UIDs. Build with `--build-arg UID=$(id -u)` once `ARG UID` / `--uid` support is added, or simply ensure your host project directory is world-readable.

**Temporary workaround** (discouraged — do not use in CI or production):

```bash
docker run --user root ...
```

This defeats the purpose of the non-root default and should never appear in `docker-compose.yml` or Kubernetes manifests without an explicit justification comment.
