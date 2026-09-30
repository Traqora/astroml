# API Key Rotation and Revocation Procedure

This document describes the API key rotation policy, expiration model, rotation workflow, and immediate revocation for AstroML machine-to-machine authentication.

---

## Expiration & Rotation Policy

1. **Expiration Policy**:
   - Every API key issued via `POST /api/v1/auth/api-keys` has a default expiration lifetime of **90 days** (`API_KEY_EXPIRE_DAYS=90`).
   - The creation timestamp is tracked on the key storage record (`created_at` / `api_key_created_at`).

2. **Rotation Mechanism**:
   - Endpoint: `POST /api/v1/auth/rotate-key`
   - Request Body: `{"name": "your-key-name"}`
   - Behavior:
     - Generates a new API key string and updates `key_hash` with a fresh 90-day expiration.
     - The previous key hash is transferred to `overlap_key_hash` with an `overlap_expires_at` set to **30 days** in the future (`API_KEY_ROTATION_OVERLAP_DAYS=30`).
     - During the 30-day overlap window, requests sent with either the old key or the new key are accepted.
     - After 30 days, the old key is automatically rejected upon authentication.

3. **Immediate Revocation**:
   - Endpoint: `POST /api/v1/auth/revoke-key`
   - Request Body: `{"name": "your-key-name"}`
   - Behavior:
     - Immediately sets `is_active = False` and clears any active `overlap_key_hash`.
     - Both primary and rotated-out overlap keys are deactivated instantly.

4. **Audit Logging**:
   - All key operations (`api_key.created`, `api_key.rotated`, `api_key.revoked`) are logged via structured logs and stored in the `audit_logs` database table for security auditing.

---

## Step-by-Step Rotation Example

### Step 1: Request Key Rotation

```bash
curl -X POST http://localhost:8000/api/v1/auth/rotate-key \
  -H "Authorization: Bearer <ADMIN_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"name": "service-worker-key"}'
```

### Step 2: Receive New Key Response

```json
{
  "new_key": "ak_abc123...",
  "name": "service-worker-key",
  "scopes": ["read:transactions"],
  "expires_at": "2026-10-25T12:00:00Z",
  "created_at": "2026-07-27T12:00:00Z",
  "overlap_expires_at": "2026-08-26T12:00:00Z",
  "message": "Key rotated. Old key remains valid until 2026-08-26 (30-day overlap). Update your clients before that date."
}
```

### Step 3: Deploy New Key to Clients

Deploy `new_key` to client applications. Services using the old key will continue to function without downtime during the 30-day overlap window.

---

## Immediate Revocation Example

In case of key compromise:

```bash
curl -X POST http://localhost:8000/api/v1/auth/revoke-key \
  -H "Authorization: Bearer <ADMIN_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"name": "service-worker-key"}'
```

Response:
```json
{
  "name": "service-worker-key",
  "revoked": true,
  "message": "Key 'service-worker-key' has been revoked immediately. Both primary and overlap keys are now invalid."
}
```
