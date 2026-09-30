# LLM API Gateway Documentation

> **Resolves #457** — Unified REST and WebSocket API for all LLM features.

## Overview

The LLM API Gateway exposes all AI/LLM capabilities through a single, consistent
REST and WebSocket interface. Authentication is enforced on all endpoints.
Rate limits apply per user tier.

**Base URL:** `https://api.astroml.io/api/v1/llm`

**Authentication:** Bearer token (JWT) required in `Authorization` header.

---

## Rate Limits

| Tier | Requests/min | Notes |
|------|-------------|-------|
| Free | 100 | Shared rate limit across all endpoints |
| Pro  | 1000 | Per-endpoint rate limits |
| Enterprise | Custom | Dedicated infrastructure |

Rate limit headers returned on every response:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1722000000
```

When exceeded:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 60s.",
    "details": { "retry_after": 60 }
  }
}
```

---

## REST Endpoints

### POST /generate
Generate a text completion.

**Request:**
```json
{
  "prompt": "Summarise this transaction in one sentence.",
  "model": "gpt-4-turbo",
  "temperature": 0.7,
  "max_tokens": 1024,
  "idempotency_key": "req_abc123"
}
```

**Response:**
```json
{
  "id": "gen_abc123",
  "model": "gpt-4-turbo",
  "content": "Response text",
  "usage": { "prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150 },
  "cost": 0.003,
  "latency_ms": 1200,
  "cached": false
}
```

---

### POST /generate/stream
Stream a completion as Server-Sent Events.

**Request:** Same as `/generate`.

**Response (SSE stream):**
```
data: {"delta": "This", "finish_reason": null}
data: {"delta": " is", "finish_reason": null}
data: {"delta": " a response.", "finish_reason": null}
data: [DONE]
```

---

### POST /embed
Generate vector embeddings.

**Request:**
```json
{
  "input": "Transaction on 2026-07-01",
  "model": "text-embedding-3-small"
}
```

**Response:**
```json
{
  "id": "emb_abc123",
  "model": "text-embedding-3-small",
  "embeddings": [[0.01, 0.23, ...]],
  "usage": { "prompt_tokens": 8, "total_tokens": 8 },
  "cost": 0.0000002
}
```

---

### POST /chat
Chat completion with a messages array.

**Request:**
```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is AstroML?" }
  ],
  "model": "gpt-4-turbo",
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chat_abc123",
  "model": "gpt-4-turbo",
  "message": { "role": "assistant", "content": "AstroML is a fraud detection platform." },
  "usage": { "prompt_tokens": 25, "completion_tokens": 80, "total_tokens": 105 },
  "cost": 0.002,
  "latency_ms": 950
}
```

---

### POST /rag/query
RAG-augmented query with document retrieval.

**Request:**
```json
{
  "query": "What transactions occurred on 2026-07-01?",
  "top_k": 5,
  "model": "gpt-4-turbo"
}
```

**Response:**
```json
{
  "id": "rag_abc123",
  "query": "What transactions occurred on 2026-07-01?",
  "answer": "On 2026-07-01, the following transactions were recorded...",
  "documents": [
    { "doc_id": "doc_0", "content": "...", "score": 0.95, "metadata": {} }
  ],
  "usage": { "prompt_tokens": 200, "completion_tokens": 150, "total_tokens": 350 },
  "cost": 0.005,
  "latency_ms": 1500
}
```

---

### GET /models
List all available models.

**Response:**
```json
{
  "models": [
    {
      "id": "gpt-4-turbo",
      "provider": "openai",
      "context_window": 128000,
      "cost_per_1k_prompt_tokens": 0.01,
      "cost_per_1k_completion_tokens": 0.03,
      "supports_streaming": true,
      "supports_vision": true
    }
  ],
  "total": 4
}
```

---

### GET /cost/usage
Get LLM cost usage for the authenticated user.

**Query params:** `?period=2026-07`

**Response:**
```json
{
  "user_id": "user_42",
  "period": "2026-07",
  "total_requests": 1500,
  "total_tokens": 3200000,
  "total_cost_usd": 45.20,
  "cost_by_model": {
    "gpt-4-turbo": 40.00,
    "text-embedding-3-small": 5.20
  },
  "cost_by_day": []
}
```

---

## Natural Language Query Endpoints

### POST /query/
Convert natural language to SQL / API / GraphQL.

**Request:**
```json
{
  "query": "Show me all transactions over $10,000 in the last 7 days",
  "target": "sql",
  "schema_hint": "transactions(id, amount, timestamp, user_id)"
}
```

---

## Explanation Endpoints

### POST /explanations/
Generate human-readable LLM explanations.

**Request:**
```json
{
  "subject": "fraud_decision",
  "context": { "score": 0.94, "reason_codes": ["V001", "V002"] },
  "audience": "user"
}
```

---

## Agent Endpoints

### POST /agents/run
Execute an LLM agent task.

### GET /agents/{task_id}
Retrieve agent task status and result.

---

## WebSocket Endpoints

### WS /chat/ws
Streaming chat over WebSocket.

**Send:**
```json
{ "messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4-turbo" }
```

**Receive chunks:**
```json
{ "delta": "Hello!", "finish_reason": null }
{ "delta": "", "finish_reason": "stop" }
```

---

### WS /stream
Generic streaming completion over WebSocket.

**Send:**
```json
{ "prompt": "Explain fraud detection", "model": "gpt-4-turbo" }
```

---

## Error Format

All errors follow this consistent structure:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message.",
    "details": {}
  }
}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `SAFETY_VIOLATION` | 400 | Request blocked by safety guardrails |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit hit; retry after `details.retry_after` seconds |
| `INVALID_REQUEST` | 422 | Validation error |
| `MODEL_NOT_FOUND` | 404 | Requested model does not exist |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Idempotency

For `POST` endpoints, include an `idempotency_key` in the request body.
Repeated requests with the same key return the cached response without
re-executing the LLM call. Keys expire after 24 hours.
