# LLM Features

AstroML exposes deterministic, testable LLM features through `/api/v1/llm`.
The current public API includes:

## API reference

| Endpoint | Method | Purpose | Minimal runnable payload |
| --- | --- | --- | --- |
| `/api/v1/llm/ask` | POST | Retrieval-augmented answer with citations. | `{"question":"How do LLM endpoints work?"}` |
| `/api/v1/llm/explain` | POST | Explain transaction details. | `{"tx_details":"payment of 10 XLM"}` |
| `/api/v1/llm/query` | POST | Translate natural language into SQL. | `{"query":"show high risk accounts"}` |
| `/api/v1/llm/context` | POST | Summarize graph/time-series context and Mermaid diagram. | `{"edges":[],"data_points":[1,2,3]}` |
| `/api/v1/llm/validate` | POST | Validate/guard model output against context. | `{"raw_response":{"answer":"ok"},"context":"answer is ok"}` |
| `/api/v1/llm/stream` | POST | Stream text chunks. | `{"prompt":"Summarize fraud risk"}` |
| `/api/v1/llm/feedback` | POST | One-click or expert-weighted feedback for an LLM output. | `{"feature":"ask","prompt":"p","output":"o","rating":5}` |
| `/api/v1/llm/feedback/dashboard` | GET | Dashboard trend metrics. | n/a |
| `/api/v1/llm/feedback/prompt-improvements` | GET | Feedback-to-prompt recommendations. | n/a |

## Usage examples

Run the API locally:

```bash
uvicorn api.app:app --reload
```

Ask a cited RAG question:

```bash
curl -s http://localhost:8000/api/v1/llm/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Where are API usage examples documented?"}'
```

Submit one-click feedback:

```bash
curl -s http://localhost:8000/api/v1/llm/feedback \
  -H 'content-type: application/json' \
  -d '{"feature":"ask","prompt":"Where are docs?","output":"See docs","rating":4}'
```

Submit expert-weighted feedback:

```bash
curl -s http://localhost:8000/api/v1/llm/feedback \
  -H 'content-type: application/json' \
  -d '{"feature":"ask","prompt":"Risk?","output":"Too vague","rating":2,"is_expert":true,"expert_weight":3,"comment":"Require citations"}'
```

View trends:

```bash
curl -s http://localhost:8000/api/v1/llm/feedback/dashboard
```

## Integration testing

Use `api/tests/llm_mocking.py` for deterministic provider behavior, latency
tracking, and cost accounting. Golden regression cases live in
`api/tests/llm_golden/ask_cases.json`. The integration suite enforces the
p95 latency gate (`<5s`) and validates chaos behavior for bad payloads and
provider failures.

## Troubleshooting

- **422 response**: validate that required JSON fields are present and non-empty.
- **Slow tests**: use `DeterministicLLMMock(delay_ms=...)` to reproduce latency
  without calling an external provider.
- **Feedback trends look unbalanced**: expert feedback intentionally receives
  `expert_weight` between `1` and `5`; regular user feedback is weight `1`.
- **Streaming output is short**: the local stream endpoint is deterministic and
  optimized for tests; production providers can replace the mock generator.
