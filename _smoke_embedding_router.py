"""Smoke test: EmbeddingRouter fallback + dimension normalisation."""
import sys, time
sys.path.insert(0, '.')

from astroml.llm.providers.embedding_base import EmbeddingProvider, EmbeddingError
from astroml.llm.providers.embedding_local import LocalEmbeddingProvider
from astroml.llm.providers.embedding_router import EmbeddingRouter, build_default_router

# --- Test 1: Local-only router always works ---
local = LocalEmbeddingProvider()
assert local.is_available()
vec = local.embed("detect fraud in blockchain transactions")
assert isinstance(vec, list) and len(vec) > 0
print("PASS: LocalEmbeddingProvider.embed() ->", len(vec), "dims")

batch = local.embed_batch(["hello world", "fraud detection"])
assert len(batch) == 2
print("PASS: LocalEmbeddingProvider.embed_batch()")

# --- Test 2: Router with always-failing primary falls through to local ---
class AlwaysFailProvider(EmbeddingProvider):
    name = "always_fail"
    output_dim = 512
    def is_available(self): return True
    def embed(self, text):
        raise EmbeddingError("intentional failure")
    def embed_batch(self, texts):
        raise EmbeddingError("intentional failure")

router = EmbeddingRouter(
    providers=[AlwaysFailProvider(), LocalEmbeddingProvider()],
    target_dim=128,
)
t0 = time.monotonic()
vec = router.embed("test fallback behaviour")
elapsed_ms = (time.monotonic() - t0) * 1000
assert len(vec) == 128, "expected target_dim=128, got %d" % len(vec)
assert elapsed_ms < 500, "fallback took %.1f ms, must be < 500 ms" % elapsed_ms
assert router.active_provider.name == "local"
print("PASS: fallback to local within %.1f ms, dim=%d" % (elapsed_ms, len(vec)))

# --- Test 3: Dimension normalisation (padding) ---
router2 = EmbeddingRouter(
    providers=[LocalEmbeddingProvider()],
    target_dim=256,
)
vec2 = router2.embed("pad me to 256")
assert len(vec2) == 256, "expected 256, got %d" % len(vec2)
print("PASS: padding to target_dim=256")

# --- Test 4: Dimension normalisation (truncation) ---
router3 = EmbeddingRouter(
    providers=[LocalEmbeddingProvider()],
    target_dim=4,
)
vec3 = router3.embed("truncate me down to 4 dimensions please")
assert len(vec3) == 4, "expected 4, got %d" % len(vec3)
print("PASS: truncation to target_dim=4")

# --- Test 5: build_default_router falls back to local (no API keys) ---
default_router = build_default_router(target_dim=64)
t0 = time.monotonic()
vec4 = default_router.embed("stellar transaction fraud score")
elapsed_ms = (time.monotonic() - t0) * 1000
assert len(vec4) == 64
assert elapsed_ms < 500
print("PASS: build_default_router -> active=%s, dim=%d, time=%.1f ms" % (
    default_router.active_provider.name, len(vec4), elapsed_ms))

# --- Test 6: provider_status() ---
status = default_router.provider_status()
assert any(s["name"] == "local" for s in status)
print("PASS: provider_status() ->", [(s["name"], s["available"]) for s in status])

# --- Test 7: all-providers-fail raises EmbeddingError ---
router_empty = EmbeddingRouter(providers=[AlwaysFailProvider()])
try:
    router_empty.embed("this should fail")
    assert False, "should have raised EmbeddingError"
except EmbeddingError as e:
    print("PASS: EmbeddingError raised when all providers fail:", str(e)[:60])

print("\nALL SMOKE TESTS PASSED")
