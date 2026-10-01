"""nginx TLS configuration regression test (#941).

``nginx/nginx.conf`` is the reverse proxy front door (see
``HTTPS_SETUP_GUIDE.md``). The guide's canonical TLS server block terminates
TLS 1.2/1.3 with an explicit certificate/key and modern ciphers, but the
checked-in config only listens on port 80, so nothing stopped a deployment
from exposing the API over plaintext forever — or from enabling TLS with
deprecated protocol versions.

This test parses the config (not the guide) and enforces the baseline:

* when a TLS (443) listener exists it must pin ``ssl_protocols`` to
  TLSv1.2/TLSv1.3, ship an explicit cert/key pair, and exclude weak ciphers
* HTTP listeners must redirect plaintext traffic to HTTPS
* HSTS is declared on TLS listeners
* the proxy chain keeps ``X-Forwarded-Proto`` so the app can detect scheme

CI runs this on every PR, so a regression that strips the TLS baseline
(e.g. a merge that reverts to a plain port-80-only config) fails loudly
instead of surfacing in a security audit.
"""

from pathlib import Path

import pytest

NGINX_CONF = Path(__file__).resolve().parents[2] / "nginx" / "nginx.conf"


@pytest.fixture(scope="module")
def nginx_conf_text() -> str:
    return NGINX_CONF.read_text(encoding="utf-8")


def _server_blocks(text: str) -> list[str]:
    """Extract each ``server { ... }`` block body as a string."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].split("#")[0].strip()
        if stripped == "server {":
            depth = 1
            body: list[str] = []
            i += 1
            while i < len(lines) and depth > 0:
                line = lines[i]
                body.append(line)
                depth += line.count("{") - line.count("}")
                i += 1
            blocks.append("\n".join(body))
        i += 1
    return blocks


def _tls_blocks(blocks: list[str]) -> list[str]:
    return [b for b in blocks if "listen" in b and "443" in b]


def _plain_blocks(blocks: list[str]) -> list[str]:
    return [b for b in blocks if "listen" in b and "443" not in b and "8443" not in b]


def test_nginx_conf_exists_and_parses_into_server_blocks(nginx_conf_text: str):
    assert NGINX_CONF.exists(), f"{NGINX_CONF} missing"
    assert nginx_conf_text.strip(), f"{NGINX_CONF} is empty"
    blocks = _server_blocks(nginx_conf_text)
    assert len(blocks) >= 1, "expected at least one server block"


def test_tls_listener_pins_modern_protocols_only(nginx_conf_text: str):
    blocks = _server_blocks(nginx_conf_text)
    tls = _tls_blocks(blocks)
    assert tls, (
        "nginx.conf defines no TLS (443) listener; add one per "
        "HTTPS_SETUP_GUIDE.md or the proxy serves plaintext only (#941)"
    )
    for block in tls:
        protocols = [line for line in block.splitlines() if "ssl_protocols" in line]
        assert protocols, "TLS listener must pin ssl_protocols explicitly"
        joined = " ".join(protocols)
        assert (
            "TLSv1.2" in joined and "TLSv1.3" in joined
        ), f"expected TLSv1.2/TLSv1.3, got: {joined}"
        assert (
            "SSLv" not in joined and "TLSv1 " not in joined and "TLSv1.0" not in joined
        ), f"deprecated protocol enabled in ssl_protocols: {joined}"


def test_tls_listener_declares_certificate_and_key(nginx_conf_text: str):
    for block in _tls_blocks(_server_blocks(nginx_conf_text)):
        assert "ssl_certificate " in block, "TLS listener must declare ssl_certificate"
        assert "ssl_certificate_key " in block, "TLS listener must declare ssl_certificate_key"


def test_tls_listener_excludes_weak_ciphers(nginx_conf_text: str):
    for block in _tls_blocks(_server_blocks(nginx_conf_text)):
        ciphers = [line for line in block.splitlines() if "ssl_ciphers" in line]
        assert ciphers, "TLS listener must pin ssl_ciphers (e.g. HIGH:!aNULL:!MD5)"
        joined = " ".join(ciphers)
        assert "!aNULL" in joined, f"ciphersuite must exclude anonymous suites: {joined}"
        assert "!MD5" in joined, f"ciphersuite must exclude MD5-based suites: {joined}"


def test_plain_http_listener_redirects_to_https(nginx_conf_text: str):
    plain = _plain_blocks(_server_blocks(nginx_conf_text))
    if not plain:
        pytest.skip("no plaintext listener declared")
    for block in plain:
        has_redirect = any(
            "return 301 https://" in line or "return 308 https://" in line
            for line in block.splitlines()
        )
        serves_tls_aware_proxy = "443" in block
        assert (
            has_redirect or serves_tls_aware_proxy
        ), "plaintext listener must 301/308-redirect to HTTPS (#941)"


def test_tls_listener_sends_hsts(nginx_conf_text: str):
    for block in _tls_blocks(_server_blocks(nginx_conf_text)):
        hsts = [line for line in block.splitlines() if "Strict-Transport-Security" in line]
        assert hsts, "TLS listener must send Strict-Transport-Security"


def test_proxy_chain_preserves_forwarded_proto(nginx_conf_text: str):
    assert (
        "X-Forwarded-Proto" in nginx_conf_text
    ), "proxy locations must forward X-Forwarded-Proto so the app can detect https"
