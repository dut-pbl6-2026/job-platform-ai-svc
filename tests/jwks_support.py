"""Test-only RS256 signing with a JWKS served from a loopback HTTP server (no key material is committed).

The server answers the same discovery and JWKS paths auth-svc publishes, so the service under test runs its real
key-loading path: discovery, jwks_uri, PyJWKClient, kid lookup.
"""
import base64
import http.server
import json
import threading

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

KID = "test-kid"
ISSUER = "job-platform"
AUDIENCE = "job-platform"

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)  # the key auth-svc would sign with
FOREIGN_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)  # never published


def _pem(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _jwks() -> dict:
    numbers = KEY.public_key().public_numbers()
    return {"keys": [{"kty": "RSA", "use": "sig", "alg": "RS256", "kid": KID,
                      "n": _b64url_uint(numbers.n), "e": _b64url_uint(numbers.e)}]}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        if self.path == "/.well-known/openid-configuration":
            body = {"issuer": ISSUER, "jwks_uri": f"{base}/.well-known/jwks.json"}
        elif self.path == "/.well-known/jwks.json":
            body = _jwks()
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep test output quiet
        pass


_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
threading.Thread(target=_server.serve_forever, daemon=True).start()
METADATA_URL = f"http://127.0.0.1:{_server.server_address[1]}/.well-known/openid-configuration"

_KEY_PEM = _pem(KEY)
_FOREIGN_PEM = _pem(FOREIGN_KEY)


def claims(**overrides) -> dict:
    base = {"sub": "user-1", "iss": ISSUER, "aud": AUDIENCE, "exp": 4102444800}
    return {**base, **overrides}


def sign(payload: dict | None = None, *, kid: str | None = KID, foreign: bool = False) -> str:
    """RS256 token. kid=None omits the header; foreign=True signs with a key that is not published."""
    headers = {"kid": kid} if kid else None
    return jwt.encode(payload if payload is not None else claims(), _FOREIGN_PEM if foreign else _KEY_PEM,
                      algorithm="RS256", headers=headers)


def sign_hs256(payload: dict | None = None, secret: str = "h" * 48) -> str:
    """An HS256 token, to prove it is refused."""
    return jwt.encode(payload if payload is not None else claims(), secret, algorithm="HS256")
