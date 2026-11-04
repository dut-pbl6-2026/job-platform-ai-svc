"""Local embeddings (ONNX via fastembed). Interface + a deterministic hashing fallback used in unit tests."""
import hashlib
import re
import threading
from collections import OrderedDict
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-9)


class OnnxEmbedder:
    """Wraps fastembed.TextEmbedding. Model files are baked into the image (offline at runtime)."""

    def __init__(self, model_name: str, cache_dir: str, cache_size: int = 4096):
        from fastembed import TextEmbedding  # imported lazily: heavy

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        with self._lock:  # one lock for cache + model: requests run in worker threads
            found = {t: self._cache[t] for t in dict.fromkeys(texts) if t in self._cache}
            missing = [t for t in dict.fromkeys(texts) if t not in found]
            if missing:
                vecs = _unit(np.array(list(self._model.embed(missing)), dtype=np.float32))
                for t, v in zip(missing, vecs, strict=True):
                    found[t] = v
                    self._cache[t] = v
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return np.stack([found[t] for t in texts])


class HashingEmbedder:
    """Bag-of-words feature hashing. No model, deterministic - for tests and as the degraded fallback."""

    name = "hashing-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9+#]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _unit(out)
