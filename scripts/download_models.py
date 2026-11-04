"""Download embedding model files at BUILD time (the runtime container is offline).

Usage: python scripts/download_models.py [model_name ...]   (default: the configured model)
"""
import sys

from fastembed import TextEmbedding

from app.config import get_settings

s = get_settings()
for name in sys.argv[1:] or [s.embedding_model]:
    m = TextEmbedding(model_name=name, cache_dir=s.embedding_cache_dir)
    list(m.embed(["warm-up"]))
    print("ok", name)
