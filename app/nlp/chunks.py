"""Split a CV into embedding-sized chunks (the model truncates ~128 tokens, so no whole-document embedding)."""
from app.nlp.normalize import split_sentences

MIN_WORDS, MAX_WORDS = 8, 55


def make_chunks(text: str, limit: int = 120) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for s in split_sentences(text):
        w = len(s.split())
        if w > MAX_WORDS:
            words = s.split()
            for i in range(0, len(words), MAX_WORDS):
                chunks.append(" ".join(words[i : i + MAX_WORDS]))
            continue
        buf.append(s)
        count += w
        if count >= MIN_WORDS:
            chunks.append(" ".join(buf))
            buf, count = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return chunks[:limit]
