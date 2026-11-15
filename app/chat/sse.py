"""Server-sent events framing for the chat stream.

The shape is the one the web and mobile clients parse (see their chat readers):
  event: token  data: {"delta": "..."}     one chunk of text; always non-empty
  event: error  data: {"code", "message"}  upstream failure, the answer text is still streamed
  event: jobs   data: [{"id", "title", "company"}]
  event: done   data: {"sessionId": "...", "done": true}   always last
Blocks end with a blank line and use \\n only. JSON is encoded on one line, so a chunk never breaks framing.
"""
import json
import re
from collections.abc import Iterator

_SPLIT = re.compile(r"(\s+)")


def frame(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def token_frames(text: str) -> Iterator[str]:
    """Word and whitespace chunks, so the text reads as it would when streamed from a model."""
    for part in _SPLIT.split(text):
        if part:
            yield frame("token", {"delta": part})
