"""Hidden-text detection helpers (SPEC section 3, 'Hidden-text defence')."""
import numpy as np

MIN_FONT_PT = 3.0
COLOR_DIST_MAX = 24.0  # text colour this close to the local background => invisible
CONTRAST_MIN = 28.0  # nothing in the bbox differs from the background by this much => no visible ink


def srgb_int_to_rgb(c: int) -> tuple[int, int, int]:
    return (c >> 16) & 255, (c >> 8) & 255, c & 255


def color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return float(np.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b))))


def region_stats(pix: np.ndarray, bbox, scale: float) -> tuple[tuple[int, int, int], float] | None:
    """Return (background RGB, max contrast) for the bbox region of a rendered page.

    Background = per-channel median of the region (text strokes are the minority of pixels).
    Contrast = 99th percentile of distance to that background (robust to a few stray pixels).
    """
    h, w = pix.shape[:2]
    x0, y0, x1, y1 = (int(max(0, v * scale)) for v in bbox)
    x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
    if x0 >= w or y0 >= h:
        return None
    region = pix[y0:y1, x0:x1].reshape(-1, pix.shape[2])[:, :3].astype(np.int16)
    if region.size == 0:
        return None
    bg = np.median(region, axis=0)
    dist = np.sqrt(((region - bg) ** 2).sum(axis=1))
    return (int(bg[0]), int(bg[1]), int(bg[2])), float(np.percentile(dist, 99))


def is_hidden_span(span: dict, page_rect, pix: np.ndarray, scale: float) -> bool:
    text = span.get("text", "")
    if not text.strip():
        return False
    if span.get("size", 12.0) < MIN_FONT_PT:
        return True
    x0, y0, x1, y1 = span["bbox"]
    inter_w = min(x1, page_rect.x1) - max(x0, page_rect.x0)
    inter_h = min(y1, page_rect.y1) - max(y0, page_rect.y0)
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    if inter_w <= 0 or inter_h <= 0 or (inter_w * inter_h) / area < 0.5:
        return True
    stats = region_stats(pix, span["bbox"], scale)
    if stats is None:
        return True
    bg, contrast = stats
    if color_distance(srgb_int_to_rgb(span.get("color", 0)), bg) <= COLOR_DIST_MAX:
        return True
    return contrast < CONTRAST_MIN
