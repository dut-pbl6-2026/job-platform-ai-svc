"""OCR path (scanned / image-only pages). Local Tesseract, no external service."""
import io

import pytesseract
from PIL import Image

CONF_RETRY_BELOW = 45.0


def ocr_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def _ocr_image(img: Image.Image, langs: str) -> tuple[str, float]:
    data = pytesseract.image_to_data(img, lang=langs, config="--psm 3", output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not word or conf < 0:
            continue
        confs.append(conf)
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    return text, (sum(confs) / len(confs) if confs else 0.0)


def ocr_png(png_bytes: bytes, langs: str) -> tuple[str, float, bool]:
    """OCR one page image. Returns (text, mean confidence, rotated). Retries once rotated by OSD when confidence is low."""
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    text, conf = _ocr_image(img, langs)
    if conf >= CONF_RETRY_BELOW and len(text) >= 40:
        return text, conf, False
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
    except Exception:  # noqa: BLE001  (OSD fails on sparse pages)
        angle = 0
    if angle:
        rotated = img.rotate(-angle, expand=True)
        text2, conf2 = _ocr_image(rotated, langs)
        if conf2 > conf:
            return text2, conf2, True
    return text, conf, False
