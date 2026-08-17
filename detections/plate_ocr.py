
import os
import string

import cv2

_reader = None

dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}
dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}


def get_reader():
    global _reader
    if _reader is None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        _reader = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _reader


def paddle_readtext(license_plate_crops):
    images = []
    for crop in license_plate_crops:
        if crop.ndim == 2:
            images.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))
        else:
            images.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    detected_chars_per_crop = []
    for prediction in get_reader().ocr(images) or []:
        texts = prediction["rec_texts"] if "rec_texts" in prediction else []
        scores = prediction["rec_scores"] if "rec_scores" in prediction else []
        detected_chars_per_crop.append(
            [(str(text), float(score)) for text, score in zip(texts, scores)]
        )
    while len(detected_chars_per_crop) < len(license_plate_crops):
        detected_chars_per_crop.append([])
    return detected_chars_per_crop


# L = letter, D = digit. Applied after spaces/hyphens are stripped.
# AB12 CDE → LLDDLLL
# A12 CDE  → LDDLLL
# A-123-CD → LDDDLL
PLATE_PATTERNS = (
    "LLDDLLL",
    "LDDLLL",
    "LDDDLL",
)


def _char_matches(ch, kind):
    if kind == "L":
        return ch in string.ascii_uppercase or ch in dict_int_to_char
    return ch in "0123456789" or ch in dict_char_to_int


def _pattern_score(text, pattern):
    if len(text) != len(pattern):
        return None
    score = 0
    for ch, kind in zip(text, pattern):
        if not _char_matches(ch, kind):
            return None
        if kind == "L":
            score += 2 if ch in string.ascii_uppercase else 1
        else:
            score += 2 if ch in "0123456789" else 1
    return score


def license_complies_format(text):
    return any(_pattern_score(text, pattern) is not None for pattern in PLATE_PATTERNS)


def format_license(text, pattern=None):
    if pattern is None:
        pattern = max(
            (p for p in PLATE_PATTERNS if _pattern_score(text, p) is not None),
            key=lambda p: _pattern_score(text, p),
            default=None,
        )
    if pattern is None:
        return text

    mapped = []
    for ch, kind in zip(text, pattern):
        table = dict_int_to_char if kind == "L" else dict_char_to_int
        mapped.append(table.get(ch, ch))
    return "".join(mapped)


def pick_license_plate(detected_chars):

    for text, score in detected_chars:
        text = text.upper().replace(" ", "").replace("-", "")

        best_pattern = None
        best_fit = -1
        for pattern in PLATE_PATTERNS:
            fit = _pattern_score(text, pattern)
            if fit is not None and fit > best_fit:
                best_pattern = pattern
                best_fit = fit

        if best_pattern is not None:
            return format_license(text, best_pattern), score

    return None, None


def read_license_plates(license_plate_crops):
    if not license_plate_crops:
        return []
    return [
        pick_license_plate(detected_chars)
        for detected_chars in paddle_readtext(license_plate_crops)
    ]
