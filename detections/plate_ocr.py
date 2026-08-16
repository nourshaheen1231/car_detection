
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


def license_complies_format(text):
    if len(text) != 7:
        return False

    return (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char)
        and (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char)
        and (text[2] in "0123456789" or text[2] in dict_char_to_int)
        and (text[3] in "0123456789" or text[3] in dict_char_to_int)
        and (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char)
        and (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char)
        and (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char)
    )


def format_license(text):
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int,
        4: dict_int_to_char,
        5: dict_int_to_char,
        6: dict_int_to_char,
    }
    license_plate_ = ""
    for j in range(7):
        if text[j] in mapping[j]:
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_


def pick_license_plate(detected_chars):

    for text, score in detected_chars:
        text = text.upper().replace(" ", "")

        if license_complies_format(text):
            return format_license(text), score
        
    return None, None


def read_license_plates(license_plate_crops):
    if not license_plate_crops:
        return []
    return [
        pick_license_plate(detected_chars)
        for detected_chars in paddle_readtext(license_plate_crops)
    ]
