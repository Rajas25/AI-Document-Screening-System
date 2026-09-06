import os
# Set DeepFace backend to PyTorch (no TensorFlow DLL issues)
os.environ["DEEPFACE_BACKEND"] = "torch"
# Suppress TensorFlow logs if it's still loaded
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
import tempfile
import hashlib
from datetime import datetime

import easyocr
from deepface import DeepFace


# ============================================================
# OPTIONAL TAMPER ENGINE
# ============================================================

try:
    from tamper_engine import analyze_metadata
    TAMPER_ENGINE_AVAILABLE = True
except Exception:
    TAMPER_ENGINE_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SSB AI Border Screening System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based Fake Identity & Document Screening System")

st.caption(
    "AI-assisted document, OCR, tamper and biometric screening prototype"
)


# ============================================================
# SESSION STATE
# ============================================================

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if "audit_chain" not in st.session_state:
    st.session_state.audit_chain = "GENESIS"

if "last_screening" not in st.session_state:
    st.session_state.last_screening = None

if "niko_messages" not in st.session_state:
    st.session_state.niko_messages = [
        {
            "role": "assistant",
            "content": "Hi, I’m NIKO 🧠 — your screening intelligence assistant. Ask me why a result was flagged, what to check next, or how the screening pipeline works."
        }
    ]


# ============================================================
# LOAD OCR
# ============================================================

@st.cache_resource
def load_ocr():

    return easyocr.Reader(
        ["en"],
        gpu=False
    )


reader = load_ocr()


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    cascade_path = (
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    return cv2.CascadeClassifier(
        cascade_path
    )


face_detector = load_face_detector()


# ============================================================
# IMAGE UTILITIES
# ============================================================

def decode_image(uploaded_file):

    data = uploaded_file.getvalue()

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    return image


def image_hash(image):

    success, encoded = cv2.imencode(
        ".jpg",
        image
    )

    if not success:
        return "HASH_ERROR"

    return hashlib.sha256(
        encoded.tobytes()
    ).hexdigest()


# ============================================================
# OCR TEXT NORMALIZATION
# ============================================================

def normalize_ocr_digits(text):

    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "!": "1",
        "S": "5",
        "B": "8",
        "G": "6"
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    return text


def clean_text(text):

    text = str(text)

    text = text.replace(
        "\n",
        " "
    )

    return text.strip()


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_ocr_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Large upscale for small document text
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


def preprocess_ocr_threshold(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return threshold


# ============================================================
# OCR WITH BOUNDING BOXES
# ============================================================

def run_ocr(image):

    all_results = []

    variants = [
        image,
        preprocess_ocr_image(image),
        preprocess_ocr_threshold(image)
    ]

    for variant in variants:

        try:

            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False
            )

            if results:
                all_results.extend(
                    results
                )

        except Exception:
            pass

    return all_results


def ocr_results_to_text(results):

    texts = []

    for item in results:

        try:
            text = clean_text(
                item[1]
            )

            if text:
                texts.append(
                    text
                )

        except Exception:
            continue

    # Remove duplicates
    unique = list(
        dict.fromkeys(texts)
    )

    return " ".join(
        unique
    )


# ============================================================
# VERHOEFF VALIDATION
# ============================================================

def validate_verhoeff(number):

    number = str(number)

    if not number.isdigit():
        return False

    if len(number) != 12:
        return False

    multiplication_table = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]

    permutation_table = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]

    checksum = 0

    reversed_number = number[::-1]

    for i, digit in enumerate(
        reversed_number
    ):

        checksum = multiplication_table[
            checksum
        ][
            permutation_table[i % 8][int(digit)]
        ]

    return checksum == 0


# ============================================================
# EXTRACT 4-4-4 NUMBER FROM OCR BOXES
# ============================================================

def get_box_center(box):

    xs = [
        point[0]
        for point in box
    ]

    ys = [
        point[1]
        for point in box
    ]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys)
    )


def extract_four_digit_token(text):

    normalized = normalize_ocr_digits(
        text
    )

    digits = re.sub(
        r"[^0-9]",
        "",
        normalized
    )

    if len(digits) == 4:
        return digits

    return None


def extract_grouped_number_candidates(results):

    tokens = []

    for item in results:

        try:

            box = item[0]
            text = item[1]
            confidence = float(
                item[2]
            )

            token = extract_four_digit_token(
                text
            )

            if token:

                cx, cy = get_box_center(
                    box
                )

                tokens.append({
                    "digits": token,
                    "x": cx,
                    "y": cy,
                    "confidence": confidence
                })

        except Exception:
            continue

    if len(tokens) < 3:
        return []


    # Sort by vertical position
    tokens = sorted(
        tokens,
        key=lambda t: t["y"]
    )

    candidates = []

    # --------------------------------------------------------
    # Group tokens that lie approximately on the same line
    # --------------------------------------------------------

    for i in range(
        len(tokens)
    ):

        first = tokens[i]

        same_line = []

        for token in tokens:

            if abs(
                token["y"] -
                first["y"]
            ) < 45:

                same_line.append(
                    token
                )

        same_line.sort(
            key=lambda t: t["x"]
        )

        # Find three consecutive 4-digit tokens
        for j in range(
            len(same_line) - 2
        ):

            a = same_line[j]
            b = same_line[j + 1]
            c = same_line[j + 2]

            # Reasonable horizontal gaps
            gap1 = b["x"] - a["x"]
            gap2 = c["x"] - b["x"]

            if gap1 <= 0 or gap2 <= 0:
                continue

            if gap1 > 500 or gap2 > 500:
                continue

            candidate = (
                a["digits"] +
                b["digits"] +
                c["digits"]
            )

            if validate_verhoeff(
                candidate
            ):

                avg_conf = (
                    a["confidence"] +
                    b["confidence"] +
                    c["confidence"]
                ) / 3

                candidates.append({
                    "number": candidate,
                    "y": (
                        a["y"] +
                        b["y"] +
                        c["y"]
                    ) / 3,
                    "confidence": avg_conf
                })

    return candidates


# ============================================================
# AADHAAR NUMBER EXTRACTION
# ============================================================

def extract_aadhaar_number(
    results,
    image
):

    candidates = (
        extract_grouped_number_candidates(
            results
        )
    )

    # --------------------------------------------------------
    # Fallback: search OCR strings for 4-4-4
    # --------------------------------------------------------

    full_text = ocr_results_to_text(
        results
    )

    normalized = normalize_ocr_digits(
        full_text
    )

    grouped_matches = re.findall(
        r"(\d{4})[\s\-]+(\d{4})[\s\-]+(\d{4})",
        normalized
    )

    for match in grouped_matches:

        candidate = "".join(
            match
        )

        if validate_verhoeff(
            candidate
        ):

            candidates.append({
                "number": candidate,
                "y": image.shape[0] * 0.70,
                "confidence": 0.5
            })

    if not candidates:
        return None, 0.0

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = {}

    for candidate in candidates:

        number = candidate["number"]

        if (
            number not in unique
            or candidate["confidence"] >
            unique[number]["confidence"]
        ):

            unique[number] = candidate

    candidates = list(
        unique.values()
    )

    # --------------------------------------------------------
    # Important:
    # Aadhaar number normally appears ABOVE the VID.
    #
    # Therefore, if multiple valid 4-4-4 numbers are found,
    # prefer the upper candidate.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x["y"],
            -x["confidence"]
        )
    )

    selected = candidates[0]

    return (
        selected["number"],
        round(
            selected["confidence"] * 100,
            1
        )
    )


# ============================================================
# MASK AADHAAR
# ============================================================

def mask_aadhaar(number):

    if not number:
        return "Not detected"

    return (
        "XXXX XXXX " +
        number[-4:]
    )


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

PAN_PATTERN = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"


def validate_pan(pan):
    if not pan:
        return False, "PAN not detected"
    pan = re.sub(r"\s+", "", str(pan).upper())
    return (True, "Valid PAN format") if re.fullmatch(PAN_PATTERN, pan) else (False, "Invalid PAN format")


def extract_pan_number(results):
    candidates = []
    for item in results or []:
        try:
            text = str(item[1]).upper()
            confidence = float(item[2])
        except Exception:
            continue
        compact = re.sub(r"[^A-Z0-9]", "", text)
        for m in re.finditer(r"[A-Z0-9]{10}", compact):
            candidate = m.group(0)
            if re.fullmatch(PAN_PATTERN, candidate):
                candidates.append((candidate, confidence))
        spaced = re.sub(r"[^A-Z0-9 ]", " ", text)
        m = re.search(r"[A-Z]{5}\s*[0-9]{4}\s*[A-Z]", spaced)
        if m:
            candidate = re.sub(r"\s+", "", m.group(0))
            if re.fullmatch(PAN_PATTERN, candidate):
                candidates.append((candidate, confidence))
    if not candidates:
        compact = re.sub(r"[^A-Z0-9]", "", str(ocr_results_to_text(results or [])).upper())
        for m in re.finditer(r"[A-Z0-9]{10}", compact):
            candidate = m.group(0)
            if re.fullmatch(PAN_PATTERN, candidate):
                candidates.append((candidate, 0.50))
    if not candidates:
        return None, 0.0
    best = {}
    for number, confidence in candidates:
        best[number] = max(confidence, best.get(number, 0.0))
    number = max(best, key=best.get)
    return number, round(best[number] * 100, 1)


def detect_document_type(text, aadhaar_number=None, pan_number=None):
    upper = str(text or "").upper()
    scores = {"AADHAAR": 0, "PAN": 0, "PASSPORT": 0, "VISA": 0}
    for k in ["AADHAAR", "AADHAR", "UIDAI", "UNIQUE IDENTIFICATION"]:
        if k in upper: scores["AADHAAR"] += 3
    if "GOVERNMENT OF INDIA" in upper: scores["AADHAAR"] += 2
    if aadhaar_number: scores["AADHAAR"] += 7
    if re.search(r"\b\d{4}\s*\d{4}\s*\d{4}\b", upper): scores["AADHAAR"] += 5
    for k in ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "PAN APPLICATION"]:
        if k in upper: scores["PAN"] += 5
    if pan_number: scores["PAN"] += 8
    if re.search(PAN_PATTERN, re.sub(r"\s+", "", upper)): scores["PAN"] += 5
    for k in ["PASSPORT", "REPUBLIC OF INDIA", "NATIONALITY", "P<"]:
        if k in upper: scores["PASSPORT"] += 4
    if "VISA" in upper: scores["VISA"] += 4
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else "UNKNOWN"


# ============================================================
# DOCUMENT ORIENTATION
# ============================================================

def score_orientation(image):

    """
    Score an orientation using:
    - Aadhaar keywords
    - grouped Aadhaar numbers
    - Government of India
    - passport keywords
    """

    try:

        # Smaller image for faster OCR
        h, w = image.shape[:2]

        scale = min(
            1.0,
            1000 / max(h, w)
        )

        if scale < 1:

            small = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

        else:

            small = image

        # One OCR pass
        results = reader.readtext(
            small,
            detail=1,
            paragraph=False
        )

        text = ocr_results_to_text(
            results
        ).upper()

        score = 0

        if "AADHAAR" in text: score += 10
        if "AADHAR" in text: score += 8
        if "GOVERNMENT OF INDIA" in text: score += 6
        if "UNIQUE IDENTIFICATION" in text: score += 8
        if re.search(r"\d{4}\s+\d{4}\s+\d{4}", text): score += 10

        if "INCOME TAX DEPARTMENT" in text: score += 12
        if "PERMANENT ACCOUNT NUMBER" in text: score += 12
        if re.search(PAN_PATTERN, re.sub(r"\s+", "", text)): score += 14

        if "PASSPORT" in text: score += 10
        if "REPUBLIC OF INDIA" in text: score += 5
        if "VISA" in text: score += 6

        # Prefer landscape card
        h, w = image.shape[:2]

        if w > h:
            score += 2

        return score

    except Exception:

        return 0


def auto_orient_document(image):

    orientations = [
        (
            "Original",
            image
        ),
        (
            "90° Clockwise",
            cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE
            )
        ),
        (
            "90° Counterclockwise",
            cv2.rotate(
                image,
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )
        ),
        (
            "180°",
            cv2.rotate(
                image,
                cv2.ROTATE_180
            )
        )
    ]

    best_name = "Original"
    best_image = image
    best_score = -1

    for name, candidate in orientations:

        score = score_orientation(
            candidate
        )

        if score > best_score:

            best_score = score
            best_name = name
            best_image = candidate

    return (
        best_image,
        best_name,
        best_score
    )


# ============================================================
# DOCUMENT PORTRAIT REGIONS
# ============================================================

def get_aadhaar_portrait_roi(image):
    h, w = image.shape[:2]
    return image[int(h * 0.20):int(h * 0.78), int(w * 0.035):int(w * 0.38)]


def get_pan_portrait_roi(image):
    h, w = image.shape[:2]
    return image[int(h * 0.20):int(h * 0.62), int(w * 0.025):int(w * 0.31)]


def get_passport_portrait_roi(image):
    h, w = image.shape[:2]
    return image[int(h * 0.10):int(h * 0.82), int(w * 0.03):int(w * 0.50)]


def get_document_portrait_roi(image, document_type):
    if document_type == "AADHAAR": return get_aadhaar_portrait_roi(image)
    if document_type == "PAN": return get_pan_portrait_roi(image)
    if document_type == "PASSPORT": return get_passport_portrait_roi(image)
    return None


def detect_face_in_roi(roi):
    if roi is None or roi.size == 0: return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scale = 3
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    enlarged = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enlarged)
    faces = face_detector.detectMultiScale(enlarged, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0: return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = [int(v / scale) for v in (x, y, fw, fh)]
    mx, my = int(fw * 0.25), int(fh * 0.30)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(roi.shape[1], x + fw + mx), min(roi.shape[0], y + fh + my)
    return roi[y1:y2, x1:x2]


def extract_document_face(image, document_type):
    roi = get_document_portrait_roi(image, document_type)
    if roi is None: return None, None
    face = detect_face_in_roi(roi)
    if face is not None: return face, roi
    h, w = image.shape[:2]
    if document_type == "AADHAAR": expanded = image[int(h * 0.12):int(h * 0.85), 0:int(w * 0.48)]
    elif document_type == "PAN": expanded = image[int(h * 0.10):int(h * 0.75), 0:int(w * 0.38)]
    elif document_type == "PASSPORT": expanded = image[int(h * 0.08):int(h * 0.85), 0:int(w * 0.58)]
    else: expanded = roi
    face = detect_face_in_roi(expanded)
    return (face, expanded) if face is not None else (None, roi)


def extract_aadhaar_face(image):
    return extract_document_face(image, "AADHAAR")


# ============================================================
# FACE QUALITY
# ============================================================

def calculate_face_quality(
    face
):

    if face is None:
        return 0

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # Sharpness
    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Brightness
    brightness = float(
        np.mean(gray)
    )

    sharpness_score = min(
        100,
        sharpness / 3
    )

    brightness_score = 100

    if brightness < 45:

        brightness_score = 50

    elif brightness > 220:

        brightness_score = 50

    quality = (
        sharpness_score * 0.6 +
        brightness_score * 0.4
    )

    return round(
        min(100, quality),
        1
    )


# ============================================================
# FACE PREPROCESSING
# ============================================================

def preprocess_face(
    face
):

    if face is None:
        return None

    face = cv2.resize(
        face,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    lab = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(
        l
    )

    lab = cv2.merge(
        (l, a, b)
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return result


# ============================================================
# SAVE TEMP FACE
# ============================================================

def save_temp_image(
    image
):

    file = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    )

    path = file.name

    file.close()

    cv2.imwrite(
        path,
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    return path


# ============================================================
# DEEPFACE COMPARISON
# ============================================================

def deepface_compare(
    document_face,
    passenger_face
):

    doc_path = None
    passenger_path = None

    try:

        doc_path = save_temp_image(
            document_face
        )

        passenger_path = save_temp_image(
            passenger_face
        )

        # ----------------------------------------------------
        # First try ArcFace
        # ----------------------------------------------------

        try:

            result = DeepFace.verify(
                img1_path=doc_path,
                img2_path=passenger_path,

                model_name="ArcFace",

                detector_backend="opencv",

                distance_metric="cosine",

                enforce_detection=False,

                align=True
            )

            model_used = "ArcFace"

        except Exception:

            # ------------------------------------------------
            # Fallback to Facenet512
            # ------------------------------------------------

            result = DeepFace.verify(
                img1_path=doc_path,
                img2_path=passenger_path,

                model_name="Facenet512",

                detector_backend="opencv",

                distance_metric="cosine",

                enforce_detection=False,

                align=True
            )

            model_used = "Facenet512"

        distance = float(
            result.get(
                "distance",
                999
            )
        )

        threshold = float(
            result.get(
                "threshold",
                0.4
            )
        )

        # ----------------------------------------------------
        # Convert distance into a display score
        #
        # This is NOT a probability.
        # ----------------------------------------------------

        score = (
            1 -
            (
                distance /
                max(
                    threshold * 1.5,
                    0.0001
                )
            )
        ) * 100

        score = max(
            0,
            min(
                100,
                score
            )
        )

        score = round(
            score,
            1
        )

        # ----------------------------------------------------
        # Graded biometric result
        # ----------------------------------------------------

        if result.get(
            "verified",
            False
        ):

            status = "STRONG MATCH"

        elif score >= 55:

            status = "POSSIBLE MATCH"

        elif score >= 40:

            status = "LOW CONFIDENCE"

        else:

            status = "NO SUFFICIENT MATCH"

        return {
            "status": status,
            "score": score,
            "distance": round(
                distance,
                4
            ),
            "threshold": round(
                threshold,
                4
            ),
            "model": model_used
        }

    except Exception as e:

        return {
            "status": "BIOMETRIC ERROR",
            "score": 0,
            "distance": None,
            "threshold": None,
            "model": "ERROR",
            "error": str(e)
        }

    finally:

        if doc_path:

            try:
                os.remove(
                    doc_path
                )
            except:
                pass

        if passenger_path:

            try:
                os.remove(
                    passenger_path
                )
            except:
                pass


# ============================================================
# PASSENGER FACE EXTRACTION
# ============================================================

def extract_passenger_face(image):
    """
    Detect the passenger's face from the supplied passenger image.
    Uses the same face detector as the document pipeline.
    Returns None if no face is confidently detected.
    """
    if image is None:
        return None

    try:
        face = detect_face_in_roi(image)
        return face
    except Exception:
        return None


# ============================================================
# BIOMETRIC PIPELINE
# ============================================================

def run_biometric(
    document_image,
    passenger_image,
    document_type="AADHAAR"
):

    document_face, portrait_roi = (
        extract_document_face(
            document_image,
            document_type
        )
    )

    passenger_face = (
        extract_passenger_face(
            passenger_image
        )
    )

    result = {
        "document_face": document_face,
        "passenger_face": passenger_face,
        "portrait_roi": portrait_roi
    }

    if document_face is None:

        result.update({
            "status":
                "DOCUMENT_FACE_NOT_FOUND",
            "score": 0,
            "distance": None,
            "threshold": None,
            "model": "N/A"
        })

        return result

    if passenger_face is None:

        result.update({
            "status":
                "PASSENGER_FACE_NOT_FOUND",
            "score": 0,
            "distance": None,
            "threshold": None,
            "model": "N/A"
        })

        return result

    document_face = preprocess_face(
        document_face
    )

    passenger_face = preprocess_face(
        passenger_face
    )

    result.update(
        deepface_compare(
            document_face,
            passenger_face
        )
    )

    result[
        "document_quality"
    ] = calculate_face_quality(
        document_face
    )

    result[
        "passenger_quality"
    ] = calculate_face_quality(
        passenger_face
    )

    result[
        "document_face"
    ] = document_face

    result[
        "passenger_face"
    ] = passenger_face

    return result


# ============================================================
# BASIC ELA
# ============================================================

def calculate_ela(
    image
):

    try:
        temp = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        )
        temp.close()
        path = temp.name

        cv2.imwrite(
            path,
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        compressed = cv2.imread(path)

        try:
            os.remove(path)
        except Exception:
            pass

        if compressed is None:
            return 0

        if compressed.shape[:2] != image.shape[:2]:
            compressed = cv2.resize(
                compressed,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_AREA
            )

        diff = cv2.absdiff(image, compressed)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        return round(float(np.mean(gray)), 2)

    except Exception:
        return 0


def analyze_image_integrity(image):
    """Multi-scale image-integrity analysis with localized anomaly detection.

    ELA alone is intentionally not treated as a forgery detector: small edits can
    disappear in a global average. This function combines multi-quality ELA,
    local ELA contrast, high-frequency residuals and edge/texture discontinuities.
    """
    try:
        if image is None or image.size == 0:
            raise ValueError("Empty image")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multi-quality JPEG residuals. Genuine images can have ELA too, so use
        # relative/local statistics rather than a single absolute threshold.
        ela_maps = []
        for quality in (75, 85, 95):
            ok, enc = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is None:
                continue
            if dec.shape[:2] != image.shape[:2]:
                dec = cv2.resize(dec, (w, h), interpolation=cv2.INTER_AREA)
            d = cv2.absdiff(image, dec)
            ela_maps.append(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32))

        if not ela_maps:
            raise ValueError("JPEG residual unavailable")

        ela_map = np.max(np.stack(ela_maps, axis=0), axis=0)
        global_ela = round(float(np.mean(ela_map)), 2)

        # Work at patch scale so a small changed region cannot be diluted by the
        # rest of a document. Patch size adapts to document resolution.
        block = int(max(24, min(64, min(h, w) // 12 if min(h, w) else 24)))
        stats = []
        for y in range(0, h, block):
            for x in range(0, w, block):
                pch = ela_map[y:min(y+block,h), x:min(x+block,w)]
                if pch.size < 100:
                    continue
                lap = cv2.Laplacian(gray[y:min(y+block,h), x:min(x+block,w)], cv2.CV_32F)
                edges = cv2.Canny(gray[y:min(y+block,h), x:min(x+block,w)], 60, 140)
                stats.append({
                    'x': x, 'y': y,
                    'x2': min(x+block,w), 'y2': min(y+block,h),
                    'ela': float(np.mean(pch)),
                    'ela_hi': float(np.percentile(pch, 95)),
                    'texture': float(np.std(pch)),
                    'lap': float(np.var(lap)),
                    'edge': float(np.mean(edges > 0))
                })

        if len(stats) < 4:
            return {'score': global_ela, 'status':'UNAVAILABLE', 'global_ela':global_ela,
                    'local_max':0.0, 'local_p95':0.0, 'hotspot_count':0,
                    'affected_area_pct':0.0, 'tamper_evidence_score':0.0,
                    'ela_pixel_hotspots':0, 'multi_quality_gain':0.0}

        vals = np.array([z['ela'] for z in stats], dtype=np.float32)
        hi = np.array([z['ela_hi'] for z in stats], dtype=np.float32)
        lapv = np.array([z['lap'] for z in stats], dtype=np.float32)
        edgev = np.array([z['edge'] for z in stats], dtype=np.float32)

        med = float(np.median(vals)); mad = float(np.median(np.abs(vals-med)))
        sigma = max(1.4826 * mad, 0.75)
        # Adaptive relative threshold plus an absolute floor.
        patch_threshold = max(2.5, med + 3.5*sigma)
        hot = vals >= patch_threshold

        # Pixel-level residual hotspots. Morphology removes isolated JPEG noise.
        pixel_base = max(5.0, float(np.percentile(ela_map, 85)))
        pixel_mask = (ela_map >= pixel_base).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        pixel_mask = cv2.morphologyEx(pixel_mask, cv2.MORPH_OPEN, kernel)
        pixel_mask = cv2.morphologyEx(pixel_mask, cv2.MORPH_CLOSE, kernel)
        nlab, labels, comps, _ = cv2.connectedComponentsWithStats(pixel_mask, 8)
        min_component = max(20, int(h*w*0.00015))
        significant_components = [i for i in range(1,nlab) if comps[i,cv2.CC_STAT_AREA] >= min_component]

        # Local discontinuity: compare each patch to its spatial neighbours.
        anomaly_scores=[]
        for i,z in enumerate(stats):
            neighbours=[]
            cx=(z['x']+z['x2'])/2; cy=(z['y']+z['y2'])/2
            for j,q in enumerate(stats):
                if i==j: continue
                qx=(q['x']+q['x2'])/2; qy=(q['y']+q['y2'])/2
                if abs(cx-qx) <= block*1.6 and abs(cy-qy) <= block*1.6:
                    neighbours.append(j)
            if neighbours:
                nb=np.array([vals[j] for j in neighbours], dtype=np.float32)
                anomaly_scores.append(abs(z['ela']-float(np.median(nb))) / max(1.0,float(np.std(nb))+1.0))
            else:
                anomaly_scores.append(0.0)
        anomaly_scores=np.array(anomaly_scores,dtype=np.float32)
        local_discontinuities=int(np.sum(anomaly_scores>=2.5))

        # Multi-quality gain: edits often react differently across JPEG qualities.
        quality_gain=float(np.max([np.mean(m) for m in ela_maps])-np.min([np.mean(m) for m in ela_maps]))

        hotspot_count=int(np.sum(hot))
        affected_area=100.0*sum((z['x2']-z['x'])*(z['y2']-z['y']) for i,z in enumerate(stats) if hot[i])/max(1,h*w)
        local_max=float(np.max(hi)); local_p95=float(np.percentile(hi,95))

        # Evidence fusion. Pixel components and local discontinuity are important
        # because obvious small edits can have modest average ELA.
        intensity=np.clip((local_p95-med)/10.0,0,1)
        patch_component=np.clip(hotspot_count/max(1,len(stats)*0.04),0,1)
        pixel_component=np.clip(len(significant_components)/3.0,0,1)
        discontinuity_component=np.clip(local_discontinuities/3.0,0,1)
        area_component=np.clip(affected_area/8.0,0,1)
        quality_component=np.clip(quality_gain/2.0,0,1)
        evidence=100*(0.25*intensity+0.20*patch_component+0.25*pixel_component+
                      0.18*discontinuity_component+0.07*area_component+0.05*quality_component)
        evidence=round(float(min(100,evidence)),1)

        strong_spatial = (affected_area >= 1.5 and (len(significant_components) >= 3 or local_discontinuities >= 3))
        strong_component_signal = (len(significant_components) >= 5 and affected_area >= 0.8)
        if evidence >= 70 and (strong_spatial or strong_component_signal or local_discontinuities >= 5):
            status='POSSIBLE MANIPULATION'
        elif evidence >= 55 and affected_area >= 1.0 and (len(significant_components) >= 2 or local_discontinuities >= 2):
            status='REVIEW'
        else:
            status='NO SIGNIFICANT ANOMALY'

        return {
            'score': global_ela, 'status':status, 'global_ela':global_ela,
            'local_max':round(local_max,2), 'local_p95':round(local_p95,2),
            'hotspot_count':hotspot_count, 'affected_area_pct':round(affected_area,2),
            'tamper_evidence_score':evidence,
            'ela_pixel_hotspots':len(significant_components),
            'local_discontinuities':local_discontinuities,
            'multi_quality_gain':round(quality_gain,2)
        }
    except Exception:
        try: g=calculate_ela(image)
        except Exception: g=0.0
        return {'score':g,'status':'UNAVAILABLE','global_ela':g,'local_max':0.0,'local_p95':0.0,
                'hotspot_count':0,'affected_area_pct':0.0,'tamper_evidence_score':0.0,
                'ela_pixel_hotspots':0,'local_discontinuities':0,'multi_quality_gain':0.0}


# ============================================================
# METADATA
# ============================================================

def analyze_document_metadata(
    uploaded_file
):

    if not TAMPER_ENGINE_AVAILABLE:

        return {
            "status": "NOT AVAILABLE",
            "details": ""
        }

    try:

        result = analyze_metadata(
            uploaded_file
        )

        # Existing tamper_engine may return:
        #
        # (False, "Clean metadata structure.")
        #

        if isinstance(
            result,
            tuple
        ):

            if len(result) >= 2:

                suspicious = result[0]
                message = result[1]

                if suspicious:

                    return {
                        "status":
                            "SUSPICIOUS",
                        "details":
                            str(message)
                    }

                return {
                    "status":
                        "CLEAR",
                    "details":
                        str(message)
                }

        if isinstance(
            result,
            dict
        ):

            return {
                "status":
                    result.get(
                        "status",
                        "UNKNOWN"
                    ),
                "details":
                    str(result)
            }

        return {
            "status":
                "UNKNOWN",
            "details":
                str(result)
        }

    except Exception as e:

        return {
            "status":
                "ERROR",
            "details":
                str(e)
        }


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(
    document_type,
    aadhaar_valid,
    pan_valid,
    biometric_status,
    biometric_score,
    ela_status,
    metadata_status,
    face_found,
    tamper_evidence_score=0.0,
    synthetic_evidence_status="NOT ASSESSED"
):

    risk = 0
    reasons = []

    # --------------------------------------------------------
    # Document
    # --------------------------------------------------------

    if document_type == "UNKNOWN":

        risk += 15

        reasons.append(
            "Document type could not be confidently identified."
        )

    # --------------------------------------------------------
    # Document validation
    # --------------------------------------------------------

    if document_type == "AADHAAR" and not aadhaar_valid:
        risk += 20
        reasons.append("Aadhaar number could not be confidently validated.")
    if document_type == "PAN" and not pan_valid:
        risk += 20
        reasons.append("PAN number could not be confidently validated.")

    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    if not face_found:

        risk += 15

        reasons.append(
            "Document/passenger face extraction requires review."
        )

    # --------------------------------------------------------
    # Biometrics
    # --------------------------------------------------------

    if biometric_status == "STRONG MATCH":

        risk += 0

    elif biometric_status == "POSSIBLE MATCH":

        risk += 5

        reasons.append(
            "Biometric result is not sufficiently strong."
        )

    elif biometric_status == "LOW CONFIDENCE":

        risk += 15

        reasons.append(
            "Low biometric similarity."
        )

    elif biometric_status == "NO SUFFICIENT MATCH":

        risk += 25

        reasons.append(
            "Insufficient biometric similarity."
        )

    elif biometric_status == "BIOMETRIC ERROR":

        risk += 10

        reasons.append(
            "Biometric engine encountered an error."
        )

    # --------------------------------------------------------
    # Image integrity
    # --------------------------------------------------------

    if ela_status == "POSSIBLE MANIPULATION" and tamper_evidence_score >= 70:
        risk += 15
        reasons.append("Corroborated localized image-integrity anomalies are consistent with possible manipulation.")
    elif ela_status == "REVIEW" and tamper_evidence_score >= 55:
        risk += 5
        reasons.append("Corroborated localized image-integrity evidence requires secondary inspection.")

    if tamper_evidence_score >= 85:
        risk += 5
        reasons.append("Very strong localized forensic anomaly signal.")

    if synthetic_evidence_status == "REVIEW":
        risk += 5
        reasons.append("Synthetic/reconstruction indicators require review.")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if metadata_status == "SUSPICIOUS":

        risk += 15

        reasons.append(
            "Suspicious document metadata."
        )

    # --------------------------------------------------------
    # Cap risk
    # --------------------------------------------------------

    risk = min(
        risk,
        100
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if risk < 20:

        level = "LOW"

        directive = (
            "CLEAR — ROUTINE PROCESSING"
        )

    elif risk < 45:

        level = "MEDIUM"

        directive = (
            "REFER TO SECONDARY INSPECTION"
        )

    elif risk < 70:

        level = "HIGH"

        directive = (
            "SECONDARY VERIFICATION REQUIRED"
        )

    else:

        level = "CRITICAL"

        directive = (
            "ESCALATE FOR HUMAN REVIEW"
        )

    return {
        "risk": risk,
        "level": level,
        "directive": directive,
        "reasons": reasons
    }


# ============================================================
# NIKO + EXPLAINABILITY + INTEGRITY EXTENSIONS
# ============================================================

def build_explainable_factors(document_type, aadhaar_valid, pan_valid, biometric_status,
                              biometric_score, ela_status, ela_score, metadata_status, face_found,
                              integrity=None, synthetic_evidence_status="NOT ASSESSED"):
    factors = [{
        "Signal": "Document type",
        "Finding": f"{document_type} identified" if document_type != "UNKNOWN" else "Unknown / not confidently classified",
        "Impact": 0 if document_type != "UNKNOWN" else 15
    }]
    if document_type == "AADHAAR":
        factors.append({"Signal": "Aadhaar validation", "Finding": "PASS" if aadhaar_valid else "NOT CONFIRMED", "Impact": 0 if aadhaar_valid else 20})
    elif document_type == "PAN":
        factors.append({"Signal": "PAN validation", "Finding": "PASS" if pan_valid else "NOT CONFIRMED", "Impact": 0 if pan_valid else 20})
    else:
        factors.append({"Signal": "Document validation", "Finding": "Not applicable", "Impact": 0})

    factors.append({"Signal": "Document/passenger face extraction", "Finding": "Available" if face_found else "Requires review", "Impact": 0 if face_found else 15})
    bio_impacts = {"STRONG MATCH": 0, "POSSIBLE MATCH": 5, "LOW CONFIDENCE": 15, "NO SUFFICIENT MATCH": 25, "BIOMETRIC ERROR": 10}
    factors.append({"Signal": "Biometric comparison", "Finding": f"{biometric_status} ({biometric_score:.1f} screening score)", "Impact": bio_impacts.get(biometric_status, 10)})

    ela_impact = {"NO SIGNIFICANT ANOMALY": 0, "REVIEW": 5, "POSSIBLE MANIPULATION": 20, "UNAVAILABLE": 0}
    if integrity:
        finding = (
            f"{ela_status}; global={integrity.get('global_ela', ela_score)}, "
            f"local max={integrity.get('local_max', 0)}, "
            f"hotspots={integrity.get('hotspot_count', 0)}, "
            f"affected={integrity.get('affected_area_pct', 0)}%"
        )
    else:
        finding = f"{ela_status} ({ela_score})"
    factors.append({"Signal": "Localized image integrity", "Finding": finding, "Impact": ela_impact.get(ela_status, 0)})

    factors.append({"Signal": "Synthetic / reconstruction indicators", "Finding": synthetic_evidence_status, "Impact": 5 if synthetic_evidence_status == "REVIEW" else 0})
    factors.append({"Signal": "Metadata", "Finding": metadata_status, "Impact": 15 if metadata_status == "SUSPICIOUS" else 0})
    return factors

def compute_audit_chain(record):
    payload = "|".join(f"{k}={record.get(k, '')}" for k in sorted(record))
    previous = st.session_state.audit_chain
    chain_hash = hashlib.sha256((previous + "|" + payload).encode("utf-8")).hexdigest()
    st.session_state.audit_chain = chain_hash
    return previous, chain_hash


def niko_reply(question, screening=None):
    q = question.lower().strip()
    if screening is None:
        return "Run a screening first. Once results are available, I can explain the risk, biometric result, document checks, tamper indicators, and recommended next step."

    if any(x in q for x in ["why", "risk", "flag", "suspicious"]):
        reasons = screening.get("reasons", [])
        if reasons:
            return "The current screening risk is **{}% ({})**. Main contributing signals: {}. These are screening indicators, not proof of fraud.".format(
                screening["risk"], screening["level"], "; ".join(reasons)
            )
        return "The current screening risk is **{}% ({})** and no major automated risk indicators were recorded.".format(
            screening["risk"], screening["level"]
        )

    if any(x in q for x in ["next", "do i do", "action", "proceed"]):
        return "Recommended workflow: **{}**. If the result is not clearly clean, perform authorized secondary inspection and verify against the authoritative source. Do not treat the AI score alone as an adverse identity decision.".format(screening["directive"])

    if any(x in q for x in ["biometric", "face", "match", "similarity"]):
        return "Biometric result: **{}**, with a screening score of **{:.1f}** using **{}**. Facial hair, lighting, pose, image quality, and document-photo quality can affect similarity, so borderline results should be reviewed by an authorized officer.".format(
            screening["biometric_status"], screening["biometric_score"], screening["model"]
        )

    if any(x in q for x in ["tamper", "ela", "edit", "forg"]):
        return "Tamper indicators show **{}**, global ELA **{}**, local max **{}**, **{} hotspots**, affected area **{}%**, and metadata **{}**. Synthetic/reconstruction indicator: **{}**. These are forensic heuristics, not proof of forgery or AI generation.".format(
            screening["ela_status"], screening.get("ela_global", screening["ela_score"]),
            screening.get("ela_local_max", 0), screening.get("ela_hotspots", 0),
            screening.get("ela_affected_area_pct", 0), screening["metadata_status"],
            screening.get("synthetic_evidence_status", "NOT ASSESSED")
        )

    if any(x in q for x in ["hash", "duplicate", "integrity"]):
        return "This screening has a SHA-256 document fingerprint and a chained audit hash. A repeated fingerprint can identify a previously screened image, while the chained hash makes later audit-log modification detectable." 

    if any(x in q for x in ["how", "pipeline", "working", "process"]):
        return "Pipeline: **orientation → OCR → document classification → document-specific identifier validation → document portrait ROI → face detection → DeepFace comparison → metadata/ELA → explainable risk fusion → audit hash**."

    if any(x in q for x in ["help", "guide", "what can you do"]):
        return "I can explain the risk score, biometric result, tamper indicators, Aadhaar validation, audit integrity, duplicate detection, and recommended workflow. Try: *Why is the risk high?*, *Explain biometric result*, or *What should I do next?*"

    return "I can guide you through the screening result. Try asking **why the risk was assigned**, **explain the biometric result**, **what tamper indicators mean**, or **what to do next**."


def create_tamper_variants(image):
    variants = {"JPEG Recompression": image.copy()}
    try:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 35])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                variants["JPEG Recompression"] = decoded
    except Exception:
        pass

    # Synthetic copy-move test: duplicate a small neutral image patch.
    h, w = image.shape[:2]
    if h > 80 and w > 120:
        copy_move = image.copy()
        ph, pw = max(20, h // 10), max(30, w // 10)
        y1, x1 = max(0, h // 3), max(0, w // 3)
        y2, x2 = min(h, y1 + ph), min(w, x1 + pw)
        patch = copy_move[y1:y2, x1:x2].copy()
        dy, dx = min(h - (y2-y1), y1 + h // 5), min(w - (x2-x1), x1 + w // 5)
        copy_move[dy:dy+(y2-y1), dx:dx+(x2-x1)] = patch
        variants["Synthetic Copy-Move"] = copy_move

        # Synthetic overlay test; deliberately obvious and only for validation.
        overlay = image.copy()
        cv2.rectangle(overlay, (max(0, w//2), max(0, h-70)), (w-5, h-5), (255, 255, 255), -1)
        cv2.putText(overlay, "TEST ALTERATION", (max(5, w//2+5), max(25, h-30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        variants["Synthetic Text Overlay"] = overlay

        reconstruction = image.copy()
        ry1, rx1 = max(0, h // 5), max(0, w // 5)
        ry2, rx2 = min(h, ry1 + max(30, h // 4)), min(w, rx1 + max(40, w // 4))
        region = reconstruction[ry1:ry2, rx1:rx2].copy()
        if region.size:
            region = cv2.GaussianBlur(region, (0, 0), 2.2)
            reconstruction[ry1:ry2, rx1:rx2] = region
        variants["Synthetic Reconstruction Stress Test"] = reconstruction
    return variants


def analyze_synthetic_evidence(image, integrity=None):
    """Synthetic/reconstruction indicator; not a definitive AI detector."""
    try:
        integrity = integrity or analyze_image_integrity(image)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape
        block=max(32,min(96,min(h,w)//8 if min(h,w) else 32))
        tex=[]; edges=[]; hf=[]
        for yy in range(0,h,block):
            for xx in range(0,w,block):
                p=gray[yy:min(yy+block,h),xx:min(xx+block,w)]
                if p.size<100: continue
                tex.append(float(np.std(p)))
                e=cv2.Canny(p,60,140); edges.append(float(np.mean(e>0)))
                hf.append(float(np.var(cv2.Laplacian(p,cv2.CV_32F))))
        if len(tex)<4: return {'status':'NOT ASSESSED','score':0.0,'reason':'Insufficient image area.'}
        tex_cv=float(np.std(tex)/max(np.mean(tex),1)); edge_cv=float(np.std(edges)/max(np.mean(edges),.01))
        hf_cv=float(np.std(hf)/max(np.mean(hf),1))
        fs=float(integrity.get('tamper_evidence_score',0))
        # Texture statistics alone are not enough: normal camera/JPEG images can
        # be highly non-uniform. Require corroborating localized forensic evidence
        # before allowing this layer to become a strong synthetic signal.
        if fs < 55:
            score = 0
        else:
            score=0
            score += 20 if tex_cv>.55 else (6 if tex_cv>.35 else 0)
            score += 15 if edge_cv>.65 else (5 if edge_cv>.45 else 0)
            score += 15 if hf_cv>.80 else (5 if hf_cv>.50 else 0)
            score += 35 if fs>=70 else (25 if fs>=60 else 15)
            score += 15 if integrity.get('ela_pixel_hotspots',0)>=3 else 0
            score += 10 if integrity.get('affected_area_pct',0)>=1.5 else 0
        score=round(min(100,float(score)),1)
        status='REVIEW' if score>=65 else 'NO STRONG SYNTHETIC INDICATOR'
        reason=('Texture/edge/high-frequency inconsistency and localized forensic evidence require review.' if status=='REVIEW' else 'No strong synthetic/reconstruction indicator was found by this heuristic layer.')
        return {'status':status,'score':score,'reason':reason}
    except Exception as e:
        return {'status':'UNAVAILABLE','score':0.0,'reason':str(e)}


def render_niko(screening):
    st.subheader("🧠 NIKO — Screening Intelligence Assistant")
    st.caption("NIKO explains the automated findings and guides the operator. It does not replace authorized human verification.")

    for msg in st.session_state.niko_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask NIKO about this screening…", key="niko_chat")
    if prompt:
        st.session_state.niko_messages.append({"role": "user", "content": prompt})
        answer = niko_reply(prompt, screening)
        st.session_state.niko_messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ----------------------------------------------------------------------
# NEW: Standalone tamper upload & analysis inside TamperLab
# ----------------------------------------------------------------------

def render_tamper_upload():
    """
    Renders an upload widget and analysis button for users to check
    any image (downloaded or otherwise) for tampering indicators.
    """
    st.markdown("---")
    st.subheader("📤 Upload an Image for Tamper Analysis")
    st.caption(
        "Upload any image (JPEG, PNG) to run the same forensic analysis "
        "used in the main screening pipeline. Results are shown instantly."
    )

    uploaded_tamper_file = st.file_uploader(
        "Choose an image…",
        type=["jpg", "jpeg", "png"],
        key="tamper_upload"
    )

    if uploaded_tamper_file is not None:
        # Decode the image
        bytes_data = uploaded_tamper_file.getvalue()
        arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Could not decode the image. Please try another file.")
            return

        col_img, col_res = st.columns([1, 2])

        with col_img:
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption="Uploaded Image",
                use_container_width=True
            )
            st.caption(f"Resolution: {img.shape[1]} x {img.shape[0]}")

        with col_res:
            if st.button("🔍 Run Tamper Analysis", key="run_tamper_analysis"):
                with st.spinner("Analyzing image integrity…"):
                    integrity = analyze_image_integrity(img)
                    synthetic = analyze_synthetic_evidence(img, integrity)

                st.success("Analysis complete!")

                # Display key metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Global ELA", integrity.get("global_ela", 0.0))
                m2.metric("Local Max", integrity.get("local_max", 0.0))
                m3.metric("Hotspots", integrity.get("hotspot_count", 0))
                m4.metric("Forensic Score", f"{integrity.get('tamper_evidence_score', 0.0):.1f}/100")

                st.write(f"**Image‑integrity status:** {integrity.get('status', 'UNAVAILABLE')}")
                st.write(
                    f"**Synthetic/reconstruction indicator:** "
                    f"{synthetic.get('status', 'UNAVAILABLE')} — "
                    f"{synthetic.get('reason', '')}"
                )

                with st.expander("📊 Detailed Metrics"):
                    detail_df = pd.DataFrame([{
                        "Metric": "Affected area (%)",
                        "Value": integrity.get("affected_area_pct", 0.0)
                    }, {
                        "Metric": "ELA pixel hotspots",
                        "Value": integrity.get("ela_pixel_hotspots", 0)
                    }, {
                        "Metric": "Local discontinuities",
                        "Value": integrity.get("local_discontinuities", 0)
                    }, {
                        "Metric": "Multi‑quality gain",
                        "Value": integrity.get("multi_quality_gain", 0)
                    }, {
                        "Metric": "Synthetic score",
                        "Value": synthetic.get("score", 0.0)
                    }])
                    st.dataframe(detail_df, hide_index=True, use_container_width=True)

                st.caption(
                    "These are forensic screening indicators. "
                    "They do not independently prove that an image is fake or AI‑generated."
                )

        # Optionally, allow download of the report as JSON
        if st.button("📥 Download Report as JSON", key="download_tamper_json"):
            import json
            report = {
                "image": uploaded_tamper_file.name,
                "resolution": f"{img.shape[1]}x{img.shape[0]}",
                "integrity": integrity,
                "synthetic": synthetic
            }
            json_str = json.dumps(report, indent=2)
            st.download_button(
                label="Click to download",
                data=json_str,
                file_name="tamper_report.json",
                mime="application/json"
            )

# ----------------------------------------------------------------------
# END OF NEW SECTION
# ----------------------------------------------------------------------


def render_tamper_lab():
    st.header("🔬 TamperLab — Controlled Validation")
    st.caption("Generate controlled synthetic modifications to validate the forensic pipeline. Use synthetic/sample documents for testing, not real people's identity documents.")
    if st.session_state.last_screening is None:
        st.info("Run a screening first to load the current document into TamperLab.")
        return

    if st.button("🧪 Generate Synthetic Tamper Tests", type="secondary"):
        st.session_state.tamper_variants = create_tamper_variants(st.session_state.last_screening["document_img"])

    variants = st.session_state.get("tamper_variants", {})
    if not variants:
        st.info("Click the button above to generate controlled test variants.")
        return

    rows = []
    cols = st.columns(min(3, len(variants)))
    for idx, (name, img) in enumerate(variants.items()):
        with cols[idx % len(cols)]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=name, use_container_width=True)
        integrity = analyze_image_integrity(img)
        synthetic = analyze_synthetic_evidence(img, integrity)
        rows.append({
            "Variant": name,
            "Global ELA": integrity["global_ela"],
            "Local Max": integrity["local_max"],
            "Hotspots": integrity["hotspot_count"],
            "Affected Area %": integrity["affected_area_pct"],
            "Forensic Score": integrity["tamper_evidence_score"],
            "Pixel Hotspots": integrity.get("ela_pixel_hotspots", 0),
            "Local Discontinuities": integrity.get("local_discontinuities", 0),
            "Quality Gain": integrity.get("multi_quality_gain", 0),
            "ELA Status": integrity["status"],
            "Synthetic Score": synthetic.get("score", 0),
            "Synthetic Indicator": synthetic["status"]
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info("Interpretation: localized ELA hotspots are more useful than the old global mean alone. A high score is a reason for secondary inspection, not proof that a document is fake.")
    st.warning("AI-generated/reconstructed document detection is only a forensic indicator here. Validate it on a representative synthetic dataset before making operational claims.")

    # ----------------------------
    # NEW: Insert the upload widget
    # ----------------------------
    render_tamper_upload()


def render_analytics():
    st.header("📊 Screening Analytics")
    if not st.session_state.audit_log:
        st.info("No screening records yet.")
        return
    df = pd.DataFrame(st.session_state.audit_log)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Screenings", len(df))
    c2.metric("Average Risk", f"{pd.to_numeric(df['Risk'], errors='coerce').mean():.1f}%")
    c3.metric("High/Critical", int(df["Risk Level"].isin(["HIGH", "CRITICAL"]).sum()))
    c4.metric("Tamper Flags", int((df["ELA Status"].isin(["REVIEW", "POSSIBLE MANIPULATION"]) | (df["Metadata"] == "SUSPICIOUS")).sum()))
    st.bar_chart(df["Risk Level"].value_counts())

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.write(
        "SSB AI Border Screening Prototype"
    )

    st.divider()

    st.write(
        "**Detection modules**"
    )

    st.write(
        "✓ Automatic orientation"
    )

    st.write(
        "✓ Multi-pass OCR"
    )

    st.write(
        "✓ Aadhaar checksum"
    )

    st.write(
        "✓ Document portrait extraction"
    )

    st.write(
        "✓ Face comparison"
    )

    st.write(
        "✓ Image integrity analysis"
    )

    st.divider()

    st.info(
        "AI outputs are screening indicators. "
        "Final identity decisions require authorized "
        "human verification."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Audit Log",
        use_container_width=True
    ):

        st.session_state.audit_log = []

        st.success(
            "Audit log cleared."
        )


# ============================================================
# DOCUMENT INPUT
# ============================================================

st.header("📄 1. Document Input")

document_source = st.radio(
    "Document source",
    [
        "Upload Document",
        "Camera"
    ],
    horizontal=True
)

document_file = None

if document_source == "Upload Document":

    document_file = st.file_uploader(
        "Upload Aadhaar / PAN / Passport / Visa",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="document_upload"
    )

else:

    document_file = st.camera_input(
        "Capture document",
        key="document_camera"
    )


# ============================================================
# PASSENGER INPUT
# ============================================================

st.header("👤 2. Passenger Face")

passenger_source = st.radio(
    "Passenger face source",
    [
        "Upload Face Photo",
        "Camera"
    ],
    horizontal=True
)

passenger_file = None

if passenger_source == "Upload Face Photo":

    passenger_file = st.file_uploader(
        "Upload passenger face photograph",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="passenger_upload"
    )

else:

    passenger_file = st.camera_input(
        "Capture passenger face",
        key="passenger_camera"
    )


# ============================================================
# SCREEN BUTTON
# ============================================================

run_screening = st.button(
    "🔍 RUN AI SCREENING",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if run_screening:

    if document_file is None:

        st.error(
            "Please upload or capture a document."
        )

        st.stop()

    if passenger_file is None:

        st.error(
            "Please upload or capture the passenger face."
        )

        st.stop()


    # ========================================================
    # DECODE
    # ========================================================

    document_img = decode_image(
        document_file
    )

    passenger_img = decode_image(
        passenger_file
    )

    if document_img is None:

        st.error(
            "Unable to decode document image."
        )

        st.stop()

    if passenger_img is None:

        st.error(
            "Unable to decode passenger image."
        )

        st.stop()


    # ========================================================
    # ORIENTATION
    # ========================================================

    with st.spinner(
        "Detecting document orientation..."
    ):

        document_img, orientation, orientation_score = (
            auto_orient_document(
                document_img
            )
        )

    if orientation != "Original":

        st.info(
            f"↻ Document automatically rotated: "
            f"{orientation}"
        )

    else:

        st.success(
            "✓ Document orientation appears correct."
        )


    # ========================================================
    # INPUT PREVIEW
    # ========================================================

    st.header("🖼️ Input Images")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Document"
        )

        st.image(
            cv2.cvtColor(
                document_img,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Passenger"
        )

        st.image(
            cv2.cvtColor(
                passenger_img,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )


    # ========================================================
    # OCR
    # ========================================================

    with st.spinner(
        "Running multi-pass OCR..."
    ):

        ocr_results = run_ocr(
            document_img
        )

        ocr_text = ocr_results_to_text(
            ocr_results
        )


    # ========================================================
    # AADHAAR EXTRACTION
    # ========================================================

    with st.spinner(
        "Validating identity number..."
    ):
        aadhaar_number, aadhaar_confidence = extract_aadhaar_number(
            ocr_results,
            document_img
        )
        pan_number, pan_confidence = extract_pan_number(ocr_results)

    document_type = detect_document_type(
        ocr_text,
        aadhaar_number=aadhaar_number,
        pan_number=pan_number
    )

    aadhaar_valid = aadhaar_number is not None and validate_verhoeff(aadhaar_number)
    pan_valid, pan_validation_message = validate_pan(pan_number)

    # ========================================================
    # DOCUMENT ANALYSIS
    # ========================================================

    st.header(
        "📄 Document Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if document_type in ["AADHAAR", "PAN", "PASSPORT", "VISA"]:
            st.success(f"Document: {document_type}")
        else:
            st.warning("Document: UNKNOWN")

    with col2:
        if document_type == "AADHAAR":
            st.success("Aadhaar Checksum: PASS" if aadhaar_valid else "Aadhaar Checksum: NOT CONFIRMED")
        elif document_type == "PAN":
            st.success("PAN Format: PASS" if pan_valid else "PAN Format: NOT CONFIRMED")
        else:
            st.info("Document Validation: N/A")

    with col3:

        st.metric(
            "Resolution",
            f"{document_img.shape[1]} × "
            f"{document_img.shape[0]}"
        )


    # ========================================================
    # AADHAAR DETAILS
    # ========================================================

    if document_type == "AADHAAR":

        st.subheader(
            "🪪 Aadhaar Verification"
        )

        if aadhaar_number:

            st.success(
                f"Aadhaar detected: "
                f"{mask_aadhaar(aadhaar_number)}"
            )

            st.caption(
                f"OCR grouping confidence: "
                f"{aadhaar_confidence}%"
            )

            st.caption(
                "Verhoeff checksum: VALID"
            )

        else:

            st.warning(
                "No Aadhaar number was confidently "
                "identified as a 4-4-4 grouped value."
            )


    # ========================================================
    # PAN DETAILS
    # ========================================================

    if document_type == "PAN":
        st.subheader("💳 PAN Verification")
        if pan_number:
            masked_pan = pan_number[:2] + "•••••••" + pan_number[-1]
            st.success(f"PAN detected: {masked_pan}")
            st.caption(f"OCR confidence: {pan_confidence}%")
            st.caption(f"PAN format validation: {pan_validation_message}")
        else:
            st.warning("PAN number could not be confidently extracted from OCR.")


    # ========================================================
    # OCR RESULTS
    # ========================================================

    with st.expander(
        "🔎 View OCR Results"
    ):

        if ocr_results:

            for item in ocr_results:

                try:

                    text = item[1]
                    confidence = float(
                        item[2]
                    )

                    st.write(
                        f"• {text} "
                        f"— {confidence * 100:.1f}%"
                    )

                except Exception:

                    pass

        else:

            st.warning(
                "OCR did not return readable text."
            )


    # ========================================================
    # METADATA
    # ========================================================

    with st.spinner(
        "Analyzing metadata..."
    ):

        metadata = analyze_document_metadata(
            document_file
        )

    metadata_status = metadata[
        "status"
    ]

    st.subheader(
        "🧬 Metadata Analysis"
    )

    if metadata_status == "CLEAR":

        st.success(
            f"Metadata: CLEAR — "
            f"{metadata['details']}"
        )

    elif metadata_status == "SUSPICIOUS":

        st.warning(
            f"Metadata: SUSPICIOUS — "
            f"{metadata['details']}"
        )

    elif metadata_status == "NOT AVAILABLE":

        st.info(
            "Metadata engine is not available."
        )

    else:

        st.warning(
            f"Metadata: {metadata_status}"
        )


    # ========================================================
    # ELA / LOCALIZED IMAGE INTEGRITY
    # ========================================================

    with st.spinner("Checking localized image integrity..."):
        ela = analyze_image_integrity(document_img)

    ela_score = ela.get("score", 0.0)
    ela_status = ela.get("status", "UNAVAILABLE")
    synthetic_evidence = analyze_synthetic_evidence(document_img, ela)

    st.subheader("🧪 Image Integrity & Forensic Evidence")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Global ELA", ela.get("global_ela", ela_score))
    c2.metric("Local Max", ela.get("local_max", 0.0))
    c3.metric("Hotspots", ela.get("hotspot_count", 0))
    c4.metric("Affected Area", f"{ela.get('affected_area_pct', 0.0):.2f}%")
    c5.metric("Forensic Score", f"{ela.get('tamper_evidence_score', 0.0):.1f}/100")
    st.write(f"**Image-integrity status:** {ela_status}")
    st.write(f"**Synthetic/reconstruction indicator:** {synthetic_evidence.get('status', 'UNAVAILABLE')} — {synthetic_evidence.get('reason', '')}")
    st.caption("These are forensic screening indicators. They do not independently establish that a document is fake or AI-generated.")


    # PORTRAIT ROI PREVIEW
    # ========================================================

    st.header(
        "👤 Biometric Verification"
    )

    portrait_roi = get_document_portrait_roi(
        document_img, document_type
    )

    with st.expander(
        f"🔍 View {document_type} Portrait Search Region"
    ):

        st.image(
            cv2.cvtColor(
                portrait_roi,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"{document_type} portrait search region",
            width=400
        )

        st.caption(
            "The system searches the expected portrait area "
            "instead of treating text elsewhere on the card "
            "as a possible face."
        )


    # ========================================================
    # BIOMETRIC
    # ========================================================

    with st.spinner(
        "Extracting document and passenger faces..."
    ):

        biometric = run_biometric(
            document_img,
            passenger_img,
            document_type=document_type
        )


    biometric_status = biometric[
        "status"
    ]

    biometric_score = biometric[
        "score"
    ]


    # ========================================================
    # FACE PREVIEW
    # ========================================================

    face_col1, face_col2 = st.columns(2)

    with face_col1:

        st.subheader(
            "Document Portrait"
        )

        if biometric[
            "document_face"
        ] is not None:

            st.image(
                cv2.cvtColor(
                    biometric[
                        "document_face"
                    ],
                    cv2.COLOR_BGR2RGB
                ),
                width=250
            )

        else:

            st.error(
                "Actual document portrait could not be detected."
            )


    with face_col2:

        st.subheader(
            "Passenger Face"
        )

        if biometric[
            "passenger_face"
        ] is not None:

            st.image(
                cv2.cvtColor(
                    biometric[
                        "passenger_face"
                    ],
                    cv2.COLOR_BGR2RGB
                ),
                width=250
            )

        else:

            st.error(
                "Passenger face could not be detected."
            )


    # ========================================================
    # FACE QUALITY
    # ========================================================

    if (
        "document_quality" in biometric
        and "passenger_quality" in biometric
    ):

        q1, q2 = st.columns(2)

        with q1:

            st.metric(
                "Document Face Quality",
                f"{biometric['document_quality']:.1f}"
            )

        with q2:

            st.metric(
                "Passenger Face Quality",
                f"{biometric['passenger_quality']:.1f}"
            )


    # ========================================================
    # BIOMETRIC RESULT
    # ========================================================

    st.subheader(
        "🧬 Face Comparison"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Similarity Score",
            f"{biometric_score:.1f}%"
        )

    with c2:

        if biometric_status == "STRONG MATCH":

            st.success(
                "STRONG MATCH"
            )

        elif biometric_status == "POSSIBLE MATCH":

            st.warning(
                "POSSIBLE MATCH"
            )

        elif biometric_status == "LOW CONFIDENCE":

            st.warning(
                "LOW CONFIDENCE"
            )

        elif biometric_status == "NO SUFFICIENT MATCH":

            st.error(
                "NO SUFFICIENT MATCH"
            )

        else:

            st.error(
                biometric_status
            )

    with c3:

        distance = biometric.get(
            "distance"
        )

        if distance is not None:

            st.metric(
                "Distance",
                distance
            )

        else:

            st.metric(
                "Distance",
                "N/A"
            )

    with c4:

        st.metric(
            "Model",
            biometric.get(
                "model",
                "N/A"
            )
        )


    if biometric.get(
        "error"
    ):

        with st.expander(
            "Biometric Error"
        ):

            st.code(
                biometric[
                    "error"
                ]
            )


    st.caption(
        "Similarity score is an AI screening indicator, "
        "not a calibrated probability of identity."
    )


    # ========================================================
    # RISK
    # ========================================================

    face_found = (
        biometric_status
        not in [
            "DOCUMENT_FACE_NOT_FOUND",
            "PASSENGER_FACE_NOT_FOUND"
        ]
    )

    risk_result = calculate_risk(
        document_type=document_type,

        aadhaar_valid=aadhaar_valid,

        pan_valid=pan_valid,

        biometric_status=biometric_status,

        biometric_score=biometric_score,

        ela_status=ela_status,

        metadata_status=metadata_status,

        face_found=face_found,

        tamper_evidence_score=ela.get("tamper_evidence_score", 0.0),

        synthetic_evidence_status=synthetic_evidence.get("status", "NOT ASSESSED")
    )


    risk = risk_result[
        "risk"
    ]

    risk_level = risk_result[
        "level"
    ]

    directive = risk_result[
        "directive"
    ]

    reasons = risk_result[
        "reasons"
    ]


    # ========================================================
    # RISK DASHBOARD
    # ========================================================

    st.header(
        "🚦 Screening Risk Assessment"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Overall Screening Risk",
            f"{risk}%"
        )

        st.progress(
            risk / 100
        )

    with c2:

        if risk_level == "LOW":

            st.success(
                f"Risk Level: {risk_level}"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                f"Risk Level: {risk_level}"
            )

        elif risk_level == "HIGH":

            st.warning(
                f"Risk Level: {risk_level}"
            )

        else:

            st.error(
                f"Risk Level: {risk_level}"
            )


    # ========================================================
    # DIRECTIVE
    # ========================================================

    st.subheader(
        "Recommended Action"
    )

    if risk < 20:

        st.success(
            f"🟢 {directive}"
        )

    elif risk < 70:

        st.warning(
            f"🟠 {directive}"
        )

    else:

        st.error(
            f"🔴 {directive}"
        )


    # ========================================================
    # RISK FACTORS
    # ========================================================

    if reasons:

        st.subheader(
            "⚠️ Risk Factors"
        )

        for reason in reasons:

            st.write(
                f"• {reason}"
            )

    else:

        st.success(
            "No major automated risk indicators detected."
        )


    # ========================================================
    # AUDIT LOG
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    audit_record = {

        "Timestamp":
            timestamp,

        "Document":
            document_file.name,

        "Document Type":
            document_type,

        "Aadhaar":
            mask_aadhaar(
                aadhaar_number
            ),

        "Aadhaar Check": (
            "PASS" if aadhaar_valid else "NOT CONFIRMED"
        ) if document_type == "AADHAAR" else "N/A",

        "PAN Check": (
            "PASS" if pan_valid else "NOT CONFIRMED"
        ) if document_type == "PAN" else "N/A",

        "Biometric":
            biometric_status,

        "Face Score":
            biometric_score,

        "Model":
            biometric.get(
                "model",
                "N/A"
            ),

        "ELA":
            ela_score,

        "ELA Status":
            ela_status,

        "Global ELA":
            ela.get("global_ela", ela_score),

        "Local ELA Max":
            ela.get("local_max", 0.0),

        "ELA Hotspots":
            ela.get("hotspot_count", 0),

        "Affected Area %":
            ela.get("affected_area_pct", 0.0),

        "Forensic Score":
            ela.get("tamper_evidence_score", 0.0),

        "Synthetic Indicator":
            synthetic_evidence.get("status", "NOT ASSESSED"),

        "Metadata":
            metadata_status,

        "Risk":
            risk,

        "Risk Level":
            risk_level,

        "Directive":
            directive,

        "Document Hash":
            image_hash(
                document_img
            ),
        "Duplicate Screening": "NO"
    }

    # Duplicate screening against this session's prior records.
    current_hash = audit_record["Document Hash"]
    previous_hashes = {r.get("Document Hash") for r in st.session_state.audit_log}
    if current_hash in previous_hashes:
        audit_record["Duplicate Screening"] = "MATCHED PREVIOUS IMAGE"

    previous_chain, current_chain = compute_audit_chain(audit_record)
    audit_record["Previous Audit Hash"] = previous_chain
    audit_record["Audit Chain Hash"] = current_chain

    st.session_state.audit_log.append(
        audit_record
    )

    st.session_state.last_screening = {
        "risk": risk,
        "level": risk_level,
        "directive": directive,
        "reasons": reasons,
        "document_type": document_type,
        "aadhaar_valid": aadhaar_valid,
        "pan_valid": pan_valid,
        "pan_number": pan_number,
        "biometric_status": biometric_status,
        "biometric_score": biometric_score,
        "model": biometric.get("model", "N/A"),
        "ela_status": ela_status,
        "ela_score": ela_score,
        "ela_global": ela.get("global_ela", ela_score),
        "ela_local_max": ela.get("local_max", 0.0),
        "ela_hotspots": ela.get("hotspot_count", 0),
        "ela_affected_area_pct": ela.get("affected_area_pct", 0.0),
        "tamper_evidence_score": ela.get("tamper_evidence_score", 0.0),
        "synthetic_evidence_status": synthetic_evidence.get("status", "NOT ASSESSED"),
        "synthetic_evidence_score": synthetic_evidence.get("score", 0.0),
        "metadata_status": metadata_status,
        "face_found": face_found,
        "document_img": document_img.copy(),
        "document_hash": current_hash
    }


    # ========================================================
    # AUDIT TABLE
    # ========================================================

    st.header(
        "📋 Audit Trail"
    )

    audit_df = pd.DataFrame(
        st.session_state.audit_log
    )

    if not audit_df.empty:

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = audit_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Export Audit Log CSV",
            data=csv_data,
            file_name="ssb_screening_audit.csv",
            mime="text/csv"
        )


    # ========================================================
    # EXPLAINABLE RISK
    # ========================================================

    st.subheader("🧩 Explainable Risk Breakdown")
    factor_df = pd.DataFrame(build_explainable_factors(
        document_type, aadhaar_valid, pan_valid, biometric_status, biometric_score,
        ela_status, ela_score, metadata_status, face_found,
        integrity=ela,
        synthetic_evidence_status=synthetic_evidence.get("status", "NOT ASSESSED")
    ))
    st.dataframe(factor_df, use_container_width=True, hide_index=True)

    # ========================================================
    # FINAL NOTICE
    # ========================================================

    st.divider()

    st.info(
        "⚠️ This is an AI-assisted screening prototype. "
        "OCR, biometric similarity, metadata and image-integrity "
        "checks may produce false positives or false negatives. "
        "Any adverse action must be based on authorized human "
        "verification and applicable operational procedures."
    )

# ============================================================
# NIKO — PERSISTENT SCREENING ASSISTANT
# ============================================================

if st.session_state.last_screening is not None:
    render_niko(st.session_state.last_screening)

# ============================================================
# ADVANCED MODULES OUTSIDE THE SCREENING TRANSACTION
# ============================================================

with st.expander("🔬 Open TamperLab", expanded=False):
    render_tamper_lab()

with st.expander("📊 Open Screening Analytics", expanded=False):
    render_analytics()