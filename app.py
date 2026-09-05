import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
import os
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

def detect_document_type(
    text,
    aadhaar_number
):

    upper = text.upper()

    aadhaar_score = 0
    passport_score = 0

    aadhaar_keywords = [
        "AADHAAR",
        "AADHAR",
        "UIDAI",
        "UNIQUE IDENTIFICATION",
        "GOVERNMENT OF INDIA"
    ]

    passport_keywords = [
        "PASSPORT",
        "REPUBLIC",
        "NATIONALITY",
        "P<"
    ]

    for keyword in aadhaar_keywords:

        if keyword in upper:
            aadhaar_score += 2

    for keyword in passport_keywords:

        if keyword in upper:
            passport_score += 2

    if aadhaar_number:
        aadhaar_score += 5

    if aadhaar_score >= passport_score and aadhaar_score > 0:

        return "AADHAAR"

    if passport_score > aadhaar_score:

        return "PASSPORT"

    return "UNKNOWN"


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

        if "AADHAAR" in text:
            score += 10

        if "AADHAR" in text:
            score += 8

        if "GOVERNMENT OF INDIA" in text:
            score += 8

        if "UNIQUE IDENTIFICATION" in text:
            score += 8

        if "DOB" in text:
            score += 2

        if "MALE" in text or "FEMALE" in text:
            score += 2

        # Check for 4-4-4
        if re.search(
            r"\d{4}\s+\d{4}\s+\d{4}",
            text
        ):
            score += 10

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
# AADHAAR PORTRAIT REGION
# ============================================================

def get_aadhaar_portrait_roi(
    image
):

    h, w = image.shape[:2]

    # Aadhaar front portrait region.
    #
    # For the orientation produced by the uploaded card:
    # photo is on the left side.
    #
    # We deliberately avoid scanning the entire card.

    x1 = int(
        w * 0.035
    )

    x2 = int(
        w * 0.38
    )

    y1 = int(
        h * 0.20
    )

    y2 = int(
        h * 0.78
    )

    roi = image[
        y1:y2,
        x1:x2
    ]

    return roi


# ============================================================
# FACE EXTRACTION
# ============================================================

def detect_face_in_roi(
    roi
):

    if roi is None:
        return None

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale small document portrait
    scale = 3

    enlarged = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enlarged = clahe.apply(
        enlarged
    )

    faces = face_detector.detectMultiScale(
        enlarged,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # Select largest detected face
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # Add margin around face
    mx = int(
        w * 0.25
    )

    my = int(
        h * 0.30
    )

    x1 = max(
        0,
        x - mx
    )

    y1 = max(
        0,
        y - my
    )

    x2 = min(
        roi.shape[1],
        x + w + mx
    )

    y2 = min(
        roi.shape[0],
        y + h + my
    )

    face = roi[
        y1:y2,
        x1:x2
    ]

    return face


def extract_aadhaar_face(
    image
):

    # --------------------------------------------------------
    # FIRST: strict Aadhaar portrait ROI
    # --------------------------------------------------------

    roi = get_aadhaar_portrait_roi(
        image
    )

    face = detect_face_in_roi(
        roi
    )

    if face is not None:

        return face, roi

    # --------------------------------------------------------
    # SECOND: slightly expanded ROI
    # --------------------------------------------------------

    h, w = image.shape[:2]

    expanded = image[
        int(h * 0.12):int(h * 0.85),
        0:int(w * 0.48)
    ]

    face = detect_face_in_roi(
        expanded
    )

    if face is not None:

        return face, expanded

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT search the entire Aadhaar.
    #
    # Searching the entire card previously caused Hindi
    # characters to be returned as the "face".
    # --------------------------------------------------------

    return None, roi


def extract_passenger_face(
    image
):

    face = detect_face_in_roi(
        image
    )

    return face


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
# BIOMETRIC PIPELINE
# ============================================================

def run_biometric(
    document_image,
    passenger_image
):

    document_face, portrait_roi = (
        extract_aadhaar_face(
            document_image
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
            [
                cv2.IMWRITE_JPEG_QUALITY,
                90
            ]
        )

        compressed = cv2.imread(
            path
        )

        os.remove(
            path
        )

        if compressed is None:
            return 0

        diff = cv2.absdiff(
            image,
            compressed
        )

        gray = cv2.cvtColor(
            diff,
            cv2.COLOR_BGR2GRAY
        )

        score = float(
            np.mean(gray)
        )

        return round(
            score,
            2
        )

    except Exception:

        return 0


def analyze_image_integrity(
    image
):

    ela = calculate_ela(
        image
    )

    if ela < 10:

        status = "CLEAR"

    elif ela < 25:

        status = "REVIEW"

    else:

        status = "POSSIBLE MANIPULATION"

    return {
        "score": ela,
        "status": status
    }


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
    biometric_status,
    biometric_score,
    ela_status,
    metadata_status,
    face_found
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
    # Aadhaar
    # --------------------------------------------------------

    if document_type == "AADHAAR":

        if not aadhaar_valid:

            risk += 20

            reasons.append(
                "Aadhaar number could not be confidently validated."
            )

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

    if ela_status == "REVIEW":

        risk += 5

        reasons.append(
            "Image compression anomaly requires review."
        )

    elif ela_status == "POSSIBLE MANIPULATION":

        risk += 20

        reasons.append(
            "Possible image manipulation detected."
        )

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
        "Upload Aadhaar / Passport / Visa",
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

        aadhaar_number, aadhaar_confidence = (
            extract_aadhaar_number(
                ocr_results,
                document_img
            )
        )


    # ========================================================
    # DOCUMENT TYPE
    # ========================================================

    document_type = detect_document_type(
        ocr_text,
        aadhaar_number
    )

    aadhaar_valid = (
        aadhaar_number is not None
        and validate_verhoeff(
            aadhaar_number
        )
    )


    # ========================================================
    # DOCUMENT ANALYSIS
    # ========================================================

    st.header(
        "📄 Document Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if document_type == "AADHAAR":

            st.success(
                "Document: AADHAAR"
            )

        elif document_type == "PASSPORT":

            st.success(
                "Document: PASSPORT"
            )

        else:

            st.warning(
                "Document: UNKNOWN"
            )


    with col2:

        if aadhaar_valid:

            st.success(
                "Aadhaar Checksum: PASS"
            )

        else:

            st.warning(
                "Aadhaar Checksum: NOT CONFIRMED"
            )


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
    # ELA
    # ========================================================

    with st.spinner(
        "Checking image integrity..."
    ):

        ela = analyze_image_integrity(
            document_img
        )

    ela_score = ela[
        "score"
    ]

    ela_status = ela[
        "status"
    ]

    st.subheader(
        "🧪 Image Integrity"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "ELA Indicator",
            ela_score
        )

    with col2:

        if ela_status == "CLEAR":

            st.success(
                "Image Integrity: CLEAR"
            )

        elif ela_status == "REVIEW":

            st.warning(
                "Image Integrity: REVIEW"
            )

        else:

            st.error(
                "Possible Manipulation"
            )

    st.caption(
        "ELA is a heuristic indicator and is not by itself "
        "proof of document manipulation."
    )


    # ========================================================
    # PORTRAIT ROI PREVIEW
    # ========================================================

    st.header(
        "👤 Biometric Verification"
    )

    portrait_roi = get_aadhaar_portrait_roi(
        document_img
    )

    with st.expander(
        "🔍 View Aadhaar Portrait Search Region"
    ):

        st.image(
            cv2.cvtColor(
                portrait_roi,
                cv2.COLOR_BGR2RGB
            ),
            caption="Aadhaar portrait search region",
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
            passenger_img
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

        biometric_status=biometric_status,

        biometric_score=biometric_score,

        ela_status=ela_status,

        metadata_status=metadata_status,

        face_found=face_found
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

        "Aadhaar Check":
            "PASS"
            if aadhaar_valid
            else "NOT CONFIRMED",

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
            )[:16]
    }

    st.session_state.audit_log.append(
        audit_record
    )


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