import streamlit as st
import cv2
import numpy as np
import re
from PIL import Image
from ultralytics import YOLO
import easyocr
from deepface import DeepFace
from verhoeff import validate_aadhaar

# Page Layout configuration for Hackathon Presentation
st.set_page_config(page_title="AI Border Screening System", layout="wide")
st.title("🛡️ AI-Based Fake Identity & Document Screening System (SSB)")

# Cache models so they download once on initialization
@st.cache_resource
def load_models():
    # Downloads standard pre-trained layout frame weights
    layout_model = YOLO("yolov8n.pt") 
    # Use PyTorch-backed EasyOCR instead of the buggy Paddle engine
    ocr_model = easyocr.Reader(['en'], gpu=False)
    return layout_model, ocr_model

with st.spinner("Initializing clean PyTorch-backed frameworks..."):
    layout_net, ocr_net = load_models()

# Layout Design Setup
col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Ingest Travel Document")
    uploaded_doc = st.file_uploader("Upload Aadhaar/PAN/Passport Image", type=["jpg", "png", "jpeg"])
    
with col2:
    st.subheader("Step 2: Ingest Live Camera Frame")
    uploaded_face = st.file_uploader("Upload Live Border Webcam Frame", type=["jpg", "png", "jpeg"])

if uploaded_doc and uploaded_face:
    # Process Uploads into CV2 matrices
    doc_bytes = np.frombuffer(uploaded_doc.read(), np.uint8)
    doc_img = cv2.imdecode(doc_bytes, cv2.IMREAD_COLOR)
    
    face_bytes = np.frombuffer(uploaded_face.read(), np.uint8)
    face_img = cv2.imdecode(face_bytes, cv2.IMREAD_COLOR)
    
    # Render Layout Check columns
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("🔍 Automated Verification Engine Execution")
        
                # MODULE 1: Smart Text extraction via EasyOCR with Auto-Rotation Correction
        with st.spinner("Extracting text via EasyOCR..."):
            raw_ocr = ocr_net.readtext(doc_img)
            
            detected_text_pool = [item[1] for item in raw_ocr] if raw_ocr else []
            full_text_dump = " ".join(detected_text_pool)
            cleaned_upper_text = full_text_dump.upper().replace(" ", "")
            
            # If no standard Indian card format is found, attempt auto-rotation checks
            if not re.search(r'\d{12}', cleaned_upper_text) and not re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', cleaned_upper_text):
                st.info("🔄 Non-horizontal layout layout suspected. Attempting automatic orientation correction...")
                
                # Test rotations: 90 degrees clockwise, 180, and 270 degrees
                for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    rotated_img = cv2.rotate(doc_img, angle)
                    rotated_ocr = ocr_net.readtext(rotated_img)
                    
                    test_pool = [item[1] for item in rotated_ocr] if rotated_ocr else []
                    test_text = " ".join(test_pool).upper().replace(" ", "")
                    
                    # Check if the rotated version successfully uncovers a valid target layout
                    if re.search(r'\d{12}', test_text) or re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', test_text):
                        doc_img = rotated_img  # Update image reference for later face cropping
                        raw_ocr = rotated_ocr
                        detected_text_pool = test_pool
                        full_text_dump = " ".join(detected_text_pool)
                        st.success("✔️ Orientation successfully corrected!")
                        break

        
        # MODULE 2: Custom Indian Validation Rules
        st.write("**📋 Mathematical Validation Status:**")
        
        # Clean string variations for regex lookups
        cleaned_upper_text = full_text_dump.upper().replace(" ", "")
        
        pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', cleaned_upper_text)
        aadhaar_match = re.search(r'\d{12}', cleaned_upper_text)
        
        is_valid_format = False
        
        if pan_match:
            pan_str = pan_match.group(0)
            st.success(f"✔️ Found PAN Card Structure: {pan_str}")
            # Rule validation: 4th char dictates individual/entity status code
            if pan_str[3] in ['P', 'C', 'H', 'F', 'A', 'T', 'B', 'L', 'J', 'G']:
                st.success("✔️ Rule Validation: PAN holder category structural code verified.")
                is_valid_format = True
            else:
                st.error("❌ Rule Anomaly: Fraudulent or non-standard PAN holder category identifier.")
                
        elif aadhaar_match:
            raw_aadhaar = aadhaar_match.group(0)
            st.info(f"Checking Aadhaar sequence match: {raw_aadhaar}")
            # Verhoeff math parsing execution
            if validate_aadhaar(raw_aadhaar):
                st.success("✔️ Rule Validation: Aadhaar 12-digit Verhoeff Checksum matches perfectly.")
                is_valid_format = True
            else:
                st.error("❌ Alteration Alert: String failed mathematical Verhoeff verification.")
        else:
            st.warning("⚠️ No clear Indian standardized ID layout formatting rules triggered via OCR.")
            
                # MODULE 4: Zero-Shot Face Verification with Auto-Alignment Rotation
        face_match = False
        confidence = 0.0
        with st.spinner("Aligning faces and running structural matching..."):
            try:
                # align=True forces DeepFace to detect facial landmarks (eyes, nose)
                # and automatically rotate both faces upright before extracting embeddings.
                verification = DeepFace.verify(
                    img1_path = doc_img, 
                    img2_path = face_img, 
                    model_name = "ArcFace", 
                    enforce_detection = False,
                    align = True  # <--- CRITICAL FIX: Automatically rotates faces to 0 degrees
                )
                face_match = verification["verified"]
                confidence = 1 - verification["distance"]
                
                if face_match:
                    st.success(f"✔️ Identity Authenticated: Face match confirmed via ArcFace (Confidence: {confidence:.2f})")
                else:
                    st.error(f"❌ Alert: Cross-verification anomaly. Facial confidence match low ({confidence:.2f})")
            except Exception as e:
                st.error(f"Facial analysis framework error: {str(e)}")


    with res_col2:
        st.subheader("🚦 Border Personnel Risk Assessment")
        
        # Calculate Unified Risk Scoring Logic dynamically
        risk_score = 100
        if is_valid_format:
            risk_score -= 40
        if face_match:
            risk_score -= 40
        if confidence > 0.6:
            risk_score -= 20
            
        if risk_score >= 60:
            st.error(f"🚨 DANGER STATUS (Risk Index Score: {risk_score}%)")
            st.metric(label="Action Required", value="DETAIN & INSPECT")
        elif risk_score >= 30:
            st.warning(f"⚠️ WARNING STATUS (Risk Index Score: {risk_score}%)")
            st.metric(label="Action Required", value="SECONDARY INTERVIEW")
        else:
            st.success(f"✅ CLEAR STATUS (Risk Index Score: {risk_score}%)")
            st.metric(label="Action Required", value="ALLOW BORDER ENTRY")
