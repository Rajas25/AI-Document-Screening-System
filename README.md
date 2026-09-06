# 🔐 AI Document & Identity Screening System

> An explainable end-to-end AI prototype for **document verification, identity validation, facial biometric comparison, tamper analysis, and risk-based screening**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-5C3EE8?logo=opencv)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-orange)
![DeepFace](https://img.shields.io/badge/Face%20Recognition-DeepFace-purple)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

## 📌 Overview

The **AI Document & Identity Screening System** is a modular prototype designed to automate multiple stages of document and identity screening.

The system combines **OCR, document classification, biometric verification, image forensics, risk scoring, explainability, and audit logging** into a single Streamlit-based application.

It is designed as a prototype for use cases such as:

* 🛂 Border control
* 🧳 Immigration screening
* 🪪 Identity validation
* 📄 Document verification
* 🔍 Fraud and tamper screening

> **Note:** This project is a screening prototype. Its outputs are indicators for further review and should not be treated as definitive proof of identity or document authenticity.

---

# ✨ Key Features

| Module                      | Description                                                                                         |
| --------------------------- | --------------------------------------------------------------------------------------------------- |
| 📄 **Document Input**       | Upload or capture images of supported identity documents.                                           |
| 🔄 **Auto Orientation**     | Tests multiple rotations and selects the orientation with the strongest OCR result.                 |
| 🔎 **OCR & Classification** | Uses multi-pass EasyOCR with image enhancement to extract text and classify documents.              |
| ✅ **Document Validation**   | Validates Aadhaar numbers using the Verhoeff checksum and PAN numbers using format validation.      |
| 👤 **Face Extraction**      | Detects and extracts document portraits using document-specific regions and Haar Cascade detection. |
| 🧬 **Biometric Comparison** | Compares a document portrait with a supplied face image using DeepFace models.                      |
| 🔬 **Tamper Analysis**      | Performs multi-quality Error Level Analysis (ELA) and additional forensic checks.                   |
| ⚠️ **Risk Engine**          | Combines multiple screening signals into a risk score from 0–100.                                   |
| 📊 **Explainability**       | Shows individual risk factors, findings, and their contribution to the final result.                |
| 🔐 **Audit Trail**          | Maintains screening records using SHA-256 document hashes and cryptographic chaining.               |
| 🤖 **NIKO Assistant**       | Rule-based assistant that explains screening results and risk factors.                              |
| 🧪 **TamperLab**            | Allows images to be analyzed for potential manipulation and supports synthetic tamper testing.      |

---

# 🧠 System Pipeline

```text
                    ┌─────────────────────┐
                    │   Document Input    │
                    │ Image / Camera      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Orientation      │
                    │   Auto Correction   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    OCR Analysis     │
                    │     EasyOCR         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Document Detection  │
                    │ Aadhaar/PAN/Passport│
                    │       /Visa         │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Document        │         │ Face Extraction │
        │ Validation      │         │ & Detection     │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 │                           ▼
                 │                  ┌─────────────────┐
                 │                  │ Biometric       │
                 │                  │ Comparison      │
                 │                  └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Tamper Analysis    │
                    │ ELA + Forensics     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Risk Engine      │
                    │    Score 0–100      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Explainability &    │
                    │    Audit Trail      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   NIKO Assistant    │
                    └─────────────────────┘
```

---

# 🛠️ Technology Stack

| Component                | Technology               |
| ------------------------ | ------------------------ |
| **Frontend / UI**        | Streamlit                |
| **Programming Language** | Python 3.10+ / 3.11      |
| **Image Processing**     | OpenCV                   |
| **Numerical Computing**  | NumPy                    |
| **OCR**                  | EasyOCR                  |
| **Deep Learning**        | PyTorch                  |
| **Face Recognition**     | DeepFace                 |
| **Data Processing**      | Pandas                   |
| **Text Processing**      | Regex / TextBlob         |
| **Forensics**            | Custom Multi-Scale ELA   |
| **Security / Integrity** | SHA-256                  |
| **Deployment**           | Local Python environment |

All major dependencies are open-source.

---

# 📁 Project Structure

```text
AI-Document-Screening-System/
│
├── app.py                  # Main Streamlit application
├── tamper_engine.py        # Optional metadata/tamper analysis module
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignored files
│
└── ...
```

---

# ⚙️ Installation

## Prerequisites

Make sure you have:

* Python 3.10 or 3.11
* Git
* pip
* A working internet connection for initial dependency/model installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd AI-Document-Screening-System
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🚀 Usage

## 📄 Document Screening

1. Launch the application.
2. Upload an identity document or capture an image.
3. Upload or capture a comparison face image.
4. Click **RUN AI SCREENING**.
5. Review the generated screening results.

The system analyzes:

* Document type
* OCR results
* Extracted identification numbers
* Document validation
* Face detection
* Biometric similarity
* Tamper indicators
* Overall risk score
* Recommended action
* Explainable risk factors

---

# 🔬 TamperLab

**TamperLab** provides an interactive environment for image-forensics testing.

Users can:

* Upload an image for forensic analysis.
* Run multi-quality ELA.
* Inspect image integrity metrics.
* Detect potential local anomalies.
* Analyze pixel hotspots.
* Evaluate spatial discontinuities.
* Test synthetic image manipulations.
* Export forensic results as JSON.

### Synthetic Testing

The system can generate test variants such as:

```text
Original Image
      │
      ├──► Copy-Move Variant
      │
      ├──► Text Overlay Variant
      │
      └──► Reconstruction Variant
```

These variants can be used to evaluate how the screening pipeline responds to known image modifications.

---

# 🧬 Biometric Verification

The biometric module uses **DeepFace** to compare facial images.

The pipeline can attempt multiple face-recognition models, including:

* ArcFace
* Facenet512
* Additional fallback models

The result includes:

* Match status
* Similarity / distance information
* Model used
* Verification result

> Biometric results should be treated as screening signals rather than absolute identity proof.

---

# 🔎 Document Validation

The system performs document-specific validation.

### Aadhaar

* OCR extraction
* Number formatting
* Verhoeff checksum validation

### PAN

* OCR extraction
* PAN format validation using regular expressions

### Passport / Visa

* OCR-based document classification
* Keyword and structural indicators

---

# ⚠️ Risk Engine

The Risk Engine combines multiple signals into a normalized **0–100 risk score**.

Example screening signals include:

```text
Document Validation
        +
OCR Confidence
        +
Biometric Result
        +
Tamper Indicators
        +
Other Screening Signals
        │
        ▼
   Risk Aggregation
        │
        ▼
    Risk Score
     0 – 100
        │
        ▼
Recommended Action
```

Possible directives include:

| Directive                  | Meaning                                             |
| -------------------------- | --------------------------------------------------- |
| 🟢 `CLEAR`                 | No significant screening indicators detected.       |
| 🟡 `SECONDARY INSPECTION`  | Additional inspection recommended.                  |
| 🟠 `VERIFICATION REQUIRED` | Identity/document verification should be performed. |
| 🔴 `ESCALATE`              | Stronger review or escalation recommended.          |

---

# 📊 Explainability

The system provides an explainable breakdown of the screening decision.

Example:

| Signal          | Finding                | Impact   |
| --------------- | ---------------------- | -------- |
| Document        | PAN format valid       | Positive |
| OCR             | High confidence        | Positive |
| Biometrics      | Potential mismatch     | Negative |
| Tamper Analysis | Local anomaly detected | Negative |

This allows users to understand **why a particular risk score was generated** instead of receiving only a final classification.

---

# 🔐 Audit Trail

Screening results can be recorded in an audit trail.

The system uses:

* SHA-256 document hashes
* Screening records
* Cryptographic chaining
* CSV export

The hash-based approach helps maintain an integrity trail for recorded screening events.

---

# 🤖 NIKO Assistant

**NIKO** is a rule-based assistant integrated into the application.

It can explain screening results without requiring an external AI API.

### Example Questions

| Question                            | Response                                                   |
| ----------------------------------- | ---------------------------------------------------------- |
| **Why is the risk high?**           | Explains the signals contributing to the risk score.       |
| **Explain biometric result**        | Provides the biometric match status and model information. |
| **What do tamper indicators mean?** | Explains the forensic findings.                            |
| **What should I do next?**          | Explains the recommended action.                           |
| **How does the pipeline work?**     | Summarizes the screening workflow.                         |

---

# 🔧 Configuration

Several parts of the system can be customized.

### Risk Weights

Modify the risk calculation logic in:

```text
calculate_risk()
```

### Tamper Sensitivity

Thresholds can be adjusted within:

```text
analyze_image_integrity()
```

### Face Recognition Models

Models used by:

```text
deepface_compare()
```

can be modified according to the requirements.

### Document Regions

Document-specific portrait regions can be adjusted through functions such as:

```text
get_aadhaar_portrait_roi()
get_pan_portrait_roi()
```

---

# 🧪 Testing & Validation

The project includes mechanisms for testing different stages of the screening pipeline.

### Tamper Testing

TamperLab can be used to test:

* Original images
* Copy-move manipulation
* Text overlays
* Reconstruction effects
* Local image anomalies

### Pipeline Testing

The system can be evaluated using different combinations of:

* Valid documents
* Invalid document numbers
* Different image qualities
* Different face images
* Manipulated images

> Forensic and biometric outputs are heuristic indicators and may produce false positives or false negatives.

---

# 🔒 Privacy & Security Considerations

This project is designed as a prototype and processes potentially sensitive identity information.

When testing the application:

* Do not upload real identity documents unnecessarily.
* Use synthetic or test data whenever possible.
* Do not commit identity documents to GitHub.
* Do not commit personal photographs containing sensitive information.
* Keep secrets and credentials outside the repository.
* Use `.gitignore` to prevent accidental uploads.

Example:

```gitignore
.env
venv/
__pycache__/
*.pyc
uploads/
logs/
*.jpg
*.jpeg
*.png
```

---

# ⚠️ Disclaimer

This project is an **educational and experimental prototype**.

It is **not intended to make autonomous decisions about a person's identity, immigration status, eligibility, or access to services**.

Biometric matching, OCR, and image-forensic techniques can produce inaccurate results. Any real-world deployment would require appropriate human review, security controls, privacy protections, legal compliance, and extensive validation.

---

# 🔮 Future Enhancements

Potential future improvements include:

* 📑 Support for additional document formats
* 🌐 Web-based deployment
* 🗄️ Secure database-backed audit storage
* 📈 Advanced analytics dashboard
* 🧠 Improved document classification
* 🔍 More advanced image-forensics techniques
* 👤 Improved face detection and alignment
* 🔐 Stronger audit-log integrity mechanisms
* 🧪 Automated testing framework
* 📊 Performance and accuracy benchmarking
* 🔒 Enhanced privacy and data-protection controls

---

# 📜 License

This project is licensed under the **Apache License 2.0**.

See the `LICENSE` file for details.

---

# 👨‍💻 Author

**Rajas Sudumbrekar**

GitHub: **Rajas25**

---

⭐ If you find this project useful, consider giving the repository a star!
