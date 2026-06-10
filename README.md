# ATS Resume Matcher & Parser

An AI-powered Applicant Tracking System (ATS) matching tool that extracts text from resumes and job descriptions, calculates semantic compatibility, and provides detailed skill gap analysis.

The application is built on a Python Flask backend utilizing advanced Natural Language Processing (NLP) via `spaCy` and `scikit-learn` and rendered on a modern slate-and-indigo dashboard UI.

---

## 🚀 Key Features

* **PDF Parsing**: Automatically extracts text from uploaded PDF documents with robust error handling for scanned or corrupted files.
* **Semantic Analysis**: Calculates overall contextual similarity using **TF-IDF Vectorization** and **Cosine Similarity** metrics.
* **Skill & Keyword Extraction**: Leverages spaCy's Part-of-Speech (POS) tagging and noun-chunking to parse technologies, frameworks, and job titles.
* **Keyword Gap Analysis**: Side-by-side visualization of **Matched Skills** vs. **Missing Skills** to give job seekers clear, actionable customization instructions.
* **Premium Dashboard UI**: Responsive slate/indigo dark-mode dashboard featuring an animated circular progress gauge, custom file drop zones, and a processing loading wheel overlay.

---

## 🛠️ Tech Stack

* **Backend**: Python 3, Flask
* **NLP & Similarity**: spaCy (using the lightweight `en_core_web_sm` model), Scikit-Learn (TF-IDF Vectorizer)
* **File Processing**: PyPDF2
* **Frontend**: HTML5, CSS3, JavaScript (Inter & Plus Jakarta Sans typography)
* **Production Server**: Gunicorn

---

## 📂 Project Directory Structure

```text
├── templates/
│   └── home.html       # Single-page frontend dashboard template
├── app.py              # Main Flask server & NLP parsing logic
├── requirements.txt    # Production dependencies
└── README.md           # Documentation
```

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd project1
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy NLP English model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Start the Flask development server:**
   ```bash
   python app.py
   ```

6. **Open your browser:**
   Navigate to `http://127.0.0.1:5000/`

---

## 🧪 Testing

The repository contains test validations to assert the extraction and matching logic. To run the automated NLP test suite:

```bash
python -m unittest scratch/test_nlp_features.py
```

---

## ☁️ Deployment on Render

This project is optimized out-of-the-box to run on free hosting tiers (such as Render) by utilizing the low-memory `en_core_web_sm` model.

1. Connect your repository to **Render**.
2. Select **Web Service** and choose **Python** runtime.
3. Configure the following build variables:
   * **Build Command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   * **Start Command**: `gunicorn app:app`
   * **Instance Type**: `Free`
