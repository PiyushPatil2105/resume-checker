# 🚀 ML-Powered ATS Resume Matcher & Classifier

An end-to-end **Machine Learning & NLP** Applicant Tracking System (ATS) matching tool that classifies candidate profiles across 10 industry tech domains, extracts categorized technical skills, computes multi-factor semantic alignment, and generates targeted resume optimization recommendations.

---

## 🌟 Key Machine Learning & NLP Features

1. **Supervised Resume Domain Classifier**:
   - Classifies candidate resumes and job postings into 10 industry engineering domains (Data Science, Machine Learning & AI, Full Stack, Backend, Frontend, DevOps & Cloud, Cybersecurity, Mobile App Dev, QA & Testing, Data Engineering).
   - Trained using **TF-IDF N-gram feature representation** (unigrams + bigrams with sublinear TF scaling) combined with a **Multinomial Logistic Regression** model with balanced class weighting.
   - Outputs calibrated probability distributions and confidence percentages for candidate profiling.

2. **Multi-Factor ML ATS Scoring Engine**:
   - Rather than simple keyword count or single-metric TF-IDF, the scoring engine uses an ensemble formula:
     - **Semantic Relevance (35%)**: TF-IDF contextual cosine similarity.
     - **Skill Coverage (25%)**: Ratio of matched vs. required hard & soft skills across technical taxonomies.
     - **Domain Alignment (20%)**: Cosine similarity between candidate and job posting probability vectors.
     - **ATS Formatting & Health (20%)**: Structural evaluation checking section completeness, action power verbs, and quantified achievements (metrics, percentages, numbers).

3. **Categorized Skill & Entity Extraction**:
   - Boundary-aware extraction across Programming Languages, Web Frameworks, Cloud & DevOps, Databases, AI/ML tools, QA Automation, and Leadership methodologies.
   - Comprehensive **Keyword Gap Analysis** detailing Matched Skills vs. Missing Skill Gaps.

4. **Automated AI Optimization Recommendations**:
   - Generates prioritized recommendations (High, Medium, Low) to improve ATS clearance, including missing high-yield keywords, power verb replacements, and structural adjustments.

5. **Dual Input & RESTful ML API**:
   - Supports both PDF file uploads (using PyMuPDF with PyPDF2 fallback) and direct text paste.
   - Exposes `/api/analyze` and `/api/model-info` JSON endpoints for programmatic integration.

---

## 🛠️ Architecture & Tech Stack

```text
├── models/
│   ├── dataset.py                # Synthetic & multi-domain training dataset generator
│   ├── train_classifier.py       # ML Pipeline training, cross-validation & artifact export
│   └── resume_classifier.joblib  # Trained model & vectorizer artifact
├── ml_engine/
│   ├── classifier.py             # Inference module for domain prediction & probability scoring
│   ├── skill_extractor.py        # Categorized skill taxonomy & boundary-aware extraction
│   ├── ats_scorer.py             # Multi-factor ATS scoring formula & formatting heuristics
│   └── recommender.py            # Automated AI recommendations generator
├── utils/
│   └── text_extractor.py         # Robust PDF extraction (PyMuPDF + PyPDF2 fallback)
├── templates/
│   └── home.html                 # Modern Slate/Indigo Dashboard with animated gauges & badges
├── app.py                        # Flask backend & RESTful inference API
├── requirements.txt              # Production dependencies
├── test_ml.py                    # Automated test suite for ML pipeline
└── scratch/                      # Test clients and temporary scripts
```

- **Core ML / NLP**: Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`), NumPy, Joblib
- **Document Extraction**: PyMuPDF (`fitz`), PyPDF2
- **Backend Web Server**: Python 3.10+, Flask, Gunicorn
- **Frontend UI**: Responsive HTML5, Vanilla CSS3 (Slate & Indigo theme, glassmorphism, SVG gauges), JavaScript

---

## 📊 Model Training & Evaluation

The repository includes a complete training pipeline with cross-validation and classification reports.

### Run Training:
```bash
python -m models.train_classifier
```

### Metrics:
- **Corpus**: 500 multi-domain technical profiles across 10 balanced domains.
- **5-Fold Cross Validation Accuracy**: ~99.5% - 100% on domain classification.
- **Feature Space**: Sublinear TF-IDF (1, 2-grams) with 2,300+ domain-specific vocabulary features.

---

## ⚡ Quickstart & Installation

### 1. Clone the repository:
```bash
git clone <your-repository-url>
cd resume-checker
```

### 2. Set up virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. (Optional) Run automated tests:
```bash
python test_ml.py
python scratch/test_server.py
```

### 5. Launch the application:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

---

## 🔌 REST API Usage

### Analyze Resume Endpoint: `POST /api/analyze`
**Request Payload (JSON or multipart/form-data):**
```json
{
  "resume_text": "Experienced Machine Learning Engineer skilled in PyTorch, Python, Docker, AWS...",
  "jd_text": "Seeking an ML Engineer with PyTorch, Computer Vision, Docker, and Kubernetes..."
}
```

**Sample Response:**
```json
{
  "status": "success",
  "data": {
    "overall_score": 78.5,
    "fit_level": "Strong Match",
    "sub_scores": {
      "semantic": 62.4,
      "skills": 85.0,
      "domain": 94.2,
      "ats_health": 80.0
    },
    "domain_classification": {
      "resume_domain": "Machine Learning & AI",
      "resume_confidence": 91.2,
      "jd_domain": "Machine Learning & AI",
      "jd_confidence": 88.7,
      "is_direct_match": true
    },
    "skills_gap_analysis": {
      "all_matched": ["PyTorch", "Python", "Docker", "AWS"],
      "all_missing": ["Kubernetes", "Computer Vision"]
    },
    "recommendations": [
      {
        "priority": "high",
        "title": "Incorporate Key Missing Technologies",
        "advice": "The job description specifically targets Kubernetes, Computer Vision..."
      }
    ]
  }
}
```

### Model Info Endpoint: `GET /api/model-info`
Returns active model parameters, vectorizer configuration, and supported classes.
