from flask import *
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def clean_function(text):
    if not text:
        return ""
    text = text.lower()
    text = nlp(text)
    tokens = [t for t in text]
    tokens = [t for t in tokens if not t.is_punct]
    tokens = [t for t in tokens if not t.is_stop]
    tokens = [t.lemma_ for t in tokens]
    tokens = [str(t) for t in tokens]
    return " ".join(tokens)

def extract_text_pdf(pdf):
    try:
        reader = PdfReader(pdf)
        text = "" 
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return ""

def extract_keywords(text):
    if not text or not text.strip():
        return []
    doc = nlp(text)
    keywords = set()
    
    # 1. Extract nouns & proper nouns as potential keywords
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 2 and token.is_alpha:
            keywords.add(token.text.lower())
            
    # 2. Extract noun chunks (to get phrase-level skills like "machine learning")
    for chunk in doc.noun_chunks:
        clean_chunk = " ".join([t.text.lower() for t in chunk if not t.is_stop and t.is_alpha])
        if clean_chunk and len(clean_chunk) > 2:
            words = clean_chunk.split()
            if 1 < len(words) <= 3:
                keywords.add(clean_chunk)
                
    return sorted(list(keywords))

def analyze_keywords(resume_text, jd_keywords):
    resume_lower = resume_text.lower()
    matched = []
    missing = []
    
    for kw in jd_keywords:
        kw_lower = kw.lower()
        idx = 0
        found = False
        
        while True:
            idx = resume_lower.find(kw_lower, idx)
            if idx == -1:
                break
                
            # Check if keyword is boundary-aligned
            # Check character before
            if idx > 0:
                char_before = resume_lower[idx - 1]
                if char_before.isalnum() or char_before in ['+', '#']:
                    idx += 1
                    continue
                    
            # Check character after
            end_idx = idx + len(kw_lower)
            if end_idx < len(resume_lower):
                char_after = resume_lower[end_idx]
                if char_after.isalnum() or char_after in ['+', '#']:
                    idx += 1
                    continue
                    
            found = True
            break
            
        if found:
            matched.append(kw)
        else:
            missing.append(kw)
            
    return matched, missing

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        resume_pdf = request.files.get("resume_pdf")
        jd_pdf = request.files.get("jd_pdf")

        if not resume_pdf or not jd_pdf or resume_pdf.filename == "" or jd_pdf.filename == "":
            return render_template("home.html", error="Please upload both a Resume and a Job Description PDF.")

        resume_text = extract_text_pdf(resume_pdf)
        jd_text = extract_text_pdf(jd_pdf)

        if not resume_text:
            return render_template("home.html", error="Could not extract text from the Resume PDF. Please ensure it is not scanned or empty.")
        if not jd_text:
            return render_template("home.html", error="Could not extract text from the Job Description PDF. Please ensure it is not scanned or empty.")

        clean_resume = clean_function(resume_text)
        clean_jd = clean_function(jd_text)

        if not clean_resume or not clean_jd:
             return render_template("home.html", error="Text cleaning resulted in empty text. Please use PDFs with more text content.")

        # Overall similarity score
        tv = TfidfVectorizer()
        vectors = tv.fit_transform([clean_resume, clean_jd])
        score = cosine_similarity(vectors[0], vectors[1])
        score = score[0][0]
        score = round(score, 2) * 100

        # Keyword matching gap analysis
        jd_keywords = extract_keywords(jd_text)
        matched_keywords, missing_keywords = analyze_keywords(resume_text, jd_keywords)

        # Capitalize keywords for clean presentation
        def format_keyword(kw):
            if len(kw) <= 3:
                return kw.upper()
            return kw.title()

        formatted_matched = [format_keyword(kw) for kw in matched_keywords]
        formatted_missing = [format_keyword(kw) for kw in missing_keywords]

        return render_template("home.html",
                               msg=f"Your Match Score is {score}%",
                               score=score,
                               matched_keywords=formatted_matched,
                               missing_keywords=formatted_missing)
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)