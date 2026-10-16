"""
Multi-Factor ATS Scoring Engine.
Calculates multi-dimensional alignment combining Semantic NLP, Domain Classification,
Skill Coverage, and Resume Structural/ATS Health.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ml_engine.classifier import calculate_domain_alignment
from ml_engine.skill_extractor import perform_skill_gap_analysis

ACTION_VERBS = {
    "architected", "spearheaded", "engineered", "developed", "designed",
    "implemented", "optimized", "automated", "streamlined", "orchestrated",
    "deployed", "accelerated", "scaled", "delivered", "constructed",
    "formulated", "led", "managed", "collaborated", "mentored", "achieved"
}

def clean_text_for_nlp(text: str) -> str:
    """Preprocesses text for TF-IDF vectorization."""
    if not text:
        return ""
    # Lowercase & remove non-alphanumeric punctuation except common tech symbols
    cleaned = text.lower()
    cleaned = re.sub(r'[^a-z0-9+#.\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def compute_semantic_similarity(resume_text: str, jd_text: str) -> float:
    """
    Computes semantic cosine similarity using TF-IDF with unigrams + bigrams
    and sublinear term-frequency scaling.
    """
    clean_res = clean_text_for_nlp(resume_text)
    clean_jd = clean_text_for_nlp(jd_text)
    
    if not clean_res or not clean_jd:
        return 0.0
        
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english'
    )
    
    try:
        matrix = vectorizer.fit_transform([clean_res, clean_jd])
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
        # Normalize and scale to percentage
        score = round(float(sim) * 100, 1)
        return min(max(score, 0.0), 100.0)
    except Exception as e:
        print(f"Error in semantic similarity calculation: {e}")
        return 50.0

def evaluate_ats_structural_health(resume_text: str) -> dict:
    """
    Evaluates ATS formatting, structure, impact metrics, and action verbs.
    """
    if not resume_text:
        return {
            "score": 0.0,
            "has_email": False,
            "has_phone": False,
            "has_linkedin_or_github": False,
            "detected_sections": [],
            "quantified_achievements_count": 0,
            "action_verbs_count": 0,
            "word_count": 0
        }
        
    lower_text = resume_text.lower()
    words = re.findall(r'\b[a-z]+\b', lower_text)
    word_count = len(words)
    
    # 1. Contact Info Checks
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text))
    has_phone = bool(re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', resume_text))
    has_links = bool(re.search(r'(linkedin\.com|github\.com|portfolio)', lower_text))
    
    # 2. Section Header Detection
    sections_found = []
    section_patterns = {
        "Summary / Objective": r'\b(summary|profile|about me|objective)\b',
        "Experience / History": r'\b(experience|work history|employment|career)\b',
        "Skills / Competencies": r'\b(skills|technologies|technical proficiencies|competencies)\b',
        "Education": r'\b(education|academic background|degree|university|college)\b',
        "Projects": r'\b(projects|open source|portfolio projects)\b',
        "Certifications": r'\b(certifications|certificates|licenses)\b'
    }
    
    for section_name, pat in section_patterns.items():
        if re.search(pat, lower_text):
            sections_found.append(section_name)
            
    # 3. Quantified Achievements (numbers, percentages, metrics)
    metrics_matches = re.findall(r'(\d+%\s|\$\d+|\d+\+?\s*(years|k|m|users|clients|rps|fps|reduction|increase))', lower_text)
    quantified_count = len(metrics_matches)
    
    # 4. Action Verbs
    found_verbs = [w for w in words if w in ACTION_VERBS]
    action_verb_count = len(set(found_verbs))
    
    # Scoring computation
    score = 0.0
    if has_email: score += 10
    if has_phone: score += 10
    if has_links: score += 5
    
    # Section score (up to 30)
    score += min(len(sections_found) * 6, 30)
    
    # Metrics score (up to 20)
    score += min(quantified_count * 5, 20)
    
    # Action verbs score (up to 15)
    score += min(action_verb_count * 3, 15)
    
    # Word count appropriateness (300 to 1200 words) (up to 10)
    if 300 <= word_count <= 1200:
        score += 10
    elif 150 <= word_count < 300 or 1200 < word_count <= 1800:
        score += 5
        
    return {
        "score": round(min(score, 100.0), 1),
        "has_email": has_email,
        "has_phone": has_phone,
        "has_linkedin_or_github": has_links,
        "detected_sections": sections_found,
        "quantified_achievements_count": quantified_count,
        "action_verbs_count": action_verb_count,
        "action_verbs_found": sorted(list(set(found_verbs))),
        "word_count": word_count
    }

def analyze_resume_ats(resume_text: str, jd_text: str) -> dict:
    """
    Full ML ATS evaluation pipeline combining all dimensions into a unified report.
    """
    # 1. Semantic Cosine Similarity
    semantic_score = compute_semantic_similarity(resume_text, jd_text)
    
    # 2. Domain & Role Classifier Inference
    domain_eval = calculate_domain_alignment(resume_text, jd_text)
    domain_score = domain_eval["alignment_score"]
    
    # 3. Categorized Skill Extraction & Gap Analysis
    skills_gap = perform_skill_gap_analysis(resume_text, jd_text)
    skills_score = skills_gap["coverage_percentage"]
    
    # 4. ATS Structural Health
    ats_health = evaluate_ats_structural_health(resume_text)
    health_score = ats_health["score"]
    
    # 5. Composite Weighted ATS Score
    # 35% Semantic Relevance, 25% Skills Coverage, 20% Domain Alignment, 20% ATS Formatting Health
    composite = (
        0.35 * semantic_score +
        0.25 * skills_score +
        0.20 * domain_score +
        0.20 * health_score
    )
    overall_score = round(min(max(composite, 0.0), 100.0), 1)
    
    # Categorize Fit Level
    if overall_score >= 80:
        fit_level = "Exceptional Fit"
        fit_color = "#10b981"
        fit_desc = "Your resume strongly aligns with this job description across technical skills, domain relevance, and ATS structure."
    elif overall_score >= 65:
        fit_level = "Strong Match"
        fit_color = "#6366f1"
        fit_desc = "Good alignment with core requirements. Adding a few missing skills and quantifiable metrics will make this stand out."
    elif overall_score >= 45:
        fit_level = "Moderate Match"
        fit_color = "#f59e0b"
        fit_desc = "Partial match. Notable skill gaps and domain discrepancies detected. Customizing your resume will substantially increase callbacks."
    else:
        fit_level = "Needs Optimization"
        fit_color = "#f43f5e"
        fit_desc = "Low alignment with role requirements. Major technical skills or domain competencies appear missing."

    return {
        "overall_score": overall_score,
        "fit_level": fit_level,
        "fit_color": fit_color,
        "fit_desc": fit_desc,
        "sub_scores": {
            "semantic": semantic_score,
            "skills": skills_score,
            "domain": domain_score,
            "ats_health": health_score
        },
        "domain_analysis": domain_eval,
        "skills_analysis": skills_gap,
        "ats_health": ats_health
    }
