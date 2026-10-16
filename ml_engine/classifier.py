"""
ML Resume & Job Description Domain Classifier Inference Module.
Loads the trained pipeline artifact and generates domain predictions with confidence distributions.
"""

import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "resume_classifier.joblib")

_model_artifact = None

def get_classifier():
    """Lazily loads the trained ML classifier artifact."""
    global _model_artifact
    if _model_artifact is None:
        if not os.path.exists(MODEL_PATH):
            # If not yet trained, train and export now
            from models.train_classifier import train_and_export_model
            _model_artifact = train_and_export_model(save_path=MODEL_PATH)
        else:
            _model_artifact = joblib.load(MODEL_PATH)
    return _model_artifact

def predict_domain(text: str) -> dict:
    """
    Predicts the technical domain of a given resume or job description text.
    Returns:
        {
            "domain": str,
            "confidence": float (0-100),
            "top_probabilities": list of {"domain": str, "prob": float}
        }
    """
    if not text or not text.strip():
        return {
            "domain": "General / Unspecified",
            "confidence": 0.0,
            "top_probabilities": []
        }
        
    artifact = get_classifier()
    pipeline = artifact['pipeline']
    
    # Predict probabilities
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    
    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    
    top_domain = classes[sorted_indices[0]]
    top_confidence = round(float(probs[sorted_indices[0]]) * 100, 1)
    
    top_probabilities = [
        {
            "domain": classes[i],
            "prob": round(float(probs[i]) * 100, 1)
        }
        for i in sorted_indices[:4]
    ]
    
    return {
        "domain": top_domain,
        "confidence": top_confidence,
        "top_probabilities": top_probabilities
    }

def calculate_domain_alignment(resume_text: str, jd_text: str) -> dict:
    """
    Evaluates whether the resume's predicted domain aligns with the job description.
    Computes cosine similarity between their probability vectors.
    """
    artifact = get_classifier()
    pipeline = artifact['pipeline']
    
    resume_probs = pipeline.predict_proba([resume_text])[0]
    jd_probs = pipeline.predict_proba([jd_text])[0]
    
    # Cosine similarity between probability distributions
    dot_product = np.dot(resume_probs, jd_probs)
    norm_res = np.linalg.norm(resume_probs)
    norm_jd = np.linalg.norm(jd_probs)
    
    if norm_res > 0 and norm_jd > 0:
        sim = dot_product / (norm_res * norm_jd)
        alignment_score = round(float(sim) * 100, 1)
    else:
        alignment_score = 50.0
        
    resume_pred = predict_domain(resume_text)
    jd_pred = predict_domain(jd_text)
    
    is_direct_match = (resume_pred["domain"] == jd_pred["domain"])
    
    return {
        "alignment_score": alignment_score,
        "resume_domain": resume_pred["domain"],
        "resume_confidence": resume_pred["confidence"],
        "jd_domain": jd_pred["domain"],
        "jd_confidence": jd_pred["confidence"],
        "is_direct_match": is_direct_match,
        "resume_top_prob": resume_pred["top_probabilities"],
        "jd_top_prob": jd_pred["top_probabilities"]
    }
