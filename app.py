"""
Flask Application for the ML-Powered ATS Resume Matcher & Parser.
Provides web dashboard interface, RESTful ML inference APIs, and PDF extraction.
"""

import os
from flask import Flask, request, render_template, jsonify
from utils.text_extractor import extract_text_from_pdf
from ml_engine.ats_scorer import analyze_resume_ats
from ml_engine.recommender import generate_recommendations
from ml_engine.classifier import get_classifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_mode = request.form.get("input_mode", "pdf")
        resume_text = ""
        jd_text = ""

        if input_mode == "pdf":
            resume_pdf = request.files.get("resume_pdf")
            jd_pdf = request.files.get("jd_pdf")

            if not resume_pdf or not jd_pdf or resume_pdf.filename == "" or jd_pdf.filename == "":
                return render_template("home.html", error="Please upload both a Resume PDF and a Job Description PDF.")

            resume_text = extract_text_from_pdf(resume_pdf)
            jd_text = extract_text_from_pdf(jd_pdf)

            if not resume_text:
                return render_template("home.html", error="Could not extract text from the Resume PDF. Please ensure it is not scanned or password-protected.")
            if not jd_text:
                return render_template("home.html", error="Could not extract text from the Job Description PDF. Please ensure it is not scanned or empty.")
        else:
            resume_text = request.form.get("resume_text", "").strip()
            jd_text = request.form.get("jd_text", "").strip()

            if not resume_text or not jd_text:
                return render_template("home.html", error="Please provide both Resume text and Job Description text.")

        # Run complete ML ATS Analysis
        try:
            analysis = analyze_resume_ats(resume_text, jd_text)
            recommendations = generate_recommendations(analysis)

            return render_template(
                "home.html",
                analysis=analysis,
                recommendations=recommendations,
                resume_snippet=resume_text[:200] + "...",
                jd_snippet=jd_text[:200] + "..."
            )
        except Exception as e:
            return render_template("home.html", error=f"An error occurred during ML analysis: {str(e)}")

    return render_template("home.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    RESTful ML inference endpoint.
    Accepts JSON payload: {"resume_text": "...", "jd_text": "..."}
    or multipart form-data with PDFs.
    """
    resume_text = ""
    jd_text = ""

    if request.is_json:
        data = request.get_json()
        resume_text = data.get("resume_text", "")
        jd_text = data.get("jd_text", "")
    elif "resume_pdf" in request.files and "jd_pdf" in request.files:
        resume_text = extract_text_from_pdf(request.files["resume_pdf"])
        jd_text = extract_text_from_pdf(request.files["jd_pdf"])
    else:
        resume_text = request.form.get("resume_text", "")
        jd_text = request.form.get("jd_text", "")

    if not resume_text or not jd_text:
        return jsonify({
            "status": "error",
            "message": "Both resume_text and jd_text are required."
        }), 400

    try:
        analysis = analyze_resume_ats(resume_text, jd_text)
        recs = generate_recommendations(analysis)
        return jsonify({
            "status": "success",
            "data": {
                "overall_score": analysis["overall_score"],
                "fit_level": analysis["fit_level"],
                "fit_description": analysis["fit_desc"],
                "sub_scores": analysis["sub_scores"],
                "domain_classification": analysis["domain_analysis"],
                "skills_gap_analysis": analysis["skills_analysis"],
                "ats_structural_health": analysis["ats_health"],
                "recommendations": recs
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Returns metadata about the active ML model."""
    try:
        artifact = get_classifier()
        pipeline = artifact['pipeline']
        vectorizer = pipeline.named_steps['tfidf']
        classifier = pipeline.named_steps['clf']

        return jsonify({
            "status": "success",
            "model_metadata": {
                "model_name": artifact.get("model_name"),
                "version": artifact.get("version"),
                "supported_domains": artifact.get("domains"),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "ngram_range": vectorizer.ngram_range,
                "classifier_type": type(classifier).__name__,
                "classes": list(classifier.classes_)
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Warm up ML model
    get_classifier()
    app.run(debug=True, use_reloader=True, host="127.0.0.1", port=5000)