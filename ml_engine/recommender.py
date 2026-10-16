"""
AI Recommendation Engine for Resume Optimization.
Generates targeted, high-impact suggestions based on ML predictions, skill gaps,
and ATS structural health.
"""

def generate_recommendations(analysis_result: dict) -> list:
    """
    Generates a list of actionable optimization recommendations based on the analysis.
    Returns: list of dicts with {"category": str, "title": str, "advice": str, "priority": "high"|"medium"|"low"}
    """
    recs = []
    
    overall_score = analysis_result.get("overall_score", 0)
    sub_scores = analysis_result.get("sub_scores", {})
    skills_analysis = analysis_result.get("skills_analysis", {})
    domain_analysis = analysis_result.get("domain_analysis", {})
    ats_health = analysis_result.get("ats_health", {})
    
    missing_skills = skills_analysis.get("all_missing", [])
    matched_skills = skills_analysis.get("all_matched", [])
    
    # 1. Critical Skill Gap Recommendations (Highest Priority)
    if missing_skills:
        top_missing = missing_skills[:5]
        recs.append({
            "category": "Skill Coverage",
            "title": f"Incorporate Key Missing Technologies ({len(top_missing)})",
            "advice": f"The job description specifically targets {', '.join(top_missing)}. If you have experience with these, explicitly incorporate them into your skills and project descriptions.",
            "priority": "high"
        })
        
    # 2. Domain Alignment Recommendations
    if not domain_analysis.get("is_direct_match", False):
        res_dom = domain_analysis.get("resume_domain", "General")
        jd_dom = domain_analysis.get("jd_domain", "Target Role")
        recs.append({
            "category": "Domain Alignment",
            "title": f"Align Resume Tone Towards '{jd_dom}'",
            "advice": f"Your resume primarily reflects '{res_dom}' patterns, while the job description emphasizes '{jd_dom}'. Tailor your summary and experience bullet points to emphasize relevant '{jd_dom}' tools and deliverables.",
            "priority": "high" if sub_scores.get("domain", 0) < 60 else "medium"
        })
        
    # 3. Quantified Achievements
    quant_count = ats_health.get("quantified_achievements_count", 0)
    if quant_count < 3:
        recs.append({
            "category": "Impact Metrics",
            "title": "Quantify Achievements with Measurable Impact",
            "advice": "Recruiters and ATS algorithms favor quantified outcomes (e.g., 'reduced latency by 35%', 'processed 5M+ daily requests', 'boosted conversion by 18%'). Replace passive duty descriptions with measurable results.",
            "priority": "high"
        })
        
    # 4. Action Verbs
    verbs_count = ats_health.get("action_verbs_count", 0)
    if verbs_count < 5:
        recs.append({
            "category": "Language & Power Words",
            "title": "Strengthen Bullet Points with Power Action Verbs",
            "advice": "Begin each experience bullet point with decisive action verbs such as 'Architected', 'Spearheaded', 'Optimized', 'Automated', or 'Streamlined' instead of 'Responsible for' or 'Helped with'.",
            "priority": "medium"
        })
        
    # 5. ATS Formatting & Section Detection
    detected_sections = ats_health.get("detected_sections", [])
    important_sections = ["Summary / Objective", "Experience / History", "Skills / Competencies", "Education", "Projects"]
    missing_sections = [s for s in important_sections if s not in detected_sections]
    
    if missing_sections:
        recs.append({
            "category": "ATS Structure",
            "title": f"Add Standard Section Headers ({', '.join(missing_sections[:2])})",
            "advice": f"ATS parsers rely on standard headings. Ensure clear, distinct section titles such as '{missing_sections[0]}' to guarantee proper parsing.",
            "priority": "high"
        })
        
    # 6. Contact & Online Presence
    if not ats_health.get("has_linkedin_or_github", False):
        recs.append({
            "category": "Professional Presence",
            "title": "Add LinkedIn & GitHub / Portfolio Profile Links",
            "advice": "Including your LinkedIn profile or GitHub repository URL provides recruiters with direct verification of your projects and professional network.",
            "priority": "low"
        })
        
    # 7. Semantic Density
    if sub_scores.get("semantic", 0) < 50:
        recs.append({
            "category": "Contextual Relevance",
            "title": "Adopt Target Job Keywords & Terminology",
            "advice": "Your semantic similarity score is under 50%. Echo terminology used in the job description's responsibilities section to ensure higher contextual match.",
            "priority": "medium"
        })
        
    # Ensure at least 3 actionable recommendations
    if len(recs) < 3:
        recs.append({
            "category": "Polish",
            "title": "Proofread for ATS Single-Column Layout",
            "advice": "Ensure your resume uses a clean single-column format without tables, text boxes, or graphics that can confuse automated applicant tracking parsers.",
            "priority": "low"
        })
        
    return recs
