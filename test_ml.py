"""
End-to-End Test Suite for the ATS Machine Learning Engine.
Validates Domain Classification, Semantic Similarity, Skill Gap Analysis,
and Recommendations Engine.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from ml_engine.classifier import predict_domain, calculate_domain_alignment
from ml_engine.skill_extractor import extract_categorized_skills, perform_skill_gap_analysis
from ml_engine.ats_scorer import analyze_resume_ats
from ml_engine.recommender import generate_recommendations

def test_pipeline():
    print("=" * 60)
    print("[TEST] Running ATS Machine Learning Engine Verification")
    print("=" * 60)
    
    sample_resume = """
    John Doe - Full Stack Developer
    Email: john.doe@example.com | Phone: (555) 123-4567 | github.com/johndoe
    
    Professional Summary:
    Results-driven Full Stack Engineer with 4+ years of experience building scalable web applications.
    Specialized in React.js, Node.js, TypeScript, and cloud-native microservices.
    
    Technical Skills:
    Languages: JavaScript, TypeScript, Python, SQL, HTML5, CSS3
    Frameworks: React, Next.js, Node.js, Express.js, Tailwind CSS
    Databases & Cloud: PostgreSQL, MongoDB, Redis, AWS, Docker
    Tools & Practices: Git, GitHub Actions, Jest, REST APIs, Agile
    
    Experience:
    Senior Software Engineer | CloudSoft Solutions (2022 - Present)
    • Architected responsive frontend with React, Next.js, and TypeScript, boosting conversion by 28%.
    • Engineered RESTful APIs and microservices in Node.js and PostgreSQL processing 2M+ requests daily.
    • Automated CI/CD deployment pipelines using GitHub Actions and Docker, reducing release time by 45%.
    • Spearheaded database indexing optimization in PostgreSQL and Redis caching, cutting latency by 60%.
    
    Education:
    Bachelor of Science in Computer Science - University of California
    """
    
    sample_jd = """
    Position: Full Stack Engineer (React / Node / AWS)
    
    About the Role:
    We are seeking a Full Stack Developer proficient in React.js, TypeScript, Node.js, and AWS.
    Experience with Docker, Kubernetes, GraphQL, and CI/CD pipelines is highly desirable.
    
    Responsibilities:
    • Build performant web interfaces using React and modern JavaScript.
    • Design and implement scalable backend APIs using Node.js and PostgreSQL.
    • Collaborate with cross-functional teams in an Agile Scrum environment.
    • Deploy and maintain containerized applications on AWS and Kubernetes.
    
    Requirements:
    • 3+ years experience with React, TypeScript, and Node.js.
    • Strong knowledge of PostgreSQL, Redis, and RESTful APIs / GraphQL.
    • Hands-on experience with Docker, AWS, and Kubernetes.
    • Excellent communication and problem-solving skills.
    """
    
    # 1. Test Domain Classifier
    print("\n1. Testing Domain Classification...")
    res_pred = predict_domain(sample_resume)
    jd_pred = predict_domain(sample_jd)
    print(f"   Resume Predicted Domain: {res_pred['domain']} ({res_pred['confidence']}% confidence)")
    print(f"   JD Predicted Domain:     {jd_pred['domain']} ({jd_pred['confidence']}% confidence)")
    assert res_pred['domain'] is not None
    assert jd_pred['domain'] is not None
    
    # 2. Test Alignment & Gap Analysis
    print("\n2. Testing Categorized Skill Gap Analysis...")
    gap_result = perform_skill_gap_analysis(sample_resume, sample_jd)
    print(f"   Matched Skills ({len(gap_result['all_matched'])}): {', '.join(gap_result['all_matched'][:8])}")
    print(f"   Missing Skills ({len(gap_result['all_missing'])}): {', '.join(gap_result['all_missing'])}")
    print(f"   Skill Coverage Percentage: {gap_result['coverage_percentage']}%")
    assert len(gap_result['all_matched']) > 0
    
    # 3. Test Full ATS Scoring Engine
    print("\n3. Testing Full Multi-Factor ATS Scoring...")
    analysis = analyze_resume_ats(sample_resume, sample_jd)
    print(f"   Overall Composite ATS Score: {analysis['overall_score']}% ({analysis['fit_level']})")
    print(f"   Sub-scores Breakdown:")
    for metric, score in analysis['sub_scores'].items():
        print(f"     - {metric.capitalize()}: {score}%")
    assert 0 <= analysis['overall_score'] <= 100
    
    # 4. Test Recommendations
    print("\n4. Testing AI Recommendations Engine...")
    recs = generate_recommendations(analysis)
    print(f"   Generated {len(recs)} Recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"     [{rec['priority'].upper()}] {rec['title']}")
    assert len(recs) >= 3
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All Machine Learning Pipeline Tests Passed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
