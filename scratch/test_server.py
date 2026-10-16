"""
Test client to verify Flask application and ML endpoints.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from app import app

def test_routes():
    client = app.test_client()
    
    # 1. GET /
    print("[TEST] 1. Testing GET / route...")
    res = client.get("/")
    assert res.status_code == 200
    assert b"ATS Resume Matcher &amp; Classifier" in res.data or b"ATS Resume Matcher & Classifier" in res.data
    print("   -> GET / passed with status 200 OK.")
    
    # 2. GET /api/model-info
    print("\n[TEST] 2. Testing GET /api/model-info...")
    res = client.get("/api/model-info")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["status"] == "success"
    print("   -> Active Model:", json_data["model_metadata"]["model_name"])
    print("   -> Supported Domains:", len(json_data["model_metadata"]["supported_domains"]))
    print("   -> Vocabulary Size:", json_data["model_metadata"]["vocabulary_size"])
    
    # 3. POST / (form text mode)
    print("\n[TEST] 3. Testing POST / with Direct Text Input...")
    resume_text = """
    Alice Smith - Machine Learning Engineer
    alice@example.com | (123) 456-7890 | github.com/alicesmith
    Experience:
    • Built and trained Deep Learning models in PyTorch and TensorFlow for Computer Vision.
    • Fine-tuned HuggingFace Transformer LLMs with LangChain and vector databases.
    • Deployed MLOps pipelines using Docker and Kubernetes, reducing inference latency by 40%.
    Skills: Python, PyTorch, TensorFlow, Scikit-Learn, Docker, Kubernetes, AWS, SQL
    Education: B.S. in Computer Science
    """
    
    jd_text = """
    We are looking for a Machine Learning Engineer with strong experience in PyTorch,
    Computer Vision, Deep Learning, and MLOps deployment on AWS and Docker.
    Knowledge of PySpark, Spark, and Kubernetes is a plus.
    """
    
    res = client.post("/", data={
        "input_mode": "text",
        "resume_text": resume_text,
        "jd_text": jd_text
    })
    assert res.status_code == 200
    assert b"Machine Learning &amp; AI" in res.data or b"Machine Learning & AI" in res.data
    print("   -> Form POST / passed with status 200 OK and rendered ML prediction.")
    
    # 4. POST /api/analyze
    print("\n[TEST] 4. Testing POST /api/analyze JSON REST API...")
    res = client.post("/api/analyze", json={
        "resume_text": resume_text,
        "jd_text": jd_text
    })
    assert res.status_code == 200
    api_data = res.get_json()
    assert api_data["status"] == "success"
    data = api_data["data"]
    print(f"   -> Overall ATS Score: {data['overall_score']}% ({data['fit_level']})")
    print(f"   -> Resume Domain: {data['domain_classification']['resume_domain']} ({data['domain_classification']['resume_confidence']}%)")
    print(f"   -> JD Domain:     {data['domain_classification']['jd_domain']} ({data['domain_classification']['jd_confidence']}%)")
    print(f"   -> Matched Skills ({len(data['skills_gap_analysis']['all_matched'])}):", data['skills_gap_analysis']['all_matched'])
    print(f"   -> Missing Skills ({len(data['skills_gap_analysis']['all_missing'])}):", data['skills_gap_analysis']['all_missing'])
    print(f"   -> Generated {len(data['recommendations'])} Recommendations.")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All Web Application & REST API Endpoints Working Perfectly!")
    print("=" * 60)

if __name__ == "__main__":
    test_routes()
