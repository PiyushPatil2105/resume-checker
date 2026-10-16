"""
Categorized Skill Extraction and ATS Keyword Gap Analysis Module.
Extracts hard skills, tools, frameworks, and methodologies across categorized taxonomies.
"""

import re

SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", ".net", "golang", "go",
        "rust", "ruby", "php", "swift", "kotlin", "dart", "scala", "r", "sql", "bash", "shell", "c"
    ],
    "Web & Mobile Frameworks": [
        "react", "react.js", "next.js", "vue", "vue.js", "angular", "node.js", "express.js",
        "django", "flask", "fastapi", "spring boot", "asp.net", "rails", "laravel",
        "flutter", "react native", "swiftui", "tailwind css", "bootstrap", "redux", "graphql", "rest api"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "ci/cd", "github actions", "gitlab ci", "helm", "linux", "nginx",
        "prometheus", "grafana", "datadog", "argocd", "cloudformation"
    ],
    "Databases & Storage": [
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "sqlite", "dynamodb", "snowflake", "bigquery", "redshift", "firebase", "oracle"
    ],
    "AI, ML & Data Science": [
        "machine learning", "deep learning", "pytorch", "tensorflow", "keras", "scikit-learn",
        "huggingface", "transformers", "nlp", "computer vision", "opencv", "llm", "langchain",
        "pandas", "numpy", "scipy", "tableau", "power bi", "mlops", "spark", "pyspark", "airflow"
    ],
    "Testing & Methodologies": [
        "selenium", "cypress", "playwright", "jest", "pytest", "postman", "jmeter",
        "git", "github", "jira", "agile", "scrum", "microservices", "system design",
        "unit testing", "tdd", "ci/cd pipelines"
    ],
    "Soft Skills & Leadership": [
        "leadership", "communication", "team collaboration", "problem solving",
        "project management", "mentorship", "cross-functional", "stakeholder management",
        "critical thinking", "adaptability", "time management"
    ]
}

def extract_categorized_skills(text: str) -> dict:
    """
    Extracts matched skills grouped by category from text.
    Uses regex boundary matching for exact technical keywords.
    """
    if not text:
        return {cat: [] for cat in SKILL_TAXONOMY}
        
    text_lower = text.lower()
    extracted = {}
    
    for category, skill_list in SKILL_TAXONOMY.items():
        found = []
        for skill in skill_list:
            # Construct boundary safe pattern
            # For skills with symbols like c++, c#, .net, node.js
            escaped = re.escape(skill)
            if skill in ['c++', 'c#', '.net']:
                pattern = r'(?:\b|\s)' + escaped + r'(?:\b|\s|[,\.\n;])'
            elif '.' in skill:
                pattern = r'(?:\b)' + escaped + r'(?:\b|[,\.\n;])'
            else:
                pattern = r'\b' + escaped + r'\b'
                
            if re.search(pattern, text_lower):
                found.append(format_skill_name(skill))
        extracted[category] = sorted(list(set(found)))
        
    return extracted

def extract_all_skills_flat(text: str) -> list:
    """Returns flat list of all unique skills found in text."""
    cat_skills = extract_categorized_skills(text)
    all_skills = []
    for s_list in cat_skills.values():
        all_skills.extend(s_list)
    return sorted(list(set(all_skills)))

def perform_skill_gap_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Performs comprehensive skill gap analysis between Resume and Job Description.
    Returns categorized matches, missing skills, and overall coverage statistics.
    """
    resume_skills = extract_categorized_skills(resume_text)
    jd_skills = extract_categorized_skills(jd_text)
    
    matched_by_cat = {}
    missing_by_cat = {}
    
    total_jd_skills_count = 0
    matched_skills_count = 0
    
    all_matched = []
    all_missing = []
    
    for category in SKILL_TAXONOMY:
        res_set = set(resume_skills.get(category, []))
        jd_set = set(jd_skills.get(category, []))
        
        matched = sorted(list(jd_set.intersection(res_set)))
        missing = sorted(list(jd_set - res_set))
        
        matched_by_cat[category] = matched
        missing_by_cat[category] = missing
        
        total_jd_skills_count += len(jd_set)
        matched_skills_count += len(matched)
        
        all_matched.extend(matched)
        all_missing.extend(missing)
        
    # Skill Coverage Score
    if total_jd_skills_count > 0:
        coverage_score = round((matched_skills_count / total_jd_skills_count) * 100, 1)
    else:
        coverage_score = 75.0  # Default if JD didn't specify distinct keyword taxonomies
        
    return {
        "matched_by_category": matched_by_cat,
        "missing_by_category": missing_by_cat,
        "all_matched": sorted(list(set(all_matched))),
        "all_missing": sorted(list(set(all_missing))),
        "total_jd_skills": total_jd_skills_count,
        "total_matched": matched_skills_count,
        "total_missing": len(all_missing),
        "coverage_percentage": coverage_score
    }

def format_skill_name(skill: str) -> str:
    """Properly formats skill capitalization (e.g. PyTorch, React.js, AWS, etc.)."""
    special_cases = {
        "aws": "AWS", "gcp": "GCP", "azure": "Azure", "k8s": "Kubernetes",
        "ci/cd": "CI/CD", "sql": "SQL", "r": "R", "c": "C", "c++": "C++",
        "c#": "C#", ".net": ".NET", "nlp": "NLP", "llm": "LLM", "mlops": "MLOps",
        "tdd": "TDD", "rest api": "REST APIs", "graphql": "GraphQL",
        "react.js": "React.js", "node.js": "Node.js", "vue.js": "Vue.js", "next.js": "Next.js",
        "express.js": "Express.js", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
        "scikit-learn": "Scikit-Learn", "huggingface": "HuggingFace", "opencv": "OpenCV",
        "tableau": "Tableau", "power bi": "Power BI", "pyspark": "PySpark",
        "github": "GitHub", "gitlab ci": "GitLab CI", "github actions": "GitHub Actions",
        "mongodb": "MongoDB", "postgresql": "PostgreSQL", "postgres": "PostgreSQL"
    }
    low = skill.lower()
    if low in special_cases:
        return special_cases[low]
    if len(skill) <= 3:
        return skill.upper()
    return skill.title()
