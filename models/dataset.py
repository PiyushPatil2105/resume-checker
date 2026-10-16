"""
Dataset definitions and generation utilities for training the Resume Domain Classifier.
Generates balanced, high-quality synthetic training resumes across 10 industry tech domains.
"""

import random

DOMAINS = [
    "Data Science",
    "Machine Learning & AI",
    "Full Stack Web Development",
    "Frontend Development",
    "Backend Engineering",
    "DevOps & Cloud Engineering",
    "Cybersecurity",
    "Mobile App Development",
    "Quality Assurance & Testing",
    "Data Engineering"
]

DOMAIN_PROFILES = {
    "Data Science": {
        "titles": ["Data Scientist", "Lead Data Scientist", "Senior Statistical Analyst", "Predictive Analytics Specialist", "Quantitative Analyst"],
        "skills": [
            "Python, R, Pandas, NumPy, Scipy, Statsmodels, Scikit-Learn, Matplotlib, Seaborn, Tableau, Power BI",
            "Exploratory Data Analysis, A/B Testing, Hypothesis Testing, Statistical Modeling, Linear Regression, Logistic Regression",
            "SQL, PostgreSQL, BigQuery, Snowflake, Data Wrangling, Feature Engineering, Dimensionality Reduction, PCA",
            "Time Series Forecasting, ARIMA, Prophet, Customer Segmentation, K-Means Clustering, Random Forest, XGBoost"
        ],
        "experiences": [
            "Built predictive churn model using XGBoost and Scikit-Learn achieving 89% accuracy, boosting retention by 14%.",
            "Designed and executed 25+ randomized A/B experiments to evaluate feature rollouts, driving $1.2M incremental revenue.",
            "Formulated statistical hypothesis tests and automated executive reporting pipelines in BigQuery and Tableau.",
            "Cleaned and preprocessed 10M+ rows of noisy customer transactional data using Pandas and PySpark."
        ]
    },
    "Machine Learning & AI": {
        "titles": ["Machine Learning Engineer", "AI Research Engineer", "Deep Learning Engineer", "NLP Specialist", "Computer Vision Engineer"],
        "skills": [
            "PyTorch, TensorFlow, Keras, HuggingFace Transformers, BERT, GPT, LLM Fine-tuning, LangChain, LlamaIndex",
            "Computer Vision, OpenCV, YOLO, Object Detection, ResNet, CNN, GANs, Stable Diffusion",
            "NLP, Named Entity Recognition, Tokenization, Word2Vec, GloVe, Vector Databases, Pinecone, ChromaDB, FAISS",
            "MLOps, MLflow, Weights & Biases, DVC, Triton Inference Server, ONNX, Docker, Kubernetes, Ray"
        ],
        "experiences": [
            "Trained and fine-tuned BERT and RoBERTa models for multi-class document classification with 94.2% F1 score.",
            "Architected Retrieval-Augmented Generation (RAG) system using LangChain, FAISS, and Llama-3 for enterprise search.",
            "Deployed real-time computer vision inference pipeline utilizing YOLOv8 and TensorRT, processing 60 FPS on edge devices.",
            "Implemented end-to-end MLOps pipeline on AWS SageMaker with automated model retraining, drift monitoring, and MLflow tracking."
        ]
    },
    "Full Stack Web Development": {
        "titles": ["Full Stack Developer", "Senior Full Stack Engineer", "MERN Stack Developer", "Full Stack Software Engineer"],
        "skills": [
            "JavaScript, TypeScript, React.js, Next.js, Node.js, Express.js, HTML5, CSS3, Tailwind CSS, Redux Toolkit",
            "RESTful APIs, GraphQL, MongoDB, PostgreSQL, Prisma, TypeORM, WebSockets, JWT Authentication, Redis",
            "Git, GitHub, CI/CD, Docker, AWS S3, Vercel, Microservices, Responsive UI Design, State Management",
            "Webpack, Vite, Jest, Cypress, Supertest, Agile Scrum, Sprint Planning, System Architecture"
        ],
        "experiences": [
            "Engineered full-stack SaaS platform utilizing Next.js, Node.js, TypeScript, and MongoDB serving 50,000+ monthly active users.",
            "Developed responsive modern user interfaces with Tailwind CSS and React, decreasing page load times by 42%.",
            "Integrated secure Stripe payment gateways, OAuth2 social login, and real-time Socket.io notifications.",
            "Designed and documented RESTful and GraphQL APIs with OpenAPI/Swagger specifications and PostgreSQL database backing."
        ]
    },
    "Frontend Development": {
        "titles": ["Frontend Developer", "Senior Frontend Engineer", "UI Engineer", "React Frontend Specialist", "Web UI Developer"],
        "skills": [
            "React.js, Vue.js, Angular, TypeScript, JavaScript ES6+, HTML5, Semantic Web, CSS3, SCSS, Sass",
            "Tailwind CSS, Material UI, Chakra UI, Framer Motion, Styled Components, Responsive Design, CSS Grid, Flexbox",
            "Redux, Zustand, Pinia, React Query, TanStack, Webpack, Vite, Web Performance Optimization, Core Web Vitals",
            "Jest, React Testing Library, Storybook, Figma, WCAG Accessibility, Cross-Browser Compatibility"
        ],
        "experiences": [
            "Spearheaded redesign of web app using React 18, TypeScript, and Tailwind CSS, increasing accessibility score to 98/100.",
            "Optimized frontend bundle size by 35% via dynamic code splitting, tree shaking, and asset lazy-loading in Vite.",
            "Constructed modular reusable design system in Storybook adopted across 4 engineering teams.",
            "Collaborated closely with UI/UX designers in Figma to translate wireframes into pixel-perfect responsive components."
        ]
    },
    "Backend Engineering": {
        "titles": ["Backend Engineer", "Senior Backend Developer", "Java API Developer", "Golang Systems Engineer", "Python Backend Developer"],
        "skills": [
            "Python, Django, FastAPI, Flask, Java, Spring Boot, Go, Golang, C#, .NET Core, Microservices Architecture",
            "PostgreSQL, MySQL, Redis, RabbitMQ, Apache Kafka, Elasticsearch, gRPC, REST APIs, GraphQL",
            "System Design, Distributed Systems, Concurrency, Multithreading, Database Indexing, Query Optimization, ORM",
            "Docker, Kubernetes, Linux CLI, Bash, Nginx, OAuth2, JWT, Unit Testing, Pytest, Mocking"
        ],
        "experiences": [
            "Architected high-throughput microservices in Go and FastAPI handling over 15,000 requests per second with <20ms p99 latency.",
            "Optimized complex PostgreSQL queries, indexes, and connection pooling, reducing database server load by 50%.",
            "Implemented event-driven asynchronous messaging pipeline utilizing Apache Kafka and Redis Pub/Sub.",
            "Designed resilient distributed cache architectures reducing backend load by 65% during peak traffic spikes."
        ]
    },
    "DevOps & Cloud Engineering": {
        "titles": ["DevOps Engineer", "Cloud Infrastructure Engineer", "Site Reliability Engineer (SRE)", "Platform Engineer", "AWS Solutions Architect"],
        "skills": [
            "AWS, Azure, Google Cloud Platform (GCP), Terraform, Ansible, Infrastructure as Code (IaC), CloudFormation",
            "Docker, Containerization, Kubernetes (K8s), Helm, Docker Compose, Service Mesh, Istio",
            "CI/CD, GitHub Actions, GitLab CI, Jenkins, ArgoCD, GitOps, SonarQube",
            "Linux Administration, Shell Scripting, Bash, Prometheus, Grafana, ELK Stack, Datadog, CloudWatch, OpenTelemetry"
        ],
        "experiences": [
            "Automated multi-region cloud infrastructure provisioning using Terraform and AWS (EKS, RDS, VPC, CloudFront).",
            "Constructed zero-downtime CI/CD deployment pipelines in GitHub Actions and ArgoCD, slashing release cycles from days to minutes.",
            "Configured robust Kubernetes clusters hosting 40+ microservices with autoscaling, ingress controllers, and Helm charts.",
            "Established comprehensive observability stack utilizing Prometheus, Grafana, and Datadog, cutting MTTR by 45%."
        ]
    },
    "Cybersecurity": {
        "titles": ["Cybersecurity Analyst", "Information Security Engineer", "Penetration Tester", "SOC Analyst", "Security Architect"],
        "skills": [
            "Network Security, Vulnerability Assessment, Penetration Testing, Ethical Hacking, Kali Linux, Metasploit, Burp Suite, Wireshark",
            "SIEM, Splunk, Microsoft Sentinel, IBM QRadar, SOC Monitoring, Incident Response, Threat Hunting, MITRE ATT&CK",
            "Identity and Access Management (IAM), Zero Trust Architecture, Cryptography, SSL/TLS, PKI, Firewalls, IDS/IPS",
            "Compliance, ISO 27001, SOC 2, NIST, GDPR, HIPAA, OWASP Top 10, Secure Code Review, SAST, DAST"
        ],
        "experiences": [
            "Executed vulnerability scans and web application penetration tests against OWASP Top 10 vulnerabilities using Burp Suite.",
            "Triaged 200+ security alerts weekly in Splunk SIEM, orchestrating incident response protocols to prevent intrusions.",
            "Implemented Zero Trust IAM policies, multi-factor authentication (MFA), and role-based access control across 1,500 enterprise users.",
            "Audited enterprise infrastructure compliance against SOC 2 Type II and ISO 27001 standards, closing 34 audit findings."
        ]
    },
    "Mobile App Development": {
        "titles": ["Mobile Application Developer", "iOS Developer", "Android Developer", "React Native Developer", "Flutter Developer"],
        "skills": [
            "React Native, Flutter, Dart, Swift, SwiftUI, Objective-C, Kotlin, Java Android SDK, Mobile Architecture (MVVM, Clean Architecture)",
            "iOS App Store Deployment, Google Play Console, TestFlight, Fastlane, Push Notifications, Firebase Cloud Messaging",
            "CoreData, Room Database, SQLite, Realm, Offline Storage, REST API integration, GraphQL, Mobile UI Animations",
            "Mobile Performance Optimization, Memory Leak Profiling, App Size Reduction, Responsive Screen Adaptability"
        ],
        "experiences": [
            "Developed and published cross-platform mobile app using Flutter and Dart with 100,000+ active downloads on Play Store and App Store.",
            "Engineered native iOS application in Swift and SwiftUI implementing offline caching with CoreData and background sync.",
            "Integrated Firebase push notifications, biometrics (FaceID/Fingerprint), and Apple Pay / Google In-App Purchases.",
            "Improved app frame rate to consistent 60 FPS by profiling memory usage and optimizing widget rebuilding lifecycles."
        ]
    },
    "Quality Assurance & Testing": {
        "titles": ["QA Automation Engineer", "Software Development Engineer in Test (SDET)", "Senior QA Analyst", "Test Automation Lead"],
        "skills": [
            "Test Automation, Selenium WebDriver, Cypress, Playwright, Appium, Cucumber, BDD, TestNG, JUnit, Pytest",
            "API Testing, Postman, RestAssured, Newman, Performance Testing, JMeter, Locust, LoadRunner",
            "Defect Tracking, Jira, TestRail, Zephyr, Test Strategy, Test Case Design, Regression Testing, Smoke Testing",
            "CI/CD Integration, Jenkins, Git, Docker for Test Environments, Cross-Browser Testing, BrowserStack"
        ],
        "experiences": [
            "Architected test automation framework in Playwright and TypeScript covering 450+ end-to-end regression test cases.",
            "Reduced regression test execution duration by 70% through parallel test execution on Docker containers in Jenkins CI/CD.",
            "Conducted load and stress tests with JMeter identifying API bottlenecks at 10,000 concurrent users.",
            "Authored comprehensive test plans, manual test suites, and API validations using Postman and RestAssured."
        ]
    },
    "Data Engineering": {
        "titles": ["Data Engineer", "Senior Big Data Engineer", "ETL Developer", "Cloud Data Pipeline Specialist"],
        "skills": [
            "Apache Spark, PySpark, Databricks, Apache Airflow, dbt, Apache Kafka, Hadoop, MapReduce",
            "SQL, PostgreSQL, Snowflake, Google BigQuery, Amazon Redshift, Delta Lake, Parquet",
            "ETL / ELT Pipelines, Data Modeling, Star Schema, Snowflake Schema, Data Warehousing, Data Lake",
            "Python, Scala, AWS Glue, AWS EMR, Azure Data Factory, S3, Docker, Git"
        ],
        "experiences": [
            "Engineered scalable ELT data pipelines in Apache Airflow and dbt processing 5TB+ daily streaming data into Snowflake.",
            "Migrated legacy on-premise Hadoop cluster to Databricks and PySpark, reducing cluster compute costs by 38%.",
            "Designed star-schema dimensional models in BigQuery enabling sub-second query latency for business intelligence dashboards.",
            "Built fault-tolerant real-time ingestion pipelines using Apache Kafka and Spark Streaming with zero data loss."
        ]
    }
}

INTRO_TEMPLATES = [
    "Experienced and detail-oriented {title} with over {years} years of demonstrated history in delivering high-impact technological solutions.",
    "Driven {title} passionate about solving complex engineering challenges and optimizing system performance with modern industry practices.",
    "Results-focused {title} having {years}+ years of expertise designing, building, and deploying enterprise-grade scalable applications.",
    "Innovative {title} adept in leveraging cutting-edge tools and methodologies to accelerate product delivery and maintain code quality.",
    "Motivated technical professional specializing as a {title} with a track record of driving operational efficiency and business value."
]

def generate_sample_resume(domain: str) -> str:
    """Generates a realistic resume text snippet for a given domain."""
    profile = DOMAIN_PROFILES[domain]
    title = random.choice(profile["titles"])
    years = random.randint(2, 9)
    intro = random.choice(INTRO_TEMPLATES).format(title=title, years=years)
    
    # Pick skills
    selected_skills = random.sample(profile["skills"], k=min(3, len(profile["skills"])))
    skills_text = " | ".join(selected_skills)
    
    # Pick experiences
    num_exp = random.randint(2, 4)
    selected_exp = random.sample(profile["experiences"], k=min(num_exp, len(profile["experiences"])))
    exp_text = "\n".join([f"• {exp}" for exp in selected_exp])
    
    resume_text = f"""
Summary:
{intro}

Core Competencies & Technical Skills:
{skills_text}

Professional Experience:
{title} | Tech Innovations Inc. ({years} years experience)
{exp_text}

Education & Certifications:
Bachelor of Science in Computer Science / Information Technology
Relevant Certifications in {domain}
"""
    return resume_text.strip()

def build_training_dataset(samples_per_domain: int = 40):
    """Generates a balanced dataset of (text, domain_label)."""
    texts = []
    labels = []
    
    for domain in DOMAINS:
        for _ in range(samples_per_domain):
            text = generate_sample_resume(domain)
            texts.append(text)
            labels.append(domain)
            
    return texts, labels

if __name__ == "__main__":
    texts, labels = build_training_dataset(samples_per_domain=2)
    print(f"Generated {len(texts)} samples across {len(set(labels))} domains.")
    print("Sample:\n", texts[0])
