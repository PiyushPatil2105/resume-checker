"""
Resume Domain Classifier Training Pipeline.
Trains, evaluates, and exports a Machine Learning pipeline to classify resumes
and job descriptions into specific industry tech domains.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from models.dataset import build_training_dataset, DOMAINS

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "resume_classifier.joblib")

def train_and_export_model(samples_per_domain: int = 50, save_path: str = MODEL_PATH):
    print("=" * 60)
    print("[INFO] Training Resume Domain Classifier ML Pipeline")
    print("=" * 60)
    
    # 1. Generate / load dataset
    print(f"Generating balanced dataset ({samples_per_domain} samples/domain across {len(DOMAINS)} domains)...")
    texts, labels = build_training_dataset(samples_per_domain=samples_per_domain)
    print(f"Total training corpus size: {len(texts)} documents")
    
    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"Split: {len(X_train)} train samples, {len(X_test)} test samples.")
    
    # 3. Define ML Pipeline
    # Using sublinear_tf=True (logarithmic scaling of term frequency) and unigrams + bigrams
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000,
            stop_words='english',
            min_df=2
        )),
        ('clf', LogisticRegression(
            C=2.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # 4. Cross validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"5-Fold Cross Validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # 5. Fit model
    print("Fitting model on full training set...")
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n[EVAL] Test Set Accuracy: {test_acc * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 7. Retrain on complete dataset for maximum production coverage
    print("Retraining pipeline on 100% of the dataset for deployment...")
    pipeline.fit(texts, labels)
    
    # 8. Save artifact
    joblib.dump({
        'pipeline': pipeline,
        'domains': DOMAINS,
        'model_name': 'TF-IDF + Multinomial Logistic Regression Domain Classifier',
        'version': '1.0.0'
    }, save_path)
    print(f"[SUCCESS] Model artifact successfully saved to: {save_path}")
    
    # 9. Quick sanity check
    sample_text = "Experienced in PyTorch, Transformer models, HuggingFace, fine-tuning LLMs, computer vision, and MLOps."
    prediction = pipeline.predict([sample_text])[0]
    probabilities = pipeline.predict_proba([sample_text])[0]
    confidence = max(probabilities) * 100
    print(f"\nSanity Check Inference:")
    print(f"Input: '{sample_text}'")
    print(f"Predicted Domain: {prediction} ({confidence:.1f}% confidence)")
    
    return pipeline

if __name__ == "__main__":
    train_and_export_model()
