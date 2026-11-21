"""
Model Training, Evaluation, and Inference Engine
Day 22 of 100 Days of Machine Learning: NLP Text Classification with TF-IDF

Trains Multinomial Naive Bayes, Logistic Regression, and Random Forest on
SMS Spam data, evaluates precision/recall/F1, extracts top predictive n-grams,
and provides real-time explainable inference.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.preprocessor import clean_text


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sms_spam_collection.csv")


def load_and_preprocess_dataset(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads CSV, applies text preprocessing, and prepares binary targets."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['message', 'label']).copy()
    # Normalize labels: ham -> 0, spam -> 1
    df['target'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})
    df['cleaned_message'] = df['message'].apply(lambda msg: clean_text(str(msg), stem=True))
    # Filter out empty cleaned strings
    df = df[df['cleaned_message'].str.len() > 0].reset_index(drop=True)
    return df


def train_models(
    df: pd.DataFrame = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Trains TF-IDF vectorizer and classifiers, evaluates metrics, and extracts
    top predictive features.
    """
    if df is None:
        df = load_and_preprocess_dataset()

    X = df['cleaned_message']
    y = df['target']

    # Stratified split to preserve 87/13 ham/spam ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 2. Candidate Models
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42)
    }

    results = {}
    trained_estimators = {}

    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1]
        }
        trained_estimators[name] = clf

    # 3. Extract Top Predictive Features (from Naive Bayes log probabilities & LogReg coefficients)
    feature_names = np.array(vectorizer.get_feature_names_out())
    nb = trained_estimators["Multinomial Naive Bayes"]
    lr = trained_estimators["Logistic Regression"]

    # Log probability difference: log P(w | spam) - log P(w | ham)
    nb_spam_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    top_spam_idx = np.argsort(nb_spam_diff)[-25:][::-1]
    top_ham_idx = np.argsort(nb_spam_diff)[:25]

    top_features = {
        "top_spam_nb": [
            {"term": feature_names[i], "score": round(float(nb_spam_diff[i]), 4)}
            for i in top_spam_idx
        ],
        "top_ham_nb": [
            {"term": feature_names[i], "score": round(float(abs(nb_spam_diff[i])), 4)}
            for i in top_ham_idx
        ],
        "lr_coefficients": [
            {"term": feature_names[i], "coef": round(float(lr.coef_[0][i]), 4)}
            for i in np.argsort(lr.coef_[0])[-25:][::-1]
        ]
    }

    # 4. Save Artifacts
    if save_artifacts:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        joblib.dump(trained_estimators["Multinomial Naive Bayes"], os.path.join(MODELS_DIR, "nb_model.joblib"))
        joblib.dump(trained_estimators["Logistic Regression"], os.path.join(MODELS_DIR, "lr_model.joblib"))
        joblib.dump(trained_estimators["Random Forest"], os.path.join(MODELS_DIR, "rf_model.joblib"))

        with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(MODELS_DIR, "top_features.json"), "w", encoding="utf-8") as f:
            json.dump(top_features, f, indent=2)

    return {
        "metrics": results,
        "top_features": top_features,
        "vectorizer": vectorizer,
        "models": trained_estimators,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "vocab_size": len(feature_names)
    }


def get_or_train_pipeline() -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Loads saved models and vectorizer, or trains them if not present.
    Returns: (vectorizer, models_dict, metrics_dict, top_features_dict)
    """
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    nb_path = os.path.join(MODELS_DIR, "nb_model.joblib")
    lr_path = os.path.join(MODELS_DIR, "lr_model.joblib")
    rf_path = os.path.join(MODELS_DIR, "rf_model.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics_summary.json")
    feat_path = os.path.join(MODELS_DIR, "top_features.json")

    all_exist = all(os.path.exists(p) for p in [vec_path, nb_path, lr_path, rf_path, metrics_path, feat_path])

    if all_exist:
        vectorizer = joblib.load(vec_path)
        models = {
            "Multinomial Naive Bayes": joblib.load(nb_path),
            "Logistic Regression": joblib.load(lr_path),
            "Random Forest": joblib.load(rf_path)
        }
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(feat_path, "r", encoding="utf-8") as f:
            top_features = json.load(f)
        return vectorizer, models, metrics, top_features
    else:
        pkg = train_models(save_artifacts=True)
        return pkg["vectorizer"], pkg["models"], pkg["metrics"], pkg["top_features"]


def predict_message(
    text: str,
    model_name: str = "Multinomial Naive Bayes"
) -> Dict[str, Any]:
    """
    Performs real-time inference on arbitrary user text with token-level attribution.
    """
    vectorizer, models, _, top_features = get_or_train_pipeline()
    model = models.get(model_name, models["Multinomial Naive Bayes"])

    cleaned = clean_text(text, stem=True)
    if not cleaned.strip():
        return {
            "label": "HAM",
            "is_spam": False,
            "spam_probability": 0.0,
            "ham_probability": 1.0,
            "cleaned_text": "",
            "token_attributions": []
        }

    vec = vectorizer.transform([cleaned])
    
    # Probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        p_ham = float(probs[0])
        p_spam = float(probs[1])
    else:
        pred = model.predict(vec)[0]
        p_spam = 1.0 if pred == 1 else 0.0
        p_ham = 1.0 - p_spam

    is_spam = bool(p_spam >= 0.50)
    label = "SPAM" if is_spam else "HAM"

    # Token attribution via Naive Bayes log-ratio or LR weights
    feature_names = vectorizer.get_feature_names_out()
    vocab_map = {name: idx for idx, name in enumerate(feature_names)}

    nb_model = models["Multinomial Naive Bayes"]
    nb_diff = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]

    tokens = cleaned.split()
    attributions = []

    for t in tokens:
        if t in vocab_map:
            feat_idx = vocab_map[t]
            score = float(nb_diff[feat_idx])
            influence = "spam" if score > 0.5 else ("ham" if score < -0.5 else "neutral")
            attributions.append({
                "token": t,
                "score": round(score, 3),
                "influence": influence
            })
        else:
            attributions.append({
                "token": t,
                "score": 0.0,
                "influence": "neutral"
            })

    return {
        "label": label,
        "is_spam": is_spam,
        "spam_probability": round(p_spam, 4),
        "ham_probability": round(p_ham, 4),
        "cleaned_text": cleaned,
        "token_attributions": attributions
    }


if __name__ == "__main__":
    print("Training SMS Spam Classifier models...")
    res = train_models(save_artifacts=True)
    print("\nTraining complete! Benchmark Metrics:")
    for m_name, m_res in res["metrics"].items():
        print(f"[{m_name}]")
        print(f"  Accuracy:  {m_res['accuracy']*100:.2f}%")
        print(f"  Precision: {m_res['precision']*100:.2f}%")
        print(f"  Recall:    {m_res['recall']*100:.2f}%")
        print(f"  F1-Score:  {m_res['f1_score']:.4f}")
        print(f"  Confusion Matrix: TN={m_res['tn']}, FP={m_res['fp']}, FN={m_res['fn']}, TP={m_res['tp']}")
