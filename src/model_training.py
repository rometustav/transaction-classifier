"""
This script trains and evaluates multiple models for transaction purpose classification.

Workflow:
    - Split data into train and test sets
    - Convert text into TF-IDF features
    - Train multiple classical ML models
    - Evaluate each model using accuracy, precision, recall, and F1-score
    - Compare results and select the best model
    - Save the best model and fitted vectorizer for later API use

Models trained:
    - Multinomial Naive Bayes
    - Logistic Regression
    - Decision Tree
    - Linear SVM

Usage:
    python src/train_models.py

Outputs:
    models/best_model.pkl
    models/vectorizer.pkl
    reports/model_comparison.csv
    reports/classification_reports.txt
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

RANDOM_STATE = 123
TEST_SIZE = 0.2

def evaluate_model(model, X_test, y_test) -> tuple[dict, str]:
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted": round(recall, 4),
        "f1_weighted": round(f1, 4)
    }

    report = classification_report(y_test, predictions, zero_division=0)

    return metrics, report


def train_and_compare() -> None:
    models_dir = Path("models")
    reports_dir = Path("reports")

    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    df = pd.read_csv(Path("data/transactions_cleaned.csv"))

    X = df["purpose_text"]
    y = df["transaction_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y # Maintain class distribution across train and test sets
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    candidate_models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    "DecisionTree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=20
    ),
    "LinearSVM": LinearSVC(
        random_state=RANDOM_STATE
    )}
    
    results = []
    reports_text = []
    fitted_pipelines = {}

    for model_name, classifier in candidate_models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False, # Already done in preprocessing
                    ngram_range=(1, 2), # To use 1-2 word features
                    min_df=1,
                    max_features=5000
                )
            ),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        metrics, report = evaluate_model(pipeline, X_test, y_test)

        result_row = {
            "model": model_name,
            **metrics
        }
        results.append(result_row)
        fitted_pipelines[model_name] = pipeline

        reports_text.append(f"{'=' * 80}\n")
        reports_text.append(f"MODEL: {model_name}\n")
        reports_text.append(f"{'-' * 80}\n")
        reports_text.append(
            f"Accuracy: {metrics['accuracy']}\n"
            f"Weighted Precision: {metrics['precision_weighted']}\n"
            f"Weighted Recall: {metrics['recall_weighted']}\n"
            f"Weighted F1: {metrics['f1_weighted']}\n\n"
        )
        reports_text.append(report)
        reports_text.append("\n\n")

        print(f"{model_name} results:")
        print(json.dumps(metrics, indent=2))

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_weighted", "accuracy"],
        ascending=False
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "model"]
    best_pipeline = fitted_pipelines[best_model_name]

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest model selected: {best_model_name}")

    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)
    with open(reports_dir / "classification_reports.txt", "w", encoding="utf-8") as f:
        f.writelines(reports_text)

    # Save the full best pipeline
    joblib.dump(best_pipeline, models_dir / "best_model.pkl")

    # Also save vectorizer separately for transparency / API flexibility
    vectorizer = best_pipeline.named_steps["tfidf"]
    joblib.dump(vectorizer, models_dir / "vectorizer.pkl")

    # Save classifier separately too
    classifier = best_pipeline.named_steps["classifier"]
    joblib.dump(classifier, models_dir / "classifier.pkl")

    print("\nSaved files:")
    print(f"- {models_dir / 'best_model.pkl'}")
    print(f"- {models_dir / 'vectorizer.pkl'}")
    print(f"- {models_dir / 'classifier.pkl'}")
    print(f"- {reports_dir / 'model_comparison.csv'}")
    print(f"- {reports_dir / 'classification_reports.txt'}")

    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    train_and_compare()