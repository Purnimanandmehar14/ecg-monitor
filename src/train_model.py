"""
train_model.py
---------------
Trains an ML model (Random Forest by default) on the extracted ECG
features and evaluates it with accuracy / precision / recall / F1.

Uses GroupShuffleSplit on patient (record) ID so that beats from the
same patient never appear in both train and test sets -- this avoids
overly optimistic results from patient-level data leakage, which is a
common mistake in ECG ML papers.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from build_dataset import build_dataset


def train_and_evaluate(X, y, groups, model=None, test_size=0.25, random_state=42):
    if model is None:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")
    print("Per-class report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
    print("Confusion matrix (rows=true, cols=pred):")
    print(sorted(set(y)))
    print(cm)

    return model, {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def save_model(model, path="models/rf_arrhythmia_model.joblib"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # Small demo run -- expand record_list for a real training run
    X, y, groups, feature_names = build_dataset(
        ["100", "101", "103", "105", "106", "108", "109", "111", "200", "201"]
    )
    model, metrics = train_and_evaluate(X, y, groups)
    save_model(model, "../models/rf_arrhythmia_model.joblib")
