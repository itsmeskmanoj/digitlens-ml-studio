"""Train and persist all five classifiers required for ML Assignment 2.

Dataset: Optical Recognition of Handwritten Digits (UCI copy distributed
with scikit-learn). The dataset has 1,797 instances and 64 numeric features.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RANDOM_SEED = 359
TARGET_COLUMN = "target"
TEST_SIZE = 0.20


def load_assignment_dataset() -> tuple[pd.DataFrame, pd.Series]:
    digits = load_digits()
    feature_names = [f"pixel_{index}" for index in range(digits.data.shape[1])]
    features = pd.DataFrame(digits.data, columns=feature_names)
    target = pd.Series(digits.target.astype(int), name=TARGET_COLUMN)
    return features, target


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "classifier",
                    DecisionTreeClassifier(
                        criterion="entropy",
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(n_neighbors=5, weights="distance", p=2),
                ),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [("classifier", GaussianNB(var_smoothing=1e-9))]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=4.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="weighted",
        ),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_assignment_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    model_files: dict[str, str] = {}
    results: list[dict[str, float | str]] = []

    for display_name, model in build_models().items():
        model.fit(x_train, y_train)
        safe_name = display_name.lower().replace("-", "_").replace(" ", "_")
        model_file = f"{safe_name}.joblib"
        joblib.dump(model, MODEL_DIR / model_file)
        model_files[display_name] = model_file
        results.append({"ML Model Name": display_name, **calculate_metrics(model, x_test, y_test)})

    metrics_frame = pd.DataFrame(results)
    metrics_frame.to_csv(ROOT / "model_metrics.csv", index=False, float_format="%.6f")

    metadata = {
        "dataset_name": "Optical Recognition of Handwritten Digits",
        "dataset_source": "UCI Machine Learning Repository; scikit-learn packaged copy",
        "instance_count": int(len(x)),
        "feature_count": int(x.shape[1]),
        "feature_columns": list(x.columns),
        "target_column": TARGET_COLUMN,
        "classes": sorted(int(value) for value in y.unique()),
        "test_instance_count": int(len(x_test)),
        "random_seed": RANDOM_SEED,
        "model_files": model_files,
    }
    (ROOT / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Training complete. Metrics:\n")
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
