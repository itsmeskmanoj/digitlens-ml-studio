"""Interactive Streamlit evaluator for ML Assignment 2.

The app only loads already-trained models and evaluates uploaded test data.
It never fits or retrains a model in the Streamlit process.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Digit Insight ML Studio",
    page_icon="🔢",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%); }
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        color: white; background: linear-gradient(120deg, #0f4c81, #1f7a8c);
        box-shadow: 0 8px 24px rgba(15, 76, 129, 0.18); margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .35rem 0 0; opacity: .92; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #dbeafe; padding: .8rem;
        border-radius: 12px; box-shadow: 0 3px 12px rgba(15, 76, 129, .06);
    }
    </style>
    <div class="hero">
      <h1>Digit Insight ML Studio</h1>
      <p>Evaluate five saved classification models on unseen handwritten-digit test data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets() -> tuple[dict, dict[str, object]]:
    metadata = json.loads((ROOT / "model_metadata.json").read_text(encoding="utf-8"))
    models = {
        name: joblib.load(ROOT / "model" / filename)
        for name, filename in metadata["model_files"].items()
    }
    return metadata, models


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float], np.ndarray]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    return metrics, predictions


def validate_test_data(frame: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.Series]:
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")].copy()
    required = metadata["feature_columns"] + [metadata["target_column"]]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing[:10]))

    extra = [column for column in frame.columns if column not in required]
    if extra:
        st.info("Extra columns were ignored: " + ", ".join(extra[:10]))

    clean = frame[required].dropna().copy()
    if clean.empty:
        raise ValueError("No complete rows remain after removing missing values.")

    x_test = clean[metadata["feature_columns"]].apply(pd.to_numeric, errors="raise")
    y_test = pd.to_numeric(clean[metadata["target_column"]], errors="raise").astype(int)
    unknown_targets = sorted(set(y_test) - set(metadata["classes"]))
    if unknown_targets:
        raise ValueError(f"Unknown target classes: {unknown_targets}")
    return x_test, y_test


metadata, models = load_assets()

with st.sidebar:
    st.header("Assignment controls")
    st.caption("BITS ID: 2026AK05359")
    model_name = st.selectbox("Select a saved model", list(models))
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded_file is None)
    st.divider()
    st.write(f"**Dataset:** {metadata['dataset_name']}")
    st.write(f"**Features:** {metadata['feature_count']}")
    st.write(f"**Classes:** {len(metadata['classes'])}")
    st.caption("Models are loaded from disk; no retraining occurs in this app.")

overview_tab, evaluation_tab, comparison_tab = st.tabs(
    ["Dataset overview", "Model evaluation", "Model comparison"]
)

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
elif use_sample:
    raw_data = pd.read_csv(ROOT / "test_data.csv")
else:
    raw_data = None

with overview_tab:
    st.subheader("Dataset description")
    st.write(
        "The Optical Recognition of Handwritten Digits dataset contains 8 x 8 grayscale "
        "digit images. Each of the 64 input features is one pixel intensity, and the target "
        "is the digit class from 0 to 9."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total instances", f"{metadata['instance_count']:,}")
    c2.metric("Input features", metadata["feature_count"])
    c3.metric("Bundled test rows", metadata["test_instance_count"])
    if raw_data is not None:
        st.dataframe(raw_data.head(12), width="stretch")

with evaluation_tab:
    if raw_data is None:
        st.warning("Upload a CSV file or enable the bundled test data.")
    else:
        try:
            x_test, y_test = validate_test_data(raw_data, metadata)
            metrics, predictions = evaluate_model(models[model_name], x_test, y_test)

            st.subheader(model_name)
            metric_columns = st.columns(6)
            for column, (name, value) in zip(metric_columns, metrics.items()):
                column.metric(name, f"{value:.4f}")

            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### Confusion matrix")
                matrix = confusion_matrix(y_test, predictions, labels=metadata["classes"])
                fig, axis = plt.subplots(figsize=(7, 5.2))
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=metadata["classes"],
                    yticklabels=metadata["classes"],
                    ax=axis,
                )
                axis.set_xlabel("Predicted digit")
                axis.set_ylabel("Actual digit")
                fig.tight_layout()
                st.pyplot(fig)

            with right:
                st.markdown("#### Classification report")
                report = classification_report(
                    y_test,
                    predictions,
                    labels=metadata["classes"],
                    output_dict=True,
                    zero_division=0,
                )
                st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")

            st.markdown("#### Sample digit preview")
            sample_index = st.slider("Select test row", 0, len(x_test) - 1, 0)
            preview = x_test.iloc[sample_index].to_numpy().reshape(8, 8)
            fig, axis = plt.subplots(figsize=(2.6, 2.6))
            axis.imshow(preview, cmap="magma")
            axis.set_title(
                f"Actual: {int(y_test.iloc[sample_index])} | "
                f"Predicted: {int(predictions[sample_index])}"
            )
            axis.axis("off")
            st.pyplot(fig)
        except Exception as error:
            st.error(f"Unable to evaluate this CSV: {error}")

with comparison_tab:
    st.subheader("Five-model comparison on the held-out test set")
    comparison = pd.read_csv(ROOT / "model_metrics.csv")
    numeric_columns = comparison.columns[1:]
    comparison[numeric_columns] = comparison[numeric_columns].round(4)
    st.dataframe(comparison, width="stretch", hide_index=True)
    winner = comparison.loc[comparison["F1"].idxmax(), "ML Model Name"]
    st.success(f"Overall winner by weighted F1 score: {winner}")
