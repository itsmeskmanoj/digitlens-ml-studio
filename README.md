# Digit Insight ML Studio - ML Assignment 2

**Student:** Manoj Kumar S  
**BITS ID:** 2026AK05359  
**Course:** Machine Learning

## A. Problem statement

The objective is to build and compare five supervised classification models that identify handwritten digits from pixel-intensity features. The trained models are exposed through an interactive Streamlit application that evaluates uploaded test data without retraining.

## B. Dataset description

The project uses the **Optical Recognition of Handwritten Digits** dataset originating from the UCI Machine Learning Repository and distributed with scikit-learn.

- Instances: 1,797
- Input features: 64 numeric pixel-intensity values
- Image representation: 8 x 8 grayscale image
- Target classes: digits 0 through 9
- Problem type: multiclass classification
- Train/test split: 80/20 stratified split
- Random seed: 359

The dataset satisfies the assignment requirements of at least 500 instances and 12 features.

## C. Links

- **GitHub repository:** https://github.com/itsmeskmanoj/digitlens-ml-studio
- **Live Streamlit application:** Not deployed yet

## D. Models and evaluation metrics

The same training and test partitions are used for every model. Precision, recall and F1 are weighted multiclass averages. AUC uses weighted one-vs-rest calculation.

<!-- RESULTS_TABLE_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9667 | 0.9993 | 0.9670 | 0.9667 | 0.9665 | 0.9630 |
| Decision Tree | 0.8472 | 0.9309 | 0.8530 | 0.8472 | 0.8451 | 0.8313 |
| K-Nearest Neighbors | 0.9806 | 0.9983 | 0.9812 | 0.9806 | 0.9805 | 0.9785 |
| Gaussian Naive Bayes | 0.7917 | 0.9639 | 0.8401 | 0.7917 | 0.7987 | 0.7743 |
| Support Vector Machine | 0.9833 | 0.9994 | 0.9842 | 0.9833 | 0.9830 | 0.9816 |
<!-- RESULTS_TABLE_END -->

## E. Model observations

### Logistic Regression

This linear baseline performs strongly because standardisation makes optimisation stable and many digit classes are separable in the 64-dimensional pixel space. L2 regularisation limits excessive coefficient growth.

### Decision Tree

The tree is interpretable and captures non-linear thresholds, but its isolated axis-aligned splits lose some spatial relationships between pixels. Depth and leaf-size controls reduce overfitting.

### K-Nearest Neighbors

KNN performs well because visually similar digits remain close in the standardised feature space. Distance weighting gives greater influence to the closest handwritten samples.

### Gaussian Naive Bayes

Naive Bayes is extremely fast but assumes conditional independence and Gaussian feature distributions. Adjacent image pixels are correlated, so its assumptions are weaker for this dataset.

### Support Vector Machine

The RBF SVM captures non-linear decision boundaries and generally provides the strongest balance across classes. Feature standardisation and the selected C value improve its margin behaviour.

### Overall winner

The **Support Vector Machine** is the overall winner. It achieved the highest accuracy (0.9833), weighted F1 score (0.9830), MCC (0.9816), and AUC (0.9994). This indicates strong discrimination and balanced multiclass performance on the held-out test data.

## F. Repository structure

```text
ml_assignment_2_digits/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model_metrics.csv
|-- model_metadata.json
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- k_nearest_neighbors.joblib
|   |-- gaussian_naive_bayes.joblib
|   `-- support_vector_machine.joblib
|-- RUN_AND_DEPLOY.md
`-- .gitignore
```

The BITS Virtual Lab execution screenshot is still required for the final submission.

## G. Local execution

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Open the local URL shown by Streamlit and upload `test_data.csv`.

## H. Streamlit functionality

- CSV upload for unseen test data
- Saved-model selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Five-model comparison
- Sample digit visualisation
- No model retraining in the web application

## I. Reproducibility

All models use random seed 359 where applicable. The training script regenerates the test CSV, model artifacts, metadata and metrics. The Streamlit app reads these saved files directly.
