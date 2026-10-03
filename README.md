# Task 1: ML Classification Project

**Alfido Tech: Artificial Intelligence Internship**

Binary classification of breast tumours (malignant vs benign) on the Breast Cancer Wisconsin (Diagnostic) dataset, which ships with scikit-learn, so no download is needed.

## Results (held-out test set, 114 samples)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | **0.9825** | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| Logistic Regression (tuned) | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9937 |
| Random Forest (tuned) | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9934 |
| Gradient Boosting | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9907 |

Five-fold stratified cross-validation was run on the training set for every model (see the notebook, section 4).

## Project structure

```
task1-ml-classification/
├── ml_classification.ipynb     # full analysis with outputs
├── results_test_metrics.csv    # final test-set metrics
├── images/                     # saved plots used in the report
├── requirements.txt            # pinned package versions
└── README.md
```

## Environment setup

Tested with **Python 3.13**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the exact package versions
pip install -r requirements.txt
```

## How to run

Option A: Jupyter

```bash
jupyter notebook ml_classification.ipynb
# then: Kernel -> Restart & Run All
```

Option B: re-execute from the command line

```bash
jupyter nbconvert --to notebook --execute --inplace ml_classification.ipynb
```

Option C: Google Colab

Upload `ml_classification.ipynb` to [colab.research.google.com](https://colab.research.google.com) and choose Runtime -> Run all. Colab already has every library used here. Create an `images/` folder first, or the plot-saving lines will fail:

```python
import os; os.makedirs("images", exist_ok=True)
```

All random seeds are fixed (`RANDOM_STATE = 42`), so the numbers above are reproducible.

## Method summary

1. **EDA:** class balance (62.7% benign / 37.3% malignant), feature distributions, correlation with the target.
2. **Preprocessing:** 80/20 stratified split; `StandardScaler` inside a scikit-learn `Pipeline` so scaling is fitted on training folds only (no data leakage).
3. **Model comparison:** Logistic Regression, Random Forest and Gradient Boosting, each with 5-fold stratified CV (accuracy, precision, recall, F1, ROC-AUC).
4. **Tuning:** `GridSearchCV` (scoring = ROC-AUC) for Logistic Regression and Random Forest.
5. **Final evaluation:** all metrics on the untouched test set, plus ROC curves, confusion matrix and feature importance.

## Author

**Ezaz Ahamad**

- Candidate ID: BS/REG/127948
- Internship domain: Artificial Intelligence
- GitHub: https://github.com/ahmed-web-oss-wj
- LinkedIn: https://www.linkedin.com/in/ezaz-ahamad-440416336
- Repository: https://github.com/ahmed-web-oss-wj/alfido-task1-ml-classification
