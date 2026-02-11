# ============================================================
# Heart Disease Classification Project
# ============================================================

# In this project, we build a machine learning model 
# to classify heart disease.

# IMPORTANT NOTES:
# - All types of disease (mild, moderate, severe, critical)
#   are treated as "disease" (binary classification).
# - We will assume that the disease is clinically significant,
#   meaning that missing a positive case could have serious consequences.
# - Therefore, the focus of the model is to be **conservative**, 
#   prioritizing recall to minimize false negatives.
# - We try to preserve precision as much as possible 
#   through techniques like threshold tuning.
# - This project is NOT intended to be clinically accurate.
#   The dataset is small and may include tests that a hospital 
#   would only request for patients already suspected of having 
#   some degree of heart disease.
# - The main goal is to explore data preprocessing, machine 
#   learning modeling, and evaluation techniques.


# ============================================================
#Imports
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# Data Loading & Cleaning
# ============================================================

df = pd.read_csv("data/heart_disease_uci.csv")

# Any disease level is treated as disease

df.loc[df["num"]>=1,"num"] = 1

# Understanding Data

print(df.info())
print(df.describe())
print(df.isna().sum())

# Replace physiologically impossible values

df["chol"].replace(0, np.nan, inplace=True)
df["trestbps"].replace(0, np.nan, inplace=True)

# Class Distribution

# The bar chart below is plotted to check whether the
# dataset is highly imbalanced. Since the classes are
# relatively balanced, no undersampling or oversampling
# techniques are applied.

y = df["num"]
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.ylabel("Number of Patients")
plt.xlabel("Class")
plt.title("Class Distribution in the Full Dataset")
plt.xticks([0, 1], ["No Disease", "Disease"])
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# Data Separation
# ============================================================
X = df.drop("num", axis=1)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=0,
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=0,
    stratify=y_temp
)

# ============================================================
# Preprocessing
# ============================================================
drop = ["id", "dataset"]
num = ["chol", "age", "trestbps", "oldpeak", "thalch"]
cat = ["cp", "slope", "restecg", "thal"]
ordinal = ["ca"]
binary = ["sex", "exang", "fbs"]

pipeline_num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

pipeline_cat = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

pipeline_bin = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("drop", "drop", drop),
    ("num", pipeline_num, num),
    ("cat", pipeline_cat, cat),
    ("ord", SimpleImputer(strategy="most_frequent"), ordinal),
    ("bin", pipeline_bin, binary)
])

preprocessor_forest = ColumnTransformer([
    ("drop", "drop", drop),
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", pipeline_cat, cat),
    ("ord", SimpleImputer(strategy="most_frequent"), ordinal),
    ("bin", pipeline_bin, binary)
])

# ============================================================
# Models
# ============================================================

forest = Pipeline([
    ("preprocessor", preprocessor_forest),
    ("model", RandomForestClassifier(random_state=0))
])

log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000
    ))
])

neural = Pipeline([
    ("preprocessor", preprocessor),
    ("model", MLPClassifier(max_iter=300, random_state=0))
])

# ============================================================
# Hyperparameter Tuning
# ============================================================

param_forest = {
    "model__n_estimators": [100, 200],
    "model__criterion": ["gini", "entropy"],
    "model__min_samples_leaf": [3, 5, 10]
}

param_log_reg = {
    "model__C": [0.01, 0.1, 0.5, 1, 5]
}

param_neural = {
    "model__hidden_layer_sizes": [(32, 32), (64, 64)]
}

gs_forest = GridSearchCV(
    forest, param_forest, scoring="recall", cv=5, n_jobs=-1
)

gs_log_reg = GridSearchCV(
    log_reg, param_log_reg, scoring="recall", cv=5, n_jobs=-1
)

gs_neural = GridSearchCV(
    neural, param_neural, scoring="recall", cv=5, n_jobs=-1
)
# ============================================================
# Training
# ============================================================

gs_forest.fit(X_train, y_train)
gs_log_reg.fit(X_train, y_train)
gs_neural.fit(X_train, y_train)

# ============================================================
#Metrics (CV and Validation Set) for Model Selection
# ============================================================

#Recall Comparison

best_scores = pd.DataFrame({
    'Random Forest': [gs_forest.best_score_],
      'Logistic Regression':  [gs_log_reg.best_score_],
     'MLP Classifier' :[gs_neural.best_score_]
    
},index=['Recall'])

plt.figure(figsize=(6, 4))
sns.heatmap(best_scores, annot=True, cmap="Blues", linewidths=0.5)
plt.title("Cross Validation Scores")
plt.tight_layout()
plt.savefig(FIG_DIR / "cv_recall_scores.png", dpi=200, bbox_inches="tight")
plt.show()

#Predictions (Validation Set)

y_forest = gs_forest.predict(X_val)
y_neural = gs_neural.predict(X_val)
y_log_reg = gs_log_reg.predict(X_val)

# Metrics Comparison (Validation Set)

metrics = pd.DataFrame({
    "Accuracy": [
        accuracy_score(y_val, y_forest),
        accuracy_score(y_val, y_neural),
        accuracy_score(y_val, y_log_reg)
    ],
    "Precision (Disease)": [
        precision_score(y_val, y_forest),
        precision_score(y_val, y_neural),
        precision_score(y_val, y_log_reg)
    ],
    "Recall (Disease)": [
        recall_score(y_val, y_forest),
        recall_score(y_val, y_neural),
        recall_score(y_val, y_log_reg)
    ],
    "F1 (Disease)": [
        f1_score(y_val, y_forest),
        f1_score(y_val, y_neural),
        f1_score(y_val, y_log_reg)
    ]
}, index=["Random Forest", "MLP", "Logistic Regression"])

plt.figure(figsize=(8, 6))
sns.heatmap(metrics, annot=True, cmap="Blues", linewidths=0.5)
plt.title("Classification Metrics (Validation Set)")
plt.tight_layout()
plt.savefig(FIG_DIR / "validation_metrics.png", dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# Model Choice & Precision–Recall Analysis
# ============================================================

# Random Forest achieved the best results and was
# selected as the intermediate model.
#First, we see if the model is calibrated or not

y_val_proba = gs_forest.best_estimator_.predict_proba(X_val)[:, 1]
prob_true, prob_pred = calibration_curve(y_val, y_val_proba, n_bins=10)

plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve ')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "calibration_curve.png", dpi=200, bbox_inches="tight")
plt.show()

# For most lower predicted probability ranges, the curve lies below the y = x line,
# indicating a tendency to overestimate disease risk.
# The model is particularly well calibrated in the 0.6–0.8 probability range,
# where calibration errors are relatively small.
# The most critical region occurs around 0.5 (decision boundary),
# where the model shows greater uncertainty and slight underestimation of probability.


# The Precision–Recall curve is analyzed to study the
# trade-off between recall and precision across different
# decision thresholds. Based on this curve, the default
# threshold (0.5) is adjusted in order to keep recall high
# while avoiding a significant loss in precision and
# overall accuracy. And this will be our final model.
# We choose a recall threshold of 0.95 to detect as many true positives as possible.
# This is a conservative approach: it prioritizes minimizing false negatives,
# even if it means accepting some false positives (lower precision).
# The threshold was selected based on the precision-recall curve on the validation set.


precision, recall, thresholds=precision_recall_curve(y_val,y_val_proba)
ap_score = average_precision_score(y_val, y_val_proba)
precision=precision[:-1]
recall=recall[:-1]
recall_min=0.95
idx_valids=np.where(recall>=recall_min)[0]
best_idx=idx_valids[np.argmax(precision[idx_valids])]
best_threshold=thresholds[best_idx]
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (AP = {ap_score:.2f})')

# Highlight the optimal point based on the best threshold
plt.scatter(
    recall[best_idx], 
    precision[best_idx], 
    color='red', 
    label=f"Best Threshold = {best_threshold:.2f}", 
    zorder=5
)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(FIG_DIR / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
plt.show()


# ============================================================
# Final Model Test (Validation Set)
# ============================================================

#Test in validation Set

y_val_final = (y_val_proba >= best_threshold).astype(int)

# Final Metrics and Confusion Matrix (Validation Set)

metrics = pd.DataFrame({
    "Accuracy": [
        accuracy_score(y_val, y_val_final)],
    "Precision (Disease)": [
        precision_score(y_val, y_val_final)
    ],
    "Recall (Disease)": [recall_score(y_val, y_val_final)],
    "F1 (Disease)": [
        f1_score(y_val, y_val_final)]
}, index=["Final Model"])

plt.figure(figsize=(8, 6))
sns.heatmap(metrics, annot=True, cmap="Blues", linewidths=0.5)
plt.title("Classification Metrics (Validation Set)")
plt.tight_layout()
plt.savefig(FIG_DIR / "validation_metrics_final_model.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_val, y_val_final),
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Confusion Matrix (Validation Set)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks([0.5, 1.5], ["No Disease", "Disease"])
plt.yticks([0.5, 1.5], ["No Disease", "Disease"])
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix_validation_final_model.png", dpi=200, bbox_inches="tight")
plt.show()


# ============================================================
# Final Model Test (Test Set)
# ============================================================

# Test 
y_test_proba= gs_forest.best_estimator_.predict_proba(X_test)[:, 1]


y_final = (y_test_proba >= best_threshold).astype(int)

# Final Metrics and Confusion Matrix (Test Set)

metrics = pd.DataFrame({
    "Accuracy": [
        accuracy_score(y_test, y_final)],
    "Precision (Disease)": [
        precision_score(y_test, y_final)
    ],
    "Recall (Disease)": [recall_score(y_test, y_final)],
    "F1 (Disease)": [
        f1_score(y_test, y_final)]
}, index=["Final Model"])

plt.figure(figsize=(8, 6))
sns.heatmap(metrics, annot=True, cmap="Blues", linewidths=0.5)
plt.title("Classification Metrics (Test Set)")
plt.tight_layout()
plt.savefig(FIG_DIR / "test_metrics_final_model.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_final),
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks([0.5, 1.5], ["No Disease", "Disease"])
plt.yticks([0.5, 1.5], ["No Disease", "Disease"])
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix_test_final_model.png", dpi=200, bbox_inches="tight")
plt.show()

# Conclusion: We built a conservative model that is able to identify most patients 
# who actually have the disease
# Furthermore, the model keeps the false positive rate relatively low. 
# The test results confirm our expectations.