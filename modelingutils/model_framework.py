import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import json
import pickle

# %%

dt_clf = DecisionTreeClassifier(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)
lm_clf = LogisticRegression(random_state=42)

dt_param_grid = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [None, 2, 4, 6],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf": [1, 2, 5, 10],
}

rf_param_grid = {
    "clf__n_estimators": [100, 500, 1000],
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [None, 2, 4, 6],
    "clf__min_samples_split": [2, 10, 20],
    "clf__min_samples_leaf": [1, 5, 10],
    "clf__max_features": ["sqrt", "log2"],
    "clf__max_samples": [0.25, 0.5, 0.75, 1],
}

gb_param_grid = {
    "clf__n_estimators": [100, 500, 1000],
    "clf__learning_rate": [0.001, 0.01, 0.3],
    "clf__max_depth": [None, 2, 5],
    "clf__min_samples_split": [2, 5, 20],
    "clf__min_samples_leaf": [1, 5, 10],
}

lm_param_grid = {
    "clf__penalty": ["l1", "l2", "elasticnet", "none"],
    "clf__C": [0.001, 0.01, 0.1, 0.2, 0.3],
    "clf__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "clf__max_iter": [100, 200, 300, 400, 500],
}


mdls_dict = {
    "Decision Tree": {
        "model": dt_clf,
        "param_grid": dt_param_grid,
        "master_pred_proba": [],
        "master_pred": [],
        "master_true": [],
        "master_index": [],
    },
    "Random Forest": {
        "model": rf_clf,
        "param_grid": rf_param_grid,
        "master_pred_proba": [],
        "master_pred": [],
        "master_true": [],
        "master_index": [],
    },
    "Gradient Boosting": {
        "model": gb_clf,
        "param_grid": gb_param_grid,
        "master_pred_proba": [],
        "master_pred": [],
        "master_true": [],
        "master_index": [],
    },
    "Logistic Regression": {
        "model": lm_clf,
        "param_grid": lm_param_grid,
        "master_pred_proba": [],
        "master_pred": [],
        "master_true": [],
        "master_index": [],
    },
}


cat_cols = []

category_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", category_transformer, cat_cols),
    ],
    remainder="passthrough",
)


def make_pipe(clf):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )


cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
