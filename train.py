import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")


# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("diabetes_prediction_dataset.csv")

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# PREPROCESSING
# -----------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

# -----------------------
# MODELS
# -----------------------
models = {
    "log_reg": {
        "model": LogisticRegression(max_iter=500),
        "params": {"model__C": [0.1, 1, 10]}
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10]
        }
    },
    "gradient_boost": {
        "model": GradientBoostingClassifier(),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1]
        }
    },
    "xgboost": {
        "model": XGBClassifier(eval_metric='logloss'),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 6]
        }
    },
    "lightgbm": {
        "model": LGBMClassifier(class_weight='balanced', verbose=-1),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1]
        }
    }
}

# -----------------------
# TRAINING LOOP
# -----------------------
best_models = {}
results = []

for name, config in models.items():
    print(f"Training {name}...")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", config["model"])
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=config["params"],
        n_iter=5,
        cv=3,
        scoring="recall",   # 🔥 focus recall
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    })

# -----------------------
# RESULTS
# -----------------------
results_df = pd.DataFrame(results).sort_values("recall", ascending=False)
print(results_df)

results_df.to_csv("model_results.csv", index=False)

# -----------------------
# BEST MODEL
# -----------------------
best_model_name = results_df.iloc[0]["model"]
best_model = best_models[best_model_name]

print("Best Model:", best_model_name)

# -----------------------
# THRESHOLD OPTIMIZATION
# -----------------------
y_prob = best_model.predict_proba(X_test)[:,1]

best_thresh = 0.5
best_recall = 0

for t in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_prob >= t).astype(int)
    rec = recall_score(y_test, y_pred_thresh)

    if rec > best_recall:
        best_recall = rec
        best_thresh = t

print("Best Threshold:", best_thresh)

# -----------------------
# SAVE
# -----------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(best_thresh, "threshold.pkl")