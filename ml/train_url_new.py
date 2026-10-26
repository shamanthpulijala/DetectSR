import os, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb

os.makedirs("models", exist_ok=True)
SEED = 42

print("="*60)
print(" 1. DATA PREPARATION (NEW URL DATASET)")
print("="*60)

df = pd.read_csv(r"E:\code things\DL_PROJECT\URL.csv")
print(f"Original shape: {df.shape}")

# Deduplicate
df.drop_duplicates(inplace=True)
print(f"After deduplication: {df.shape}")

# The last column is 'label'
X = df.drop("label", axis=1)
y = df["label"].values
feature_cols = list(X.columns)

X_train_tmp, X_test, y_train_tmp, y_test = train_test_split(
    X.values, y, test_size=0.15, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_tmp, y_train_tmp, test_size=0.15/0.85, random_state=SEED, stratify=y_train_tmp
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

print("\n" + "="*60)
print(" 2. MODEL TRAINING (XGBoost)")
print("="*60)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    early_stopping_rounds=15,
    eval_metric="auc"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(f"Best iteration: {model.best_iteration}")

print("\n" + "="*60)
print(" 3. TEST SET EVALUATION")
print("="*60)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malicious (1)"]))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\n" + "="*60)
print(" 4. SAVING ARTIFACTS")
print("="*60)
with open("models/url_xgb_new.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/url_xgb_features.json", "w") as f:
    json.dump(feature_cols, f)

print("Saved models/url_xgb_new.pkl")
