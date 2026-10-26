import os, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import xgboost as xgb

os.makedirs("models", exist_ok=True)
SEED = 42
np.random.seed(SEED)

print("="*60)
print(" 1. DATA PREPARATION (APP METADATA)")
print("="*60)

df = pd.read_csv(
    r"E:\code things\DL_PROJECT\app_dataset.csv",
    sep=";", low_memory=False
)
# Deduplicate
df.drop_duplicates(inplace=True)

label_col = df.columns[-1]
y_raw = df[label_col].astype(str).str.strip().str.lower()

# Permission columns
perm_cols = sorted([c for c in df.columns if "android.permission" in c or "com.android" in c])

# Convert to numeric
for col in perm_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

X = df[perm_cols].copy()
print(f"Total features used: {X.shape[1]}")

le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Splitting 70/15/15
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

# Compute scale_pos_weight
neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
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
y_prob = model.predict_proba(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

print("\n" + "="*60)
print(" 4. SAVING ARTIFACTS")
print("="*60)

with open("models/app_model_xgb.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/app_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("models/app_feature_cols.json", "w") as f:
    json.dump(perm_cols, f)

print("Saved models/app_model_xgb.pkl and app_feature_cols.json")
