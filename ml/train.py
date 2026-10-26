"""
DetectShield — Spam Message Classifier
Train a Bi-LSTM (and plain LSTM baseline) on the SMS Spam Collection dataset.

Usage:
    python train.py

Outputs:
    models/bilstm_spam.keras      <- best Bi-LSTM model
    models/lstm_spam.keras        <- best plain LSTM model
    models/tokenizer.pkl          <- fitted Keras tokenizer
    models/label_encoder.pkl      <- label encoder (ham=0, spam=1)
    models/training_config.json   <- vocab_size, max_len, thresholds
"""

import os, json, pickle, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ─────────────────────────── CONFIG ────────────────────────────
DATASET_PATH  = "../Spam_SMS.csv"   # adjust if running from a different CWD
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"
RANDOM_SEED   = 42
TEST_SIZE     = 0.15               # 15 % test
VAL_SIZE      = 0.15               # 15 % validation (of the remaining 85 %)
VOCAB_SIZE    = 10_000             # top 10 k words (covers >99 % of dataset vocabulary)
MAX_LEN       = 150                # 99th-percentile message length in tokens
EMBEDDING_DIM = 64
LSTM_UNITS    = 64
DROPOUT_RATE  = 0.4
EPOCHS        = 30
BATCH_SIZE    = 64
# ───────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. LOADING & EXPLORING DATASET")
print("="*60)

df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", usecols=["Class", "Message"])
df.columns = ["label", "message"]
df.dropna(inplace=True)

print(f"Total samples : {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\nMessage length (chars) — mean: {df['message'].str.len().mean():.1f}, "
      f"max: {df['message'].str.len().max()}, min: {df['message'].str.len().min()}")


# ════════════════════════════════════════════════════════════════
# 2. TEXT CLEANING
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. TEXT CLEANING")
print("="*60)

import re

def clean_text(text: str) -> str:
    """
    Cleaning rules:
      - Lowercase
      - Replace currency symbols with the token CURRENCY (signal for spam)
      - Collapse digit sequences to NUM (keeps number-as-signal, removes noise)
      - Keep URLs as URL_TOKEN (strong spam signal)
      - Strip leftover punctuation / extra whitespace
    We intentionally KEEP urgency words like 'free', 'win', 'claim'.
    """
    text = str(text).lower()
    # Preserve URL structure as token
    text = re.sub(r"https?://\S+|www\.\S+", " URL_TOKEN ", text)
    # Preserve currency as token
    text = re.sub(r"[£$€¥₹]", " CURRENCY ", text)
    # Collapse numbers
    text = re.sub(r"\b\d+\b", " NUM ", text)
    # Remove remaining special characters (keep alphanumeric + space)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df["message"].apply(clean_text)
print("Sample cleaned messages:")
print(df[["message", "clean"]].head(3).to_string())


# ════════════════════════════════════════════════════════════════
# 3. ENCODE LABELS
# ════════════════════════════════════════════════════════════════
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])   # ham=0, spam=1
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# ════════════════════════════════════════════════════════════════
# 4. TRAIN / VAL / TEST SPLIT
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4. TRAIN / VAL / TEST SPLIT")
print("="*60)

X = df["clean"].values
y = df["label_enc"].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=VAL_SIZE / (1 - TEST_SIZE),
    stratify=y_trainval, random_state=RANDOM_SEED
)
print(f"Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")


# ════════════════════════════════════════════════════════════════
# 5. TOKENISE & PAD
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  5. TOKENISE & PAD  (vocab_size={}, max_len={})".format(VOCAB_SIZE, MAX_LEN))
print("="*60)
"""
Why VOCAB_SIZE=10,000?
  The SMS Spam Collection has ~8,700 unique tokens after cleaning.
  10,000 covers 100 % of them with room for OOV.

Why MAX_LEN=150?
  99th-percentile message length in word-tokens is ~130.
  We use 150 to capture virtually all messages without padding waste.
  (The dataset's longest message is 910 chars ≈ ~170 tokens — we truncate the tail.)
"""
tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>",
    lower=True
)
tokenizer.fit_on_texts(X_train)       # fit ONLY on training data — no data leakage
word_index = tokenizer.word_index
print(f"Actual vocabulary size (after cleaning): {len(word_index)}")

def texts_to_padded(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

X_train_pad = texts_to_padded(X_train)
X_val_pad   = texts_to_padded(X_val)
X_test_pad  = texts_to_padded(X_test)

print(f"X_train_pad shape: {X_train_pad.shape}")


# ════════════════════════════════════════════════════════════════
# 6. CLASS WEIGHTS (handle 6.5:1 imbalance)
# ════════════════════════════════════════════════════════════════
n_ham  = np.sum(y_train == 0)
n_spam = np.sum(y_train == 1)
total  = n_ham + n_spam
class_weights = {0: total / (2 * n_ham), 1: total / (2 * n_spam)}
print(f"\nClass weights -> ham: {class_weights[0]:.3f}, spam: {class_weights[1]:.3f}")


# ════════════════════════════════════════════════════════════════
# 7. BUILD MODELS
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  7. BUILDING MODELS")
print("="*60)

def build_bilstm(vocab_size, max_len, embed_dim, lstm_units, dropout):
    """
    Architecture:
      Embedding (64-d) → Bidirectional LSTM (64 forward + 64 backward)
      → Dropout → Dense(32, relu) → Dropout → Dense(1, sigmoid)
    """
    inp = keras.Input(shape=(max_len,), name="token_ids")
    x   = layers.Embedding(vocab_size, embed_dim, name="embedding")(inp)
    x   = layers.Bidirectional(
              layers.LSTM(lstm_units, recurrent_dropout=0.1), name="bi_lstm")(x)
    x   = layers.Dropout(dropout, name="dropout_1")(x)
    x   = layers.Dense(32, activation="relu", name="dense_1")(x)
    x   = layers.Dropout(dropout * 0.5, name="dropout_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inp, out, name="BiLSTM_Spam")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")]
    )
    return model

def build_lstm(vocab_size, max_len, embed_dim, lstm_units, dropout):
    """Plain LSTM baseline (same hyper-parameters, no Bidirectional wrapper)."""
    inp = keras.Input(shape=(max_len,), name="token_ids")
    x   = layers.Embedding(vocab_size, embed_dim, name="embedding")(inp)
    x   = layers.LSTM(lstm_units, recurrent_dropout=0.1, name="lstm")(x)
    x   = layers.Dropout(dropout, name="dropout_1")(x)
    x   = layers.Dense(32, activation="relu", name="dense_1")(x)
    x   = layers.Dropout(dropout * 0.5, name="dropout_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inp, out, name="LSTM_Spam")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")]
    )
    return model

bilstm_model = build_bilstm(VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE)
lstm_model   = build_lstm(VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE)

print("\nBi-LSTM summary:")
bilstm_model.summary()
print("\nLSTM (baseline) summary:")
lstm_model.summary()


# ════════════════════════════════════════════════════════════════
# 8. CALLBACKS
# ════════════════════════════════════════════════════════════════
def get_callbacks(checkpoint_path: str):
    return [
        EarlyStopping(
            monitor="val_auc", mode="max",
            patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
    ]


# ════════════════════════════════════════════════════════════════
# 9. TRAIN BI-LSTM
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  9. TRAINING Bi-LSTM")
print("="*60)
bilstm_history = bilstm_model.fit(
    X_train_pad, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_pad, y_val),
    class_weight=class_weights,
    callbacks=get_callbacks(f"{MODELS_DIR}/bilstm_spam.keras"),
    verbose=1
)


# ════════════════════════════════════════════════════════════════
# 10. TRAIN PLAIN LSTM
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  10. TRAINING LSTM BASELINE")
print("="*60)
lstm_history = lstm_model.fit(
    X_train_pad, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_pad, y_val),
    class_weight=class_weights,
    callbacks=get_callbacks(f"{MODELS_DIR}/lstm_spam.keras"),
    verbose=1
)


# ════════════════════════════════════════════════════════════════
# 11. EVALUATE ON TEST SET
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  11. TEST SET EVALUATION")
print("="*60)

def evaluate_model(model, X_test_pad, y_test, name):
    y_prob = model.predict(X_test_pad, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print(f"  ROC-AUC : {auc:.4f}")
    return report, auc, y_prob, y_pred

bilstm_report, bilstm_auc, bilstm_prob, bilstm_pred = evaluate_model(
    bilstm_model, X_test_pad, y_test, "Bi-LSTM")
lstm_report, lstm_auc, lstm_prob, lstm_pred = evaluate_model(
    lstm_model, X_test_pad, y_test, "LSTM (baseline)")


# ════════════════════════════════════════════════════════════════
# 12. COMPARISON TABLE
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  12. COMPARISON TABLE")
print("="*60)
comparison = pd.DataFrame({
    "Model":     ["Bi-LSTM", "LSTM (baseline)"],
    "Accuracy":  [bilstm_report["accuracy"],  lstm_report["accuracy"]],
    "Precision": [bilstm_report["Spam"]["precision"],  lstm_report["Spam"]["precision"]],
    "Recall":    [bilstm_report["Spam"]["recall"],     lstm_report["Spam"]["recall"]],
    "F1-Score":  [bilstm_report["Spam"]["f1-score"],   lstm_report["Spam"]["f1-score"]],
    "ROC-AUC":   [bilstm_auc, lstm_auc],
})
comparison = comparison.round(4)
print(comparison.to_string(index=False))


# ════════════════════════════════════════════════════════════════
# 13. PLOTS
# ════════════════════════════════════════════════════════════════
print("\n  Saving plots …")

def plot_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14)

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/{model_name.lower().replace(' ', '_')}_curves.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

plot_history(bilstm_history, "BiLSTM")
plot_history(lstm_history,   "LSTM Baseline")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, title in zip(axes,
                            [bilstm_pred, lstm_pred],
                            ["Bi-LSTM", "LSTM Baseline"]):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrices.png", dpi=150)
plt.close()
print(f"  Saved: {PLOTS_DIR}/confusion_matrices.png")

# ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for prob, name in [(bilstm_prob, f"Bi-LSTM (AUC={bilstm_auc:.3f})"),
                   (lstm_prob,   f"LSTM (AUC={lstm_auc:.3f})")]:
    RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax)
ax.set_title("ROC Curves — Spam Detection")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=150)
plt.close()
print(f"  Saved: {PLOTS_DIR}/roc_curves.png")


# ════════════════════════════════════════════════════════════════
# 14. TEST ON 10 SAMPLE MESSAGES
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  14. SAMPLE PREDICTIONS (Bi-LSTM)")
print("="*60)
sample_messages = [
    # 5 spam
    "WINNER! You have been selected for a FREE prize. Call 09061701461 now to claim!",
    "Congratulations! Your mobile number has won £1000 cash. Text WIN to 80123.",
    "FREE entry in our weekly competition. Text FA to 87121. T&Cs at www.luckydraw.co.uk",
    "URGENT: Your account will be suspended. Verify now at http://secure-login.fake.com",
    "You've been approved for a £5000 loan. No credit checks. Call 08000 123 456.",
    # 5 ham
    "Hey, are you coming to the party tonight? Let me know!",
    "Your package has been delivered. Check the tracking at ups.com.",
    "Can you pick up some milk on your way home please?",
    "Reminder: Doctor's appointment at 3pm tomorrow. See you then.",
    "I'll be there in 10 minutes, just parking now.",
]

cleaned_samples = [clean_text(m) for m in sample_messages]
padded_samples  = texts_to_padded(cleaned_samples)
probs = bilstm_model.predict(padded_samples, verbose=0).flatten()
labels_pred = le.inverse_transform((probs >= 0.5).astype(int))

print(f"\n{'─'*80}")
print(f"{'Message':<55} {'Label':>8} {'Confidence':>12}")
print(f"{'─'*80}")
for msg, lbl, prob in zip(sample_messages, labels_pred, probs):
    spam_conf = prob if lbl == "spam" else 1 - prob
    print(f"{msg[:54]:<55} {lbl.upper():>8} {spam_conf*100:>10.1f}%")
print(f"{'─'*80}")


# ════════════════════════════════════════════════════════════════
# 15. SAVE ARTEFACTS
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  15. SAVING ARTEFACTS")
print("="*60)

# Tokenizer
with open(f"{MODELS_DIR}/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print(f"  Saved: {MODELS_DIR}/tokenizer.pkl")

# Label encoder
with open(f"{MODELS_DIR}/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"  Saved: {MODELS_DIR}/label_encoder.pkl")

# Training config (needed by the API server)
config = {
    "vocab_size": VOCAB_SIZE,
    "max_len": MAX_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "lstm_units": LSTM_UNITS,
    "dropout_rate": DROPOUT_RATE,
    "prediction_threshold": 0.5,
    "label_map": {"0": "ham", "1": "spam"},
    "bilstm_metrics": {
        "accuracy": float(bilstm_report["accuracy"]),
        "spam_precision": float(bilstm_report["Spam"]["precision"]),
        "spam_recall": float(bilstm_report["Spam"]["recall"]),
        "spam_f1": float(bilstm_report["Spam"]["f1-score"]),
        "roc_auc": float(bilstm_auc),
    },
    "lstm_baseline_metrics": {
        "accuracy": float(lstm_report["accuracy"]),
        "spam_precision": float(lstm_report["Spam"]["precision"]),
        "spam_recall": float(lstm_report["Spam"]["recall"]),
        "spam_f1": float(lstm_report["Spam"]["f1-score"]),
        "roc_auc": float(lstm_auc),
    }
}
with open(f"{MODELS_DIR}/training_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved: {MODELS_DIR}/training_config.json")

print("\n" + "="*60)
print("  TRAINING COMPLETE ✓")
print("="*60)
print(f"\n  Bi-LSTM  → {MODELS_DIR}/bilstm_spam.keras")
print(f"  LSTM     → {MODELS_DIR}/lstm_spam.keras")
print(f"  Tokenizer→ {MODELS_DIR}/tokenizer.pkl")
print(f"  Config   → {MODELS_DIR}/training_config.json")
print(f"  Plots    → {PLOTS_DIR}/")
