"""
DetectShield — Spam Message Classifier
Google Colab Notebook (run cell-by-cell)

Upload your Spam_SMS.csv to the Colab session before running.
After training, download:
  - bilstm_spam.keras
  - tokenizer.pkl
  - training_config.json
"""

# ─────────────────────────── CELL 1: Install ───────────────────────────
"""
!pip install -q tensorflow scikit-learn matplotlib pandas
"""

# ─────────────────────────── CELL 2: Imports ───────────────────────────
import os, json, pickle, re, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# ─────────────────────────── CELL 3: Config ────────────────────────────
DATASET_PATH  = "Spam_SMS.csv"  # upload this to Colab first
VOCAB_SIZE    = 10_000
MAX_LEN       = 150
EMBEDDING_DIM = 64
LSTM_UNITS    = 64
DROPOUT_RATE  = 0.4
EPOCHS        = 30
BATCH_SIZE    = 64
RANDOM_SEED   = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.makedirs("models", exist_ok=True)

# ─────────────────────────── CELL 4: Load & Explore ────────────────────
df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", usecols=["Class", "Message"])
df.columns = ["label", "message"]
df.dropna(inplace=True)

print("Shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())
print(f"\nClass balance: {df['label'].value_counts(normalize=True).round(3).to_dict()}")

# Message length distribution
df["msg_len"] = df["message"].str.len()
print(f"\nMessage length — mean: {df.msg_len.mean():.1f}, "
      f"95th-pct: {df.msg_len.quantile(0.95):.0f}, max: {df.msg_len.max()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df.groupby("label")["msg_len"].hist(bins=40, alpha=0.7, ax=axes[0])
axes[0].set_title("Message Length by Label"); axes[0].legend(["ham", "spam"])
df["label"].value_counts().plot(kind="bar", ax=axes[1], color=["steelblue", "tomato"])
axes[1].set_title("Class Distribution"); axes[1].set_xticklabels(["Ham", "Spam"], rotation=0)
plt.tight_layout(); plt.show()

# ─────────────────────────── CELL 5: Clean Text ────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL_TOKEN ", text)
    text = re.sub(r"[£$€¥₹]", " CURRENCY ", text)
    text = re.sub(r"\b\d+\b", " NUM ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df["message"].apply(clean_text)
print("Sample cleaned messages:")
df[["message", "clean"]].head(3)

# ─────────────────────────── CELL 6: Encode & Split ────────────────────
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])  # ham=0, spam=1
print("Encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

X, y = df["clean"].values, df["label_enc"].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15/0.85, stratify=y_trainval, random_state=RANDOM_SEED)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ─────────────────────────── CELL 7: Tokenise ──────────────────────────
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", lower=True)
tokenizer.fit_on_texts(X_train)
print(f"Vocabulary size: {len(tokenizer.word_index)}")

def to_padded(texts):
    return pad_sequences(tokenizer.texts_to_sequences(texts),
                         maxlen=MAX_LEN, padding="post", truncating="post")

X_train_p = to_padded(X_train)
X_val_p   = to_padded(X_val)
X_test_p  = to_padded(X_test)
print(f"X_train_p shape: {X_train_p.shape}")

# ─────────────────────────── CELL 8: Class Weights ─────────────────────
n_ham, n_spam = np.sum(y_train==0), np.sum(y_train==1)
total = n_ham + n_spam
class_weights = {0: total/(2*n_ham), 1: total/(2*n_spam)}
print(f"Class weights: {class_weights}")

# ─────────────────────────── CELL 9: Bi-LSTM Model ─────────────────────
def build_bilstm():
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inp)
    x   = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=0.1))(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.Dropout(DROPOUT_RATE * 0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out, name="BiLSTM_Spam")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy",
                       keras.metrics.Precision(name="precision"),
                       keras.metrics.Recall(name="recall"),
                       keras.metrics.AUC(name="auc")])
    return m

bilstm = build_bilstm()
bilstm.summary()

# ─────────────────────────── CELL 10: Train Bi-LSTM ────────────────────
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=5,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/bilstm_spam.keras", monitor="val_auc",
                    mode="max", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                      min_lr=1e-6, verbose=1)
]

h_bilstm = bilstm.fit(X_train_p, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                       validation_data=(X_val_p, y_val),
                       class_weight=class_weights, callbacks=callbacks)

# ─────────────────────────── CELL 11: LSTM Baseline ────────────────────
def build_lstm():
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inp)
    x   = layers.LSTM(LSTM_UNITS, recurrent_dropout=0.1)(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.Dropout(DROPOUT_RATE * 0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out, name="LSTM_Spam")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy",
                       keras.metrics.Precision(name="precision"),
                       keras.metrics.Recall(name="recall"),
                       keras.metrics.AUC(name="auc")])
    return m

lstm = build_lstm()
callbacks_lstm = [
    EarlyStopping(monitor="val_auc", mode="max", patience=5,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/lstm_spam.keras", monitor="val_auc",
                    mode="max", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                      min_lr=1e-6, verbose=1)
]
h_lstm = lstm.fit(X_train_p, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_val_p, y_val),
                  class_weight=class_weights, callbacks=callbacks_lstm)

# ─────────────────────────── CELL 12: Evaluate ─────────────────────────
def eval_model(model, name):
    y_prob = model.predict(X_test_p, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    return y_prob, y_pred, auc

bi_prob, bi_pred, bi_auc = eval_model(bilstm, "Bi-LSTM")
ls_prob, ls_pred, ls_auc = eval_model(lstm,   "LSTM Baseline")

# ─────────────────────────── CELL 13: Comparison Table ─────────────────
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1-Score":  round(f1_score(y_true, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
    }

comp = pd.DataFrame([
    {"Model": "Bi-LSTM",          **metrics(y_test, bi_pred, bi_prob)},
    {"Model": "LSTM (baseline)",  **metrics(y_test, ls_pred, ls_prob)},
])
print("\nCOMPARISON TABLE:")
print(comp.to_string(index=False))

# ─────────────────────────── CELL 14: Plots ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (hist, name) in enumerate([(h_bilstm, "Bi-LSTM"), (h_lstm, "LSTM")]):
    axes[0][i].plot(hist.history["loss"],     label="Train"); axes[0][i].plot(hist.history["val_loss"], label="Val")
    axes[0][i].set_title(f"{name} — Loss"); axes[0][i].legend()
    axes[1][i].plot(hist.history["accuracy"], label="Train"); axes[1][i].plot(hist.history["val_accuracy"], label="Val")
    axes[1][i].set_title(f"{name} — Accuracy"); axes[1][i].legend()

plt.tight_layout(); plt.show()

fig2, ax2 = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, bi_prob, name=f"Bi-LSTM (AUC={bi_auc:.3f})", ax=ax2)
RocCurveDisplay.from_predictions(y_test, ls_prob, name=f"LSTM (AUC={ls_auc:.3f})",    ax=ax2)
ax2.set_title("ROC Curves"); plt.show()

# ─────────────────────────── CELL 15: Sample Predictions ───────────────
sample_msgs = [
    "WINNER! You have been selected for a FREE prize. Call 09061701461 now to claim!",
    "Congratulations! Your mobile number has won 1000 cash. Text WIN to 80123.",
    "FREE entry in our weekly competition. Text FA to 87121.",
    "URGENT: Your account will be suspended. Verify now at http://secure-login.fake.com",
    "You've been approved for a loan. No credit checks. Call now.",
    "Hey, are you coming to the party tonight? Let me know!",
    "Your package has been delivered. Check the tracking at ups.com.",
    "Can you pick up some milk on your way home please?",
    "Reminder: Doctor's appointment at 3pm tomorrow.",
    "I'll be there in 10 minutes, just parking now.",
]

cleaned  = [clean_text(m) for m in sample_msgs]
padded   = to_padded(cleaned)
probs    = bilstm.predict(padded, verbose=0).flatten()
lbls     = le.inverse_transform((probs >= 0.5).astype(int))

print(f"\n{'─'*80}")
print(f"{'Message':<55} {'Label':>8} {'Confidence':>12}")
print(f"{'─'*80}")
for msg, lbl, p in zip(sample_msgs, lbls, probs):
    conf = p if lbl == "spam" else 1 - p
    print(f"{msg[:54]:<55} {lbl.upper():>8} {conf*100:>10.1f}%")

# ─────────────────────────── CELL 16: Save & Download ──────────────────
with open("models/tokenizer.pkl", "wb") as f: pickle.dump(tokenizer, f)

config_data = {
    "vocab_size": VOCAB_SIZE, "max_len": MAX_LEN,
    "embedding_dim": EMBEDDING_DIM, "lstm_units": LSTM_UNITS,
    "dropout_rate": DROPOUT_RATE, "prediction_threshold": 0.5,
    "label_map": {"0": "ham", "1": "spam"},
}
with open("models/training_config.json", "w") as f: json.dump(config_data, f, indent=2)

print("Saved: models/bilstm_spam.keras, models/tokenizer.pkl, models/training_config.json")

# Download from Colab (uncomment these lines if running in Colab):
# from google.colab import files
# files.download("models/bilstm_spam.keras")
# files.download("models/tokenizer.pkl")
# files.download("models/training_config.json")
