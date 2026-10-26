import os, pickle, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import clean_spam_text

os.makedirs("models", exist_ok=True)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("="*60)
print(" 1. DATA PREPARATION (Spam)")
print("="*60)

# Load data
df = pd.read_csv(r"E:\code things\DL_PROJECT\Spam_SMS.csv")
df.rename(columns={'Class': 'label', 'Message': 'text'}, inplace=True)
print(f"Original shape: {df.shape}")

# Deduplicate BEFORE splitting
df.drop_duplicates(subset=['text'], inplace=True)
print(f"Shape after deduplication: {df.shape}")

# Clean text
df['clean_text'] = df['text'].apply(clean_spam_text)
# Drop empty
df = df[df['clean_text'].str.len() > 0]

# Labels (0=ham, 1=spam)
df['target'] = (df['label'] == 'spam').astype(int)
print(f"Class Balance:\n{df['target'].value_counts(normalize=True)}")

# Stratified split: 70/15/15
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['clean_text'], df['target'], test_size=0.30, random_state=SEED, stratify=df['target']
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
)
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

print("\n" + "="*60)
print(" 2. TOKENIZATION & PIPELINE")
print("="*60)
MAX_WORDS = 10000
MAX_LEN = 150

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding='post', truncating='post')
X_val   = pad_sequences(tokenizer.texts_to_sequences(val_texts),   maxlen=MAX_LEN, padding='post', truncating='post')
X_test  = pad_sequences(tokenizer.texts_to_sequences(test_texts),  maxlen=MAX_LEN, padding='post', truncating='post')

y_train = train_labels.values
y_val   = val_labels.values
y_test  = test_labels.values

# Compute class weights for imbalance
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
print(f"Class Weights: {class_weights}")

print("\n" + "="*60)
print(" 3. MODEL TRAINING (Bi-LSTM)")
print("="*60)

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR')
    ]
)

callbacks = [
    EarlyStopping(monitor='val_prc', mode='max', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/bilstm_spam_best.keras", monitor='val_prc', mode='max', save_best_only=True, verbose=0)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print(" 4. THRESHOLD TUNING (Validation Set)")
print("="*60)
val_probs = model.predict(X_val).flatten()

best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.1, 0.9, 0.05):
    preds = (val_probs >= thresh).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Best Threshold based on Val F1: {best_thresh:.3f} (F1 = {best_f1:.3f})")

print("\n" + "="*60)
print(" 5. FINAL EVALUATION (Test Set)")
print("="*60)
test_probs = model.predict(X_test).flatten()
test_preds = (test_probs >= best_thresh).astype(int)

print(f"Accuracy  : {accuracy_score(y_test, test_preds):.4f}")
print(f"Precision : {precision_score(y_test, test_preds):.4f}")
print(f"Recall    : {recall_score(y_test, test_preds):.4f}")
print(f"F1-Score  : {f1_score(y_test, test_preds):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, test_probs):.4f}")
print(f"PR-AUC    : {average_precision_score(y_test, test_probs):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, test_preds)}")

print("\n" + "="*60)
print(" 6. SAVING ARTIFACTS")
print("="*60)
# Overwrite the generic one with the best one
model.save("models/bilstm_spam.keras")
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

config = {
    "max_len": MAX_LEN,
    "max_words": MAX_WORDS,
    "prediction_threshold": float(best_thresh)
}
with open("models/training_config.json", "w") as f:
    json.dump(config, f)

print("Saved models/bilstm_spam.keras and models/tokenizer.pkl")
