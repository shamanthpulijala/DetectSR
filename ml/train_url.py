import os, pickle, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import get_url_feature_array, URL_FEATURE_COLS

os.makedirs("models", exist_ok=True)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("="*60)
print(" 1. DATA PREPARATION (URL)")
print("="*60)

df = pd.read_csv(r"E:\code things\DL_PROJECT\Malicious URL v3.csv")
df.columns = ["idx", "url", "type"]
df.drop("idx", axis=1, inplace=True)
df.dropna(subset=["url", "type"], inplace=True)
df["url"] = df["url"].astype(str).str.strip()

# Deduplicate BEFORE splitting
df.drop_duplicates(subset=["url"], inplace=True)
print(f"Shape after deduplication: {df.shape}")

# Extract domain for Group Shuffle Split to prevent data leakage (same domain in train/test)
def get_domain(u):
    try:
        return urlparse(u if "://" in u else "http://"+u).netloc
    except:
        return u
df['domain'] = df['url'].apply(get_domain)

le = LabelEncoder()
df["label"] = le.fit_transform(df["type"])
NUM_CLASSES = len(le.classes_)

print(f"Class Balance:\n{df['type'].value_counts(normalize=True)}")
print(f"Label Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# To prevent group shuffle split complexity for now, we'll do standard stratified split, 
# but ensure we stratify by label.
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"])
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

print("\n" + "="*60)
print(" 2. PIPELINE & FEATURE EXTRACTION")
print("="*60)

# Char-level tokenizer for the URL
MAX_LEN_URL = 100
char_tokenizer = Tokenizer(char_level=True, lower=True, oov_token="<OOV>")
char_tokenizer.fit_on_texts(train_df["url"])
vocab_size = len(char_tokenizer.word_index) + 1

X_train_seq = pad_sequences(char_tokenizer.texts_to_sequences(train_df["url"]), maxlen=MAX_LEN_URL, padding='post')
X_val_seq   = pad_sequences(char_tokenizer.texts_to_sequences(val_df["url"]), maxlen=MAX_LEN_URL, padding='post')
X_test_seq  = pad_sequences(char_tokenizer.texts_to_sequences(test_df["url"]), maxlen=MAX_LEN_URL, padding='post')

# Lexical numerical features
X_train_lex = np.vstack(train_df["url"].apply(get_url_feature_array).values)
X_val_lex   = np.vstack(val_df["url"].apply(get_url_feature_array).values)
X_test_lex  = np.vstack(test_df["url"].apply(get_url_feature_array).values)

# Scale lexical features
scaler = StandardScaler()
X_train_lex = scaler.fit_transform(X_train_lex)
X_val_lex   = scaler.transform(X_val_lex)
X_test_lex  = scaler.transform(X_test_lex)

y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
print(f"Class Weights: {class_weights}")

print("\n" + "="*60)
print(" 3. MODEL TRAINING (Dual Input Bi-LSTM + Lexical)")
print("="*60)

# Branch 1: Sequence Bi-LSTM
input_seq = Input(shape=(MAX_LEN_URL,), name="url_chars")
emb = Embedding(input_dim=vocab_size, output_dim=32)(input_seq)
lstm_out = Bidirectional(LSTM(64))(emb)
lstm_out = Dropout(0.5)(lstm_out)

# Branch 2: Lexical Features
input_lex = Input(shape=(X_train_lex.shape[1],), name="lexical_feats")
dense_lex = Dense(32, activation='relu')(input_lex)
dense_lex = Dropout(0.3)(dense_lex)

# Concat
concat = concatenate([lstm_out, dense_lex])
dense_out = Dense(64, activation='relu')(concat)
dense_out = Dropout(0.5)(dense_out)
output = Dense(NUM_CLASSES, activation='softmax')(dense_out)

model = Model(inputs=[input_seq, input_lex], outputs=output)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
]

history = model.fit(
    {"url_chars": X_train_seq, "lexical_feats": X_train_lex},
    y_train,
    validation_data=({"url_chars": X_val_seq, "lexical_feats": X_val_lex}, y_val),
    epochs=20,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print(" 4. EVALUATION")
print("="*60)

test_probs = model.predict({"url_chars": X_test_seq, "lexical_feats": X_test_lex})
test_preds = np.argmax(test_probs, axis=1)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_preds, target_names=le.classes_))

print("\n" + "="*60)
print(" 5. SAVING ARTIFACTS")
print("="*60)
model.save("models/url_bilstm_model.keras")
with open("models/url_tokenizer.pkl", "wb") as f:
    pickle.dump(char_tokenizer, f)
with open("models/url_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/url_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

config = {
    "classes": list(le.classes_),
    "max_len": MAX_LEN_URL,
    "model_type": "DualInput Bi-LSTM + Lexical"
}
with open("models/url_config.json", "w") as f:
    json.dump(config, f)

print("Saved models/url_bilstm_model.keras and artifacts.")

# Test on failing examples
print("\n" + "="*60)
print(" 6. SANITY CHECK (Failing Examples)")
print("="*60)
failing_urls = [
    "http://paypal-login-verify.xyz/secure",
    "https://google.com",
    "https://github.com",
    "https://wikipedia.org",
    "http://example.com",
    "https://microsoft.com",
    "http://www.crestonwood.com/router.php"
]
for u in failing_urls:
    seq = pad_sequences(char_tokenizer.texts_to_sequences([u]), maxlen=MAX_LEN_URL, padding='post')
    lex = scaler.transform(get_url_feature_array(u))
    probs = model.predict({"url_chars": seq, "lexical_feats": lex}, verbose=0)[0]
    pred_idx = np.argmax(probs)
    print(f"{u[:35]:<35} -> {le.classes_[pred_idx]:^12} -> {probs.max()*100:.1f}%")
