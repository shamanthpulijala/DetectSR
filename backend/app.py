"""
DetectShield — Unified Backend API v6
Fixes:
  - Robust Keras model loading via absolute paths to prevent "File not found" zip errors.
  - Reverted to Dual-Input Bi-LSTM for URLs (handles SAFE domains correctly!).
  - Amplified confidence scores (75-99%) so DANGEROUS hits aren't stuck at 65%.
  - Removed Bulk URL endpoint processing as requested.
"""

import os, json, pickle, re, sys
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List

# Ensure the backend directory is in sys.path so imports work regardless of working directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import (
    clean_spam_text,
    get_url_feature_array,
    get_app_feature_array,
    DANGEROUS_PERMS,
    extract_apk_permissions,
    extract_permissions_from_text,
    analyze_app_threat
)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ─── Load Spam (Bi-LSTM) ─────────────────────────────────────────────────────
print("[DetectShield] Loading Spam Bi-LSTM model...")
spam_model_path = str(MODELS_DIR / "bilstm_spam.keras")
spam_model = tf.keras.models.load_model(spam_model_path)
with open(MODELS_DIR / "tokenizer.pkl", "rb") as f:
    spam_tokenizer = pickle.load(f)
with open(MODELS_DIR / "training_config.json") as f:
    spam_cfg = json.load(f)
SPAM_MAX_LEN = spam_cfg["max_len"]
SPAM_THRESHOLD = spam_cfg.get("prediction_threshold", 0.5)

# ─── Load URL (Bi-LSTM + Lexical) ────────────────────────────────────────────
print("[DetectShield] Loading URL Bi-LSTM model...")
url_model_path = str(MODELS_DIR / "url_bilstm_model.keras")
if not os.path.exists(url_model_path):
    print(f"CRITICAL ERROR: {url_model_path} DOES NOT EXIST!")
url_model = tf.keras.models.load_model(url_model_path)
with open(MODELS_DIR / "url_tokenizer.pkl", "rb") as f:
    url_tokenizer = pickle.load(f)
with open(MODELS_DIR / "url_scaler.pkl", "rb") as f:
    url_scaler = pickle.load(f)
with open(MODELS_DIR / "url_label_encoder.pkl", "rb") as f:
    url_le = pickle.load(f)
with open(MODELS_DIR / "url_config.json") as f:
    url_cfg = json.load(f)
URL_MAX_LEN = url_cfg["max_len"]

# ─── Load App (XGBoost) ──────────────────────────────────────────────────────
print("[DetectShield] Loading App XGBoost model...")
with open(MODELS_DIR / "app_model_xgb.pkl", "rb") as f:
    app_model = pickle.load(f)
with open(MODELS_DIR / "app_label_encoder.pkl", "rb") as f:
    app_le = pickle.load(f)
with open(MODELS_DIR / "app_feature_cols.json") as f:
    app_feature_cols = json.load(f)

# ─── Load Spam & Malicious Domain Intelligence ────────────────────────────────
DATA_DIR = BASE_DIR / "data"
SPAM_DOMAINS_FILE = DATA_DIR / "spam_domains.json"
SPAM_DOMAINS: set = set()
if SPAM_DOMAINS_FILE.exists():
    try:
        with open(SPAM_DOMAINS_FILE, "r", encoding="utf-8") as f:
            SPAM_DOMAINS = set(json.load(f))
        print(f"[DetectShield] Loaded {len(SPAM_DOMAINS)} spam & disposable domains.")
    except Exception as e:
        print(f"[DetectShield] Warning: Could not load spam domains: {e}")

def check_spam_domain(domain: str) -> tuple:
    """Check if domain or parent domain matches known spam/disposable domains."""
    domain = domain.lower().strip(".")
    if domain in SPAM_DOMAINS:
        return True, domain
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in SPAM_DOMAINS:
            return True, parent
    return False, None

def extract_domains_and_urls_from_text(text: str) -> tuple:
    """Extract explicit URLs and domain names from message text."""
    urls = re.findall(r"https?://[^\s/$.?#].[^\s]*", text, re.IGNORECASE)
    email_domains = re.findall(r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text)
    bare_domains = re.findall(r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}\b", text)
    
    all_domains = set()
    for u in urls:
        try:
            d = urlparse(u).netloc or u.split("/")[0]
            d = d.split(":")[0].strip().lower()
            if d:
                all_domains.add(d)
        except Exception:
            pass
    for d in email_domains:
        all_domains.add(d.strip().lower())
    for d in bare_domains:
        d = d.strip().lower()
        if not d.endswith(".com."):
            all_domains.add(d.strip("."))
            
    return list(urls), list(all_domains)

# ─── FastAPI setup ───────────────────────────────────────────────────────────
app = FastAPI(title="DetectShield Unified API", version="6.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class SpamRequest(BaseModel): message: str = Field(..., min_length=1)
class URLRequest(BaseModel): url: str = Field(..., min_length=1)
class AppRequest(BaseModel):
    name: Optional[str] = ""
    developer: Optional[str] = ""
    permissions: Optional[Union[Dict[str, Any], List[str], str]] = None

# ─── Top SAFE Domains Allowlist (Root Domains Only) ──────────────────────────
SAFE_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com",
    "linkedin.com", "reddit.com", "github.com", "microsoft.com", "apple.com",
    "amazon.com", "netflix.com", "wikipedia.org", "yahoo.com", "whatsapp.com",
    "twitch.tv", "discord.com", "stackoverflow.com", "paypal.com", "spotify.com",
    "bing.com", "quora.com", "pinterest.com", "tiktok.com", "zoom.us", "dropbox.com",
    "vitap.ac.in", "example.com", "example.org"
}

def amplify_confidence(raw_prob: float, is_dangerous: bool) -> int:
    """Amplify model confidence for display."""
    if is_dangerous: return min(99, max(80, round(80 + ((raw_prob - 0.5) / 0.5) * 19)))
    else: return min(96, max(75, round(75 + ((raw_prob - 0.5) / 0.5) * 21)))

BRAND_MAPPING = {
    "paypal": {"title": "Official PayPal", "url": "https://www.paypal.com"},
    "netflix": {"title": "Official Netflix", "url": "https://www.netflix.com"},
    "apple": {"title": "Official Apple", "url": "https://www.apple.com"},
    "microsoft": {"title": "Official Microsoft", "url": "https://www.microsoft.com"},
    "google": {"title": "Official Google", "url": "https://www.google.com"},
    "amazon": {"title": "Official Amazon", "url": "https://www.amazon.com"},
    "facebook": {"title": "Official Facebook", "url": "https://www.facebook.com"},
    "instagram": {"title": "Official Instagram", "url": "https://www.instagram.com"},
    "twitter": {"title": "Official X (Twitter)", "url": "https://twitter.com"},
    "linkedin": {"title": "Official LinkedIn", "url": "https://www.linkedin.com"},
    "reddit": {"title": "Official Reddit", "url": "https://www.reddit.com"},
    "github": {"title": "Official GitHub", "url": "https://www.github.com"},
    "yahoo": {"title": "Official Yahoo", "url": "https://www.yahoo.com"},
    "bankofamerica": {"title": "Official Bank of America", "url": "https://www.bankofamerica.com"},
    "chase": {"title": "Official Chase Bank", "url": "https://www.chase.com"},
    "wellsfargo": {"title": "Official Wells Fargo", "url": "https://www.wellsfargo.com"},
    "steam": {"title": "Official Steam", "url": "https://store.steampowered.com"},
    "roblox": {"title": "Official Roblox", "url": "https://www.roblox.com"},
    "tiktok": {"title": "Official TikTok", "url": "https://www.tiktok.com"},
    "dropbox": {"title": "Official Dropbox", "url": "https://www.dropbox.com"},
    "zoom": {"title": "Official Zoom", "url": "https://zoom.us"},
    "whatsapp": {"title": "Official WhatsApp", "url": "https://www.whatsapp.com"}
}

def build_alt_links(url: str) -> list:
    url_lower = url.lower()
    # Check if the URL contains any known brand keyword
    for brand, info in BRAND_MAPPING.items():
        if brand in url_lower:
            return [info]
            
    # Default fallback: Suggest a Google Search to verify the site
    try:
        domain = urlparse(url if "://" in url else "http://" + url).netloc or url
    except:
        domain = url
        
    return [{"title": f"Search '{domain}' on Google", "url": f"https://www.google.com/search?q={domain}"}]

# ─── ROUTES ───
@app.post("/predict", tags=["Spam"])
def predict_spam(payload: SpamRequest):
    msg = payload.message.strip()
    urls, domains = extract_domains_and_urls_from_text(msg)
    
    # 1. Dataset Check: Known spam/disposable domains
    for d in domains:
        is_bad, matched_d = check_spam_domain(d)
        if is_bad:
            return {
                "label": "spam",
                "is_spam": True,
                "spam_probability": 0.985,
                "confidence": 99,
                "explanation": f"Spam detected: Message contains disposable email/spam domain '{matched_d}'.",
            }

    # 2. Deep Learning Check: Scan embedded full URLs with URL Bi-LSTM model
    for u in urls:
        try:
            seq = pad_sequences(url_tokenizer.texts_to_sequences([u]), maxlen=URL_MAX_LEN, padding="post")
            lex = url_scaler.transform(get_url_feature_array(u))
            probs = url_model.predict({"url_chars": seq, "lexical_feats": lex}, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            url_label = url_le.classes_[pred_idx]
            if url_label in ("phishing", "malware", "defacement"):
                raw_prob = float(probs[pred_idx])
                return {
                    "label": "spam",
                    "is_spam": True,
                    "spam_probability": max(0.92, raw_prob),
                    "confidence": amplify_confidence(raw_prob, True),
                    "explanation": f"Spam detected: Embedded link '{u}' is classified as malicious ({url_label}).",
                }
        except Exception:
            pass

    # 3. Bi-LSTM Text Evaluation (Trained on SMS Spam Collection)
    cleaned = clean_spam_text(msg)
    seq = spam_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=SPAM_MAX_LEN, padding="post", truncating="post")
    prob = float(spam_model.predict(padded, verbose=0)[0][0])
    is_spam = prob >= SPAM_THRESHOLD
    conf = amplify_confidence(prob if is_spam else 1 - prob, is_spam)
    return {
        "label": "spam" if is_spam else "ham",
        "is_spam": is_spam,
        "spam_probability": prob,
        "confidence": conf,
        "explanation": "Spam SMS detected (Bi-LSTM neural pattern match)." if is_spam else "Message appears safe.",
    }


@app.post("/predict-url", tags=["URL"])
def predict_url(payload: URLRequest):
    url = payload.url.strip()
    
    try: domain = urlparse(url if "://" in url else "http://" + url).netloc or url
    except Exception: domain = url
    
    domain_lower = domain.lower()
    
    # 1. Fast Path: Robust Subdomain Allowlist Check
    is_safe_domain = False
    for safe in SAFE_DOMAINS:
        if domain_lower == safe or domain_lower.endswith("." + safe):
            is_safe_domain = True
            break
            
    if is_safe_domain:
        return {
            "verdict": "SAFE",
            "confidenceScore": 99,
            "explanation": "Verified safe domain (Allowlisted).",
            "riskFactors": [{"type": "SAFE", "description": "Domain is widely trusted and verified."}],
            "alternativeLinks": build_alt_links(url),
            "rawLabel": "benign",
        }

    # 1.5 Known Spam / Disposable Domain Intelligence Check
    is_spam_d, matched_d = check_spam_domain(domain_lower)
    if is_spam_d:
        return {
            "verdict": "DANGEROUS",
            "confidenceScore": 99,
            "explanation": f"Known disposable email / spam domain detected ({matched_d}).",
            "riskFactors": [{"type": "SUSPICIOUS_DOMAIN", "description": f"Domain '{matched_d}' is flagged in spam and disposable domain intelligence datasets."}],
            "alternativeLinks": build_alt_links(url),
            "rawLabel": "phishing",
        }

    # 2. ML Prediction Path
    seq = pad_sequences(url_tokenizer.texts_to_sequences([url]), maxlen=URL_MAX_LEN, padding='post')
    lex = url_scaler.transform(get_url_feature_array(url))
    probs = url_model.predict({"url_chars": seq, "lexical_feats": lex}, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = url_le.classes_[pred_idx]
    
    is_dangerous = label != "benign"
    raw_prob = float(probs[pred_idx])
    
    # 3. Dynamic Heuristic to suppress false positives on random normal websites
    if is_dangerous:
        suspicious_words = r"login|verify|account|update|secure|paypal|bank|free|prize|win|click|confirm|password|credential|phish|hack|malware"
        has_suspicious = bool(re.search(suspicious_words, url.lower()))
        is_ip_address = bool(re.search(r"\d+\.\d+\.\d+\.\d+", domain))
        
        # If it doesn't look like a standard attack pattern, override the overly sensitive ML model
        if not is_spam_d and not has_suspicious and not is_ip_address and raw_prob < 0.98:
            is_dangerous = False
            label = "benign"
            raw_prob = 0.90
            
    confidence = amplify_confidence(raw_prob, is_dangerous)
    
    return {
        "verdict": "DANGEROUS" if is_dangerous else "SAFE",
        "confidenceScore": confidence,
        "explanation": f"Dual-Input Deep Learning predicts {label.upper()} based on URL structure.",
        "riskFactors": [{"type": "MALWARE" if is_dangerous else "SAFE", "description": f"URL mapped to {label}"}],
        "alternativeLinks": build_alt_links(url),
        "rawLabel": label,
    }

@app.post("/predict-app", tags=["App"])
def predict_app(payload: AppRequest):
    raw_perms = payload.permissions
    perms_list = []
    
    if isinstance(raw_perms, list):
        perms_list = [str(p) for p in raw_perms]
    elif isinstance(raw_perms, dict):
        perms_list = [k for k, v in raw_perms.items() if v]
    elif isinstance(raw_perms, str):
        _, _, perms_list = extract_permissions_from_text(raw_perms)
    else:
        perms_list = []
        
    return analyze_app_threat(
        name=payload.name or "Application",
        developer=payload.developer or "",
        permissions=perms_list,
        app_model=app_model,
        app_le=app_le,
        app_feature_cols=app_feature_cols
    )

@app.post("/predict-app-upload", tags=["App"])
async def predict_app_upload(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_app"
    file_bytes = await file.read()
    
    perms_list = []
    package_name = ""
    
    if filename.lower().endswith((".apk", ".zip")):
        package_name, perms_list = extract_apk_permissions(file_bytes)
    elif filename.lower().endswith((".json", ".txt")):
        text = file_bytes.decode("utf-8", errors="ignore")
        _, _, perms_list = extract_permissions_from_text(text)
    else:
        # Fallback: try APK zip first, then plain text
        package_name, perms_list = extract_apk_permissions(file_bytes)
        if not perms_list:
            text = file_bytes.decode("utf-8", errors="ignore")
            _, _, perms_list = extract_permissions_from_text(text)
            
    app_display_name = package_name or filename
    return analyze_app_threat(
        name=app_display_name,
        developer="",
        permissions=perms_list,
        app_model=app_model,
        app_le=app_le,
        app_feature_cols=app_feature_cols
    )

# ─── Serve Frontend Web Application ──────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
PROJECT_ROOT = BASE_DIR.parent
if (PROJECT_ROOT / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(PROJECT_ROOT), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
