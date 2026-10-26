import re
import numpy as np

# ==========================================
# SPAM PREPROCESSING
# ==========================================
def clean_spam_text(text: str) -> str:
    """Clean text by standardising URLs, numbers, and currency."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL_TOKEN ", text)
    text = re.sub(r"[£$€¥₹]", " CURRENCY ", text)
    text = re.sub(r"\b\d+\b", " NUM ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ==========================================
# URL PREPROCESSING
# ==========================================
# Define features in a fixed list to guarantee ordering!
URL_FEATURE_COLS = [
    "url_length", "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_question", "num_equals", "num_at", "num_ampersand", "num_percent",
    "num_digits", "num_letters", "num_special", "has_ip", "has_https",
    "has_http", "suspicious_words", "domain_length", "has_subdomain",
    "path_length", "ratio_digits", "ratio_letters"
]

def extract_url_features(url: str) -> dict:
    """Extract numerical features from raw URL string."""
    try:
        url = str(url)
        features = {}
        features["url_length"]       = len(url)
        features["num_dots"]         = url.count(".")
        features["num_hyphens"]      = url.count("-")
        features["num_underscores"]  = url.count("_")
        features["num_slashes"]      = url.count("/")
        features["num_question"]     = url.count("?")
        features["num_equals"]       = url.count("=")
        features["num_at"]           = url.count("@")
        features["num_ampersand"]    = url.count("&")
        features["num_percent"]      = url.count("%")
        features["num_digits"]       = sum(c.isdigit() for c in url)
        features["num_letters"]      = sum(c.isalpha() for c in url)
        features["num_special"]      = sum(not c.isalnum() and c not in ".-_/:" for c in url)
        features["has_ip"]           = int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url)))
        features["has_https"]        = int(url.lower().startswith("https"))
        features["has_http"]         = int(url.lower().startswith("http"))
        features["suspicious_words"] = int(bool(re.search(
            r"login|verify|account|update|secure|paypal|bank|free|prize|win|click|confirm|password|credential|phish|hack|malware",
            url.lower()
        )))
        
        try:
            domain = url.split("/")[2] if "://" in url else url.split("/")[0].split("?")[0]
            features["domain_length"] = len(domain)
            features["has_subdomain"] = int(domain.count(".") > 1)
        except:
            domain = ""
            features["domain_length"] = 0
            features["has_subdomain"] = 0
            
        features["path_length"]      = max(0, len(url) - len(domain))
        features["ratio_digits"]     = features["num_digits"] / max(len(url), 1)
        features["ratio_letters"]    = features["num_letters"] / max(len(url), 1)
        
        return features
    except:
        return {k: 0 for k in URL_FEATURE_COLS}

def get_url_feature_array(url: str) -> np.ndarray:
    """Returns a numpy array of features in the EXACT guaranteed order."""
    feats = extract_url_features(url)
    return np.array([[feats.get(k, 0) for k in URL_FEATURE_COLS]], dtype=np.float32)

# ==========================================
# APP PREPROCESSING
# ==========================================
DANGEROUS_PERMS = {
    "android.permission.READ_SMS", "android.permission.SEND_SMS", "android.permission.RECEIVE_SMS",
    "android.permission.READ_CALL_LOG", "android.permission.PROCESS_OUTGOING_CALLS",
    "android.permission.READ_CONTACTS", "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.RECORD_AUDIO", "android.permission.CAMERA", "android.permission.READ_PHONE_STATE",
    "android.permission.WRITE_EXTERNAL_STORAGE", "android.permission.INSTALL_PACKAGES",
    "android.permission.DELETE_PACKAGES", "android.permission.READ_LOGS",
    "android.permission.SYSTEM_ALERT_WINDOW", "android.permission.WRITE_SECURE_SETTINGS",
    "android.permission.AUTHENTICATE_ACCOUNTS", "android.permission.MANAGE_ACCOUNTS"
}

def get_app_feature_array(permissions_dict: dict, feature_cols: list) -> np.ndarray:
    """Creates a feature vector based on exactly the columns the model was trained on."""
    vec = [float(permissions_dict.get(col, 0)) for col in feature_cols]
    return np.array([vec], dtype=np.float32)
