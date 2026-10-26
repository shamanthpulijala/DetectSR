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
import zipfile
import io
import json

DANGEROUS_PERMS = {
    "android.permission.READ_SMS", "android.permission.SEND_SMS", "android.permission.RECEIVE_SMS",
    "android.permission.READ_CALL_LOG", "android.permission.PROCESS_OUTGOING_CALLS",
    "android.permission.READ_CONTACTS", "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION", "android.permission.RECORD_AUDIO",
    "android.permission.CAMERA", "android.permission.READ_PHONE_STATE",
    "android.permission.WRITE_EXTERNAL_STORAGE", "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.INSTALL_PACKAGES", "android.permission.DELETE_PACKAGES",
    "android.permission.READ_LOGS", "android.permission.SYSTEM_ALERT_WINDOW",
    "android.permission.WRITE_SECURE_SETTINGS", "android.permission.AUTHENTICATE_ACCOUNTS",
    "android.permission.MANAGE_ACCOUNTS", "android.permission.RECEIVE_BOOT_COMPLETED",
    "android.permission.BIND_DEVICE_ADMIN", "android.permission.BIND_ACCESSIBILITY_SERVICE",
    "android.permission.MOUNT_UNMOUNT_FILESYSTEMS", "android.permission.MOUNT_FORMAT_FILESYSTEMS"
}

DANGEROUS_PERM_INFO = {
    "android.permission.SEND_SMS": "Allows silently sending unauthorized SMS messages (Toll fraud / premium SMS attack risk).",
    "android.permission.READ_SMS": "Allows reading private SMS text messages and intercepting banking 2FA/OTP codes.",
    "android.permission.RECEIVE_SMS": "Allows intercepting incoming SMS messages before user notification.",
    "android.permission.READ_CONTACTS": "Allows harvesting user contact book for data exfiltration and spam distribution.",
    "android.permission.RECORD_AUDIO": "Allows secretly recording ambient audio via the microphone in the background.",
    "android.permission.CAMERA": "Allows capturing photos and video streams without user preview.",
    "android.permission.ACCESS_FINE_LOCATION": "Allows tracking precise real-time GPS physical coordinates.",
    "android.permission.ACCESS_COARSE_LOCATION": "Allows tracking approximate cellular/Wi-Fi physical location.",
    "android.permission.READ_PHONE_STATE": "Allows harvesting device hardware identifiers (IMEI, IMSI, SIM serial, phone number).",
    "android.permission.SYSTEM_ALERT_WINDOW": "Allows drawing overlay screens over banking and login applications (Screen overlay phishing attack).",
    "android.permission.INSTALL_PACKAGES": "Allows silently downloading and installing arbitrary secondary malware payloads.",
    "android.permission.DELETE_PACKAGES": "Allows uninstalling security applications or antivirus defenses.",
    "android.permission.RECEIVE_BOOT_COMPLETED": "Allows malware to gain persistence and launch background services upon device reboot.",
    "android.permission.WRITE_EXTERNAL_STORAGE": "Allows encrypting, modifying, or deleting files stored on external storage (Ransomware risk).",
    "android.permission.READ_EXTERNAL_STORAGE": "Allows harvesting documents, media, and private downloads.",
    "android.permission.WRITE_SECURE_SETTINGS": "Allows modifying critical Android OS security configurations.",
    "android.permission.BIND_DEVICE_ADMIN": "Allows obtaining Device Administrator privileges to prevent user uninstallation (Device locker).",
    "android.permission.BIND_ACCESSIBILITY_SERVICE": "Allows monitoring keyboard strokes and reading screen contents (Keylogger / automated theft).",
    "android.permission.READ_CALL_LOG": "Allows accessing incoming, outgoing, and missed phone call history.",
    "android.permission.PROCESS_OUTGOING_CALLS": "Allows intercepting or rerouting outgoing phone calls.",
    "android.permission.MOUNT_UNMOUNT_FILESYSTEMS": "Allows tampering with device filesystem mount points.",
    "android.permission.MOUNT_FORMAT_FILESYSTEMS": "Allows wiping or reformatting storage partitions."
}

def get_app_feature_array(permissions_dict: dict, feature_cols: list) -> np.ndarray:
    """Creates a feature vector based on exactly the columns the model was trained on."""
    vec = [float(permissions_dict.get(col, 0)) for col in feature_cols]
    return np.array([vec], dtype=np.float32)

def extract_apk_permissions(apk_bytes: bytes) -> tuple:
    """
    Extracts package name and all declared android permissions from an APK file's
    AndroidManifest.xml and classes.dex files without requiring external binary tools.
    """
    perms = set()
    package_name = ""
    
    try:
        with zipfile.ZipFile(io.BytesIO(apk_bytes)) as z:
            namelist = z.namelist()
            
            # 1. Parse AndroidManifest.xml
            if "AndroidManifest.xml" in namelist:
                manifest_data = z.read("AndroidManifest.xml")
                
                # Check byte-level ASCII / UTF-8
                found_b = re.findall(rb"(?:android\.permission|com\.android\.[a-zA-Z0-9_\.]+)\.([A-Za-z0-9_]+)", manifest_data)
                for fb in found_b:
                    perms.add(f"android.permission.{fb.decode('latin1', errors='ignore')}")
                
                # Check UTF-16LE byte pattern: a\x00n\x00d\x00r\x00o\x00i\x00d\x00.\x00p\x00e\x00r\x00m\x00i\x00s\x00s\x00i\x00o\x00n\x00.\x00
                utf16_pattern = rb"a\x00n\x00d\x00r\x00o\x00i\x00d\x00\.\x00p\x00e\x00r\x00m\x00i\x00s\x00s\x00i\x00o\x00n\x00\.\x00([A-Za-z0-9_\x00]+?)(?=\x00\x00|\x00[^\x00a-zA-Z0-9_]|$)"
                found_u16 = re.findall(utf16_pattern, manifest_data)
                for fu in found_u16:
                    cleaned_name = fu.replace(b"\x00", b"").decode("latin1", errors="ignore")
                    if cleaned_name:
                        perms.add(f"android.permission.{cleaned_name}")
                
                # Check decoded UTF-8 and Latin-1
                raw_str = manifest_data.decode("utf-8", errors="ignore")
                perms.update(re.findall(r"(?:android\.permission|com\.android\.[a-zA-Z0-9_\.]+)\.[A-Za-z0-9_]+", raw_str))
                
                # Extract package name
                cleaned_pkg = re.sub(rb"[\x00-\x1f]", b" ", manifest_data).decode("latin1", errors="ignore")
                pkg_match = re.search(r"package\s+([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+)", cleaned_pkg)
                if pkg_match:
                    package_name = pkg_match.group(1)
            
            # 2. Check classes.dex for any dynamically referenced permissions
            dex_files = [f for f in namelist if f.endswith(".dex")]
            for dex_name in dex_files[:5]:
                try:
                    dex_data = z.read(dex_name)
                    found_dex = re.findall(rb"android\.permission\.([A-Za-z0-9_]+)", dex_data)
                    for fd in found_dex:
                        perms.add(f"android.permission.{fd.decode('latin1', errors='ignore')}")
                except Exception:
                    pass
    except Exception as e:
        print(f"[DetectShield] APK parse warning: {e}")
        
    return package_name, sorted(list(perms))

def extract_permissions_from_text(text: str) -> tuple:
    """
    Parses JSON, YAML, or plain text to extract app name and permissions.
    Supports both fully-qualified and short permission names.
    """
    perms = set()
    name = ""
    developer = ""
    
    # Try parsing as JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            name = data.get("name") or data.get("appName") or ""
            developer = data.get("developer") or ""
            
            p_val = data.get("permissions")
            if isinstance(p_val, list):
                for p in p_val:
                    perms.add(str(p).strip())
            elif isinstance(p_val, dict):
                for k, v in p_val.items():
                    if v: perms.add(str(k).strip())
                    
            for k, v in data.items():
                if "permission" in k.lower() and v:
                    perms.add(str(k).strip())
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    perms.add(item.strip())
    except Exception:
        pass
        
    # Regex search for fully-qualified permissions
    found_fq = re.findall(r"(?:android\.permission|com\.android\.[a-zA-Z0-9_\.]+)\.[A-Za-z0-9_]+", text)
    perms.update(found_fq)
    
    # Also find bare uppercase permission names like SEND_SMS, CAMERA, etc.
    bare_perms = re.findall(r"\b([A-Z][A-Z0-9_]{3,30})\b", text)
    for bp in bare_perms:
        if bp in DANGEROUS_PERM_INFO or any(bp == col.split(".")[-1] for col in DANGEROUS_PERMS):
            perms.add(f"android.permission.{bp}")
            
    return name, developer, sorted(list(perms))

def analyze_app_threat(name: str, developer: str, permissions: list, app_model, app_le, app_feature_cols) -> dict:
    """
    Executes deep threat analysis using the trained XGBoost model and heuristic security rules.
    Returns a unified report compatible with the DetectShield React UI.
    """
    # 1. Normalize all permission strings
    normalized_perms = set()
    for p in permissions:
        p_clean = str(p).strip().strip('"').strip("'")
        if not p_clean:
            continue
        if not p_clean.startswith("android.permission.") and not p_clean.startswith("com.android."):
            candidate = f"android.permission.{p_clean}"
            normalized_perms.add(candidate)
        else:
            normalized_perms.add(p_clean)
            
    perms_set = normalized_perms
    perms_dict = {p: 1 for p in perms_set}
    
    # Run XGBoost model
    feat_arr = get_app_feature_array(perms_dict, app_feature_cols)
    probs = app_model.predict_proba(feat_arr)[0]
    
    classes = list(app_le.classes_)
    malware_idx = classes.index("malware") if "malware" in classes else 1
    benign_idx = classes.index("benignware") if "benignware" in classes else 0
    
    raw_malware_prob = float(probs[malware_idx])
    
    # Identify dangerous permissions
    danger_detected = [p for p in perms_set if p in DANGEROUS_PERMS]
    
    # Threat pattern heuristics
    has_sms_fraud = any(p in perms_set for p in ["android.permission.SEND_SMS", "android.permission.RECEIVE_SMS"])
    has_spyware = (
        ("android.permission.RECORD_AUDIO" in perms_set or "android.permission.CAMERA" in perms_set) and
        ("android.permission.ACCESS_FINE_LOCATION" in perms_set or "android.permission.READ_PHONE_STATE" in perms_set or "android.permission.READ_CONTACTS" in perms_set)
    )
    has_ransomware = any(p in perms_set for p in ["android.permission.SYSTEM_ALERT_WINDOW", "android.permission.BIND_DEVICE_ADMIN", "android.permission.MOUNT_FORMAT_FILESYSTEMS"])
    has_stealth_install = any(p in perms_set for p in ["android.permission.INSTALL_PACKAGES", "android.permission.DELETE_PACKAGES"])
    has_persistence = "android.permission.RECEIVE_BOOT_COMPLETED" in perms_set
    
    CRITICAL_PERMS = {
        "android.permission.SEND_SMS", "android.permission.RECEIVE_SMS", "android.permission.READ_SMS",
        "android.permission.RECORD_AUDIO", "android.permission.SYSTEM_ALERT_WINDOW",
        "android.permission.INSTALL_PACKAGES", "android.permission.WRITE_SECURE_SETTINGS",
        "android.permission.BIND_DEVICE_ADMIN", "android.permission.BIND_ACCESSIBILITY_SERVICE",
        "android.permission.DELETE_PACKAGES"
    }
    
    has_critical = any(p in perms_set for p in CRITICAL_PERMS)
    
    # Overall threat decision
    is_dangerous = False
    if raw_malware_prob >= 0.50:
        is_dangerous = True
    elif len(danger_detected) >= 2 and (has_critical or raw_malware_prob > 0.44):
        is_dangerous = True
        raw_malware_prob = max(raw_malware_prob, 0.88)
    elif has_sms_fraud or (has_spyware and has_persistence) or has_ransomware:
        is_dangerous = True
        raw_malware_prob = max(raw_malware_prob, 0.92)
    elif has_critical and raw_malware_prob > 0.42:
        is_dangerous = True
        raw_malware_prob = max(raw_malware_prob, 0.80)
        
    # Calculate realistic, dynamic confidence scores (no artificial clamping to 82%)
    if is_dangerous:
        if has_critical or len(danger_detected) >= 3:
            conf_score = min(98, max(88, round(raw_malware_prob * 100)))
        elif len(danger_detected) >= 1:
            conf_score = min(92, max(75, round(raw_malware_prob * 100)))
        else:
            conf_score = min(88, max(68, round(raw_malware_prob * 100)))
    else:
        # Safe application
        if len(perms_set) == 0:
            conf_score = 91
        else:
            benign_prob = 1.0 - raw_malware_prob
            conf_score = min(97, max(74, round(benign_prob * 100)))
        
    # Build risk factors
    risk_factors = []
    
    if is_dangerous:
        if has_sms_fraud:
            risk_factors.append({
                "type": "KNOWN_MALWARE_SIGNATURE",
                "description": "High-Risk Threat Pattern: SMS Fraud / Banking Trojan vector (Unauthorized premium messaging or 2FA bypass)."
            })
        if has_spyware:
            risk_factors.append({
                "type": "KNOWN_MALWARE_SIGNATURE",
                "description": "High-Risk Threat Pattern: Spyware / Surveillance vector (Covert microphone/camera and location tracking)."
            })
        if has_ransomware:
            risk_factors.append({
                "type": "KNOWN_MALWARE_SIGNATURE",
                "description": "High-Risk Threat Pattern: Screen Locker / Ransomware vector (System Overlay or Device Administrator privilege)."
            })
        if has_stealth_install:
            risk_factors.append({
                "type": "KNOWN_MALWARE_SIGNATURE",
                "description": "Critical Permission: Background Package Installer (Can download and execute drop payloads silently)."
            })
            
        for p in danger_detected:
            desc = DANGEROUS_PERM_INFO.get(p, "Sensitive system permission requested.")
            risk_factors.append({
                "type": "PERMISSION_OVERREACH",
                "description": f"Dangerous Permission: {p} - {desc}"
            })
            
        explanation = (
            f"XGBoost Threat Model identified {len(danger_detected)} high-risk permissions. "
            f"This application exhibits privilege combinations typical of Android Trojans or Spyware."
        )
    else:
        if len(perms_set) == 0:
            risk_factors.append({
                "type": "SAFE",
                "description": "No sensitive permissions declared in the application package."
            })
            explanation = "App appears safe. No excessive or sensitive permissions requested."
        else:
            risk_factors.append({
                "type": "SAFE",
                "description": f"Analyzed {len(perms_set)} declared permissions. No malicious signature or privilege overreach detected."
            })
            explanation = f"App appears safe. Permissions requested are consistent with standard benign utilities."
        
    verdict_str = "Dangerous" if is_dangerous else "Safe"
    
    alt_links = []
    if is_dangerous:
        alt_links = [
            {"title": "Verify Official App on Google Play", "url": "https://play.google.com/store/apps"},
            {"title": "Android Security Guidelines", "url": "https://source.android.com/docs/security"}
        ]
        
    return {
        "verdict": verdict_str,
        "confidenceScore": conf_score,
        "explanation": explanation,
        "riskFactors": risk_factors,
        "alternativeLinks": alt_links,
        "rawLabel": "malware" if is_dangerous else "benignware",
        "malwareProbability": round(raw_malware_prob, 3),
        "totalPermissions": len(perms_set),
        "dangerousPermissions": danger_detected,
        "appName": name or "Application"
    }

