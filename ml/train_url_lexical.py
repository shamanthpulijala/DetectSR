import os, json, pickle, warnings, re
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

os.makedirs("models", exist_ok=True)
SEED = 42

# ── 1. Feature engineering from a raw URL string ─────────────────────────────
# Maps directly to the new dataset's lexical columns (indices 0–96 only)
KNOWN_TLDS = {'.com','.net','.org','.edu','.gov','.co','.io','.xyz','.info','.biz',
              '.club','.online','.site','.store','.tech','.vu','.tk','.ml','.ga','.cf'}

def extract_lexical_features(url: str) -> dict:
    """Extract 97 lexical features that exactly match the URL.csv column ordering."""
    url = str(url).strip()
    try:
        parsed = urlparse(url if '://' in url else 'http://' + url)
        domain   = parsed.netloc or ''
        path     = parsed.path   or ''
        params   = parsed.query  or ''
        # split path into directory / file
        parts    = path.rsplit('/', 1)
        dir_part = parts[0] if len(parts) > 1 else ''
        file_part= parts[1] if len(parts) > 1 else parts[0] if parts else ''
    except Exception:
        domain = dir_part = file_part = params = ''

    def count_char(s, c): return s.count(c)

    f = {}
    # ── URL level ────────────────────────────────────────────
    f['qty_dot_url']         = count_char(url, '.')
    f['qty_hyphen_url']      = count_char(url, '-')
    f['qty_underline_url']   = count_char(url, '_')
    f['qty_slash_url']       = count_char(url, '/')
    f['qty_questionmark_url']= count_char(url, '?')
    f['qty_equal_url']       = count_char(url, '=')
    f['qty_at_url']          = count_char(url, '@')
    f['qty_and_url']         = count_char(url, '&')
    f['qty_exclamation_url'] = count_char(url, '!')
    f['qty_space_url']       = count_char(url, ' ')
    f['qty_tilde_url']       = count_char(url, '~')
    f['qty_comma_url']       = count_char(url, ',')
    f['qty_plus_url']        = count_char(url, '+')
    f['qty_asterisk_url']    = count_char(url, '*')
    f['qty_hashtag_url']     = count_char(url, '#')
    f['qty_dollar_url']      = count_char(url, '$')
    f['qty_percent_url']     = count_char(url, '%')
    tld = next((t for t in KNOWN_TLDS if domain.endswith(t)), '')
    f['qty_tld_url']         = len(tld)
    f['length_url']          = len(url)
    # ── Domain level ─────────────────────────────────────────
    f['qty_dot_domain']         = count_char(domain, '.')
    f['qty_hyphen_domain']      = count_char(domain, '-')
    f['qty_underline_domain']   = count_char(domain, '_')
    f['qty_slash_domain']       = count_char(domain, '/')
    f['qty_questionmark_domain']= count_char(domain, '?')
    f['qty_equal_domain']       = count_char(domain, '=')
    f['qty_at_domain']          = count_char(domain, '@')
    f['qty_and_domain']         = count_char(domain, '&')
    f['qty_exclamation_domain'] = count_char(domain, '!')
    f['qty_space_domain']       = count_char(domain, ' ')
    f['qty_tilde_domain']       = count_char(domain, '~')
    f['qty_comma_domain']       = count_char(domain, ',')
    f['qty_plus_domain']        = count_char(domain, '+')
    f['qty_asterisk_domain']    = count_char(domain, '*')
    f['qty_hashtag_domain']     = count_char(domain, '#')
    f['qty_dollar_domain']      = count_char(domain, '$')
    f['qty_percent_domain']     = count_char(domain, '%')
    f['qty_vowels_domain']      = sum(c in 'aeiouAEIOU' for c in domain)
    f['domain_length']          = len(domain)
    f['domain_in_ip']           = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', domain)))
    f['server_client_domain']   = int('client' in domain.lower() or 'server' in domain.lower())
    # ── Directory level ───────────────────────────────────────
    f['qty_dot_directory']         = count_char(dir_part, '.')
    f['qty_hyphen_directory']      = count_char(dir_part, '-')
    f['qty_underline_directory']   = count_char(dir_part, '_')
    f['qty_slash_directory']       = count_char(dir_part, '/')
    f['qty_questionmark_directory']= count_char(dir_part, '?')
    f['qty_equal_directory']       = count_char(dir_part, '=')
    f['qty_at_directory']          = count_char(dir_part, '@')
    f['qty_and_directory']         = count_char(dir_part, '&')
    f['qty_exclamation_directory'] = count_char(dir_part, '!')
    f['qty_space_directory']       = count_char(dir_part, ' ')
    f['qty_tilde_directory']       = count_char(dir_part, '~')
    f['qty_comma_directory']       = count_char(dir_part, ',')
    f['qty_plus_directory']        = count_char(dir_part, '+')
    f['qty_asterisk_directory']    = count_char(dir_part, '*')
    f['qty_hashtag_directory']     = count_char(dir_part, '#')
    f['qty_dollar_directory']      = count_char(dir_part, '$')
    f['qty_percent_directory']     = count_char(dir_part, '%')
    f['directory_length']          = len(dir_part)
    # ── File level ────────────────────────────────────────────
    f['qty_dot_file']         = count_char(file_part, '.')
    f['qty_hyphen_file']      = count_char(file_part, '-')
    f['qty_underline_file']   = count_char(file_part, '_')
    f['qty_slash_file']       = count_char(file_part, '/')
    f['qty_questionmark_file']= count_char(file_part, '?')
    f['qty_equal_file']       = count_char(file_part, '=')
    f['qty_at_file']          = count_char(file_part, '@')
    f['qty_and_file']         = count_char(file_part, '&')
    f['qty_exclamation_file'] = count_char(file_part, '!')
    f['qty_space_file']       = count_char(file_part, ' ')
    f['qty_tilde_file']       = count_char(file_part, '~')
    f['qty_comma_file']       = count_char(file_part, ',')
    f['qty_plus_file']        = count_char(file_part, '+')
    f['qty_asterisk_file']    = count_char(file_part, '*')
    f['qty_hashtag_file']     = count_char(file_part, '#')
    f['qty_dollar_file']      = count_char(file_part, '$')
    f['qty_percent_file']     = count_char(file_part, '%')
    f['file_length']          = len(file_part)
    # ── Params level ──────────────────────────────────────────
    f['qty_dot_params']         = count_char(params, '.')
    f['qty_hyphen_params']      = count_char(params, '-')
    f['qty_underline_params']   = count_char(params, '_')
    f['qty_slash_params']       = count_char(params, '/')
    f['qty_questionmark_params']= count_char(params, '?')
    f['qty_equal_params']       = count_char(params, '=')
    f['qty_at_params']          = count_char(params, '@')
    f['qty_and_params']         = count_char(params, '&')
    f['qty_exclamation_params'] = count_char(params, '!')
    f['qty_space_params']       = count_char(params, ' ')
    f['qty_tilde_params']       = count_char(params, '~')
    f['qty_comma_params']       = count_char(params, ',')
    f['qty_plus_params']        = count_char(params, '+')
    f['qty_asterisk_params']    = count_char(params, '*')
    f['qty_hashtag_params']     = count_char(params, '#')
    f['qty_dollar_params']      = count_char(params, '$')
    f['qty_percent_params']     = count_char(params, '%')
    f['params_length']          = len(params)
    f['tld_present_params']     = int(any(t in params for t in KNOWN_TLDS))
    f['qty_params']             = len(params.split('&')) if params else 0
    f['email_in_url']           = int(bool(re.search(r'[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+', url)))
    return f

# Ordered columns — same insertion order as above
LEXICAL_COLS = list(extract_lexical_features('http://test.com').keys())
assert len(LEXICAL_COLS) == 97, f"Expected 97, got {len(LEXICAL_COLS)}"
print(f"Lexical feature count: {len(LEXICAL_COLS)}")

# ── 2. Load and prepare data ─────────────────────────────────────────────────
print("="*60)
print(" LOADING URL.csv (147K rows)")
print("="*60)
df = pd.read_csv(r"E:\code things\DL_PROJECT\URL.csv")
print(f"Original: {df.shape}")
df.drop_duplicates(inplace=True)
print(f"After dedup: {df.shape}")

# Use only the first 97 lexical columns (no network features)
lexical_feature_cols = LEXICAL_COLS  # from URL.csv the same order
X = df[LEXICAL_COLS].fillna(-1).values
y = df['label'].values
print(f"Class balance: {pd.Series(y).value_counts().to_dict()}")

X_train_tmp, X_test, y_train_tmp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_tmp, y_train_tmp, test_size=0.15/0.85, random_state=SEED, stratify=y_train_tmp
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 3. Train XGBoost ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" TRAINING XGBoost (lexical only, 97 features)")
print("="*60)

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
spw = neg / pos

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    scale_pos_weight=spw,
    random_state=SEED,
    early_stopping_rounds=20,
    eval_metric='auc'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}")

# ── 4. Evaluate ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" TEST SET EVALUATION")
print("="*60)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# ── 5. Sanity check on known URLs ────────────────────────────────────────────
print("\n" + "="*60)
print(" SANITY CHECK")
print("="*60)
test_urls = [
    ('http://paypal-login-verify.xyz/secure', 'DANGEROUS'),
    ('https://google.com',                    'SAFE'),
    ('https://github.com',                    'SAFE'),
    ('https://microsoft.com',                 'SAFE'),
    ('http://9779.info/malware.php',          'DANGEROUS'),
    ('http://www.crestonwood.com/router.php', 'DANGEROUS'),
]
for u, expected in test_urls:
    feats = extract_lexical_features(u)
    arr = np.array([[feats.get(k, -1) for k in LEXICAL_COLS]])
    prob = model.predict_proba(arr)[0]
    label = 'Malicious' if prob[1] >= 0.5 else 'Benign'
    verdict = 'DANGEROUS' if label == 'Malicious' else 'SAFE'
    match = 'PASS' if verdict == expected else 'FAIL'
    print(match, u[:45], '->', label, 'Mal:', round(prob[1]*100,1), '% (expected', expected, ')')

# ── 6. Save artifacts ────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" SAVING ARTIFACTS")
print("="*60)
with open("models/url_xgb_lexical.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/url_xgb_lexical_cols.json", "w") as f:
    json.dump(LEXICAL_COLS, f)
print("Saved: models/url_xgb_lexical.pkl")
print("Saved: models/url_xgb_lexical_cols.json")
