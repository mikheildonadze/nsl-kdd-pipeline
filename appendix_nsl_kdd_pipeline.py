"""
Appendix: Reproducible Python Pipeline for NSL-KDD Intrusion Detection
=======================================================================
This script reproduces the baseline experiments reported in Section 6 of the paper.

Dataset: NSL-KDD — standard benchmark split
    Training:   KDDTrain+.txt  (125,973 records)
    Evaluation: KDDTest+.txt   (22,544 records — includes attack subtypes
                                 NOT present in KDDTrain+, by design, to
                                 test generalisation to unseen attacks)

IMPORTANT: Models are trained ONLY on KDDTrain+ and evaluated ONLY on the
held-out KDDTest+ file. This is the standard NSL-KDD evaluation protocol
used in the literature (Tavallaee et al., 2009; Shone et al., 2018, etc.).
Randomly re-splitting KDDTest+ alone (as in the previous version of this
script) mixes the same distribution into train and test, which inflates
accuracy and is NOT comparable to published NSL-KDD results.

Tasks: (i) Binary classification (Normal vs. Attack)
       (ii) Multi-class classification (Normal, DoS, Probe, R2L, U2R)
Models: Random Forest, SVM (RBF), Deep Neural Network (MLP)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. DATA LOADING — separate train and test files (standard NSL-KDD protocol)
# ---------------------------------------------------------------------------

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Adjust paths as needed — both files ship together in the standard NSL-KDD archive
df_train = pd.read_csv('KDDTrain+.txt', names=COLUMNS).drop('difficulty', axis=1)
df_test = pd.read_csv('KDDTest+.txt', names=COLUMNS).drop('difficulty', axis=1)

print(f"Train shape: {df_train.shape}   Test shape: {df_test.shape}")
print(f"Train label distribution:\n{df_train['label'].value_counts()}")
print(f"\nTest label distribution:\n{df_test['label'].value_counts()}")

# Sanity check: attack types present in test but never seen during training
train_attack_types = set(df_train['label'].unique())
test_attack_types = set(df_test['label'].unique())
unseen_in_train = test_attack_types - train_attack_types
print(f"\nAttack types in KDDTest+ NOT present in KDDTrain+ ({len(unseen_in_train)}): "
      f"{sorted(unseen_in_train)}")
print("These unseen-attack records are exactly what make the standard NSL-KDD "
      "evaluation a genuine generalisation test rather than a same-distribution split.\n")

# ---------------------------------------------------------------------------
# 2. PREPROCESSING — fit on train, apply to test (no leakage)
# ---------------------------------------------------------------------------

X_train_raw = df_train.drop('label', axis=1)
y_train_raw = df_train['label']
X_test_raw = df_test.drop('label', axis=1)
y_test_raw = df_test['label']

CATEGORICAL = ['protocol_type', 'service', 'flag']

# Fit LabelEncoder on the union of train+test categories for these columns only.
# This is pure feature-vocabulary alignment (no label information involved),
# needed because KDDTest+ contains 'service' values absent from KDDTrain+.
X_train_enc = X_train_raw.copy()
X_test_enc = X_test_raw.copy()
for col in CATEGORICAL:
    le = LabelEncoder()
    le.fit(pd.concat([X_train_raw[col], X_test_raw[col]], axis=0))
    X_train_enc[col] = le.transform(X_train_raw[col])
    X_test_enc[col] = le.transform(X_test_raw[col])

# Standardise numerical features — fit ONLY on training data, then apply
# the same transform to the test set (this is the leakage-safe way to scale).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

# ---------------------------------------------------------------------------
# 3. BINARY CLASSIFICATION: Normal (0) vs. Attack (1)
# ---------------------------------------------------------------------------

y_train_binary = (y_train_raw != 'normal').astype(int)
y_test_binary = (y_test_raw != 'normal').astype(int)

# --- Random Forest ---
rf_b = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_b.fit(X_train_scaled, y_train_binary)
rf_b_pred = rf_b.predict(X_test_scaled)
print("[Binary] Random Forest Accuracy:", accuracy_score(y_test_binary, rf_b_pred))

# --- SVM (RBF) ---
svm_b = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_b.fit(X_train_scaled, y_train_binary)
svm_b_pred = svm_b.predict(X_test_scaled)
print("[Binary] SVM Accuracy:", accuracy_score(y_test_binary, svm_b_pred))

# --- DNN (MLP) ---
mlp_b = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    max_iter=300, random_state=42,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
)
mlp_b.fit(X_train_scaled, y_train_binary)
mlp_b_pred = mlp_b.predict(X_test_scaled)
print("[Binary] DNN Accuracy:", accuracy_score(y_test_binary, mlp_b_pred))

# ---------------------------------------------------------------------------
# 4. MULTI-CLASS CLASSIFICATION: Normal, DoS, Probe, R2L, U2R
# ---------------------------------------------------------------------------

ATTACK_MAP = {
    'normal': 'Normal',
    # DoS
    'neptune': 'DoS', 'smurf': 'DoS', 'back': 'DoS', 'teardrop': 'DoS',
    'pod': 'DoS', 'land': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
    'mailbomb': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L',
    'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L',
    'spy': 'R2L', 'named': 'R2L', 'sendmail': 'R2L', 'snmpgetattack': 'R2L',
    'snmpguess': 'R2L', 'worm': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'httptunnel': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'rootkit': 'U2R', 'loadmodule': 'U2R',
    'perl': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

y_train_multi = y_train_raw.map(ATTACK_MAP)
y_test_multi = y_test_raw.map(ATTACK_MAP)

missing_train = y_train_multi.isna().sum()
missing_test = y_test_multi.isna().sum()
if missing_train or missing_test:
    print(f"\nWARNING: {missing_train} train / {missing_test} test labels "
          f"were not found in ATTACK_MAP — check for unmapped attack names "
          f"before proceeding.")

# Fit the class-label encoder on train+test category names jointly (again,
# this only aligns the *names* Normal/DoS/Probe/R2L/U2R to integer codes;
# it does not use any test feature or leak test identities into training).
le_multi = LabelEncoder()
le_multi.fit(pd.concat([y_train_multi, y_test_multi], axis=0))
y_train_multi_enc = le_multi.transform(y_train_multi)
y_test_multi_enc = le_multi.transform(y_test_multi)
print(f"\nMulti-class classes: {list(le_multi.classes_)}")

# --- Random Forest ---
rf_m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_m.fit(X_train_scaled, y_train_multi_enc)
rf_m_pred = rf_m.predict(X_test_scaled)
print("[Multi] Random Forest Accuracy:", accuracy_score(y_test_multi_enc, rf_m_pred))

# --- SVM (RBF) ---
svm_m = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_m.fit(X_train_scaled, y_train_multi_enc)
svm_m_pred = svm_m.predict(X_test_scaled)
print("[Multi] SVM Accuracy:", accuracy_score(y_test_multi_enc, svm_m_pred))

# --- DNN (MLP) ---
mlp_m = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    max_iter=300, random_state=42,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
)
mlp_m.fit(X_train_scaled, y_train_multi_enc)
mlp_m_pred = mlp_m.predict(X_test_scaled)
print("[Multi] DNN Accuracy:", accuracy_score(y_test_multi_enc, mlp_m_pred))

# ---------------------------------------------------------------------------
# 5. DETAILED REPORTS — now printed for ALL THREE models (RF, SVM, DNN),
#    both binary and multi-class, so results are fully comparable across
#    models rather than only available for Random Forest.
# ---------------------------------------------------------------------------

binary_models = [
    ("Random Forest", rf_b_pred),
    ("SVM (RBF)", svm_b_pred),
    ("DNN (MLP)", mlp_b_pred),
]
for name, pred in binary_models:
    print("\n" + "="*60)
    print(f"BINARY CLASSIFICATION REPORT — {name}")
    print("="*60)
    print(classification_report(y_test_binary, pred, target_names=['Normal', 'Attack']))

multi_models = [
    ("Random Forest", rf_m_pred),
    ("SVM (RBF)", svm_m_pred),
    ("DNN (MLP)", mlp_m_pred),
]
for name, pred in multi_models:
    print("\n" + "="*60)
    print(f"MULTI-CLASS CLASSIFICATION REPORT — {name}")
    print("="*60)
    print(classification_report(y_test_multi_enc, pred, target_names=le_multi.classes_))

# ---------------------------------------------------------------------------
# 6. CONFUSION MATRIX (Random Forest, multi-class — the best-performing model)
# ---------------------------------------------------------------------------

def plot_confusion(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

plot_confusion(y_test_multi_enc, rf_m_pred, le_multi.classes_,
               'Random Forest — Multi-Class Confusion Matrix (KDDTest+)',
               'confusion_matrix_multiclass.png')

# Also print the raw confusion matrix as text, so the exact cell counts are
# available even without opening the saved PNG.
print("\n" + "="*60)
print("CONFUSION MATRIX (RAW COUNTS) — Random Forest, Multi-Class")
print("="*60)
cm_multi = confusion_matrix(y_test_multi_enc, rf_m_pred)
header = "        " + "".join(f"{c:>10}" for c in le_multi.classes_)
print(header)
for cls, row in zip(le_multi.classes_, cm_multi):
    print(f"{cls:>8}" + "".join(f"{v:>10}" for v in row))

# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

importances = rf_m.feature_importances_
indices = np.argsort(importances)[::-1][:10]
feature_names = X_train_enc.columns.tolist()

plt.figure(figsize=(6, 3))
plt.barh(range(10), importances[indices], color='#2ecc71')
plt.yticks(range(10), [feature_names[i] for i in indices])
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (Random Forest, trained on KDDTrain+)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=200)
plt.close()

# Also print the top-10 feature importances as text, with exact values.
print("\n" + "="*60)
print("TOP 10 FEATURE IMPORTANCES — Random Forest, Multi-Class")
print("="*60)
for rank, i in enumerate(indices, start=1):
    print(f"{rank:>2}. {feature_names[i]:<30} {importances[i]:.4f}")

print("\nAll figures saved. Pipeline complete.")
print("\nNOTE: If accuracy is noticeably lower than in the previous (incorrect) "
      "80/20-within-KDDTest+ version, that is expected and correct — it now "
      "reflects genuine generalisation to attack types absent from training, "
      "consistent with published NSL-KDD benchmark results in the literature.")
