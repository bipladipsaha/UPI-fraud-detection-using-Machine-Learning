import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
from feature_engineering import add_location_device_features

# =========================================================
# LOAD DATA
# =========================================================
print("Loading transaction data...")
df = pd.read_csv("upi_fraud_synthetic.csv")
df = add_location_device_features(df)


print(f"Total transactions: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# =========================================================
# FEATURE ENGINEERING (SIMPLE & SAFE)
# =========================================================
print("\nFEATURE ENGINEERING")

# Balance error
df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]

# Time features
df["is_night"] = df["hour"].apply(lambda x: 1 if 0 <= x <= 5 else 0)
df["is_peak"] = df["hour"].apply(lambda x: 1 if 9 <= x <= 17 else 0)

# One-hot encode transaction type
df = pd.get_dummies(
    df,
    columns=["transaction_type", "device_type", "user_city", "merchant_city"],
    drop_first=False
)


print("Features created successfully.")
print(df.head())

# =========================================================
# PREPARE DATA
# =========================================================
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Scaling (kept for Flask compatibility)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# RANDOM FOREST
# =========================================================
print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nRandom Forest Results")
print(f"Accuracy : {accuracy_score(y_test, rf_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"Recall   : {recall_score(y_test, rf_pred):.4f}")
print(f"F1-score : {f1_score(y_test, rf_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, rf_proba):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

# =========================================================
# GRADIENT BOOSTING (OPTIONAL COMPARISON)
# =========================================================
print("\nTraining Gradient Boosting...")

gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)

print("\nGradient Boosting Results")
print(classification_report(y_test, gb_pred))

# =========================================================
# THRESHOLD OPTIMIZATION
# =========================================================
print("\nThreshold Optimization")

thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    preds = (rf_proba >= t).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

print(f"Best Threshold: {best_threshold}")
print(f"Best F1 Score: {best_f1:.4f}")

# =========================================================
# SAVE MODEL & FILES
# =========================================================
joblib.dump(rf_model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nMODEL FILES SAVED ✅")
print("fraud_model.pkl")
print("scaler.pkl")
print("model_columns.pkl")
