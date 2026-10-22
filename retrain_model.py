"""
Retrain the SVM model with probability=True and save the StandardScaler.

This preserves the original training approach (SVC with RBF kernel, random_state=42)
but enables predict_proba and saves the scaler for consistent inference.

The original model is preserved as models/final_model_original_backup.pkl.
"""

import os
import shutil

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
X_PATH = os.path.join(DATA_DIR, "cleaned_X.csv")
Y_PATH = os.path.join(DATA_DIR, "clean_y.csv")
ORIGINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.pkl")
BACKUP_MODEL_PATH = os.path.join(MODELS_DIR, "final_model_original_backup.pkl")
NEW_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Load data
X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH)["target"]

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target distribution: {dict(y.value_counts().sort_index())}")

# Same split as the original training (test_size=0.2, random_state=42, no stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM with probability=True (same kernel, same random_state)
model = SVC(kernel="rbf", random_state=42, probability=True)
model.fit(X_train_scaled, y_train)

# Evaluate
preds = model.predict(X_test_scaled)
acc = accuracy_score(y_test, preds)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds))

# Backup original model
if os.path.exists(ORIGINAL_MODEL_PATH) and not os.path.exists(BACKUP_MODEL_PATH):
    shutil.copy2(ORIGINAL_MODEL_PATH, BACKUP_MODEL_PATH)
    print(f"Original model backed up to {BACKUP_MODEL_PATH}")

# Save new model and scaler
joblib.dump(model, NEW_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"New model saved to {NEW_MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")
print(f"\nModel now supports predict_proba: {hasattr(model, 'predict_proba')}")
