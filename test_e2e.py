"""End-to-end test of the CardioCare prediction pipeline."""

import sys
import os
import io
import sqlite3
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

# 1. Load model & scaler
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("[OK] Model loaded:", type(model).__name__)
print("[OK] Scaler loaded:", type(scaler).__name__)

# 2. Load features
X = pd.read_csv("data/cleaned_X.csv")
feature_columns = list(X.columns)
print(f"[OK] {len(feature_columns)} feature columns loaded")

# 3. Build input vector
def split_feature_groups(feature_columns):
    groups = {"continuous": [], "sex": [], "cp": [], "restecg": [],
              "slope": [], "fbs": [], "exang": [], "thal": []}
    for col in feature_columns:
        if col in {"age", "trestbps", "chol", "thalach", "oldpeak", "ca"}:
            groups["continuous"].append(col)
        elif col.startswith("sex_"): groups["sex"].append(col)
        elif col.startswith("cp_"): groups["cp"].append(col)
        elif col.startswith("restecg_"): groups["restecg"].append(col)
        elif col.startswith("slope_"): groups["slope"].append(col)
        elif col.startswith("fbs_"): groups["fbs"].append(col)
        elif col.startswith("exang_"): groups["exang"].append(col)
        elif col.startswith("thal_"): groups["thal"].append(col)
    for key in groups:
        groups[key] = [c for c in feature_columns if c in groups[key]]
    return groups

groups = split_feature_groups(feature_columns)

user_inputs = {
    "age": 55, "trestbps": 130, "chol": 240,
    "thalach": 150, "oldpeak": 1.0, "ca": 0,
    "sex": "1", "cp": "1", "restecg": "2",
    "slope": "3", "fbs": "1", "exang": "0", "thal": "6.0",
}

values = {c: 0.0 for c in feature_columns}
for c in groups["continuous"]:
    values[c] = float(user_inputs[c])
for gn in ["sex", "cp", "restecg", "slope", "fbs", "exang", "thal"]:
    sel = str(user_inputs.get(gn))
    for col in groups[gn]:
        suffix = col.split("_", 1)[1]
        values[col] = 1.0 if str(suffix) == sel else 0.0

features_df = pd.DataFrame([values], columns=feature_columns)
print(f"[OK] Input vector shape: {features_df.shape}")

# 4. Scale and predict
features_scaled = scaler.transform(features_df)
pred = int(model.predict(features_scaled)[0])
proba = model.predict_proba(features_scaled)[0]
prob_positive = float(proba[1])

label = "Disease" if pred == 1 else "No Disease"
print(f"[OK] Prediction: {pred} ({label})")
print(f"[OK] Probability: {prob_positive:.4f}")

# 5. Test PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

buffer = io.BytesIO()
doc = SimpleDocTemplate(buffer, pagesize=A4)
styles = getSampleStyleSheet()
elements = [
    Paragraph("CardioCare Test Report", styles["Title"]),
    Paragraph(f"Prediction: {label}", styles["Normal"]),
    Paragraph(f"Probability: {prob_positive:.1%}", styles["Normal"]),
]
doc.build(elements)
pdf_bytes = buffer.getvalue()
print(f"[OK] PDF generated: {len(pdf_bytes)} bytes")

# 6. Test SQLite history
db_path = "data/prediction_history.db"
conn = sqlite3.connect(db_path)
conn.execute("""CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    probability REAL,
    age INTEGER, sex TEXT, chest_pain TEXT, resting_bp INTEGER,
    cholesterol INTEGER, max_hr INTEGER, oldpeak REAL, vessels INTEGER,
    fbs TEXT, exang TEXT, restecg TEXT, slope TEXT, thal TEXT
)""")
conn.execute(
    "INSERT INTO prediction_history (timestamp, prediction, probability, age, sex) VALUES (?, ?, ?, ?, ?)",
    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pred, prob_positive, 55, "Male"),
)
conn.commit()
cursor = conn.execute("SELECT COUNT(*) FROM prediction_history")
count = cursor.fetchone()[0]
print(f"[OK] History records: {count}")
conn.close()

# 7. Test multi-model evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

y = pd.read_csv("data/clean_y.csv")["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
Xtr = sc.fit_transform(X_train)
Xte = sc.transform(X_test)

for name, clf in [
    ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=100)),
    ("SVM (RBF)", SVC(kernel="rbf", random_state=42, probability=True)),
]:
    clf.fit(Xtr, y_train)
    p = clf.predict(Xte)
    a = accuracy_score(y_test, p)
    print(f"[OK] {name}: {a:.4f}")

# 8. Test Plotly
import plotly.graph_objects as go
fig = go.Figure(go.Bar(x=["A", "B"], y=[1, 2]))
print("[OK] Plotly charts work")

print()
print("=" * 40)
print("ALL END-TO-END TESTS PASSED")
print("=" * 40)
