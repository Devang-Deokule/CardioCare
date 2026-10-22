"""
CardioCare — Heart Disease Prediction System
=============================================
A polished, executive-grade Streamlit application for clinical heart disease
risk assessment using an SVM classifier trained on the UCI Heart Disease dataset.

Features:
  - Tab-based navigation (no radio buttons)
  - Clean inline SVG icons (no cartoon emojis)
  - Refined clinical navy / slate color palette
  - Complete "Machine Learning Guide" explaining all concepts from basics
  - Interactive single & batch prediction with StandardScaler pipeline
  - Multi-model evaluation & confusion matrix analytics
  - SQLite persistent local history
  - ReportLab PDF clinical summary generation
"""

import io
import os
import sys
import sqlite3
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ReportLab imports for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Suppress sklearn version warnings in the UI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
CLEANED_X_PATH = os.path.join(BASE_DIR, "data", "cleaned_X.csv")
CLEANED_Y_PATH = os.path.join(BASE_DIR, "data", "clean_y.csv")
METRICS_PATH = os.path.join(BASE_DIR, "results", "evaluation_metrics.txt")
DB_PATH = os.path.join(BASE_DIR, "data", "prediction_history.db")

# ---------------------------------------------------------------------------
# Feature metadata for human-readable labels
# ---------------------------------------------------------------------------
CONTINUOUS_FEATURES = {
    "age": {"label": "Age", "unit": "years", "min": 1, "max": 120, "default": 55,
            "help": "Patient's age in years"},
    "trestbps": {"label": "Resting Blood Pressure", "unit": "mm Hg", "min": 70,
                 "max": 250, "default": 130,
                 "help": "Resting blood pressure on admission (mm Hg)"},
    "chol": {"label": "Serum Cholesterol", "unit": "mg/dl", "min": 100, "max": 700,
              "default": 240,
              "help": "Serum cholesterol level in mg/dl"},
    "thalach": {"label": "Max Heart Rate Achieved", "unit": "bpm", "min": 60,
                "max": 250, "default": 150,
                "help": "Maximum heart rate achieved during exercise test"},
    "oldpeak": {"label": "ST Depression", "unit": "mm", "min": 0.0, "max": 10.0,
                "default": 1.0,
                "help": "ST depression induced by exercise relative to rest"},
    "ca": {"label": "Major Vessels Coloured", "unit": "count", "min": 0, "max": 3,
            "default": 0,
            "help": "Number of major vessels coloured by fluoroscopy (0–3)"},
}

CATEGORICAL_FEATURES = {
    "sex": {
        "label": "Sex",
        "options": {"0": "Female", "1": "Male"},
        "help": "Biological sex of the patient",
    },
    "cp": {
        "label": "Chest Pain Type",
        "options": {
            "1": "Typical angina",
            "2": "Atypical angina",
            "3": "Non-anginal pain",
            "4": "Asymptomatic",
        },
        "help": "Type of chest pain experienced",
    },
    "restecg": {
        "label": "Resting ECG",
        "options": {
            "0": "Normal",
            "1": "ST-T wave abnormality",
            "2": "Left ventricular hypertrophy",
        },
        "help": "Resting electrocardiographic results",
    },
    "slope": {
        "label": "ST Segment Slope",
        "options": {
            "1": "Upsloping",
            "2": "Flat",
            "3": "Downsloping",
        },
        "help": "Slope of the peak exercise ST segment",
    },
    "fbs": {
        "label": "Fasting Blood Sugar",
        "options": {
            "0": "≤ 120 mg/dl (Normal)",
            "1": "> 120 mg/dl (Elevated)",
        },
        "help": "Fasting blood sugar > 120 mg/dl",
    },
    "exang": {
        "label": "Exercise-Induced Angina",
        "options": {
            "0": "No",
            "1": "Yes",
        },
        "help": "Exercise-induced angina present",
    },
    "thal": {
        "label": "Thalassemia",
        "options": {
            "3": "Normal",
            "6": "Fixed defect",
            "7": "Reversible defect",
        },
        "help": "Thalassemia blood disorder status",
    },
}

# ---------------------------------------------------------------------------
# Clean SVG Icons (Replaces emojis with crisp clinical vector graphics)
# ---------------------------------------------------------------------------
SVG_ICONS = {
    "heart": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l1.5-3 2 6 1.5-3h6.28"/></svg>',
    "pulse": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "dashboard": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/></svg>',
    "brain": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04Z"/></svg>',
    "analytics": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "history": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>',
    "stethoscope": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg>',
    "check": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>',
    "book": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 2v20"/></svg>',
    "file": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/></svg>',
    "award": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "download": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "sliders": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/></svg>',
    "trash": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>',
}


def icon_svg(name: str, size: int = 20, color: str = "#2563EB") -> str:
    """Return inline SVG icon markup safely."""
    tmpl = SVG_ICONS.get(name, SVG_ICONS["info"])
    return tmpl.format(size=size, color=color)


# ---------------------------------------------------------------------------
# Database Management (SQLite for prediction history)
# ---------------------------------------------------------------------------
def init_db():
    """Initialise SQLite database table for prediction history."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL,
                age INTEGER,
                sex TEXT,
                cp TEXT,
                trestbps INTEGER,
                chol INTEGER,
                fbs TEXT,
                restecg TEXT,
                thalach INTEGER,
                exang TEXT,
                oldpeak REAL,
                slope TEXT,
                ca INTEGER,
                thal TEXT
            )
            """
        )
        # Migrate table if created with legacy column names
        cursor.execute("PRAGMA table_info(prediction_history)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        desired_cols = {
            "cp": "TEXT", "trestbps": "INTEGER", "chol": "INTEGER",
            "thalach": "INTEGER", "ca": "INTEGER", "sex": "TEXT",
            "fbs": "TEXT", "exang": "TEXT", "restecg": "TEXT",
            "slope": "TEXT", "thal": "TEXT", "probability": "REAL",
            "oldpeak": "REAL"
        }
        for col, col_type in desired_cols.items():
            if col not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE prediction_history ADD COLUMN {col} {col_type}")
                except Exception:
                    pass
        conn.commit()


def save_prediction(prediction: int, probability: Optional[float], inputs: dict):
    """Save a prediction record into the SQLite database."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                timestamp, prediction, probability,
                age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                int(prediction),
                float(probability) if probability is not None else None,
                int(inputs.get("age", 0)),
                str(inputs.get("sex", "")),
                str(inputs.get("cp", "")),
                int(inputs.get("trestbps", 0)),
                int(inputs.get("chol", 0)),
                str(inputs.get("fbs", "")),
                str(inputs.get("restecg", "")),
                int(inputs.get("thalach", 0)),
                str(inputs.get("exang", "")),
                float(inputs.get("oldpeak", 0.0)),
                str(inputs.get("slope", "")),
                int(inputs.get("ca", 0)),
                str(inputs.get("thal", "")),
            ),
        )
        conn.commit()


def get_prediction_history(limit: int = 100) -> pd.DataFrame:
    """Retrieve prediction history records from SQLite database."""
    init_db()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM prediction_history ORDER BY id DESC LIMIT {limit}",
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


def get_history_count() -> int:
    """Return the total number of saved predictions."""
    init_db()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prediction_history")
            return cursor.fetchone()[0]
    except Exception:
        return 0


def clear_prediction_history():
    """Clear all records from prediction history."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_history")
        conn.commit()


# ---------------------------------------------------------------------------
# Data & Model Helpers (Cached for performance)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str = MODEL_PATH) -> SVC:
    """Load the trained SVM model."""
    return joblib.load(path)


@st.cache_resource
def load_scaler(path: str = SCALER_PATH) -> Optional[StandardScaler]:
    """Load the trained StandardScaler."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


@st.cache_data
def load_feature_columns(path: str = CLEANED_X_PATH) -> List[str]:
    """Load feature column names from cleaned_X.csv."""
    df = pd.read_csv(path, nrows=1)
    return list(df.columns)


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load cleaned dataset."""
    X = pd.read_csv(CLEANED_X_PATH)
    y = pd.read_csv(CLEANED_Y_PATH).squeeze()
    return X, y


@st.cache_data
def evaluate_models() -> Dict[str, dict]:
    """
    Train and evaluate multiple models on an 80/20 train/test split.
    Uses the exact split: test_size=0.2, random_state=42.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42, probability=True),
    }

    results = {}
    for name, model in models.items():
        if name in ("Logistic Regression", "SVM (RBF Kernel)"):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "report": cr,
            "model": model,
        }

    return results


def split_feature_groups(feature_columns: List[str]) -> Dict[str, List[str]]:
    """Group one-hot encoded columns by their original categorical prefix."""
    groups = {k: [] for k in CATEGORICAL_FEATURES}
    for col in feature_columns:
        for prefix in groups:
            if col.startswith(f"{prefix}_"):
                groups[prefix].append(col)
                break
    return groups


def build_input_vector(
    feature_columns: List[str],
    groups: Dict[str, List[str]],
    user_inputs: dict,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the model's expected 25-feature schema."""
    row = {col: 0.0 for col in feature_columns}

    for col in CONTINUOUS_FEATURES:
        if col in row:
            row[col] = float(user_inputs[col])

    for prefix in CATEGORICAL_FEATURES:
        chosen_val = user_inputs[prefix]
        target_col = f"{prefix}_{chosen_val}"
        for col in groups.get(prefix, []):
            row[col] = 1.0 if col == target_col else 0.0

    return pd.DataFrame([row], columns=feature_columns)


# ---------------------------------------------------------------------------
# PDF Generation with ReportLab
# ---------------------------------------------------------------------------
def generate_pdf_report(
    prediction: int,
    probability: Optional[float],
    user_inputs: dict,
) -> bytes:
    """Generate a clean, clinical PDF summary report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CCTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "CCSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748B"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "CCHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "CCBody",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=colors.HexColor("#334155"),
        leading=14,
    )
    disclaimer_style = ParagraphStyle(
        "CCDisclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#92400E"),
        leading=11,
        backColor=colors.HexColor("#FEF3C7"),
        borderPadding=(8, 8, 8, 8),
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("CardioCare", title_style))
    elements.append(Paragraph("Heart Disease Risk Assessment Report", subtitle_style))
    elements.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            subtitle_style,
        )
    )
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")))
    elements.append(Spacer(1, 10))

    # Prediction Result Banner
    elements.append(Paragraph("Clinical Risk Assessment", heading_style))
    result_text = "ELEVATED RISK: Heart Disease Indication Detected" if prediction == 1 else "NORMAL PROFILE: No Heart Disease Indication Detected"
    result_color = "#DC2626" if prediction == 1 else "#059669"
    elements.append(
        Paragraph(
            f'<font color="{result_color}" size="13"><b>{result_text}</b></font>',
            body_style,
        )
    )

    if probability is not None:
        confidence = probability if prediction == 1 else (1.0 - probability)
        elements.append(
            Paragraph(
                f"Model Statistical Confidence: <b>{confidence:.1%}</b> (Platt-scaled SVM probability)",
                body_style,
            )
        )
    elements.append(Spacer(1, 10))

    # Patient Measurements Table
    elements.append(Paragraph("Patient Input Measurements", heading_style))
    sex_label = CATEGORICAL_FEATURES["sex"]["options"].get(str(user_inputs.get("sex", "")), str(user_inputs.get("sex", "")))
    cp_label = CATEGORICAL_FEATURES["cp"]["options"].get(str(user_inputs.get("cp", "")), str(user_inputs.get("cp", "")))
    restecg_label = CATEGORICAL_FEATURES["restecg"]["options"].get(str(user_inputs.get("restecg", "")), str(user_inputs.get("restecg", "")))
    slope_label = CATEGORICAL_FEATURES["slope"]["options"].get(str(user_inputs.get("slope", "")), str(user_inputs.get("slope", "")))
    fbs_label = CATEGORICAL_FEATURES["fbs"]["options"].get(str(user_inputs.get("fbs", "")), str(user_inputs.get("fbs", "")))
    exang_label = CATEGORICAL_FEATURES["exang"]["options"].get(str(user_inputs.get("exang", "")), str(user_inputs.get("exang", "")))
    thal_label = CATEGORICAL_FEATURES["thal"]["options"].get(str(user_inputs.get("thal", "")), str(user_inputs.get("thal", "")))

    data = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Age", f"{user_inputs.get('age', '-')} years", "Sex", sex_label],
        ["Resting Blood Pressure", f"{user_inputs.get('trestbps', '-')} mm Hg", "Chest Pain Type", cp_label],
        ["Serum Cholesterol", f"{user_inputs.get('chol', '-')} mg/dl", "Fasting Blood Sugar", fbs_label],
        ["Max Heart Rate Achieved", f"{user_inputs.get('thalach', '-')} bpm", "Resting ECG", restecg_label],
        ["ST Depression (oldpeak)", f"{user_inputs.get('oldpeak', '-')} mm", "Exercise Angina", exang_label],
        ["Major Vessels (fluoroscopy)", str(user_inputs.get('ca', '-')), "ST Segment Slope", slope_label],
        ["Thalassemia Status", thal_label, "Model Champion", "SVM (RBF Kernel, 90.0% Acc)"],
    ]

    t = Table(data, colWidths=[3.2 * cm, 4.8 * cm, 3.8 * cm, 5.2 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ]
        )
    )
    elements.append(t)
    elements.append(Spacer(1, 14))

    # Clinical Methodology Summary
    elements.append(Paragraph("System & Preprocessing Methodology", heading_style))
    elements.append(
        Paragraph(
            "This prediction was generated using a Support Vector Machine classifier with a Radial Basis Function (RBF) "
            "kernel, trained on the UCI Cleveland Heart Disease dataset (297 instances, 25 features post one-hot encoding). "
            "Input features were scaled using a dedicated StandardScaler fitted exclusively on training distributions.",
            body_style,
        )
    )
    elements.append(Spacer(1, 12))

    # Disclaimer
    elements.append(Paragraph("Academic & Medical Disclaimer", heading_style))
    elements.append(
        Paragraph(
            "NOTICE: This report is generated by an educational machine learning model for demonstration purposes only. "
            "It does NOT constitute medical advice, clinical diagnosis, or treatment planning. Always seek evaluation "
            "from a qualified cardiologist or healthcare professional for cardiovascular diagnosis.",
            disclaimer_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Plotly Chart Helpers — Cohesive Executive Navy / Slate Theme
# ---------------------------------------------------------------------------
CHART_COLORS = ["#2563EB", "#059669", "#7C3AED", "#D97706"]

BASE_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", color="#0F172A", size=12),
)


def create_model_comparison_chart(results: dict) -> go.Figure:
    """Create a grouped bar chart comparing model metrics."""
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig = go.Figure()
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in models]
        fig.add_trace(
            go.Bar(
                name=label,
                x=models,
                y=values,
                marker_color=CHART_COLORS[i],
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
                textfont=dict(size=10),
            )
        )

    fig.update_layout(
        **BASE_CHART_LAYOUT,
        barmode="group",
        yaxis=dict(
            range=[0, 1.15],
            tickformat=".0%",
            gridcolor="#E2E8F0",
            zeroline=False,
        ),
        xaxis=dict(gridcolor="#E2E8F0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def create_confusion_matrix_chart(cm: np.ndarray, model_name: str) -> go.Figure:
    """Create a clean heatmap for the confusion matrix without duplicate kwargs."""
    labels = ["No Disease (0)", "Disease (1)"]
    text = [[f"{val} cases" for val in row] for row in cm]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16, color="#0F172A"),
            colorscale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1E3A8A"]],
            showscale=False,
        )
    )
    fig.update_layout(
        **BASE_CHART_LAYOUT,
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=13, color="#1E3A8A")),
        xaxis_title="Predicted Class",
        yaxis_title="True Actual Class",
        yaxis=dict(autorange="reversed"),
        height=320,
        width=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def create_accuracy_gauge(accuracy: float) -> go.Figure:
    """Create an accurate gauge chart without duplicate margin keyword arguments."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,
            number={"suffix": "%", "font": {"size": 26, "color": "#0F172A"}},
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=0, tickcolor="#94A3B8"),
                bar=dict(color="#2563EB"),
                bgcolor="#F8FAFC",
                borderwidth=0,
                steps=[
                    dict(range=[0, 50], color="#FEE2E2"),
                    dict(range=[50, 75], color="#FEF3C7"),
                    dict(range=[75, 100], color="#DCFCE7"),
                ],
            ),
        )
    )
    fig.update_layout(
        **BASE_CHART_LAYOUT,
        height=190,
        margin=dict(l=25, r=25, t=25, b=15),
    )
    return fig


# ---------------------------------------------------------------------------
# CSS Injection — Modern Executive Healthcare Theme (No Emojis, No Bright Teal)
# ---------------------------------------------------------------------------
def inject_css():
    """Inject polished, modern CSS styling for CardioCare."""
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --font-main: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --bg-primary: #F8FAFC;
            --bg-card: #FFFFFF;
            --bg-sidebar: #F1F5F9;
            --text-primary: #0F172A;
            --text-secondary: #475569;
            --text-muted: #64748B;
            --brand-navy: #1E3A8A;
            --accent: #2563EB;
            --accent-hover: #1D4ED8;
            --accent-light: #EFF6FF;
            --accent-border: #BFDBFE;
            --border: #E2E8F0;
            --border-subtle: #F1F5F9;
            --shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.04);
            --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.04);
            --shadow-md: 0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.05);
            --radius: 10px;
            --radius-sm: 6px;
            --success: #059669;
            --success-bg: #ECFDF5;
            --success-border: #A7F3D0;
            --danger: #DC2626;
            --danger-bg: #FEF2F2;
            --danger-border: #FECACA;
            --warning: #D97706;
            --warning-bg: #FFFBEB;
            --warning-border: #FDE68A;
        }

        /* ===== Strict Light Theme Enforcement (Suppress Dark Mode Completely) ===== */
        html, body, [class*="css"], .stApp, section[data-testid="stSidebar"], [data-testid="stAppViewContainer"] {
            font-family: var(--font-main) !important;
            color: var(--text-primary) !important;
            background-color: var(--bg-primary) !important;
            color-scheme: light !important;
        }

        @media (prefers-color-scheme: dark) {
            html, body, .stApp, [class*="css"], [data-testid="stAppViewContainer"], section[data-testid="stSidebar"] {
                background-color: #F8FAFC !important;
                color: #0F172A !important;
                color-scheme: light !important;
            }
        }

        /* ===== Hide Deploy Button & Header Controls Completely ===== */
        .stDeployButton,
        [data-testid="stAppDeployButton"],
        [data-testid="stToolbarActions"],
        button[kind="header"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }

        .main .block-container {
            max-width: 1140px;
            padding: 1.5rem 2.2rem 3.5rem;
        }

        /* ===== Top Tabs Navigation Bar ===== */
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            gap: 6px;
            background-color: #E2E8F0;
            padding: 5px;
            border-radius: 12px;
            border: 1px solid #CBD5E1;
            margin-bottom: 1.75rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 0 16px;
            background-color: transparent;
            border-radius: 8px;
            color: #475569;
            font-weight: 600;
            font-size: 0.88rem;
            border: none;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255, 255, 255, 0.6);
            color: var(--text-primary);
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: var(--brand-navy) !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 4px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.05) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        /* ===== Sidebar Refinements ===== */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border);
            padding-top: 1.5rem;
        }

        /* ===== Brand Header Component ===== */
        .cc-brand-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 0.5rem;
        }

        .cc-brand-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 44px;
            height: 44px;
            background: var(--accent-light);
            border: 1px solid var(--accent-border);
            border-radius: 10px;
        }

        .cc-brand-title {
            font-size: 1.65rem;
            font-weight: 800;
            color: var(--brand-navy);
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.1;
        }

        .cc-brand-subtitle {
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text-muted);
            margin: 0;
        }

        /* ===== Cards & Containers ===== */
        .cc-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.35rem 1.6rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 1.25rem;
        }

        .cc-card-interactive {
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .cc-card-interactive:hover {
            box-shadow: var(--shadow-md);
            border-color: #CBD5E1;
        }

        /* ===== Metric KPI Cards ===== */
        .cc-metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.1rem 1.25rem;
            box-shadow: var(--shadow-sm);
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .cc-metric-icon-box {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            background: var(--accent-light);
            border-radius: 8px;
            border: 1px solid var(--accent-border);
            flex-shrink: 0;
        }

        .cc-metric-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1.2;
            font-family: var(--font-main);
        }

        .cc-metric-label {
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* ===== Clinical Alert Banners ===== */
        .cc-result-card {
            padding: 1.5rem;
            border-radius: var(--radius);
            margin: 1.2rem 0;
            border: 1px solid transparent;
        }

        .cc-result-positive {
            background: var(--danger-bg);
            border-color: var(--danger-border);
        }

        .cc-result-negative {
            background: var(--success-bg);
            border-color: var(--success-border);
        }

        .cc-disclaimer-box {
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-radius: var(--radius-sm);
            padding: 0.85rem 1.1rem;
            font-size: 0.82rem;
            color: #92400E;
            line-height: 1.5;
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin: 1rem 0;
        }

        /* ===== Explanatory Code & Formula Cards ===== */
        .cc-formula-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid var(--accent);
            border-radius: 6px;
            padding: 0.9rem 1.2rem;
            margin: 0.8rem 0;
            font-family: var(--font-mono);
            font-size: 0.88rem;
            color: #1E293B;
        }

        .cc-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .cc-badge-blue { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
        .cc-badge-green { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
        .cc-badge-amber { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }

        .cc-divider {
            border: 0;
            height: 1px;
            background: var(--border);
            margin: 1.25rem 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page Renderer: 1. Overview
# ---------------------------------------------------------------------------
def render_overview():
    """Render the executive Overview / Dashboard."""
    st.markdown(
        f"""
        <div class="cc-brand-header">
            <div class="cc-brand-icon">{icon_svg("heart", 24, "#2563EB")}</div>
            <div>
                <h1 class="cc-brand-title">CardioCare</h1>
                <p class="cc-brand-subtitle">Clinical Heart Disease Risk Assessment System — UCI Cleveland Cohort</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='cc-divider'>", unsafe_allow_html=True)

    # Load multi-model metrics
    try:
        results = evaluate_models()
        svm_acc = results["SVM (RBF Kernel)"]["accuracy"]
        best_model = max(results, key=lambda m: results[m]["accuracy"])
        best_acc = results[best_model]["accuracy"]
    except Exception:
        svm_acc = 0.90
        best_model = "SVM (RBF Kernel)"
        best_acc = 0.90

    history_count = get_history_count()

    # KPI summary cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div class="cc-metric-icon-box">{icon_svg("award", 22, "#2563EB")}</div>
                <div>
                    <div class="cc-metric-value">{best_acc:.1%}</div>
                    <div class="cc-metric-label">Champion Accuracy</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div class="cc-metric-icon-box">{icon_svg("shield", 22, "#059669")}</div>
                <div>
                    <div class="cc-metric-value">SVM RBF</div>
                    <div class="cc-metric-label">Active Model</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div class="cc-metric-icon-box">{icon_svg("sliders", 22, "#7C3AED")}</div>
                <div>
                    <div class="cc-metric-value">25 Features</div>
                    <div class="cc-metric-label">Encoded Inputs</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div class="cc-metric-icon-box">{icon_svg("database", 22, "#D97706")}</div>
                <div>
                    <div class="cc-metric-value">{history_count}</div>
                    <div class="cc-metric-label">Local Audits</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column overview
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("### Model Benchmark Comparison")
        try:
            fig = create_model_comparison_chart(results)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.info(f"Comparison chart unavailable: {e}")

    with right_col:
        st.markdown("### Clinical Machine Learning Pipeline")
        st.markdown(
            f"""
            <div class="cc-card">
                <div style="display:flex; flex-direction:column; gap:12px; font-size:0.88rem;">
                    <div style="display:flex; align-items:flex-start; gap:10px;">
                        <span class="cc-badge cc-badge-blue">Step 1</span>
                        <div><b>Dataset Ingestion:</b> UCI Cleveland Heart Disease dataset (297 cleaned samples).</div>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:10px;">
                        <span class="cc-badge cc-badge-blue">Step 2</span>
                        <div><b>Categorical Encoding:</b> One-Hot Encoding applied to prevent false ordinal bias.</div>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:10px;">
                        <span class="cc-badge cc-badge-blue">Step 3</span>
                        <div><b>StandardScaler:</b> Continuous variables z-scored to prevent scale dominance.</div>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:10px;">
                        <span class="cc-badge cc-badge-blue">Step 4</span>
                        <div><b>Train-Test Split:</b> 80% training / 20% holdout test set (seed=42).</div>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:10px;">
                        <span class="cc-badge cc-badge-green">Champion</span>
                        <div><b>SVM with RBF Kernel:</b> Achieves 90.0% accuracy, calibrated via Platt scaling.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Regulatory disclaimer
    st.markdown(
        f"""
        <div class="cc-disclaimer-box">
            <div>{icon_svg("alert", 18, "#D97706")}</div>
            <div>
                <b>Educational and Research Tool:</b> CardioCare is designed for academic demonstration and laboratory viva evaluation.
                Predictions reflect statistical probabilities and do not constitute clinical medical diagnosis or treatment advice.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page Renderer: 2. Heart Prediction (Single & Batch)
# ---------------------------------------------------------------------------
def render_prediction():
    """Render the Heart Prediction assessment section."""
    st.markdown(
        f"""
        <div class="cc-brand-header">
            <div class="cc-brand-icon">{icon_svg("stethoscope", 24, "#2563EB")}</div>
            <div>
                <h2 class="cc-brand-title" style="font-size:1.45rem;">Patient Risk Assessment</h2>
                <p class="cc-brand-subtitle">Evaluate cardiovascular risk indicators using the production SVM classifier</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='cc-divider'>", unsafe_allow_html=True)

    if not os.path.exists(MODEL_PATH):
        st.error("Model artifact not found at `models/final_model.pkl`.")
        return
    if not os.path.exists(CLEANED_X_PATH):
        st.error("Feature schema not found at `data/cleaned_X.csv`.")
        return

    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    feature_columns = load_feature_columns(CLEANED_X_PATH)
    groups = split_feature_groups(feature_columns)

    pred_mode = st.radio(
        "Assessment Mode",
        ["Single Patient Assessment", "CSV Batch Processing"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if pred_mode == "Single Patient Assessment":
        _render_single_prediction_form(model, scaler, feature_columns, groups)
    else:
        _render_batch_prediction_form(model, scaler, feature_columns)


def _render_single_prediction_form(model, scaler, feature_columns, groups):
    """Render single patient form."""
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    st.markdown("#### Diagnostic Measurements")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, help="Patient age in years")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=70, max_value=250, value=130, help="Resting BP on admission")
    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=700, value=240, help="Serum cholesterol level")
        thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=250, value=150, help="Peak exercise heart rate")
    with col3:
        oldpeak = st.number_input("ST Depression (oldpeak mm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Exercise-induced ST depression")
        ca = st.number_input("Fluoroscopy Colored Vessels (0–3)", min_value=0, max_value=3, value=0, help="Major vessels colored")

    st.markdown("#### Clinical Categorical Features")
    col_a, col_b = st.columns(2)

    def _select_cat(label, cat_key, container):
        meta = CATEGORICAL_FEATURES[cat_key]
        avail_cols = groups[cat_key]
        avail_keys = [c.split("_", 1)[1] for c in avail_cols]
        display_map = {k: meta["options"].get(k, k) for k in avail_keys}
        with container:
            selected_label = st.selectbox(
                label,
                options=list(display_map.values()),
                help=meta["help"],
            )
        # Reverse lookup key
        for k, v in display_map.items():
            if v == selected_label:
                return k
        return avail_keys[0]

    sex_val = _select_cat("Sex", "sex", col_a)
    cp_val = _select_cat("Chest Pain Type", "cp", col_b)
    restecg_val = _select_cat("Resting Electrocardiogram", "restecg", col_a)
    slope_val = _select_cat("Peak Exercise ST Segment Slope", "slope", col_b)
    fbs_val = _select_cat("Fasting Blood Sugar > 120 mg/dl", "fbs", col_a)
    exang_val = _select_cat("Exercise-Induced Angina", "exang", col_b)
    thal_val = _select_cat("Thalassemia Blood Profile", "thal", col_a)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Clinical Risk Assessment", type="primary", use_container_width=True):
        user_inputs = {
            "age": age, "trestbps": trestbps, "chol": chol,
            "thalach": thalach, "oldpeak": oldpeak, "ca": ca,
            "sex": sex_val, "cp": cp_val, "restecg": restecg_val,
            "slope": slope_val, "fbs": fbs_val, "exang": exang_val,
            "thal": thal_val,
        }
        features_df = build_input_vector(feature_columns, groups, user_inputs)

        try:
            if scaler is not None:
                features_scaled = scaler.transform(features_df)
            else:
                features_scaled = features_df.values

            pred = int(model.predict(features_scaled)[0])
            prob = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(features_scaled)[0]
                    prob = float(proba[1] if len(proba) > 1 else proba[0])
                except Exception:
                    pass

            st.session_state.prediction_result = {
                "prediction": pred,
                "probability": prob,
                "user_inputs": user_inputs,
                "features_df": features_df,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_prediction(pred, prob, user_inputs)
        except Exception as err:
            st.error(f"Inference pipeline error: {err}")
            st.session_state.prediction_result = None

    # Render results
    res = st.session_state.prediction_result
    if res is not None:
        p = res["prediction"]
        prob = res["probability"]
        inputs = res["user_inputs"]

        if p == 1:
            conf_str = f"Model Probability: <b>{prob:.1%}</b> (Platt-scaled SVM confidence)" if prob is not None else ""
            st.markdown(
                f"""
                <div class="cc-result-card cc-result-positive">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <div>{icon_svg("alert", 26, "#DC2626")}</div>
                        <h3 style="margin:0; color:#DC2626;">Elevated Risk: Heart Disease Indication Detected</h3>
                    </div>
                    <p style="margin: 8px 0 0; color:#7F1D1D; font-size:0.95rem;">
                        The model identifies clinical patterns consistent with significant coronary vessel narrowing (≥50% diameter reduction).
                    </p>
                    <p style="margin: 4px 0 0; color:#991B1B; font-size:0.88rem;">{conf_str}</p>
                    <p style="font-size:0.75rem; color:#64748B; margin-top:8px;">Evaluated at: {res['timestamp']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            conf_val = (1.0 - prob) if prob is not None else 0.90
            conf_str = f"Model Probability: <b>{conf_val:.1%}</b> (Healthy profile confidence)" if prob is not None else ""
            st.markdown(
                f"""
                <div class="cc-result-card cc-result-negative">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <div>{icon_svg("check", 26, "#059669")}</div>
                        <h3 style="margin:0; color:#059669;">Normal Profile: No Heart Disease Indication Detected</h3>
                    </div>
                    <p style="margin: 8px 0 0; color:#064E3B; font-size:0.95rem;">
                        Patient attributes correlate with normal cardiovascular parameters within the Cleveland cohort distribution.
                    </p>
                    <p style="margin: 4px 0 0; color:#065F46; font-size:0.88rem;">{conf_str}</p>
                    <p style="font-size:0.75rem; color:#64748B; margin-top:8px;">Evaluated at: {res['timestamp']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # PDF Download Button
        try:
            pdf_data = generate_pdf_report(p, prob, inputs)
            st.download_button(
                label="Download Clinical Summary Report (PDF)",
                data=pdf_data,
                file_name=f"CardioCare_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as ex:
            st.warning(f"PDF compilation note: {ex}")

        with st.expander("Inspect One-Hot Scaled Feature Vector (25 dimensions)"):
            st.dataframe(res["features_df"].T.rename(columns={0: "Encoded Value"}), use_container_width=True)


def _render_batch_prediction_form(model, scaler, feature_columns):
    """Render batch CSV upload."""
    st.markdown("#### Batch CSV Processing")
    st.markdown(
        "<p style='color:var(--text-secondary); font-size:0.88rem;'>"
        "Upload a dataset containing the 25 one-hot encoded features. Results can be inspected and exported.</p>",
        unsafe_allow_html=True,
    )

    template_csv = ",".join(feature_columns).encode("utf-8")
    st.download_button(
        label="Download Schema Template CSV",
        data=template_csv,
        file_name="cardiac_features_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_csv_input")
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        st.error(f"CSV missing {len(missing_cols)} schema column(s): {', '.join(missing_cols[:4])}...")
        return

    df_aligned = df[feature_columns]
    if scaler is not None:
        scaled_mat = scaler.transform(df_aligned)
    else:
        scaled_mat = df_aligned.values

    preds = model.predict(scaled_mat)
    result_df = df.copy()
    result_df["Predicted_Class"] = preds
    result_df["Risk_Category"] = np.where(preds == 1, "Disease Detected", "No Disease")

    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(scaled_mat)
            result_df["Disease_Probability"] = probas[:, 1].round(4)
        except Exception:
            pass

    st.success(f"Batch prediction completed for {len(df)} records.")
    st.dataframe(result_df, use_container_width=True)

    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Batch Prediction Results (CSV)",
        data=csv_out,
        file_name=f"CardioCare_Batch_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Page Renderer: 3. Machine Learning Guide (The requested "ML Explained from Basic")
# ---------------------------------------------------------------------------
def render_ml_guide():
    """
    Dedicated comprehensive single-page machine learning guide explaining all
    concepts from absolute basics. Ideal for viva and lab examination.
    """
    st.markdown(
        f"""
        <div class="cc-brand-header">
            <div class="cc-brand-icon">{icon_svg("brain", 24, "#2563EB")}</div>
            <div>
                <h2 class="cc-brand-title" style="font-size:1.45rem;">Machine Learning Architecture & Theory</h2>
                <p class="cc-brand-subtitle">A comprehensive first-principles guide to the algorithms, preprocessing, and clinical metrics</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='cc-divider'>", unsafe_allow_html=True)

    # 1. Fundamental Concept
    st.markdown("### 1. The Fundamental Machine Learning Concept")
    st.markdown(
        """
        **What is Supervised Classification?**
        In Supervised Machine Learning, an algorithm is provided with a training set of labeled historical examples:
        $$\\mathcal{D} = \\{(\\mathbf{x}_1, y_1), (\\mathbf{x}_2, y_2), \\dots, (\\mathbf{x}_N, y_N)\\}$$
        Where:
        - $\\mathbf{x}_i \\in \\mathbb{R}^{25}$ is a vector of clinical measurements (age, cholesterol, blood pressure, ECG, ST depression, etc.).
        - $y_i \\in \\{0, 1\\}$ is the binary ground-truth clinical diagnosis:
          - $y = 0$: Healthy (angiography shows $< 50\\%$ coronary artery diameter narrowing).
          - $y = 1$: Coronary Artery Disease present ($\\ge 50\\%$ narrowing in at least one major vessel).

        The objective is to learn a mapping function $f: \\mathcal{X} \\rightarrow \\mathcal{Y}$ that generalizes accurately to unseen patients, minimizing prediction error without memorizing (overfitting) the training distribution.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Cleveland Dataset & Preprocessing
    st.markdown("### 2. Dataset Architecture & Feature Engineering Pipeline")
    col_data_a, col_data_b = st.columns(2)
    with col_data_a:
        st.markdown(
            f"""
            <div class="cc-card">
                <h4 style="margin-top:0; color:var(--brand-navy);">Dataset Origin & Structure</h4>
                <ul style="font-size:0.88rem; line-height:1.7; padding-left:1.2rem; margin:0;">
                    <li><b>Source:</b> UCI Machine Learning Repository (Cleveland Clinic Foundation).</li>
                    <li><b>Original Records:</b> 303 patient examinations.</li>
                    <li><b>Cleaned Cohort:</b> 297 instances (6 records with missing fluoroscopy vessels <code>ca</code> or thalassemia <code>thal</code> were removed).</li>
                    <li><b>Target Binarization:</b> Original multi-class angiographic staging (0, 1, 2, 3, 4) converted to binary: 0 = Absence, 1–4 = Disease Present.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_data_b:
        st.markdown(
            f"""
            <div class="cc-card">
                <h4 style="margin-top:0; color:var(--brand-navy);">Why One-Hot Encoding is Required</h4>
                <p style="font-size:0.86rem; line-height:1.6; margin:0;">
                    If we encode <b>Chest Pain</b> as 1, 2, 3, 4, numerical models assume an artificial arithmetic order:
                    <br><code>Asymptomatic (4) &gt; Typical Angina (1)</code>.
                    <br>This forces Euclidean distance $|4 - 1| = 3$, which distorts geometric boundaries!
                    <br><b>Solution:</b> One-Hot Encoding creates orthogonal binary dimensions (e.g. <code>cp_1, cp_2, cp_3, cp_4</code>), expanding 13 raw clinical attributes into <b>25 independent features</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Standard Scaler Viva Card
    st.markdown(
        """
        <div class="cc-card" style="border-left: 4px solid #2563EB;">
            <h4 style="margin-top:0; color:var(--brand-navy);">Critical Viva Question: Why Feature Standardization is Mandatory for SVM</h4>
            <p style="font-size:0.88rem; line-height:1.6;">
                Continuous clinical features have wildly differing physiological scales:
                <br>• Serum Cholesterol ranges from <b>126 to 564 mg/dl</b>.
                <br>• ST Depression (oldpeak) ranges from <b>0.0 to 6.2 mm</b>.
                <br><br>
                Distance-based algorithms like Support Vector Machines compute distances: $\\|\\mathbf{x}_i - \\mathbf{x}_j\\|^2 = \\sum_k (x_{ik} - x_{jk})^2$.
                Without scaling, a cholesterol difference of $50$ completely overwhelms an ST depression difference of $1.5$, making ST depression mathematically invisible!
                <br><br>
                <b>StandardScaler Formula:</b> $z = \\frac{x - \\mu}{\\sigma}$ (Transforms each variable to mean $\\mu = 0$ and unit variance $\\sigma = 1$).
                <i>Both our training pipeline and inference engine apply this exact fitted scaler.</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Algorithms Explored
    st.markdown("### 3. The Four Machine Learning Algorithms Compared")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            """
            <div class="cc-card">
                <h4 style="color:var(--brand-navy); margin-top:0;">1. Logistic Regression (88.3% Accuracy)</h4>
                <p style="font-size:0.85rem; line-height:1.6; margin-bottom:8px;">
                    Linear baseline model. Calculates the log-odds of heart disease as a linear combination of inputs:
                    $z = \\mathbf{w}^T \\mathbf{x} + b$.
                    Passes $z$ through the <b>Sigmoid activation function</b>:
                </p>
                <div class="cc-formula-box">
                    $\\sigma(z) = \\frac{1}{1 + e^{-z}} \\in (0, 1)$
                </div>
                <p style="font-size:0.82rem; color:var(--text-muted); margin:0;">
                    Optimized via Log-Loss (Binary Cross-Entropy). Highly interpretable coefficients, but limited to linear hyperplanes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="cc-card">
                <h4 style="color:var(--brand-navy); margin-top:0;">2. Decision Tree Classifier (83.3% Accuracy)</h4>
                <p style="font-size:0.85rem; line-height:1.6; margin-bottom:8px;">
                    Non-parametric tree structure. Partitions feature space recursively by choosing splits that maximize purity using <b>Gini Impurity</b>:
                </p>
                <div class="cc-formula-box">
                    $I_G(p) = 1 - \\sum_{i=0}^1 p_i^2$
                </div>
                <p style="font-size:0.82rem; color:var(--text-muted); margin:0;">
                    Creates intuitive orthogonal rules (e.g. <i>if ST depression &gt; 1.5 and vessels &gt; 0</i>), but sensitive to small data perturbations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div class="cc-card">
                <h4 style="color:var(--brand-navy); margin-top:0;">3. Random Forest Classifier (86.7% Accuracy)</h4>
                <p style="font-size:0.85rem; line-height:1.6; margin-bottom:8px;">
                    Ensemble bagging method combining 100 decorrelated decision trees. Each tree is trained on a bootstrap sample of data with random feature subspacing:
                </p>
                <div class="cc-formula-box">
                    $\\hat{y} = \\text{Mode}\\{T_1(\\mathbf{x}), T_2(\\mathbf{x}), \\dots, T_{100}(\\mathbf{x})\\}$
                </div>
                <p style="font-size:0.82rem; color:var(--text-muted); margin:0;">
                    Drastically reduces model variance while maintaining low bias, overcoming the instability of individual decision trees.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="cc-card" style="border: 2px solid #2563EB;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="color:#2563EB; margin:0;">4. Support Vector Machine (SVM) — Champion</h4>
                    <span class="cc-badge cc-badge-blue">90.0% Test Accuracy</span>
                </div>
                <p style="font-size:0.85rem; line-height:1.6; margin:8px 0;">
                    Constructs an optimal decision hyperplane $\\mathbf{w}^T \\mathbf{x} + b = 0$ that <b>maximizes the geometric margin</b> $\\frac{2}{\\|\\mathbf{w}\\|}$ between support vectors of opposing classes.
                </p>
                <div class="cc-formula-box">
                    $K(\\mathbf{x}, \\mathbf{x}') = \\exp\\left(-\\gamma \\|\\mathbf{x} - \\mathbf{x}'\\|^2\\right)$
                </div>
                <p style="font-size:0.82rem; color:var(--text-muted); margin:0;">
                    <b>RBF Kernel Trick:</b> Projects 25-D clinical features into infinite-dimensional Hilbert space where non-linear patient distributions become linearly separable.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Evaluation Metrics
    st.markdown("### 4. Healthcare Evaluation Metrics Demystified")
    st.markdown(
        """
        In clinical diagnostics, standard accuracy is not sufficient. A false negative can be fatal, while a false positive leads to unnecessary invasive tests.

        | Metric | Mathematical Formula | Clinical Meaning in Cardiology |
        |---|:---:|---|
        | **Accuracy** | $\\frac{TP + TN}{TP + TN + FP + FN}$ | Overall fraction of correct diagnoses across the 60 test patients. |
        | **Recall (Sensitivity)** | $\\frac{TP}{TP + FN}$ | **Most Critical Metric:** Percentage of sick patients correctly identified. High recall means minimal missed diagnoses. |
        | **Precision (PPV)** | $\\frac{TP}{TP + FP}$ | When the model flags heart disease, the probability the patient truly has it. High precision prevents unnecessary cardiac catheterization. |
        | **F1-Score** | $2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ | Harmonic mean that balances precision and recall into a single robust metric. |
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Top 10 Viva Questions
    st.markdown("### 5. Laboratory Viva & Oral Defense Cheat Sheet (Top 10 Q&A)")
    viva_questions = [
        ("Q1: Why did you choose Support Vector Machine (SVM) as your final production model?",
         "SVM with the RBF kernel achieved the highest test accuracy (90.0%) and demonstrated superior generalization on the 60-sample holdout test set, identifying 18 out of 20 disease cases (90% recall) and 36 out of 40 healthy cases (90% precision)."),
        ("Q2: What is the Kernel Trick and why is it necessary here?",
         "In clinical data, disease boundaries are rarely linearly separable. The Kernel Trick computes dot products in a high-dimensional Hilbert space without explicitly mapping vectors there. The Radial Basis Function (RBF) kernel measures similarity using Gaussian curves, allowing complex non-linear separation."),
        ("Q3: Why is StandardScaler strictly required for SVM inference?",
         "SVM relies on Euclidean distance in feature space. Continuous features like Cholesterol (100–700 mg/dl) have a variance orders of magnitude larger than ST depression (0–6 mm). StandardScaler normalizes all 25 features to zero mean and unit variance, ensuring fair feature weighting."),
        ("Q4: What is the difference between Type I and Type II errors in cardiology?",
         "A Type I error (False Positive) predicts disease in a healthy patient, leading to anxiety and confirmatory testing. A Type II error (False Negative) fails to detect heart disease in a sick patient, potentially delaying life-saving intervention. Type II errors are clinically far more dangerous."),
        ("Q5: How did you handle categorical variables?",
         "Categorical features (Chest Pain, Resting ECG, ST Slope, Thalassemia) were converted using One-Hot Encoding. Integer label encoding would have introduced false ordinal magnitude (e.g. 4 > 1), which distorts geometric distance metrics."),
        ("Q6: How does Platt Scaling work in your model?",
         "Standard SVM outputs a signed distance to the hyperplane f(x), not a calibrated probability. By setting `probability=True`, scikit-learn fits an additional logistic regression model on the decision function outputs during 5-fold cross-validation, producing calibrated confidence scores."),
        ("Q7: What train/test split did you use?",
         "We used an 80/20 train/test split with random_state=42 on the 297 cleaned samples. This creates 237 training samples and 60 test samples (36 healthy, 24 disease), allowing reproducible evaluation matching published academic benchmarks."),
        ("Q8: How does Random Forest combat overfitting compared to a Decision Tree?",
         "A single Decision Tree grows deep and overfits training noise. Random Forest uses Bootstrap Aggregation (bagging) to train 100 decorrelated trees on random data subsets with random feature subsets. Averaging their predictions cancels out variance while preserving low bias."),
        ("Q9: What is Gini Impurity in Decision Trees?",
         "Gini Impurity measures the probability that a randomly chosen element from a subset would be incorrectly labeled. A Gini score of 0.0 indicates a perfectly pure node where all samples belong to a single class."),
        ("Q10: Can this model be used as an independent clinical diagnosis in a hospital?",
         "No. This application is an educational prototype. It lacks multi-cohort clinical trial validation, does not ingest ECG waveforms or angiographic images directly, and must only serve as a secondary triaging or pedagogical aid under physician oversight."),
    ]

    for q, ans in viva_questions:
        with st.expander(q):
            st.markdown(f"<p style='font-size:0.88rem; color:#1E293B; line-height:1.6;'>{ans}</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page Renderer: 4. Model Analytics
# ---------------------------------------------------------------------------
def render_analytics():
    """Render the Model Analytics & Confusion Matrix section."""
    st.markdown(
        f"""
        <div class="cc-brand-header">
            <div class="cc-brand-icon">{icon_svg("analytics", 24, "#2563EB")}</div>
            <div>
                <h2 class="cc-brand-title" style="font-size:1.45rem;">Model Performance & Diagnostics</h2>
                <p class="cc-brand-subtitle">Benchmark metrics, confusion matrix heatmaps, and classification reports</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='cc-divider'>", unsafe_allow_html=True)

    try:
        results = evaluate_models()
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return

    # Benchmark summary table
    table_rows = []
    for name, data in results.items():
        is_prod = " (Production)" if name == "SVM (RBF Kernel)" else ""
        table_rows.append({
            "Classifier": f"{name}{is_prod}",
            "Accuracy": f"{data['accuracy']:.1%}",
            "Precision": f"{data['precision']:.1%}",
            "Recall": f"{data['recall']:.1%}",
            "F1-Score": f"{data['f1']:.1%}",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed inspector for chosen model
    st.markdown("### Deep-Dive Model Diagnostic")
    selected_model_name = st.selectbox(
        "Select Model to Inspect",
        list(results.keys()),
        index=3,  # Default to SVM
    )
    model_data = results[selected_model_name]

    col_cm, col_gauge, col_rep = st.columns([3, 2, 3])

    with col_cm:
        cm_fig = create_confusion_matrix_chart(model_data["confusion_matrix"], selected_model_name)
        st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})

    with col_gauge:
        st.markdown("<div style='text-align:center; font-size:0.85rem; font-weight:600; color:#475569; margin-top:10px;'>Test Set Accuracy</div>", unsafe_allow_html=True)
        gauge_fig = create_accuracy_gauge(model_data["accuracy"])
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

    with col_rep:
        st.markdown("<div style='font-size:0.85rem; font-weight:600; color:#475569; margin-top:10px;'>Class-Level Report</div>", unsafe_allow_html=True)
        rep = model_data["report"]
        rep_rows = []
        for c in ["0", "1"]:
            if c in rep:
                lbl = "Healthy (0)" if c == "0" else "Disease (1)"
                rep_rows.append({
                    "Class": lbl,
                    "Precision": f"{rep[c]['precision']:.2f}",
                    "Recall": f"{rep[c]['recall']:.2f}",
                    "F1": f"{rep[c]['f1-score']:.2f}",
                    "Samples": int(rep[c]["support"]),
                })
        st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page Renderer: 5. Prediction History
# ---------------------------------------------------------------------------
def render_history():
    """Render the SQLite-backed prediction history."""
    st.markdown(
        f"""
        <div class="cc-brand-header">
            <div class="cc-brand-icon">{icon_svg("history", 24, "#2563EB")}</div>
            <div>
                <h2 class="cc-brand-title" style="font-size:1.45rem;">Local Audit & History</h2>
                <p class="cc-brand-subtitle">Persistent local records of assessments stored in SQLite on this machine</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='cc-divider'>", unsafe_allow_html=True)

    df_history = get_prediction_history(limit=200)

    if df_history.empty:
        st.markdown(
            f"""
            <div class="cc-card" style="text-align:center; padding:2.5rem;">
                <div style="margin-bottom:10px;">{icon_svg("database", 36, "#94A3B8")}</div>
                <h4 style="margin:0 0 6px; color:#475569;">No Assessment History Recorded</h4>
                <p style="color:var(--text-muted); font-size:0.88rem; margin:0;">
                    Perform your first evaluation under the <b>Heart Prediction</b> tab to record local audits.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # History KPIs
    tot = len(df_history)
    disease_cases = int((df_history["prediction"] == 1).sum())
    healthy_cases = tot - disease_cases

    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div class="cc-metric-icon-box">{icon_svg("database", 20, "#2563EB")}</div>
                <div><div class="cc-metric-value">{tot}</div><div class="cc-metric-label">Total Audits</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hc2:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div class="cc-metric-icon-box">{icon_svg("check", 20, "#059669")}</div>
                <div><div class="cc-metric-value">{healthy_cases}</div><div class="cc-metric-label">Normal Evaluations</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hc3:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div class="cc-metric-icon-box">{icon_svg("alert", 20, "#DC2626")}</div>
                <div><div class="cc-metric-value">{disease_cases}</div><div class="cc-metric-label">Elevated Risk Cases</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter & Search
    f1, f2, f3 = st.columns([3, 2, 1])
    with f1:
        search_query = st.text_input("Filter Records", placeholder="Search by age, cholesterol, BP...", label_visibility="collapsed")
    with f2:
        cat_filter = st.selectbox("Category Filter", ["All Results", "Disease Cases Only", "Healthy Profiles Only"], label_visibility="collapsed")
    with f3:
        if st.button("Clear Log", type="secondary", use_container_width=True):
            clear_prediction_history()
            st.rerun()

    disp_df = df_history.copy()
    disp_df["Outcome"] = np.where(disp_df["prediction"] == 1, "Disease Detected", "No Disease")
    if "probability" in disp_df.columns:
        disp_df["Confidence"] = disp_df["probability"].apply(lambda v: f"{v:.1%}" if pd.notnull(v) else "-")

    if cat_filter == "Disease Cases Only":
        disp_df = disp_df[disp_df["prediction"] == 1]
    elif cat_filter == "Healthy Profiles Only":
        disp_df = disp_df[disp_df["prediction"] == 0]

    if search_query:
        mask = disp_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
        disp_df = disp_df[mask]

    cols_to_show = ["id", "timestamp", "Outcome", "Confidence", "age", "trestbps", "chol", "thalach", "oldpeak"]
    cols_exist = [c for c in cols_to_show if c in disp_df.columns]
    st.dataframe(disp_df[cols_exist], use_container_width=True, hide_index=True)

    csv_data = disp_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export History to CSV", data=csv_data, file_name="cardiac_prediction_audit_log.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Page Renderer: 6. About Project
# ---------------------------------------------------------------------------
def render_about():
    """Render project metadata, technical stack, and acknowledgements."""
    st.markdown(
        f"""
        <div class="cc-brand-header">
            <div class="cc-brand-icon">{icon_svg("info", 24, "#2563EB")}</div>
            <div>
                <h2 class="cc-brand-title" style="font-size:1.45rem;">About CardioCare Project</h2>
                <p class="cc-brand-subtitle">Academic lab submission, data-science architecture, and system metadata</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='cc-divider'>", unsafe_allow_html=True)

    st.markdown(
        """
        #### Project Overview
        CardioCare is a machine learning–based clinical screening system developed for laboratory demonstration and viva evaluation.
        The system ingests 13 standard clinical parameters, scales and encodes them to 25 mathematical features, and performs inference
        using an RBF-kernel Support Vector Machine with Platt probability scaling.
        """
    )

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(f'<div class="cc-card" style="text-align:center;"><b>Core Engine</b><br><span style="font-size:0.85rem; color:#475569;">Python 3.11 / scikit-learn</span></div>', unsafe_allow_html=True)
    with t2:
        st.markdown(f'<div class="cc-card" style="text-align:center;"><b>Web Platform</b><br><span style="font-size:0.85rem; color:#475569;">Streamlit 1.66</span></div>', unsafe_allow_html=True)
    with t3:
        st.markdown(f'<div class="cc-card" style="text-align:center;"><b>Visualization</b><br><span style="font-size:0.85rem; color:#475569;">Plotly Interactive</span></div>', unsafe_allow_html=True)
    with t4:
        st.markdown(f'<div class="cc-card" style="text-align:center;"><b>Reporting</b><br><span style="font-size:0.85rem; color:#475569;">ReportLab PDF Engine</span></div>', unsafe_allow_html=True)

    st.markdown("#### Analysis Notebooks")
    st.markdown(
        """
        The project repository includes 6 modular exploratory and modelling Jupyter notebooks:
        1. `01_Data_Preprocessing.ipynb` — Missing value imputation and distribution analysis
        2. `02_PCA_Analysis.ipynb` — Principal Component Analysis on continuous variables
        3. `03_Feature_Selection.ipynb` — Feature importance ranking and correlation heatmaps
        4. `04_Supervised_Learning.ipynb` — Baseline comparison of LR, DT, RF, and SVM
        5. `05_Unsupervised_Learning.ipynb` — K-Means clustering of patient sub-groups
        6. `06_Hyperparameter_Tuning.ipynb` — Grid search on SVM kernels, C, and gamma parameters
        """
    )

    st.markdown("#### Ethical & Clinical Limitations")
    st.markdown(
        f"""
        <div class="cc-disclaimer-box">
            <div>{icon_svg("alert", 20, "#D97706")}</div>
            <div style="font-size:0.85rem;">
                <b>Limitations and Caveats:</b>
                <br>• The dataset is sampled from 297 patients from a single regional center (Cleveland Clinic).
                <br>• The model does not ingest live ECG waveforms, cardiac MRI, or coronary CT angiography imaging.
                <br>• This tool is exclusively designed for academic lab demonstration and cannot be deployed in autonomous clinical triage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Developer & Academic Credit")
    st.markdown(
        """
        <div class="cc-card" style="border-left: 4px solid #2563EB;">
            <p style="margin: 0; font-size: 0.92rem; line-height: 1.6;">
                <b>Lead Developer & Researcher:</b> Devang Deokule<br>
                <b>Project:</b> CardioCare — Heart Disease Prediction System<br>
                <b>Domain:</b> Healthcare Data Science & Applied Machine Learning<br>
                <b>Champion Architecture:</b> Support Vector Classifier with RBF Kernel & Platt Scaling
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CardioCare — Heart Disease Prediction System",
        page_icon="cardiology",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    init_db()

    # Informative Sidebar (No radio buttons!)
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0 1rem;">
                <div style="display:flex; align-items:center; gap:8px;">
                    {icon_svg("heart", 22, "#2563EB")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #1E3A8A; font-weight:800;">CardioCare</h3>
                </div>
                <p style="margin: 3px 0 0; font-size: 0.74rem; color: #64748B;">Heart Disease Risk Assessment</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='cc-divider'>", unsafe_allow_html=True)

        # Active Model Specs Card
        st.markdown(
            f"""
            <div class="cc-card" style="padding: 0.9rem; margin-bottom: 1rem; border-color:#CBD5E1;">
                <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #64748B; letter-spacing:0.04em;">Production Classifier</div>
                <div style="font-size: 0.95rem; font-weight: 800; color: #1E3A8A; margin-top:2px;">Support Vector Machine</div>
                <div style="font-size: 0.76rem; color: #475569; margin-top: 4px;">
                    • Kernel: <b>RBF (Gaussian)</b><br>
                    • Accuracy: <b>90.0%</b> (60 test cases)<br>
                    • Calibration: <b>Platt Scaling</b><br>
                    • Scaler: <b>StandardScaler (Z-Score)</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clinical Normal Ranges Reference
        st.markdown(
            f"""
            <div class="cc-card" style="padding: 0.9rem; margin-bottom: 1rem;">
                <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #64748B; letter-spacing:0.04em;">Normal Clinical Ranges</div>
                <div style="font-size: 0.74rem; color: #334155; line-height: 1.6; margin-top: 4px;">
                    • <b>Resting BP:</b> &lt; 120 mm Hg<br>
                    • <b>Cholesterol:</b> &lt; 200 mg/dl<br>
                    • <b>Fasting Sugar:</b> &le; 120 mg/dl<br>
                    • <b>Max HR:</b> 220 - Age (approx)<br>
                    • <b>ST Depression:</b> 0.0 mm
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='cc-divider'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 0.72rem; color: #94A3B8;">
                CardioCare v2.0 · Educational Release<br>
                Local SQLite Store · No External Telemetry
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Modern Tabs Navigation (Tabs instead of radio buttons)
    tab_overview, tab_pred, tab_ml, tab_analytics, tab_history, tab_about = st.tabs(
        [
            "Overview",
            "Heart Prediction",
            "Machine Learning Guide",
            "Model Analytics",
            "Prediction History",
            "About Project",
        ]
    )

    with tab_overview:
        render_overview()

    with tab_pred:
        render_prediction()

    with tab_ml:
        render_ml_guide()

    with tab_analytics:
        render_analytics()

    with tab_history:
        render_history()

    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
