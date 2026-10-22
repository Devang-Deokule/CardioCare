# CardioCare — Heart Disease Prediction System

A complete, end-to-end clinical machine learning application built on the UCI Cleveland Heart Disease dataset. Developed for academic data-science lab demonstration and viva evaluation.

---

## Highlights & Features

- **Tab-Based Navigation:** Clean, modern segmented tabs (`Overview`, `Heart Prediction`, `Machine Learning Guide`, `Model Analytics`, `Prediction History`, `About Project`).
- **Icons & Color Theme:** Crisp, professional SVG vector icons (no cartoon emojis) paired with an executive clinical navy/slate color palette.
- **Machine Learning Guide:** Comprehensive, first-principles explanation of supervised classification, the Cleveland dataset, one-hot encoding, feature standardization ($z = \frac{x-\mu}{\sigma}$), four core algorithms (Logistic Regression, Decision Tree, Random Forest, SVM with RBF kernel), evaluation metrics, and a Top 10 Viva Q&A cheat sheet.
- **Single & Batch Prediction:** Interactive patient assessment form with automatic One-Hot Encoding and fitted `StandardScaler` transformation. Supports CSV batch prediction.
- **Model Analytics & Heatmaps:** Multi-model benchmarking, interactive Plotly confusion matrices, and test accuracy gauges.
- **Local SQLite Audit Log:** Persistent, private record keeping stored in `data/prediction_history.db`.
- **PDF Report Generation:** Instant clinical summary report generation powered by ReportLab.

---

## Project Structure

```
Diabetes-Predictor/
├─ data/
│  ├─ heart_disease.csv            # Raw UCI Cleveland dataset (303 records)
│  ├─ cleaned_X.csv                # Cleaned, one-hot encoded features (297 samples, 25 columns)
│  ├─ clean_y.csv                  # Binarized target labels (0 = Healthy, 1 = Disease)
│  └─ prediction_history.db        # SQLite database storing local prediction audits
├─ models/
│  ├─ final_model.pkl              # Production SVC (RBF kernel, probability=True, 90.0% accuracy)
│  ├─ scaler.pkl                   # Fitted StandardScaler artifact
│  └─ final_model_original_backup.pkl # Backup of legacy model artifact
├─ notebooks/                      # Exploratory Data Analysis & Modeling Pipeline
│  ├─ 01_data_preprocessing.ipynb
│  ├─ 02_pca_analysis.ipynb
│  ├─ 03_feature_selection.ipynb
│  ├─ 04_supervised_learning.ipynb
│  ├─ 05_unsupervised_learning.ipynb
│  └─ 06_hyperparameter_tuning.ipynb
├─ results/
│  └─ evaluation_metrics.txt       # Benchmarking metrics
├─ UI/
│  └─ app.py                       # Main Streamlit web application
├─ requirements.txt                # Python dependencies
└─ README.md
```

---

## Machine Learning Pipeline Summary

1. **Dataset:** UCI Cleveland Heart Disease dataset (303 raw $\rightarrow$ 297 cleaned instances after removing 6 samples with missing values in `ca` and `thal`).
2. **Target Binarization:** Multi-class coronary angiographic diagnosis (0 to 4) binarized into 0 (healthy) vs $\ge 1$ (disease present).
3. **Categorical Encoding:** One-Hot Encoding applied to `sex`, `cp`, `restecg`, `slope`, `fbs`, `exang`, and `thal`, yielding 25 orthogonal feature dimensions.
4. **Feature Standardization:** `StandardScaler` fits $z = \frac{x - \mu}{\sigma}$ on training data to ensure Euclidean distance-based models (SVM) are not dominated by large-magnitude variables like Cholesterol.
5. **Model Champion:** Support Vector Machine (SVC) with Radial Basis Function (RBF) kernel and Platt scaling (`probability=True`), achieving **90.0% test accuracy** on an 80/20 train/test split.

---

## Installation & Setup

1. **Clone or Navigate to the Repository:**
   ```powershell
   cd "c:\Users\Devang\OneDrive\Desktop\DSL\Diabetes-Predictor"
   ```

2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```powershell
   python -m streamlit run UI/app.py
   ```

4. **Access the Web Interface:**
   Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Developer & Author

**Devang Deokule**  
*Lead Developer & Machine Learning Researcher*  
- GitHub: [@Devang-Deokule](https://github.com/Devang-Deokule)  
- Project: CardioCare — Clinical Heart Disease Prediction System  
- Academic Lab & Viva Evaluation

---

## Academic & Medical Disclaimer

This project is developed exclusively for educational demonstration and university laboratory evaluation. It does NOT provide certified medical diagnoses, clinical guidance, or autonomous triage. All predictions represent statistical estimates.
