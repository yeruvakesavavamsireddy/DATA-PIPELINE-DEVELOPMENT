import pandas as pd
import numpy as np
import warnings
from io import StringIO
 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
warnings.filterwarnings("ignore")
 
# STEP 1: EXTRACT — Simulate loading raw data

 
def extract_data() -> pd.DataFrame:
    """
    Extract: Load raw data from a CSV-like source.
    In production this would be a database query, API call, or file read.
    Here we generate a realistic synthetic dataset to demonstrate the pipeline.
    """
    print("\n" + "="*70)
    print("STEP 1: EXTRACT — Loading Raw Data")
    print("="*70)
 
    np.random.seed(42)
    n = 500
 
    raw_csv = """age,income,education,experience,marital_status,job_type,churned
"""
    rows = []
    for _ in range(n):
        age        = np.random.randint(18, 65)
        income     = round(np.random.normal(60000, 20000), 2) if np.random.rand() > 0.05 else np.nan
        education  = np.random.choice(["High School", "Bachelor", "Master", "PhD", None],
                                       p=[0.25, 0.40, 0.20, 0.10, 0.05])
        experience = max(0, age - 22 + np.random.randint(-3, 5))
        marital    = np.random.choice(["Single", "Married", "Divorced", None],
                                       p=[0.35, 0.50, 0.10, 0.05])
        job_type   = np.random.choice(["Tech", "Finance", "Healthcare", "Retail", "Other"])
        # Churn depends loosely on income & age
        churn_prob = 0.3 - (income or 60000) / 400000 + (age < 30) * 0.1
        churned    = int(np.random.rand() < max(0.05, min(0.95, churn_prob)))
        rows.append([age, income if income else "", education or "", experience,
                     marital or "", job_type, churned])
 
    df = pd.DataFrame(rows, columns=[
        "age","income","education","experience","marital_status","job_type","churned"])
 
    print(f"  ✔  Loaded {len(df)} records × {len(df.columns)} columns")
    print(f"\n  Preview (first 5 rows):\n{df.head().to_string(index=False)}")
    print(f"\n  Missing values per column:\n{df.replace('', np.nan).isnull().sum().to_string()}")
    return df.replace('', np.nan)

# STEP 2: TRANSFORM — Clean, engineer features, and build the SK-Learn pipeline

 
def transform_data(df: pd.DataFrame):
    """
    Transform: Apply cleaning, feature engineering, encoding, and scaling.
    Returns (X_train, X_test, y_train, y_test, preprocessing_pipeline).
    """
    print("\n" + "="*70)
    print("STEP 2: TRANSFORM — Data Cleaning & Feature Engineering")
    print("="*70)
 
    # --- 2a. Basic Cleaning ---
    df = df.copy()
 
    # Convert numeric columns that may have been read as object
    df["age"]        = pd.to_numeric(df["age"],        errors="coerce")
    df["income"]     = pd.to_numeric(df["income"],     errors="coerce")
    df["experience"] = pd.to_numeric(df["experience"], errors="coerce")
    df["churned"]    = pd.to_numeric(df["churned"],    errors="coerce").astype(int)
 
    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  ✔  Duplicate rows removed : {before - len(df)}")
 
    # --- 2b. Feature Engineering ---
    df["income_per_age"]         = df["income"] / (df["age"] + 1)
    df["experience_income_ratio"] = df["experience"] / (df["income"] / 1000 + 1)
    df["is_senior"]               = (df["age"] >= 45).astype(int)
    print("  ✔  New features created  : income_per_age, experience_income_ratio, is_senior")
 
    # --- 2c. Define feature groups ---
    numerical_features   = ["age", "income", "experience",
                             "income_per_age", "experience_income_ratio"]
    categorical_features = ["education", "marital_status", "job_type"]
    binary_features      = ["is_senior"]
 
    X = df[numerical_features + categorical_features + binary_features]
    y = df["churned"]
 
    # --- 2d. Build Scikit-Learn ColumnTransformer ---
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # fill missing with median
        ("scaler",  StandardScaler())                    # z-score normalisation
    ])
 
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing mode
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
 
    preprocessor = ColumnTransformer(transformers=[
        ("num",  numerical_pipeline,   numerical_features),
        ("cat",  categorical_pipeline, categorical_features),
        ("bin",  "passthrough",        binary_features)
    ])
 
    # --- 2e. Train / Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
 
    print(f"  ✔  Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"  ✔  Class balance (train) — Churned: {y_train.sum()} | Not Churned: {(y_train==0).sum()}")
 
    return X_train, X_test, y_train, y_test, preprocessor
 
 
# STEP 3: LOAD — Train models and persist the processed dataset

 
def load_and_model(X_train, X_test, y_train, y_test, preprocessor):
    """
    Load: Fit the preprocessing pipeline + model, evaluate, and save outputs.
    """
    print("\n" + "="*70)
    print("STEP 3: LOAD — Model Training & Evaluation")
    print("="*70)
 
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42)
    }
 
    results = {}
    for name, clf in models.items():
        full_pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier",    clf)
        ])
        full_pipeline.fit(X_train, y_train)
        y_pred   = full_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {"pipeline": full_pipeline, "accuracy": accuracy, "y_pred": y_pred}
 
        print(f"\n  ── {name} ──")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Not Churned','Churned'])}")
 
    # Pick the best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n  ✔  Best model: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")
 
    # Save processed test predictions to CSV
    output = X_test.copy()
    output["actual_churn"]    = y_test.values
    output["predicted_churn"] = results[best_name]["y_pred"]
    output.to_csv("/mnt/user-data/outputs/task1_predictions.csv", index=False)
    print("  ✔  Predictions saved  → task1_predictions.csv")
 
    return results[best_name]["pipeline"]
 
 
 
def main():
    print("\n" + "█"*70)
    print("  CODTECH INTERNSHIP — TASK 1: DATA PIPELINE DEVELOPMENT (ETL)")
    print("█"*70)
 
    # ETL Pipeline
    raw_df              = extract_data()
    X_train, X_test, y_train, y_test, preprocessor = transform_data(raw_df)
    trained_pipeline    = load_and_model(X_train, X_test, y_train, y_test, preprocessor)
 
    print("\n" + "="*70)
    print("  ✅  ETL Pipeline completed successfully!")
    print("="*70 + "\n")
 
 
if __name__ == "__main__":
    main()
 
