# DATA-PIPELINE-DEVELOPMENT
Full ETL pipeline on a 500-record customer churn dataset. Extracts raw data with missing values, applies a Scikit-learn ColumnTransformer (median imputation + StandardScaler for numerics, mode imputation ), engineers 3 new features, then trains Logistic Regression and Random Forest. Best model: Random Forest at 84% accuracy.
