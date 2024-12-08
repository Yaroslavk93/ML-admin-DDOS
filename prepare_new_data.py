# prepare_new_data.py
# ---------------------------------------------
# Подготовка новых данных (new_data_raw.csv) с помощью сохраненного scaler
# ---------------------------------------------
# prepare_new_data.py
import pandas as pd
import numpy as np
import joblib
from preprocessing.transform import DataPreprocessor

new_df = pd.read_csv("data/raw/new_data_raw.csv")

preprocessor = DataPreprocessor(target_col="Label")
new_df = preprocessor.clean_data(new_df)
new_df = preprocessor.encode_target(new_df)
new_df = preprocessor.feature_engineering(new_df)

y_new = new_df['Label']
y_new_original = new_df['OriginalLabel']  # Важно, чтобы эта колонка появилась в encode_target

X_new = new_df.drop(columns=["Label", "OriginalLabel", "SourceFile"], errors='ignore')
X_new = X_new.replace([np.inf, -np.inf], np.nan)
X_new = X_new.dropna()

scaler = joblib.load("models/scaler.pkl")
numeric_cols = X_new.select_dtypes(include=[np.number]).columns
X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])

X_new.to_csv("data/processed/new_data.csv", index=False)
y_new.to_csv("data/processed/y_new.csv", index=False)
y_new_original.to_csv("data/processed/y_new_original.csv", index=False)
print("Новые данные подготовлены и сохранены в data/processed/new_data.csv и data/processed/y_new_original.csv")