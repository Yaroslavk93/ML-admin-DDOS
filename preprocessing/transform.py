# preprocessing/transform.py
# ---------------------------------------------
# Предобработка данных: заполнение пропусков, нормализация, кодирование меток, разделение
# Добавлено: Очистка inf и NaN перед масштабированием
# ---------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, target_col: str = "Label"):
        self.target_col = target_col
        self.scaler = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Заполняем пропуски в столбце Flow Bytes/s медианой
        if "Flow Bytes/s" in df.columns:
            df["Flow Bytes/s"] = df["Flow Bytes/s"].fillna(df["Flow Bytes/s"].median())
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        # Сохраняем исходную метку в отдельной колонке "OriginalLabel"
        if self.target_col in df.columns:
            df["OriginalLabel"] = df[self.target_col]
            # BENIGN = 0, остальные = 1
            df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x == "BENIGN" else 1)
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # Место для отбора признаков или генерации новых признаков при необходимости
        return df

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler = StandardScaler()
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        return X

    def prepare_data(self, df: pd.DataFrame):
        df = self.clean_data(df)
        df = self.encode_target(df)
        df = self.feature_engineering(df)

        # Разделяем целевую переменную и оригинальную метку
        y = df[self.target_col]
        y_original = df["OriginalLabel"]
        
        X = df.drop(columns=[self.target_col, "OriginalLabel", "SourceFile"], errors='ignore')

        # Масштабируем
        # Заменим inf на NaN и удалим строки с NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        
        X = self.scale_features(X)

        # Синхронизируем индексы y и y_original с X
        y = y.loc[X.index]
        y_original = y_original.loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Синхронизируем оригинальные метки с разделением
        y_original_train = y_original.loc[X_train.index]
        y_original_test = y_original.loc[X_test.index]

        return X_train, X_test, y_train, y_test, y_original_train, y_original_test