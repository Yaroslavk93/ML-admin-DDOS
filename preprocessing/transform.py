# preprocessing/transform.py
# ---------------------------------------------
# Добавляем опцию feature selection и scaler_type
# feature_selection: если True, отбираем top-N признаков по важности из RandomForest
# ---------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    Предобработка данных:
    - Заполнение пропусков.
    - Кодирование меток (Label: BENIGN=0, остальные=1).
    - Масштабирование признаков (MinMaxScaler или StandardScaler).
    - Опциональный отбор признаков по важности через RandomForest на тренировочной выборке.
    """
    def __init__(self, target_col: str = "Label", scaler_type="minmax", do_feature_selection=False, n_features=30):
        self.target_col = target_col
        self.scaler_type = scaler_type
        self.scaler = None
        self.do_feature_selection = do_feature_selection
        self.n_features = n_features
        self.selected_features = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Заполняем пропуски в Flow Bytes/s медианой, если есть столбец
        if "Flow Bytes/s" in df.columns:
            df["Flow Bytes/s"] = df["Flow Bytes/s"].fillna(df["Flow Bytes/s"].median())
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        # Перекодируем метку: BENIGN=0, остальные=1
        if self.target_col in df.columns:
            df["OriginalLabel"] = df[self.target_col]
            df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x == "BENIGN" else 1)
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # Здесь можно добавить логику отбора или генерации признаков, если нужно.
        return df

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        # Сохраняем scaler для использования в инференсе
        import joblib
        joblib.dump(self.scaler, "models/scaler.pkl")
        return X

    def select_features(self, X_train, y_train):
        # Отбор признаков по важности через RandomForestClassifier (бин. классификация)
        # Обучаем RF на X_train, y_train, затем берем top-n_features
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        importances = pd.Series(rf.feature_importances_, index=X_train.columns)
        importances = importances.sort_values(ascending=False)
        self.selected_features = importances.iloc[:self.n_features].index.tolist()
        return X_train[self.selected_features]

    def prepare_data(self, df: pd.DataFrame):
        # Полный цикл предобработки
        df = self.clean_data(df)
        df = self.encode_target(df)
        df = self.feature_engineering(df)

        y = df[self.target_col]
        y_original = df["OriginalLabel"]
        X = df.drop(columns=[self.target_col, "OriginalLabel", "SourceFile"], errors='ignore')

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()

        # Масштабируем данные
        X = self.scale_features(X)

        print("Статистика X_train после масштабирования:")
        print(X.describe().T)

        y = y.loc[X.index]
        y_original = y_original.loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        y_train_original = y_original.loc[X_train.index]
        y_test_original = y_original.loc[X_test.index]

        if self.do_feature_selection:
            # Выполняем отбор признаков на X_train
            X_train_selected = self.select_features(X_train, y_train)
            # Применяем те же признаки к X_test
            X_test_selected = X_test[self.selected_features]

            # Перезаписываем X_train, X_test
            X_train, X_test = X_train_selected, X_test_selected

        # Сохраняем список отобранных признаков (или все, если нет отбора)
        if not self.do_feature_selection:
            self.selected_features = X_train.columns.tolist()

        # Сохраняем выбранные признаки
        # Это позволит в ноутбуке понять, какие признаки были использованы
        return X_train, X_test, y_train, y_test, y_train_original, y_test_original, self.selected_features