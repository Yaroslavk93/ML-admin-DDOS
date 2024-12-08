# integration/app.py
# ---------------------------------------------
# Консольное приложение для инференса на новых данных:
# Загружает обученную модель и предсказывает аномалии
# ---------------------------------------------
import pandas as pd
import joblib
import sys
import numpy as np

class App:
    def __init__(self, model_path:str):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def predict_anomalies(self, input_data_path:str):
        X = pd.read_csv(input_data_path)

        # Попытка загрузить оригинальные метки
        original_labels_path = input_data_path.replace("new_data.csv", "y_new_original.csv")
        y_original = None
        try:
            # Загружаем оригинальные метки как DataFrame с одной колонкой OriginalLabel
            y_original = pd.read_csv(original_labels_path, header=None, names=["OriginalLabel"])
        except FileNotFoundError:
            print("Original labels file not found. Cannot identify exact attack type.")

        # Предсказание
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X) * -1
        elif hasattr(self.model, 'predict_proba'):
            scores = self.model.predict_proba(X)[:, 1]
        else:
            print("Модель не поддерживает инференс.")
            return pd.DataFrame()

        threshold = 0.5
        anomaly_mask = (scores > threshold)
        anomalies = X[anomaly_mask].copy()  # Делаем копию, чтобы избежать SettingWithCopyWarning
        anomaly_indices = anomalies.index

        # Обратное преобразование масштабирования
        scaler = joblib.load("models/scaler.pkl")
        numeric_cols = anomalies.select_dtypes(include=[np.number]).columns
        anomalies.loc[:, numeric_cols] = scaler.inverse_transform(anomalies[numeric_cols])

        # Добавляем OriginalLabel, если файл нашелся
        if y_original is not None:
            # y_original_anomalies - это Series (строки по индексам anomalies)
            y_original_anomalies = y_original.loc[anomaly_indices, "OriginalLabel"]
            # Теперь y_original_anomalies - это Series с тем же индексом, что и anomalies
            # Присваиваем одномерный массив значений этой Series в новую колонку anomalies
            anomalies["OriginalLabel"] = y_original_anomalies.values

        return anomalies

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python app.py <model_path> <input_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_file = sys.argv[2]

    app = App(model_path)
    app.load_model()
    anomalies = app.predict_anomalies(input_file)
    print("Найденные аномалии:")
    print(anomalies.head())
