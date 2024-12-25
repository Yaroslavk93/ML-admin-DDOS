# integration/app.py
# ---------------------------------------------
# Консольное приложение для инференса на новых данных
# ---------------------------------------------
import sys
import os
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Добавляем путь к корню проекта, чтобы импортировать models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from models.autoencoder import sampling_func

class App:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.selected_features = None
        self.scaler = None

    def load_model(self):
        _, ext = os.path.splitext(self.model_path)
        if ext == ".pkl":
            self.model = joblib.load(self.model_path)
        else:
            # Передаем custom_objects
            self.model = load_model(self.model_path, compile=False, custom_objects={"MyVAE>sampling_func": sampling_func})

        self.selected_features = pd.read_csv("data/processed/selected_features.csv", header=None).iloc[:,0].tolist()
        self.scaler = joblib.load("models/scaler.pkl")

    def predict_anomalies(self, input_data_path: str):
        X = pd.read_csv(input_data_path)

        missing = set(self.selected_features) - set(X.columns)
        if missing:
            print("Отобранные признаки не найдены в данных:", missing)
            return pd.DataFrame()

        X = X[self.selected_features]

        original_labels_path = input_data_path.replace("new_data.csv", "y_new_original.csv")
        y_original = None
        try:
            y_original = pd.read_csv(original_labels_path, header=None, names=["OriginalLabel"])
        except FileNotFoundError:
            print("Original labels file not found. Cannot identify exact attack type.")

        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X)*-1
        elif hasattr(self.model, 'predict_proba'):
            scores = self.model.predict_proba(X)[:, 1]
        else:
            # Автоэнкодер
            X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))
            reconstructed = self.model.predict(X_reshaped, verbose=0)
            scores = ((X_reshaped - reconstructed)**2).mean(axis=(1,2))

        threshold = 0.5
        anomaly_mask = (scores > threshold)
        anomalies = X[anomaly_mask].copy()

        if anomalies.empty:
            print("Аномалии не найдены при данном пороге.")
            print("Найденные аномалии:")
            print(anomalies.head())
            return anomalies

        numeric_cols = anomalies.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == len(self.scaler.min_):
            anomalies.loc[:, numeric_cols] = self.scaler.inverse_transform(anomalies[numeric_cols])
        else:
            print("Предупреждение: Число признаков для обратного преобразования не совпадает с обучением, пропускаем inverse_transform")

        if y_original is not None:
            y_original_anomalies = y_original.loc[anomalies.index, "OriginalLabel"]
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
