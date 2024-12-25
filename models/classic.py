# models/classic.py
# ---------------------------------------------
# IsolationForest обучение только на нормальных данных
# RandomForest будет тюнинговаться в train.py
# ---------------------------------------------
from sklearn.ensemble import IsolationForest

class ClassicModels:
    """
    Класс для обучения классических моделей.
    Пока реализуем только IsolationForest, обучая его на нормальных данных.
    """
    @staticmethod
    def train_isolation_forest(X_train_normal):
        # Обучение на нормальных данных, чтобы модель учила "норму".
        iso = IsolationForest(contamination=0.01, random_state=42)
        iso.fit(X_train_normal)
        return iso