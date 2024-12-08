# models/classic.py
# ---------------------------------------------
# Классические модели для сравнения с нейросетевыми:
# IsolationForest - для обнаружения аномалий
# RandomForestClassifier - для бинарной классификации (норма/атака)
# ---------------------------------------------
from sklearn.ensemble import IsolationForest, RandomForestClassifier

class ClassicModels:
    @staticmethod
    def train_isolation_forest(X_train):
        iso = IsolationForest(contamination=0.01, random_state=42)
        iso.fit(X_train)
        return iso

    @staticmethod
    def train_random_forest(X_train, y_train):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        return rf