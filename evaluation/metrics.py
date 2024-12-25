# evaluation/metrics.py
# ---------------------------------------------
# Метрики оценки: Precision, Recall, F1, ROC-AUC
# ---------------------------------------------
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class MetricsEvaluator:
    """
    Класс для расчета метрик классификации:
    y_true: истинные метки (0 или 1)
    y_pred_scores: непрерывные оценки (скоры)
    threshold: порог для бинаризации
    Возвращает словарь с precision, recall, f1, roc_auc
    """
    @staticmethod
    def evaluate(y_true, y_pred_scores, threshold=0.5):
        y_pred = (y_pred_scores > threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_scores)
        return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}