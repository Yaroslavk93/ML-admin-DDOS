# evaluation/metrics.py
# ---------------------------------------------
# Метрики оценки качества модели: Precision, Recall, F1, ROC-AUC
# ---------------------------------------------
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class MetricsEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred_scores, threshold=0.5):
        # Преобразуем непрерывные оценки в предсказания классов по порогу
        y_pred = (y_pred_scores > threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_scores)
        return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}