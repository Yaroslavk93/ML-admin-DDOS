# evaluation/visualization.py
# ---------------------------------------------
# Визуализация: ROC-кривые, гистограммы, boxplots и т.д.
# ---------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

class Visualization:
    @staticmethod
    def plot_roc(y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label='Model')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.legend()
        plt.show()