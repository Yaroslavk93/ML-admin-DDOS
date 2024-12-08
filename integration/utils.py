# integration/utils.py
# ---------------------------------------------
# Вспомогательные функции для инференса, логирования, форматов вывода
# ---------------------------------------------

def print_summary(results_dict: dict):
    # Печатает сводку результатов
    for k, v in results_dict.items():
        print(f"{k}: {v}")