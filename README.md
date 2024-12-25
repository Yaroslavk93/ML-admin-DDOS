# Обнаружение аномалий в сетевом трафике (CICIDS2017)

Данный проект нацелен на обнаружение аномалий в сетевом трафике с использованием датасета [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html). В нём реализованы несколько методов:

- **Autoencoder** (Anomaly Detection): Позволяет выявлять неизвестные ранее атаки, обучаясь на нормальном трафике.
- **Isolation Forest** (Unsupervised): Классический алгоритм выявления аномалий.
- **Random Forest** (Supervised): Обучается на размеченных данных и показывает высокое качество на известных атаках, но менее эффективен против новых, неизвестных угроз.

**Почему Anomaly Detection?**  
Супервайзед методы (как Random Forest) показывают почти 100% качество на знакомых атаках, но их гибкость ограничена: новые атаки, не встречавшиеся при обучении, они не распознают. Anomaly Detection модели (Autoencoder, IsolationForest) более универсальны в реальной среде, где постоянно появляются новые типы атак.

## Структура проекта
```
CICIDS2017_AnomalyDetection/
├─ venv/                 # виртуальное окружение
├─ data/
│  ├─ raw/               # сырые данные (CICIDS2017 CSV)
│  └─ processed/          # предобработанные данные (X_train.csv, ...)
├─ models/
│  ├─ autoencoder_model.keras
│  ├─ isolation_forest_model.pkl
│  ├─ random_forest_model.pkl
│  ├─ scaler.pkl
│  └─ thresholds.json
├─ notebooks/
│  ├─ EDA.ipynb           # Исследовательский анализ данных
│  ├─ Model_Evaluation.ipynb # Анализ результатов моделей
├─ preprocessing/
│  ├─ data_loader.py      # Загрузка исходных CSV
│  ├─ transform.py        # Предобработка данных: заполнение пропусков, нормализация, отбор признаков
├─ evaluation/
│  ├─ metrics.py          # Метрики оценки: precision, recall, f1, roc-auc
│  ├─ visualization.py     # Визуализация ROC-кривых
├─ integration/
│  ├─ app.py              # Консольное приложение для инференса
│  ├─ utils.py
├─ train.py               # Основной скрипт обучения и логирования экспериментов в MLflow
├─ prepare_new_data_raw.py  # Генерация тестового набора для инференса (DDoS+BENIGN)
├─ prepare_new_data.py     # Предобработка новых данных для инференса
├─ requirements.txt
└─ README.md              # Описание проекта
```

- `data/raw/` — Исходные CSV файлы.
- `data/processed/` — Обработанные данные (X_train, X_test, и т.п.).
- `models/` — Сохранённые модели, scaler, thresholds.
- `preprocessing/` — Код предобработки данных (transform, data_loader).
- `evaluation/` — Метрики и визуализация.
- `integration/` — Интеграционное приложение (app.py).
- `notebooks/` — Jupyter ноутбуки для EDA и оценки моделей.
- `mlruns/` — Логи MLflow экспериментов.


1. Создать виртуальное окружение и установить зависимости:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```


Подготовка среды и запуск проекта:

Установите Python (ставил 3.11.9).  
Создайте виртуальное окружение командой:  
```bash
python -m venv venv
```
После выполнения команды просто появится папка venv.
Активируйте окружение:
```bash
Windows: venv\Scripts\activate
Linux/MacOS: source venv/bin/activate
```
Установите зависимости:
```bash
pip install -r requirements.txt
```
Поместите исходные CSV-файлы CICIDS2017 в data/raw/.  
Запустите процесс обучения:
```bash
python train.py
```
По окончании обучения модели сохранятся в папке models/.  
Для анализа данных и результатов:  
jupyter notebook  
Откройте notebooks/EDA.ipynb или Model_Evaluation.ipynb.

Для запуска mlflow, используем нижеприведённую команду и переходим по адресу http://127.0.0.1:5000
```bash
mlflow ui
```

(venv) C:\AnomalyDetection>python prepare_new_data_raw.py
new_data_raw.csv создан, содержит 500 DDoS и 500 BENIGN.

(venv) C:\AnomalyDetection>python prepare_new_data.py
Новые данные подготовлены (new_data.csv) для инференса.

Инференс новых данных
Создайте new_data_raw.csv:
bash
Копировать код
python prepare_new_data_raw.py
Предобработайте их:
bash
Копировать код
python prepare_new_data.py
Запустите инференс:
bash
Копировать код
python integration/app.py data/processed/new_data.csv
Скрипт покажет, сколько аномалий нашел VAE и ISO.



Описание дипломной работы  
Цель дипломной работы: Создать систему обнаружения аномалий в сетевом трафике, способную выявлять неизвестные атаки без необходимости иметь обучающую выборку, помеченную всеми типами атак. Датасет CICIDS2017 содержит трафик с различными типами атак (DDoS, PortScan, Web Attacks и др.), а также большой объем нормального трафика.  
  
Основная идея:  
Использовать anomaly detection подходы (VAE, ISO), которые учатся на нормальных данных и затем выявляют аномалии по степени отклонения от нормы. Сравнить с RandomForest, который требует меток атак, чтобы понять верхний предел качества при известной разметке.  
  
Результаты:  
  
RandomForest: Почти идеальные метрики (F1>0.99, ROC-AUC>0.999), но это при условии знания меток.  
Без дополнительных улучшений VAE и ISO давали ~0.40 F1 и ~0.70 ROC-AUC.  
После отбор признаков и настройки параметров F1 для ISO вырос до ~0.53, VAE до ~0.48–0.49 F1, ROC-AUC ~0.76–0.78.  
Это улучшение подтверждает, что уменьшение размерности и отбор информативных признаков помогает anomaly detection методам.
Почему такой подход:  

В реальных сценариях атак могут быть новые типы, не встречавшиеся в обучении. Супервайзед методы, как RandomForest, не обнаружат новые атаки.  
VAE и ISO способны зафиксировать "нормальный профиль" и сигнализировать о любых отклонениях. Это даёт преимущество при появлении новых, неизвестных угроз.  
Источники:  

CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html  
VAE: D. P. Kingma, M. Welling. "Auto-Encoding Variational Bayes." ICLR, 2014.  
Isolation Forest: Liu et al., "Isolation Forest", ICDM 2008.  
RandomForest: L. Breiman "Random Forests", Machine Learning, 2001.  
Анализ:  

Результаты показывают, что anomaly detection (VAE, ISO) сложнее повысить до уровня супервайзед методов без меток.
Но достигнутый прогресс (F1 ~0.53 у ISO вместо 0.40) важен, так как в реальных условиях часто не хватает данных о новых атаках.  
Будущее улучшение — использование более продвинутых архитектур (например, GAN-based anomaly detection), более тщательно тюнинговать гиперпараметры (Optuna), другие типы нормализации, а также рассмотреть временную корреляцию более глубоко, если она присутствует.  






Проблема - файл selected_features.csv содержит лишнюю строку '0' в начале. Это происходит из-за того, что при сохранении pd.Series(selected_features) по умолчанию добавляет имя серии или заголовок.

# train.py
# ...
pd.Series(selected_features).to_csv("data/processed/selected_features.csv", index=False, header=False)
# ...


проверка результата:

```bash
python integration/app.py models/random_forest_model.pkl data/processed/new_data.csv


Metric             Value
iso_f1             0.5329110367551276
iso_precision      0.6818937342422262
iso_recall         0.4373558846957782
iso_roc_auc        0.783035966845494
rf_f1              0.9971119912281228
rf_precision       0.9975303757785011
rf_recall          0.9966939574886056
rf_roc_auc         0.9997745114234884
runtime_seconds    1404.5809271335602
vae_f1             0.4851443379738444
vae_precision      0.6536677459627896
vae_recall         0.38570495966268786
vae_roc_auc        0.7645058557897825


Найденные аномалии:
   Packet Length Std  Avg Bwd Segment Size  Max Packet Length  ...  ACK Flag Count  Fwd IAT Max  OriginalLabel
0           0.475643              0.400207           0.290693  ...             1.0     0.050629  OriginalLabel
1           0.539818              0.400207           0.235294  ...             0.0     0.000005           DDoS
2           0.516043              0.400000           0.290693  ...             0.0     0.000092           DDoS
3           0.000000              0.000000           0.000242  ...             1.0     0.069367           DDoS
4           0.000000              0.000000           0.000242  ...             1.0     0.016985           DDoS

[5 rows x 31 columns]
```


