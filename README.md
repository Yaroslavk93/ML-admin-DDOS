Проект: Обнаружение аномалий в сетевом трафике (CICIDS2017)
Описание проекта: Данный проект предназначен для обнаружения аномалий в сетевом трафике с использованием датасета CICIDS2017 (https://www.unb.ca/cic/datasets/ids-2017.html).
Используются методы глубокого обучения (автоэнкодер) и классические алгоритмы машинного обучения (IsolationForest, RandomForest).

Цель проекта – сравнить различные подходы, оценить их эффективность и предоставить удобный интерфейс для анализа результатов и проведения инференса на новых данных.

```
Структура проекта: CICIDS2017_AnomalyDetection/
CICIDS2017_AnomalyDetection/
    ├── venv/                        # виртуальное окружение
    ├── data/
    │   ├── raw/                     # необработанные данные (CSV CICIDS2017)
    │   └── processed/               # обработанные данные (после предобработки)
    ├── models/
    │   ├── autoencoder.py           # Модель автоэнкодера
    │   ├── classic.py               # Классические модели: IsolationForest, RandomForest
    │   └── __init__.py
    ├── preprocessing/
    │   ├── data_loader.py           # Загрузка данных CICIDS2017
    │   ├── transform.py             # Предобработка данных (изменения здесь!)
    │   └── __init__.py
    ├── evaluation/
    │   ├── metrics.py               # Метрики оценки
    │   ├── visualization.py         # Визуализация результатов
    │   └── __init__.py
    ├── integration/
    │   ├── app.py                   # Консольное приложение для инференса
    │   ├── utils.py                 # Утилитные функции
    │   └── __init__.py
    ├── notebooks/
    │   ├── EDA.ipynb                # Исследовательский анализ данных
    │   ├── Model_Evaluation.ipynb   # Анализ результатов обучения моделей
    │   └── __init__.py
    ├── train.py                     # Основной скрипт обучения и оценки
    ├── requirements.txt
    └── README.md
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


Обучение модели:
```bash
python train.py
2024-12-08 19:06:18.137706: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-12-08 19:06:18.938513: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-12-08 19:06:48.505784: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/5
3480/3480 ━━━━━━━━━━━━━━━━━━━━ 5s 1ms/step - loss: 0.7413 - val_loss: 0.5977
Epoch 2/5
3480/3480 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6513 - val_loss: 0.5964
Epoch 3/5
3480/3480 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6272 - val_loss: 0.5960
Epoch 4/5
3480/3480 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6361 - val_loss: 0.5959
Epoch 5/5
3480/3480 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6255 - val_loss: 0.5956
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
Autoencoder metrics: {'precision': np.float64(0.03531300160513644), 'recall': np.float64(0.0001317625638599244), 'f1': np.float64(0.0002625454979414046), 'roc_auc': np.float64(0.7227188357192318)}
Isolation Forest metrics: {'precision': np.float64(0.4985070694651796), 'recall': np.float64(0.4079728329550151), 'f1': np.float64(0.448718919933731), 'roc_auc': np.float64(0.6826796367880535)}
Random Forest metrics: {'precision': np.float64(0.9971094287680661), 'recall': np.float64(0.9978738313558967), 'f1': np.float64(0.9974914836168136), 'roc_auc': np.float64(0.9999214276325252)}
Обучение завершено. Модели сохранены в папке models/.
```
Проверка результатов на созданном датасете из исходных данных:
```bash
python integration/app.py models/random_forest_model.pkl data/processed/new_data.csv
Найденные аномалии:
   Destination Port  Flow Duration  Total Fwd Packets  Total Backward Packets  ...      Idle Std      Idle Max      Idle Min  OriginalLabel
0              80.0      4047917.0                5.0            6.217249e-14  ...  3.492460e-10  1.862645e-09  1.862645e-09           DDos
1              80.0      5997859.0                5.0            6.217249e-14  ...  3.492460e-10  5.995860e+06  5.995860e+06           DDoS
2              80.0      1806469.0                3.0            6.000000e+00  ...  3.492460e-10  1.862645e-09  1.862645e-09           DDoS
3              80.0      2234959.0                4.0            6.217249e-14  ...  3.492460e-10  1.862645e-09  1.862645e-09           DDoS
4              80.0       872466.0                2.0            6.000000e+00  ...  3.492460e-10  1.862645e-09  1.862645e-09           DDoS

[5 rows x 79 columns]
```