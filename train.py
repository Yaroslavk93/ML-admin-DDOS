# train.py
# ---------------------------------------------
# Расширяем код для перебора параметров VAE, добавляем отбор признаков и сохранение дополнительных данных.
# ---------------------------------------------
import os
import subprocess
import mlflow
import mlflow.sklearn
import time
import json

from preprocessing.data_loader import DataLoader
from preprocessing.transform import DataPreprocessor
from evaluation.metrics import MetricsEvaluator
from models.classic import ClassicModels
from models.autoencoder import LSTMVAE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import joblib

if __name__ == "__main__":
    # Пытаемся получить commit hash для логирования эксперимента.
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except:
        commit_hash = "unknown_commit"

    experiment_name = "Anomaly_Detection_Exp"
    mlflow.set_experiment(experiment_name)

    # Путь к сырым данным
    raw_data_path = "data/raw"
    loader = DataLoader(raw_data_path)
    df = loader.load_data()
    if df.empty:
        print("Нет данных: проверьте наличие исходных CSV в data/raw/.")
        exit(1)

    # Инициализируем препроцессор: по умолчанию MinMaxScaler, можно переключить на StandardScaler внутри transform.py
    preprocessor = DataPreprocessor(target_col="Label", scaler_type="minmax", do_feature_selection=True, n_features=30)
    # do_feature_selection=True и n_features=30 означает, что отберём 30 самых важных признаков

    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig, selected_features = preprocessor.prepare_data(df)

    # Сохраняем обучающий набор
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    y_train_orig.to_csv("data/processed/y_train_original.csv", index=False)
    y_test_orig.to_csv("data/processed/y_test_original.csv", index=False)


    # Сохраняем список отобранных признаков для анализа в ноутбуке
    pd.Series(selected_features).to_csv("data/processed/selected_features.csv", index=False)

    # Нормальные данные для обучения VAE и ISO (берем только BENIGN)
    X_train_normal = X_train[y_train==0].values
    X_test_normal = X_test[y_test==0].values

    # Формируем данные для LSTMVAE, если нет реальной временной структуры, timesteps=1
    X_train_normal = X_train_normal.reshape((X_train_normal.shape[0], 1, X_train_normal.shape[1]))
    X_test_normal = X_test_normal.reshape((X_test_normal.shape[0], 1, X_test_normal.shape[1]))
    X_test_all = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Список параметров для перебора
    vae_params_list = [
        {"latent_dim": 16, "intermediate_dim": 32},
        {"latent_dim": 32, "intermediate_dim": 64},
        {"latent_dim": 64, "intermediate_dim": 64},
        {"latent_dim": 32, "intermediate_dim": 128}
    ]

    best_vae_f1 = 0.0
    best_params = None
    best_run_id = None

    for params in vae_params_list:
        with mlflow.start_run() as run:
            # Логируем параметры эксперимента
            mlflow.log_param("commit_hash", commit_hash)
            mlflow.log_param("latent_dim", params["latent_dim"])
            mlflow.log_param("intermediate_dim", params["intermediate_dim"])
            mlflow.log_param("scaler", "MinMaxScaler")
            mlflow.log_param("feature_selection", True)
            mlflow.log_param("n_features_selected", len(selected_features))

            start_time = time.time()
            mlflow.log_param("start_time", start_time)

            # Обучаем VAE
            vae = LSTMVAE(input_dim=X_train.shape[1], timesteps=1,
                          latent_dim=params["latent_dim"], 
                          intermediate_dim=params["intermediate_dim"])
            vae.train(X_train_normal, epochs=20, batch_size=256)  # при желании epochs/batch_size можно вынести в список параметров

            vae.model.save("models/autoencoder_model.keras")
            mlflow.log_artifact("models/autoencoder_model.keras")

            # Подбираем порог для VAE
            test_scores_vae_normal = vae.reconstruct_error(X_test_normal)
            vae_threshold = np.percentile(test_scores_vae_normal, 95)
            test_scores_vae = vae.reconstruct_error(X_test_all)
            vae_metrics = MetricsEvaluator.evaluate(y_test, test_scores_vae, threshold=vae_threshold)
            for k,v in vae_metrics.items():
                mlflow.log_metric("vae_"+k, v)

            # Isolation Forest обучение на норме
            iso = ClassicModels.train_isolation_forest(X_train_normal.reshape(X_train_normal.shape[0], -1))
            iso_scores_test = iso.decision_function(X_test.values)*-1
            iso_scores_normal = iso.decision_function(X_test_normal.reshape(X_test_normal.shape[0], -1))*-1
            iso_threshold = np.percentile(iso_scores_normal,95)
            iso_metrics = MetricsEvaluator.evaluate(y_test, iso_scores_test, threshold=iso_threshold)
            for k,v in iso_metrics.items():
                mlflow.log_metric("iso_"+k, v)
            joblib.dump(iso, "models/isolation_forest_model.pkl")
            mlflow.log_artifact("models/isolation_forest_model.pkl")

            # RandomForest тюнинг (упрощённый)
            param_dist = {
                'n_estimators': [100,200],
                'max_depth': [None,10],
                'min_samples_split': [2,5],
                'min_samples_leaf': [1,2]
            }
            rf_base = RandomForestClassifier(random_state=42)
            rf_search = RandomizedSearchCV(rf_base, param_dist, n_iter=2, cv=3, random_state=42, verbose=1, n_jobs=-1)
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            mlflow.log_params(rf_search.best_params_)
            rf_scores = best_rf.predict_proba(X_test)[:,1]
            rf_metrics = MetricsEvaluator.evaluate(y_test, rf_scores)
            for k,v in rf_metrics.items():
                mlflow.log_metric("rf_"+k, v)
            joblib.dump(best_rf, "models/random_forest_model.pkl")
            mlflow.log_artifact("models/random_forest_model.pkl")

            # Сохраняем пороги для VAE и ISO
            thresholds = {
                "vae_threshold": float(vae_threshold),
                "iso_threshold": float(iso_threshold)
            }
            with open("models/thresholds.json", "w") as f:
                json.dump(thresholds, f)
            mlflow.log_artifact("models/thresholds.json")

            end_time = time.time()
            mlflow.log_metric("runtime_seconds", end_time - start_time)
            mlflow.log_param("status", "completed")

            # Сравниваем vae_f1 c лучшим результатом, выбираем лучшие параметры
            if vae_metrics["f1"] > best_vae_f1:
                best_vae_f1 = vae_metrics["f1"]
                best_params = params
                best_run_id = run.info.run_id

    print("Лучшие параметры VAE по F1:", best_params, "с F1:", best_vae_f1, "Run ID:", best_run_id)
    print("Откройте MLflow UI для сравнения результатов.")