# train.py
# ---------------------------------------------
# Основной скрипт: загрузка данных, предобработка, обучение моделей, оценка.
# ---------------------------------------------
import os
from preprocessing.data_loader import DataLoader
from preprocessing.transform import DataPreprocessor
from models.autoencoder import AutoencoderModel
from models.classic import ClassicModels
from evaluation.metrics import MetricsEvaluator
import joblib

if __name__ == "__main__":
    # Путь к сырым данным
    raw_data_path = "data/raw"

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    loader = DataLoader(raw_data_path)
    df = loader.load_data()
    if df.empty:
        print("Данные не загружены. Проверьте путь к датасету.")
        exit(1)

    preprocessor = DataPreprocessor(target_col="Label")
    # Теперь prepare_data возвращает и оригинальные метки
    X_train, X_test, y_train, y_test, y_train_original, y_test_original = preprocessor.prepare_data(df)

    # Сохраняем предобработанные данные и оригинальные метки
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    y_train_original.to_csv("data/processed/y_train_original.csv", index=False)
    y_test_original.to_csv("data/processed/y_test_original.csv", index=False)

    # Сохраняем scaler
    joblib.dump(preprocessor.scaler, "models/scaler.pkl")

    # Обучение автоэнкодера
    ae = AutoencoderModel(input_dim=X_train.shape[1])
    ae.train(X_train, epochs=5, batch_size=512)
    ae.model.save("models/autoencoder_model.h5")

    # Оценка автоэнкодера
    test_scores = ae.reconstruct_error(X_test)
    threshold = test_scores.mean() + test_scores.std()
    ae_metrics = MetricsEvaluator.evaluate(y_test, test_scores, threshold=threshold)
    print("Autoencoder metrics:", ae_metrics)

    # Обучение Isolation Forest
    iso = ClassicModels.train_isolation_forest(X_train)
    iso_scores = iso.decision_function(X_test) * -1
    iso_threshold = iso_scores.mean() + iso_scores.std()
    iso_metrics = MetricsEvaluator.evaluate(y_test, iso_scores, threshold=iso_threshold)
    print("Isolation Forest metrics:", iso_metrics)
    joblib.dump(iso, "models/isolation_forest_model.pkl")

    # Обучение Random Forest
    rf = ClassicModels.train_random_forest(X_train, y_train)
    rf_scores = rf.predict_proba(X_test)[:, 1]
    rf_metrics = MetricsEvaluator.evaluate(y_test, rf_scores)
    print("Random Forest metrics:", rf_metrics)
    joblib.dump(rf, "models/random_forest_model.pkl")

    print("Обучение завершено. Модели сохранены в папке models/.")