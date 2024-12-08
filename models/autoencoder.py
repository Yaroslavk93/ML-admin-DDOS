# models/autoencoder.py
# ---------------------------------------------
# Класс для создания и обучения автоэнкодера, используемого для выявления аномалий
# ---------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, models

class AutoencoderModel:
    def __init__(self, input_dim: int, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = self._build_model()

    def _build_model(self):
        # Строим простую архитектуру автоэнкодера
        input_layer = tf.keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(encoded)
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train(self, X_train, epochs=10, batch_size=256):
        # Обучаем автоэнкодер реконструировать нормальные данные
        self.model.fit(X_train, X_train, 
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.1,
                       verbose=1)

    def reconstruct_error(self, X):
        # Рассчитываем MSE между входом и реконструкцией
        reconstructed = self.model.predict(X, verbose=0)
        mse = ((X - reconstructed)**2).mean(axis=1)
        return mse