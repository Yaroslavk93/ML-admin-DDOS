# models/autoencoder.py
# ---------------------------------------------
# LSTM Variational Autoencoder (VAE) для anomaly detection.
# Лосс считается через VAELossLayer, которая вызывает add_loss в call().
# Если нет временной структуры, timesteps=1.
# ---------------------------------------------
import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

@keras.saving.register_keras_serializable(package="MyVAE")
def sampling_func(args):
    # Кастомная функция семплирования z ~ N(z_mean, exp(z_log_var))
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
    return z_mean + tf.exp(0.5 * z_log_var)*epsilon

class VAELossLayer(layers.Layer):
    """
    Кастомный слой для вычисления VAE лосса:
    diff = inputs - outputs
    reconstruction_loss = sum((diff^2), axis=[1,2])
    kl_loss = -0.5 * sum(1 + z_log_var - z_mean^2 - exp(z_log_var), axis=1)
    total_loss = mean(reconstruction_loss + kl_loss)
    """
    def call(self, inputs):
        inputs_data, outputs, z_mean, z_log_var = inputs

        diff_squared = tf.square(inputs_data - outputs)
        reconstruction_loss = tf.reduce_sum(diff_squared, axis=[1,2])

        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1
        )

        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)

        return outputs

class LSTMVAE:
    def __init__(self, input_dim, timesteps=1, latent_dim=16, intermediate_dim=32):
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self._build_model()

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.timesteps, self.input_dim))

        # Энкодер
        x = layers.LSTM(self.intermediate_dim)(inputs)
        x = layers.Dropout(0.2)(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)

        # Используем зарегистрированную функцию для семплирования
        z = layers.Lambda(sampling_func, name="z_sampling")([z_mean, z_log_var])

        # Декодер
        decoder_inputs = layers.RepeatVector(self.timesteps)(z)
        decoder_x = layers.LSTM(self.intermediate_dim, return_sequences=True)(decoder_inputs)
        decoder_x = layers.Dropout(0.2)(decoder_x)
        outputs = layers.TimeDistributed(layers.Dense(self.input_dim, activation='sigmoid'))(decoder_x)

        # Лосс-слой
        loss_outputs = VAELossLayer()([inputs, outputs, z_mean, z_log_var])

        self.model = models.Model(inputs, loss_outputs)
        self.model.compile(optimizer='adam')

    def train(self, X_train, epochs=10, batch_size=256):
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(X_train, None,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.1,
                       callbacks=[es],
                       verbose=1)

    def reconstruct_error(self, X):
        reconstructed = self.model.predict(X, verbose=0)
        mse = ((X - reconstructed)**2).mean(axis=(1,2))
        return mse