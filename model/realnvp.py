import functools
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np
from model.layers import AffineCoupling


class SimpleRealNVP(K.Model):
    """Simplified RealNVP.
    Differences:
    * no multi-scale architecture 
    * no squeezing operation
    * no ResNet
    """
    def __init__(self, data_shape, n_coupling):
        super(SimpleRealNVP, self).__init__()
        self.loss_tracker = K.metrics.Mean(name="loss")
        # assert 3-ch image
        assert len(data_shape) == 3
        self.data_shape = data_shape
        data_dim = functools.reduce(lambda x, y: x * y, data_shape)

        self.couplings = []
        # in original paper [Dinh+, 2017], deep ResNet was used 
        # in the coupling layers
        for i in range(n_coupling):
            nn_s = K.Sequential([L.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal"), 
                                 L.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
                                 L.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
                                 L.Flatten(), 
                                 L.Dense(data_dim, activation="tanh", kernel_initializer="glorot_normal"),
                                 L.Reshape(self.data_shape)])
            nn_t = K.Sequential([L.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal"), 
                                 L.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
                                 L.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
                                 L.Flatten(), 
                                 L.Dense(data_dim, kernel_initializer="glorot_normal"),
                                 L.Reshape(self.data_shape)])

            self.couplings.append(AffineCoupling(i % 2, nn_s, nn_t))

    def call(self, inputs):
        h = inputs
        for layer in self.couplings:
            h = layer(h)
        return h

    def inverse(self, inputs):
        h = inputs
        for layer in reversed(self.couplings):
            h = layer.inverse(h)
        return h

    def _loss_f(self, h):
        # nll of standard normal prior
        loss = 0.5 * tf.reduce_mean(
            tf.reduce_sum(
                tf.square(h) + tf.math.log(tf.constant(np.pi, dtype="float64") * 2.),
                axis=[1, 2, 3]  # sum over all pixels
            )
        )
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self(data, training=True)
            loss = self._loss_f(z)
            loss += sum(self.lossses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return { m.name : m.result() for m in self.metrics }