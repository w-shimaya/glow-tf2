import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np
from model.layers import AdditiveCoupling
from model.layers import Reverse
from model.layers import Rescaling


class NICE(K.Model):
    def __init__(self, data_shape, n_coupling):
        super(NICE, self).__init__()
        self.loss_tracker = K.metrics.Mean(name="loss")
        # currently 1-dim data assumed
        assert type(data_shape) == int
        self.data_shape = data_shape

        # model def
        self.couplings = []
        for _ in range(n_coupling):
            m = K.Sequential([L.Dense(128, activation="relu", kernel_initializer="he_normal"), 
                              L.Dense(128, activation="relu", kernel_initializer="he_normal"), 
                              L.Dense(128, activation="relu", kernel_initializer="he_normal"), 
                              L.Dense(128, activation="relu", kernel_initializer="he_normal"), 
                              L.Dense(data_shape // 2)])
            self.couplings.append(AdditiveCoupling(m=m))
        self.reverse_layer = Reverse()
        self.rescaling = Rescaling()

    def call(self, inputs):
        h = inputs
        for layer in self.couplings:
            h = layer(h)
            h = self.reverse_layer(h)
        h = self.rescaling(h)
        return h

    def _loss_f(self, h):
        # negetive log-likelihood of standard normal prior
        # loss = 0.5 * tf.reduce_mean(
        #     tf.reduce_sum(
        #         tf.square(h) + tf.math.log(tf.constant(np.pi, dtype="float64") * 2.), 
        #         axis=-1
        #     )
        # )

        # logistic prior
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.math.log(1 + tf.exp(h)) + tf.math.log(1 + tf.exp(-h)), 
                axis=-1
            )
        )
        # Additive Coupling log-det = 1
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self(data, training=True)
            loss = self._loss_f(z)
            loss += sum(self.losses)  
        grads = tape.gradient(loss, self.trainable_variables)
        # metrics (NLL)
        self.loss_tracker.update_state(loss)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return { m.name : m.result() for m in self.metrics }

    def inverse(self, inputs):
        h = self.rescaling.inverse(inputs)
        for layer in reversed(self.couplings):
            h = self.reverse_layer.inverse(h)
            h = layer.inverse(h)
        return h