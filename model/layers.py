import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np


class AdditiveCoupling(L.Layer):
    def __init__(self, m):
        super(AdditiveCoupling, self).__init__()
        self.m = m

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0
        self.split_size = input_shape[-1] // 2

    def call(self, inputs):
        x1, x2 = tf.split(inputs, [self.split_size, self.split_size], axis=-1)
        y1 = x1
        y2 = x2 + self.m(x1)
        return tf.concat([y1, y2], axis=-1)
    
    def inverse(self, inputs):
        y1, y2 = tf.split(inputs, [self.split_size, self.split_size], axis=-1)
        x1 = y1
        x2 = y2 - self.m(y1)
        return tf.concat([x1, x2], axis=-1)

class AffineCoupling(L.Layer):
    def __init__(self, parity, nn_s, nn_t):
        super(AffineCoupling, self).__init__()
        assert parity == 0 or parity == 1
        self.parity = parity
        # nn for scale and translation
        self.nn_s = nn_s
        self.nn_t = nn_t

    def build(self, input_shape):
        # checker board masks
        assert len(input_shape) == 4 and input_shape[1] % 2 == 0 and input_shape[2] % 2 == 0
        h, w = input_shape[1], input_shape[2]
        mask = np.array([[self.parity, 1 - self.parity] * (w // 2), 
                         [1 - self.parity, self.parity] * (w // 2)], dtype="float32")
        mask = np.tile(mask, (h // 2, 1))
        mask = mask[:, :, np.newaxis]
        self.mask = tf.constant(mask)

    def call(self, inputs):
        x_masked = inputs * self.mask
        s = tf.exp(self.nn_s(x_masked))
        t = self.nn_t(x_masked)
        y = x_masked + (1. - self.mask) * (inputs * s + t)
        
        # add log-det of Jacobian to the NEGATIVE log-likelihood loss
        self.add_loss(
            tf.reduce_mean(
                -tf.reduce_sum(s * (1. - self.mask), axis=[1, 2, 3])
            )
        )

        return y

    def inverse(self, inputs):
        y_masked = inputs * self.mask
        s = tf.exp(-self.nn_s(y_masked))
        t = self.nn_t(y_masked)
        x = y_masked + (1. - self.mask) * ((inputs - t) * s)
        return x


class Rescaling(L.Layer):
    def __init__(self):
        super(Rescaling, self).__init__()
    
    def build(self, input_shape):
        self.log_s = tf.Variable(initial_value=tf.zeros_initializer()(shape=input_shape[1:], dtype="float32"), trainable=True)
    
    def call(self, inputs):
        # log-det of Jacobian
        self.add_loss(-tf.reduce_sum(self.log_s))  # note: NEGATIVE log-likelihood
        return inputs * tf.exp(self.log_s)

    def inverse(self, inputs):
        return inputs * tf.exp(-self.log_s)

class Reverse(L.Layer):
    def call(self, inputs):
        return tf.reverse(inputs, axis=[-1])

    def inverse(self, inputs):
        return tf.reverse(inputs, axis=[-1])

