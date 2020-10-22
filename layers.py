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

class Rescaling(L.Layer):
    def __init__(self):
        super(Rescaling, self).__init__()
    
    def build(self, input_shape):
        self.log_s = tf.Variable(initial_value=tf.zeros_initializer()(shape=input_shape[1:], dtype="float64"), trainable=True)
    
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

