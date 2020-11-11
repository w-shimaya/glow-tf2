import abc
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np

class CouplingBase(L.Layer, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def inverse(self, inputs):
        pass

    @property
    @abc.abstractmethod
    def log_det_jacobian(self):
        pass


class AdditiveCoupling(CouplingBase):
    def __init__(self, parity, m):
        super(AdditiveCoupling, self).__init__()
        self.m = m
        self.parity = parity

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0
        self.data_dim = input_shape[-1]

    @property
    def log_det_jacobian(self):
        return 0.

    def _concat_alt(self, t1, t2):
        # concat t1 and t2 in alternate fashion
        # e.g., [t1_1, t2_1, t1_2, ...]
        t1_ex = tf.expand_dims(t1, axis=2)
        t2_ex = tf.expand_dims(t2, axis=2)
        t = tf.concat([t1_ex, t2_ex], axis=2)
        return tf.reshape(t, [-1, self.data_dim])

    def call(self, inputs):
        x1, x2 = inputs[:, self.parity::2], inputs[:, 1 - self.parity::2]
        y1 = x1
        y2 = x2 + self.m(x1)
        if self.parity == 0:
            return self._concat_alt(y1, y2)
        else:
            return self._concat_alt(y2, y1)
    
    def inverse(self, inputs):
        y1, y2 = inputs[:, self.parity::2], inputs[:, 1 - self.parity::2]
        x1 = y1
        x2 = y2 - self.m(y1)
        if self.parity == 0:
            return self._concat_alt(x1, x2)
        else:
            return self._concat_alt(x2, x1)

class AffineCoupling(CouplingBase):
    def __init__(self, parity, nn_s, nn_t):
        super(AffineCoupling, self).__init__()
        assert parity == 0 or parity == 1
        self.parity = parity
        # nn for scale and translation
        self.nn_s = nn_s
        self.nn_t = nn_t
        self._log_det_jacobian = 0.

    @property
    def log_det_jacobian(self):
        return self._log_det_jacobian

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
        self._log_det_jacobian = tf.reduce_sum(s * (1. - self.mask), axis=[1, 2, 3])

        return y

    def inverse(self, inputs):
        y_masked = inputs * self.mask
        s = tf.exp(-self.nn_s(y_masked))
        t = self.nn_t(y_masked)
        x = y_masked + (1. - self.mask) * ((inputs - t) * s)
        return x


class Rescaling(CouplingBase):
    def __init__(self):
        super(Rescaling, self).__init__()
    
    @property
    def log_det_jacobian(self):
        return tf.reduce_sum(self.log_s)

    def build(self, input_shape):
        self.log_s = tf.Variable(initial_value=tf.zeros_initializer()(shape=[1, input_shape[1]], dtype="float32"), trainable=True)
        super(Rescaling, self).build(input_shape)
    
    def call(self, inputs):
        return inputs * tf.exp(self.log_s)

    def inverse(self, inputs):
        return inputs * tf.exp(-self.log_s)