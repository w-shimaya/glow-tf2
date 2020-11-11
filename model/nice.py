import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np
from model.layers import AdditiveCoupling
from model.layers import Rescaling

class NICE(K.Model):
    def __init__(self, data_shape, n_coupling, regularize=False):
        super(NICE, self).__init__()
        self.loss_tracker = K.metrics.Mean(name="loss")
        # currently 1-dim data assumed
        assert type(data_shape) == int
        self.data_shape = data_shape
        self.regularize = regularize

        # model def
        self.couplings = []
        for i in range(n_coupling):
            m = self._build_network()
            self.couplings.append(AdditiveCoupling(i % 2, m))
        self.couplings.append(Rescaling())

    def call(self, inputs):
        h = inputs
        for layer in self.couplings:
            h = layer(h)
        return h

    def _build_network(self):
        if self.regularize:
            dense = lambda n: L.Dense(n, activation="relu", kernel_initializer="he_normal", kernel_regularizer="l1", bias_regularizer="l1")
        else:
            dense = lambda n: L.Dense(n, activation="relu", kernel_initializer="he_normal")
        # 4 hidden layer rectified network       
        m = K.Sequential([dense(1000) for _ in range(4)])
        m.add(L.Dense(self.data_shape // 2))
        return m

    def _loss_f(self, h):
        # negetive log-likelihood of standard normal prior
        # loss = 0.5 * tf.reduce_sum(
        #    tf.square(h) + tf.math.log(tf.constant(np.pi, dtype="float32") * 2.), 
        #    axis=-1
        # )

        # logistic prior 
        # note: use softplus to improve stability
        loss = tf.reduce_sum(
                tf.nn.softplus(h) + tf.nn.softplus(-h), 
                axis=-1
        )

        # note: Additive Coupling log-det = 0
        for layer in self.couplings:
            loss += -layer.log_det_jacobian  # note: NEGATIVE log likelihood

        loss = tf.reduce_mean(loss)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self(data, training=True)
            tf.debugging.assert_all_finite(z, "z has nan/inf")
            # compute NLL for each data point
            loss = self._loss_f(z)
            tf.debugging.assert_all_finite(loss, "loss has nan/inf")
            # metrics (NLL)
            self.loss_tracker.update_state(loss)
            # add other losses such as regularization
            loss += sum(self.losses)  
            
        grads = tape.gradient(loss, self.trainable_variables)
        for g in grads:
            tf.debugging.assert_all_finite(g, "grad has nan/inf")
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return { m.name : m.result() for m in self.metrics }

    def inverse(self, inputs):
        h = inputs
        for layer in reversed(self.couplings):
            h = layer.inverse(h)
        return h