import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L

# AdditiveCoupling has the Jacobian whose log-det is 1
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
        self.log_s = tf.Variable(initial_value=tf.zeros_initializer()(shape=input_shape, dtype="float64"), trainable=True)
    
    def call(self, inputs):
        # log-det of Jacobian
        self.add_loss(tf.reduce_logsumexp(self.log_s))
        return inputs * tf.exp(self.log_s)

    def inverse(self, inputs):
        return inputs * tf.exp(-self.log_s)

class Reverse(L.Layer):
    def call(self, inputs):
        return tf.reverse(inputs, axis=[-1])

    def inverse(self, inputs):
        return tf.reverse(inputs, axis=[-1])


class NICE(K.Model):
    def __init__(self, data_shape, n_coupling):
        super(NICE, self).__init__()

        self.data_shape = data_shape
        self.couplings = []
        for _ in range(n_coupling):
            m = K.Sequential([L.Dense(128, activation="relu"), 
                              L.Dense(data_shape // 2, activation="relu")])
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
        # log-likelihood of standard normal prior
        loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                tf.square(h) + tf.log(np.pi * 2.), 
                axis=-1
            )
        )
        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self(data, training=True)
            loss = self._loss_f(z)
            assert len(self.losses) == 1
            loss += sum(self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return { m.name : m.result() for m in self.metrics }

    def inverse(self, inputs):
        h = self.rescaling.inverse(inputs)
        for layer in reversed(self.couplings):
            h = self.reverse_layer.inverse(h)
            h = layer.inverse(h)
        return h