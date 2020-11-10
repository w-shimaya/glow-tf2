import tensorflow as tf
import tensorflow.keras as K
import numpy as np

class RandomSampleImageCallback(K.callbacks.Callback):
    def __init__(self, logdir, sample_shape, image_shape):
        super(RandomSampleImageCallback, self).__init__()
        self.writer = tf.summary.create_file_writer(logdir)
        self.sample_shape = sample_shape
        self.image_shape = image_shape
        self.z = tf.constant(np.random.normal(size=[64] + self.sample_shape), dtype="float32")

    def on_epoch_end(self, epoch, logs=None):
        x = self.model.inverse(self.z)
        x = tf.reshape(x, [-1] + self.image_shape)
        with self.writer.as_default():
            tf.summary.image("sample", x, step=epoch)
            tf.summary.scalar("loss", logs["loss"], step=epoch)