import tensorflow as tf
import tensorflow.keras as K

class RandomSampleImageCallback(K.callbacks.Callback):
    def __init__(self, logdir, sample_shape, image_shape):
        super(RandomSampleImageCallback, self).__init__()
        self.writer = tf.summary.create_file_writer(logdir)
        self.sample_shape = sample_shape
        self.image_shape = image_shape

    def on_epoch_end(self, epoch, logs=None):
        z = tf.random.normal([32] + self.sample_shape, dtype="float64")
        x = self.model.inverse(z)
        x = tf.reshape(x, [-1] + self.image_shape)
        with self.writer.as_default():
            tf.summary.image("sample", x, step=epoch)
            tf.summary.scalar("loss", logs["loss"], step=epoch)