import tensorflow as tf
import tensorflow.keras as K
import numpy as np

class RandomSampleImageCallback(K.callbacks.Callback):
    def __init__(self, logdir, sample_shape, image_shape, writer):
        super(RandomSampleImageCallback, self).__init__()
        self.writer = writer
        self.sample_shape = sample_shape
        self.image_shape = image_shape
        self.z = tf.constant(np.random.normal(size=[64] + self.sample_shape), dtype="float32")

    def on_epoch_end(self, epoch, logs=None):
        x = self.model.inverse(self.z)
        x = tf.reshape(x, [-1] + self.image_shape)
        with self.writer.as_default():
            tf.summary.image("sample", x, step=epoch)
            tf.summary.scalar("loss", logs["loss"], step=epoch)

class InterpolationCallback(K.callbacks.Callback):
    def __init__(self, logdir, sample_shape, corner_images, writer):
        super(InterpolationCallback, self).__init__()
        x = np.linspace(0., np.pi / 2., 5)
        y = np.linspace(0., np.pi / 2., 5)
        x_grid, y_grid = np.meshgrid(x, y)
        x_flat = np.reshape(x_grid, (-1, ))
        y_flat = np.reshape(y_grid, (-1, ))
        coeff = np.stack([
            np.cos(x_flat) * np.cos(y_flat), 
            np.cos(x_flat) * np.sin(y_flat), 
            np.sin(x_flat) * np.cos(y_flat), 
            np.sin(x_flat) * np.sin(y_flat), 
        ], axis=1)
        self.coeff = np.reshape(coeff, [25, 4, 1, 1, 1])
        self.corner_images = corner_images
        self.image_shape = corner_images.shape[1:]
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        z_corners = mdl(x_corner_images, training=False)
        z_manifold = np.sum(self.coeff * z_corners, axis=1)
        x_manifold = mdl.inverse(z_corners)
        x_visualize = np.reshape(x_manifold, [5, 5] + self.image_shape)

        h, w = self.image_shape[0], self.image_shape[1]
        img = np.empty((5 * self.image_shape[0], 
                        5 * self.image_shape[1], 1), dtype="float32")
        for i in range(5):
            for j in range(5):
                img[i*h:(i+1)*h, j*w:(j+1)*w, :] = x_visualize[i, j, ...]
                
        with self.writer.as_default():
            tf.summary.image("manifold", img[np.newaxis, ...], step=epoch)
