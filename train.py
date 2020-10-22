import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model
import callbacks

if __name__ == "__main__":
    K.backend.set_floatx("float64")
    # prepare mnist dataset
    dataset = K.datasets.mnist
    (x_train, _), (x_test, _) = dataset.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_train = np.reshape(x_train, [-1, 28*28])
    x_train = x_train + np.random.normal(size=x_train.shape) / 255.
    x_tensor = tf.data.Dataset.from_tensor_slices(x_train)
    x_tensor = x_tensor.batch(32)

    # set callback
    callback = callbacks.RandomSampleImageCallback(
        logdir="logtest/logistic", 
        sample_shape=[28*28], 
        image_shape=[28, 28, 1]
    )

    # train the NICE model
    nice = model.NICE(28 * 28, 4)
    nice.compile(optimizer=K.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.9, epsilon=1e-4))
    nice.fit(x_tensor, epochs=100, callbacks=[callback])