import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model
import callbacks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", "-l", type=str, required=True)
    parser.add_argument("--model", "-m", choices=["nice", "realnvp"], required=True)
    args = parser.parse_args()

    # prepare mnist dataset
    dataset = K.datasets.mnist
    (x_train, _), (x_test, _) = dataset.load_data()
    x_train = x_train.astype(np.float32)
    # reduce the impact of boundary effects
    x_train = .05 + .95 * x_train / 256.
    # reshape to fit inputs of the model
    if args.model == "nice":
        x_train = np.reshape(x_train, [-1, 28*28])
    elif args.model == "realnvp":
        x_train = np.reshape(x_train, [-1, 28, 28, 1])
    # tf dataset
    x_tensor = tf.data.Dataset.from_tensor_slices(x_train)
    x_tensor = x_tensor.batch(32)

    # latent dim
    if args.model == "nice":
        sample_shape = [28*28]
    elif args.model == "realnvp":
        sample_shape = [28, 28, 1]

    # set callback
    callback = callbacks.RandomSampleImageCallback(
        logdir=args.logdir, 
        sample_shape=sample_shape, 
        image_shape=[28, 28, 1]
    )

    # train the model
    if args.model == "nice":
        mdl = model.NICE(28 * 28, 4)
    elif args.model == "realnvp":
        mdl = model.SimpleRealNVP([28, 28, 1], 4)
    # compile
    mdl.compile(optimizer=K.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.9, epsilon=1e-4))
    mdl.fit(x_tensor, epochs=100, callbacks=[callback])