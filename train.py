import argparse
import os
import sys
import functools
import pprint
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds
import model
import callbacks

def get_mnist_dataset(model):
    # prepare mnist dataset
    dataset = K.datasets.mnist
    (x_train, _), (x_test, _) = dataset.load_data()
    x_train = x_train.astype(np.float32)
    # dequantization (?)
    x_train = x_train + np.random.normal(size=x_train.shape)
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    # reduce the impact of boundary effects (?)
    x_train = .05 + .95 * x_train
    # reshape to fit inputs of the model
    if model == "nice":
        x_train = np.reshape(x_train, [-1, 28*28])
    elif model == "realnvp":
        x_train = np.reshape(x_train, [-1, 28, 28, 1])
    # tf dataset
    x_tensor = tf.data.Dataset.from_tensor_slices(x_train)
    return x_tensor

def get_dsprites_dataset(model):
    def preprocess(x):
        ret = tf.cast(x["image"], "float32")
        ret = ret + tf.random.normal((64, 64, 1))
        ret /= 255.
        if model == "nice":
            ret = tf.reshape(ret, (-1, 64 * 64))
        return ret
    dataset = tfds.load("dsprites", data_dir="./tfds")
    dataset = dataset["train"].map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", "-l", type=str, required=True)
    parser.add_argument("--model", "-m", choices=["nice", "realnvp"], required=True)
    parser.add_argument("--vdl", "-v", type=int, required=True)
    parser.add_argument("--epoch", "-e", type=int, default=500)
    parser.add_argument("--learning-rate", "-r", type=float, default=1e-3)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--dataset", "-d", choices=["mnist", "dsprites"], required=True)
    args = parser.parse_args()

    # set visible devices
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[args.vdl], "GPU")

    if args.dataset == "mnist":
        x_tensor = get_mnist_dataset(args.model)
        image_shape = [28, 28, 1]
    elif args.dataset == "dsprites":
        # prepare dsprites dataset
        x_tensor = get_dsprites_dataset(args.model)
        image_shape = [64, 64, 1]

    x_tensor = x_tensor.shuffle(buffer_size=737280, reshuffle_each_iteration=True)
    x_tensor = x_tensor.batch(args.batch_size)
    x_tensor = x_tensor.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # latent dim
    if args.model == "nice":
        sample_shape = [functools.reduce(lambda x, y: x * y, image_shape, 1)]
    elif args.model == "realnvp":
        sample_shape = image_shape

    # prepare a directory to save the model
    save_dir = os.path.join(args.logdir, "saved_model")
    if os.path.exists(save_dir):
        print(f"{save_dir} already exists. Overwrite? [y/N]")
        c = input()
        if not c.upper() == "Y":
            print("abort")
            exit()
    else:
        # make directory to save the model
        os.makedirs(save_dir)

    # set callback
    image_callback = callbacks.RandomSampleImageCallback(
        logdir=args.logdir, 
        sample_shape=sample_shape, 
        image_shape=image_shape
    )
    save_model_callback = K.callbacks.ModelCheckpoint(
        os.path.join(save_dir, "epoch-{epoch:02d}"), 
        monitor="loss", 
        save_best_only=True, 
        save_weights_only=False
    )

    # train the model
    if args.model == "nice":
        mdl = model.NICE(sample_shape[0], 4, regularize=True)
    elif args.model == "realnvp":
        mdl = model.SimpleRealNVP(image_shape, 4)
    # compile
    mdl.compile(optimizer=K.optimizers.Adam(args.learning_rate, beta_1=0.9, beta_2=0.01, epsilon=1e-4))
    mdl.fit(x_tensor, epochs=args.epoch, callbacks=[image_callback, save_model_callback])

    # debug
    with open(os.path.join(args.logdir, "var.txt"), "w") as f:
        pprint.pprint(mdl.trainable_weights, stream=f)