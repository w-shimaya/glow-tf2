import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import model
import callbacks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", "-l", type=str, required=True)
    parser.add_argument("--model", "-m", choices=["nice", "realnvp"], required=True)
    parser.add_argument("--vdl", "-v", type=int, required=True)
    args = parser.parse_args()

    # set visible devices
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[args.vdl], "GPU")

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
    if args.model == "nice":
        x_train = np.reshape(x_train, [-1, 28*28])
    elif args.model == "realnvp":
        x_train = np.reshape(x_train, [-1, 28, 28, 1])
    # tf dataset
    x_tensor = tf.data.Dataset.from_tensor_slices(x_train)
    x_tensor = x_tensor.batch(64)

    # latent dim
    if args.model == "nice":
        sample_shape = [28*28]
    elif args.model == "realnvp":
        sample_shape = [28, 28, 1]

    # prepare a directory to save the model
    save_dir = os.path.join(args.logdir, "saved_model")
    if os.path.exists(save_dir):
        print(f"{save_dir} already exists. Overwrite? [y/N]")
        c = input()
        if not c.upper() == "Y":
            print("abort")
            exit()

    # make directory to save the model
    os.makedirs(save_dir)

    # set callback
    image_callback = callbacks.RandomSampleImageCallback(
        logdir=args.logdir, 
        sample_shape=sample_shape, 
        image_shape=[28, 28, 1]
    )
    save_model_callback = K.callbacks.ModelCheckpoint(
        os.path.join(save_dir, "epoch-{epoch:02d}"), 
        monitor="loss", 
        save_best_only=True
    )

    # train the model
    if args.model == "nice":
        mdl = model.NICE(28 * 28, 4, regularize=True)
    elif args.model == "realnvp":
        mdl = model.SimpleRealNVP([28, 28, 1], 8)
    # compile
    mdl.compile(optimizer=K.optimizers.Adam(1e-7, beta_1=0.9, beta_2=0.01, epsilon=1e-4))
    mdl.fit(x_tensor, epochs=500, callbacks=[image_callback, save_model_callback])

    mdl.save(os.path.join(save_dir, "model"))