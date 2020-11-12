import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--outfile", "-o", type=str, required=True)
    args = parser.parse_args()

    mdl = K.models.load_model(args.model_path)
    # mdl = model.SimpleRealNVP([28, 28, 1], 8)
    # mdl.load_weights(args.model_path)


    dataset = K.datasets.mnist
    (_, _), (x_test, _) = dataset.load_data()
    x_test = x_test.astype("float32")
    x_test = x_test / 255.
    x_test = np.reshape(x_test, [-1, 28, 28, 1])

    rand_idx = np.random.randint(0, x_test.shape[0], (4, ))
    x_corners = x_test[rand_idx, ...]

    z_corners = mdl(x_corners, training=False)

    x = np.linspace(0., np.pi / 2., 5)
    y = np.linspace(0., np.pi / 2., 5)
    x_grid, y_grid = np.meshgrid(x, y)
    x_flat = np.reshape(x_grid, (-1, ))  # [25]
    y_flat = np.reshape(y_grid, (-1, ))

    coeff = np.stack([
        np.cos(x_flat) * np.cos(y_flat), 
        np.cos(x_flat) * np.sin(y_flat), 
        np.sin(x_flat) * np.cos(y_flat), 
        np.sin(x_flat) * np.sin(y_flat), 
    ], axis=1)  # [25, 4]

    expand_axes = [1 for _ in [28, 28, 1]]
    coeff = np.reshape(coeff, [25, 4] + expand_axes)  # [25, 4, 1...]

    z_manifold = np.sum(coeff * z_corners, axis=1)  # [25, ...]
    
    # vis
    x_mani = np.sum(coeff * x_corners, axis=1)
    x_mani = np.clip(x_mani, 0., 1.)
    img = np.empty((5 * 28, 5 * 28, 1), dtype="float32")
    for i in range(5):
        for j in range(5):
            img[i*28:(i+1)*28, j*28:(j+1)*28, :] = np.reshape(z_manifold[i + 5 * j, :], (28, 28, 1))
    
    plt.imshow(img, cmap="Greys_r")
    plt.colorbar()
    plt.show()   


    # error!?
    # x_manifold = mdl.inverse(z_manifold)
    x_manifold = z_manifold
    h, w = 28, 28

    print(len(mdl.couplings))
    for i, layer in enumerate(reversed(mdl.couplings)):
        # x_manifold = layer.inverse(x_manifold)
        parity = (len(mdl.couplings) - i - 1) % 2
        mask = np.array([[parity, 1 - parity] * (w // 2), 
                         [1 - parity, parity] * (w // 2)], dtype="float32")
        mask = np.tile(mask, (h // 2, 1))
        mask = mask[:, :, np.newaxis]

        y_masked = x_manifold * mask
        s = tf.exp(-layer.nn_s(y_masked))
        t = layer.nn_t(y_masked)
        x_manifold = y_masked + (1. - mask) * ((x_manifold - t) * s)

    x_visualize = np.reshape(x_manifold, [5, 5, -1])
    x_visualize = np.clip(x_visualize, 0., 1.)

    img = np.zeros((5 * 28, 5 * 28, 1), dtype="float32")
    for i in range(5):
        for j in range(5):
            img[i*28:(i+1)*28, j*28:(j+1)*28, :] = np.reshape(x_visualize[i, j, :], (28, 28, 1))
    
    plt.imshow(img, cmap="Greys_r")
    plt.colorbar()
    plt.show()