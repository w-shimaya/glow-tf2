import unittest
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import model

class InversibleTest(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-5

    def test_nice_inverse(self):
        test_tensor = np.array([[1., 0.], [0., 1.], [-1., 1.]], dtype="float32")

        nice = model.NICE(2, 4)
        result_tensor = nice.inverse(nice(test_tensor))
        self.assertTrue(
            tf.reduce_all(
                tf.abs(result_tensor - test_tensor) < self.eps
            ).numpy()
        )

    def test_realnvp_inverse(self):
        test_tensor = np.array([
            [[1., 0.], 
             [0., 1.]], 
            [[-1., 1.], 
             [1., -1.]],
            [[0.2, 0.5], 
             [0.3, 0.8]], 
        ], dtype="float32")
        test_tensor = test_tensor[:, :, :, np.newaxis]

        realnvp = model.SimpleRealNVP((2, 2, 1), 4)
        result_tensor = realnvp.inverse(realnvp(test_tensor))
        self.assertTrue(
            tf.reduce_all(
                tf.abs(result_tensor - test_tensor) < self.eps
            ).numpy()
        )

    def test_realnvp_inverse_4x4(self):
        test_tensor = np.array([
            [[[0.1, 0.3], [0.2, 0.1], [0.3, 1.0], [0.4, 0.4]], 
             [[0.2, 0.1], [0.3, 0.5], [0.4, 0.8], [0.5, 1.0]], 
             [[0.3, 0.8], [0.4, 0.2], [0.5, -1.], [0.6, 0.0]],
             [[0.1, 0.4], [-1., 0.5], [0.6, 0.0], [0.7, 0.3]]], 
        ], dtype="float32")

        realnvp = model.SimpleRealNVP((4, 4, 2), 6)
        result_tensor = realnvp.inverse(realnvp(test_tensor))
        self.assertTrue(
            tf.reduce_all(
                tf.abs(result_tensor - test_tensor) < self.eps
            ).numpy()
        )

if __name__ == "__main__":
    unittest.main()