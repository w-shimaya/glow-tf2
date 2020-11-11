import unittest
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import model

class InversibleTest(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-5

    def test_nice_inverse(self):
        test_tensor = np.random.normal(size=(4, 8)).astype("float32")

        nice = model.NICE(8, 4)
        result_tensor = nice.inverse(nice(test_tensor))
        self.assertTrue(
            tf.reduce_all(
                tf.abs(result_tensor - test_tensor) < self.eps
            ).numpy()
        )

    def test_realnvp_inverse(self):
        test_tensor = np.random.normal(size=(4, 16, 16, 3)).astype("float32")

        realnvp = model.SimpleRealNVP((16, 16, 3), 4)
        result_tensor = realnvp.inverse(realnvp(test_tensor))
        self.assertTrue(
            tf.reduce_all(
                tf.abs(result_tensor - test_tensor) < self.eps
            ).numpy()
        )

if __name__ == "__main__":
    unittest.main()