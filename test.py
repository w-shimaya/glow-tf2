import unittest
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import model

class InversibleTest(unittest.TestCase):
    def setUp(self):
        self.nice = model.nice.NICE(2, 4)
        self.realnvp = model.realnvp.SimpleRealNVP((2, 2, 1), 4)
        self.eps = 1e-5

    def test_nice_inverse(self):
        test_tensor = np.array([[1., 0.], [0., 1.], [-1., 1.]], dtype="float32")
        result_tensor = self.nice.inverse(self.nice(test_tensor))
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
        result_tensor = self.realnvp.inverse(self.realnvp(test_tensor))
        self.assertTrue(
            tf.reduce_all(
                tf.abs(result_tensor - test_tensor) < self.eps
            ).numpy()
        )

if __name__ == "__main__":
    unittest.main()