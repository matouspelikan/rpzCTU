import unittest

import numpy as np
import minimax

data = np.load("data_33rpz_03_minimax.npz", allow_pickle=True)
alphabet = data["alphabet"]
images_tst = data["images_test"]
labels_tst = data["labels_test"]
cont = data["cont"].item()


class TestMinimaxCont(unittest.TestCase):
    def test_minimax_strategy_cont(self):
        D1 = cont['M']
        D2 = cont['D']
        q, risk = minimax.minimax_strategy_cont(D1, D2)

        np.testing.assert_almost_equal(q['t1'], -230.44601288127865, decimal=4)
        np.testing.assert_almost_equal(q['t2'], 3705.0796592073866, decimal=4)
        np.testing.assert_equal(q['decision'], np.array([1, 0, 1], dtype=np.int32))
        np.testing.assert_almost_equal(risk, 0.2774251103401184)

    def test_classification_error(self):
        letters = 'MD'
        q = {'t1': -230.44601288127865,
             't2': 3705.0796592073866,
             'decision': np.array([1, 0, 1], dtype=np.int32)}
        images_test_cont, labels_test_cont = minimax.create_test_set(images_tst, labels_tst, letters, alphabet)
        measurements_cont = minimax.compute_measurement_lr_cont(images_test_cont)
        labels_estimated_cont = minimax.classify_2normal(measurements_cont, q)
        error_cont = minimax.classification_error(labels_estimated_cont, labels_test_cont)

        np.testing.assert_almost_equal(error_cont, 0.3167, decimal=4)


if __name__ == "__main__":
    unittest.main()
