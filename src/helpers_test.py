import unittest

from numpy.testing import *

from src.helpers import *


class TestHelpers(unittest.TestCase):

    def test_replace_nan_with_mean_columns(self):
        provided = np.array([[1, 3],
                             [float('NaN'), 4],
                             [5, float('Nan')],
                             [10, 20],
                             [float('NaN'), float('NaN')]])
        expected = np.array([[1, 3],
                             [5.333333333333333, 4],
                             [5, 9.0],
                             [10, 20],
                             [5.333333333333333, 9.0]])

        assert_array_almost_equal(replace_nan_with_mean_columns(provided), expected)

    def test_get_nearest_neighbors(self):
        pass

    def test_repair_individual(self):
        provided = [[1, 2, 3, 4]]
        provided_mask = [100, 200, float('NaN'), float('NaN')]

        expected = [[100, 200, 3, 4]]
        repair_individuals(provided, provided_mask)
        assert_array_equal(provided, expected)

    def test_evaluation_function(self):
        self.assertEqual(evaluation_function(np.arange(10), [np.arange(10)]), (1,))
        self.assertEqual(evaluation_function(np.arange(10), [np.arange(10)[::-1]]), (0,))
        self.assertEqual(evaluation_function(np.arange(10), [np.arange(10)[::-1], np.arange(10)]), (0.5,))  # (-1 +1 / 2)


if __name__ == '__main__':
    unittest.main()
