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

    def test_mask_matrix(self):
        provided = np.array([[1, 2, 3, 4],
                             [10, 20, 30, 40],
                             [5, 10, 15, 20]])
        provided_mask = np.array([[100, 200, float('NaN'), float('NaN')],
                                  [float('NaN'), float('NaN'), float('NaN'), float('NaN')],
                                  [50, 100, float('NaN'), float('NaN')]])

        expected = np.array([[100, 200, 3, 4],
                             [10, 20, 30, 40],
                             [50, 100, 15, 20]])

        assert_array_equal(mask_matrix(provided, provided_mask), expected)


if __name__ == '__main__':
    unittest.main()
