import unittest
from scripts.data_helpers_night_lights import *


class TestDataHelpers(unittest.TestCase):

    def test_calibration(self):

        c0, c1, c2 = get_calib_coefficients('testtestF152007', 2007)
        self.assertEqual(c0, 1.3606)
        self.assertEqual(c1, 1.2974)
        self.assertEqual(c2, -0.0045)


if __name__ == '__main__':
    unittest.main()
