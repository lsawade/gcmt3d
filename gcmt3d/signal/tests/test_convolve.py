import os
import inspect
import unittest
import numpy as np
import gcmt3d.signal.convolve as cv

def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(os.path.abspath(
    inspect.getfile(inspect.currentframe())), 4)
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")



class TestWavelets(unittest.TestCase):


    def test_gaussian(self):
        """Tests gaussian function in convolve.py """

        t = np.array([0, 1, 2, 3, 4, 5])
        to = 2.5
        sig = 1
        amp = 1

        g = cv.gaussian(t, to, sig, amp)

        np.testing.assert_array_almost_equal(g, np.array([0.04393693,
                                                          0.32465247,
                                                          0.8824969,
                                                          0.8824969,
                                                          0.32465247,
                                                          0.04393693]))

    def test_bimodal_gaussian(self):
        """Tests bimodal gaussian function in convolve.py"""

        # Setup parameters
        t = np.array([0, 1, 2, 3, 4, 5])
        to1 = 2.5
        sig1 = 1
        amp1 = 1
        to2 = 4
        sig2 = 0.5
        amp2 = 0.5

        # Compute gaussian
        g = cv.bimodal_gaussian(t, amp1=amp1, to1=to1, sig1=sig1,
                                amp2=amp2, to2=to2, sig2=sig2)

        # Test it
        np.testing.assert_array_almost_equal(g, np.array([0.043937,
                                                          0.324652,
                                                          0.882665,
                                                          0.950165,
                                                          0.824652,
                                                          0.111605]))

    def test_wavelet(self):
        """Testing the wavelet creation."""