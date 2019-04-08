"""
Tests for the specfem source generation class.


copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""


import os
import inspect
from gcmt3d.data.management import SpecfemSources
import unittest


# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")
CMTFILE = os.path.join(DATA_DIR, "CMTSOLUTION")



class TestIO(unittest.TestCase):
    '''
    Testing the IO of the Specfemsources class.
    '''

    def test_noCMTinput(self):
        """Testing no input.
        """
        cmt = 3
        npar = 2
        with self.assertRaises(ValueError):
            SpecfemSources(cmt,npar)


    def test_bad_npar_input(self):
        """Testing no input.
        """
        cmt = 3
        npar = 2
        with self.assertRaises(ValueError):
            SpecfemSources(cmt,npar)