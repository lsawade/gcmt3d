"""
Tests for the specfem source generation class.


copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""


import os
from gcmt3d.data.management.specfem_sources import SpecfemSources
import pytest


# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")
CMTFILE = os.path.join(DATA_DIR, "CMTSOLUTION")



class TestIO(object):
    '''
    Testing the IO of the Specfemsources class.
    '''