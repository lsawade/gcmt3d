"""

Test suite for Specfem driver.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2019

"""

import unittest
import tempfile
import os
import inspect
from gcmt3d.runSF3D.runSF3D import DATAFixer


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


# Most generic way to get the data folder path.
TESTBASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TESTBASE_DIR, "data")


class TestParfileFixer(unittest.TestCase):
    """Class that handles testing of the Data Base structure creator"""

    def test_replace_var(self):
        """Tests whether the replace variable function works.
        """

        # Test par file
        parfile = os.path.join(DATA_DIR, "Par_file")

        # Set new value
        newval = 4

        with tempfile.NamedTemporaryFile(mode='w+t') as tmpfile:
            # Copy parfile content to temporary file.
            print(tmpfile)
            with open(parfile) as PF:
                for line in PF:
                    tmpfile.write(line)

            # Back to start of file
            tmpfile.seek(0)

            # Replace value
            DATAFixer.replace_varval(tmpfile.name, "NCHUNKS", newval)

            # Check if value is
            val = DATAFixer.get_val(tmpfile.name, "NCHUNKS")

            # Check if value has changed
            self.assertTrue(newval == int(val))

    def test_get_val(self):
        """ Tests static method get_val"""

        # Test par file
        parfile = os.path.join(DATA_DIR, "Par_file")

        # Solution
        original_value = 6

        # Get the value from the file
        val = DATAFixer.get_val(parfile, 'NCHUNKS')

        print(val)
        # Check if values match
        self.assertTrue(original_value == int(val))
