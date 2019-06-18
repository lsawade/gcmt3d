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
import glob

from gcmt3d.data.management import SpecfemSources
from gcmt3d.source import CMTSource
import unittest
import tempfile

# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data/SpecFEMSources")
CMTFILE = os.path.join(DATA_DIR, "CMTSOLUTION")


class Test_SpecfemSources(unittest.TestCase):
    """
    Test class for specfem_sources.py. The main reason for this being necessary
    is that the class includes a mechanism to test error raising.
    The writing is set to be written into the different subdirectories of the
    database skeleton created by the Skeleton class.
    """

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmpdir = self.test_dir.name

        # Create subdirectories for each simulation
        self.attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
                     "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]
        for at in self.attr:
            # create each sim dir
            os.makedirs(os.path.join(self.tmpdir, at))
            os.makedirs(os.path.join(self.tmpdir, at, "DATA"))
            os.makedirs(os.path.join(self.tmpdir, at, "OUTPUT_FILES"))

        self.cmt = CMTSource.from_CMTSOLUTION_file(CMTFILE)

    def tearDown(self):
        # Close the file, the directory will be removed after the test
        self.test_dir.cleanup()

    def test_noCMTinput(self):
        """Testing no input.
        """
        cmt = 3
        npar = 2
        with self.assertRaises(ValueError):
            SpecfemSources(cmt, npar)

    def test_bad_npar_input(self):
        """Testing no input.
        """
        cmt = self.cmt
        npar = 2
        with self.assertRaises(ValueError):
            SpecfemSources(cmt, npar)

    def test_standard_init(self):
        """Testing basic initalization. How does the tmpdir thing work???"""

        # Basic standard parameters
        cmt = self.cmt
        npar = 9
        dm = 10.0 ** 24
        dx = 2.
        ddeg = 0.01
        outputdir = self.tmpdir

        sfsource = SpecfemSources(cmt, npar=npar, dm=dm, dx=dx, ddeg=ddeg,
                                  outdir=outputdir)

        # Assert that correct values are assigned.
        self.assertEqual(sfsource.cmt, cmt)
        self.assertEqual(sfsource.dm, dm)
        self.assertEqual(sfsource.dx, dx)
        self.assertEqual(sfsource.ddeg, ddeg)
        self.assertEqual(sfsource.outdir, outputdir)

    def test_write_6par(self):
        """Testing the writing to file
        """

        # Basic standard parameters
        cmt = CMTSource.from_CMTSOLUTION_file(CMTFILE)
        npar = 6
        dm = 10.0 ** 24
        dx = 2000.
        ddeg = 0.01
        outputdir = self.tmpdir

        sfsource = SpecfemSources(cmt, npar=npar, dm=dm, dx=dx, ddeg=ddeg,
                                  outdir=outputdir)

        sfsource.write_sources()

        # Get files of the written files
        written_files = glob.glob(outputdir + "CMT*/DATA/CMTSOLUTION")

        # Get files and names of the test files
        test_files = glob.glob(DATA_DIR + "/*")
        test_names = [os.path.basename(x) for x in test_files]

        # Assert each file is equal to the written one
        for file in written_files:

            # Get filename
            name = os.path.basename(file)

            # Find corresponding file in test directory
            index = test_names.index(name)

            # Opening both files and notebooks them string for string
            # somehow a simple file compare was not do-able.
            with open(file, 'r') as written_file:
                with open(test_files[index], 'r') as test_file:

                    # Get lines
                    for line in written_file:
                        written_line = line.split()
                        test_line = test_file.readline().split()

                        # Get Strings
                        for index, teststring in enumerate(test_line):
                            assert teststring == str(written_line[index])

    def test_write_9par(self):
        """Testing the writing to file
        """

        # Basic standard parameters
        cmt = CMTSource.from_CMTSOLUTION_file(CMTFILE)
        npar = 9
        dm = 10.0 ** 24
        dx = 2000.
        ddeg = 0.02
        outputdir = self.tmpdir

        sfsource = SpecfemSources(cmt, npar=npar, dm=dm, dx=dx, ddeg=ddeg,
                                  outdir=outputdir)

        sfsource.write_sources()

        # Get files and names of the written files
        written_files = glob.glob(outputdir + "CMT*/DATA/CMTSOLUTION")

        # Get files and names of the test files
        test_files = glob.glob(DATA_DIR + "/*")
        test_names = [os.path.basename(x) for x in test_files]

        # Assert each file is equal to the written one
        for file in written_files:

            # Get filename
            name = os.path.basename(file)

            # Find corresponding file in test directory
            index = test_names.index(name)

            # Opening both files and notebooks them string for string
            # somehow a simple file compare was not do-able.
            with open(file, 'r') as written_file:
                with open(test_files[index], 'r') as test_file:

                    # Getting the lines
                    for line in written_file:
                        written_line = line.split()
                        test_line = test_file.readline().split()

                        # Getting the strings
                        for index, teststring in enumerate(test_line):
                            assert teststring == str(written_line[index])


if __name__ == '__main__':
    unittest.main()
