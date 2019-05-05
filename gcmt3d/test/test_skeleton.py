"""

Test suite for skeleton creator.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2019

"""

import unittest
from gcmt3d.data.management.skeleton import DataBaseSkeleton
from gcmt3d.source import CMTSource
import tempfile
import os
import inspect


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(os.path.abspath(
    inspect.getfile(inspect.currentframe())), 1)
DATA_DIR = os.path.join(TESTBASE_DIR, "data")


class TestSkeleton(unittest.TestCase):
    """Class that handles testing of the Data Base structure creator"""

    def setUp(self):
        # Create a temporary specfem like directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.specfem_dir = self.test_dir.name
        os.makedirs(os.path.join(self.specfem_dir, "DATA"))
        os.makedirs(os.path.join(self.specfem_dir, "DATABASES_MPI"))
        os.makedirs(os.path.join(self.specfem_dir, "OUTPUT_FILES"))
        os.makedirs(os.path.join(self.specfem_dir, "bin"))

    def tearDown(self):
        # Close the file, the directory will be removed after the test
        self.test_dir.cleanup()

    def test_IO(self):
        """Testing if strings are set correctly."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=os.path.join(DATA_DIR, "CMTSOLUTION"),
                                  specfem_dir=self.specfem_dir, verbose=False)

            self.assertEqual(DB.specfem_dir, self.specfem_dir)
            self.assertEqual(DB.cmt_fn, os.path.join(DATA_DIR, "CMTSOLUTION"))
            self.assertEqual(DB.basedir, tmp_dir)

    def test_create_db_dir(self):
        """This test the creation of the EQ directory."""

        # Test tmp_dir exist already
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=os.path.join(DATA_DIR, "CMTSOLUTION"),
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            DB.create_base()

            self.assertTrue(os.path.exists(DB.basedir))

        # Test tmp_dir exist already
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=os.path.join(tmp_dir, "db"),
                                  cmt_fn=os.path.join(DATA_DIR, "CMTSOLUTION"),
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            DB.create_base()

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "db")))

    def test_create_eq_dir(self):
        """Tests creation of earthquake directory and copying of the cmt
        solution"""

        # Check one cmt file
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")
            # create CMT
            cmt = CMTSource.from_CMTSOLUTION_file(cmtfile)

            # Create CMTSource to extract the file name
            eq_id = cmt.eventname

            # Earthquake directory
            eq_dir = os.path.join(tmp_dir, "eq_" + eq_id)

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # Create eq directory
            DB.create_eq_dirs()

            # check if new path exists
            new_cmt_path = os.path.join(eq_dir, "eq_" + eq_id + ".cmt")
            self.assertTrue(os.path.exists(new_cmt_path) and os.path.isfile(
                new_cmt_path))
            self.assertTrue(CMTSource.from_CMTSOLUTION_file(new_cmt_path)
                            == cmt)

    def test_create_eq_dir_mult(self):
        """Tests creation of earthquake directory and copying of the cmt
                solution for multiple cmt solution files."""

        # Check multiple cmt files
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Cmtfile path
            cmtfile1 = os.path.join(DATA_DIR, "CMTSOLUTION_TRUE")
            cmtfile2 = os.path.join(DATA_DIR, "CMTSOLUTION_VAR")

            # create CMT
            cmt1 = CMTSource.from_CMTSOLUTION_file(cmtfile1)
            cmt2 = CMTSource.from_CMTSOLUTION_file(cmtfile2)

            # Create CMTSource to extract the file name
            eq_id1 = cmt1.eventname
            eq_id2 = cmt2.eventname

            # Earthquake directory
            eq_dir1 = os.path.join(tmp_dir, "eq_" + eq_id1)
            eq_dir2 = os.path.join(tmp_dir, "eq_" + eq_id2)

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=os.path.join(DATA_DIR,
                                                      "CMTSOLUTION_*"),
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # Create eq directory
            DB.create_eq_dirs()

            # check if new path exists
            new_cmt_path1 = os.path.join(eq_dir1, "eq_" + eq_id1 + ".cmt")
            new_cmt_path2 = os.path.join(eq_dir2, "eq_" + eq_id2 + ".cmt")

            self.assertTrue(os.path.exists(new_cmt_path1) and os.path.isfile(
                new_cmt_path1))
            self.assertTrue(os.path.exists(new_cmt_path2) and os.path.isfile(
                new_cmt_path2))

    def test__create_dir(self):
        """Test the create directory method"""
        # Check one cmt file
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # create new directory
            test_dir = os.path.join(tmp_dir, "test")
            DB._create_dir(test_dir)

            self.assertTrue(os.path.exists(test_dir)
                            and os.path.isdir(test_dir))

    def test__copy_cmt(self):
        """Test the create directory method"""
        # Check one cmt file
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # create new directory
            new_cmt_path = os.path.join(tmp_dir, "blub.cmt")
            DB._copy_cmt(cmtfile, new_cmt_path)

            self.assertTrue(os.path.exists(new_cmt_path)
                            and os.path.isfile(new_cmt_path))

            self.assertTrue(CMTSource.from_CMTSOLUTION_file(new_cmt_path)
                            == CMTSource.from_CMTSOLUTION_file(cmtfile))

    def test_create_SIM_dir(self):
        """Tests the function that creates the Simulation directories and the
        copies the necessary files from the specfem directory."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # create CMT
            cmt = CMTSource.from_CMTSOLUTION_file(cmtfile)

            # Create CMTSource to extract the file name
            eq_id = cmt.eventname

            # Earthquake directory
            eq_dir = os.path.join(tmp_dir, "eq_" + eq_id)

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # Create earthquake solution directory
            DB.create_eq_dirs()

            # Create CMT simulation directory
            DB.create_CMT_SIM_dir()

            # Parameters
            attr = ["CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp", "CMT_tp",
                    "CMT_depth", "CMT_lat", "CMT_lon"]

            # Subdirectories
            subdirs = ["DATA", "DATABASES_MPI", "OUTPUT_FILES"]

            # Check for all directories
            for at in attr:

                # Attribute path
                test_dir = os.path.join(eq_dir, "CMT_SIMs", at)

                self.assertTrue(os.path.isdir(test_dir))

                # Now check if subdirectories are created
                for _k, _subdir in enumerate(subdirs):
                    test_dir2 = os.path.join(test_dir, _subdir)

                    self.assertTrue(os.path.isdir(test_dir2))

                # Check if link is created
                self.assertTrue(os.path.islink(os.path.join(test_dir, "bin")))

    def test__copy_dir(self):
        """Tests the copy dir function in skeleton."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            # Create on directory in temporary directory
            test_dir1 = os.path.join(tmp_dir, "test1")
            test_dir2 = os.path.join(tmp_dir, "test2")
            os.makedirs(test_dir1)

            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            DB._copy_dir(test_dir1, test_dir2)

            self.assertTrue(os.path.isdir(test_dir2))


if __name__ == "__main__":
    unittest.main()
