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
                                  cmt_fn="./data/CMTSOLUTION",
                                  specfem_dir=self.specfem_dir, verbose=False)

            self.assertEqual(DB.specfem_dir, self.specfem_dir)
            self.assertEqual(DB.cmt_fn, "./data/CMTSOLUTION")
            self.assertEqual(DB.basedir, tmp_dir)

    def test_create_db_dir(self):
        """This test the creation of the EQ directory."""

        # Test tmp_dir exist already
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn="./data/CMTSOLUTION",
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            DB.create_base()

            self.assertTrue(os.path.exists(DB.basedir))

        # Test tmp_dir exist already
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=os.path.join(tmp_dir, "db"),
                                  cmt_fn="./data/CMTSOLUTION",
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
            cmtfile = "./data/CMTSOLUTION"
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
            DB.create_eq_dir()

            # check if new path exists
            new_cmt_path = os.path.join(eq_dir, "eq_" + eq_id + ".cmt")
            self.assertTrue(os.path.exists(new_cmt_path) and os.path.isfile(
                new_cmt_path))
            self.assertTrue(CMTSource.from_CMTSOLUTION_file(new_cmt_path) ==
                            cmt)

    def test_create_eq_dir_mult(self):
        """Tests creation of earthquake directory and copying of the cmt
                solution for multiple cmt solution files."""

        # Check multiple cmt files
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Cmtfile path
            cmtfile1 = "./data/CMTSOLUTION_TRUE"
            cmtfile2 = "./data/CMTSOLUTION_VAR"

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
                                  cmt_fn="./data/CMTSOLUTION_*",
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # Create eq directory
            DB.create_eq_dir()

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
            cmtfile = "./data/CMTSOLUTION"

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
            cmtfile = "./data/CMTSOLUTION"

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

            self.assertTrue(CMTSource.from_CMTSOLUTION_file(new_cmt_path) ==
                            CMTSource.from_CMTSOLUTION_file(cmtfile))


if __name__ == "__main__":
    unittest.main()

