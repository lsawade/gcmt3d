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
# from gcmt3d.data.management.skeleton import get_Centry_path
from gcmt3d.source import CMTSource
from gcmt3d.asdf.utils import smart_read_yaml
import tempfile
import os
import inspect
from .functions_for_testing import assertDictAlmostEqual


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


# Recreates bash touch behaviour
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


class TestSkeleton(unittest.TestCase):
    """Class that handles testing of the Data Base structure creator"""

    def setUp(self):
        # Create a temporary specfem like directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.specfem_dir = self.test_dir.name
        os.makedirs(os.path.join(self.specfem_dir, "DATA"))
        touch(os.path.join(self.specfem_dir, "DATA", "STATIONS"))
        touch(os.path.join(self.specfem_dir, "DATA", "Par_file"))
        touch(os.path.join(self.specfem_dir, "DATA", "CMTSOLUTION"))
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

    def test_create_Cdir(self):
        """Tests creation of earthquake directory and copying of the cmt
        solution"""

        # Check one cmt file
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")
            # create CMT
            cmt = CMTSource.from_CMTSOLUTION_file(cmtfile)

            # Create CMTSource to extract the file name
            Cid = cmt.eventname

            # Earthquake directory
            Cdir = os.path.join(tmp_dir, "C" + Cid)

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # Create eq directory
            DB.create_Cdirs()

            # check if new path exists
            new_cmt_path = os.path.join(Cdir, "C" + Cid + ".cmt")
            self.assertTrue(os.path.exists(new_cmt_path) and os.path.isfile(
                new_cmt_path))
            self.assertTrue(CMTSource.from_CMTSOLUTION_file(new_cmt_path)
                            == cmt)

    def test_create_Cdir_mult(self):
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
            Cid1 = cmt1.eventname
            Cid2 = cmt2.eventname

            # Earthquake directory
            Cdir1 = os.path.join(tmp_dir, "C" + Cid1)
            Cdir2 = os.path.join(tmp_dir, "C" + Cid2)

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=os.path.join(DATA_DIR,
                                                      "CMTSOLUTION_*"),
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # Create eq directory
            DB.create_Cdirs()

            # check if new path exists
            new_cmt_path1 = os.path.join(Cdir1, "C" + Cid1 + ".cmt")
            new_cmt_path2 = os.path.join(Cdir2, "C" + Cid2 + ".cmt")

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
            DB._create_dir(test_dir, False)

            self.assertTrue(os.path.exists(test_dir)
                            and os.path.isdir(test_dir))

    def test__copy_file(self):
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
            DB._copy_file(cmtfile, new_cmt_path, False)

            self.assertTrue(os.path.exists(new_cmt_path)
                            and os.path.isfile(new_cmt_path))

            self.assertTrue(CMTSource.from_CMTSOLUTION_file(new_cmt_path)
                            == CMTSource.from_CMTSOLUTION_file(cmtfile))

    def test__copy_quake(self):
        """Test the create directory method"""
        # Check one cmt file
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "testCMT")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # create new directory
            new_cmt_path = os.path.join(tmp_dir, "blub.xml")
            DB._write_quakeml(cmtfile, new_cmt_path, True)

            self.assertTrue(os.path.exists(new_cmt_path)
                            and os.path.isfile(new_cmt_path))

            print("QuakeML\n", CMTSource.from_quakeml_file(new_cmt_path))
            print("CMT\n", CMTSource.from_CMTSOLUTION_file(cmtfile))
            assertDictAlmostEqual(CMTSource.from_quakeml_file(new_cmt_path),
                                  CMTSource.from_CMTSOLUTION_file(cmtfile))

    def test_create_SIM_dir(self):
        """Tests the function that creates the Simulation directories and the
        copies the necessary files from the specfem directory."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # create CMT
            cmt = CMTSource.from_CMTSOLUTION_file(cmtfile)

            # Create CMTSource to extract the file name
            Cid = cmt.eventname

            # Earthquake directory
            Cdir = os.path.join(tmp_dir, "C" + Cid)

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  overwrite=0,
                                  verbose=True)

            # Create earthquake solution directory
            DB.create_Cdirs()

            # Create CMT simulation directory
            DB.create_CMT_SIM_dir()

            # Parameters
            attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
                    "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

            # Subdirectories
            subdirs = ["DATA", "DATABASES_MPI", "OUTPUT_FILES"]

            # Check for all directories
            for at in attr:

                # Attribute path
                test_dir = os.path.join(Cdir, "CMT_SIMs", at)

                self.assertTrue(os.path.isdir(test_dir))

                # Check if yaml path file exist.
                self.assertTrue(os.path.isfile(os.path.join(test_dir,
                                                            at + ".yml")))

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

            DB._copy_dir(test_dir1, test_dir2, False)

            self.assertTrue(os.path.isdir(test_dir2))

    def test__create_syn_path_yaml(self):
        """Testing the creation of the yaml file."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # Create database
            DB.create_all()

            # Read the yaml_file which should be created in the CMT directory
            yaml_file = os.path.join(DB.Cdirs[0], "CMT_SIMs", "CMT_rr",
                                     "CMT_rr.yml")

            # Solution should be:
            waveform_dir = os.path.join(DB.Cdirs[0], "CMT_SIMs", "CMT_rr",
                                        "OUTPUT_FILES")
            tag = 'syn'
            filetype = 'sac'
            output_file = os.path.join(DB.Cdirs[0], "seismograms", "syn",
                                       "CMT_rr.h5")
            quakeml_file = os.path.join(DB.Cdirs[0], "CMT_SIMs", "CMT_rr",
                                        "OUTPUT_FILES", "Quake.xml")

            d = smart_read_yaml(yaml_file, mpi_mode=False)

            # Assessing correctness of yaml file
            self.assertTrue(d["quakeml_file"] == quakeml_file)
            self.assertTrue(d["tag"] == tag)
            self.assertTrue(d["output_file"] == output_file)
            self.assertTrue(d["filetype"] == filetype)
            self.assertTrue(d["waveform_dir"] == waveform_dir)

    def test__create_obs_path_yaml(self):
        """Testing the creation of the yaml file."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir,
                                  verbose=True)

            # Create database
            DB.create_all()

            # Read the yaml_file which should be created in the CMT directory
            yaml_file = os.path.join(DB.Cdirs[0], "seismograms", "obs",
                                     "observed.yml")

            # Solution should be:
            waveform_files = os.path.join(DB.Cdirs[0], "seismograms", "obs",
                                          "*.mseed")
            staxml = os.path.join(DB.Cdirs[0], "station_data", "*.xml")
            tag = 'obs'
            output_file = os.path.join(DB.Cdirs[0], "seismograms", "obs",
                                       "raw_observed.h5")
            quakeml_file = os.path.join(DB.Cdirs[0],
                                        "C" + DB.Cids[0] + ".xml")

            d = smart_read_yaml(yaml_file, mpi_mode=False)
            # Assessing correctness of yaml file
            self.assertTrue(d["quakeml_file"] == quakeml_file)
            self.assertTrue(d["tag"] == tag)
            self.assertTrue(d["output_file"] == output_file)
            self.assertTrue(d["waveform_files"] == waveform_files)
            self.assertTrue(d["staxml_files"] == staxml)


if __name__ == "__main__":
    unittest.main()
