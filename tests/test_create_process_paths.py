"""

Test suite for the creation of processing path_files.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2019

"""
import tempfile
import inspect
import os
import copy
import glob
import pprint
import unittest

from gcmt3d.utils.io import read_yaml_file
from gcmt3d.data.management.process_classifier import ProcessParams
from gcmt3d.data.management.create_process_paths \
    import create_processing_dictionary
from gcmt3d.data.management.create_process_paths \
    import get_window_parameter_dict
from gcmt3d.data.management.create_process_paths\
    import create_windowing_dictionary
from gcmt3d.data.management.create_process_paths import PathCreator
from gcmt3d.data.management.skeleton import DataBaseSkeleton
# from gcmt3d.data.management.create_process_paths import create_process_path_syn
# from gcmt3d.data.management.create_process_paths import create_process_path_obs
# from gcmt3d.data.management.create_process_paths import create_window_path
# from gcmt3d.data.management.create_process_paths import get_processing_list
# from gcmt3d.data.management.create_process_paths import get_windowing_list
# from gcmt3d.data.management.inversion_dicts \
#     import create_full_inversion_dict_list
# from gcmt3d.data.management.inversion_dicts import write_inversion_dicts


attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp", "CMT_tp",
        "CMT_depth", "CMT_lat", "CMT_lon"]


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


# Recreates bash touch behaviour
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def test_create_process_dictionary():
    """This one tests the processing dictionaries that are created to
    simplify process path file creation."""


    # Setup test dictionaries

    obsd_dict = {"remove_response_flag": True,
                 "water_level": 100.0,
                 "filter_flag": True,
                 "pre_filt": [0.0030, 0.0040, 0.0111, 0.0139],
                 "relative_starttime": 0,
                 "relative_endtime": 7150,
                 "resample_flag": True,
                 "sampling_rate": 5,
                 "taper_type": "hann",
                 "taper_percentage": 0.05,
                 "rotate_flag": True,
                 "sanity_check": True}

    synt_dict = {"remove_response_flag": False,
                 "filter_flag": True,
                 "pre_filt": [0.0030, 0.0040, 0.0111, 0.0139],
                 "relative_starttime": 0,
                 "relative_endtime": 7150,
                 "resample_flag": True,
                 "sampling_rate": 5,
                 "taper_type": "hann",
                 "taper_percentage": 0.05,
                 "rotate_flag": True,
                 "sanity_check": False}

    # Update base dictionary with the good values.
    body_obsddict = copy.deepcopy(obsd_dict)
    body_syntdict = copy.deepcopy(synt_dict)
    body_obsddict["pre_filt"] = [1.0 / 150.0, 1.0 / 100.0,
                                 1.0 / 60.0, 1.0 / 50.0]
    body_syntdict["pre_filt"] = [1.0 / 150.0, 1.0 / 100.0,
                                 1.0 / 60.0, 1.0 / 50.0]
    body_obsddict["relative_endtime"] = 3600.0
    body_syntdict["relative_endtime"] = 3600.0

    surface_obsddict = copy.deepcopy(obsd_dict)
    surface_syntdict = copy.deepcopy(synt_dict)
    surface_obsddict["pre_filt"] = [1.0 / 150.0, 1.0 / 100.0,
                                    1.0 / 60.0, 1.0 / 50.0]
    surface_syntdict["pre_filt"] = [1.0 / 150.0, 1.0 / 100.0,
                                    1.0 / 60.0, 1.0 / 50.0]
    surface_obsddict["relative_endtime"] = 7200.0
    surface_syntdict["relative_endtime"] = 7200.0

    mantle_obsddict = copy.deepcopy(obsd_dict)
    mantle_syntdict = copy.deepcopy(synt_dict)
    mantle_obsddict["pre_filt"] = [1.0 / 350.0, 1.0 / 300.0, 1.0 / 150.0,
                                   1.0 / 125.0]
    mantle_syntdict["pre_filt"] = [1.0 / 350.0, 1.0 / 300.0, 1.0 / 150.0,
                                   1.0 / 125.0]
    mantle_obsddict["relative_endtime"] = 4.5 * 3600.0
    mantle_syntdict["relative_endtime"] = 4.5 * 3600.0

    test_dict = {"body":
                     {"obsd": body_obsddict,
                      "synt": body_syntdict},
                 "surface":
                     {"obsd": surface_obsddict,
                      "synt": surface_syntdict},
                 "mantle":
                     {"obsd": mantle_obsddict,
                      "synt": mantle_syntdict}
    }


    # Actual Testing
    mw = 6.5
    depth = 200000  # in meters
    p = ProcessParams(mw, depth)
    pdict = p.determine_all()

    pprint.pprint(pdict)

    # Load base parameter files:
    obsd_params = read_yaml_file(os.path.join(DATA_DIR, 'params', "Process",
                                              "process_observed.yml"))
    synt_params = read_yaml_file(os.path.join(DATA_DIR, 'params', "Process",
                                              "process_synthetic.yml"))

    # Create the dictionary
    created_dict = create_processing_dictionary(pdict, obsd_params, synt_params)

    print("Body")
    pprint.pprint(created_dict["body"])
    pprint.pprint(test_dict["body"])
    print("Surface")
    pprint.pprint(created_dict["surface"])
    pprint.pprint(test_dict["surface"])
    print("Mantle")
    pprint.pprint(created_dict["mantle"])
    pprint.pprint(test_dict["mantle"])

    assert created_dict == test_dict



def test_get_wave_dict():
    """Tests whether the wavefiles are loaded correctly into the dictionaries"""

    # List of possible wavetypes
    waves = ["body", "surface", "mantle"]


    test_dict = dict()
    for wave in waves:
        wavedict = read_yaml_file(os.path.join(DATA_DIR, 'params', "Window",
                                               "window." + wave + ".yml"))
        test_dict[wave] = wavedict

    assert get_window_parameter_dict(
        os.path.join(DATA_DIR, "params", "Window")) == test_dict


def test_create_windowing_dictionary():

    # Get windowparameter dictionary
    windowconfig_dict = get_window_parameter_dict(
        os.path.join(DATA_DIR, 'params', "Window"))

    # Get cmtconfigoparams
    mw = 6.5
    depth = 200000  # in meters
    p = ProcessParams(mw, depth)
    cmtconfigdict = p.determine_all()

    # Create full windowing dictionary
    fulldict = create_windowing_dictionary(cmtconfigdict, windowconfig_dict)

    assert fulldict["body"]["min_period"] == 60.0
    assert fulldict["body"]["max_period"] == 100.0
    assert fulldict["surface"]["min_period"] == 60.0
    assert fulldict["surface"]["max_period"] == 100.0
    assert fulldict["mantle"]["min_period"] == 150.0
    assert fulldict["mantle"]["max_period"] == 300.0

    # One more test to check if certain slots are empty depending on the
    # cmtconfig
    mw = 5.25
    depth = 200000  # in meters
    p = ProcessParams(mw, depth)
    cmtconfigdict = p.determine_all()

    # Create full windowing dictionary
    fulldict = create_windowing_dictionary(cmtconfigdict, windowconfig_dict)

    assert fulldict["body"]["min_period"] == 50.0
    assert fulldict["body"]["max_period"] == 100.0
    assert fulldict["surface"]["min_period"] == 60.0
    assert fulldict["surface"]["max_period"] == 100.0

    assert "mantle" not in fulldict.keys()

class TestCreatePaths(unittest.TestCase):
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
        # Close the file, the directory will be removed after the tests
        self.test_dir.cleanup()

    def test_IO(self):
        """Testing if strings are set correctly."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=os.path.join(DATA_DIR, "CMTSOLUTION"),
                                  specfem_dir=self.specfem_dir)

            self.assertEqual(DB.specfem_dir, self.specfem_dir)
            self.assertEqual(DB.cmt_fn, os.path.join(DATA_DIR, "CMTSOLUTION"))
            self.assertEqual(DB.basedir, tmp_dir)


    def test_PathCreator(self):
        """Tests the path crator"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir)

            # Create database
            DB.create_all()

            # Read the yaml_file which should be created in the CMT directory
            windowbasedir = os.path.join(DATA_DIR, "params", "Window")
            processbasedir = os.path.join(DATA_DIR, "params", "Process")

            # CMT filename in database
            cmt_in_db = os.path.join(DB.Cdirs[0], "C" + DB.Cids[0] + ".cmt")

            # Workflow directory
            workflow_dir = os.path.join(DB.Cdirs[0], "workflow_files")
            procpaths = os.path.join(workflow_dir, "path_files",
                                     "process_paths")
            procparams = os.path.join(workflow_dir, "params",
                                      "process_params")
            windowparams = os.path.join(workflow_dir, "params",
                                        "window_params")
            # Trying the thing
            p = PathCreator(cmt_in_db, windowbasedir, processbasedir)
            p.write_all()

            checklist = [os.path.join(procparams,
                                      "body.obsd.process.yml"),
                         os.path.join(procparams,
                                      "body.synt.process.yml"),
                         os.path.join(procparams,
                                      "surface.obsd.process.yml"),
                         os.path.join(procparams,
                                      "surface.synt.process.yml"),
                         os.path.join(windowparams,
                                      "body.window.yml"),
                         os.path.join(windowparams,
                                      "surface.window.yml"),
                         os.path.join(procpaths, "body.obsd.process_path.yml"),
                         os.path.join(procpaths,
                                      "surface.obsd.process_path.yml")
                         ]

            for at in attr:
                checklist.append(os.path.join(procpaths,
                                 "body.synt." + at + ".process_path.yml"))
                checklist.append(os.path.join(procpaths,
                                 "surface.synt." + at + ".process_path.yml"))

            for file in checklist:
                fullfile = os.path.join(file)
                print(fullfile)
                assert os.path.exists(fullfile)


    def test_PathCreator_procpaths(self):
        """Checks Processing path file content."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir)

            # Create database
            DB.create_all()

            # Read the yaml_file which should be created in the CMT directory
            windowbasedir = os.path.join(DATA_DIR, "params", "Window")
            processbasedir = os.path.join(DATA_DIR, "params", "Process")

            # CMT filename in database
            cmt_in_db = os.path.join(DB.Cdirs[0], "C" + DB.Cids[0] + ".cmt")

            # Workflow directory
            workflow_dir = os.path.join(DB.Cdirs[0], "workflow_files")
            procpaths = os.path.join(workflow_dir, "path_files",
                                     "process_paths")
            procparams = os.path.join(workflow_dir, "params",
                                      "process_params")
            windowparams = os.path.join(workflow_dir, "params",
                                        "window_params")
            # Trying the thing
            p = PathCreator(cmt_in_db, windowbasedir, processbasedir)
            p.write_all()

            # Now that the files are checked for existence, we can check the
            # contents (of three path files)
            process_file = os.path.join(procpaths,
                                        "body.obsd.process_path.yml")

            # Solution should be:
            input_asdf = os.path.join(DB.Cdirs[0], "seismograms", "obs",
                                      "raw_observed.h5")
            input_tag = "obs"
            output_asdf = os.path.join(DB.Cdirs[0], "seismograms",
                                       "processed_seismograms",
                                       "body.obsd.h5")

            output_tag = "body"
            process_param_file = os.path.join(procparams, "body.obsd.process.yml")

            d = read_yaml_file(process_file)

            # Assessing correctness of yaml file
            self.assertTrue(d["input_asdf"] == input_asdf)
            self.assertTrue(d["input_tag"] == input_tag)
            self.assertTrue(d["output_asdf"] == output_asdf)
            self.assertTrue(d["output_tag"] == output_tag)
            self.assertTrue(d["process_param_file"] == process_param_file)

            # Try a synthetic one
            process_file = os.path.join(procpaths,
                                        "body.synt.CMT_rr.process_path.yml")

            # Solution should be:
            input_asdf = os.path.join(DB.Cdirs[0], "seismograms", "syn",
                                      "CMT_rr.h5")
            input_tag = "syn"
            output_asdf = os.path.join(DB.Cdirs[0], "seismograms",
                                       "processed_seismograms",
                                       "body.synt.CMT_rr.h5")

            output_tag = "body_cmt_rr"
            process_param_file = os.path.join(procparams,
                                              "body.synt.process.yml")

            d = read_yaml_file(process_file)

            # Assessing correctness of yaml file
            self.assertTrue(d["input_asdf"] == input_asdf)
            self.assertTrue(d["input_tag"] == input_tag)
            self.assertTrue(d["output_asdf"] == output_asdf)
            self.assertTrue(d["output_tag"] == output_tag)
            self.assertTrue(d["process_param_file"] == process_param_file)



    def test_PathCreator_windpaths(self):
        """Tests the windowpaths"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir)

            # Create database
            DB.create_all()

            # Read the yaml_file which should be created in the CMT directory
            windowbasedir = os.path.join(DATA_DIR, "params", "Window")
            processbasedir = os.path.join(DATA_DIR, "params", "Process")

            # CMT filename in database
            cmt_in_db = os.path.join(DB.Cdirs[0], "C" + DB.Cids[0] + ".cmt")

            # Workflow directory
            workflow_dir = os.path.join(DB.Cdirs[0], "workflow_files")
            winpaths = os.path.join(workflow_dir, "path_files",
                                     "window_paths")
            windowparams = os.path.join(workflow_dir, "params",
                                        "window_params")
            # Trying the thing
            p = PathCreator(cmt_in_db, windowbasedir, processbasedir)
            p.write_all()

            # Try one window file
            procdir = os.path.join(DB.Cdirs[0], "seismograms",
                                  "processed_seismograms")
            windir = os.path.join(DB.Cdirs[0], "window_data")

            # Content
            obsd_asdf = os.path.join(procdir, "body.obsd.h5")
            obsd_tag = "body"
            synt_asdf = os.path.join(procdir, "body.synt.CMT.h5")
            synt_tag = "body_cmt"
            output_file = os.path.join(windir, "body.windows.json")
            figure_dir = os.path.join(windir, "body_wave_plots")
            window_param_file = os.path.join(windowparams, "body.window.yml")

            # Load written window path_file
            winpath = read_yaml_file(os.path.join(winpaths,
                                                  "body.window_path.yml"))

            # Check if correct
            self.assertTrue(winpath["obsd_asdf"] == obsd_asdf)
            self.assertTrue(winpath["obsd_tag"] == obsd_tag)
            self.assertTrue(winpath["synt_asdf"] == synt_asdf)
            self.assertTrue(winpath["synt_tag"] == synt_tag)
            self.assertTrue(winpath["output_file"] == output_file)
            self.assertTrue(winpath["figure_dir"] == figure_dir)
            self.assertTrue(winpath["window_param_file"] == window_param_file)

            # Also run check of surface waves
            # Content
            obsd_asdf = os.path.join(procdir, "surface.obsd.h5")
            obsd_tag = "surface"
            synt_asdf = os.path.join(procdir, "surface.synt.CMT.h5")
            synt_tag = "surface_cmt"
            output_file = os.path.join(windir, "surface.windows.json")
            figure_dir = os.path.join(windir, "surface_wave_plots")
            window_param_file = os.path.join(windowparams, "surface.window.yml")

            # Load written window path_file
            winpath = read_yaml_file(os.path.join(winpaths,
                                                  "surface.window_path.yml"))

            # Check if correct
            self.assertTrue(winpath["obsd_asdf"] == obsd_asdf)
            self.assertTrue(winpath["obsd_tag"] == obsd_tag)
            self.assertTrue(winpath["synt_asdf"] == synt_asdf)
            self.assertTrue(winpath["synt_tag"] == synt_tag)
            self.assertTrue(winpath["output_file"] == output_file)
            self.assertTrue(winpath["figure_dir"] == figure_dir)
            self.assertTrue(winpath["window_param_file"] == window_param_file)


    def test_PathCreator_invdict(self):
        """Test the inversion dictionary creation"""


        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir)

            # Create database
            DB.create_all()

            # Read the yaml_file which should be created in the CMT directory
            windowbasedir = os.path.join(DATA_DIR, "params", "Window")
            processbasedir = os.path.join(DATA_DIR, "params", "Process")

            # CMT filename in database
            cmt_in_db = os.path.join(DB.Cdirs[0], "C" + DB.Cids[0] + ".cmt")

            # Workflow directory
            workflow_dir = os.path.join(DB.Cdirs[0], "workflow_files")
            invdir = os.path.join(workflow_dir, "inversion_dicts")

            # Trying the thing
            p = PathCreator(cmt_in_db, windowbasedir, processbasedir)
            p.write_all()

            # Try one window file
            procdir = os.path.join(DB.Cdirs[0], "seismograms",
                                  "processed_seismograms")
            windir = os.path.join(DB.Cdirs[0], "window_data")

            asdf_dict = dict()
            asdf_dict['Mpp'] = os.path.join(procdir, "body.synt.CMT_pp.h5")
            asdf_dict['Mrp'] = os.path.join(procdir, "body.synt.CMT_rp.h5")
            asdf_dict['Mrt'] = os.path.join(procdir, "body.synt.CMT_pt.h5")
            asdf_dict['Mrr'] = os.path.join(procdir, "body.synt.CMT_rr.h5")
            asdf_dict['Mrt'] = os.path.join(procdir, "body.synt.CMT_rt.h5")
            asdf_dict['Mtp'] = os.path.join(procdir, "body.synt.CMT_tp.h5")
            asdf_dict['Mtt'] = os.path.join(procdir, "body.synt.CMT_tt.h5")
            asdf_dict['dep'] = os.path.join(procdir, "body.synt.CMT_depth.h5")
            asdf_dict['lat'] = os.path.join(procdir, "body.synt.CMT_lat.h5")
            asdf_dict['lon'] = os.path.join(procdir, "body.synt.CMT_lon.h5")
            asdf_dict['synt'] = os.path.join(procdir, "body.synt.CMT.h5")
            asdf_dict['obsd'] = os.path.join(procdir, "body.obsd.h5")
            window_file = os.path.join(windir, "body.windows.json")

            # load inversion dict
            invdict = read_yaml_file(os.path.join(invdir,
                                                  "cmt3d.body.inv_dict.yml"))

            self.assertTrue(invdict["asdf_dict"] == asdf_dict)
            self.assertTrue(invdict["window_file"] == window_file)

    def test_PathCreator_invdict_g3d(self):
        """Test the inversion dictionary creation"""


        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cmtfile path
            cmtfile = os.path.join(DATA_DIR, "CMTSOLUTION")

            # Initialize database skeleton class
            DB = DataBaseSkeleton(basedir=tmp_dir,
                                  cmt_fn=cmtfile,
                                  specfem_dir=self.specfem_dir)

            # Create database
            DB.create_all()

            # Read the yaml_file which should be created in the CMT directory
            windowbasedir = os.path.join(DATA_DIR, "params", "Window")
            processbasedir = os.path.join(DATA_DIR, "params", "Process")

            # CMT filename in database
            cmt_in_db = os.path.join(DB.Cdirs[0], "C" + DB.Cids[0] + ".cmt")

            # Workflow directory
            workflow_dir = os.path.join(DB.Cdirs[0], "workflow_files")
            invdir = os.path.join(workflow_dir, "inversion_dicts")

            # Trying the thing
            p = PathCreator(cmt_in_db, windowbasedir, processbasedir)
            p.write_all()

            # Try one window file
            procdir = os.path.join(DB.Cdirs[0], "seismograms",
                                  "processed_seismograms")
            windir = os.path.join(DB.Cdirs[0], "window_data")
            invoutdir = os.path.join(DB.Cdirs[0], "inversion", "cmt3d",
                                     "new_synt")

            asdf_dict = dict()
            asdf_dict['synt'] = os.path.join(invoutdir, "*body_synt.h5")
            asdf_dict['obsd'] = os.path.join(procdir, "body.obsd.h5")
            window_file = os.path.join(windir, "body.windows.json")

            # load inversion dict
            invdict = read_yaml_file(os.path.join(invdir,
                                                  "g3d.body.inv_dict.yml"))

            self.assertTrue(invdict["asdf_dict"] == asdf_dict)
            self.assertTrue(invdict["window_file"] == window_file)





if __name__ == "__main__":
    unittest.main()
